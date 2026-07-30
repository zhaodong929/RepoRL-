"""Deterministic, CPU-only fixture for exercising the materialization trust boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path

from reporl.schemas import DatasetSplit, TaskProvenance, TaskSpec
from reporl.tasks.adapters import SWESmithExportV1, SWESmithTestIdMapping
from reporl.tasks.admission import (
    AdmissionEvidence,
    LeakScan,
    LicenseReview,
    RegressionExpectation,
    SnapshotKind,
    SnapshotValidation,
    TestRunOutcome,
    validate_task_admission,
)
from reporl.tasks.canonical import artifact_sha256, canonical_json
from reporl.tasks.dataset import DatasetManifest, DatasetTaskEntry, seal_dataset_manifest
from reporl.tasks.lineage import RepositoryRecord
from reporl.tasks.manifest import (
    AgentTestSuite,
    ArtifactReference,
    TrustedCommand,
    VerifierManifest,
    VerifierTestSuite,
)


def _fake_digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _reference(artifact_root: Path, path: Path) -> ArtifactReference:
    digest, size_bytes = artifact_sha256(artifact_root / path)
    return ArtifactReference(path=path.as_posix(), sha256=digest, size_bytes=size_bytes)


def _outcomes(suite: str, *, passes: bool) -> tuple[TestRunOutcome, ...]:
    return tuple(
        TestRunOutcome(
            suite=suite,  # type: ignore[arg-type]
            attempt=attempt,
            exit_code=0 if passes else 1,
            passed=1 if passes else 0,
            failed=0 if passes else 1,
            errors=0,
            duration_ms=5,
            report_sha256=_fake_digest(f"{suite}-{passes}-{attempt}"),
        )
        for attempt in range(1, 4)
    )


def _build_task(
    artifact_root: Path,
    split: DatasetSplit,
    index: int,
) -> tuple[VerifierManifest, AdmissionEvidence, RepositoryRecord, SWESmithExportV1]:
    task_id = f"fixture-{split.value}-001"
    repository_id = f"fixture-repo-{split.value}"
    source_url = f"https://github.com/reporl-fixtures/{repository_id}"
    lineage = f"fixture-lineage-{split.value}"
    base_commit = f"{index:x}" * 40
    relative_root = Path("tasks") / task_id
    clean = relative_root / "states" / "clean"
    buggy = relative_root / "states" / "buggy"
    reference = relative_root / "states" / "reference"
    hidden = relative_root / "sealed" / "hidden-tests"
    mutation_patch = relative_root / "sealed" / "mutation.patch"
    patch = relative_root / "sealed" / "reference.patch"

    public_tests = (
        "from src.calculator import add, identity\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n"
        "def test_identity():\n"
        "    assert identity(7) == 7\n"
    )
    for snapshot, implementation in (
        (
            clean,
            "def add(left, right):\n"
            "    return left + right\n\n"
            "def identity(value):\n"
            "    return value\n",
        ),
        (
            buggy,
            "def add(left, right):\n"
            "    return left - right\n\n"
            "def identity(value):\n"
            "    return value\n",
        ),
        (
            reference,
            "def add(left, right):\n"
            "    # Reference repair.\n"
            "    return left + right\n\n"
            "def identity(value):\n"
            "    return value\n",
        ),
    ):
        _write(artifact_root / snapshot / "src" / "__init__.py", "")
        _write(artifact_root / snapshot / "src" / "calculator.py", implementation)
        _write(artifact_root / snapshot / "tests" / "test_public.py", public_tests)
    _write(
        artifact_root / hidden / "test_target.py",
        "from src.calculator import add\n\ndef test_hidden_add():\n    assert add(-2, 5) == 3\n",
    )
    _write(
        artifact_root / mutation_patch,
        "diff --git a/src/calculator.py b/src/calculator.py\n"
        "--- a/src/calculator.py\n"
        "+++ b/src/calculator.py\n"
        "@@ -1,5 +1,5 @@\n"
        " def add(left, right):\n"
        "-    return left + right\n"
        "+    return left - right\n"
        " \n"
        " def identity(value):\n"
        "     return value\n",
    )
    _write(
        artifact_root / patch,
        "diff --git a/src/calculator.py b/src/calculator.py\n"
        "--- a/src/calculator.py\n"
        "+++ b/src/calculator.py\n"
        "@@ -1,5 +1,6 @@\n"
        " def add(left, right):\n"
        "-    return left - right\n"
        "+    # Reference repair.\n"
        "+    return left + right\n"
        " \n"
        " def identity(value):\n"
        "     return value\n",
    )

    clean_ref = _reference(artifact_root, clean)
    buggy_ref = _reference(artifact_root, buggy)
    reference_ref = _reference(artifact_root, reference)
    hidden_ref = _reference(artifact_root, hidden)
    patch_ref = _reference(artifact_root, patch)
    task = TaskSpec(
        task_id=task_id,
        issue="The add function subtracts its right operand instead of adding it.",
        split=split,
        agent_image_digest=_fake_digest(f"agent-image-{split.value}"),
        provenance=TaskProvenance(
            source_repository=source_url,
            source_license="MIT",
            base_commit=base_commit,
            lineage_group=lineage,
            generator="SWE-smith",
            generator_version="9b74ac0",
        ),
        allowed_paths=("src/calculator.py",),
    )
    evidence = AdmissionEvidence(
        task=task,
        clean=SnapshotValidation(
            kind=SnapshotKind.CLEAN,
            snapshot_sha256=clean_ref.sha256,
            outcomes=_outcomes("regression", passes=True),
        ),
        buggy=SnapshotValidation(
            kind=SnapshotKind.BUGGY,
            snapshot_sha256=buggy_ref.sha256,
            outcomes=_outcomes("target", passes=False) + _outcomes("regression", passes=True),
        ),
        reference=SnapshotValidation(
            kind=SnapshotKind.REFERENCE,
            snapshot_sha256=reference_ref.sha256,
            outcomes=_outcomes("target", passes=True) + _outcomes("regression", passes=True),
        ),
        buggy_regression_expectation=RegressionExpectation.PASSES,
        license_review=LicenseReview(
            repository_url=source_url,
            spdx_identifier="MIT",
            license_file_sha256=_fake_digest(f"license-{split.value}"),
            use_approved=True,
            redistribution_approved=True,
            reviewed_by="fixture-builder",
        ),
        leak_scan=LeakScan(
            issue_text_clean=True,
            agent_image_clean=True,
            agent_image_digest=task.agent_image_digest,
        ),
    )
    admission = validate_task_admission(evidence)
    agent_suites = (
        AgentTestSuite(
            name="target",
            command=TrustedCommand(
                argv=("python", "-m", "pytest", "tests/test_public.py::test_add")
            ),
        ),
        AgentTestSuite(
            name="regression",
            command=TrustedCommand(
                argv=("python", "-m", "pytest", "tests/test_public.py::test_identity")
            ),
        ),
    )
    verifier_suites = (
        VerifierTestSuite(
            name="target",
            command=TrustedCommand(
                argv=(
                    "python",
                    "-m",
                    "pytest",
                    "/verifier-tests/test_target.py",
                    "--junitxml=/tmp/reporl-junit/target.xml",
                )
            ),
            junit_path="/tmp/reporl-junit/target.xml",
            expected_test_ids=("test_target.py::test_hidden_add",),
        ),
        VerifierTestSuite(
            name="regression",
            command=TrustedCommand(
                argv=(
                    "python",
                    "-m",
                    "pytest",
                    "tests/test_public.py::test_identity",
                    "--junitxml=/tmp/reporl-junit/regression.xml",
                )
            ),
            junit_path="/tmp/reporl-junit/regression.xml",
            expected_test_ids=("tests/test_public.py::test_identity",),
        ),
    )
    manifest = VerifierManifest(
        task=task,
        verifier_image_digest=_fake_digest(f"verifier-image-{split.value}"),
        clean_snapshot=clean_ref,
        buggy_snapshot=buggy_ref,
        reference_snapshot=reference_ref,
        hidden_tests=hidden_ref,
        reference_patch=patch_ref,
        agent_test_suites=agent_suites,
        test_suites=verifier_suites,
        admission_result_sha256=admission.digest(),
    )
    repository = RepositoryRecord(
        repository_id=repository_id,
        source_url=source_url,
        lineage_group=lineage,
        split=split,
        commits=(base_commit,),
        content_fingerprints=(_fake_digest(f"repository-{split.value}"),),
    )
    export = SWESmithExportV1(
        schema_version="reporl.swe-smith-export/v1",
        instance_id=task.task_id,
        repo=f"reporl-fixtures/{repository_id}",
        patch=(artifact_root / mutation_patch).read_text(encoding="utf-8"),
        FAIL_TO_PASS=("test_target.py::test_hidden_add",),
        PASS_TO_PASS=("tests/test_public.py::test_identity",),
        image_name=task.agent_image_digest,
        problem_statement=task.issue,
        resolved_agent_image=task.agent_image_digest,
        target_test_id_map=(
            SWESmithTestIdMapping(
                official_id="test_target.py::test_hidden_add",
                junit_id="test_target.py::test_hidden_add",
            ),
        ),
        regression_test_id_map=(
            SWESmithTestIdMapping(
                official_id="tests/test_public.py::test_identity",
                junit_id="tests/test_public.py::test_identity",
            ),
        ),
        task=task,
        repository=repository,
        mutation_patch_path=mutation_patch.as_posix(),
        clean_snapshot_path=clean.as_posix(),
        buggy_snapshot_path=buggy.as_posix(),
        reference_snapshot_path=reference.as_posix(),
        hidden_tests_path=hidden.as_posix(),
        reference_patch_path=patch.as_posix(),
        verifier_image_digest=manifest.verifier_image_digest,
        agent_test_suites=agent_suites,
        verifier_test_suites=verifier_suites,
        admission_evidence=evidence,
    )
    return manifest, evidence, repository, export


def _write_jsonl(path: Path, records: tuple[object, ...]) -> None:
    _write(path, "".join(f"{canonical_json(record)}\n" for record in records))


def build_materialization_fixture(output_root: Path) -> dict[str, Path]:
    """Create three tiny sealed tasks and all trust evidence without running Docker."""

    root = output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"fixture output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    artifact_root = root / "artifacts"
    built = tuple(
        _build_task(artifact_root, split, index)
        for index, split in enumerate(DatasetSplit, start=10)
    )
    manifests = tuple(item[0] for item in built)
    evidence = tuple(item[1] for item in built)
    repositories = tuple(item[2] for item in built)
    admissions = tuple(validate_task_admission(item) for item in evidence)
    entries = tuple(
        DatasetTaskEntry(
            task_id=manifest.task.task_id,
            split=manifest.task.split,
            repository_id=repository.repository_id,
            source_url=repository.source_url,
            lineage_group=repository.lineage_group,
            verifier_manifest_sha256=manifest.digest(),
            admission_result_sha256=admission.digest(),
        )
        for manifest, repository, admission in zip(manifests, repositories, admissions, strict=True)
    )
    dataset_manifest = DatasetManifest(
        dataset_id="reporl-materialization-fixture",
        version="1",
        tasks=entries,
    )
    split_seal = seal_dataset_manifest(dataset_manifest)
    paths = {
        "artifact_root": artifact_root,
        "manifests": root / "verifier-manifests.jsonl",
        "dataset_manifest": root / "dataset-manifest.json",
        "split_seal": root / "split-seal.json",
        "repositories": root / "repositories.jsonl",
        "admission_evidence": root / "admission-evidence.jsonl",
        "admission_results": root / "admission-results.jsonl",
        "swe_smith_export": root / "swe-smith-export-v1.jsonl",
    }
    _write_jsonl(paths["manifests"], manifests)
    _write(paths["dataset_manifest"], f"{canonical_json(dataset_manifest)}\n")
    _write(paths["split_seal"], f"{canonical_json(split_seal)}\n")
    _write_jsonl(paths["repositories"], repositories)
    _write_jsonl(paths["admission_evidence"], evidence)
    _write_jsonl(paths["admission_results"], admissions)
    _write_jsonl(paths["swe_smith_export"], tuple(item[3] for item in built))
    return paths
