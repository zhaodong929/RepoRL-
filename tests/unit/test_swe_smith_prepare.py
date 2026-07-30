from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from reporl.schemas import DatasetSplit, TaskProvenance, TaskSpec
from reporl.tasks.adapters import (
    SWESmithExportAdapterV1,
    SWESmithExportV1,
    SWESmithTestIdMapping,
)
from reporl.tasks.admission import (
    LeakScan,
    LicenseReview,
    SnapshotKind,
    SnapshotValidation,
)
from reporl.tasks.admission import (
    TestRunOutcome as AdmissionTestRunOutcome,
)
from reporl.tasks.admission_docker import DockerAdmissionExecutor
from reporl.tasks.canonical import artifact_sha256
from reporl.tasks.loader import TaskDataError, load_jsonl
from reporl.tasks.manifest import AgentTestSuite, TrustedCommand, VerifierTestSuite
from reporl.tasks.swe_smith_prepare import (
    SWE_SMITH_COMMIT,
    OfficialSWESmithInstance,
    SWESmithPreparationSpec,
    load_official_swe_smith_instances,
    prepare_swe_smith_bundle,
)


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _repository(root: Path) -> tuple[Path, str, str]:
    repository = root / "demo"
    (repository / "src").mkdir(parents=True)
    (repository / "tests").mkdir()
    (repository / "src" / "calculator.py").write_text(
        "def add(left, right):\n    return left + right\n\n"
        "def identity(value):\n    return value\n",
        encoding="utf-8",
    )
    (repository / "tests" / "test_calc.py").write_text(
        "from src.calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    (repository / "tests" / "test_regression.py").write_text(
        "from src.calculator import identity\n\n"
        "def test_identity():\n    assert identity(7) == 7\n",
        encoding="utf-8",
    )
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Fixture")
    _git(repository, "config", "user.email", "fixture@invalid")
    _git(repository, "add", "--all", "--", ".")
    _git(repository, "commit", "--quiet", "-m", "clean")
    commit = _git(repository, "rev-parse", "HEAD").strip()
    calculator = repository / "src" / "calculator.py"
    calculator.write_text(
        "def add(left, right):\n    return left - right\n\n"
        "def identity(value):\n    return value\n",
        encoding="utf-8",
    )
    patch = _git(repository, "diff", "--binary", "--", "src/calculator.py")
    _git(repository, "restore", "--", "src/calculator.py")
    return repository, commit, patch


def _inputs(root: Path) -> tuple[OfficialSWESmithInstance, SWESmithPreparationSpec]:
    repository, commit, patch = _repository(root)
    del repository
    task = TaskSpec(
        task_id="demo.bug_001",
        issue="The add function subtracts instead of adding its operands.",
        split=DatasetSplit.TRAIN,
        agent_image_digest=_digest("a"),
        provenance=TaskProvenance(
            source_repository="https://github.com/example/demo",
            source_license="MIT",
            base_commit=commit,
            lineage_group="example-demo",
            generator="SWE-smith",
            generator_version=SWE_SMITH_COMMIT,
        ),
        allowed_paths=("src/calculator.py",),
    )
    target_mapping = SWESmithTestIdMapping(
        official_id="tests/test_calc.py::test_add",
        junit_id="tests.test_calc::test_add",
    )
    regression_mapping = SWESmithTestIdMapping(
        official_id="tests/test_regression.py::test_identity",
        junit_id="tests.test_regression::test_identity",
    )
    agent_suites = (
        AgentTestSuite(
            name="target",
            command=TrustedCommand(argv=("python", "-m", "pytest", "tests/test_regression.py")),
        ),
        AgentTestSuite(
            name="regression",
            command=TrustedCommand(argv=("python", "-m", "pytest", "tests/test_regression.py")),
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
                    "/verifier-tests/tests/test_calc.py",
                    "--junitxml=/tmp/reporl-junit/target.xml",
                )
            ),
            junit_path="/tmp/reporl-junit/target.xml",
            expected_test_ids=(target_mapping.junit_id,),
        ),
        VerifierTestSuite(
            name="regression",
            command=TrustedCommand(
                argv=(
                    "python",
                    "-m",
                    "pytest",
                    "tests/test_regression.py",
                    "--junitxml=/tmp/reporl-junit/regression.xml",
                )
            ),
            junit_path="/tmp/reporl-junit/regression.xml",
            expected_test_ids=(regression_mapping.junit_id,),
        ),
    )
    instance = OfficialSWESmithInstance(
        instance_id=task.task_id,
        repo="swesmith/example__demo.12345678",
        patch=patch,
        FAIL_TO_PASS=(target_mapping.official_id,),
        PASS_TO_PASS=(regression_mapping.official_id,),
        image_name="example/agent:fixed",
    )
    spec = SWESmithPreparationSpec(
        schema_version="reporl.swe-smith-prepare/v1",
        instance_id=task.task_id,
        gather_repo=instance.repo,
        repository_id="example-demo",
        repository_path="demo",
        task=task,
        agent_image=f"{instance.image_name}@{task.agent_image_digest}",
        verifier_image=f"example/verifier:fixed@{_digest('b')}",
        hidden_test_paths=("tests/test_calc.py",),
        target_test_id_map=(target_mapping,),
        regression_test_id_map=(regression_mapping,),
        agent_test_suites=agent_suites,
        verifier_test_suites=verifier_suites,
        license_review=LicenseReview(
            repository_url=task.provenance.source_repository,
            spdx_identifier="MIT",
            license_file_sha256=_digest("c"),
            use_approved=True,
            redistribution_approved=True,
            reviewed_by="fixture-reviewer",
        ),
        leak_scan=LeakScan(
            issue_text_clean=True,
            agent_image_clean=True,
            agent_image_digest=task.agent_image_digest,
        ),
    )
    return instance, spec


class _DeterministicExecutor:
    def __init__(self) -> None:
        self.kinds: list[SnapshotKind] = []

    def run_snapshot(
        self,
        *,
        kind: SnapshotKind,
        snapshot: Path,
        snapshot_sha256: str,
        hidden_tests: Path,
        image: str,
        suites: tuple[VerifierTestSuite, ...],
        repetitions: int,
    ) -> SnapshotValidation:
        del image
        self.kinds.append(kind)
        implementation = (snapshot / "src" / "calculator.py").read_text(encoding="utf-8")
        assert ("left - right" in implementation) is (kind is SnapshotKind.BUGGY)
        assert not (snapshot / "tests" / "test_calc.py").exists()
        assert (hidden_tests / "tests" / "test_calc.py").is_file()
        outcomes: list[AdmissionTestRunOutcome] = []
        for suite in suites:
            for attempt in range(1, repetitions + 1):
                fails = kind is SnapshotKind.BUGGY and suite.name == "target"
                outcomes.append(
                    AdmissionTestRunOutcome(
                        suite=suite.name,
                        attempt=attempt,
                        exit_code=1 if fails else 0,
                        passed=0 if fails else 1,
                        failed=1 if fails else 0,
                        errors=0,
                        duration_ms=1,
                        report_sha256=_digest(str(attempt)),
                    )
                )
        return SnapshotValidation(
            kind=kind,
            snapshot_sha256=snapshot_sha256,
            outcomes=tuple(outcomes),
        )


def test_real_git_preparation_builds_portable_admitted_export(tmp_path: Path) -> None:
    repositories = tmp_path / "repositories"
    repositories.mkdir()
    instance, spec = _inputs(repositories)
    executor = _DeterministicExecutor()

    result = prepare_swe_smith_bundle(
        instances=(instance,),
        specs=(spec,),
        repositories_root=repositories,
        output_root=tmp_path / "prepared",
        executor=executor,
    )

    assert executor.kinds == [SnapshotKind.CLEAN, SnapshotKind.BUGGY, SnapshotKind.REFERENCE]
    records = load_jsonl(result.export_jsonl, SWESmithExportV1)
    assert len(records) == 1
    record = records[0]
    assert record.repo == instance.repo
    assert record.image_name == instance.image_name
    assert record.resolved_agent_image == spec.agent_image
    assert not record.task.provenance.source_repository.startswith("swesmith/")
    for snapshot_path in (
        record.clean_snapshot_path,
        record.buggy_snapshot_path,
        record.reference_snapshot_path,
    ):
        snapshot = result.artifacts_root / snapshot_path
        assert not any(entry.name == ".git" for entry in snapshot.rglob(".git"))
        assert not (snapshot / "tests" / "test_calc.py").exists()
    assert (result.artifacts_root / record.hidden_tests_path / "tests/test_calc.py").is_file()
    assert result.metadata.instance_count == 1
    imported = SWESmithExportAdapterV1().import_records(
        records,
        result.artifacts_root,
        dataset_id="prepared-fixture",
        dataset_version="1",
    )
    assert imported.manifests[0].task == spec.task


def test_import_merges_multiple_commits_from_the_same_repository(tmp_path: Path) -> None:
    repositories = tmp_path / "repositories"
    repositories.mkdir()
    instance, spec = _inputs(repositories)
    prepared = prepare_swe_smith_bundle(
        instances=(instance,),
        specs=(spec,),
        repositories_root=repositories,
        output_root=tmp_path / "prepared",
        executor=_DeterministicExecutor(),
    )
    first = load_jsonl(prepared.export_jsonl, SWESmithExportV1)[0]
    second_commit = "d" * 40
    second_task = first.task.model_copy(
        update={
            "task_id": "demo.bug_002",
            "provenance": first.task.provenance.model_copy(update={"base_commit": second_commit}),
        }
    )
    second_payload = first.model_dump(mode="json")
    second_payload.update(
        {
            "instance_id": second_task.task_id,
            "task": second_task.model_dump(mode="json"),
            "repository": first.repository.model_copy(
                update={
                    "commits": (second_commit,),
                    "content_fingerprints": (_digest("e"),),
                }
            ).model_dump(mode="json"),
            "admission_evidence": first.admission_evidence.model_copy(
                update={"task": second_task}
            ).model_dump(mode="json"),
        }
    )
    second = SWESmithExportV1.model_validate(second_payload)

    imported = SWESmithExportAdapterV1().import_records(
        (first, second),
        prepared.artifacts_root,
        dataset_id="multi-commit-fixture",
        dataset_version="1",
    )

    assert len(imported.repositories) == 1
    repository = imported.repositories[0]
    assert repository.commits == tuple(sorted((first.task.provenance.base_commit, second_commit)))
    assert repository.content_fingerprints == tuple(
        sorted((*first.repository.content_fingerprints, _digest("e")))
    )

    conflicting_task = first.task.model_copy(update={"task_id": "demo.bug_003"})
    conflicting_payload = first.model_dump(mode="json")
    conflicting_payload.update(
        {
            "instance_id": conflicting_task.task_id,
            "task": conflicting_task.model_dump(mode="json"),
            "repository": first.repository.model_copy(
                update={"content_fingerprints": (_digest("f"),)}
            ).model_dump(mode="json"),
            "admission_evidence": first.admission_evidence.model_copy(
                update={"task": conflicting_task}
            ).model_dump(mode="json"),
        }
    )
    conflicting = SWESmithExportV1.model_validate(conflicting_payload)
    with pytest.raises(ValueError, match="content fingerprint"):
        SWESmithExportAdapterV1().import_records(
            (first, conflicting),
            prepared.artifacts_root,
            dataset_id="conflicting-fingerprint-fixture",
            dataset_version="1",
        )


def test_preparation_rejects_dirty_checkout_without_publishing(tmp_path: Path) -> None:
    repositories = tmp_path / "repositories"
    repositories.mkdir()
    instance, spec = _inputs(repositories)
    (repositories / "demo" / "untracked.txt").write_text("dirty", encoding="utf-8")
    output = tmp_path / "prepared"

    with pytest.raises(ValueError, match="dirty"):
        prepare_swe_smith_bundle(
            instances=(instance,),
            specs=(spec,),
            repositories_root=repositories,
            output_root=output,
            executor=_DeterministicExecutor(),
        )

    assert not output.exists()


def test_official_loader_accepts_gather_json_and_rejects_unknown_fields(tmp_path: Path) -> None:
    repositories = tmp_path / "repositories"
    repositories.mkdir()
    instance, _ = _inputs(repositories)
    payload = instance.model_dump(mode="json")
    gathered = tmp_path / "gathered.json"
    gathered.write_text(json.dumps([payload]), encoding="utf-8")
    assert load_official_swe_smith_instances(gathered) == (instance,)

    payload["unknown"] = True
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(TaskDataError, match="invalid official"):
        load_official_swe_smith_instances(invalid)


class _FakeContainer:
    def __init__(self) -> None:
        self.removed = False
        self.archives = 0

    def put_archive(self, destination: str, archive: bytes) -> bool:
        assert destination == "/workspace"
        assert archive
        self.archives += 1
        return True

    def exec_run(self, argv: tuple[str, ...], **kwargs: Any) -> SimpleNamespace:
        del kwargs
        if "head" in argv:
            xml = (
                b'<testsuite><testcase classname="tests.test_calc" '
                b'name="test_add" time="0.01" /></testsuite>'
            )
            return SimpleNamespace(exit_code=0, output=(xml, b""))
        return SimpleNamespace(exit_code=0, output=(b"", b""))

    def remove(self, *, force: bool) -> None:
        assert force
        self.removed = True


class _FakeContainers:
    def __init__(self) -> None:
        self.created: list[_FakeContainer] = []

    def run(self, image: str, command: tuple[str, ...], **kwargs: Any) -> _FakeContainer:
        del kwargs
        assert image.endswith(f"@{_digest('b')}")
        assert command == ("sleep", "infinity")
        container = _FakeContainer()
        self.created.append(container)
        return container


def test_docker_admission_executor_uses_fresh_junit_backed_attempts(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    hidden = tmp_path / "hidden"
    snapshot.mkdir()
    hidden.mkdir()
    (snapshot / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (hidden / "test_calc.py").write_text("def test_add(): pass\n", encoding="utf-8")
    snapshot_sha256, _ = artifact_sha256(snapshot)
    suite = VerifierTestSuite(
        name="target",
        command=TrustedCommand(
            argv=(
                "python",
                "-m",
                "pytest",
                "/verifier-tests/test_calc.py",
                "--junitxml=/tmp/reporl-junit/target.xml",
            )
        ),
        junit_path="/tmp/reporl-junit/target.xml",
        expected_test_ids=("tests.test_calc::test_add",),
    )
    containers = _FakeContainers()
    client = SimpleNamespace(containers=containers)

    validation = DockerAdmissionExecutor(client=client).run_snapshot(  # type: ignore[arg-type]
        kind=SnapshotKind.REFERENCE,
        snapshot=snapshot,
        snapshot_sha256=snapshot_sha256,
        hidden_tests=hidden,
        image=f"example/verifier:fixed@{_digest('b')}",
        suites=(suite,),
        repetitions=3,
    )

    assert len(validation.outcomes) == 3
    assert all(outcome.is_pass for outcome in validation.outcomes)
    assert len(containers.created) == 3
    assert all(container.removed and container.archives == 2 for container in containers.created)
