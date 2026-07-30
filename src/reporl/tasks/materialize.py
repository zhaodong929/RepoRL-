"""Materialize sealed task manifests into split-specific rollout runtime JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import Field

from reporl.rollouts.config import AgentRunSpec, RolloutTaskSpec
from reporl.sandbox.base import CommandSpec
from reporl.schemas import DatasetSplit, StrictModel
from reporl.tasks.adapters import (
    JsonlSealedManifestSource,
    SealedManifestSource,
    SWESmithExportAdapterV1,
    SWESmithExportV1,
    SWESmithImportBundle,
)
from reporl.tasks.admission import AdmissionEvidence, TaskAdmissionResult, validate_task_admission
from reporl.tasks.canonical import artifact_sha256, canonical_json, canonical_sha256
from reporl.tasks.dataset import (
    DatasetManifest,
    SealedSplitAudit,
    SplitSeal,
    audit_sealed_split,
)
from reporl.tasks.lineage import RepositoryRecord, normalize_source_url
from reporl.tasks.loader import load_json, load_jsonl
from reporl.tasks.manifest import ArtifactReference, TrustedCommand, VerifierManifest
from reporl.verifier.models import VerifierRunSpec, VerifierSuiteSpec

_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST)
_RUNTIME_NAME = {split: f"{split.value}-runtimes.jsonl" for split in _SPLITS}
_METADATA_NAME = "materialization-metadata.json"


class RuntimeFileSeal(StrictModel):
    split: DatasetSplit
    path: str
    task_count: int = Field(ge=1)
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1)
    split_membership_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class MaterializationMetadata(StrictModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    dataset_id: str
    dataset_version: str
    dataset_manifest_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    split_seal_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    split_assignment_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    verifier_manifests_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    repository_records_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    lineage_audit: SealedSplitAudit
    runtime_files: tuple[RuntimeFileSeal, ...]

    def digest(self) -> str:
        return canonical_sha256(self)


@dataclass(frozen=True)
class DatasetTrustContext:
    manifest: DatasetManifest
    seal: SplitSeal
    repositories: tuple[RepositoryRecord, ...]
    audit: SealedSplitAudit

    @property
    def split_seal_sha256(self) -> str:
        return self.seal.digest()

    @property
    def repository_records_sha256(self) -> str:
        return self.audit.lineage_audit.records_sha256

    def membership_sha256(self, split: DatasetSplit) -> str:
        return next(
            item.assignment_sha256 for item in self.seal.split_digests if item.split == split
        )


@dataclass(frozen=True)
class MaterializationPlan:
    splits: Mapping[DatasetSplit, tuple[RolloutTaskSpec, ...]]
    context: DatasetTrustContext
    verifier_manifests_sha256: str


@dataclass(frozen=True)
class PublishedMaterialization:
    runtime_paths: Mapping[DatasetSplit, Path]
    metadata_path: Path
    metadata: MaterializationMetadata


def write_swe_smith_import(output_dir: Path, bundle: SWESmithImportBundle) -> dict[str, Path]:
    """Publish generic sealed inputs produced by the versioned SWE-smith adapter."""

    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    names = {
        "manifests": "verifier-manifests.jsonl",
        "dataset_manifest": "dataset-manifest.json",
        "split_seal": "split-seal.json",
        "repositories": "repositories.jsonl",
        "admission_evidence": "admission-evidence.jsonl",
        "admission_results": "admission-results.jsonl",
    }
    paths = {key: destination / name for key, name in names.items()}
    existing = tuple(path for path in paths.values() if path.exists())
    if existing:
        raise FileExistsError(f"refusing to overwrite imported task data: {existing[0]}")

    def jsonl(records: tuple[object, ...]) -> str:
        return "".join(f"{canonical_json(record)}\n" for record in records)

    payloads = {
        "manifests": jsonl(bundle.manifests),
        "dataset_manifest": f"{canonical_json(bundle.dataset_manifest)}\n",
        "split_seal": f"{canonical_json(bundle.split_seal)}\n",
        "repositories": jsonl(bundle.repositories),
        "admission_evidence": jsonl(bundle.admission_evidence),
        "admission_results": jsonl(bundle.admission_results),
    }
    temporary: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for key, target in paths.items():
            temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            with temp.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(payloads[key])
                handle.flush()
                os.fsync(handle.fileno())
            temporary[target] = temp
        for target in paths.values():
            os.link(temporary[target], target)
            published.append(target)
        return paths
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def _image_reference(repository: str | None, digest: str) -> str:
    if repository is None:
        return digest
    if (
        not repository.strip()
        or repository != repository.strip()
        or "@" in repository
        or any(character.isspace() for character in repository)
    ):
        raise ValueError("image repository must be a non-empty Docker name without @ or whitespace")
    return f"{repository}@{digest}"


def _runtime_command(command: TrustedCommand) -> CommandSpec:
    workdir = "/workspace/repo" if command.cwd == "." else f"/workspace/repo/{command.cwd}"
    return CommandSpec(
        argv=command.argv,
        timeout_seconds=min(command.timeout_seconds, 3_600),
        environment={item.name: item.value for item in command.environment},
        workdir=workdir,
    )


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_artifact(
    artifact_root: Path,
    reference: ArtifactReference,
) -> Path:
    relative = Path(*PurePosixPath(reference.path).parts)
    candidate = artifact_root / relative
    cursor = artifact_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"sealed artifact path contains a symlink: {reference.path}")
    resolved = candidate.resolve(strict=True)
    if not _contains(artifact_root, resolved):
        raise ValueError(f"sealed artifact escapes artifact_root: {reference.path}")
    digest, size_bytes = artifact_sha256(resolved)
    if digest != reference.sha256 or size_bytes != reference.size_bytes:
        raise ValueError(f"sealed artifact does not match its digest: {reference.path}")
    return resolved


def _validate_repository_snapshot(path: Path, label: str) -> None:
    if not path.is_dir():
        raise ValueError(f"{label} must be a directory")
    if any(entry.name == ".git" for entry in path.rglob(".git")):
        raise ValueError(f"{label} must be sanitized and contain no .git entry")


def _validate_suite_workdirs(manifest: VerifierManifest, buggy_snapshot: Path) -> None:
    suites = tuple(manifest.agent_test_suites) + tuple(manifest.test_suites)
    for suite in suites:
        relative = Path(*PurePosixPath(suite.command.cwd).parts)
        workdir = buggy_snapshot if suite.command.cwd == "." else buggy_snapshot / relative
        resolved = workdir.resolve(strict=True)
        if not resolved.is_dir() or not _contains(buggy_snapshot, resolved):
            raise ValueError(f"suite {suite.name!r} has an invalid repository cwd")


def _validated_artifacts(
    manifest: VerifierManifest,
    artifact_root: Path,
) -> Mapping[str, Path]:
    references = {
        "clean_snapshot": manifest.clean_snapshot,
        "buggy_snapshot": manifest.buggy_snapshot,
        "reference_snapshot": manifest.reference_snapshot,
        "hidden_tests": manifest.hidden_tests,
        "reference_patch": manifest.reference_patch,
    }
    resolved = {
        name: _resolve_artifact(artifact_root, reference) for name, reference in references.items()
    }
    for label in ("clean_snapshot", "buggy_snapshot", "reference_snapshot"):
        _validate_repository_snapshot(resolved[label], label)
    if not resolved["hidden_tests"].is_dir():
        raise ValueError("hidden_tests must be a directory")
    if not resolved["reference_patch"].is_file():
        raise ValueError("reference_patch must be a regular file")

    directory_names = ("clean_snapshot", "buggy_snapshot", "reference_snapshot", "hidden_tests")
    for index, left_name in enumerate(directory_names):
        for right_name in directory_names[index + 1 :]:
            left, right = resolved[left_name], resolved[right_name]
            if _contains(left, right) or _contains(right, left):
                raise ValueError(f"sealed artifact directories overlap: {left_name}, {right_name}")
    if any(_contains(resolved[name], resolved["reference_patch"]) for name in directory_names):
        raise ValueError("reference_patch must not be stored inside a snapshot or hidden tests")

    _validate_suite_workdirs(manifest, resolved["buggy_snapshot"])
    return resolved


def validate_dataset_trust(
    manifests: tuple[VerifierManifest, ...],
    dataset_manifest: DatasetManifest,
    split_seal: SplitSeal,
    repositories: tuple[RepositoryRecord, ...],
    admission_evidence: tuple[AdmissionEvidence, ...],
    admission_results: tuple[TaskAdmissionResult, ...],
) -> DatasetTrustContext:
    """Bind task manifests to admitted evidence and a clean repository-level split seal."""

    audit = audit_sealed_split(dataset_manifest, split_seal, repositories)
    if not audit.is_valid:
        issues = ", ".join(issue.value for issue in audit.issues)
        raise ValueError(f"sealed dataset audit failed: {issues}")

    def unique_by_task_id(records: tuple[object, ...], label: str) -> dict[str, object]:
        indexed: dict[str, object] = {}
        for record in records:
            task_id = getattr(record, "task_id", None)
            if not isinstance(task_id, str):
                task = getattr(record, "task", None)
                task_id = getattr(task, "task_id", None)
            if not isinstance(task_id, str):
                raise ValueError(f"{label} record has no task_id")
            if task_id in indexed:
                raise ValueError(f"duplicate task_id in {label}: {task_id}")
            indexed[task_id] = record
        return indexed

    sealed_by_id = {manifest.task.task_id: manifest for manifest in manifests}
    entries_by_id = {entry.task_id: entry for entry in dataset_manifest.tasks}
    evidence_by_id = unique_by_task_id(admission_evidence, "admission evidence")
    results_by_id = unique_by_task_id(admission_results, "admission results")
    expected_ids = set(entries_by_id)
    sources = {
        "sealed manifests": set(sealed_by_id),
        "admission evidence": set(evidence_by_id),
        "admission results": set(results_by_id),
    }
    for label, task_ids in sources.items():
        if task_ids != expected_ids:
            missing = sorted(expected_ids - task_ids)
            extra = sorted(task_ids - expected_ids)
            raise ValueError(
                f"{label} task IDs differ from DatasetManifest; missing={missing}, extra={extra}"
            )

    repositories_by_id = {repository.repository_id: repository for repository in repositories}
    if len(repositories_by_id) != len(repositories):
        raise ValueError("repository records must have unique repository_id values")
    for task_id, entry in entries_by_id.items():
        sealed = sealed_by_id[task_id]
        evidence = evidence_by_id[task_id]
        result = results_by_id[task_id]
        assert isinstance(evidence, AdmissionEvidence)
        assert isinstance(result, TaskAdmissionResult)
        computed_result = validate_task_admission(evidence)
        if computed_result != result:
            raise ValueError(f"admission result does not match recomputed evidence: {task_id}")
        if not result.admitted:
            raise ValueError(f"task was not admitted: {task_id}")
        if (
            sealed.digest() != entry.verifier_manifest_sha256
            or result.digest() != entry.admission_result_sha256
            or result.digest() != sealed.admission_result_sha256
        ):
            raise ValueError(f"sealed manifest or admission digest mismatch: {task_id}")
        if evidence.task != sealed.task:
            raise ValueError(f"admission TaskSpec differs from sealed TaskSpec: {task_id}")
        if (
            evidence.clean.snapshot_sha256 != sealed.clean_snapshot.sha256
            or evidence.buggy.snapshot_sha256 != sealed.buggy_snapshot.sha256
            or evidence.reference.snapshot_sha256 != sealed.reference_snapshot.sha256
        ):
            raise ValueError(f"admission snapshot digest differs from sealed artifacts: {task_id}")
        if (
            entry.split != sealed.task.split
            or entry.source_url != normalize_source_url(sealed.task.provenance.source_repository)
            or entry.lineage_group != sealed.task.provenance.lineage_group.strip().casefold()
        ):
            raise ValueError(f"DatasetManifest metadata differs from sealed TaskSpec: {task_id}")
        repository = repositories_by_id[entry.repository_id]
        if sealed.task.provenance.base_commit not in repository.commits:
            raise ValueError(f"repository record does not bind the task base commit: {task_id}")

    return DatasetTrustContext(
        manifest=dataset_manifest,
        seal=split_seal,
        repositories=repositories,
        audit=audit,
    )


def materialize_manifest(
    manifest: VerifierManifest,
    artifact_root: Path,
    context: DatasetTrustContext,
    *,
    agent_image_repository: str | None = None,
    verifier_image_repository: str | None = None,
    baseline_target_pass_fraction: float = 0,
) -> RolloutTaskSpec:
    """Validate all sealed inputs and construct one immutable rollout runtime."""

    root = artifact_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("artifact_root must be a directory")
    _validated_artifacts(manifest, root)
    agent_suites: dict[str, CommandSpec] = {
        suite.name: _runtime_command(suite.command) for suite in manifest.agent_test_suites
    }
    verifier_suites = tuple(
        VerifierSuiteSpec(
            name=suite.name,
            command=_runtime_command(suite.command),
            junit_path=suite.junit_path,
            expected_test_ids=suite.expected_test_ids,
        )
        for suite in manifest.test_suites
    )
    runtime = RolloutTaskSpec(
        sealed_manifest_sha256=manifest.digest(),
        dataset_manifest_sha256=context.manifest.digest(),
        split_seal_sha256=context.split_seal_sha256,
        split_assignment_sha256=context.seal.assignment_sha256,
        split_membership_sha256=context.membership_sha256(manifest.task.split),
        repository_records_sha256=context.repository_records_sha256,
        sealed_manifest=manifest,
        artifact_root=Path("."),
        task=manifest.agent_view(),
        agent=AgentRunSpec(
            repository_snapshot=Path(manifest.buggy_snapshot.path),
            image=_image_reference(agent_image_repository, manifest.task.agent_image_digest),
            suite_commands=agent_suites,
        ),
        verifier=VerifierRunSpec(
            task_id=manifest.task.task_id,
            image=_image_reference(verifier_image_repository, manifest.verifier_image_digest),
            image_digest=manifest.verifier_image_digest,
            repository_snapshot=Path(manifest.buggy_snapshot.path),
            hidden_tests_path=Path(manifest.hidden_tests.path),
            suites=verifier_suites,
            allowed_paths=manifest.task.allowed_paths,
            forbidden_globs=manifest.task.forbidden_globs,
            max_patch_bytes=manifest.task.budgets.max_patch_bytes,
        ),
        baseline_target_pass_fraction=baseline_target_pass_fraction,
    )
    runtime.validate_materialized_paths(root)
    return runtime


def materialize_source(
    source: SealedManifestSource,
    artifact_root: Path,
    *,
    dataset_manifest: DatasetManifest,
    split_seal: SplitSeal,
    repositories: tuple[RepositoryRecord, ...],
    admission_evidence: tuple[AdmissionEvidence, ...],
    admission_results: tuple[TaskAdmissionResult, ...],
    agent_image_repository: str | None = None,
    verifier_image_repository: str | None = None,
    baseline_target_pass_fraction: float = 0,
) -> MaterializationPlan:
    """Materialize and deterministically partition a complete three-way dataset."""

    manifests = source.load()
    context = validate_dataset_trust(
        manifests,
        dataset_manifest,
        split_seal,
        repositories,
        admission_evidence,
        admission_results,
    )
    grouped: dict[DatasetSplit, list[RolloutTaskSpec]] = {split: [] for split in _SPLITS}
    for manifest in manifests:
        runtime = materialize_manifest(
            manifest,
            artifact_root,
            context,
            agent_image_repository=agent_image_repository,
            verifier_image_repository=verifier_image_repository,
            baseline_target_pass_fraction=baseline_target_pass_fraction,
        )
        grouped[runtime.task.split].append(runtime)
    missing = tuple(split.value for split in _SPLITS if not grouped[split])
    if missing:
        raise ValueError(
            "sealed manifests must contain every dataset split; missing: " + ", ".join(missing)
        )
    splits = {
        split: tuple(sorted(grouped[split], key=lambda item: item.task.task_id))
        for split in _SPLITS
    }
    return MaterializationPlan(
        splits=splits,
        context=context,
        verifier_manifests_sha256=canonical_sha256(
            tuple(sorted(manifests, key=lambda item: item.task.task_id))
        ),
    )


def write_runtime_splits(
    output_dir: Path,
    plan: MaterializationPlan,
) -> PublishedMaterialization:
    """Publish all split files without overwriting existing runtime evidence."""

    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    targets = {split: destination / _RUNTIME_NAME[split] for split in _SPLITS}
    metadata_target = destination / _METADATA_NAME
    all_targets = (*targets.values(), metadata_target)
    existing = tuple(path for path in all_targets if path.exists())
    if existing:
        raise FileExistsError(f"refusing to overwrite runtime file: {existing[0]}")

    payloads: dict[DatasetSplit, str] = {}
    runtime_seals: list[RuntimeFileSeal] = []
    for split in _SPLITS:
        records = tuple(plan.splits.get(split, ()))
        if not records:
            raise ValueError(f"cannot write an empty {split.value} runtime split")
        if any(record.task.split != split for record in records):
            raise ValueError(f"runtime record is assigned to the wrong {split.value} output")
        payload = "".join(f"{canonical_json(record)}\n" for record in records)
        payloads[split] = payload
        encoded = payload.encode("utf-8")
        runtime_seals.append(
            RuntimeFileSeal(
                split=split,
                path=_RUNTIME_NAME[split],
                task_count=len(records),
                sha256=f"sha256:{hashlib.sha256(encoded).hexdigest()}",
                size_bytes=len(encoded),
                split_membership_sha256=plan.context.membership_sha256(split),
            )
        )

    metadata = MaterializationMetadata(
        dataset_id=plan.context.manifest.dataset_id,
        dataset_version=plan.context.manifest.version,
        dataset_manifest_sha256=plan.context.manifest.digest(),
        split_seal_sha256=plan.context.split_seal_sha256,
        split_assignment_sha256=plan.context.seal.assignment_sha256,
        verifier_manifests_sha256=plan.verifier_manifests_sha256,
        repository_records_sha256=plan.context.repository_records_sha256,
        lineage_audit=plan.context.audit,
        runtime_files=tuple(runtime_seals),
    )
    metadata_payload = f"{canonical_json(metadata)}\n"

    temporary: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for split, target in targets.items():
            temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            with temp.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(payloads[split])
                handle.flush()
                os.fsync(handle.fileno())
            temporary[target] = temp
        metadata_temp = metadata_target.with_name(f".{metadata_target.name}.{uuid.uuid4().hex}.tmp")
        with metadata_temp.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(metadata_payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary[metadata_target] = metadata_temp
        for target in all_targets:
            os.link(temporary[target], target)
            published.append(target)
        return PublishedMaterialization(
            runtime_paths=targets,
            metadata_path=metadata_target,
            metadata=metadata,
        )
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def verify_runtime_splits(
    runtime_dir: Path,
    artifact_root: Path,
    *,
    expected_metadata_sha256: str | None = None,
) -> tuple[dict[DatasetSplit, int], MaterializationMetadata]:
    """Revalidate transferred runtime files and every referenced sealed artifact."""

    metadata = load_json(runtime_dir / _METADATA_NAME, MaterializationMetadata)
    if expected_metadata_sha256 is not None and metadata.digest() != expected_metadata_sha256:
        raise ValueError("materialization metadata digest differs from the expected digest")
    if not metadata.lineage_audit.is_valid or not metadata.lineage_audit.lineage_audit.is_clean:
        raise ValueError("materialization metadata does not contain a clean sealed lineage audit")
    seals = {item.split: item for item in metadata.runtime_files}
    if set(seals) != set(_SPLITS) or len(seals) != len(metadata.runtime_files):
        raise ValueError("materialization metadata must seal exactly one file per split")

    seen: set[str] = set()
    counts: dict[DatasetSplit, int] = {}
    for split in _SPLITS:
        path = runtime_dir / _RUNTIME_NAME[split]
        file_sha256, size_bytes = artifact_sha256(path)
        seal = seals[split]
        if (
            seal.path != _RUNTIME_NAME[split]
            or seal.sha256 != file_sha256
            or seal.size_bytes != size_bytes
        ):
            raise ValueError(f"runtime file differs from materialization metadata: {path}")
        records = load_jsonl(path, RolloutTaskSpec)
        if any(record.task.split != split for record in records):
            raise ValueError(f"runtime file contains a record from another split: {path}")
        for record in records:
            if record.task.task_id in seen:
                raise ValueError("duplicate task_id across runtime split files")
            seen.add(record.task.task_id)
            if (
                record.dataset_manifest_sha256 != metadata.dataset_manifest_sha256
                or record.split_seal_sha256 != metadata.split_seal_sha256
                or record.split_assignment_sha256 != metadata.split_assignment_sha256
                or record.split_membership_sha256 != seal.split_membership_sha256
                or record.repository_records_sha256 != metadata.repository_records_sha256
            ):
                raise ValueError(f"runtime binding differs from materialization metadata: {path}")
            record.validate_materialized_paths(artifact_root)
        counts[split] = len(records)
        if counts[split] != seal.task_count:
            raise ValueError(f"runtime task count differs from materialization metadata: {path}")
    return counts, metadata


def _render_counts(
    paths: Mapping[DatasetSplit, Path],
    counts: Mapping[DatasetSplit, int],
    *,
    metadata_path: Path,
    metadata_sha256: str,
) -> str:
    payload = {
        "metadata": {"path": str(metadata_path), "sha256": metadata_sha256},
        "splits": {
            split.value: {"count": counts[split], "path": str(paths[split])} for split in _SPLITS
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize", help="create split runtime JSONL files")
    materialize.add_argument("--manifests", type=Path, required=True)
    materialize.add_argument("--artifact-root", "--artifacts-root", type=Path, required=True)
    materialize.add_argument("--dataset-manifest", type=Path, required=True)
    materialize.add_argument("--split-seal", type=Path, required=True)
    materialize.add_argument("--repositories", type=Path, required=True)
    materialize.add_argument("--admission-evidence", type=Path, required=True)
    materialize.add_argument("--admission-results", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.add_argument("--agent-image-repository")
    materialize.add_argument("--verifier-image-repository")
    materialize.add_argument("--baseline-target-pass-fraction", type=float, default=0)

    verify = subparsers.add_parser("verify", help="revalidate materialized runtime files")
    verify.add_argument("--runtime-dir", type=Path, required=True)
    verify.add_argument("--artifact-root", "--artifacts-root", type=Path, required=True)
    verify.add_argument("--expected-metadata-sha256")

    swe_smith = subparsers.add_parser(
        "import-swe-smith",
        help="convert a reporl.swe-smith-export/v1 JSONL file into sealed task inputs",
    )
    swe_smith.add_argument("--export-jsonl", type=Path, required=True)
    swe_smith.add_argument("--artifact-root", "--artifacts-root", type=Path, required=True)
    swe_smith.add_argument("--output-dir", type=Path, required=True)
    swe_smith.add_argument("--dataset-id", required=True)
    swe_smith.add_argument("--dataset-version", required=True)

    fixture = subparsers.add_parser(
        "build-fixture", help="create a deterministic no-Docker materialization fixture"
    )
    fixture.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build-fixture":
            from reporl.tasks.fixture import build_materialization_fixture

            fixture = build_materialization_fixture(args.output_root)
            print(
                json.dumps(
                    {key: str(value) for key, value in fixture.items()}, indent=2, sort_keys=True
                )
            )
            return 0
        if args.command == "verify":
            counts, metadata = verify_runtime_splits(
                args.runtime_dir,
                args.artifact_root,
                expected_metadata_sha256=args.expected_metadata_sha256,
            )
            paths = {split: args.runtime_dir / _RUNTIME_NAME[split] for split in _SPLITS}
            print(
                _render_counts(
                    paths,
                    counts,
                    metadata_path=args.runtime_dir / _METADATA_NAME,
                    metadata_sha256=metadata.digest(),
                )
            )
            return 0
        if args.command == "import-swe-smith":
            records = load_jsonl(args.export_jsonl, SWESmithExportV1)
            bundle = SWESmithExportAdapterV1().import_records(
                records,
                args.artifact_root,
                dataset_id=args.dataset_id,
                dataset_version=args.dataset_version,
            )
            imported = write_swe_smith_import(args.output_dir, bundle)
            print(
                json.dumps(
                    {key: str(value) for key, value in imported.items()}, indent=2, sort_keys=True
                )
            )
            return 0

        source = JsonlSealedManifestSource(args.manifests)
        plan = materialize_source(
            source,
            args.artifact_root,
            dataset_manifest=load_json(args.dataset_manifest, DatasetManifest),
            split_seal=load_json(args.split_seal, SplitSeal),
            repositories=load_jsonl(args.repositories, RepositoryRecord),
            admission_evidence=load_jsonl(args.admission_evidence, AdmissionEvidence),
            admission_results=load_jsonl(args.admission_results, TaskAdmissionResult),
            agent_image_repository=args.agent_image_repository,
            verifier_image_repository=args.verifier_image_repository,
            baseline_target_pass_fraction=args.baseline_target_pass_fraction,
        )
        published = write_runtime_splits(args.output_dir, plan)
        counts = {split: len(records) for split, records in plan.splits.items()}
        print(
            _render_counts(
                published.runtime_paths,
                counts,
                metadata_path=published.metadata_path,
                metadata_sha256=published.metadata.digest(),
            )
        )
        return 0
    except (OSError, ValueError) as error:
        print(f"materialization failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
