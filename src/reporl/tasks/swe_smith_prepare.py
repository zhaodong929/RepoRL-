"""Fail-closed preparation of real SWE-smith instances for RepoRL admission."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from reporl.schemas import StrictModel, TaskSpec
from reporl.tasks.adapters import SWESmithExportV1, SWESmithTestIdMapping
from reporl.tasks.admission import (
    AdmissionEvidence,
    LeakScan,
    LicenseReview,
    RegressionExpectation,
    SnapshotKind,
    TaskAdmissionResult,
    validate_task_admission,
)
from reporl.tasks.admission_docker import (
    DockerAdmissionConfig,
    DockerAdmissionExecutor,
    SnapshotAdmissionExecutor,
)
from reporl.tasks.canonical import artifact_sha256, canonical_json, canonical_sha256
from reporl.tasks.lineage import RepositoryRecord, normalize_source_url
from reporl.tasks.loader import TaskDataError, load_jsonl
from reporl.tasks.manifest import AgentTestSuite, VerifierTestSuite
from reporl.tools.patch import PatchPolicy

SWE_SMITH_COMMIT = "9b74ac08118a85c39c356802f7961893af73e07f"
_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"


class OfficialSWESmithInstance(StrictModel):
    """Exact gathered-instance fields at the pinned SWE-smith commit."""

    instance_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
    repo: str = Field(min_length=1)
    patch: str = Field(min_length=1)
    FAIL_TO_PASS: tuple[str, ...] = Field(min_length=1)
    PASS_TO_PASS: tuple[str, ...] = Field(min_length=1)
    image_name: str = Field(min_length=1)
    problem_statement: str | None = Field(default=None, min_length=1)

    @field_validator("patch")
    @classmethod
    def patch_is_a_git_diff(cls, value: str) -> str:
        if "\x00" in value or "diff --git " not in value:
            raise ValueError("SWE-smith patch must be a null-free Git diff")
        return value

    @field_validator("FAIL_TO_PASS", "PASS_TO_PASS")
    @classmethod
    def tests_are_canonical(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() or "\x00" in value for value in values):
            raise ValueError("SWE-smith test IDs must be non-empty and null-free")
        if len(values) != len(set(values)):
            raise ValueError("SWE-smith test IDs must be unique")
        return values


def _relative_path(value: str, *, label: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or "\x00" in normalized
        or path.is_absolute()
        or ".." in path.parts
        or any(part in {"", "."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a normalized relative path")
    return path.as_posix()


def _pinned_digest(image: str) -> str:
    if image.startswith("sha256:") and len(image) == 71:
        digest = image
    elif "@" in image:
        digest = image.rsplit("@", maxsplit=1)[-1]
    else:
        raise ValueError("Docker image must be pinned by @sha256 digest")
    if len(digest) != 71 or not digest.startswith("sha256:"):
        raise ValueError("Docker image must contain a full SHA-256 digest")
    try:
        int(digest.removeprefix("sha256:"), 16)
    except ValueError as error:
        raise ValueError("Docker image digest must be lowercase hexadecimal") from error
    if digest != digest.lower():
        raise ValueError("Docker image digest must be lowercase hexadecimal")
    return digest


class SWESmithPreparationSpec(StrictModel):
    """Trusted enrichment inputs that SWE-smith gather cannot provide."""

    schema_version: Literal["reporl.swe-smith-prepare/v1"]
    instance_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
    gather_repo: str = Field(min_length=1)
    repository_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{1,127}$")
    repository_path: str
    task: TaskSpec
    agent_image: str = Field(min_length=1)
    verifier_image: str = Field(min_length=1)
    hidden_test_paths: tuple[str, ...] = Field(min_length=1)
    target_test_id_map: tuple[SWESmithTestIdMapping, ...] = Field(min_length=1)
    regression_test_id_map: tuple[SWESmithTestIdMapping, ...] = Field(min_length=1)
    agent_test_suites: tuple[AgentTestSuite, ...]
    verifier_test_suites: tuple[VerifierTestSuite, ...]
    license_review: LicenseReview
    leak_scan: LeakScan
    buggy_regression_expectation: RegressionExpectation = RegressionExpectation.PASSES

    @field_validator("repository_path")
    @classmethod
    def validate_repository_path(cls, value: str) -> str:
        return _relative_path(value, label="repository_path")

    @field_validator("hidden_test_paths")
    @classmethod
    def validate_hidden_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_relative_path(value, label="hidden test path") for value in values)
        if len(normalized) != len(set(normalized)):
            raise ValueError("hidden test paths must be unique")
        for index, left in enumerate(normalized):
            if any(_paths_overlap(left, right) for right in normalized[index + 1 :]):
                raise ValueError("hidden test paths must not overlap")
        return tuple(sorted(normalized))

    @model_validator(mode="after")
    def trusted_inputs_are_consistent(self) -> SWESmithPreparationSpec:
        if self.task.task_id != self.instance_id:
            raise ValueError("preparation instance_id must equal TaskSpec task_id")
        if len(self.task.provenance.base_commit) != 40:
            raise ValueError("preparation requires a full 40-character base commit")
        if self.task.provenance.generator != "SWE-smith":
            raise ValueError("TaskSpec generator must be 'SWE-smith'")
        if self.task.provenance.generator_version != SWE_SMITH_COMMIT:
            raise ValueError("TaskSpec generator_version must equal the pinned SWE-smith commit")
        if _pinned_digest(self.agent_image) != self.task.agent_image_digest:
            raise ValueError("agent image does not match TaskSpec agent_image_digest")
        _pinned_digest(self.verifier_image)
        if self.license_review.repository_url != normalize_source_url(
            self.task.provenance.source_repository
        ):
            raise ValueError("license review repository differs from TaskSpec provenance")
        if self.leak_scan.agent_image_digest != self.task.agent_image_digest:
            raise ValueError("leak scan image digest differs from TaskSpec")
        expected = set(self.task.available_test_suites)
        if {suite.name for suite in self.agent_test_suites} != expected:
            raise ValueError("agent suites must match TaskSpec aliases")
        if {suite.name for suite in self.verifier_test_suites} != expected:
            raise ValueError("verifier suites must match TaskSpec aliases")
        hidden = self.hidden_test_paths
        for suite in self.agent_test_suites:
            command_text = "\n".join(
                (*suite.command.argv, *(item.value for item in suite.command.environment))
            )
            if any(path in command_text for path in hidden):
                raise ValueError("agent suite command must not name a hidden test path")
        return self


class PreparationMetadata(StrictModel):
    schema_version: Literal[1] = 1
    swe_smith_commit: Literal["9b74ac08118a85c39c356802f7961893af73e07f"] = (
        "9b74ac08118a85c39c356802f7961893af73e07f"
    )
    instance_count: int = Field(ge=1)
    instances_sha256: str = Field(pattern=_DIGEST_PATTERN)
    preparation_specs_sha256: str = Field(pattern=_DIGEST_PATTERN)
    enriched_export_sha256: str = Field(pattern=_DIGEST_PATTERN)


@dataclass(frozen=True)
class PreparedArtifactPaths:
    mutation_patch: str
    clean_snapshot: str
    buggy_snapshot: str
    reference_snapshot: str
    hidden_tests: str
    reference_patch: str
    repository_fingerprint: str


@dataclass(frozen=True)
class PreparedSWESmithBundle:
    root: Path
    artifacts_root: Path
    export_jsonl: Path
    metadata_path: Path
    metadata: PreparationMetadata


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TaskDataError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise TaskDataError(f"non-standard JSON constant: {value}")


def _decode_json(raw: str, source: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except TaskDataError:
        raise
    except (json.JSONDecodeError, ValueError) as error:
        raise TaskDataError(f"invalid JSON in {source}: {error}") from error


def load_official_swe_smith_instances(path: Path) -> tuple[OfficialSWESmithInstance, ...]:
    """Load the pinned gather JSON array or a line-delimited equivalent."""

    try:
        raw = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise TaskDataError(f"cannot read SWE-smith instances from {path}: {error}") from error
    if not raw:
        raise TaskDataError(f"SWE-smith instance file is empty: {path}")
    payloads: list[Any]
    if path.suffix.casefold() == ".jsonl":
        payloads = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                raise TaskDataError(f"blank JSONL record at {path}:{line_number}")
            payloads.append(_decode_json(line, f"{path}:{line_number}"))
    else:
        decoded = _decode_json(raw, str(path))
        if not isinstance(decoded, list):
            raise TaskDataError("official SWE-smith gather JSON must contain an array")
        payloads = decoded
    try:
        records = tuple(
            OfficialSWESmithInstance.model_validate_json(
                json.dumps(payload, allow_nan=False, separators=(",", ":")),
                strict=True,
            )
            for payload in payloads
        )
    except ValueError as error:
        raise TaskDataError(f"invalid official SWE-smith instance: {error}") from error
    task_ids = tuple(record.instance_id for record in records)
    if not records or len(task_ids) != len(set(task_ids)):
        raise TaskDataError("SWE-smith instances must be non-empty with unique instance_id values")
    return records


def _git(repository: Path, *arguments: str, text: bool = True) -> subprocess.CompletedProcess[Any]:
    command = ("git", "-C", str(repository), *arguments)
    result: subprocess.CompletedProcess[Any]
    if text:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    else:
        result = subprocess.run(command, check=False, capture_output=True)
    if result.returncode != 0:
        stderr = (
            result.stderr
            if isinstance(result.stderr, str)
            else result.stderr.decode(errors="replace")
        )
        raise RuntimeError(f"git {' '.join(arguments)} failed: {stderr.strip()}")
    return result


def _extract_git_archive(archive_bytes: bytes, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not name.parts:
                raise ValueError("Git archive contains an unsafe path")
            if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                raise ValueError("repository snapshots must contain only files and directories")
            target = destination.joinpath(*name.parts)
            resolved_parent = target.parent.resolve()
            if destination.resolve() not in (resolved_parent, *resolved_parent.parents):
                raise ValueError("Git archive entry escapes snapshot root")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Git archive file has no payload")
            with source, target.open("xb") as handle:
                shutil.copyfileobj(source, handle)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)


def _initialize_temporary_repository(path: Path) -> None:
    _git(path, "init", "--quiet")
    _git(path, "config", "user.name", "RepoRL Task Preparation")
    _git(path, "config", "user.email", "preparation@invalid")
    _git(path, "add", "--all", "--", ".")
    _git(path, "-c", "commit.gpgSign=false", "commit", "--quiet", "-m", "baseline")


def _remove_git_directory(path: Path) -> None:
    git_directory = (path / ".git").resolve(strict=True)
    try:
        git_directory.relative_to(path.resolve(strict=True))
    except ValueError as error:
        raise ValueError("temporary Git directory escaped task workspace") from error
    for directory, child_directories, files in os.walk(git_directory, topdown=False):
        current = Path(directory)
        for name in files:
            (current / name).chmod(stat.S_IREAD | stat.S_IWRITE)
        for name in child_directories:
            (current / name).chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
    git_directory.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
    shutil.rmtree(git_directory)


def _paths_overlap(left: str, right: str) -> bool:
    left_path, right_path = PurePosixPath(left), PurePosixPath(right)
    return (
        left_path == right_path
        or left_path in right_path.parents
        or right_path in left_path.parents
    )


def _copy_hidden_tests(source: Path, destination: Path, paths: tuple[str, ...]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for relative in paths:
        source_path = source.joinpath(*PurePosixPath(relative).parts)
        if not source_path.exists():
            raise ValueError(f"declared hidden test path does not exist: {relative}")
        target = destination.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            shutil.copytree(source_path, target)
        elif source_path.is_file():
            shutil.copy2(source_path, target)
        else:
            raise ValueError(f"hidden test path is not a regular file or directory: {relative}")


def _remove_hidden_tests(snapshot: Path, paths: tuple[str, ...]) -> None:
    root = snapshot.resolve(strict=True)
    for relative in paths:
        target = snapshot.joinpath(*PurePosixPath(relative).parts)
        resolved = target.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError("hidden test removal escaped snapshot") from error
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()


def _validate_checked_out_repository(
    repositories_root: Path,
    spec: SWESmithPreparationSpec,
) -> Path:
    root = repositories_root.resolve(strict=True)
    repository = (root / Path(spec.repository_path)).resolve(strict=True)
    try:
        repository.relative_to(root)
    except ValueError as error:
        raise ValueError("checked-out repository escapes repositories_root") from error
    top_level = Path(str(_git(repository, "rev-parse", "--show-toplevel").stdout).strip()).resolve()
    if top_level != repository:
        raise ValueError("repository_path must identify the checkout root")
    head = str(_git(repository, "rev-parse", "HEAD").stdout).strip()
    if head != spec.task.provenance.base_commit:
        raise ValueError(f"checked-out repository HEAD differs for {spec.instance_id}")
    status = str(_git(repository, "status", "--porcelain", "--untracked-files=all").stdout)
    if status:
        raise ValueError(f"checked-out repository is dirty: {spec.instance_id}")
    return repository


def _prepare_artifacts(
    instance: OfficialSWESmithInstance,
    spec: SWESmithPreparationSpec,
    repositories_root: Path,
    artifact_root: Path,
    work_root: Path,
) -> PreparedArtifactPaths:
    repository = _validate_checked_out_repository(repositories_root, spec)
    archive = _git(
        repository,
        "archive",
        "--format=tar",
        spec.task.provenance.base_commit,
        text=False,
    )
    archive_bytes = bytes(archive.stdout)
    full_clean = work_root / "full-clean"
    _extract_git_archive(archive_bytes, full_clean)
    repository_fingerprint, _ = artifact_sha256(full_clean)

    relative_root = PurePosixPath("tasks") / spec.instance_id
    task_root = artifact_root.joinpath(*relative_root.parts)
    state_root = task_root / "states"
    sealed_root = task_root / "sealed"
    mutation_path = sealed_root / "mutation.patch"
    reference_patch_path = sealed_root / "reference.patch"
    hidden_root = sealed_root / "hidden-tests"
    clean_snapshot = state_root / "clean"
    buggy_snapshot = state_root / "buggy"
    reference_snapshot = state_root / "reference"
    sealed_root.mkdir(parents=True, exist_ok=False)
    mutation_path.write_text(instance.patch, encoding="utf-8", newline="\n")

    buggy_full = work_root / "buggy-full"
    shutil.copytree(full_clean, buggy_full)
    _initialize_temporary_repository(buggy_full)
    _git(buggy_full, "apply", "--check", "--binary", "--whitespace=nowarn", str(mutation_path))
    _git(buggy_full, "apply", "--binary", "--whitespace=nowarn", str(mutation_path))
    changed_raw = str(_git(buggy_full, "diff", "--name-only", "-z", "--").stdout)
    changed_paths = tuple(path for path in changed_raw.split("\x00") if path)
    if not changed_paths:
        raise ValueError(f"SWE-smith mutation produced no repository change: {spec.instance_id}")
    for changed in changed_paths:
        if any(_paths_overlap(changed, hidden) for hidden in spec.hidden_test_paths):
            raise ValueError("mutation patch modifies a declared hidden test artifact")
    reference_patch = str(_git(buggy_full, "diff", "-R", "--binary", "--no-ext-diff", "--").stdout)
    inspection = PatchPolicy(
        allowed_paths=spec.task.allowed_paths,
        forbidden_globs=spec.task.forbidden_globs,
        max_patch_bytes=spec.task.budgets.max_patch_bytes,
    ).inspect(reference_patch)
    if not inspection.accepted:
        violations = ", ".join(item.code.value for item in inspection.violations)
        raise ValueError(f"reference repair violates TaskSpec patch policy: {violations}")
    reference_patch_path.write_text(reference_patch, encoding="utf-8", newline="\n")

    _copy_hidden_tests(full_clean, hidden_root, spec.hidden_test_paths)
    shutil.copytree(full_clean, clean_snapshot)
    shutil.copytree(full_clean, reference_snapshot)
    _remove_git_directory(buggy_full)
    shutil.copytree(buggy_full, buggy_snapshot)
    for snapshot in (clean_snapshot, buggy_snapshot, reference_snapshot):
        _remove_hidden_tests(snapshot, spec.hidden_test_paths)

    with tempfile.TemporaryDirectory(prefix="repair-check-", dir=work_root) as check_text:
        repaired = Path(check_text) / "repo"
        shutil.copytree(buggy_snapshot, repaired)
        _initialize_temporary_repository(repaired)
        _git(
            repaired,
            "apply",
            "--check",
            "--binary",
            "--whitespace=nowarn",
            str(reference_patch_path),
        )
        _git(
            repaired,
            "apply",
            "--binary",
            "--whitespace=nowarn",
            str(reference_patch_path),
        )
        _remove_git_directory(repaired)
        repaired_digest, _ = artifact_sha256(repaired)
        reference_digest, _ = artifact_sha256(reference_snapshot)
        if repaired_digest != reference_digest:
            raise ValueError("generated reference patch does not reproduce reference snapshot")

    def relative(path: Path) -> str:
        return path.relative_to(artifact_root).as_posix()

    return PreparedArtifactPaths(
        mutation_patch=relative(mutation_path),
        clean_snapshot=relative(clean_snapshot),
        buggy_snapshot=relative(buggy_snapshot),
        reference_snapshot=relative(reference_snapshot),
        hidden_tests=relative(hidden_root),
        reference_patch=relative(reference_patch_path),
        repository_fingerprint=repository_fingerprint,
    )


def _pair_inputs(
    instances: tuple[OfficialSWESmithInstance, ...],
    specs: tuple[SWESmithPreparationSpec, ...],
) -> tuple[tuple[OfficialSWESmithInstance, SWESmithPreparationSpec], ...]:
    spec_by_id = {spec.instance_id: spec for spec in specs}
    if len(spec_by_id) != len(specs):
        raise ValueError("preparation specs contain duplicate instance_id values")
    instance_ids = {instance.instance_id for instance in instances}
    if instance_ids != set(spec_by_id):
        missing = sorted(instance_ids - set(spec_by_id))
        extra = sorted(set(spec_by_id) - instance_ids)
        raise ValueError(f"instance/spec IDs differ; missing={missing}, extra={extra}")
    return tuple((instance, spec_by_id[instance.instance_id]) for instance in instances)


def _validate_pair(
    instance: OfficialSWESmithInstance,
    spec: SWESmithPreparationSpec,
) -> None:
    if instance.repo != spec.gather_repo:
        raise ValueError(f"gather repository mismatch: {instance.instance_id}")
    if instance.problem_statement is not None and instance.problem_statement != spec.task.issue:
        raise ValueError(f"gather problem statement mismatch: {instance.instance_id}")
    if not (
        spec.agent_image == instance.image_name
        or spec.agent_image.startswith(f"{instance.image_name}@")
    ):
        raise ValueError(f"resolved agent image differs from gather image: {instance.instance_id}")
    verifier: dict[str, VerifierTestSuite] = {
        suite.name: suite for suite in spec.verifier_test_suites
    }
    mappings = (
        ("target", instance.FAIL_TO_PASS, spec.target_test_id_map),
        ("regression", instance.PASS_TO_PASS, spec.regression_test_id_map),
    )
    for name, official_ids, mapping in mappings:
        if set(item.official_id for item in mapping) != set(official_ids):
            raise ValueError(f"{name} mapping differs from official test IDs")
        if set(item.junit_id for item in mapping) != set(verifier[name].expected_test_ids):
            raise ValueError(f"{name} mapping differs from canonical JUnit IDs")


def _collect_admission(
    spec: SWESmithPreparationSpec,
    paths: PreparedArtifactPaths,
    artifact_root: Path,
    executor: SnapshotAdmissionExecutor,
    repetitions: int,
) -> tuple[AdmissionEvidence, TaskAdmissionResult]:
    suites = {suite.name: suite for suite in spec.verifier_test_suites}

    def artifact(path: str) -> Path:
        return artifact_root.joinpath(*PurePosixPath(path).parts)

    clean_digest, _ = artifact_sha256(artifact(paths.clean_snapshot))
    buggy_digest, _ = artifact_sha256(artifact(paths.buggy_snapshot))
    reference_digest, _ = artifact_sha256(artifact(paths.reference_snapshot))
    hidden = artifact(paths.hidden_tests)
    clean = executor.run_snapshot(
        kind=SnapshotKind.CLEAN,
        snapshot=artifact(paths.clean_snapshot),
        snapshot_sha256=clean_digest,
        hidden_tests=hidden,
        image=spec.verifier_image,
        suites=(suites["regression"],),
        repetitions=repetitions,
    )
    buggy = executor.run_snapshot(
        kind=SnapshotKind.BUGGY,
        snapshot=artifact(paths.buggy_snapshot),
        snapshot_sha256=buggy_digest,
        hidden_tests=hidden,
        image=spec.verifier_image,
        suites=(suites["target"], suites["regression"]),
        repetitions=repetitions,
    )
    reference = executor.run_snapshot(
        kind=SnapshotKind.REFERENCE,
        snapshot=artifact(paths.reference_snapshot),
        snapshot_sha256=reference_digest,
        hidden_tests=hidden,
        image=spec.verifier_image,
        suites=(suites["target"], suites["regression"]),
        repetitions=repetitions,
    )
    evidence = AdmissionEvidence(
        task=spec.task,
        clean=clean,
        buggy=buggy,
        reference=reference,
        buggy_regression_expectation=spec.buggy_regression_expectation,
        required_repetitions=repetitions,
        license_review=spec.license_review,
        leak_scan=spec.leak_scan,
    )
    result = validate_task_admission(evidence)
    if not result.admitted:
        failures = ", ".join(failure.value for failure in result.failures)
        raise ValueError(f"task failed admission ({spec.instance_id}): {failures}")
    return evidence, result


def prepare_swe_smith_bundle(
    *,
    instances: tuple[OfficialSWESmithInstance, ...],
    specs: tuple[SWESmithPreparationSpec, ...],
    repositories_root: Path,
    output_root: Path,
    executor: SnapshotAdmissionExecutor,
    repetitions: int = 3,
) -> PreparedSWESmithBundle:
    """Build portable artifacts and an admitted enriched export, publishing atomically."""

    output_root = output_root.resolve()
    if repetitions < 3:
        raise ValueError("admission requires at least three repetitions")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite preparation output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    pairs = _pair_inputs(instances, specs)
    with tempfile.TemporaryDirectory(
        prefix=".reporl-swe-smith-prepare-",
        dir=output_root.parent,
    ) as temporary_text:
        temporary = Path(temporary_text)
        bundle_root = temporary / "bundle"
        artifact_root = bundle_root / "artifacts"
        work_root = temporary / "work"
        artifact_root.mkdir(parents=True)
        work_root.mkdir()
        exports: list[SWESmithExportV1] = []
        for instance, spec in sorted(pairs, key=lambda item: item[0].instance_id):
            _validate_pair(instance, spec)
            instance_work = work_root / spec.instance_id
            instance_work.mkdir()
            paths = _prepare_artifacts(
                instance,
                spec,
                repositories_root,
                artifact_root,
                instance_work,
            )
            evidence, _ = _collect_admission(
                spec,
                paths,
                artifact_root,
                executor,
                repetitions,
            )
            repository = RepositoryRecord(
                repository_id=spec.repository_id,
                source_url=spec.task.provenance.source_repository,
                lineage_group=spec.task.provenance.lineage_group,
                split=spec.task.split,
                commits=(spec.task.provenance.base_commit,),
                content_fingerprints=(paths.repository_fingerprint,),
            )
            exports.append(
                SWESmithExportV1(
                    schema_version="reporl.swe-smith-export/v1",
                    instance_id=instance.instance_id,
                    repo=instance.repo,
                    patch=instance.patch,
                    FAIL_TO_PASS=instance.FAIL_TO_PASS,
                    PASS_TO_PASS=instance.PASS_TO_PASS,
                    image_name=instance.image_name,
                    problem_statement=spec.task.issue,
                    resolved_agent_image=spec.agent_image,
                    target_test_id_map=spec.target_test_id_map,
                    regression_test_id_map=spec.regression_test_id_map,
                    task=spec.task,
                    repository=repository,
                    mutation_patch_path=paths.mutation_patch,
                    clean_snapshot_path=paths.clean_snapshot,
                    buggy_snapshot_path=paths.buggy_snapshot,
                    reference_snapshot_path=paths.reference_snapshot,
                    hidden_tests_path=paths.hidden_tests,
                    reference_patch_path=paths.reference_patch,
                    verifier_image_digest=_pinned_digest(spec.verifier_image),
                    agent_test_suites=spec.agent_test_suites,
                    verifier_test_suites=spec.verifier_test_suites,
                    admission_evidence=evidence,
                )
            )
        export_payload = "".join(f"{canonical_json(record)}\n" for record in exports)
        export_path = bundle_root / "swe-smith-export-v1.jsonl"
        export_path.write_text(export_payload, encoding="utf-8", newline="\n")
        metadata = PreparationMetadata(
            instance_count=len(exports),
            instances_sha256=canonical_sha256(instances),
            preparation_specs_sha256=canonical_sha256(specs),
            enriched_export_sha256=(
                f"sha256:{hashlib.sha256(export_payload.encode()).hexdigest()}"
            ),
        )
        metadata_path = bundle_root / "preparation-metadata.json"
        metadata_path.write_text(f"{canonical_json(metadata)}\n", encoding="utf-8", newline="\n")
        os.rename(bundle_root, output_root)
    return PreparedSWESmithBundle(
        root=output_root,
        artifacts_root=output_root / "artifacts",
        export_jsonl=output_root / "swe-smith-export-v1.jsonl",
        metadata_path=output_root / "preparation-metadata.json",
        metadata=metadata,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--preparation-specs", type=Path, required=True)
    parser.add_argument("--repositories-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--memory-limit", default="12g")
    parser.add_argument("--nano-cpus", type=int, default=4_000_000_000)
    parser.add_argument("--pids-limit", type=int, default=512)
    parser.add_argument("--workspace-size", default="8g")
    parser.add_argument("--user", default="1000:1000")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        instances = load_official_swe_smith_instances(args.instances)
        specs = load_jsonl(args.preparation_specs, SWESmithPreparationSpec)
        executor = DockerAdmissionExecutor(
            DockerAdmissionConfig(
                user=args.user,
                memory_limit=args.memory_limit,
                nano_cpus=args.nano_cpus,
                pids_limit=args.pids_limit,
                workspace_size=args.workspace_size,
            )
        )
        result = prepare_swe_smith_bundle(
            instances=instances,
            specs=specs,
            repositories_root=args.repositories_root,
            output_root=args.output_root,
            executor=executor,
            repetitions=args.repetitions,
        )
        print(
            json.dumps(
                {
                    "artifact_root": str(result.artifacts_root),
                    "export_jsonl": str(result.export_jsonl),
                    "metadata": str(result.metadata_path),
                    "metadata_sha256": canonical_sha256(result.metadata),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"SWE-smith preparation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
