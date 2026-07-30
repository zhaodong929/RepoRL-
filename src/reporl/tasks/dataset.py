"""Dataset manifest hashing, split sealing, and seal verification."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator

from reporl.schemas import DatasetSplit, StrictModel
from reporl.tasks.canonical import canonical_sha256
from reporl.tasks.lineage import (
    LineageAuditReport,
    RepositoryRecord,
    audit_repository_splits,
    normalize_source_url,
)

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"


class DatasetTaskEntry(StrictModel):
    task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
    split: DatasetSplit
    repository_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{1,127}$")
    source_url: str
    lineage_group: str = Field(min_length=1, max_length=256)
    verifier_manifest_sha256: str = Field(pattern=_DIGEST_PATTERN)
    admission_result_sha256: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator("source_url")
    @classmethod
    def normalize_url(cls, url: str) -> str:
        return normalize_source_url(url)

    @field_validator("lineage_group")
    @classmethod
    def normalize_lineage(cls, lineage: str) -> str:
        normalized = lineage.strip().casefold()
        if not normalized:
            raise ValueError("lineage group must be non-empty")
        return normalized


class DatasetManifest(StrictModel):
    schema_version: Literal[1] = 1
    dataset_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
    version: str = Field(min_length=1, max_length=128)
    tasks: tuple[DatasetTaskEntry, ...] = Field(min_length=1)

    @field_validator("tasks")
    @classmethod
    def sort_and_validate_tasks(
        cls, tasks: tuple[DatasetTaskEntry, ...]
    ) -> tuple[DatasetTaskEntry, ...]:
        task_ids = tuple(task.task_id for task in tasks)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("dataset task IDs must be unique")
        return tuple(sorted(tasks, key=lambda task: task.task_id))

    def digest(self) -> str:
        return canonical_sha256(self)


class SplitDigest(StrictModel):
    split: DatasetSplit
    task_count: int = Field(ge=0)
    assignment_sha256: str = Field(pattern=_DIGEST_PATTERN)


def split_assignment_sha256(manifest: DatasetManifest) -> str:
    """Hash only immutable split and repository assignments."""

    assignments = tuple(
        {
            "lineage_group": task.lineage_group,
            "repository_id": task.repository_id,
            "source_url": task.source_url,
            "split": task.split.value,
            "task_id": task.task_id,
        }
        for task in manifest.tasks
    )
    return canonical_sha256(assignments)


def _per_split_digests(manifest: DatasetManifest) -> tuple[SplitDigest, ...]:
    digests: list[SplitDigest] = []
    for split in DatasetSplit:
        tasks = tuple(
            {
                "lineage_group": task.lineage_group,
                "repository_id": task.repository_id,
                "source_url": task.source_url,
                "task_id": task.task_id,
            }
            for task in manifest.tasks
            if task.split is split
        )
        digests.append(
            SplitDigest(
                split=split,
                task_count=len(tasks),
                assignment_sha256=canonical_sha256(tasks),
            )
        )
    return tuple(sorted(digests, key=lambda digest: digest.split.value))


class SplitSeal(StrictModel):
    schema_version: Literal[1] = 1
    dataset_id: str
    dataset_version: str
    manifest_sha256: str = Field(pattern=_DIGEST_PATTERN)
    assignment_sha256: str = Field(pattern=_DIGEST_PATTERN)
    split_digests: tuple[SplitDigest, ...]

    @field_validator("split_digests")
    @classmethod
    def validate_split_digests(cls, digests: tuple[SplitDigest, ...]) -> tuple[SplitDigest, ...]:
        splits = tuple(digest.split for digest in digests)
        if set(splits) != set(DatasetSplit) or len(splits) != len(set(splits)):
            raise ValueError("a seal must contain exactly one digest for every split")
        return tuple(sorted(digests, key=lambda digest: digest.split.value))

    def digest(self) -> str:
        return canonical_sha256(self)


def seal_dataset_manifest(manifest: DatasetManifest) -> SplitSeal:
    """Create the reproducible aggregate hashes published before evaluation."""

    return SplitSeal(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.version,
        manifest_sha256=manifest.digest(),
        assignment_sha256=split_assignment_sha256(manifest),
        split_digests=_per_split_digests(manifest),
    )


class SealedSplitIssue(StrEnum):
    DATASET_ID_MISMATCH = "dataset_id_mismatch"
    DATASET_VERSION_MISMATCH = "dataset_version_mismatch"
    MANIFEST_HASH_MISMATCH = "manifest_hash_mismatch"
    ASSIGNMENT_HASH_MISMATCH = "assignment_hash_mismatch"
    PER_SPLIT_HASH_MISMATCH = "per_split_hash_mismatch"
    REPOSITORY_MISSING = "repository_missing"
    REPOSITORY_AMBIGUOUS = "repository_ambiguous"
    REPOSITORY_METADATA_MISMATCH = "repository_metadata_mismatch"
    LINEAGE_LEAKAGE = "lineage_leakage"


class SealedSplitAudit(StrictModel):
    manifest_sha256: str = Field(pattern=_DIGEST_PATTERN)
    assignment_sha256: str = Field(pattern=_DIGEST_PATTERN)
    lineage_audit: LineageAuditReport
    issues: tuple[SealedSplitIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues


def audit_sealed_split(
    manifest: DatasetManifest,
    seal: SplitSeal,
    repositories: tuple[RepositoryRecord, ...] | list[RepositoryRecord],
    *,
    check_fingerprints: bool = True,
) -> SealedSplitAudit:
    """Verify seal hashes, task/repository metadata, and lineage isolation together."""

    manifest_hash = manifest.digest()
    assignment_hash = split_assignment_sha256(manifest)
    per_split = _per_split_digests(manifest)
    issues: set[SealedSplitIssue] = set()

    if seal.dataset_id != manifest.dataset_id:
        issues.add(SealedSplitIssue.DATASET_ID_MISMATCH)
    if seal.dataset_version != manifest.version:
        issues.add(SealedSplitIssue.DATASET_VERSION_MISMATCH)
    if seal.manifest_sha256 != manifest_hash:
        issues.add(SealedSplitIssue.MANIFEST_HASH_MISMATCH)
    if seal.assignment_sha256 != assignment_hash:
        issues.add(SealedSplitIssue.ASSIGNMENT_HASH_MISMATCH)
    if seal.split_digests != per_split:
        issues.add(SealedSplitIssue.PER_SPLIT_HASH_MISMATCH)

    by_repository: dict[str, list[RepositoryRecord]] = {}
    for repository in repositories:
        by_repository.setdefault(repository.repository_id, []).append(repository)
    for task in manifest.tasks:
        candidates = by_repository.get(task.repository_id, [])
        if not candidates:
            issues.add(SealedSplitIssue.REPOSITORY_MISSING)
            continue
        if len(candidates) != 1:
            issues.add(SealedSplitIssue.REPOSITORY_AMBIGUOUS)
            continue
        repository = candidates[0]
        if (
            repository.split != task.split
            or repository.source_url != task.source_url
            or repository.lineage_group != task.lineage_group
        ):
            issues.add(SealedSplitIssue.REPOSITORY_METADATA_MISMATCH)

    lineage_audit = audit_repository_splits(repositories, check_fingerprints=check_fingerprints)
    if not lineage_audit.is_clean:
        issues.add(SealedSplitIssue.LINEAGE_LEAKAGE)

    return SealedSplitAudit(
        manifest_sha256=manifest_hash,
        assignment_sha256=assignment_hash,
        lineage_audit=lineage_audit,
        issues=tuple(sorted(issues, key=lambda issue: issue.value)),
    )
