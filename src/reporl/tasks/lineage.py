"""Repository identity normalization and cross-split lineage auditing."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from enum import StrEnum
from urllib.parse import unquote, urlsplit

from pydantic import Field, field_validator

from reporl.schemas import DatasetSplit, StrictModel
from reporl.tasks.canonical import canonical_json, canonical_sha256

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{7,64}$")
_SCP_URL = re.compile(r"^(?:[^@/]+@)?(?P<host>[^:/]+):(?P<path>[^?#]+)$")


def normalize_source_url(value: str) -> str:
    """Normalize common HTTPS, SSH, git, and scp-style repository URLs.

    The scheme and credentials are deliberately discarded. Hosting services treat repository
    paths case-insensitively in the common case, so both host and path are case-folded to avoid a
    trivial split leak through URL spelling.
    """

    candidate = value.strip().replace("\\", "/")
    if not candidate or "\x00" in candidate:
        raise ValueError("source URL must be non-empty and contain no null bytes")

    scp_match = _SCP_URL.fullmatch(candidate) if "://" not in candidate else None
    if scp_match is not None:
        host = scp_match.group("host")
        path = scp_match.group("path")
    else:
        parsed = urlsplit(candidate if "://" in candidate else f"https://{candidate}")
        if parsed.query or parsed.fragment:
            raise ValueError("source URL must not contain a query or fragment")
        host = parsed.hostname or ""
        path = parsed.path
        if parsed.port is not None:
            default_port = (
                (parsed.scheme == "http" and parsed.port == 80)
                or (parsed.scheme == "https" and parsed.port == 443)
                or (parsed.scheme == "ssh" and parsed.port == 22)
            )
            if not default_port:
                host = f"{host}:{parsed.port}"

    host = host.casefold().removeprefix("www.")
    path = re.sub(r"/+", "/", unquote(path)).strip("/")
    if path.casefold().endswith(".git"):
        path = path[:-4]
    path = path.strip("/").casefold()
    if not host or not path or path in {".", ".."} or ".." in path.split("/"):
        raise ValueError("source URL must identify a hosted repository")
    return f"{host}/{path}"


class RepositoryRecord(StrictModel):
    """Repository-level evidence used before assigning task splits."""

    repository_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{1,127}$")
    source_url: str
    lineage_group: str = Field(min_length=1, max_length=256)
    split: DatasetSplit
    commits: tuple[str, ...] = ()
    content_fingerprints: tuple[str, ...] = ()

    @field_validator("source_url")
    @classmethod
    def normalize_url(cls, source_url: str) -> str:
        return normalize_source_url(source_url)

    @field_validator("lineage_group")
    @classmethod
    def normalize_lineage(cls, lineage_group: str) -> str:
        normalized = lineage_group.strip().casefold()
        if not normalized:
            raise ValueError("lineage group must be non-empty")
        return normalized

    @field_validator("commits")
    @classmethod
    def normalize_commits(cls, commits: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(commit.strip().casefold() for commit in commits)
        if any(_COMMIT_PATTERN.fullmatch(commit) is None for commit in normalized):
            raise ValueError("commit IDs must be lowercase hexadecimal Git object IDs")
        return tuple(sorted(set(normalized)))

    @field_validator("content_fingerprints")
    @classmethod
    def normalize_fingerprints(cls, fingerprints: tuple[str, ...]) -> tuple[str, ...]:
        if any(re.fullmatch(_DIGEST_PATTERN, item) is None for item in fingerprints):
            raise ValueError("content fingerprints must be tagged SHA-256 digests")
        return tuple(sorted(set(fingerprints)))


class LineageConflictKind(StrEnum):
    DIRECT_LINEAGE = "direct_lineage"
    SOURCE_URL = "source_url"
    SHARED_COMMIT = "shared_commit"
    CONTENT_FINGERPRINT = "content_fingerprint"


class LineageConflict(StrictModel):
    kind: LineageConflictKind
    evidence: str = Field(min_length=1)
    repository_ids: tuple[str, ...] = Field(min_length=1)
    splits: tuple[DatasetSplit, ...] = Field(min_length=2)


class LineageAuditReport(StrictModel):
    record_count: int = Field(ge=0)
    records_sha256: str = Field(pattern=_DIGEST_PATTERN)
    fingerprints_checked: bool
    conflicts: tuple[LineageConflict, ...]

    @property
    def is_clean(self) -> bool:
        return not self.conflicts


def _cross_split_conflicts(
    records: tuple[RepositoryRecord, ...],
    kind: LineageConflictKind,
    evidence_getter: Callable[[RepositoryRecord], tuple[str, ...]],
) -> list[LineageConflict]:
    grouped: dict[str, list[RepositoryRecord]] = defaultdict(list)
    for record in records:
        for evidence in evidence_getter(record):
            grouped[evidence].append(record)

    conflicts: list[LineageConflict] = []
    for evidence, group in sorted(grouped.items()):
        splits = tuple(sorted({item.split for item in group}, key=lambda split: split.value))
        if len(splits) < 2:
            continue
        repository_ids = tuple(sorted({item.repository_id for item in group}))
        conflicts.append(
            LineageConflict(
                kind=kind,
                evidence=evidence,
                repository_ids=repository_ids,
                splits=splits,
            )
        )
    return conflicts


def audit_repository_splits(
    records: tuple[RepositoryRecord, ...] | list[RepositoryRecord],
    *,
    check_fingerprints: bool = True,
) -> LineageAuditReport:
    """Detect known ways one repository family can cross dataset splits."""

    ordered = tuple(sorted(records, key=canonical_json))
    evidence_sets: list[
        tuple[LineageConflictKind, Callable[[RepositoryRecord], tuple[str, ...]]]
    ] = [
        (
            LineageConflictKind.DIRECT_LINEAGE,
            lambda record: (record.lineage_group,),
        ),
        (
            LineageConflictKind.SOURCE_URL,
            lambda record: (record.source_url,),
        ),
        (
            LineageConflictKind.SHARED_COMMIT,
            lambda record: record.commits,
        ),
    ]
    if check_fingerprints:
        evidence_sets.append(
            (
                LineageConflictKind.CONTENT_FINGERPRINT,
                lambda record: record.content_fingerprints,
            )
        )

    conflicts: list[LineageConflict] = []
    for kind, evidence in evidence_sets:
        conflicts.extend(_cross_split_conflicts(ordered, kind, evidence))
    conflicts.sort(key=lambda item: (item.kind.value, item.evidence, item.repository_ids))

    return LineageAuditReport(
        record_count=len(ordered),
        records_sha256=canonical_sha256(ordered),
        fingerprints_checked=check_fingerprints,
        conflicts=tuple(conflicts),
    )
