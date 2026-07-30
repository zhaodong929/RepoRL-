"""Upstream task-adapter boundary and sealed JSONL ingestion.

RepoRL deliberately does not depend on a particular SWE-smith release. Dataset generators can
implement :class:`UpstreamTaskAdapter` and must emit the same strict ``VerifierManifest`` used by
the generic JSONL path before any task reaches materialization.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field, field_validator, model_validator

from reporl.schemas import StrictModel, TaskSpec
from reporl.tasks.admission import AdmissionEvidence, TaskAdmissionResult, validate_task_admission
from reporl.tasks.canonical import artifact_sha256
from reporl.tasks.dataset import (
    DatasetManifest,
    DatasetTaskEntry,
    SplitSeal,
    audit_sealed_split,
    seal_dataset_manifest,
)
from reporl.tasks.lineage import RepositoryRecord, normalize_source_url
from reporl.tasks.loader import load_verifier_manifests_jsonl
from reporl.tasks.manifest import (
    AgentTestSuite,
    ArtifactReference,
    VerifierManifest,
    VerifierTestSuite,
)


@runtime_checkable
class UpstreamTaskAdapter(Protocol):
    """Convert one generator record into RepoRL's sealed trust-boundary schema."""

    def adapt(self, record: Mapping[str, object], artifact_root: Path) -> VerifierManifest:
        """Validate and convert one upstream record without exposing hidden fields."""


@runtime_checkable
class SealedManifestSource(Protocol):
    """Load a complete set of already sealed verifier manifests."""

    def load(self) -> tuple[VerifierManifest, ...]:
        """Return unique, strictly validated manifests."""


@dataclass(frozen=True)
class JsonlSealedManifestSource:
    """Generic, dependency-free sealed-manifest source used by the CLI."""

    path: Path

    def load(self) -> tuple[VerifierManifest, ...]:
        return load_verifier_manifests_jsonl(self.path)


def _export_artifact_path(value: str) -> str:
    return ArtifactReference(
        path=value,
        sha256="sha256:" + "0" * 64,
        size_bytes=0,
    ).path


class SWESmithTestIdMapping(StrictModel):
    official_id: str = Field(min_length=1)
    junit_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def ids_are_safe(self) -> SWESmithTestIdMapping:
        if "\x00" in self.official_id or "\x00" in self.junit_id:
            raise ValueError("test ID mappings must not contain null bytes")
        return self


class SWESmithExportV1(StrictModel):
    """Versioned RepoRL envelope around an official SWE-smith gathered instance.

    The first seven task fields follow the public ``swesmith.harness.gather`` output documented
    at SWE-smith commit ``9b74ac0``. RepoRL enrichment supplies immutable local artifacts and
    admission evidence that the upstream format does not contain. Unknown versions and fields
    are rejected rather than guessed.
    """

    schema_version: Literal["reporl.swe-smith-export/v1"]
    instance_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
    repo: str = Field(min_length=1)
    patch: str = Field(min_length=1)
    FAIL_TO_PASS: tuple[str, ...] = Field(min_length=1)
    PASS_TO_PASS: tuple[str, ...] = Field(min_length=1)
    image_name: str = Field(min_length=1)
    problem_statement: str = Field(min_length=1)
    resolved_agent_image: str = Field(min_length=1)
    target_test_id_map: tuple[SWESmithTestIdMapping, ...] = Field(min_length=1)
    regression_test_id_map: tuple[SWESmithTestIdMapping, ...] = Field(min_length=1)

    task: TaskSpec
    repository: RepositoryRecord
    mutation_patch_path: str
    clean_snapshot_path: str
    buggy_snapshot_path: str
    reference_snapshot_path: str
    hidden_tests_path: str
    reference_patch_path: str
    verifier_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    agent_test_suites: tuple[AgentTestSuite, ...]
    verifier_test_suites: tuple[VerifierTestSuite, ...]
    admission_evidence: AdmissionEvidence

    @field_validator(
        "mutation_patch_path",
        "clean_snapshot_path",
        "buggy_snapshot_path",
        "reference_snapshot_path",
        "hidden_tests_path",
        "reference_patch_path",
    )
    @classmethod
    def validate_artifact_path(cls, value: str) -> str:
        return _export_artifact_path(value)

    @field_validator("FAIL_TO_PASS", "PASS_TO_PASS")
    @classmethod
    def validate_test_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() or "\x00" in value for value in values):
            raise ValueError("SWE-smith test IDs must be non-empty and contain no null bytes")
        if len(values) != len(set(values)):
            raise ValueError("SWE-smith test IDs must be unique")
        return values

    @model_validator(mode="after")
    def official_and_enriched_fields_match(self) -> SWESmithExportV1:
        if self.task.task_id != self.instance_id:
            raise ValueError("task_id must equal the SWE-smith instance_id")
        if self.task.issue != self.problem_statement:
            raise ValueError("TaskSpec issue must equal the SWE-smith problem_statement")
        if self.task.provenance.generator != "SWE-smith":
            raise ValueError("exported TaskSpec generator must be 'SWE-smith'")
        if self.repository.split != self.task.split:
            raise ValueError("repository and task splits must match")
        if self.repository.commits != (self.task.provenance.base_commit,):
            raise ValueError("each export must bind exactly its TaskSpec base commit")
        if len(self.repository.content_fingerprints) != 1:
            raise ValueError("each export must bind exactly one repository content fingerprint")
        if (
            self.repository.source_url
            != normalize_source_url(self.task.provenance.source_repository)
            or self.repository.lineage_group
            != self.task.provenance.lineage_group.strip().casefold()
            or self.task.provenance.base_commit not in self.repository.commits
        ):
            raise ValueError("repository record does not bind TaskSpec provenance")
        if not (
            self.resolved_agent_image == self.task.agent_image_digest
            or self.resolved_agent_image.endswith(f"@{self.task.agent_image_digest}")
        ):
            raise ValueError("resolved agent image must be pinned to the TaskSpec digest")
        if self.admission_evidence.task != self.task:
            raise ValueError("admission evidence must describe the exported TaskSpec")
        agent_names = {suite.name for suite in self.agent_test_suites}
        verifier: dict[str, VerifierTestSuite] = {
            suite.name: suite for suite in self.verifier_test_suites
        }
        expected_names = set(self.task.available_test_suites)
        if agent_names != expected_names or set(verifier) != expected_names:
            raise ValueError("exported suites must exactly match TaskSpec aliases")
        mappings = (
            ("target", self.FAIL_TO_PASS, self.target_test_id_map),
            ("regression", self.PASS_TO_PASS, self.regression_test_id_map),
        )
        for name, official_ids, mapping in mappings:
            mapped_official = tuple(item.official_id for item in mapping)
            mapped_junit = tuple(item.junit_id for item in mapping)
            if len(mapped_official) != len(set(mapped_official)) or len(mapped_junit) != len(
                set(mapped_junit)
            ):
                raise ValueError(f"{name} test ID mapping must be one-to-one")
            if set(mapped_official) != set(official_ids):
                raise ValueError(f"{name} mapping must cover every official SWE-smith test ID")
            if set(mapped_junit) != set(verifier[name].expected_test_ids):
                raise ValueError(f"{name} mapping must cover every canonical JUnit test ID")
        return self


@dataclass(frozen=True)
class AdaptedSWESmithTask:
    manifest: VerifierManifest
    evidence: AdmissionEvidence
    admission: TaskAdmissionResult
    repository: RepositoryRecord


@dataclass(frozen=True)
class SWESmithImportBundle:
    manifests: tuple[VerifierManifest, ...]
    admission_evidence: tuple[AdmissionEvidence, ...]
    admission_results: tuple[TaskAdmissionResult, ...]
    repositories: tuple[RepositoryRecord, ...]
    dataset_manifest: DatasetManifest
    split_seal: SplitSeal


class SWESmithExportAdapterV1:
    """Concrete adapter for ``reporl.swe-smith-export/v1`` JSONL records."""

    def adapt(self, record: Mapping[str, object], artifact_root: Path) -> VerifierManifest:
        return self.adapt_record(
            SWESmithExportV1.model_validate(record, strict=True), artifact_root
        ).manifest

    def adapt_record(self, record: SWESmithExportV1, artifact_root: Path) -> AdaptedSWESmithTask:
        root = artifact_root.resolve(strict=True)

        def reference(path: str) -> ArtifactReference:
            candidate = root / Path(path)
            resolved = candidate.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError(f"SWE-smith artifact escapes artifact_root: {path}") from error
            digest, size_bytes = artifact_sha256(resolved)
            return ArtifactReference(path=path, sha256=digest, size_bytes=size_bytes)

        mutation = (root / Path(record.mutation_patch_path)).resolve(strict=True)
        try:
            mutation.relative_to(root)
        except ValueError as error:
            raise ValueError("SWE-smith mutation patch escapes artifact_root") from error
        if not mutation.is_file() or mutation.read_text(encoding="utf-8") != record.patch:
            raise ValueError("SWE-smith patch differs from mutation_patch_path")
        clean = reference(record.clean_snapshot_path)
        buggy = reference(record.buggy_snapshot_path)
        repaired = reference(record.reference_snapshot_path)
        hidden = reference(record.hidden_tests_path)
        repair_patch = reference(record.reference_patch_path)
        admission = validate_task_admission(record.admission_evidence)
        if not admission.admitted:
            raise ValueError(f"SWE-smith task failed RepoRL admission: {record.instance_id}")
        if (
            record.admission_evidence.clean.snapshot_sha256 != clean.sha256
            or record.admission_evidence.buggy.snapshot_sha256 != buggy.sha256
            or record.admission_evidence.reference.snapshot_sha256 != repaired.sha256
        ):
            raise ValueError("SWE-smith admission snapshot hashes differ from exported artifacts")
        manifest = VerifierManifest(
            task=record.task,
            verifier_image_digest=record.verifier_image_digest,
            clean_snapshot=clean,
            buggy_snapshot=buggy,
            reference_snapshot=repaired,
            hidden_tests=hidden,
            reference_patch=repair_patch,
            agent_test_suites=record.agent_test_suites,
            test_suites=record.verifier_test_suites,
            admission_result_sha256=admission.digest(),
        )
        return AdaptedSWESmithTask(
            manifest=manifest,
            evidence=record.admission_evidence,
            admission=admission,
            repository=record.repository,
        )

    def import_records(
        self,
        records: tuple[SWESmithExportV1, ...],
        artifact_root: Path,
        *,
        dataset_id: str,
        dataset_version: str,
    ) -> SWESmithImportBundle:
        adapted = tuple(self.adapt_record(record, artifact_root) for record in records)
        task_ids = tuple(item.manifest.task.task_id for item in adapted)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("duplicate instance_id in SWE-smith export")
        repositories_by_id: dict[str, RepositoryRecord] = {}
        fingerprints_by_commit: dict[tuple[str, str], str] = {}
        for item in adapted:
            commit_key = (
                item.repository.repository_id,
                item.manifest.task.provenance.base_commit,
            )
            fingerprint = item.repository.content_fingerprints[0]
            prior_fingerprint = fingerprints_by_commit.get(commit_key)
            if prior_fingerprint is not None and prior_fingerprint != fingerprint:
                raise ValueError("conflicting SWE-smith content fingerprint for repository commit")
            fingerprints_by_commit[commit_key] = fingerprint
            existing = repositories_by_id.get(item.repository.repository_id)
            if existing is None:
                repositories_by_id[item.repository.repository_id] = item.repository
                continue
            identity = (existing.source_url, existing.lineage_group, existing.split)
            candidate_identity = (
                item.repository.source_url,
                item.repository.lineage_group,
                item.repository.split,
            )
            if identity != candidate_identity:
                raise ValueError("conflicting SWE-smith repository records")
            repositories_by_id[item.repository.repository_id] = RepositoryRecord(
                repository_id=existing.repository_id,
                source_url=existing.source_url,
                lineage_group=existing.lineage_group,
                split=existing.split,
                commits=(*existing.commits, *item.repository.commits),
                content_fingerprints=(
                    *existing.content_fingerprints,
                    *item.repository.content_fingerprints,
                ),
            )
        entries = tuple(
            DatasetTaskEntry(
                task_id=item.manifest.task.task_id,
                split=item.manifest.task.split,
                repository_id=item.repository.repository_id,
                source_url=item.repository.source_url,
                lineage_group=item.repository.lineage_group,
                verifier_manifest_sha256=item.manifest.digest(),
                admission_result_sha256=item.admission.digest(),
            )
            for item in adapted
        )
        dataset = DatasetManifest(dataset_id=dataset_id, version=dataset_version, tasks=entries)
        seal = seal_dataset_manifest(dataset)
        repositories = tuple(
            sorted(repositories_by_id.values(), key=lambda item: item.repository_id)
        )
        audit = audit_sealed_split(dataset, seal, repositories)
        if not audit.is_valid:
            issues = ", ".join(issue.value for issue in audit.issues)
            raise ValueError(f"SWE-smith export has an invalid split audit: {issues}")
        return SWESmithImportBundle(
            manifests=tuple(item.manifest for item in adapted),
            admission_evidence=tuple(item.evidence for item in adapted),
            admission_results=tuple(item.admission for item in adapted),
            repositories=repositories,
            dataset_manifest=dataset,
            split_seal=seal,
        )
