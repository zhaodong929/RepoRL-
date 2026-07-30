"""Deterministic three-state admission checks for generated repair tasks."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from reporl.schemas import StrictModel, TaskSpec
from reporl.tasks.canonical import canonical_sha256
from reporl.tasks.lineage import normalize_source_url

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"


class SnapshotKind(StrEnum):
    CLEAN = "clean"
    BUGGY = "buggy"
    REFERENCE = "reference"


class RegressionExpectation(StrEnum):
    PASSES = "passes"
    FAILS = "fails"


class TestRunOutcome(StrictModel):
    """One independently executed suite outcome with content-addressed evidence."""

    suite: Literal["target", "regression"]
    attempt: int = Field(ge=1)
    exit_code: int
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    errors: int = Field(ge=0)
    skipped: int = Field(default=0, ge=0)
    duration_ms: int = Field(ge=0)
    report_sha256: str = Field(pattern=_DIGEST_PATTERN)
    infrastructure_error: bool = False

    @property
    def collected(self) -> int:
        return self.passed + self.failed + self.errors + self.skipped

    @property
    def is_pass(self) -> bool:
        return (
            not self.infrastructure_error
            and self.exit_code == 0
            and self.passed > 0
            and self.failed == 0
            and self.errors == 0
        )

    @property
    def is_test_failure(self) -> bool:
        return (
            not self.infrastructure_error
            and self.collected > 0
            and (self.failed > 0 or self.errors > 0)
            and self.exit_code != 0
        )


class SnapshotValidation(StrictModel):
    kind: SnapshotKind
    snapshot_sha256: str = Field(pattern=_DIGEST_PATTERN)
    outcomes: tuple[TestRunOutcome, ...]

    @field_validator("outcomes")
    @classmethod
    def validate_attempts(cls, outcomes: tuple[TestRunOutcome, ...]) -> tuple[TestRunOutcome, ...]:
        keys = tuple((outcome.suite, outcome.attempt) for outcome in outcomes)
        if len(keys) != len(set(keys)):
            raise ValueError("suite attempt numbers must be unique within a snapshot")
        for suite in ("target", "regression"):
            attempts = sorted(outcome.attempt for outcome in outcomes if outcome.suite == suite)
            if attempts and attempts != list(range(1, len(attempts) + 1)):
                raise ValueError("suite attempts must be contiguous and one-based")
        return tuple(sorted(outcomes, key=lambda outcome: (outcome.suite, outcome.attempt)))


class LicenseReview(StrictModel):
    repository_url: str
    spdx_identifier: str = Field(min_length=1, max_length=128)
    license_file_sha256: str = Field(pattern=_DIGEST_PATTERN)
    use_approved: bool
    redistribution_approved: bool
    reviewed_by: str = Field(min_length=1, max_length=256)

    @field_validator("repository_url")
    @classmethod
    def normalize_repository_url(cls, url: str) -> str:
        return normalize_source_url(url)


class LeakScan(StrictModel):
    issue_text_clean: bool
    agent_image_clean: bool
    agent_image_digest: str = Field(pattern=_DIGEST_PATTERN)
    findings: tuple[str, ...] = ()

    @field_validator("findings")
    @classmethod
    def normalize_findings(cls, findings: tuple[str, ...]) -> tuple[str, ...]:
        if any(not finding.strip() for finding in findings):
            raise ValueError("leak scan findings must be non-empty strings")
        return tuple(sorted(set(finding.strip() for finding in findings)))

    @model_validator(mode="after")
    def findings_match_status(self) -> LeakScan:
        clean = self.issue_text_clean and self.agent_image_clean
        if clean == bool(self.findings):
            raise ValueError("leak findings must be present exactly when a scan is not clean")
        return self


class AdmissionEvidence(StrictModel):
    task: TaskSpec
    clean: SnapshotValidation
    buggy: SnapshotValidation
    reference: SnapshotValidation
    buggy_regression_expectation: RegressionExpectation
    required_repetitions: int = Field(default=3, ge=3, le=20)
    license_review: LicenseReview
    leak_scan: LeakScan

    @model_validator(mode="after")
    def kinds_are_fixed(self) -> AdmissionEvidence:
        actual = (self.clean.kind, self.buggy.kind, self.reference.kind)
        expected = (SnapshotKind.CLEAN, SnapshotKind.BUGGY, SnapshotKind.REFERENCE)
        if actual != expected:
            raise ValueError("clean, buggy, and reference evidence must use matching state kinds")
        return self


class AdmissionFailure(StrEnum):
    SNAPSHOT_HASH_COLLISION = "snapshot_hash_collision"
    INSUFFICIENT_REPETITIONS = "insufficient_repetitions"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    CLEAN_REGRESSION_FAILED = "clean_regression_failed"
    BUGGY_TARGET_NOT_REPRODUCIBLE = "buggy_target_not_reproducible"
    BUGGY_REGRESSION_MISMATCH = "buggy_regression_mismatch"
    REFERENCE_TARGET_FAILED = "reference_target_failed"
    REFERENCE_REGRESSION_FAILED = "reference_regression_failed"
    SOURCE_PROVENANCE_MISMATCH = "source_provenance_mismatch"
    LICENSE_MISMATCH = "license_mismatch"
    LICENSE_NOT_APPROVED = "license_not_approved"
    REDISTRIBUTION_NOT_APPROVED = "redistribution_not_approved"
    LEAK_DETECTED = "leak_detected"
    AGENT_IMAGE_MISMATCH = "agent_image_mismatch"


class TaskAdmissionResult(StrictModel):
    task_id: str
    evidence_sha256: str = Field(pattern=_DIGEST_PATTERN)
    admitted: bool
    failures: tuple[AdmissionFailure, ...]

    @model_validator(mode="after")
    def status_matches_failures(self) -> TaskAdmissionResult:
        if self.admitted == bool(self.failures):
            raise ValueError("admitted must be true exactly when there are no failures")
        return self

    def digest(self) -> str:
        return canonical_sha256(self)


def _suite_runs(snapshot: SnapshotValidation, suite: str) -> tuple[TestRunOutcome, ...]:
    return tuple(outcome for outcome in snapshot.outcomes if outcome.suite == suite)


def validate_task_admission(evidence: AdmissionEvidence) -> TaskAdmissionResult:
    """Evaluate all admission gates without retrying or mutating evidence."""

    failures: set[AdmissionFailure] = set()
    repetitions = evidence.required_repetitions

    if evidence.clean.snapshot_sha256 == evidence.buggy.snapshot_sha256:
        failures.add(AdmissionFailure.SNAPSHOT_HASH_COLLISION)
    if evidence.reference.snapshot_sha256 == evidence.buggy.snapshot_sha256:
        failures.add(AdmissionFailure.SNAPSHOT_HASH_COLLISION)

    required_runs = (
        (evidence.clean, "regression"),
        (evidence.buggy, "target"),
        (evidence.buggy, "regression"),
        (evidence.reference, "target"),
        (evidence.reference, "regression"),
    )
    for snapshot, suite in required_runs:
        runs = _suite_runs(snapshot, suite)
        if len(runs) < repetitions:
            failures.add(AdmissionFailure.INSUFFICIENT_REPETITIONS)
        if any(run.infrastructure_error for run in runs):
            failures.add(AdmissionFailure.INFRASTRUCTURE_ERROR)

    clean_regression = _suite_runs(evidence.clean, "regression")
    if not clean_regression or not all(run.is_pass for run in clean_regression):
        failures.add(AdmissionFailure.CLEAN_REGRESSION_FAILED)

    buggy_target = _suite_runs(evidence.buggy, "target")
    if not buggy_target or not all(run.is_test_failure for run in buggy_target):
        failures.add(AdmissionFailure.BUGGY_TARGET_NOT_REPRODUCIBLE)

    buggy_regression = _suite_runs(evidence.buggy, "regression")
    expected_regression = (
        all(run.is_pass for run in buggy_regression)
        if evidence.buggy_regression_expectation is RegressionExpectation.PASSES
        else all(run.is_test_failure for run in buggy_regression)
    )
    if not buggy_regression or not expected_regression:
        failures.add(AdmissionFailure.BUGGY_REGRESSION_MISMATCH)

    reference_target = _suite_runs(evidence.reference, "target")
    if not reference_target or not all(run.is_pass for run in reference_target):
        failures.add(AdmissionFailure.REFERENCE_TARGET_FAILED)
    reference_regression = _suite_runs(evidence.reference, "regression")
    if not reference_regression or not all(run.is_pass for run in reference_regression):
        failures.add(AdmissionFailure.REFERENCE_REGRESSION_FAILED)

    provenance = evidence.task.provenance
    try:
        source_matches = (
            normalize_source_url(provenance.source_repository)
            == evidence.license_review.repository_url
        )
    except ValueError:
        source_matches = False
    if not source_matches:
        failures.add(AdmissionFailure.SOURCE_PROVENANCE_MISMATCH)
    if provenance.source_license.casefold() != evidence.license_review.spdx_identifier.casefold():
        failures.add(AdmissionFailure.LICENSE_MISMATCH)
    if not evidence.license_review.use_approved:
        failures.add(AdmissionFailure.LICENSE_NOT_APPROVED)
    if not evidence.license_review.redistribution_approved:
        failures.add(AdmissionFailure.REDISTRIBUTION_NOT_APPROVED)
    if not evidence.leak_scan.issue_text_clean or not evidence.leak_scan.agent_image_clean:
        failures.add(AdmissionFailure.LEAK_DETECTED)
    if evidence.leak_scan.agent_image_digest != evidence.task.agent_image_digest:
        failures.add(AdmissionFailure.AGENT_IMAGE_MISMATCH)

    ordered = tuple(sorted(failures, key=lambda failure: failure.value))
    return TaskAdmissionResult(
        task_id=evidence.task.task_id,
        evidence_sha256=canonical_sha256(evidence),
        admitted=not ordered,
        failures=ordered,
    )
