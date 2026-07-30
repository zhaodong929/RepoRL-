"""Trusted verifier manifests and structured, auditable outcomes."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from reporl.sandbox.base import CommandSpec
from reporl.schemas import StrictModel
from reporl.tools.patch import PatchInspection


class VerifierStatus(StrEnum):
    PASSED = "pass"
    AGENT_FAILURE = "agent_failure"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class FailureKind(StrEnum):
    PATCH_POLICY = "patch_policy"
    UNSAFE_REPOSITORY_ENTRY = "unsafe_repository_entry"
    PATCH_APPLY = "patch_apply"
    TARGET_TESTS = "target_tests"
    REGRESSION_TESTS = "regression_tests"
    TEST_TIMEOUT = "test_timeout"
    VERIFIER_CONFIGURATION = "verifier_configuration"
    SANDBOX_INFRASTRUCTURE = "sandbox_infrastructure"


class TestCaseStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class JUnitTestCase(StrictModel):
    test_id: str = Field(min_length=1, max_length=2_000)
    classname: str = ""
    name: str = Field(min_length=1)
    status: TestCaseStatus
    duration_seconds: float = Field(default=0.0, ge=0)
    message: str = Field(default="", max_length=4_000)


class JUnitReport(StrictModel):
    cases: tuple[JUnitTestCase, ...]
    expected_test_ids: tuple[str, ...] = ()
    missing_test_ids: tuple[str, ...] = ()
    unexpected_test_ids: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(case.status == TestCaseStatus.PASSED for case in self.cases)

    @property
    def failed(self) -> int:
        return sum(case.status == TestCaseStatus.FAILED for case in self.cases)

    @property
    def errors(self) -> int:
        return sum(case.status == TestCaseStatus.ERROR for case in self.cases)

    @property
    def skipped(self) -> int:
        return sum(case.status == TestCaseStatus.SKIPPED for case in self.cases)

    @property
    def all_passed(self) -> bool:
        return (
            bool(self.cases)
            and self.passed == self.total
            and not self.missing_test_ids
            and not self.unexpected_test_ids
        )


class VerifierSuiteSpec(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    command: CommandSpec
    junit_path: str = Field(pattern=r"^/[^\x00]*\.xml$")
    expected_test_ids: tuple[str, ...] = ()

    @field_validator("expected_test_ids")
    @classmethod
    def unique_expected_ids(cls, ids: tuple[str, ...]) -> tuple[str, ...]:
        if len(ids) != len(set(ids)):
            raise ValueError("expected JUnit test IDs must be unique")
        return ids

    @model_validator(mode="after")
    def command_writes_expected_junit_path(self) -> VerifierSuiteSpec:
        if not self.junit_path.startswith("/tmp/reporl-junit/"):
            raise ValueError("JUnit output must be under /tmp/reporl-junit")
        if not any(self.junit_path in argument for argument in self.command.argv):
            raise ValueError("suite command must name its declared JUnit output path")
        return self


class VerifierRunSpec(StrictModel):
    """Verifier-only data that must never be serialized into an agent prompt."""

    task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
    image: str = Field(min_length=1)
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    repository_snapshot: Path
    hidden_tests_path: Path | None = None
    suites: tuple[VerifierSuiteSpec, ...]
    allowed_paths: tuple[str, ...] = ()
    forbidden_globs: tuple[str, ...] = ()
    max_patch_bytes: int = Field(default=100_000, ge=1)

    @model_validator(mode="after")
    def validate_suites(self) -> VerifierRunSpec:
        names = tuple(suite.name for suite in self.suites)
        if not names or len(names) != len(set(names)):
            raise ValueError("verifier suite names must be non-empty and unique")
        if "target" not in names:
            raise ValueError("verifier manifest must include a target suite")
        if self.image != self.image_digest and not self.image.endswith(f"@{self.image_digest}"):
            raise ValueError("verifier image must be pinned to image_digest")
        return self


class RepositoryEntryKind(StrEnum):
    MISSING = "missing"
    REGULAR = "regular"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    SUBMODULE = "submodule"
    OTHER = "other"


class RepositoryEntry(StrictModel):
    path: str
    kind: RepositoryEntryKind


class SuiteExecution(StrictModel):
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = Field(ge=0)
    timed_out: bool = False
    junit_xml: bytes | None = None


class SuiteVerification(StrictModel):
    suite: str
    passed: bool
    exit_code: int
    duration_ms: int = Field(ge=0)
    timed_out: bool = False
    report: JUnitReport | None = None
    output_excerpt: str = Field(default="", max_length=20_000)
    detail: str = Field(default="", max_length=2_000)


class VerificationResult(StrictModel):
    task_id: str
    patch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: VerifierStatus
    failure_kind: FailureKind | None = None
    detail: str = Field(default="", max_length=4_000)
    patch_inspection: PatchInspection
    suites: tuple[SuiteVerification, ...] = ()

    @model_validator(mode="after")
    def status_is_consistent(self) -> VerificationResult:
        if self.status == VerifierStatus.PASSED:
            if self.failure_kind is not None or not self.suites:
                raise ValueError("a passing result requires suites and no failure kind")
            if not all(suite.passed for suite in self.suites):
                raise ValueError("a passing result cannot contain a failed suite")
        elif self.failure_kind is None:
            raise ValueError("a non-passing result requires a failure kind")
        return self
