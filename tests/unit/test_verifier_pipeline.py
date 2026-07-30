from __future__ import annotations

from pathlib import Path

from reporl.sandbox.base import CommandSpec, PatchArtifact, ProcessResult
from reporl.tools.patch import PatchViolationCode
from reporl.verifier.models import (
    FailureKind,
    RepositoryEntry,
    RepositoryEntryKind,
    SuiteExecution,
    VerifierRunSpec,
    VerifierStatus,
    VerifierSuiteSpec,
)
from reporl.verifier.pipeline import Verifier

VALID_PATCH = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -1 +1 @@
-a = 1
+a = 2
"""
PASSING_XML = b"<testsuite><testcase classname='tests.test_a' name='test_fix'/></testsuite>"


def _manifest(tmp_path: Path, *, expected: tuple[str, ...] = ()) -> VerifierRunSpec:
    return VerifierRunSpec(
        task_id="task-001",
        image=f"verifier@sha256:{'a' * 64}",
        image_digest=f"sha256:{'a' * 64}",
        repository_snapshot=tmp_path,
        suites=(
            VerifierSuiteSpec(
                name="target",
                command=CommandSpec(
                    argv=("pytest", "-q", "--junitxml=/tmp/reporl-junit/target.xml")
                ),
                junit_path="/tmp/reporl-junit/target.xml",
                expected_test_ids=expected,
            ),
        ),
        allowed_paths=("src",),
        forbidden_globs=("tests/**",),
    )


class FakeVerifierSandbox:
    def __init__(self) -> None:
        self.entries = (RepositoryEntry(path="src/a.py", kind=RepositoryEntryKind.REGULAR),)
        self.apply_result = ProcessResult(argv=("git", "apply"), exit_code=0, duration_ms=1)
        self.execution = SuiteExecution(
            exit_code=0,
            duration_ms=2,
            junit_xml=PASSING_XML,
        )
        self.closed = False
        self.applied_paths = ("src/a.py",)

    def inspect_entries(self, paths: tuple[str, ...]) -> tuple[RepositoryEntry, ...]:
        assert paths == ("src/a.py",)
        return self.entries

    def apply_patch(self, patch: PatchArtifact) -> ProcessResult:
        assert patch.content == VALID_PATCH
        return self.apply_result

    def changed_paths(self) -> tuple[str, ...]:
        return self.applied_paths

    def run_suite(self, suite: VerifierSuiteSpec) -> SuiteExecution:
        assert suite.name == "target"
        return self.execution

    def close(self) -> None:
        self.closed = True


class FakeFactory:
    def __init__(self, sandbox: FakeVerifierSandbox) -> None:
        self.sandbox = sandbox
        self.calls = 0

    def create(self, manifest: VerifierRunSpec) -> FakeVerifierSandbox:
        self.calls += 1
        return self.sandbox


def test_verifier_passes_only_with_patch_and_canonical_junit(tmp_path: Path) -> None:
    sandbox = FakeVerifierSandbox()
    expected = ("tests.test_a::test_fix",)

    result = Verifier(FakeFactory(sandbox)).verify(
        _manifest(tmp_path, expected=expected),
        PatchArtifact(content=VALID_PATCH),
    )

    assert result.status == VerifierStatus.PASSED
    assert result.failure_kind is None
    assert result.suites[0].report is not None
    assert result.suites[0].report.all_passed
    assert sandbox.closed


def test_verifier_rejects_paths_that_disagree_with_git(tmp_path: Path) -> None:
    sandbox = FakeVerifierSandbox()
    sandbox.applied_paths = ("tests/test_a.py",)

    result = Verifier(FakeFactory(sandbox)).verify(
        _manifest(tmp_path),
        PatchArtifact(content=VALID_PATCH),
    )

    assert result.status == VerifierStatus.AGENT_FAILURE
    assert result.failure_kind == FailureKind.PATCH_POLICY
    assert PatchViolationCode.MALFORMED in {
        violation.code for violation in result.patch_inspection.violations
    }


def test_verifier_rejects_patch_before_allocating_sandbox(tmp_path: Path) -> None:
    sandbox = FakeVerifierSandbox()
    factory = FakeFactory(sandbox)
    forbidden = VALID_PATCH.replace("src/a.py", "tests/test_a.py")

    result = Verifier(factory).verify(
        _manifest(tmp_path),
        PatchArtifact(content=forbidden),
    )

    assert result.status == VerifierStatus.AGENT_FAILURE
    assert result.failure_kind == FailureKind.PATCH_POLICY
    assert PatchViolationCode.FORBIDDEN_PATH in {
        violation.code for violation in result.patch_inspection.violations
    }
    assert factory.calls == 0


def test_symlink_target_is_agent_failure(tmp_path: Path) -> None:
    sandbox = FakeVerifierSandbox()
    sandbox.entries = (RepositoryEntry(path="src/a.py", kind=RepositoryEntryKind.SYMLINK),)

    result = Verifier(FakeFactory(sandbox)).verify(
        _manifest(tmp_path), PatchArtifact(content=VALID_PATCH)
    )

    assert result.status == VerifierStatus.AGENT_FAILURE
    assert result.failure_kind == FailureKind.UNSAFE_REPOSITORY_ENTRY


def test_failed_test_is_valid_negative_training_signal(tmp_path: Path) -> None:
    sandbox = FakeVerifierSandbox()
    sandbox.execution = SuiteExecution(
        exit_code=1,
        duration_ms=2,
        junit_xml=(
            b"<testsuite><testcase classname='tests.test_a' name='test_fix'>"
            b"<failure message='still broken'/></testcase></testsuite>"
        ),
    )

    result = Verifier(FakeFactory(sandbox)).verify(
        _manifest(tmp_path), PatchArtifact(content=VALID_PATCH)
    )

    assert result.status == VerifierStatus.AGENT_FAILURE
    assert result.failure_kind == FailureKind.TARGET_TESTS


def test_missing_junit_after_zero_exit_is_infrastructure_error(tmp_path: Path) -> None:
    sandbox = FakeVerifierSandbox()
    sandbox.execution = SuiteExecution(exit_code=0, duration_ms=2, junit_xml=None)

    result = Verifier(FakeFactory(sandbox)).verify(
        _manifest(tmp_path), PatchArtifact(content=VALID_PATCH)
    )

    assert result.status == VerifierStatus.INFRASTRUCTURE_ERROR
    assert result.failure_kind == FailureKind.VERIFIER_CONFIGURATION


def test_canonical_test_set_mismatch_fails_even_with_zero_exit(tmp_path: Path) -> None:
    result = Verifier(FakeFactory(FakeVerifierSandbox())).verify(
        _manifest(tmp_path, expected=("tests.test_a::different",)),
        PatchArtifact(content=VALID_PATCH),
    )

    assert result.status == VerifierStatus.AGENT_FAILURE
    assert result.failure_kind == FailureKind.TARGET_TESTS


def test_factory_exception_is_excluded_as_infrastructure(tmp_path: Path) -> None:
    class BrokenFactory:
        def create(self, manifest: VerifierRunSpec) -> FakeVerifierSandbox:
            raise RuntimeError("daemon down")

    result = Verifier(BrokenFactory()).verify(
        _manifest(tmp_path), PatchArtifact(content=VALID_PATCH)
    )

    assert result.status == VerifierStatus.INFRASTRUCTURE_ERROR
    assert result.failure_kind == FailureKind.SANDBOX_INFRASTRUCTURE
