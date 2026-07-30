"""Deterministic verifier pipeline over a pristine, isolated workspace."""

from __future__ import annotations

from reporl.sandbox.base import PatchArtifact, SandboxInfrastructureError
from reporl.tools.output import format_process_output, truncate_output
from reporl.tools.patch import PatchInspection, PatchPolicy
from reporl.verifier.base import VerifierSandbox, VerifierSandboxFactory
from reporl.verifier.junit import JUnitParseError, parse_junit_xml
from reporl.verifier.models import (
    FailureKind,
    RepositoryEntryKind,
    SuiteExecution,
    SuiteVerification,
    VerificationResult,
    VerifierRunSpec,
    VerifierStatus,
    VerifierSuiteSpec,
)


class Verifier:
    """Apply and test a patch without reusing any agent-controlled workspace."""

    def __init__(self, factory: VerifierSandboxFactory, *, output_chars: int = 8_000) -> None:
        self._factory = factory
        self._output_chars = output_chars

    def verify(
        self,
        manifest: VerifierRunSpec,
        patch: PatchArtifact,
    ) -> VerificationResult:
        policy = PatchPolicy(
            allowed_paths=manifest.allowed_paths,
            forbidden_globs=manifest.forbidden_globs,
            max_patch_bytes=manifest.max_patch_bytes,
        )
        inspection = policy.inspect(patch.content)
        digest = patch.sha256
        assert digest is not None
        if not inspection.accepted:
            return self._failure(
                manifest,
                digest,
                inspection,
                FailureKind.PATCH_POLICY,
                "patch failed static safety policy",
            )

        sandbox: VerifierSandbox | None = None
        result: VerificationResult | None = None
        try:
            sandbox = self._factory.create(manifest)
            entries = sandbox.inspect_entries(inspection.paths)
            unsafe = tuple(
                entry
                for entry in entries
                if entry.kind
                in {
                    RepositoryEntryKind.SYMLINK,
                    RepositoryEntryKind.SUBMODULE,
                    RepositoryEntryKind.OTHER,
                }
            )
            if unsafe:
                detail = ", ".join(f"{entry.path} ({entry.kind.value})" for entry in unsafe)
                result = self._failure(
                    manifest,
                    digest,
                    inspection,
                    FailureKind.UNSAFE_REPOSITORY_ENTRY,
                    f"patch targets unsafe repository entries: {detail}",
                )
            else:
                applied = sandbox.apply_patch(patch)
                if applied.exit_code != 0 or applied.timed_out:
                    detail, _ = truncate_output(
                        format_process_output(applied.stdout, applied.stderr),
                        self._output_chars,
                    )
                    result = self._failure(
                        manifest,
                        digest,
                        inspection,
                        FailureKind.PATCH_APPLY,
                        f"patch could not be applied: {detail}",
                    )
                else:
                    applied_inspection = policy.inspect_applied_paths(
                        inspection,
                        sandbox.changed_paths(),
                    )
                    if not applied_inspection.accepted:
                        result = self._failure(
                            manifest,
                            digest,
                            applied_inspection,
                            FailureKind.PATCH_POLICY,
                            "Git's applied paths failed patch safety policy",
                        )
                    else:
                        result = self._run_suites(
                            manifest,
                            digest,
                            applied_inspection,
                            sandbox,
                        )
        except SandboxInfrastructureError as error:
            result = self._infrastructure_error(manifest, digest, inspection, str(error))
        except Exception as error:
            result = self._infrastructure_error(
                manifest,
                digest,
                inspection,
                f"unexpected verifier failure: {type(error).__name__}",
            )
        finally:
            if sandbox is not None:
                try:
                    sandbox.close()
                except Exception as error:
                    result = self._infrastructure_error(
                        manifest,
                        digest,
                        inspection,
                        f"failed to dispose verifier sandbox: {type(error).__name__}",
                    )
        assert result is not None
        return result

    def _run_suites(
        self,
        manifest: VerifierRunSpec,
        digest: str,
        inspection: PatchInspection,
        sandbox: VerifierSandbox,
    ) -> VerificationResult:
        suite_results: list[SuiteVerification] = []
        for suite in manifest.suites:
            execution = sandbox.run_suite(suite)
            verification, configuration_error = self._evaluate_suite(suite, execution)
            suite_results.append(verification)
            if configuration_error:
                return VerificationResult(
                    task_id=manifest.task_id,
                    patch_sha256=digest,
                    status=VerifierStatus.INFRASTRUCTURE_ERROR,
                    failure_kind=FailureKind.VERIFIER_CONFIGURATION,
                    detail=configuration_error,
                    patch_inspection=inspection,
                    suites=tuple(suite_results),
                )

        if all(suite.passed for suite in suite_results):
            return VerificationResult(
                task_id=manifest.task_id,
                patch_sha256=digest,
                status=VerifierStatus.PASSED,
                patch_inspection=inspection,
                suites=tuple(suite_results),
            )

        first_failure = next(suite for suite in suite_results if not suite.passed)
        if first_failure.timed_out:
            kind = FailureKind.TEST_TIMEOUT
        elif first_failure.suite == "regression":
            kind = FailureKind.REGRESSION_TESTS
        else:
            kind = FailureKind.TARGET_TESTS
        return VerificationResult(
            task_id=manifest.task_id,
            patch_sha256=digest,
            status=VerifierStatus.AGENT_FAILURE,
            failure_kind=kind,
            detail=f"suite {first_failure.suite!r} did not pass",
            patch_inspection=inspection,
            suites=tuple(suite_results),
        )

    def _evaluate_suite(
        self,
        suite: VerifierSuiteSpec,
        execution: SuiteExecution,
    ) -> tuple[SuiteVerification, str | None]:
        output, _ = truncate_output(
            format_process_output(execution.stdout, execution.stderr),
            self._output_chars,
        )
        if execution.junit_xml is None:
            detail = "suite did not produce JUnit XML"
            verification = SuiteVerification(
                suite=suite.name,
                passed=False,
                exit_code=execution.exit_code,
                duration_ms=execution.duration_ms,
                timed_out=execution.timed_out,
                output_excerpt=output,
                detail=detail,
            )
            return verification, detail if execution.exit_code == 0 else None
        try:
            report = parse_junit_xml(
                execution.junit_xml,
                expected_test_ids=suite.expected_test_ids,
            )
        except JUnitParseError as error:
            detail = str(error)
            verification = SuiteVerification(
                suite=suite.name,
                passed=False,
                exit_code=execution.exit_code,
                duration_ms=execution.duration_ms,
                timed_out=execution.timed_out,
                output_excerpt=output,
                detail=detail,
            )
            return verification, detail if execution.exit_code == 0 else None

        passed = execution.exit_code == 0 and not execution.timed_out and report.all_passed
        return (
            SuiteVerification(
                suite=suite.name,
                passed=passed,
                exit_code=execution.exit_code,
                duration_ms=execution.duration_ms,
                timed_out=execution.timed_out,
                report=report,
                output_excerpt=output,
                detail="" if passed else "JUnit outcomes or canonical test IDs did not pass",
            ),
            None,
        )

    @staticmethod
    def _failure(
        manifest: VerifierRunSpec,
        digest: str,
        inspection: PatchInspection,
        kind: FailureKind,
        detail: str,
    ) -> VerificationResult:
        return VerificationResult(
            task_id=manifest.task_id,
            patch_sha256=digest,
            status=VerifierStatus.AGENT_FAILURE,
            failure_kind=kind,
            detail=detail,
            patch_inspection=inspection,
        )

    @staticmethod
    def _infrastructure_error(
        manifest: VerifierRunSpec,
        digest: str,
        inspection: PatchInspection,
        detail: str,
    ) -> VerificationResult:
        return VerificationResult(
            task_id=manifest.task_id,
            patch_sha256=digest,
            status=VerifierStatus.INFRASTRUCTURE_ERROR,
            failure_kind=FailureKind.SANDBOX_INFRASTRUCTURE,
            detail=detail,
            patch_inspection=inspection,
        )
