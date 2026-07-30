"""Trusted dispatch from validated policy actions to an isolated sandbox."""

from __future__ import annotations

from time import monotonic

from reporl.sandbox.base import AgentSandbox, PatchArtifact, ProcessResult
from reporl.schemas import (
    ApplyPatch,
    Finish,
    ReadFile,
    RunTests,
    SearchCode,
    TaskSpec,
    ToolCall,
    ToolResult,
)
from reporl.tools.output import format_process_output, truncate_output
from reporl.tools.patch import PatchPolicy
from reporl.tools.paths import normalize_repo_path


class ToolGateway:
    """Execute the closed action set and enforce task-specific policy."""

    def __init__(self, task: TaskSpec, sandbox: AgentSandbox) -> None:
        self._task = task
        self._sandbox = sandbox
        self._patch_policy = PatchPolicy(
            allowed_paths=task.allowed_paths,
            forbidden_globs=task.forbidden_globs,
            max_patch_bytes=task.budgets.max_patch_bytes,
        )

    def execute(
        self,
        call: ToolCall,
        *,
        timeout_seconds: float | None = None,
    ) -> ToolResult:
        started = monotonic()
        action = call.action

        if isinstance(action, Finish):
            return ToolResult(
                call_id=call.call_id,
                ok=True,
                output=action.summary,
                duration_ms=self._elapsed_ms(started),
            )

        if isinstance(action, SearchCode):
            result = self._sandbox.search_code(
                action.query,
                normalize_repo_path(action.path, allow_dot=True),
                action.max_results,
                timeout_seconds=timeout_seconds,
            )
        elif isinstance(action, ReadFile):
            result = self._sandbox.read_file(
                normalize_repo_path(action.path),
                action.start_line,
                action.end_line,
                timeout_seconds=timeout_seconds,
            )
        elif isinstance(action, ApplyPatch):
            inspection = self._patch_policy.inspect(action.unified_diff)
            if not inspection.accepted:
                details = "; ".join(
                    f"{violation.code.value}: {violation.message}"
                    for violation in inspection.violations
                )
                return self._bounded_result(
                    call.call_id,
                    ok=False,
                    output=f"patch rejected: {details}",
                    exit_code=None,
                    started=started,
                )
            result = self._sandbox.apply_patch(
                action.unified_diff,
                timeout_seconds=timeout_seconds,
            )
        elif isinstance(action, RunTests):
            if (
                action.suite not in self._task.available_test_suites
                or action.suite not in self._sandbox.available_suites
            ):
                return self._bounded_result(
                    call.call_id,
                    ok=False,
                    output=f"test suite {action.suite!r} is unavailable",
                    exit_code=None,
                    started=started,
                )
            result = self._sandbox.run_suite(action.suite, timeout_seconds=timeout_seconds)
        else:  # pragma: no cover - Pydantic's discriminated union makes this unreachable.
            raise TypeError(f"unsupported action type: {type(action).__name__}")

        return self._process_result(call.call_id, result, started)

    def export_patch(self, *, timeout_seconds: float | None = None) -> PatchArtifact:
        """Export the content-addressed patch for the isolated verifier."""

        return self._sandbox.diff(timeout_seconds=timeout_seconds)

    def _process_result(
        self,
        call_id: str,
        result: ProcessResult,
        started: float,
    ) -> ToolResult:
        output = format_process_output(result.stdout, result.stderr)
        return self._bounded_result(
            call_id,
            ok=result.exit_code == 0 and not result.timed_out,
            output=output,
            exit_code=result.exit_code,
            started=started,
        )

    def _bounded_result(
        self,
        call_id: str,
        *,
        ok: bool,
        output: str,
        exit_code: int | None,
        started: float,
    ) -> ToolResult:
        bounded, truncated = truncate_output(
            output,
            self._task.budgets.max_tool_output_chars,
        )
        return ToolResult(
            call_id=call_id,
            ok=ok,
            output=bounded,
            exit_code=exit_code,
            duration_ms=self._elapsed_ms(started),
            truncated=truncated,
        )

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, round((monotonic() - started) * 1_000))
