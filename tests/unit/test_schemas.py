from __future__ import annotations

import pytest
from pydantic import ValidationError

from reporl.schemas import (
    ApplyPatch,
    Finish,
    ReadFile,
    RunTests,
    SearchCode,
    TerminationReason,
    ToolCall,
    ToolResult,
    Trajectory,
    TrajectoryEvent,
    parse_action,
)


@pytest.mark.parametrize(
    "payload, expected_type",
    [
        ({"kind": "search_code", "query": "needle"}, SearchCode),
        ({"kind": "read_file", "path": "src/module.py"}, ReadFile),
        (
            {
                "kind": "apply_patch",
                "unified_diff": "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-a\n+b\n",
            },
            ApplyPatch,
        ),
        ({"kind": "run_tests", "suite": "target"}, RunTests),
    ],
)
def test_parse_action_discriminates(
    payload: dict[str, object], expected_type: type[object]
) -> None:
    assert isinstance(parse_action(payload), expected_type)


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "read_file", "path": "../secret"},
        {"kind": "read_file", "path": "/etc/passwd"},
        {"kind": "read_file", "path": "C:\\Windows\\win.ini"},
        {"kind": "read_file", "path": "."},
        {"kind": "read_file", "path": "src/module.py\x00.txt"},
        {"kind": "search_code", "query": "x", "path": "src/../../secret"},
    ],
)
def test_actions_reject_path_escape(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        parse_action(payload)


def test_run_tests_rejects_arbitrary_shell() -> None:
    with pytest.raises(ValidationError):
        parse_action({"kind": "run_tests", "suite": "pytest; cat /etc/passwd"})


def test_read_file_bounds_window() -> None:
    with pytest.raises(ValidationError):
        ReadFile(path="src/module.py", start_line=1, end_line=501)


def test_trajectory_requires_contiguous_steps() -> None:
    call = ToolCall(call_id="call-1", action=Finish())
    result = ToolResult(call_id="call-1", ok=True, output="", duration_ms=1)
    event = TrajectoryEvent(step=1, tool_call=call, tool_result=result)

    with pytest.raises(ValidationError, match="contiguous"):
        Trajectory(
            trajectory_id="trajectory-1",
            task_id="task-1",
            policy_id="example/policy",
            policy_revision="revision-1",
            config_digest=f"sha256:{'a' * 64}",
            seed=0,
            events=(event,),
            termination_reason=TerminationReason.FINISHED,
            patch="",
        )
