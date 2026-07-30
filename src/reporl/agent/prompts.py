"""Stable prompts for the structured repository repair loop."""

from __future__ import annotations

import json

from reporl.agent.models import ChatMessage
from reporl.schemas import TaskSpec

SYSTEM_PROMPT = """You are RepoRL, a repository repair agent.
Work only through the available structured actions. Return exactly one JSON object per turn,
without Markdown or explanatory prose. Inspect before editing, make the smallest justified patch,
run named tests, and finish only when the patch is ready for independent verification.

Actions:
{"kind":"search_code","query":"text","path":".","max_results":50}
{"kind":"read_file","path":"relative/path.py","start_line":1,"end_line":200}
{"kind":"apply_patch","unified_diff":"<unified diff>"}
{"kind":"run_tests","suite":"target"}
{"kind":"run_tests","suite":"regression"}
{"kind":"finish","summary":"short factual summary"}

Never attempt path traversal, shell execution, network access, test modification, verifier
inspection, or access outside the repository. Tool errors are observations: revise the next action
instead of repeating an invalid request.
"""


def initial_messages(task: TaskSpec) -> tuple[ChatMessage, ...]:
    task_payload = {
        "task_id": task.task_id,
        "issue": task.issue,
        "allowed_paths": list(task.allowed_paths),
        "forbidden_globs": list(task.forbidden_globs),
        "available_test_suites": list(task.available_test_suites),
        "budgets": task.budgets.model_dump(mode="json"),
    }
    return (
        ChatMessage(role="system", content=SYSTEM_PROMPT.strip()),
        ChatMessage(
            role="user",
            content="Repair this issue:\n" + json.dumps(task_payload, sort_keys=True),
        ),
    )


def tool_observation_message(call_id: str, payload: dict[str, object]) -> ChatMessage:
    return ChatMessage(
        role="tool",
        name="reporl_tool",
        tool_call_id=call_id,
        content=json.dumps(payload, sort_keys=True),
    )


def parser_error_message(error: str) -> ChatMessage:
    return ChatMessage(
        role="tool",
        name="action_parser",
        content=json.dumps(
            {
                "ok": False,
                "error": error,
                "instruction": "Return exactly one valid JSON action object.",
            },
            sort_keys=True,
        ),
    )
