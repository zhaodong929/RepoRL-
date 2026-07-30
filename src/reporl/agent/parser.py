"""Strict parsing for a single JSON action emitted by a policy."""

from __future__ import annotations

import json

from pydantic import ValidationError

from reporl.schemas import Action, parse_action


class ActionParseError(ValueError):
    """The policy response is not exactly one valid RepoRL action."""


def _strip_single_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        raise ActionParseError("unterminated JSON code fence")
    opening = lines[0].strip().lower()
    if opening not in {"```", "```json"}:
        raise ActionParseError("only a single JSON code fence is accepted")
    return "\n".join(lines[1:-1]).strip()


def parse_policy_action(raw_output: str) -> Action:
    """Parse one JSON object while rejecting prose and multiple actions."""

    candidate = _strip_single_fence(raw_output)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ActionParseError(f"invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ActionParseError("the policy response must be one JSON object")
    try:
        return parse_action(payload)
    except ValidationError as error:
        first = error.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first["loc"])
        raise ActionParseError(f"invalid action at {location}: {first['msg']}") from error
