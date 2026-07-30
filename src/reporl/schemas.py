"""Trusted data contracts shared by the runner, tools, and verifier."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator


class StrictModel(BaseModel):
    """Immutable model that rejects unknown fields at trust boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _repo_relative_path(value: str, *, allow_dot: bool = False) -> str:
    normalized = value.replace("\\", "/")
    if "\x00" in normalized:
        raise ValueError("path must not contain a null byte")
    if normalized == ".":
        if allow_dot:
            return normalized
        raise ValueError("path must identify a repository file")
    if not normalized or normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError("path must be relative to the repository root")
    if ".." in PurePosixPath(normalized).parts:
        raise ValueError("path traversal is not allowed")
    return normalized


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class TaskBudgets(StrictModel):
    max_steps: int = Field(default=20, ge=1, le=100)
    max_policy_tokens: int = Field(default=16_384, ge=256)
    max_wall_time_seconds: int = Field(default=1_800, ge=1)
    max_tool_output_chars: int = Field(default=20_000, ge=256)
    max_patch_bytes: int = Field(default=100_000, ge=1)


class TaskProvenance(StrictModel):
    source_repository: str = Field(min_length=1)
    source_license: str = Field(min_length=1)
    base_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    lineage_group: str = Field(min_length=1)
    generator: str = Field(min_length=1)
    generator_version: str = Field(min_length=1)


class TaskSpec(StrictModel):
    """Agent-visible task metadata; hidden tests and gold patches are deliberately absent."""

    task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
    issue: str = Field(min_length=1, max_length=50_000)
    split: DatasetSplit
    agent_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    provenance: TaskProvenance
    allowed_paths: tuple[str, ...] = ()
    forbidden_globs: tuple[str, ...] = (
        "test*.py",
        "**/test*.py",
        "*_test.py",
        "**/*_test.py",
        "tests/**",
        "**/tests/**",
        "conftest.py",
        "**/conftest.py",
        "pytest.ini",
        "tox.ini",
        "setup.cfg",
        "setup.py",
        "sitecustomize.py",
        "**/sitecustomize.py",
        "requirements*.txt",
        "pyproject.toml",
        "Pipfile",
        "Pipfile.lock",
        "poetry.lock",
        "uv.lock",
        ".github/**",
    )
    available_test_suites: tuple[Literal["target", "regression"], ...] = (
        "target",
        "regression",
    )
    budgets: TaskBudgets = Field(default_factory=TaskBudgets)

    @field_validator("allowed_paths")
    @classmethod
    def validate_allowed_paths(cls, paths: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_repo_relative_path(path, allow_dot=True) for path in paths)


class SearchCode(StrictModel):
    kind: Literal["search_code"] = "search_code"
    query: str = Field(min_length=1, max_length=1_000)
    path: str = "."
    max_results: int = Field(default=50, ge=1, le=200)

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        return _repo_relative_path(path, allow_dot=True)


class ReadFile(StrictModel):
    kind: Literal["read_file"] = "read_file"
    path: str
    start_line: int = Field(default=1, ge=1)
    end_line: int = Field(default=200, ge=1)

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        return _repo_relative_path(path)

    @model_validator(mode="after")
    def validate_window(self) -> ReadFile:
        if self.end_line < self.start_line:
            raise ValueError("end_line must not precede start_line")
        if self.end_line - self.start_line + 1 > 500:
            raise ValueError("a read_file window cannot exceed 500 lines")
        return self


class ApplyPatch(StrictModel):
    kind: Literal["apply_patch"] = "apply_patch"
    unified_diff: str = Field(min_length=1, max_length=100_000)

    @field_validator("unified_diff")
    @classmethod
    def validate_unified_diff(cls, patch: str) -> str:
        if "diff --git " not in patch and not patch.startswith("--- "):
            raise ValueError("patch must be a unified diff")
        return patch


class RunTests(StrictModel):
    kind: Literal["run_tests"] = "run_tests"
    suite: Literal["target", "regression"]


class Finish(StrictModel):
    kind: Literal["finish"] = "finish"
    summary: str = Field(default="", max_length=1_000)


Action: TypeAlias = Annotated[
    SearchCode | ReadFile | ApplyPatch | RunTests | Finish,
    Field(discriminator="kind"),
]
_ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)


def parse_action(payload: str | bytes | dict[str, Any]) -> Action:
    """Validate a policy action before it reaches the tool gateway."""

    if isinstance(payload, (str, bytes)):
        return _ACTION_ADAPTER.validate_json(payload)
    return _ACTION_ADAPTER.validate_python(payload)


class TokenUsage(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class GenerationTrace(StrictModel):
    """Token-level policy evidence required for on-policy updates."""

    prompt_input_ids: tuple[int, ...] = Field(min_length=1)
    generated_token_ids: tuple[int, ...] = Field(min_length=1)
    old_logprobs: tuple[float, ...] = ()
    sampling_temperature: float = Field(default=1.0, ge=0)
    sampling_top_p: float = Field(default=1.0, gt=0, le=1)

    @model_validator(mode="after")
    def validate_alignment(self) -> GenerationTrace:
        if any(token_id < 0 for token_id in (*self.prompt_input_ids, *self.generated_token_ids)):
            raise ValueError("token IDs must be non-negative")
        if self.old_logprobs and len(self.old_logprobs) != len(self.generated_token_ids):
            raise ValueError("old_logprobs must align with generated_token_ids")
        if any(not math.isfinite(value) for value in self.old_logprobs):
            raise ValueError("old_logprobs must be finite")
        return self


class ToolCall(StrictModel):
    call_id: str = Field(min_length=1, max_length=128)
    action: Action


class ToolResult(StrictModel):
    call_id: str = Field(min_length=1, max_length=128)
    ok: bool
    output: str
    exit_code: int | None = None
    duration_ms: int = Field(ge=0)
    executed: bool = True
    truncated: bool = False
    artifact_path: str | None = None


class TerminationReason(StrEnum):
    FINISHED = "finished"
    STEP_BUDGET = "step_budget"
    TOKEN_BUDGET = "token_budget"
    WALL_TIME_BUDGET = "wall_time_budget"
    CONTEXT_BUDGET = "context_budget"
    INVALID_ACTION = "invalid_action"
    POLICY_ERROR = "policy_error"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class TrajectoryEvent(StrictModel):
    step: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    raw_policy_output: str = ""
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    parse_error: str | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    generation_trace: GenerationTrace | None = None
    policy_latency_ms: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def event_is_consistent(self) -> TrajectoryEvent:
        if (self.tool_call is None) != (self.tool_result is None):
            raise ValueError("tool call and result must either both exist or both be absent")
        if self.tool_call is not None and self.tool_result is not None:
            if self.tool_call.call_id != self.tool_result.call_id:
                raise ValueError("tool call and result IDs must match")
        if self.parse_error is not None and self.tool_call is not None:
            raise ValueError("a parse-error event cannot contain a tool call")
        if self.parse_error is None and self.tool_call is None:
            raise ValueError("an event without a tool call must describe a parse error")
        if self.generation_trace is not None:
            if len(self.generation_trace.generated_token_ids) != self.token_usage.output_tokens:
                raise ValueError("generated token count must match output token usage")
        return self


class Trajectory(StrictModel):
    trajectory_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    task_id: str = Field(min_length=1, max_length=128)
    policy_id: str = Field(min_length=1, max_length=256)
    policy_revision: str = Field(min_length=1, max_length=256)
    policy_adapter_sha256: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    max_conversation_bytes: int = Field(default=3_000, ge=2_048, le=100_000)
    seed: int
    events: tuple[TrajectoryEvent, ...]
    termination_reason: TerminationReason
    patch: str
    patch_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_event_sequence(self) -> Trajectory:
        steps = tuple(event.step for event in self.events)
        if steps != tuple(range(len(self.events))):
            raise ValueError("trajectory event steps must be contiguous and zero-based")
        return self
