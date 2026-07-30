"""Structured task metrics and aggregate summaries."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

from pydantic import Field, model_validator

from reporl.schemas import StrictModel


class EvaluationRecord(StrictModel):
    task_id: str = Field(min_length=1)
    lineage_group: str = Field(min_length=1)
    method: str = Field(min_length=1)
    replicate: str = Field(default="0", min_length=1)
    success: bool
    target_pass_fraction: float = Field(ge=0, le=1)
    regression_pass_fraction: float = Field(ge=0, le=1)
    valid_patch: bool
    policy_violation: bool = False
    infrastructure_error: bool = False
    tool_calls: int = Field(default=0, ge=0)
    invalid_actions: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    wall_time_seconds: float = Field(default=0, ge=0)
    test_cpu_seconds: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def success_is_consistent(self) -> EvaluationRecord:
        expected = (
            self.target_pass_fraction == 1
            and self.regression_pass_fraction == 1
            and self.valid_patch
            and not self.policy_violation
            and not self.infrastructure_error
        )
        if self.success != expected:
            raise ValueError("success disagrees with executable verifier fields")
        return self

    @property
    def unit_id(self) -> str:
        return f"{self.task_id}:{self.replicate}"


class MethodSummary(StrictModel):
    method: str
    attempted: int
    infrastructure_errors: int
    micro_success: float
    repository_macro_success: float
    regression_break_rate: float
    policy_violation_rate: float
    invalid_actions_per_task: float
    mean_tool_calls_all: float
    mean_total_tokens_all: float
    mean_output_tokens_all: float
    mean_tool_calls_successes: float | None
    mean_total_tokens_successes: float | None
    mean_output_tokens_successes: float | None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_method(records: Iterable[EvaluationRecord], method: str) -> MethodSummary:
    selected = [record for record in records if record.method == method]
    if not selected:
        raise ValueError(f"no records found for method {method!r}")
    usable = [record for record in selected if not record.infrastructure_error]
    if not usable:
        raise ValueError(f"all records for method {method!r} are infrastructure errors")
    by_repository: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in usable:
        by_repository[record.lineage_group].append(record)
    successful = [record for record in usable if record.success]
    return MethodSummary(
        method=method,
        attempted=len(selected),
        infrastructure_errors=len(selected) - len(usable),
        micro_success=_mean([float(record.success) for record in usable]),
        repository_macro_success=_mean(
            [
                _mean([float(record.success) for record in repository_records])
                for repository_records in by_repository.values()
            ]
        ),
        regression_break_rate=_mean(
            [float(record.regression_pass_fraction < 1) for record in usable]
        ),
        policy_violation_rate=_mean([float(record.policy_violation) for record in usable]),
        invalid_actions_per_task=_mean([float(record.invalid_actions) for record in usable]),
        mean_tool_calls_all=_mean([float(record.tool_calls) for record in usable]),
        mean_total_tokens_all=_mean(
            [float(record.input_tokens + record.output_tokens) for record in usable]
        ),
        mean_output_tokens_all=_mean([float(record.output_tokens) for record in usable]),
        mean_tool_calls_successes=(
            _mean([float(record.tool_calls) for record in successful]) if successful else None
        ),
        mean_total_tokens_successes=(
            _mean([float(record.input_tokens + record.output_tokens) for record in successful])
            if successful
            else None
        ),
        mean_output_tokens_successes=(
            _mean([float(record.output_tokens) for record in successful]) if successful else None
        ),
    )


def unbiased_pass_at_k(*, samples: int, successes: int, k: int) -> float:
    if samples < 1 or k < 1 or k > samples:
        raise ValueError("require samples >= k >= 1")
    if successes < 0 or successes > samples:
        raise ValueError("successes must be between zero and samples")
    if samples - successes < k:
        return 1.0
    log_failure = 0.0
    for offset in range(k):
        log_failure += math.log(samples - successes - offset) - math.log(samples - offset)
    return 1.0 - math.exp(log_failure)


def paired_records(
    records: Iterable[EvaluationRecord],
    *,
    baseline: str,
    candidate: str,
) -> tuple[tuple[EvaluationRecord, EvaluationRecord], ...]:
    table: dict[tuple[str, str, str], dict[str, EvaluationRecord]] = defaultdict(dict)
    for record in records:
        if record.method not in {baseline, candidate} or record.infrastructure_error:
            continue
        key = (record.lineage_group, record.task_id, record.replicate)
        if record.method in table[key]:
            raise ValueError(f"duplicate evaluation record for {key} and {record.method}")
        table[key][record.method] = record
    incomplete = [key for key, methods in table.items() if set(methods) != {baseline, candidate}]
    if incomplete:
        raise ValueError(f"unpaired evaluation records, first missing key: {incomplete[0]}")
    return tuple((methods[baseline], methods[candidate]) for _, methods in sorted(table.items()))
