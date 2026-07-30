from __future__ import annotations

import pytest

from reporl.evaluation.bootstrap import hierarchical_paired_bootstrap, leave_one_lineage_out
from reporl.evaluation.metrics import (
    EvaluationRecord,
    paired_records,
    summarize_method,
    unbiased_pass_at_k,
)


def record(
    task: str,
    lineage: str,
    method: str,
    success: bool,
    *,
    calls: int = 3,
) -> EvaluationRecord:
    return EvaluationRecord(
        task_id=task,
        lineage_group=lineage,
        method=method,
        success=success,
        target_pass_fraction=float(success),
        regression_pass_fraction=1,
        valid_patch=success,
        tool_calls=calls,
    )


def records() -> tuple[EvaluationRecord, ...]:
    return (
        record("a", "repo-1", "base", False, calls=5),
        record("a", "repo-1", "rl", True, calls=3),
        record("b", "repo-1", "base", False, calls=4),
        record("b", "repo-1", "rl", False, calls=3),
        record("c", "repo-2", "base", True, calls=4),
        record("c", "repo-2", "rl", True, calls=2),
    )


def test_method_summary_reports_micro_and_macro() -> None:
    summary = summarize_method(records(), "rl")
    assert summary.micro_success == pytest.approx(2 / 3)
    assert summary.repository_macro_success == pytest.approx(0.75)


def test_hierarchical_bootstrap_is_paired_and_reproducible() -> None:
    pairs = paired_records(records(), baseline="base", candidate="rl")
    first = hierarchical_paired_bootstrap(
        pairs,
        baseline_metric=lambda item: float(item.success),
        candidate_metric=lambda item: float(item.success),
        resamples=1_000,
        seed=11,
    )
    second = hierarchical_paired_bootstrap(
        pairs,
        baseline_metric=lambda item: float(item.success),
        candidate_metric=lambda item: float(item.success),
        resamples=1_000,
        seed=11,
    )
    assert first == second
    assert first.estimate == pytest.approx(1 / 3)
    assert first.repositories == 2


def test_macro_bootstrap_and_leave_one_lineage_out_are_reported() -> None:
    pairs = paired_records(records(), baseline="base", candidate="rl")
    macro = hierarchical_paired_bootstrap(
        pairs,
        baseline_metric=lambda item: float(item.success),
        candidate_metric=lambda item: float(item.success),
        resamples=1_000,
        seed=11,
        repository_weighting="macro",
    )
    sensitivity = leave_one_lineage_out(
        pairs,
        baseline_metric=lambda item: float(item.success),
        candidate_metric=lambda item: float(item.success),
    )

    assert macro.estimate == pytest.approx(0.25)
    assert sensitivity == {"repo-1": 0.0, "repo-2": 0.5}


def test_unbiased_pass_at_k() -> None:
    assert unbiased_pass_at_k(samples=5, successes=1, k=1) == pytest.approx(0.2)
    assert unbiased_pass_at_k(samples=5, successes=5, k=3) == 1.0
