from __future__ import annotations

import pytest

from reporl.rewards import RewardConfig, RewardSignals, compute_terminal_reward


def test_clean_success_outscores_best_failure() -> None:
    expensive_success = compute_terminal_reward(
        RewardSignals(
            target_pass_fraction=1,
            progress_potential_delta=1,
            regression_pass_fraction=1,
            valid_patch=True,
            step_fraction=1,
            token_fraction=1,
            budget_exhausted=True,
        )
    )
    best_failure = compute_terminal_reward(
        RewardSignals(
            target_pass_fraction=1,
            progress_potential_delta=1,
            regression_pass_fraction=1,
            valid_patch=False,
        )
    )

    assert expensive_success.success is True
    assert best_failure.success is False
    assert expensive_success.total > best_failure.total


def test_policy_violation_prevents_success() -> None:
    result = compute_terminal_reward(
        RewardSignals(
            target_pass_fraction=1,
            regression_pass_fraction=1,
            valid_patch=True,
            policy_violation=True,
        )
    )

    assert result.success is False
    assert result.policy < 0


def test_infrastructure_errors_are_excluded_from_training() -> None:
    result = compute_terminal_reward(
        RewardSignals(
            target_pass_fraction=0,
            regression_pass_fraction=0,
            valid_patch=False,
            infrastructure_error=True,
        )
    )

    assert result.eligible_for_training is False
    assert result.total == 0


def test_cost_is_bounded() -> None:
    result = compute_terminal_reward(
        RewardSignals(
            target_pass_fraction=0,
            regression_pass_fraction=0,
            valid_patch=False,
            step_fraction=1,
            token_fraction=1,
        )
    )

    assert result.cost == pytest.approx(-RewardConfig().cost_weight)


def test_invalid_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="success dominance"):
        RewardConfig(success_bonus=0.1, cost_weight=0.1)
