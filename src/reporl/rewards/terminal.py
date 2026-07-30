"""Bounded terminal reward with explicit, auditable components."""

from __future__ import annotations

from pydantic import Field, model_validator

from reporl.schemas import StrictModel


class RewardConfig(StrictModel):
    success_bonus: float = Field(default=1.0, gt=0)
    target_progress_weight: float = Field(default=0.05, ge=0)
    regression_weight: float = Field(default=0.03, ge=0)
    valid_patch_weight: float = Field(default=0.02, ge=0)
    cost_weight: float = Field(default=0.05, ge=0)
    invalid_action_weight: float = Field(default=0.02, ge=0)
    budget_exhausted_penalty: float = Field(default=0.02, ge=0)

    @model_validator(mode="after")
    def successful_trajectory_dominates(self) -> RewardConfig:
        maximum_failure = (
            self.target_progress_weight + self.regression_weight + self.valid_patch_weight
        )
        conservative_success_floor = self.success_bonus - (
            self.target_progress_weight
            + self.cost_weight
            + self.invalid_action_weight
            + self.budget_exhausted_penalty
        )
        if conservative_success_floor <= maximum_failure:
            raise ValueError("reward weights do not guarantee success dominance")
        return self


class RewardSignals(StrictModel):
    target_pass_fraction: float = Field(ge=0, le=1)
    progress_potential_delta: float = Field(default=0, ge=-1, le=1)
    regression_pass_fraction: float = Field(ge=0, le=1)
    valid_patch: bool
    policy_violation: bool = False
    invalid_action_fraction: float = Field(default=0, ge=0, le=1)
    step_fraction: float = Field(default=0, ge=0, le=1)
    token_fraction: float = Field(default=0, ge=0, le=1)
    budget_exhausted: bool = False
    infrastructure_error: bool = False


class RewardBreakdown(StrictModel):
    success: bool
    eligible_for_training: bool
    outcome: float
    target_progress: float
    regression: float
    valid_patch: float
    policy: float
    invalid_actions: float
    cost: float
    budget: float
    total: float


class TrajectoryReward(StrictModel):
    trajectory_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    breakdown: RewardBreakdown


def compute_terminal_reward(
    signals: RewardSignals,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Compose verifier signals without treating the scalar as an evaluation metric."""

    weights = config or RewardConfig()
    if signals.infrastructure_error:
        return RewardBreakdown(
            success=False,
            eligible_for_training=False,
            outcome=0.0,
            target_progress=0.0,
            regression=0.0,
            valid_patch=0.0,
            policy=0.0,
            invalid_actions=0.0,
            cost=0.0,
            budget=0.0,
            total=0.0,
        )

    success = (
        signals.target_pass_fraction == 1.0
        and signals.regression_pass_fraction == 1.0
        and signals.valid_patch
        and not signals.policy_violation
    )
    outcome = weights.success_bonus if success else 0.0
    target_progress = weights.target_progress_weight * signals.progress_potential_delta
    regression = weights.regression_weight * signals.regression_pass_fraction
    valid_patch = weights.valid_patch_weight if signals.valid_patch else 0.0
    policy = -weights.success_bonus if signals.policy_violation else 0.0
    invalid_actions = -weights.invalid_action_weight * signals.invalid_action_fraction
    normalized_cost = min(1.0, 0.5 * signals.step_fraction + 0.5 * signals.token_fraction)
    cost = -weights.cost_weight * normalized_cost
    budget = -weights.budget_exhausted_penalty if signals.budget_exhausted else 0.0
    total = sum(
        (
            outcome,
            target_progress,
            regression,
            valid_patch,
            policy,
            invalid_actions,
            cost,
            budget,
        )
    )

    return RewardBreakdown(
        success=success,
        eligible_for_training=True,
        outcome=outcome,
        target_progress=target_progress,
        regression=regression,
        valid_patch=valid_patch,
        policy=policy,
        invalid_actions=invalid_actions,
        cost=cost,
        budget=budget,
        total=total,
    )
