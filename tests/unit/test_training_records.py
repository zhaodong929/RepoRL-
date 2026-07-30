from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from reporl.rewards import RewardSignals, compute_terminal_reward
from reporl.schemas import (
    Finish,
    GenerationTrace,
    TerminationReason,
    ToolCall,
    ToolResult,
    Trajectory,
    TrajectoryEvent,
)
from reporl.tools.patch import PatchInspection
from reporl.training.math import grouped_standardized_advantages
from reporl.training.prepare_sft import is_verified_sft_success
from reporl.training.records import GRPOEpisode, GRPOGroup, trajectory_to_grpo_episode
from reporl.verifier.models import FailureKind, VerificationResult, VerifierStatus


def episode(identifier: str, reward: float, *, group_id: str = "group-1") -> GRPOEpisode:
    breakdown = compute_terminal_reward(
        RewardSignals(
            target_pass_fraction=reward,
            regression_pass_fraction=1,
            valid_patch=bool(reward),
        )
    )
    return GRPOEpisode(
        episode_id=identifier,
        group_id=group_id,
        task_id="task-1",
        policy_id="Qwen/Qwen2.5-Coder-3B-Instruct",
        policy_revision=f"sha256:{'b' * 64}",
        policy_adapter_sha256=f"sha256:{'a' * 64}",
        reward=breakdown.total,
        reward_breakdown=breakdown,
        traces=(
            GenerationTrace(
                prompt_input_ids=(1, 2),
                generated_token_ids=(3,),
                old_logprobs=(-0.5,),
            ),
        ),
    )


def test_group_detects_zero_variance() -> None:
    group = GRPOGroup(group_id="group-1", episodes=(episode("a", 1), episode("b", 1)))
    assert group.zero_variance is True


def test_group_rejects_mixed_tasks() -> None:
    second = episode("b", 0).model_copy(update={"task_id": "task-2"})
    with pytest.raises(ValidationError, match="one task"):
        GRPOGroup(group_id="group-1", episodes=(episode("a", 1), second))


def test_grouped_advantages_center_each_group() -> None:
    advantages, zero_variance = grouped_standardized_advantages(
        [0.0, 1.0, 4.0, 4.0],
        ["a", "a", "b", "b"],
    )
    assert advantages[0] == pytest.approx(-1.0)
    assert advantages[1] == pytest.approx(1.0)
    assert advantages[2:] == (0.0, 0.0)
    assert zero_variance == frozenset({"b"})


def test_grouped_advantages_reject_singletons() -> None:
    with pytest.raises(ValueError, match="fewer than two"):
        grouped_standardized_advantages([1.0], ["a"])


def test_grpo_conversion_rejects_any_policy_turn_without_trace() -> None:
    call = ToolCall(call_id="call-1", action=Finish())
    result = ToolResult(call_id="call-1", ok=True, output="done", duration_ms=0)
    trajectory = Trajectory(
        trajectory_id="trajectory-1",
        task_id="task-1",
        policy_id="Qwen/Qwen2.5-Coder-3B-Instruct",
        policy_revision=f"sha256:{'b' * 64}",
        policy_adapter_sha256=f"sha256:{'a' * 64}",
        config_digest=f"sha256:{'c' * 64}",
        seed=1,
        events=(TrajectoryEvent(step=0, tool_call=call, tool_result=result),),
        termination_reason=TerminationReason.FINISHED,
        patch="diff",
    )
    breakdown = compute_terminal_reward(
        RewardSignals(target_pass_fraction=1, regression_pass_fraction=1, valid_patch=True)
    )

    with pytest.raises(ValueError, match="missing its generation trace"):
        trajectory_to_grpo_episode(
            trajectory,
            group_id="group-1",
            reward_breakdown=breakdown,
        )


def test_sft_admission_skips_failed_empty_patch_before_hash_comparison() -> None:
    verification = VerificationResult(
        task_id="task-1",
        patch_sha256=hashlib.sha256(b"").hexdigest(),
        status=VerifierStatus.AGENT_FAILURE,
        failure_kind=FailureKind.PATCH_POLICY,
        patch_inspection=PatchInspection(paths=(), byte_size=0, hunk_count=0),
    )

    assert not is_verified_sft_success("task-1", "", None, verification)
