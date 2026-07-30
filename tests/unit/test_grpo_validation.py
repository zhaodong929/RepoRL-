from __future__ import annotations

from pathlib import Path

import pytest

from reporl.rewards import RewardSignals, compute_terminal_reward
from reporl.schemas import GenerationTrace
from reporl.training.config import GRPOConfig
from reporl.training.grpo import _validate_groups
from reporl.training.records import GRPOEpisode, GRPOGroup

POLICY_REVISION = f"sha256:{'b' * 64}"


def group(revision: str = POLICY_REVISION) -> GRPOGroup:
    episodes = []
    for index in range(2):
        breakdown = compute_terminal_reward(
            RewardSignals(
                target_pass_fraction=float(index),
                regression_pass_fraction=1,
                valid_patch=bool(index),
            )
        )
        episodes.append(
            GRPOEpisode(
                episode_id=f"episode-{index}",
                group_id="group-1",
                task_id="task-1",
                policy_id="Qwen/Qwen2.5-Coder-3B-Instruct",
                policy_revision=revision,
                policy_adapter_sha256=f"sha256:{'a' * 64}",
                reward=breakdown.total,
                reward_breakdown=breakdown,
                traces=(
                    GenerationTrace(
                        prompt_input_ids=(1,),
                        generated_token_ids=(2,),
                        old_logprobs=(-0.2,),
                    ),
                ),
            )
        )
    return GRPOGroup(group_id="group-1", episodes=tuple(episodes))


def config() -> GRPOConfig:
    return GRPOConfig(
        initial_adapter=Path("adapter"),
        rollout_groups_file=Path("groups.jsonl"),
        output_dir=Path("output"),
        expected_policy_revision=POLICY_REVISION,
    )


def test_validate_groups_accepts_matching_revision() -> None:
    _validate_groups((group(),), config())


def test_validate_groups_rejects_stale_policy() -> None:
    with pytest.raises(ValueError, match="revision"):
        _validate_groups((group(f"sha256:{'c' * 64}"),), config())


def test_validate_groups_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _validate_groups((group(), group()), config())
