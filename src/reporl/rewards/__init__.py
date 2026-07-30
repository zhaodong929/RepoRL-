"""Executable reward composition."""

from reporl.rewards.terminal import (
    RewardBreakdown,
    RewardConfig,
    RewardSignals,
    TrajectoryReward,
    compute_terminal_reward,
)

__all__ = [
    "RewardBreakdown",
    "RewardConfig",
    "RewardSignals",
    "TrajectoryReward",
    "compute_terminal_reward",
]
