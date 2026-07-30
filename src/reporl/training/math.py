"""Small framework-independent pieces of GRPO math."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence


def grouped_standardized_advantages(
    rewards: Sequence[float],
    group_ids: Sequence[str],
    *,
    epsilon: float = 1e-8,
) -> tuple[tuple[float, ...], frozenset[str]]:
    if len(rewards) != len(group_ids):
        raise ValueError("rewards and group_ids must have equal length")
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for index, (group_id, reward) in enumerate(zip(group_ids, rewards, strict=True)):
        if not math.isfinite(reward):
            raise ValueError("rewards must be finite")
        grouped[group_id].append((index, reward))

    advantages = [0.0] * len(rewards)
    zero_variance: set[str] = set()
    for group_id, values in grouped.items():
        if len(values) < 2:
            raise ValueError(f"group {group_id!r} has fewer than two episodes")
        mean = sum(value for _, value in values) / len(values)
        variance = sum((value - mean) ** 2 for _, value in values) / len(values)
        if variance <= epsilon:
            zero_variance.add(group_id)
            continue
        scale = math.sqrt(variance + epsilon)
        for index, value in values:
            advantages[index] = (value - mean) / scale
    return tuple(advantages), frozenset(zero_variance)
