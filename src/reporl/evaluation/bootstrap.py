"""Two-level paired bootstrap over repository lineages and tasks."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import Field

from reporl.evaluation.metrics import EvaluationRecord
from reporl.schemas import StrictModel


class BootstrapInterval(StrictModel):
    estimate: float
    lower: float
    upper: float
    confidence: float = Field(gt=0, lt=1)
    resamples: int = Field(ge=1)
    repositories: int = Field(ge=1)
    paired_units: int = Field(ge=1)


Metric = Callable[[EvaluationRecord], float]


def hierarchical_paired_bootstrap(
    pairs: Sequence[tuple[EvaluationRecord, EvaluationRecord]],
    *,
    baseline_metric: Metric,
    candidate_metric: Metric,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    repository_weighting: Literal["micro", "macro"] = "micro",
) -> BootstrapInterval:
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if not pairs:
        raise ValueError("at least one paired record is required")
    grouped: dict[str, list[tuple[EvaluationRecord, EvaluationRecord]]] = defaultdict(list)
    for baseline, candidate in pairs:
        if baseline.lineage_group != candidate.lineage_group:
            raise ValueError("paired records must have the same lineage group")
        if baseline.unit_id != candidate.unit_id:
            raise ValueError("paired records must have the same task and replicate")
        grouped[baseline.lineage_group].append((baseline, candidate))
    repository_ids = sorted(grouped)
    observed = _grouped_difference(
        grouped,
        baseline_metric,
        candidate_metric,
        repository_weighting,
    )
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        sampled_repositories: list[list[tuple[EvaluationRecord, EvaluationRecord]]] = []
        for _ in repository_ids:
            repository = rng.choice(repository_ids)
            repository_pairs = grouped[repository]
            sampled_repositories.append([rng.choice(repository_pairs) for _ in repository_pairs])
        if repository_weighting == "micro":
            sampled_pairs = [pair for repository in sampled_repositories for pair in repository]
            draws.append(_paired_difference(sampled_pairs, baseline_metric, candidate_metric))
        else:
            draws.append(
                sum(
                    _paired_difference(repository, baseline_metric, candidate_metric)
                    for repository in sampled_repositories
                )
                / len(sampled_repositories)
            )
    draws.sort()
    alpha = (1 - confidence) / 2
    return BootstrapInterval(
        estimate=observed,
        lower=_percentile(draws, alpha),
        upper=_percentile(draws, 1 - alpha),
        confidence=confidence,
        resamples=resamples,
        repositories=len(repository_ids),
        paired_units=len(pairs),
    )


def leave_one_lineage_out(
    pairs: Sequence[tuple[EvaluationRecord, EvaluationRecord]],
    *,
    baseline_metric: Metric,
    candidate_metric: Metric,
) -> dict[str, float]:
    lineages = sorted({baseline.lineage_group for baseline, _ in pairs})
    if len(lineages) < 2:
        raise ValueError("leave-one-lineage-out requires at least two repository lineages")
    results: dict[str, float] = {}
    for omitted in lineages:
        retained = tuple(pair for pair in pairs if pair[0].lineage_group != omitted)
        results[omitted] = _paired_difference(
            retained,
            baseline_metric,
            candidate_metric,
        )
    return results


def _grouped_difference(
    grouped: dict[str, list[tuple[EvaluationRecord, EvaluationRecord]]],
    baseline_metric: Metric,
    candidate_metric: Metric,
    weighting: Literal["micro", "macro"],
) -> float:
    if weighting == "micro":
        return _paired_difference(
            tuple(pair for repository in grouped.values() for pair in repository),
            baseline_metric,
            candidate_metric,
        )
    return sum(
        _paired_difference(repository, baseline_metric, candidate_metric)
        for repository in grouped.values()
    ) / len(grouped)


def _paired_difference(
    pairs: Sequence[tuple[EvaluationRecord, EvaluationRecord]],
    baseline_metric: Metric,
    candidate_metric: Metric,
) -> float:
    differences = [
        candidate_metric(candidate) - baseline_metric(baseline) for baseline, candidate in pairs
    ]
    if any(not math.isfinite(value) for value in differences):
        raise ValueError("metric values must be finite")
    return sum(differences) / len(differences)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
