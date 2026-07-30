"""Generate a machine-readable evaluation summary and paired confidence intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reporl.evaluation.bootstrap import hierarchical_paired_bootstrap, leave_one_lineage_out
from reporl.evaluation.metrics import EvaluationRecord, paired_records, summarize_method
from reporl.tasks.loader import load_jsonl


def load_records(path: Path) -> tuple[EvaluationRecord, ...]:
    return load_jsonl(path, EvaluationRecord)


def build_report(
    records: tuple[EvaluationRecord, ...],
    *,
    baseline: str,
    candidate: str,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    methods = sorted({record.method for record in records})
    pairs = paired_records(records, baseline=baseline, candidate=candidate)
    success_micro_interval = hierarchical_paired_bootstrap(
        pairs,
        baseline_metric=lambda record: float(record.success),
        candidate_metric=lambda record: float(record.success),
        resamples=resamples,
        seed=seed,
    )
    success_macro_interval = hierarchical_paired_bootstrap(
        pairs,
        baseline_metric=lambda record: float(record.success),
        candidate_metric=lambda record: float(record.success),
        resamples=resamples,
        seed=seed,
        repository_weighting="macro",
    )
    tool_interval = hierarchical_paired_bootstrap(
        pairs,
        baseline_metric=lambda record: float(record.tool_calls),
        candidate_metric=lambda record: float(record.tool_calls),
        resamples=resamples,
        seed=seed + 1,
    )
    token_interval = hierarchical_paired_bootstrap(
        pairs,
        baseline_metric=lambda record: float(record.input_tokens + record.output_tokens),
        candidate_metric=lambda record: float(record.input_tokens + record.output_tokens),
        resamples=resamples,
        seed=seed + 2,
    )
    return {
        "methods": {
            method: summarize_method(records, method).model_dump(mode="json") for method in methods
        },
        "comparison": {
            "baseline": baseline,
            "candidate": candidate,
            "success_micro_absolute_difference": success_micro_interval.model_dump(mode="json"),
            "success_repository_macro_absolute_difference": success_macro_interval.model_dump(
                mode="json"
            ),
            "success_leave_one_lineage_out": leave_one_lineage_out(
                pairs,
                baseline_metric=lambda record: float(record.success),
                candidate_metric=lambda record: float(record.success),
            ),
            "tool_call_difference": tool_interval.model_dump(mode="json"),
            "total_token_difference": token_interval.model_dump(mode="json"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    report = build_report(
        load_records(args.records),
        baseline=args.baseline,
        candidate=args.candidate,
        resamples=args.resamples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
