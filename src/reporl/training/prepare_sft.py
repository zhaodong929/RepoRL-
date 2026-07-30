"""Build an SFT JSONL file from independently verified successful trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

from reporl.rollouts.config import RolloutTaskSpec
from reporl.rollouts.store import TrajectoryStore
from reporl.schemas import DatasetSplit
from reporl.tasks.loader import load_jsonl
from reporl.training.records import SFTRecord, trajectory_to_sft_record, write_jsonl
from reporl.verifier.models import VerificationResult, VerifierStatus


def is_verified_sft_success(
    trajectory_task_id: str,
    trajectory_patch: str,
    trajectory_patch_sha256: str | None,
    verification: VerificationResult,
) -> bool:
    if verification.task_id != trajectory_task_id:
        raise ValueError("verification and trajectory task IDs do not match")
    if verification.status != VerifierStatus.PASSED:
        return False
    if not trajectory_patch or trajectory_patch_sha256 is None:
        raise ValueError("a passing verification must correspond to a non-empty patch")
    if verification.patch_sha256 != trajectory_patch_sha256:
        raise ValueError("verification and trajectory patch hashes do not match")
    return True


def prepare(
    *,
    tasks_file: Path,
    trajectories_dir: Path,
    verifications_dir: Path,
    output: Path,
    split: DatasetSplit,
) -> int:
    runtimes = load_jsonl(tasks_file, RolloutTaskSpec)
    tasks = {
        runtime.task.task_id: runtime.task for runtime in runtimes if runtime.task.split == split
    }
    store = TrajectoryStore(trajectories_dir)
    records: list[SFTRecord] = []
    for trajectory in store.iter_all():
        task = tasks.get(trajectory.task_id)
        if task is None:
            continue
        verification_path = verifications_dir / f"{trajectory.trajectory_id}.json"
        if not verification_path.exists():
            continue
        verification = VerificationResult.model_validate_json(
            verification_path.read_text(encoding="utf-8")
        )
        if not is_verified_sft_success(
            trajectory.task_id,
            trajectory.patch,
            trajectory.patch_sha256,
            verification,
        ):
            continue
        records.append(trajectory_to_sft_record(task, trajectory, verified_success=True))
    if not records:
        raise ValueError(f"no verified SFT records found for split {split.value}")
    write_jsonl(output, records)
    return len(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--verifications", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=[split.value for split in DatasetSplit],
        default=DatasetSplit.TRAIN.value,
    )
    args = parser.parse_args(argv)
    count = prepare(
        tasks_file=args.tasks,
        trajectories_dir=args.trajectories,
        verifications_dir=args.verifications,
        output=args.output,
        split=DatasetSplit(args.split),
    )
    print(f"wrote {count} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
