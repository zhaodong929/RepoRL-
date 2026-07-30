"""Collect grouped interactive trajectories and independently verify every patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from reporl.agent import (
    AgentRunner,
    DockerTaskEnvironment,
    OpenAICompatiblePolicy,
    RemoteTracePolicy,
    TransformersPolicy,
)
from reporl.agent.policy import PolicyBackend
from reporl.agent.remote_policy import fetch_policy_server_info
from reporl.evaluation.metrics import EvaluationRecord
from reporl.rewards import (
    RewardBreakdown,
    RewardSignals,
    TrajectoryReward,
    compute_terminal_reward,
)
from reporl.rollouts.config import RolloutCollectionConfig, RolloutPolicyConfig, RolloutTaskSpec
from reporl.rollouts.store import ArtifactStore, TrajectoryStore
from reporl.sandbox.base import PatchArtifact
from reporl.schemas import TerminationReason, Trajectory
from reporl.tasks.canonical import artifact_sha256
from reporl.tasks.loader import load_jsonl
from reporl.training.records import (
    GRPOEpisode,
    GRPOGroup,
    trajectory_to_grpo_episode,
    write_jsonl,
)
from reporl.verifier.docker import DockerVerifierFactory
from reporl.verifier.models import (
    FailureKind,
    SuiteVerification,
    VerificationResult,
    VerifierStatus,
)
from reporl.verifier.pipeline import Verifier

_INFRASTRUCTURE_TERMINATIONS = frozenset(
    {TerminationReason.INFRASTRUCTURE_ERROR, TerminationReason.POLICY_ERROR}
)


def load_collection_config(path: Path) -> RolloutCollectionConfig:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    if "rollout" not in payload:
        raise ValueError("configuration file is missing [rollout]")
    return RolloutCollectionConfig.model_validate(payload["rollout"])


def create_policy(config: RolloutPolicyConfig) -> PolicyBackend:
    if config.backend == "transformers":
        return TransformersPolicy.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            adapter_path=str(config.adapter_path) if config.adapter_path is not None else None,
            load_in_4bit=config.load_in_4bit,
            max_input_tokens=config.max_input_tokens,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
    if config.backend == "remote_trace":
        assert config.base_url is not None
        secret = config.secret()
        info = fetch_policy_server_info(
            config.base_url,
            bearer_token=secret,
            timeout_seconds=config.timeout_seconds,
        )
        if info.policy_identity is None:
            raise ValueError("trace policy server did not publish a complete policy identity")
        config.validate_server_identity(info.policy_identity)
        return RemoteTracePolicy(
            base_url=config.base_url,
            policy_id=info.policy_id,
            policy_revision=info.policy_revision,
            adapter_sha256=info.policy_identity.adapter_sha256,
            policy_identity=info.policy_identity,
            bearer_token=secret,
            timeout_seconds=config.timeout_seconds,
        )
    assert config.backend == "openai"
    assert config.base_url is not None
    return OpenAICompatiblePolicy(
        base_url=config.base_url,
        model=config.model_id,
        api_key=config.secret(),
        revision=config.model_revision,
        timeout_seconds=config.timeout_seconds,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens,
    )


def collect(config: RolloutCollectionConfig) -> Path:
    tasks_file_sha256, _ = artifact_sha256(config.tasks_file)
    if tasks_file_sha256 != config.expected_tasks_file_sha256:
        raise ValueError("tasks_file differs from expected_tasks_file_sha256")
    portable_tasks = load_jsonl(config.tasks_file, RolloutTaskSpec)
    if config.maximum_tasks is not None:
        portable_tasks = portable_tasks[: config.maximum_tasks]
    config.validate_task_bindings(portable_tasks)
    tasks = tuple(
        runtime.rebase_artifact_root(config.task_artifacts_root) for runtime in portable_tasks
    )
    for runtime in tasks:
        runtime.validate_materialized_paths()

    run_dir = config.artifacts_root / config.run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    policy = create_policy(config.policy)
    started_at = datetime.now(UTC).isoformat()
    partial_manifest_path = run_dir / "run-manifest.partial.json"
    manifest_base = {
        "kind": "rollout-collection",
        "run_id": config.run_id,
        "method": config.method,
        "started_at": started_at,
        "git": _git_state(),
        "policy_id": policy.policy_id,
        "policy_revision": policy.policy_revision,
        "policy_identity": (
            identity.model_dump(mode="json")
            if (identity := getattr(policy, "policy_identity", None)) is not None
            else None
        ),
        "config": config.model_dump(mode="json"),
        "task_count": len(tasks),
    }
    partial_manifest_path.write_text(
        json.dumps({**manifest_base, "status": "running"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    runner = AgentRunner(config.runner)
    verifier = Verifier(DockerVerifierFactory())
    trajectories = TrajectoryStore(run_dir / "trajectories")
    evidence = ArtifactStore(run_dir / "evidence")
    verification_dir = run_dir / "verifications"
    verification_dir.mkdir(parents=True, exist_ok=True)
    reward_dir = run_dir / "rewards"
    reward_dir.mkdir(parents=True, exist_ok=True)
    evaluation_records: list[EvaluationRecord] = []
    groups: list[GRPOGroup] = []
    dropped_groups: list[dict[str, object]] = []
    candidate_exclusions: list[dict[str, str]] = []

    for task_index, runtime in enumerate(tasks):
        group_id = _group_id(config.run_id, runtime.task.task_id, policy.policy_revision)
        episodes: list[GRPOEpisode] = []
        for candidate_index in range(config.group_size):
            rollout_seed = config.seed + task_index * config.group_size + candidate_index
            environment = DockerTaskEnvironment(
                runtime.agent.repository_snapshot,
                runtime.agent.docker_config(),
            )
            started = time.monotonic()
            trajectory = runner.run(runtime.task, policy, environment, seed=rollout_seed)
            wall_time = time.monotonic() - started
            trajectories.append(trajectory)
            verification: VerificationResult | None = None
            if trajectory.termination_reason not in _INFRASTRUCTURE_TERMINATIONS:
                verification = verifier.verify(
                    runtime.verifier,
                    PatchArtifact(content=trajectory.patch),
                )
                evidence.put_text(
                    verification.model_dump_json(),
                    suffix=".verification.json",
                )
                with (verification_dir / f"{trajectory.trajectory_id}.json").open(
                    "x", encoding="utf-8", newline="\n"
                ) as handle:
                    handle.write(verification.model_dump_json())
                    handle.write("\n")
            reward = _reward(trajectory, verification, runtime, config)
            with (reward_dir / f"{trajectory.trajectory_id}.json").open(
                "x", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(
                    TrajectoryReward(
                        trajectory_id=trajectory.trajectory_id,
                        task_id=trajectory.task_id,
                        breakdown=reward,
                    ).model_dump_json()
                )
                handle.write("\n")
            evaluation_records.append(
                _evaluation_record(
                    trajectory,
                    verification,
                    runtime,
                    method=config.method,
                    replicate=str(rollout_seed),
                    wall_time_seconds=wall_time,
                )
            )
            if reward.eligible_for_training:
                try:
                    episodes.append(
                        trajectory_to_grpo_episode(
                            trajectory,
                            group_id=group_id,
                            reward_breakdown=reward,
                        )
                    )
                except ValueError as error:
                    candidate_exclusions.append(
                        {
                            "trajectory_id": trajectory.trajectory_id,
                            "task_id": trajectory.task_id,
                            "reason": str(error),
                        }
                    )
        if len(episodes) == config.group_size and len(episodes) >= 2:
            groups.append(GRPOGroup(group_id=group_id, episodes=tuple(episodes)))
        elif config.group_size >= 2:
            dropped_groups.append(
                {
                    "group_id": group_id,
                    "task_id": runtime.task.task_id,
                    "expected_episodes": config.group_size,
                    "trainable_episodes": len(episodes),
                }
            )

    write_jsonl(run_dir / "evaluation.jsonl", evaluation_records)
    if groups:
        write_jsonl(run_dir / "grpo-groups.jsonl", groups)
    manifest = {
        **manifest_base,
        "status": "completed",
        "finished_at": datetime.now(UTC).isoformat(),
        "trajectory_count": len(evaluation_records),
        "trainable_group_count": len(groups),
        "zero_variance_group_count": sum(group.zero_variance for group in groups),
        "dropped_groups": dropped_groups,
        "candidate_exclusions": candidate_exclusions,
    }
    (run_dir / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    partial_manifest_path.unlink()
    return run_dir


def _suite_fraction(suite: SuiteVerification | None) -> float:
    if suite is None or suite.report is None or suite.report.total == 0:
        return 0.0
    fraction = suite.report.passed / suite.report.total
    if not suite.passed and fraction == 1:
        return 0.0
    return fraction


def _suite(result: VerificationResult | None, name: str) -> SuiteVerification | None:
    if result is None:
        return None
    return next((suite for suite in result.suites if suite.suite == name), None)


def _reward(
    trajectory: Trajectory,
    verification: VerificationResult | None,
    runtime: RolloutTaskSpec,
    config: RolloutCollectionConfig,
) -> RewardBreakdown:
    target_fraction = _suite_fraction(_suite(verification, "target"))
    regression_fraction = _suite_fraction(_suite(verification, "regression"))
    infrastructure_error = (
        _is_infrastructure_trajectory(trajectory)
        or verification is None
        or verification.status == VerifierStatus.INFRASTRUCTURE_ERROR
    )
    policy_violation = verification is not None and verification.failure_kind in {
        FailureKind.PATCH_POLICY,
        FailureKind.UNSAFE_REPOSITORY_ENTRY,
    }
    valid_patch = bool(trajectory.patch.strip()) and (
        verification is not None and verification.patch_inspection.accepted
    )
    invalid_actions = sum(event.parse_error is not None for event in trajectory.events)
    total_policy_tokens = _policy_token_count(trajectory)
    signals = RewardSignals(
        target_pass_fraction=target_fraction,
        progress_potential_delta=max(
            -1.0,
            min(1.0, target_fraction - runtime.baseline_target_pass_fraction),
        ),
        regression_pass_fraction=regression_fraction,
        valid_patch=valid_patch,
        policy_violation=policy_violation,
        invalid_action_fraction=invalid_actions / max(1, len(trajectory.events)),
        step_fraction=len(trajectory.events) / runtime.task.budgets.max_steps,
        token_fraction=min(
            1.0,
            total_policy_tokens / runtime.task.budgets.max_policy_tokens,
        ),
        budget_exhausted=trajectory.termination_reason
        in {
            TerminationReason.STEP_BUDGET,
            TerminationReason.TOKEN_BUDGET,
            TerminationReason.WALL_TIME_BUDGET,
            TerminationReason.CONTEXT_BUDGET,
        },
        infrastructure_error=infrastructure_error,
    )
    return compute_terminal_reward(signals, config.reward)


def _evaluation_record(
    trajectory: Trajectory,
    verification: VerificationResult | None,
    runtime: RolloutTaskSpec,
    *,
    method: str,
    replicate: str,
    wall_time_seconds: float,
) -> EvaluationRecord:
    target_fraction = _suite_fraction(_suite(verification, "target"))
    regression_fraction = _suite_fraction(_suite(verification, "regression"))
    infrastructure_error = (
        _is_infrastructure_trajectory(trajectory)
        or verification is None
        or verification.status == VerifierStatus.INFRASTRUCTURE_ERROR
    )
    policy_violation = verification is not None and verification.failure_kind in {
        FailureKind.PATCH_POLICY,
        FailureKind.UNSAFE_REPOSITORY_ENTRY,
    }
    valid_patch = bool(trajectory.patch.strip()) and (
        verification is not None and verification.patch_inspection.accepted
    )
    success = verification is not None and verification.status == VerifierStatus.PASSED
    return EvaluationRecord(
        task_id=runtime.task.task_id,
        lineage_group=runtime.task.provenance.lineage_group,
        method=method,
        replicate=replicate,
        success=success,
        target_pass_fraction=target_fraction,
        regression_pass_fraction=regression_fraction,
        valid_patch=valid_patch,
        policy_violation=policy_violation,
        infrastructure_error=infrastructure_error,
        tool_calls=_executed_tool_calls(trajectory),
        invalid_actions=sum(event.parse_error is not None for event in trajectory.events),
        input_tokens=sum(event.token_usage.input_tokens for event in trajectory.events),
        output_tokens=sum(event.token_usage.output_tokens for event in trajectory.events),
        wall_time_seconds=wall_time_seconds,
        test_cpu_seconds=(
            sum(suite.duration_ms for suite in verification.suites) / 1_000
            if verification is not None
            else 0
        ),
    )


def _group_id(run_id: str, task_id: str, policy_revision: str) -> str:
    payload = f"{run_id}\0{task_id}\0{policy_revision}".encode()
    return hashlib.sha256(payload).hexdigest()[:32]


def _is_infrastructure_trajectory(trajectory: Trajectory) -> bool:
    return trajectory.termination_reason in _INFRASTRUCTURE_TERMINATIONS


def _policy_token_count(trajectory: Trajectory) -> int:
    return sum(event.token_usage.total_tokens for event in trajectory.events)


def _executed_tool_calls(trajectory: Trajectory) -> int:
    return sum(
        event.tool_result is not None and event.tool_result.executed for event in trajectory.events
    )


def _git_state() -> dict[str, object]:
    def command(*args: str) -> str:
        result = subprocess.run(
            ("git", *args),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": command("rev-parse", "HEAD"),
            "dirty": bool(command("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    print(collect(load_collection_config(args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
