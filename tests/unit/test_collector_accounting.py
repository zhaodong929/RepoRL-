from __future__ import annotations

from pathlib import Path

import pytest

from reporl.agent import RunnerConfig
from reporl.rollouts.collector import (
    _executed_tool_calls,
    _is_infrastructure_trajectory,
    _policy_token_count,
)
from reporl.rollouts.config import RolloutCollectionConfig, RolloutPolicyConfig
from reporl.schemas import (
    DatasetSplit,
    Finish,
    SearchCode,
    TerminationReason,
    TokenUsage,
    ToolCall,
    ToolResult,
    Trajectory,
    TrajectoryEvent,
)


def _collection_config(**updates: object) -> RolloutCollectionConfig:
    values: dict[str, object] = {
        "run_id": "runner-config-test",
        "method": "test",
        "tasks_file": Path("tasks.jsonl"),
        "task_artifacts_root": Path("sealed"),
        "artifacts_root": Path("rollouts"),
        "expected_split": DatasetSplit.TRAIN,
        "expected_dataset_manifest_sha256": f"sha256:{'1' * 64}",
        "expected_split_seal_sha256": f"sha256:{'2' * 64}",
        "expected_split_assignment_sha256": f"sha256:{'3' * 64}",
        "expected_split_membership_sha256": f"sha256:{'4' * 64}",
        "expected_repository_records_sha256": f"sha256:{'5' * 64}",
        "expected_tasks_file_sha256": f"sha256:{'6' * 64}",
        "group_size": 1,
    }
    values.update(updates)
    return RolloutCollectionConfig.model_validate(values)


def test_collection_config_persists_runner_limits() -> None:
    config = _collection_config(
        runner=RunnerConfig(
            max_consecutive_invalid_actions=2,
            max_policy_output_chars=8_000,
            max_conversation_bytes=3_000,
            context_token_reserve=768,
        )
    )

    assert config.runner.max_consecutive_invalid_actions == 2
    assert config.model_dump(mode="json")["runner"]["max_conversation_bytes"] == 3_000


def test_collection_config_rejects_context_far_beyond_model_window() -> None:
    with pytest.raises(ValueError, match="context bytes"):
        _collection_config(
            runner=RunnerConfig(max_conversation_bytes=4_000, context_token_reserve=768),
            policy=RolloutPolicyConfig(max_input_tokens=4_096),
        )


def test_collector_accounting_matches_runner_budget_and_execution() -> None:
    search = ToolCall(call_id="search", action=SearchCode(query="needle"))
    finish = ToolCall(call_id="finish", action=Finish())
    trajectory = Trajectory(
        trajectory_id="trajectory-1",
        task_id="task-1",
        policy_id="policy",
        policy_revision=f"sha256:{'a' * 64}",
        config_digest=f"sha256:{'b' * 64}",
        seed=1,
        events=(
            TrajectoryEvent(
                step=0,
                tool_call=search,
                tool_result=ToolResult(
                    call_id="search",
                    ok=True,
                    output="match",
                    duration_ms=1,
                ),
                token_usage=TokenUsage(input_tokens=100, output_tokens=2),
            ),
            TrajectoryEvent(
                step=1,
                tool_call=finish,
                tool_result=ToolResult(
                    call_id="finish",
                    ok=True,
                    output="done",
                    duration_ms=0,
                    executed=False,
                ),
                token_usage=TokenUsage(input_tokens=10, output_tokens=1),
            ),
        ),
        termination_reason=TerminationReason.POLICY_ERROR,
        patch="",
    )

    assert _is_infrastructure_trajectory(trajectory)
    assert _policy_token_count(trajectory) == 113
    assert _executed_tool_calls(trajectory) == 1
