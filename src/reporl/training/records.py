"""Serializable, trainer-facing records derived from verified trajectories."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import Field, model_validator

from reporl.agent.models import ChatMessage
from reporl.agent.runner import replay_messages
from reporl.rewards import RewardBreakdown
from reporl.schemas import GenerationTrace, StrictModel, TaskSpec, Trajectory
from reporl.tasks.loader import load_jsonl


class SFTRecord(StrictModel):
    record_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)
    messages: tuple[ChatMessage, ...]


class GRPOEpisode(StrictModel):
    episode_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    policy_revision: str = Field(min_length=1)
    policy_adapter_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    reward: float
    reward_breakdown: RewardBreakdown
    traces: tuple[GenerationTrace, ...]

    @model_validator(mode="after")
    def traces_are_trainable(self) -> GRPOEpisode:
        if not self.traces:
            raise ValueError("a GRPO episode must contain at least one generation trace")
        if any(not trace.old_logprobs for trace in self.traces):
            raise ValueError("every GRPO trace requires behavior-policy logprobs")
        if any(trace.sampling_temperature <= 0 for trace in self.traces):
            raise ValueError("GRPO traces must come from a stochastic behavior policy")
        sampling_settings = {
            (trace.sampling_temperature, trace.sampling_top_p) for trace in self.traces
        }
        if len(sampling_settings) != 1:
            raise ValueError("all traces in an episode must use one sampling configuration")
        if not self.reward_breakdown.eligible_for_training:
            raise ValueError("ineligible rewards cannot enter GRPO")
        if self.reward != self.reward_breakdown.total:
            raise ValueError("scalar reward must equal the stored reward breakdown")
        return self

    @property
    def generated_tokens(self) -> int:
        return sum(len(trace.generated_token_ids) for trace in self.traces)


class GRPOGroup(StrictModel):
    group_id: str = Field(min_length=1)
    episodes: tuple[GRPOEpisode, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def episodes_match_group(self) -> GRPOGroup:
        episode_ids = tuple(episode.episode_id for episode in self.episodes)
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("episode IDs must be unique within a rollout group")
        if any(episode.group_id != self.group_id for episode in self.episodes):
            raise ValueError("episode group IDs must match the enclosing group")
        revisions = {episode.policy_revision for episode in self.episodes}
        if len(revisions) != 1:
            raise ValueError("a rollout group must come from one policy revision")
        policy_ids = {episode.policy_id for episode in self.episodes}
        if len(policy_ids) != 1:
            raise ValueError("a rollout group must come from one policy model")
        adapters = {episode.policy_adapter_sha256 for episode in self.episodes}
        if len(adapters) != 1:
            raise ValueError("a rollout group must come from one adapter artifact")
        sampling_settings = {
            (trace.sampling_temperature, trace.sampling_top_p)
            for episode in self.episodes
            for trace in episode.traces
        }
        if len(sampling_settings) != 1:
            raise ValueError("a rollout group must use one sampling configuration")
        task_ids = {episode.task_id for episode in self.episodes}
        if len(task_ids) != 1:
            raise ValueError("a rollout group must contain candidates for one task")
        return self

    @property
    def zero_variance(self) -> bool:
        return len({episode.reward for episode in self.episodes}) == 1


def trajectory_to_sft_record(
    task: TaskSpec,
    trajectory: Trajectory,
    *,
    verified_success: bool,
) -> SFTRecord:
    if task.task_id != trajectory.task_id:
        raise ValueError("task and trajectory IDs do not match")
    if not verified_success:
        raise ValueError("only independently verified successes may enter SFT")
    if any(event.parse_error is not None for event in trajectory.events):
        raise ValueError("SFT admission rejects trajectories with invalid actions")
    return SFTRecord(
        record_id=f"{task.task_id}:{trajectory.trajectory_id}",
        task_id=task.task_id,
        trajectory_id=trajectory.trajectory_id,
        messages=replay_messages(
            task,
            trajectory.events,
            max_conversation_bytes=trajectory.max_conversation_bytes,
        ),
    )


def trajectory_to_grpo_episode(
    trajectory: Trajectory,
    *,
    group_id: str,
    reward_breakdown: RewardBreakdown,
) -> GRPOEpisode:
    if not trajectory.events:
        raise ValueError("a GRPO trajectory must contain at least one policy turn")
    traces: list[GenerationTrace] = []
    for event in trajectory.events:
        if event.generation_trace is None:
            raise ValueError(f"policy turn {event.step} is missing its generation trace")
        traces.append(event.generation_trace)
    if trajectory.policy_adapter_sha256 is None:
        raise ValueError("a GRPO trajectory must identify its behavior-policy adapter")
    return GRPOEpisode(
        episode_id=trajectory.trajectory_id,
        group_id=group_id,
        task_id=trajectory.task_id,
        policy_id=trajectory.policy_id,
        policy_revision=trajectory.policy_revision,
        policy_adapter_sha256=trajectory.policy_adapter_sha256,
        reward=reward_breakdown.total,
        reward_breakdown=reward_breakdown,
        traces=tuple(traces),
    )


def write_jsonl(path: Path, records: Iterable[StrictModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")
        handle.flush()
    temporary.replace(path)


def read_sft_jsonl(path: Path) -> tuple[SFTRecord, ...]:
    return load_jsonl(path, SFTRecord)


def read_grpo_groups_jsonl(path: Path) -> tuple[GRPOGroup, ...]:
    return load_jsonl(path, GRPOGroup)
