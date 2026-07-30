from __future__ import annotations

from collections.abc import Sequence

from reporl.agent import AgentRunner, RunnerConfig, ScriptedPolicy
from reporl.agent.models import ChatMessage, PolicyStep
from reporl.agent.parser import ActionParseError, parse_policy_action
from reporl.agent.policy import PolicyContextLengthError
from reporl.agent.runner import compact_messages, replay_messages
from reporl.sandbox.base import SandboxInfrastructureError
from reporl.schemas import (
    Action,
    DatasetSplit,
    SearchCode,
    TaskBudgets,
    TaskProvenance,
    TaskSpec,
    TerminationReason,
    TokenUsage,
    ToolResult,
)


class MemorySandbox:
    def __init__(self) -> None:
        self.actions: list[Action] = []
        self.timeouts: list[float | None] = []
        self.closed = False

    def reset(self, task: TaskSpec) -> None:
        self.task = task

    def execute(
        self,
        action: Action,
        *,
        call_id: str,
        timeout_seconds: float | None = None,
    ) -> ToolResult:
        self.timeouts.append(timeout_seconds)
        self.actions.append(action)
        return ToolResult(call_id=call_id, ok=True, output="ok", duration_ms=1)

    def diff(self, *, timeout_seconds: float | None = None) -> str:
        del timeout_seconds
        return "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"

    def close(self) -> None:
        self.closed = True


class MeteredPolicy:
    def __init__(self, steps: Sequence[PolicyStep]) -> None:
        self._steps = tuple(steps)
        self._index = 0

    @property
    def policy_id(self) -> str:
        return "metered"

    @property
    def policy_revision(self) -> str:
        return "fixture-v1"

    def act(
        self,
        messages: Sequence[ChatMessage],
        *,
        seed: int,
        timeout_seconds: float | None = None,
    ) -> PolicyStep:
        del messages, seed, timeout_seconds
        step = self._steps[self._index]
        self._index += 1
        return step


def make_task(
    *,
    max_steps: int = 4,
    max_policy_tokens: int = 16_384,
    max_wall_time_seconds: int = 1_800,
) -> TaskSpec:
    return TaskSpec(
        task_id="fixture-task",
        issue="Fix the fixture.",
        split=DatasetSplit.TRAIN,
        agent_image_digest=f"sha256:{'a' * 64}",
        provenance=TaskProvenance(
            source_repository="https://example.test/repo",
            source_license="MIT",
            base_commit="a" * 40,
            lineage_group="fixture",
            generator="manual",
            generator_version="1",
        ),
        budgets=TaskBudgets(
            max_steps=max_steps,
            max_policy_tokens=max_policy_tokens,
            max_wall_time_seconds=max_wall_time_seconds,
        ),
    )


def test_parser_accepts_one_fenced_json_action() -> None:
    action = parse_policy_action('```json\n{"kind":"search_code","query":"x"}\n```')
    assert isinstance(action, SearchCode)


def test_parser_rejects_prose() -> None:
    try:
        parse_policy_action('I will search. {"kind":"search_code","query":"x"}')
    except ActionParseError as error:
        assert "invalid JSON" in str(error)
    else:
        raise AssertionError("prose must be rejected")


def test_runner_executes_until_finish_and_closes_sandbox() -> None:
    policy = ScriptedPolicy(
        [
            '{"kind":"search_code","query":"bug"}',
            '{"kind":"finish","summary":"done"}',
        ]
    )
    sandbox = MemorySandbox()
    trajectory = AgentRunner().run(make_task(), policy, sandbox, seed=7)

    assert trajectory.termination_reason == TerminationReason.FINISHED
    assert len(trajectory.events) == 2
    assert len(sandbox.actions) == 1
    assert sandbox.closed is True
    assert trajectory.patch_sha256 is not None


def test_runner_records_parse_errors_then_recovers() -> None:
    policy = ScriptedPolicy(
        [
            "not json",
            '{"kind":"finish","summary":"stop"}',
        ]
    )
    trajectory = AgentRunner().run(make_task(), policy, MemorySandbox(), seed=0)

    assert trajectory.termination_reason == TerminationReason.FINISHED
    assert trajectory.events[0].parse_error is not None
    assert trajectory.events[0].tool_call is None


def test_runner_stops_at_step_budget() -> None:
    policy = ScriptedPolicy(['{"kind":"search_code","query":"x"}'])
    trajectory = AgentRunner().run(make_task(max_steps=1), policy, MemorySandbox(), seed=0)

    assert trajectory.termination_reason == TerminationReason.STEP_BUDGET


def test_runner_returns_value_errors_to_policy_and_continues() -> None:
    class InvalidInputSandbox(MemorySandbox):
        def execute(
            self,
            action: Action,
            *,
            call_id: str,
            timeout_seconds: float | None = None,
        ) -> ToolResult:
            del timeout_seconds
            self.actions.append(action)
            raise ValueError("repository path does not exist")

    policy = ScriptedPolicy(
        [
            '{"kind":"search_code","query":"bug"}',
            '{"kind":"finish","summary":"done"}',
        ]
    )
    sandbox = InvalidInputSandbox()

    trajectory = AgentRunner().run(make_task(), policy, sandbox, seed=0)

    assert trajectory.termination_reason == TerminationReason.FINISHED
    assert trajectory.events[0].tool_result is not None
    assert trajectory.events[0].tool_result.ok is False
    assert "repository path does not exist" in trajectory.events[0].tool_result.output
    assert len(trajectory.events) == 2


def test_runner_terminates_on_explicit_sandbox_infrastructure_error() -> None:
    class BrokenSandbox(MemorySandbox):
        def execute(
            self,
            action: Action,
            *,
            call_id: str,
            timeout_seconds: float | None = None,
        ) -> ToolResult:
            del action, call_id, timeout_seconds
            raise SandboxInfrastructureError("daemon unavailable")

    policy = ScriptedPolicy(['{"kind":"search_code","query":"bug"}'])

    trajectory = AgentRunner().run(make_task(), policy, BrokenSandbox(), seed=0)

    assert trajectory.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
    assert len(trajectory.events) == 1


def test_runner_counts_input_and_output_toward_policy_token_budget() -> None:
    policy = MeteredPolicy(
        [
            PolicyStep(
                raw_output='{"kind":"search_code","query":"bug"}',
                token_usage=TokenUsage(input_tokens=250, output_tokens=7),
            )
        ]
    )
    sandbox = MemorySandbox()

    trajectory = AgentRunner().run(make_task(max_policy_tokens=256), policy, sandbox, seed=0)

    assert trajectory.termination_reason == TerminationReason.TOKEN_BUDGET
    assert sandbox.actions == []
    assert trajectory.events[0].tool_result is not None
    assert trajectory.events[0].tool_result.output == "policy token budget exceeded"


def test_runner_enforces_token_budget_on_parse_error_turn() -> None:
    policy = MeteredPolicy(
        [
            PolicyStep(
                raw_output="not json",
                token_usage=TokenUsage(input_tokens=250, output_tokens=7),
            )
        ]
    )

    trajectory = AgentRunner().run(
        make_task(max_policy_tokens=256), policy, MemorySandbox(), seed=0
    )

    assert trajectory.termination_reason == TerminationReason.TOKEN_BUDGET
    assert len(trajectory.events) == 1
    assert trajectory.events[0].parse_error is not None


def test_runner_does_not_start_tool_after_policy_exhausts_wall_budget() -> None:
    now = [0.0]

    class AdvancingPolicy(ScriptedPolicy):
        def act(
            self,
            messages: Sequence[ChatMessage],
            *,
            seed: int,
            timeout_seconds: float | None = None,
        ) -> PolicyStep:
            now[0] = 2.0
            return super().act(messages, seed=seed, timeout_seconds=timeout_seconds)

    sandbox = MemorySandbox()
    trajectory = AgentRunner(clock=lambda: now[0]).run(
        make_task(max_wall_time_seconds=1),
        AdvancingPolicy(['{"kind":"search_code","query":"bug"}']),
        sandbox,
        seed=0,
    )

    assert trajectory.termination_reason == TerminationReason.WALL_TIME_BUDGET
    assert sandbox.actions == []
    assert trajectory.events[0].tool_result is not None
    assert trajectory.events[0].tool_result.executed is False


def test_runner_passes_remaining_deadline_to_policy_and_sandbox() -> None:
    class CapturingPolicy(ScriptedPolicy):
        def __init__(self) -> None:
            super().__init__(
                (
                    '{"kind":"search_code","query":"bug"}',
                    '{"kind":"finish"}',
                )
            )
            self.timeouts: list[float | None] = []

        def act(
            self,
            messages: Sequence[ChatMessage],
            *,
            seed: int,
            timeout_seconds: float | None = None,
        ) -> PolicyStep:
            self.timeouts.append(timeout_seconds)
            return super().act(messages, seed=seed, timeout_seconds=timeout_seconds)

    policy = CapturingPolicy()
    sandbox = MemorySandbox()

    trajectory = AgentRunner().run(
        make_task(max_wall_time_seconds=10),
        policy,
        sandbox,
        seed=0,
    )

    assert trajectory.termination_reason == TerminationReason.FINISHED
    assert all(timeout is not None and 0 < timeout <= 10 for timeout in policy.timeouts)
    assert len(sandbox.timeouts) == 1
    assert sandbox.timeouts[0] is not None and 0 < sandbox.timeouts[0] <= 10


def test_runner_classifies_policy_context_overflow_as_budget_exhaustion() -> None:
    class OverflowPolicy(ScriptedPolicy):
        def act(
            self,
            messages: Sequence[ChatMessage],
            *,
            seed: int,
            timeout_seconds: float | None = None,
        ) -> PolicyStep:
            del messages, seed, timeout_seconds
            raise PolicyContextLengthError("too many tokens")

    trajectory = AgentRunner().run(
        make_task(),
        OverflowPolicy(['{"kind":"finish"}']),
        MemorySandbox(),
        seed=0,
    )

    assert trajectory.termination_reason == TerminationReason.CONTEXT_BUDGET


def test_runner_compacts_large_tool_observation_before_next_policy_turn() -> None:
    class LargeOutputSandbox(MemorySandbox):
        def execute(
            self,
            action: Action,
            *,
            call_id: str,
            timeout_seconds: float | None = None,
        ) -> ToolResult:
            del timeout_seconds
            self.actions.append(action)
            return ToolResult(call_id=call_id, ok=True, output="x" * 20_000, duration_ms=1)

    class CapturingPolicy(MeteredPolicy):
        def __init__(self) -> None:
            super().__init__(
                (
                    PolicyStep(raw_output='{"kind":"search_code","query":"bug"}'),
                    PolicyStep(raw_output='{"kind":"finish","summary":"done"}'),
                )
            )
            self.contexts: list[tuple[ChatMessage, ...]] = []

        def act(
            self,
            messages: Sequence[ChatMessage],
            *,
            seed: int,
            timeout_seconds: float | None = None,
        ) -> PolicyStep:
            self.contexts.append(tuple(messages))
            return super().act(messages, seed=seed, timeout_seconds=timeout_seconds)

    policy = CapturingPolicy()
    trajectory = AgentRunner(RunnerConfig(max_conversation_bytes=2_048)).run(
        make_task(),
        policy,
        LargeOutputSandbox(),
        seed=0,
    )

    assert trajectory.termination_reason == TerminationReason.FINISHED
    assert sum(len(message.content.encode("utf-8")) + 64 for message in policy.contexts[1]) <= 2_048
    assert "context compacted" in policy.contexts[1][-1].content
    replayed = replay_messages(
        make_task(),
        trajectory.events,
        max_conversation_bytes=2_048,
    )
    assert replayed[:-2] == policy.contexts[1]


def test_runner_compaction_counts_multibyte_utf8_content() -> None:
    messages = (
        ChatMessage(role="system", content="system"),
        ChatMessage(role="user", content="issue"),
        ChatMessage(role="assistant", content="{}"),
        ChatMessage(role="tool", content="错" * 2_000),
    )

    compacted = compact_messages(messages, 2_048)

    assert sum(len(message.content.encode("utf-8")) + 64 for message in compacted) <= 2_048
    assert "context compacted" in compacted[-1].content
