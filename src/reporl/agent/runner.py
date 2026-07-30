"""Budgeted, deterministic orchestration of policy and sandbox actions."""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from pydantic import Field

from reporl.agent.models import ChatMessage
from reporl.agent.parser import ActionParseError, parse_policy_action
from reporl.agent.policy import PolicyBackend, PolicyContextLengthError, PolicyTimeoutError
from reporl.agent.prompts import initial_messages, parser_error_message, tool_observation_message
from reporl.sandbox.base import SandboxInfrastructureError, SandboxStateError
from reporl.schemas import (
    Action,
    Finish,
    StrictModel,
    TaskSpec,
    TerminationReason,
    ToolCall,
    ToolResult,
    Trajectory,
    TrajectoryEvent,
)


@runtime_checkable
class SandboxProtocol(Protocol):
    def reset(self, task: TaskSpec) -> None: ...

    def execute(
        self,
        action: Action,
        *,
        call_id: str,
        timeout_seconds: float | None = None,
    ) -> ToolResult: ...

    def diff(self, *, timeout_seconds: float | None = None) -> str: ...

    def close(self) -> None: ...


class RunnerConfig(StrictModel):
    max_consecutive_invalid_actions: int = Field(default=3, ge=1, le=20)
    max_policy_output_chars: int = Field(default=100_000, ge=256)
    max_conversation_bytes: int = Field(default=3_000, ge=2_048, le=100_000)
    context_token_reserve: int = Field(default=768, ge=256, le=8_192)

    @property
    def digest(self) -> str:
        payload = self.model_dump_json().encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()


class AgentRunner:
    def __init__(
        self,
        config: RunnerConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or RunnerConfig()
        self._clock = clock

    def run(
        self,
        task: TaskSpec,
        policy: PolicyBackend,
        sandbox: SandboxProtocol,
        *,
        seed: int,
        trajectory_id: str | None = None,
    ) -> Trajectory:
        events: list[TrajectoryEvent] = []
        messages: list[ChatMessage] = list(initial_messages(task))
        termination: TerminationReason | None = None
        total_policy_tokens = 0
        consecutive_invalid = 0
        patch = ""
        started = self._clock()
        sandbox_ready = False

        try:
            sandbox.reset(task)
            sandbox_ready = True
            for step in range(task.budgets.max_steps):
                if self._elapsed(started) >= task.budgets.max_wall_time_seconds:
                    termination = TerminationReason.WALL_TIME_BUDGET
                    break
                try:
                    messages = list(compact_messages(messages, self.config.max_conversation_bytes))
                    policy_step = policy.act(
                        tuple(messages),
                        seed=seed + step,
                        timeout_seconds=self._remaining(
                            started,
                            task.budgets.max_wall_time_seconds,
                        ),
                    )
                except PolicyContextLengthError:
                    termination = TerminationReason.CONTEXT_BUDGET
                    break
                except PolicyTimeoutError as error:
                    termination = (
                        TerminationReason.WALL_TIME_BUDGET
                        if error.task_deadline
                        else TerminationReason.POLICY_ERROR
                    )
                    break
                except Exception:
                    termination = TerminationReason.POLICY_ERROR
                    break

                total_policy_tokens += policy_step.token_usage.total_tokens
                token_budget_exceeded = total_policy_tokens > task.budgets.max_policy_tokens
                wall_budget_exceeded = self._elapsed(started) >= task.budgets.max_wall_time_seconds
                raw_output = policy_step.raw_output
                if len(raw_output) > self.config.max_policy_output_chars:
                    parse_error = "policy output exceeds the configured character limit"
                    events.append(
                        TrajectoryEvent(
                            step=step,
                            raw_policy_output=raw_output[: self.config.max_policy_output_chars],
                            parse_error=parse_error,
                            token_usage=policy_step.token_usage,
                            generation_trace=policy_step.generation_trace,
                            policy_latency_ms=policy_step.latency_ms,
                        )
                    )
                    termination = (
                        TerminationReason.TOKEN_BUDGET
                        if token_budget_exceeded
                        else (
                            TerminationReason.WALL_TIME_BUDGET
                            if wall_budget_exceeded
                            else TerminationReason.INVALID_ACTION
                        )
                    )
                    break

                try:
                    action = parse_policy_action(raw_output)
                except ActionParseError as error:
                    consecutive_invalid += 1
                    error_text = str(error)
                    events.append(
                        TrajectoryEvent(
                            step=step,
                            raw_policy_output=raw_output,
                            parse_error=error_text,
                            token_usage=policy_step.token_usage,
                            generation_trace=policy_step.generation_trace,
                            policy_latency_ms=policy_step.latency_ms,
                        )
                    )
                    if token_budget_exceeded:
                        termination = TerminationReason.TOKEN_BUDGET
                        break
                    if wall_budget_exceeded:
                        termination = TerminationReason.WALL_TIME_BUDGET
                        break
                    messages.extend(
                        (
                            ChatMessage(role="assistant", content=raw_output),
                            parser_error_message(error_text),
                        )
                    )
                    if consecutive_invalid >= self.config.max_consecutive_invalid_actions:
                        termination = TerminationReason.INVALID_ACTION
                        break
                    continue

                consecutive_invalid = 0
                call_id = f"step-{step}-{uuid.uuid4().hex[:12]}"
                tool_call = ToolCall(call_id=call_id, action=action)
                if token_budget_exceeded:
                    result = ToolResult(
                        call_id=call_id,
                        ok=False,
                        output="policy token budget exceeded",
                        duration_ms=0,
                        executed=False,
                    )
                    termination = TerminationReason.TOKEN_BUDGET
                elif wall_budget_exceeded:
                    result = ToolResult(
                        call_id=call_id,
                        ok=False,
                        output="trajectory wall-time budget exceeded",
                        duration_ms=0,
                        executed=False,
                    )
                    termination = TerminationReason.WALL_TIME_BUDGET
                elif isinstance(action, Finish):
                    result = ToolResult(
                        call_id=call_id,
                        ok=True,
                        output="finish accepted; patch will be independently verified",
                        duration_ms=0,
                        executed=False,
                    )
                    termination = TerminationReason.FINISHED
                else:
                    try:
                        result = sandbox.execute(
                            action,
                            call_id=call_id,
                            timeout_seconds=self._remaining(
                                started,
                                task.budgets.max_wall_time_seconds,
                            ),
                        )
                    except ValueError as error:
                        result = ToolResult(
                            call_id=call_id,
                            ok=False,
                            output=f"tool error: {type(error).__name__}: {error}",
                            duration_ms=0,
                        )
                    except (SandboxInfrastructureError, SandboxStateError) as error:
                        result = ToolResult(
                            call_id=call_id,
                            ok=False,
                            output=f"sandbox infrastructure error: {type(error).__name__}",
                            duration_ms=0,
                        )
                        termination = TerminationReason.INFRASTRUCTURE_ERROR
                    except Exception as error:
                        result = ToolResult(
                            call_id=call_id,
                            ok=False,
                            output=f"sandbox infrastructure error: {type(error).__name__}",
                            duration_ms=0,
                        )
                        termination = TerminationReason.INFRASTRUCTURE_ERROR

                events.append(
                    TrajectoryEvent(
                        step=step,
                        raw_policy_output=raw_output,
                        tool_call=tool_call,
                        tool_result=result,
                        token_usage=policy_step.token_usage,
                        generation_trace=policy_step.generation_trace,
                        policy_latency_ms=policy_step.latency_ms,
                    )
                )
                messages.extend(
                    (
                        ChatMessage(role="assistant", content=raw_output),
                        tool_observation_message(
                            call_id,
                            {
                                "ok": result.ok,
                                "output": result.output,
                                "exit_code": result.exit_code,
                                "duration_ms": result.duration_ms,
                                "truncated": result.truncated,
                            },
                        ),
                    )
                )
                if termination in {
                    TerminationReason.FINISHED,
                    TerminationReason.TOKEN_BUDGET,
                    TerminationReason.WALL_TIME_BUDGET,
                    TerminationReason.INFRASTRUCTURE_ERROR,
                }:
                    break
                if self._elapsed(started) >= task.budgets.max_wall_time_seconds:
                    termination = TerminationReason.WALL_TIME_BUDGET
                    break
            else:
                termination = TerminationReason.STEP_BUDGET

            remaining = self._remaining(started, task.budgets.max_wall_time_seconds)
            if sandbox_ready and remaining > 0:
                patch = sandbox.diff(timeout_seconds=remaining)
                if len(patch.encode("utf-8")) > task.budgets.max_patch_bytes:
                    patch = ""
                    termination = TerminationReason.INVALID_ACTION
            elif sandbox_ready:
                termination = TerminationReason.WALL_TIME_BUDGET
        except Exception:
            termination = TerminationReason.INFRASTRUCTURE_ERROR
            patch = ""
        finally:
            try:
                sandbox.close()
            except Exception:
                termination = TerminationReason.INFRASTRUCTURE_ERROR

        patch_sha256 = hashlib.sha256(patch.encode("utf-8")).hexdigest() if patch else None
        return Trajectory(
            trajectory_id=trajectory_id or uuid.uuid4().hex,
            task_id=task.task_id,
            policy_id=policy.policy_id,
            policy_revision=policy.policy_revision,
            policy_adapter_sha256=getattr(policy, "adapter_sha256", None),
            config_digest=self.config.digest,
            max_conversation_bytes=self.config.max_conversation_bytes,
            seed=seed,
            events=tuple(events),
            termination_reason=termination or TerminationReason.INFRASTRUCTURE_ERROR,
            patch=patch,
            patch_sha256=patch_sha256,
        )

    def _elapsed(self, started: float) -> float:
        return max(0.0, self._clock() - started)

    def _remaining(self, started: float, budget: float) -> float:
        return max(0.0, budget - self._elapsed(started))


def replay_messages(
    task: TaskSpec,
    events: Sequence[TrajectoryEvent],
    *,
    max_conversation_bytes: int = RunnerConfig().max_conversation_bytes,
) -> tuple[ChatMessage, ...]:
    """Reconstruct the exact text conversation used for SFT conversion."""

    messages = list(initial_messages(task))
    for event in events:
        messages = list(compact_messages(messages, max_conversation_bytes))
        messages.append(ChatMessage(role="assistant", content=event.raw_policy_output))
        if event.parse_error is not None:
            messages.append(parser_error_message(event.parse_error))
        elif event.tool_call is not None and event.tool_result is not None:
            messages.append(
                tool_observation_message(
                    event.tool_call.call_id,
                    {
                        "ok": event.tool_result.ok,
                        "output": event.tool_result.output,
                        "exit_code": event.tool_result.exit_code,
                        "duration_ms": event.tool_result.duration_ms,
                        "truncated": event.tool_result.truncated,
                    },
                )
            )
    return tuple(messages)


def compact_messages(
    messages: Sequence[ChatMessage],
    max_bytes: int,
) -> tuple[ChatMessage, ...]:
    """Bound UTF-8 context bytes while preserving the task and newest tool evidence."""

    if max_bytes < 2_048:
        raise ValueError("max_bytes must be at least 2048")
    compacted = list(messages)
    while len(compacted) > 4 and _conversation_bytes(compacted) > max_bytes:
        del compacted[2:4]

    if len(compacted) >= 2 and _conversation_bytes(compacted) > max_bytes:
        excess = _conversation_bytes(compacted) - max_bytes
        issue = compacted[1]
        target = max(512, _utf8_size(issue.content) - excess)
        compacted[1] = issue.model_copy(
            update={"content": _truncate_middle_bytes(issue.content, target)}
        )

    while _conversation_bytes(compacted) > max_bytes:
        candidates = [
            (_utf8_size(message.content), index)
            for index, message in enumerate(compacted)
            if index != 0 and _utf8_size(message.content) > 256
        ]
        if not candidates:
            raise ValueError("conversation header exceeds max_conversation_bytes")
        _, index = max(candidates)
        message = compacted[index]
        excess = _conversation_bytes(compacted) - max_bytes
        target = max(256, _utf8_size(message.content) - excess)
        compacted[index] = message.model_copy(
            update={"content": _truncate_middle_bytes(message.content, target)}
        )
    return tuple(compacted)


def _conversation_bytes(messages: Sequence[ChatMessage]) -> int:
    return sum(_utf8_size(message.content) + 64 for message in messages)


def _utf8_size(value: str) -> int:
    return len(value.encode("utf-8"))


def _truncate_middle_bytes(value: str, limit: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    marker = b"\n...[context compacted]...\n"
    if limit <= len(marker):
        return marker[:limit].decode("ascii")
    remaining = limit - len(marker)
    head = (remaining + 1) // 2
    tail = remaining - head
    prefix = encoded[:head].decode("utf-8", errors="ignore")
    suffix = encoded[-tail:].decode("utf-8", errors="ignore") if tail else ""
    return prefix + marker.decode("ascii") + suffix
