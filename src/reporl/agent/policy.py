"""Policy protocol plus deterministic and OpenAI-compatible implementations."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from reporl.agent.models import ChatMessage, PolicyStep
from reporl.schemas import TokenUsage


class PolicyContextLengthError(RuntimeError):
    """The prepared prompt exceeds the policy's fixed tokenizer window."""


class PolicyTimeoutError(RuntimeError):
    """A policy call exceeded either the task deadline or backend timeout."""

    def __init__(self, message: str, *, task_deadline: bool) -> None:
        super().__init__(message)
        self.task_deadline = task_deadline


@runtime_checkable
class PolicyBackend(Protocol):
    @property
    def policy_id(self) -> str: ...

    @property
    def policy_revision(self) -> str: ...

    def act(
        self,
        messages: Sequence[ChatMessage],
        *,
        seed: int,
        timeout_seconds: float | None = None,
    ) -> PolicyStep: ...


class ScriptedPolicy:
    """Deterministic policy used by contract and integration tests."""

    def __init__(self, outputs: Sequence[str], *, policy_id: str = "scripted") -> None:
        if not outputs:
            raise ValueError("ScriptedPolicy requires at least one output")
        self._outputs = tuple(outputs)
        self._index = 0
        self._policy_id = policy_id

    @property
    def policy_id(self) -> str:
        return self._policy_id

    @property
    def policy_revision(self) -> str:
        return "deterministic-v1"

    def act(
        self,
        messages: Sequence[ChatMessage],
        *,
        seed: int,
        timeout_seconds: float | None = None,
    ) -> PolicyStep:
        del messages, seed, timeout_seconds
        if self._index >= len(self._outputs):
            raise RuntimeError("scripted policy exhausted")
        output = self._outputs[self._index]
        self._index += 1
        return PolicyStep(raw_output=output)


class OpenAICompatiblePolicy:
    """Small dependency-free client for a local or remote vLLM-compatible endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        revision: str = "server-managed",
        timeout_seconds: float = 120.0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._api_key = api_key
        self._revision = revision
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens

    @property
    def policy_id(self) -> str:
        return self._model

    @property
    def policy_revision(self) -> str:
        return self._revision

    def act(
        self,
        messages: Sequence[ChatMessage],
        *,
        seed: int,
        timeout_seconds: float | None = None,
    ) -> PolicyStep:
        wire_messages: list[dict[str, str]] = []
        for message in messages:
            if message.role == "tool":
                wire_messages.append(
                    {
                        "role": "user",
                        "content": f"TOOL_OBSERVATION\n{message.content}",
                    }
                )
            else:
                wire_messages.append(
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                )
        payload = {
            "model": self._model,
            "messages": wire_messages,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
            "seed": seed,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.monotonic()
        effective_timeout = _effective_timeout(self._timeout_seconds, timeout_seconds)
        try:
            with urllib.request.urlopen(request, timeout=effective_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise PolicyTimeoutError(
                    "policy endpoint request timed out",
                    task_deadline=_uses_task_deadline(self._timeout_seconds, timeout_seconds),
                ) from error
            raise RuntimeError(f"policy endpoint request failed: {error}") from error
        except TimeoutError as error:
            raise PolicyTimeoutError(
                "policy endpoint request timed out",
                task_deadline=_uses_task_deadline(self._timeout_seconds, timeout_seconds),
            ) from error
        except json.JSONDecodeError as error:
            raise RuntimeError(f"policy endpoint request failed: {error}") from error
        latency_ms = round((time.monotonic() - started) * 1_000)
        try:
            content = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
            if not isinstance(content, str):
                raise TypeError("message content is not text")
            input_tokens = int(usage.get("prompt_tokens", 0))
            output_tokens = int(usage.get("completion_tokens", 0))
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise RuntimeError("policy endpoint returned an invalid response") from error
        return PolicyStep(
            raw_output=content,
            latency_ms=latency_ms,
            token_usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )


def _effective_timeout(configured: float, remaining: float | None) -> float:
    if remaining is None:
        return configured
    if remaining <= 0:
        raise PolicyTimeoutError("task deadline expired before policy call", task_deadline=True)
    return min(configured, remaining)


def _uses_task_deadline(configured: float, remaining: float | None) -> bool:
    return remaining is not None and remaining <= configured
