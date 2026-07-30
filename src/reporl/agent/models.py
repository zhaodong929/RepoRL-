"""Policy-facing message and generation models."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, model_validator

from reporl.schemas import GenerationTrace, StrictModel, TokenUsage


class PolicyIdentity(StrictModel):
    """Complete, content-bound identity for a rollout policy."""

    schema_version: Literal[1] = 1
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    tokenizer_class: str = Field(min_length=1)
    adapter_sha256: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    quantization: str = Field(min_length=1)
    model_preparation: str = Field(min_length=1)
    torch_dtype: str = Field(min_length=1)
    attention_implementation: str | None = None
    transformers_version: str = Field(min_length=1)
    chat_template_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    generation_config_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    special_token_ids: tuple[tuple[str, int | None], ...]
    max_input_tokens: int = Field(ge=1)
    max_new_tokens: int = Field(ge=1)
    sampling_temperature: float = Field(ge=0)
    sampling_top_p: float = Field(gt=0, le=1)
    sampling_top_k: int = Field(default=0, ge=0)

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()


class ChatMessage(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class PolicyStep(StrictModel):
    raw_output: str
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: int = Field(default=0, ge=0)
    generation_trace: GenerationTrace | None = None

    @model_validator(mode="after")
    def usage_matches_trace(self) -> PolicyStep:
        if self.generation_trace is not None:
            if len(self.generation_trace.prompt_input_ids) != self.token_usage.input_tokens:
                raise ValueError("prompt token count must match input token usage")
            if len(self.generation_trace.generated_token_ids) != self.token_usage.output_tokens:
                raise ValueError("generated token count must match output token usage")
        return self
