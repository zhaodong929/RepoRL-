from __future__ import annotations

import pytest

from reporl.agent.models import ChatMessage
from reporl.training.records import SFTRecord
from reporl.training.sft import SFTTokenizationError, tokenize_sft_record


class PrefixStableTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        values = [1]
        for message in messages:
            role_id = {"system": 10, "user": 20, "assistant": 30}[message["role"]]
            values.extend((role_id, *message["content"].encode("utf-8"), 2))
        if add_generation_prompt:
            values.append(30)
        return values


def record() -> SFTRecord:
    return SFTRecord(
        record_id="record-1",
        task_id="task-1",
        trajectory_id="trajectory-1",
        messages=(
            ChatMessage(role="system", content="system"),
            ChatMessage(role="user", content="issue"),
            ChatMessage(role="assistant", content="action"),
            ChatMessage(role="tool", content="result"),
            ChatMessage(role="assistant", content="finish"),
        ),
    )


def test_tokenization_masks_non_assistant_tokens() -> None:
    tokenized = tokenize_sft_record(PrefixStableTokenizer(), record(), max_length=1_000)
    labels = tokenized["labels"]
    assert any(label == -100 for label in labels)
    assert any(label != -100 for label in labels)
    assert len(labels) == len(tokenized["input_ids"])


def test_tokenization_rejects_overlong_records() -> None:
    with pytest.raises(SFTTokenizationError, match="limit"):
        tokenize_sft_record(PrefixStableTokenizer(), record(), max_length=5)
