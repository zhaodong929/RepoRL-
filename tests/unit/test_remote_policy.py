from __future__ import annotations

import json
import threading

import pytest

from reporl.agent.models import ChatMessage, PolicyIdentity, PolicyStep
from reporl.agent.policy import PolicyTimeoutError, ScriptedPolicy
from reporl.agent.policy_server import (
    PolicyServerInfo,
    PolicyServerRequest,
    PolicyServerResponse,
    execute_request,
)
from reporl.agent.remote_policy import RemoteTracePolicy, fetch_policy_server_info


def policy_identity() -> PolicyIdentity:
    return PolicyIdentity(
        model_id="model",
        model_revision="commit",
        tokenizer_revision="commit",
        tokenizer_class="Tokenizer",
        quantization="bnb-nf4-double",
        model_preparation="inference-only",
        torch_dtype="bfloat16",
        attention_implementation="sdpa",
        transformers_version="4.48.0",
        chat_template_sha256=f"sha256:{'a' * 64}",
        generation_config_sha256=f"sha256:{'b' * 64}",
        special_token_ids=(("eos", 1),),
        max_input_tokens=4096,
        max_new_tokens=256,
        sampling_temperature=0.7,
        sampling_top_p=1.0,
    )


def test_remote_policy_round_trips_a_policy_step() -> None:
    expected = PolicyStep(raw_output='{"kind":"finish"}')

    def transport(url: str, payload: bytes, headers: dict[str, str], timeout: float) -> bytes:
        assert url == "http://gpu:8010/action"
        assert json.loads(payload)["seed"] == 7
        assert headers["Authorization"] == "Bearer secret"
        assert timeout == 10
        return (
            PolicyServerResponse(
                policy_id="model",
                policy_revision="revision",
                step=expected,
            )
            .model_dump_json()
            .encode("utf-8")
        )

    policy = RemoteTracePolicy(
        base_url="http://gpu:8010",
        policy_id="model",
        policy_revision="revision",
        bearer_token="secret",
        timeout_seconds=10,
        transport=transport,
    )
    result = policy.act((ChatMessage(role="user", content="task"),), seed=7)
    assert result == expected


def test_remote_policy_caps_transport_timeout_at_task_deadline() -> None:
    def transport(url: str, payload: bytes, headers: dict[str, str], timeout: float) -> bytes:
        del url, payload, headers
        assert timeout == 3
        return (
            PolicyServerResponse(
                policy_id="model",
                policy_revision="revision",
                step=PolicyStep(raw_output='{"kind":"finish"}'),
            )
            .model_dump_json()
            .encode()
        )

    policy = RemoteTracePolicy(
        base_url="http://gpu:8010",
        policy_id="model",
        policy_revision="revision",
        bearer_token="secret",
        timeout_seconds=10,
        transport=transport,
    )

    policy.act(
        (ChatMessage(role="user", content="task"),),
        seed=7,
        timeout_seconds=3,
    )


def test_remote_policy_marks_task_deadline_timeout() -> None:
    def transport(url: str, payload: bytes, headers: dict[str, str], timeout: float) -> bytes:
        del url, payload, headers, timeout
        raise TimeoutError

    policy = RemoteTracePolicy(
        base_url="http://gpu:8010",
        policy_id="model",
        policy_revision="revision",
        bearer_token="secret",
        timeout_seconds=10,
        transport=transport,
    )

    with pytest.raises(PolicyTimeoutError) as caught:
        policy.act(
            (ChatMessage(role="user", content="task"),),
            seed=7,
            timeout_seconds=3,
        )

    assert caught.value.task_deadline


def test_policy_server_serializes_generation_with_a_lock() -> None:
    request = PolicyServerRequest(
        messages=(ChatMessage(role="user", content="task"),),
        seed=0,
    )
    result = execute_request(
        ScriptedPolicy(['{"kind":"finish"}']),
        request,
        lock=threading.Lock(),
    )
    assert result.raw_output == '{"kind":"finish"}'


def test_health_handshake_is_authenticated_and_fingerprint_bound() -> None:
    identity = policy_identity()

    def transport(url: str, headers: dict[str, str], timeout: float) -> bytes:
        assert url == "http://gpu:8010/health"
        assert headers["Authorization"] == "Bearer secret"
        assert timeout == 10
        return (
            PolicyServerInfo(
                policy_id=identity.model_id,
                policy_revision=identity.digest,
                policy_identity=identity,
            )
            .model_dump_json()
            .encode()
        )

    info = fetch_policy_server_info(
        "http://gpu:8010",
        bearer_token="secret",
        timeout_seconds=10,
        transport=transport,
    )

    assert info.policy_revision == identity.digest


def test_policy_fingerprint_changes_with_sampling_or_adapter_content() -> None:
    identity = policy_identity()

    assert identity.model_copy(update={"sampling_top_p": 0.95}).digest != identity.digest
    assert (
        identity.model_copy(update={"adapter_sha256": f"sha256:{'c' * 64}"}).digest
        != identity.digest
    )
