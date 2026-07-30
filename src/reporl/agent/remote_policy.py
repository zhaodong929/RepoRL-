"""Trace-preserving client for split GPU-policy and CPU-sandbox deployments."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence

from reporl.agent.models import ChatMessage, PolicyIdentity, PolicyStep
from reporl.agent.policy import PolicyTimeoutError, _effective_timeout, _uses_task_deadline
from reporl.agent.policy_server import PolicyServerInfo, PolicyServerResponse

Transport = Callable[[str, bytes, dict[str, str], float], bytes]
InfoTransport = Callable[[str, dict[str, str], float], bytes]


def _default_transport(url: str, payload: bytes, headers: dict[str, str], timeout: float) -> bytes:
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            if not isinstance(body, bytes):
                raise RuntimeError("trace policy server returned a non-bytes response")
            return body
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise TimeoutError("trace policy server request timed out") from error
        raise RuntimeError(f"trace policy server request failed: {error}") from error
    except TimeoutError:
        raise


def _default_info_transport(url: str, headers: dict[str, str], timeout: float) -> bytes:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            if not isinstance(body, bytes):
                raise RuntimeError("trace policy server returned a non-bytes health response")
            return body
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"trace policy server health request failed: {error}") from error


def fetch_policy_server_info(
    base_url: str,
    *,
    bearer_token: str,
    timeout_seconds: float,
    transport: InfoTransport = _default_info_transport,
) -> PolicyServerInfo:
    payload = transport(
        base_url.rstrip("/") + "/health",
        {"Authorization": f"Bearer {bearer_token}"},
        timeout_seconds,
    )
    try:
        info = PolicyServerInfo.model_validate_json(payload)
    except ValueError as error:
        raise RuntimeError("trace policy server returned invalid health metadata") from error
    if info.policy_identity is not None:
        if info.policy_identity.model_id != info.policy_id:
            raise RuntimeError("trace policy server health metadata has inconsistent model IDs")
        if info.policy_identity.digest != info.policy_revision:
            raise RuntimeError("trace policy server health metadata has an invalid fingerprint")
    return info


class RemoteTracePolicy:
    """Call RepoRL's policy server while retaining exact rollout token evidence."""

    def __init__(
        self,
        *,
        base_url: str,
        policy_id: str,
        policy_revision: str,
        adapter_sha256: str | None = None,
        policy_identity: PolicyIdentity | None = None,
        bearer_token: str,
        timeout_seconds: float = 180.0,
        transport: Transport = _default_transport,
    ) -> None:
        if not bearer_token:
            raise ValueError("the trace policy server requires a bearer token")
        self._url = base_url.rstrip("/") + "/action"
        self._policy_id = policy_id
        self._policy_revision = policy_revision
        self._adapter_sha256 = adapter_sha256
        self._policy_identity = policy_identity
        self._token = bearer_token
        self._timeout = timeout_seconds
        self._transport = transport

    @property
    def policy_id(self) -> str:
        return self._policy_id

    @property
    def policy_revision(self) -> str:
        return self._policy_revision

    @property
    def adapter_sha256(self) -> str | None:
        return self._adapter_sha256

    @property
    def policy_identity(self) -> PolicyIdentity | None:
        return self._policy_identity

    def act(
        self,
        messages: Sequence[ChatMessage],
        *,
        seed: int,
        timeout_seconds: float | None = None,
    ) -> PolicyStep:
        payload = json.dumps(
            {
                "messages": [message.model_dump(mode="json") for message in messages],
                "seed": seed,
            }
        ).encode("utf-8")
        effective_timeout = _effective_timeout(self._timeout, timeout_seconds)
        try:
            response = self._transport(
                self._url,
                payload,
                {
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                effective_timeout,
            )
        except TimeoutError as error:
            raise PolicyTimeoutError(
                "trace policy server request timed out",
                task_deadline=_uses_task_deadline(self._timeout, timeout_seconds),
            ) from error
        try:
            parsed = PolicyServerResponse.model_validate_json(response)
        except ValueError as error:
            raise RuntimeError("trace policy server returned an invalid response") from error
        if (
            parsed.policy_id != self.policy_id
            or parsed.policy_revision != self.policy_revision
            or parsed.adapter_sha256 != self.adapter_sha256
        ):
            raise RuntimeError("trace policy server identity changed during rollout collection")
        return parsed.step
