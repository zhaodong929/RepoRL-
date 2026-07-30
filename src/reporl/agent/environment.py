"""Adapter from the AgentRunner protocol to the hardened tool gateway."""

from __future__ import annotations

from pathlib import Path

from reporl.sandbox.base import AgentSandbox
from reporl.sandbox.docker import DockerAgentSandbox, DockerClientLike, DockerSandboxConfig
from reporl.schemas import Action, TaskSpec, ToolCall, ToolResult
from reporl.tools.gateway import ToolGateway


class DockerTaskEnvironment:
    def __init__(
        self,
        repository_snapshot: Path,
        config: DockerSandboxConfig,
        *,
        client: DockerClientLike | None = None,
    ) -> None:
        self._snapshot = repository_snapshot
        self._config = config
        self._client = client
        self._sandbox: AgentSandbox | None = None
        self._gateway: ToolGateway | None = None

    def reset(self, task: TaskSpec) -> None:
        if self._sandbox is not None:
            raise RuntimeError("environment is already active")
        if not (
            self._config.image == task.agent_image_digest
            or self._config.image.endswith(f"@{task.agent_image_digest}")
        ):
            raise ValueError("agent image is not pinned to the TaskSpec digest")
        sandbox = DockerAgentSandbox.start(
            self._snapshot,
            self._config,
            client=self._client,
        )
        self._sandbox = sandbox
        self._gateway = ToolGateway(task, sandbox)

    def execute(
        self,
        action: Action,
        *,
        call_id: str,
        timeout_seconds: float | None = None,
    ) -> ToolResult:
        gateway = self._require_gateway()
        return gateway.execute(
            ToolCall(call_id=call_id, action=action),
            timeout_seconds=timeout_seconds,
        )

    def diff(self, *, timeout_seconds: float | None = None) -> str:
        gateway = self._require_gateway()
        patch = gateway.export_patch(timeout_seconds=timeout_seconds)
        return patch.content

    def close(self) -> None:
        if self._sandbox is None:
            return
        try:
            self._sandbox.close()
        finally:
            self._sandbox = None
            self._gateway = None

    def _require_gateway(self) -> ToolGateway:
        if self._gateway is None:
            raise RuntimeError("environment is not active")
        return self._gateway
