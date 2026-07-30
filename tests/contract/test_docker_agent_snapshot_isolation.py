from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Any

import pytest

from reporl.sandbox.base import CommandSpec
from reporl.sandbox.docker import DockerAgentSandbox, DockerSandboxConfig


class ExecResult:
    exit_code = 0
    output = (b"", b"")


class FakeContainer:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.archives: list[tuple[str, bytes]] = []
        self.removed = False

    def exec_run(self, argv: tuple[str, ...], **_: Any) -> ExecResult:
        self.commands.append(tuple(argv[3:]))
        return ExecResult()

    def put_archive(self, destination: str, archive: bytes) -> bool:
        self.archives.append((destination, archive))
        return bool(destination and archive)

    def remove(self, *, force: bool) -> None:
        self.removed = force


class FakeContainers:
    def __init__(self, container: FakeContainer) -> None:
        self.container = container
        self.call: dict[str, Any] = {}

    def run(self, image: str, command: tuple[str, ...], **kwargs: Any) -> FakeContainer:
        self.call = {"image": image, "command": command, **kwargs}
        return self.container


class FakeClient:
    def __init__(self, container: FakeContainer) -> None:
        self.containers = FakeContainers(container)


def test_agent_snapshot_is_uploaded_without_original_git_or_bind_mount(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    (snapshot / ".git").mkdir(parents=True)
    (snapshot / ".git" / "config").write_text("private history", encoding="utf-8")
    (snapshot / "src").mkdir()
    (snapshot / "src" / "a.py").write_text("a = 1\n", encoding="utf-8")
    container = FakeContainer()
    client = FakeClient(container)
    config = DockerSandboxConfig(
        image=f"agent@sha256:{'a' * 64}",
        suite_commands={"target": CommandSpec(argv=("pytest", "-q"))},
    )

    sandbox = DockerAgentSandbox.start(snapshot, config, client=client)

    assert client.containers.call["volumes"] == {}
    assert client.containers.call["working_dir"] == "/workspace"
    assert len(container.archives) == 1
    destination, payload = container.archives[0]
    assert destination == "/workspace"
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        members = {member.name: member for member in archive.getmembers()}
    assert "repo/src/a.py" in members
    assert all(".git" not in name.split("/") for name in members)
    assert all((member.uid, member.gid) == (1000, 1000) for member in members.values())
    assert members["repo/src/a.py"].mode & 0o600 == 0o600
    assert all("/snapshot" not in " ".join(command) for command in container.commands)
    assert all(command[:2] != ("cp", "-a") for command in container.commands)
    sandbox.close()
    assert container.removed


def test_agent_container_user_must_be_numeric_and_non_root() -> None:
    with pytest.raises(ValueError, match="(?i)string should match pattern"):
        DockerSandboxConfig(
            image="agent",
            user="0:0",
            suite_commands={"target": CommandSpec(argv=("pytest", "-q"))},
        )
