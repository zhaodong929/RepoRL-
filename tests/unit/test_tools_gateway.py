from __future__ import annotations

from pathlib import Path

import pytest

from reporl.sandbox.base import (
    CommandSpec,
    PatchArtifact,
    ProcessResult,
    SandboxInfrastructureError,
)
from reporl.sandbox.docker import DockerAgentSandbox, DockerSandboxConfig, _BoundedBytes
from reporl.schemas import (
    ApplyPatch,
    DatasetSplit,
    ReadFile,
    RunTests,
    SearchCode,
    TaskBudgets,
    TaskProvenance,
    TaskSpec,
    ToolCall,
)
from reporl.tools.gateway import ToolGateway
from reporl.tools.patch import PatchPolicy

VALID_PATCH = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -1 +1 @@
-a = 1
+a = 2
"""


def test_stream_capture_retains_bounded_head_and_tail() -> None:
    capture = _BoundedBytes(10)
    capture.add(b"abcdefgh")
    capture.add(b"ijklmnop")

    value = capture.value()
    assert value.startswith(b"abcde")
    assert value.endswith(b"lmnop")
    assert b"6 output bytes omitted" in value


def _task(*, output_chars: int = 256) -> TaskSpec:
    return TaskSpec(
        task_id="task-001",
        issue="Fix the result",
        split=DatasetSplit.TRAIN,
        agent_image_digest=f"sha256:{'a' * 64}",
        provenance=TaskProvenance(
            source_repository="example/repo",
            source_license="MIT",
            base_commit="abcdef1",
            lineage_group="example",
            generator="fixture",
            generator_version="1",
        ),
        allowed_paths=("src",),
        forbidden_globs=("tests/**",),
        budgets=TaskBudgets(max_tool_output_chars=output_chars),
    )


class FakeSandbox:
    def __init__(self) -> None:
        self.available_suites = frozenset({"target", "regression"})
        self.calls: list[tuple[object, ...]] = []
        self.result = ProcessResult(argv=("fixed",), exit_code=0, stdout="ok", duration_ms=1)

    def search_code(
        self,
        query: str,
        path: str,
        max_results: int,
        *,
        timeout_seconds: float | None = None,
    ) -> ProcessResult:
        del timeout_seconds
        self.calls.append(("search", query, path, max_results))
        return self.result

    def read_file(
        self,
        path: str,
        start_line: int,
        end_line: int,
        *,
        timeout_seconds: float | None = None,
    ) -> ProcessResult:
        del timeout_seconds
        self.calls.append(("read", path, start_line, end_line))
        return self.result

    def apply_patch(
        self,
        patch: str,
        *,
        timeout_seconds: float | None = None,
    ) -> ProcessResult:
        del timeout_seconds
        self.calls.append(("patch", patch))
        return self.result

    def run_suite(
        self,
        suite: str,
        *,
        timeout_seconds: float | None = None,
    ) -> ProcessResult:
        del timeout_seconds
        self.calls.append(("suite", suite))
        return self.result

    def diff(self, *, timeout_seconds: float | None = None) -> PatchArtifact:
        del timeout_seconds
        return PatchArtifact(content=VALID_PATCH)

    def close(self) -> None:
        self.calls.append(("close",))


def test_gateway_dispatches_closed_action_set() -> None:
    sandbox = FakeSandbox()
    gateway = ToolGateway(_task(), sandbox)

    search = gateway.execute(ToolCall(call_id="1", action=SearchCode(query="needle")))
    read = gateway.execute(ToolCall(call_id="2", action=ReadFile(path="src/a.py")))
    tests = gateway.execute(ToolCall(call_id="3", action=RunTests(suite="target")))

    assert search.ok and read.ok and tests.ok
    assert sandbox.calls == [
        ("search", "needle", ".", 50),
        ("read", "src/a.py", 1, 200),
        ("suite", "target"),
    ]


def test_gateway_rejects_forbidden_patch_before_sandbox() -> None:
    sandbox = FakeSandbox()
    gateway = ToolGateway(_task(), sandbox)
    patch = VALID_PATCH.replace("src/a.py", "tests/test_a.py")

    result = gateway.execute(ToolCall(call_id="1", action=ApplyPatch(unified_diff=patch)))

    assert not result.ok
    assert "forbidden_path" in result.output
    assert sandbox.calls == []


def test_gateway_bounds_combined_stdout_and_stderr() -> None:
    sandbox = FakeSandbox()
    sandbox.result = ProcessResult(
        argv=("fixed",),
        exit_code=1,
        stdout="A" * 300,
        stderr="important tail",
        duration_ms=1,
    )
    gateway = ToolGateway(_task(output_chars=256), sandbox)

    result = gateway.execute(ToolCall(call_id="1", action=RunTests(suite="target")))

    assert not result.ok
    assert result.truncated
    assert len(result.output) == 256
    assert result.output.endswith("important tail")


def test_gateway_does_not_hide_infrastructure_failures() -> None:
    class BrokenSandbox(FakeSandbox):
        def run_suite(
            self,
            suite: str,
            *,
            timeout_seconds: float | None = None,
        ) -> ProcessResult:
            del suite, timeout_seconds
            raise SandboxInfrastructureError("daemon unavailable")

    with pytest.raises(SandboxInfrastructureError, match="daemon unavailable"):
        ToolGateway(_task(), BrokenSandbox()).execute(
            ToolCall(call_id="1", action=RunTests(suite="target"))
        )


class _ExecResult:
    exit_code = 0
    output = (b"", b"")


class _FakeContainer:
    def __init__(self) -> None:
        self.exec_calls: list[tuple[str, ...]] = []
        self.removed = False

    def exec_run(self, argv: tuple[str, ...], **_: object) -> _ExecResult:
        self.exec_calls.append(tuple(argv))
        return _ExecResult()

    def put_archive(self, destination: str, archive: bytes) -> bool:
        return bool(destination and archive)

    def remove(self, *, force: bool) -> None:
        self.removed = force


class _FakeContainers:
    def __init__(self, container: _FakeContainer) -> None:
        self.container = container
        self.kwargs: dict[str, object] = {}

    def run(
        self,
        image: str,
        command: tuple[str, ...],
        **kwargs: object,
    ) -> _FakeContainer:
        self.kwargs = {"image": image, "command": command, **kwargs}
        return self.container


class _FakeClient:
    def __init__(self, container: _FakeContainer) -> None:
        self.containers = _FakeContainers(container)


def test_docker_adapter_is_securely_configured_and_mockable(tmp_path: Path) -> None:
    container = _FakeContainer()
    client = _FakeClient(container)
    config = DockerSandboxConfig(
        image=f"example@sha256:{'b' * 64}",
        suite_commands={"target": CommandSpec(argv=("pytest", "-q"))},
    )

    sandbox = DockerAgentSandbox.start(tmp_path, config, client=client)

    assert client.containers.kwargs["network_disabled"] is True
    assert client.containers.kwargs["read_only"] is True
    assert client.containers.kwargs["cap_drop"] == ("ALL",)
    mounted = client.containers.kwargs["volumes"]
    assert mounted == {}
    commands = [call[3:] for call in container.exec_calls]
    assert ("rm", "-rf", "--", "/workspace/repo/.git") in commands
    assert all("/snapshot" not in " ".join(command) for command in commands)
    sandbox.close()
    assert container.removed


def test_docker_adapter_caps_suite_timeout_to_remaining_deadline(tmp_path: Path) -> None:
    container = _FakeContainer()
    client = _FakeClient(container)
    config = DockerSandboxConfig(
        image=f"example@sha256:{'b' * 64}",
        suite_commands={"target": CommandSpec(argv=("pytest", "-q"), timeout_seconds=60)},
    )
    sandbox = DockerAgentSandbox.start(tmp_path, config, client=client)

    sandbox.run_suite("target", timeout_seconds=2.5)

    timeout_argv = container.exec_calls[-1]
    assert timeout_argv[:2] == ("timeout", "--signal=KILL")
    assert 1 <= int(timeout_argv[2]) <= 3
    sandbox.close()


def test_patch_policy_used_by_gateway_accepts_valid_patch() -> None:
    assert PatchPolicy(allowed_paths=("src",)).inspect(VALID_PATCH).accepted
