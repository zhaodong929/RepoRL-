"""Docker-backed agent sandbox with lazy SDK loading and injectable clients."""

from __future__ import annotations

import importlib
import io
import math
import posixpath
import stat
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any, Protocol, cast

from pydantic import Field, field_validator

from reporl.sandbox.base import (
    AgentSandbox,
    CommandSpec,
    PatchArtifact,
    ProcessResult,
    SandboxInfrastructureError,
    SandboxStateError,
)
from reporl.schemas import StrictModel
from reporl.tools.paths import normalize_repo_path


class DockerUnavailableError(SandboxInfrastructureError):
    """The optional Docker SDK or daemon is unavailable."""


class _ContainerCollection(Protocol):
    def run(self, image: str, command: Sequence[str], **kwargs: Any) -> Any: ...


class DockerClientLike(Protocol):
    containers: _ContainerCollection


class DockerSandboxConfig(StrictModel):
    """Security and resource settings for an ephemeral agent container."""

    image: str = Field(min_length=1)
    suite_commands: Mapping[str, CommandSpec]
    workspace_path: str = "/workspace/repo"
    user: str = Field(default="1000:1000", pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")
    memory_limit: str = "12g"
    nano_cpus: int = Field(default=4_000_000_000, ge=100_000_000)
    pids_limit: int = Field(default=512, ge=16, le=4_096)
    workspace_size: str = "8g"
    default_timeout_seconds: int = Field(default=60, ge=1, le=3_600)
    max_process_output_bytes: int = Field(default=2_000_000, ge=4_096, le=50_000_000)

    @field_validator("workspace_path")
    @classmethod
    def validate_container_path(cls, path: str) -> str:
        if not path.startswith("/") or ".." in PurePosixPath(path).parts or "\x00" in path:
            raise ValueError("container paths must be absolute normalized POSIX paths")
        normalized = path.rstrip("/")
        if normalized == "/workspace" or not normalized.startswith("/workspace/"):
            raise ValueError("workspace_path must be a child of /workspace")
        return normalized


class DockerAgentSandbox(AgentSandbox):
    """A disposable, network-disabled agent workspace in Docker."""

    def __init__(self, container: Any, config: DockerSandboxConfig) -> None:
        self._container = container
        self._config = config
        self._closed = False

    @classmethod
    def start(
        cls,
        repository_snapshot: Path,
        config: DockerSandboxConfig,
        *,
        client: DockerClientLike | None = None,
    ) -> DockerAgentSandbox:
        """Upload a sanitized snapshot into a disposable in-container tmpfs."""

        snapshot = repository_snapshot.resolve(strict=True)
        if not snapshot.is_dir():
            raise ValueError("repository_snapshot must be a directory")
        archive_root = PurePosixPath(config.workspace_path).relative_to("/workspace").as_posix()
        snapshot_archive = _directory_archive(snapshot, archive_root, config.user)
        docker_client = client if client is not None else _docker_client_from_environment()
        try:
            container = docker_client.containers.run(
                config.image,
                ("sleep", "infinity"),
                detach=True,
                network_disabled=True,
                read_only=True,
                user=config.user,
                working_dir="/workspace",
                mem_limit=config.memory_limit,
                nano_cpus=config.nano_cpus,
                pids_limit=config.pids_limit,
                cap_drop=("ALL",),
                security_opt=("no-new-privileges:true",),
                volumes={},
                tmpfs={
                    "/tmp": "rw,nosuid,nodev,size=256m,mode=1777",
                    "/workspace": (f"rw,nosuid,nodev,size={config.workspace_size},mode=1777"),
                },
                labels={"reporl.role": "agent"},
            )
        except Exception as error:
            raise DockerUnavailableError("failed to start agent container") from error

        sandbox = cls(container, config)
        try:
            sandbox._put_archive("/workspace", snapshot_archive)
            sandbox._initialize_fresh_repository()
            return sandbox
        except Exception:
            sandbox.close()
            raise

    @property
    def available_suites(self) -> frozenset[str]:
        return frozenset(self._config.suite_commands)

    def search_code(
        self,
        query: str,
        path: str,
        max_results: int,
        *,
        timeout_seconds: float | None = None,
    ) -> ProcessResult:
        deadline = self._deadline(timeout_seconds)
        resolved = self._resolve_workspace_path(path, allow_dot=True, deadline=deadline)
        spec = CommandSpec(
            argv=(
                "rg",
                "--line-number",
                "--no-heading",
                "--color=never",
                "--glob=!.git/**",
                "--",
                query,
                resolved,
            ),
            timeout_seconds=self._command_timeout(deadline),
        )
        result = self._exec(spec)
        lines = result.stdout.splitlines(keepends=True)
        if len(lines) > max_results:
            result = result.model_copy(
                update={
                    "stdout": "".join(lines[:max_results])
                    + f"...[limited to {max_results} matches]\n"
                }
            )
        return result

    def read_file(
        self,
        path: str,
        start_line: int,
        end_line: int,
        *,
        timeout_seconds: float | None = None,
    ) -> ProcessResult:
        deadline = self._deadline(timeout_seconds)
        resolved = self._resolve_workspace_path(path, deadline=deadline)
        spec = CommandSpec(
            argv=("sed", "-n", f"{start_line},{end_line}p", "--", resolved),
            timeout_seconds=self._command_timeout(deadline),
        )
        return self._exec(spec)

    def apply_patch(
        self,
        patch: str,
        *,
        timeout_seconds: float | None = None,
    ) -> ProcessResult:
        deadline = self._deadline(timeout_seconds)
        digest = PatchArtifact(content=patch).sha256
        assert digest is not None
        archive_name = f"reporl-{digest}.patch"
        self._put_text_archive("/tmp", archive_name, patch)
        patch_path = f"/tmp/{archive_name}"
        check = CommandSpec(
            argv=("git", "apply", "--check", "--whitespace=nowarn", patch_path),
            timeout_seconds=self._config.default_timeout_seconds,
        )
        apply = CommandSpec(
            argv=("git", "apply", "--whitespace=nowarn", patch_path),
            timeout_seconds=self._config.default_timeout_seconds,
        )
        cleanup = CommandSpec(
            argv=("rm", "-f", "--", patch_path),
            timeout_seconds=self._config.default_timeout_seconds,
        )
        try:
            checked = self._exec(self._with_deadline(check, deadline))
            if checked.exit_code != 0:
                return checked
            return self._exec(self._with_deadline(apply, deadline))
        finally:
            self._exec(
                self._with_deadline(cleanup, deadline),
                raise_on_transport=False,
            )

    def run_suite(
        self,
        suite: str,
        *,
        timeout_seconds: float | None = None,
    ) -> ProcessResult:
        try:
            spec = self._config.suite_commands[suite]
        except KeyError as error:
            raise ValueError(f"unknown test suite: {suite}") from error
        deadline = self._deadline(timeout_seconds)
        return self._exec(
            spec.model_copy(
                update={
                    "timeout_seconds": self._command_timeout(
                        deadline,
                        maximum=spec.timeout_seconds,
                    )
                }
            )
        )

    def diff(self, *, timeout_seconds: float | None = None) -> PatchArtifact:
        deadline = self._deadline(timeout_seconds)
        intent = CommandSpec(
            argv=("git", "add", "--intent-to-add", "--", "."),
            timeout_seconds=self._command_timeout(deadline),
        )
        intent_result = self._exec(intent)
        if intent_result.exit_code != 0:
            raise SandboxInfrastructureError(
                f"failed to enumerate workspace changes: {intent_result.stderr}"
            )
        spec = CommandSpec(
            argv=("git", "diff", "--binary", "--no-ext-diff", "--"),
            timeout_seconds=self._command_timeout(deadline),
        )
        result = self._exec(spec)
        if result.exit_code != 0:
            raise SandboxInfrastructureError(f"failed to export patch: {result.stderr}")
        return PatchArtifact(content=result.stdout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._container.remove(force=True)
        except Exception as error:
            raise SandboxInfrastructureError("failed to remove agent container") from error

    def _resolve_workspace_path(
        self,
        path: str,
        *,
        allow_dot: bool = False,
        deadline: float | None = None,
    ) -> str:
        normalized = normalize_repo_path(path, allow_dot=allow_dot)
        requested = posixpath.join(self._config.workspace_path, normalized)
        spec = CommandSpec(
            argv=("realpath", "-e", "--", requested),
            timeout_seconds=self._command_timeout(deadline),
        )
        result = self._exec(spec)
        if result.exit_code != 0:
            raise ValueError(f"repository path does not exist: {path}")
        resolved = result.stdout.strip()
        if not resolved or "\n" in resolved:
            raise SandboxInfrastructureError("realpath returned an invalid path")
        try:
            common = posixpath.commonpath((self._config.workspace_path, resolved))
        except ValueError as error:
            raise ValueError("repository path escapes the workspace") from error
        if common != self._config.workspace_path:
            raise ValueError("repository path escapes the workspace")
        return resolved

    def _initialize_fresh_repository(self) -> None:
        commands = (
            CommandSpec(
                argv=("rm", "-rf", "--", f"{self._config.workspace_path}/.git"),
                timeout_seconds=self._config.default_timeout_seconds,
            ),
            CommandSpec(
                argv=("git", "init", "--quiet"),
                timeout_seconds=self._config.default_timeout_seconds,
            ),
            CommandSpec(
                argv=("git", "config", "user.name", "RepoRL Sandbox"),
                timeout_seconds=self._config.default_timeout_seconds,
            ),
            CommandSpec(
                argv=("git", "config", "user.email", "sandbox@invalid"),
                timeout_seconds=self._config.default_timeout_seconds,
            ),
            CommandSpec(
                argv=("git", "add", "--all", "--", "."),
                timeout_seconds=self._config.default_timeout_seconds,
            ),
            CommandSpec(
                argv=(
                    "git",
                    "-c",
                    "commit.gpgSign=false",
                    "commit",
                    "--quiet",
                    "--no-verify",
                    "-m",
                    "RepoRL isolated baseline",
                ),
                timeout_seconds=self._config.default_timeout_seconds,
            ),
        )
        for command in commands:
            result = self._exec(command)
            if result.exit_code != 0:
                raise DockerUnavailableError(
                    f"failed to create isolated baseline: {result.stderr or result.stdout}"
                )

    def _exec(
        self,
        spec: CommandSpec,
        *,
        workdir: str | None = None,
        raise_on_transport: bool = True,
    ) -> ProcessResult:
        if self._closed:
            raise SandboxStateError("sandbox is closed")
        argv = (
            "timeout",
            "--signal=KILL",
            str(spec.timeout_seconds),
            *spec.argv,
        )
        started = monotonic()
        try:
            exit_code, output = _bounded_container_exec(
                self._container,
                argv,
                environment=dict(spec.environment),
                workdir=workdir or spec.workdir or self._config.workspace_path,
                max_output_bytes=self._config.max_process_output_bytes,
            )
        except Exception as error:
            if not raise_on_transport:
                return ProcessResult(
                    argv=spec.argv,
                    exit_code=125,
                    stderr="sandbox transport error during cleanup",
                    duration_ms=_elapsed_ms(started),
                )
            raise SandboxInfrastructureError("Docker exec failed") from error
        stdout_bytes, stderr_bytes = output
        return ProcessResult(
            argv=spec.argv,
            exit_code=exit_code,
            stdout=_decode_output(stdout_bytes),
            stderr=_decode_output(stderr_bytes),
            duration_ms=_elapsed_ms(started),
            timed_out=exit_code == 124,
        )

    @staticmethod
    def _deadline(timeout_seconds: float | None) -> float | None:
        if timeout_seconds is None:
            return None
        return monotonic() + max(0.0, timeout_seconds)

    def _command_timeout(self, deadline: float | None, *, maximum: int | None = None) -> int:
        limit = maximum or self._config.default_timeout_seconds
        if deadline is None:
            return limit
        return max(1, min(limit, math.ceil(max(0.0, deadline - monotonic()))))

    def _with_deadline(self, spec: CommandSpec, deadline: float | None) -> CommandSpec:
        return spec.model_copy(
            update={
                "timeout_seconds": self._command_timeout(
                    deadline,
                    maximum=spec.timeout_seconds,
                )
            }
        )

    def _put_text_archive(self, destination: str, name: str, content: str) -> None:
        payload = content.encode("utf-8")
        uid, gid = _numeric_owner(self._config.user)
        archive_buffer = io.BytesIO()
        with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mode = 0o600
            info.uid = uid
            info.gid = gid
            archive.addfile(info, io.BytesIO(payload))
        self._put_archive(destination, archive_buffer.getvalue())

    def _put_archive(self, destination: str, archive: bytes) -> None:
        try:
            accepted = self._container.put_archive(destination, archive)
        except Exception as error:
            raise SandboxInfrastructureError("failed to upload archive to sandbox") from error
        if not accepted:
            raise SandboxInfrastructureError("sandbox rejected archive upload")


def _numeric_owner(user: str) -> tuple[int, int]:
    """Return the numeric owner used for daemon-extracted workspace archives."""

    try:
        uid_text, gid_text = user.split(":", maxsplit=1)
        uid, gid = int(uid_text), int(gid_text)
    except (TypeError, ValueError) as error:
        raise ValueError("sandbox user must be a numeric non-root UID:GID") from error
    if uid <= 0 or gid <= 0:
        raise ValueError("sandbox user must be a numeric non-root UID:GID")
    return uid, gid


def _directory_archive(source: Path, root_name: str, user: str) -> bytes:
    """Build a writable, non-root-owned repository archive without source Git metadata."""

    root = source.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("archive source must be a directory")
    root_path = PurePosixPath(root_name)
    if (
        root_path.is_absolute()
        or not root_path.parts
        or any(part in {"", ".", ".."} for part in root_path.parts)
    ):
        raise ValueError("archive root must be a normalized relative path")
    root_name = root_path.as_posix()
    uid, gid = _numeric_owner(user)
    buffer = io.BytesIO()

    def add_entry(archive: tarfile.TarFile, path: Path, archive_name: str) -> None:
        try:
            metadata = path.lstat()
            info = archive.gettarinfo(str(path), arcname=archive_name)
        except OSError as error:
            raise SandboxInfrastructureError(
                f"failed to read repository snapshot entry: {path.name}"
            ) from error
        info.uid = uid
        info.gid = gid
        info.uname = ""
        info.gname = ""

        if stat.S_ISDIR(metadata.st_mode):
            info.mode = (info.mode & 0o077) | 0o700
            archive.addfile(info)
            try:
                children = sorted(path.iterdir(), key=lambda child: child.name)
            except OSError as error:
                raise SandboxInfrastructureError(
                    f"failed to enumerate repository snapshot entry: {path.name}"
                ) from error
            for child in children:
                if child.name == ".git":
                    continue
                child_name = f"{archive_name}/{child.name}"
                add_entry(archive, child, child_name)
            return
        if stat.S_ISREG(metadata.st_mode):
            info.mode = (info.mode & 0o111) | 0o600
            try:
                with path.open("rb") as payload:
                    archive.addfile(info, payload)
            except OSError as error:
                raise SandboxInfrastructureError(
                    f"failed to archive repository snapshot entry: {path.name}"
                ) from error
            return
        if stat.S_ISLNK(metadata.st_mode):
            info.mode = 0o777
            archive.addfile(info)
            return
        raise SandboxInfrastructureError(
            f"repository snapshot contains unsupported entry: {path.name}"
        )

    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        add_entry(archive, root, root_name)
    return buffer.getvalue()


def _docker_client_from_environment() -> DockerClientLike:
    try:
        docker_module = importlib.import_module("docker")
        client = docker_module.from_env()
        client.ping()
        return cast(DockerClientLike, client)
    except (ImportError, AttributeError, OSError) as error:
        raise DockerUnavailableError(
            "Docker support requires the 'sandbox' extra and a reachable daemon"
        ) from error
    except Exception as error:
        raise DockerUnavailableError("the Docker daemon is not reachable") from error


def _unpack_exec_result(raw: Any) -> tuple[int, tuple[bytes | None, bytes | None]]:
    if hasattr(raw, "exit_code") and hasattr(raw, "output"):
        exit_code = int(raw.exit_code)
        output = raw.output
    elif isinstance(raw, tuple) and len(raw) == 2:
        exit_code = int(raw[0])
        output = raw[1]
    else:
        raise TypeError("unexpected Docker exec result")
    if isinstance(output, tuple) and len(output) == 2:
        return exit_code, (output[0], output[1])
    if isinstance(output, bytes):
        return exit_code, (output, None)
    raise TypeError("unexpected Docker exec output")


class _BoundedBytes:
    def __init__(self, limit: int) -> None:
        self._head_limit = limit // 2
        self._tail_limit = limit - self._head_limit
        self._head = bytearray()
        self._tail = bytearray()
        self._total = 0

    def add(self, value: bytes | None) -> None:
        if not value:
            return
        self._total += len(value)
        head_remaining = self._head_limit - len(self._head)
        if head_remaining > 0:
            self._head.extend(value[:head_remaining])
            value = value[head_remaining:]
        if value:
            self._tail.extend(value)
            if len(self._tail) > self._tail_limit:
                del self._tail[: len(self._tail) - self._tail_limit]

    def value(self) -> bytes:
        retained = len(self._head) + len(self._tail)
        if self._total <= retained:
            return bytes(self._head + self._tail)
        omitted = self._total - retained
        marker = f"\n...[{omitted} output bytes omitted]...\n".encode()
        return bytes(self._head) + marker + bytes(self._tail)


def _bounded_container_exec(
    container: Any,
    argv: Sequence[str],
    *,
    environment: Mapping[str, str],
    workdir: str,
    max_output_bytes: int,
) -> tuple[int, tuple[bytes | None, bytes | None]]:
    """Stream real Docker exec output while retaining bounded head and tail evidence."""

    api = getattr(getattr(container, "client", None), "api", None)
    container_id = getattr(container, "id", None)
    if api is None or not isinstance(container_id, str):
        raw = container.exec_run(
            argv,
            environment=dict(environment),
            workdir=workdir,
            demux=True,
        )
        return _unpack_exec_result(raw)

    created = api.exec_create(
        container_id,
        list(argv),
        stdout=True,
        stderr=True,
        environment=dict(environment),
        workdir=workdir,
    )
    exec_id = created.get("Id") if isinstance(created, dict) else None
    if not isinstance(exec_id, str) or not exec_id:
        raise TypeError("Docker exec_create returned no exec ID")
    stdout = _BoundedBytes(max_output_bytes)
    stderr = _BoundedBytes(max_output_bytes)
    stream = api.exec_start(exec_id, stream=True, demux=True)
    for chunk in stream:
        if isinstance(chunk, tuple) and len(chunk) == 2:
            stdout.add(chunk[0])
            stderr.add(chunk[1])
        elif isinstance(chunk, bytes):
            stdout.add(chunk)
        else:
            raise TypeError("Docker exec stream returned an invalid chunk")
    inspected = api.exec_inspect(exec_id)
    exit_code = inspected.get("ExitCode") if isinstance(inspected, dict) else None
    if not isinstance(exit_code, int):
        raise TypeError("Docker exec_inspect returned no exit code")
    return exit_code, (stdout.value(), stderr.value())


def _decode_output(value: bytes | None) -> str:
    return "" if value is None else value.decode("utf-8", errors="replace")


def _elapsed_ms(started: float) -> int:
    return max(0, round((monotonic() - started) * 1_000))
