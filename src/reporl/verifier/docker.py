"""Two-phase Docker verifier: patch first, hidden tests only in the test phase."""

from __future__ import annotations

import io
import posixpath
import secrets
import stat
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any

from pydantic import Field

from reporl.sandbox.base import (
    CommandSpec,
    PatchArtifact,
    ProcessResult,
    SandboxInfrastructureError,
)
from reporl.sandbox.docker import (
    DockerClientLike,
    _bounded_container_exec,
    _directory_archive,
    _docker_client_from_environment,
    _numeric_owner,
)
from reporl.schemas import StrictModel
from reporl.tools.output import truncate_output
from reporl.verifier.base import VerifierSandbox, VerifierSandboxFactory
from reporl.verifier.models import (
    RepositoryEntry,
    RepositoryEntryKind,
    SuiteExecution,
    VerifierRunSpec,
    VerifierSuiteSpec,
)


class DockerVerifierConfig(StrictModel):
    user: str = Field(default="1000:1000", pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")
    memory_limit: str = "12g"
    nano_cpus: int = Field(default=4_000_000_000, ge=100_000_000)
    pids_limit: int = Field(default=512, ge=16, le=4_096)
    workspace_size: str = "8g"
    setup_timeout_seconds: int = Field(default=60, ge=1, le=600)
    max_console_chars: int = Field(default=20_000, ge=256, le=1_000_000)
    max_process_output_bytes: int = Field(default=2_000_000, ge=4_096, le=50_000_000)
    max_junit_bytes: int = Field(default=10_000_000, ge=1, le=50_000_000)


class DockerVerifierFactory(VerifierSandboxFactory):
    """Create verifier containers without sharing an agent container or volume."""

    def __init__(
        self,
        config: DockerVerifierConfig | None = None,
        *,
        client: DockerClientLike | None = None,
    ) -> None:
        self._config = config or DockerVerifierConfig()
        self._client = client

    def create(self, manifest: VerifierRunSpec) -> VerifierSandbox:
        client = self._client or _docker_client_from_environment()
        return DockerVerifierSandbox.start(client, manifest, self._config)


class DockerVerifierSandbox(VerifierSandbox):
    """Verifier implementation whose patch-validation phase cannot see hidden tests."""

    _WORKSPACE = "/workspace/repo"
    _HIDDEN_TESTS = "/verifier-tests"

    def __init__(
        self,
        client: DockerClientLike,
        manifest: VerifierRunSpec,
        config: DockerVerifierConfig,
        validation_container: Any,
        snapshot_archive: bytes,
        hidden_tests_archive: bytes | None,
    ) -> None:
        self._client = client
        self._manifest = manifest
        self._config = config
        self._validation_container = validation_container
        self._test_container: Any | None = None
        self._snapshot_archive = snapshot_archive
        self._hidden_tests_archive = hidden_tests_archive
        self._patch: PatchArtifact | None = None
        self._closed = False

    @classmethod
    def start(
        cls,
        client: DockerClientLike,
        manifest: VerifierRunSpec,
        config: DockerVerifierConfig,
    ) -> DockerVerifierSandbox:
        snapshot = manifest.repository_snapshot.resolve(strict=True)
        snapshot_archive = _directory_archive(snapshot, "repo", config.user)
        hidden_tests_archive: bytes | None = None
        if manifest.hidden_tests_path is not None:
            hidden = manifest.hidden_tests_path.resolve(strict=True)
            hidden_tests_archive = _readonly_directory_archive(hidden, "tests")
        sandbox = cls(
            client,
            manifest,
            config,
            validation_container=None,
            snapshot_archive=snapshot_archive,
            hidden_tests_archive=hidden_tests_archive,
        )
        try:
            sandbox._validation_container = sandbox._new_container(role="verifier-patch")
            sandbox._initialize_workspace(sandbox._validation_container)
        except Exception:
            sandbox.close()
            raise
        return sandbox

    def inspect_entries(self, paths: tuple[str, ...]) -> tuple[RepositoryEntry, ...]:
        container = self._require_validation_container()
        entries: list[RepositoryEntry] = []
        for path in paths:
            requested = posixpath.join(self._WORKSPACE, path)
            resolved = self._exec(
                container,
                CommandSpec(
                    argv=("realpath", "-m", "--", requested),
                    timeout_seconds=self._config.setup_timeout_seconds,
                ),
            )
            canonical = resolved.stdout.strip()
            if resolved.exit_code != 0 or not canonical or "\n" in canonical:
                entries.append(RepositoryEntry(path=path, kind=RepositoryEntryKind.OTHER))
                continue
            try:
                contained = posixpath.commonpath((self._WORKSPACE, canonical)) == self._WORKSPACE
            except ValueError:
                contained = False
            if not contained:
                entries.append(RepositoryEntry(path=path, kind=RepositoryEntryKind.OTHER))
                continue

            index = self._exec(
                container,
                CommandSpec(
                    argv=("git", "ls-files", "--stage", "--", path),
                    timeout_seconds=self._config.setup_timeout_seconds,
                ),
            )
            mode = index.stdout.split(maxsplit=1)[0] if index.stdout.strip() else ""
            if mode == "120000":
                kind = RepositoryEntryKind.SYMLINK
            elif mode == "160000":
                kind = RepositoryEntryKind.SUBMODULE
            else:
                kind = self._filesystem_kind(container, requested)
            entries.append(RepositoryEntry(path=path, kind=kind))
        return tuple(entries)

    def apply_patch(self, patch: PatchArtifact) -> ProcessResult:
        container = self._require_validation_container()
        result = self._apply_to(container, patch)
        if result.exit_code == 0 and not result.timed_out:
            self._patch = patch
        return result

    def changed_paths(self) -> tuple[str, ...]:
        container = self._require_validation_container()
        intent = self._exec(
            container,
            CommandSpec(
                argv=("git", "add", "--intent-to-add", "--", "."),
                timeout_seconds=self._config.setup_timeout_seconds,
            ),
        )
        if intent.exit_code != 0:
            raise SandboxInfrastructureError("failed to enumerate applied patch paths")
        result = self._exec(
            container,
            CommandSpec(
                argv=("git", "diff", "--name-only", "-z", "--no-ext-diff", "HEAD", "--"),
                timeout_seconds=self._config.setup_timeout_seconds,
            ),
        )
        if result.exit_code != 0 or not result.stdout.endswith("\0"):
            raise SandboxInfrastructureError("Git returned invalid applied-path evidence")
        paths = tuple(path for path in result.stdout.split("\0")[:-1] if path)
        if any("\ufffd" in path or "\n" in path or "\r" in path for path in paths):
            raise SandboxInfrastructureError("Git returned an unsupported repository path")
        return tuple(sorted(paths))

    def run_suite(self, suite: VerifierSuiteSpec) -> SuiteExecution:
        if self._patch is None:
            raise SandboxInfrastructureError("verifier attempted tests before applying a patch")
        self._dispose_validation_container()
        container = self._new_container(role="verifier-suite", suite=suite.name)
        self._test_container = container
        execution: SuiteExecution | None = None
        primary_error: BaseException | None = None
        try:
            self._initialize_workspace(container)
            reapplied = self._apply_to(container, self._patch)
            if reapplied.exit_code != 0 or reapplied.timed_out:
                raise SandboxInfrastructureError(
                    "approved patch was not reproducible in a fresh suite container"
                )
            execution = self._execute_suite(container, suite)
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                container.remove(force=True)
            except Exception as error:
                if primary_error is None:
                    raise SandboxInfrastructureError(
                        "failed to dispose isolated suite container"
                    ) from error
            finally:
                self._test_container = None
        assert execution is not None
        return execution

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[Exception] = []
        for container in (self._validation_container, self._test_container):
            if container is None:
                continue
            try:
                container.remove(force=True)
            except Exception as error:
                errors.append(error)
        self._validation_container = None
        self._test_container = None
        if errors:
            raise SandboxInfrastructureError("failed to remove verifier container") from errors[0]

    def _new_container(self, *, role: str, suite: str | None = None) -> Any:
        labels = {"reporl.role": role}
        if suite is not None:
            labels["reporl.suite"] = suite
        try:
            return self._client.containers.run(
                self._manifest.image,
                ("sleep", "infinity"),
                detach=True,
                network_disabled=True,
                read_only=True,
                user=self._config.user,
                working_dir="/workspace",
                mem_limit=self._config.memory_limit,
                nano_cpus=self._config.nano_cpus,
                pids_limit=self._config.pids_limit,
                cap_drop=("ALL",),
                security_opt=("no-new-privileges:true",),
                volumes={},
                tmpfs={
                    "/tmp": "rw,nosuid,nodev,size=256m,mode=1777",
                    "/workspace": (f"rw,nosuid,nodev,size={self._config.workspace_size},mode=1777"),
                },
                labels=labels,
            )
        except Exception as error:
            raise SandboxInfrastructureError("failed to start verifier container") from error

    def _initialize_workspace(self, container: Any) -> None:
        self._put_archive_bytes(container, "/workspace", self._snapshot_archive)
        self._initialize_fresh_repository(container)

    def _initialize_fresh_repository(self, container: Any) -> None:
        commands = (
            CommandSpec(
                argv=("rm", "-rf", "--", f"{self._WORKSPACE}/.git"),
                timeout_seconds=self._config.setup_timeout_seconds,
            ),
            CommandSpec(
                argv=("git", "init", "--quiet"),
                timeout_seconds=self._config.setup_timeout_seconds,
            ),
            CommandSpec(
                argv=("git", "config", "user.name", "RepoRL Verifier"),
                timeout_seconds=self._config.setup_timeout_seconds,
            ),
            CommandSpec(
                argv=("git", "config", "user.email", "verifier@invalid"),
                timeout_seconds=self._config.setup_timeout_seconds,
            ),
            CommandSpec(
                argv=("git", "add", "--all", "--", "."),
                timeout_seconds=self._config.setup_timeout_seconds,
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
                    "RepoRL verifier baseline",
                ),
                timeout_seconds=self._config.setup_timeout_seconds,
            ),
        )
        for command in commands:
            result = self._exec(container, command)
            if result.exit_code != 0:
                raise SandboxInfrastructureError(
                    f"failed to create verifier baseline: {result.stderr or result.stdout}"
                )

    def _dispose_validation_container(self) -> None:
        if self._validation_container is not None:
            try:
                self._validation_container.remove(force=True)
            except Exception as error:
                raise SandboxInfrastructureError(
                    "failed to dispose patch-validation container"
                ) from error
            self._validation_container = None

    def _execute_suite(self, container: Any, suite: VerifierSuiteSpec) -> SuiteExecution:
        nonce = secrets.token_hex(16)
        stage_name = f".reporl-suite-{nonce}"
        stage_root = f"/tmp/{stage_name}"
        hidden_tests = f"{stage_root}/tests"
        evidence_path = f"{stage_root}/evidence/{suite.name}.xml"
        self._put_archive_bytes(
            container,
            "/tmp",
            _suite_staging_archive(stage_name, self._config.user),
        )
        if self._hidden_tests_archive is not None:
            self._put_archive_bytes(container, stage_root, self._hidden_tests_archive)

        def relocate(value: str) -> str:
            return value.replace(self._HIDDEN_TESTS, hidden_tests).replace(
                suite.junit_path, evidence_path
            )

        command = suite.command.model_copy(
            update={
                "argv": tuple(relocate(argument) for argument in suite.command.argv),
                "environment": {
                    name: relocate(value) for name, value in suite.command.environment.items()
                },
                "workdir": (
                    relocate(suite.command.workdir) if suite.command.workdir is not None else None
                ),
            }
        )
        result = self._exec(container, command)
        output_stdout, _ = truncate_output(result.stdout, self._config.max_console_chars)
        output_stderr, _ = truncate_output(result.stderr, self._config.max_console_chars)
        self._pause_container(container)
        junit = self._read_junit(container, evidence_path)
        return SuiteExecution(
            exit_code=result.exit_code,
            stdout=output_stdout,
            stderr=output_stderr,
            duration_ms=result.duration_ms,
            timed_out=result.timed_out,
            junit_xml=junit,
        )

    def _apply_to(self, container: Any, patch: PatchArtifact) -> ProcessResult:
        digest = patch.sha256
        assert digest is not None
        name = f"reporl-{digest}.patch"
        self._put_archive(container, "/tmp", name, patch.content)
        path = f"/tmp/{name}"
        check = self._exec(
            container,
            CommandSpec(
                argv=("git", "apply", "--check", "--whitespace=nowarn", path),
                timeout_seconds=self._config.setup_timeout_seconds,
            ),
        )
        if check.exit_code != 0:
            return check
        return self._exec(
            container,
            CommandSpec(
                argv=("git", "apply", "--whitespace=nowarn", path),
                timeout_seconds=self._config.setup_timeout_seconds,
            ),
        )

    def _filesystem_kind(self, container: Any, requested: str) -> RepositoryEntryKind:
        result = self._exec(
            container,
            CommandSpec(
                argv=("stat", "-c", "%F", "--", requested),
                timeout_seconds=self._config.setup_timeout_seconds,
            ),
        )
        if result.exit_code != 0:
            return RepositoryEntryKind.MISSING
        kind = result.stdout.strip().lower()
        if "symbolic link" in kind:
            return RepositoryEntryKind.SYMLINK
        if "regular file" in kind:
            return RepositoryEntryKind.REGULAR
        if "directory" in kind:
            return RepositoryEntryKind.DIRECTORY
        return RepositoryEntryKind.OTHER

    def _read_junit(self, container: Any, path: str) -> bytes | None:
        try:
            stream, metadata = container.get_archive(path, chunk_size=2 * 1024 * 1024)
        except Exception as error:
            if _docker_status_code(error) == 404:
                return None
            raise SandboxInfrastructureError("failed to retrieve JUnit evidence") from error

        expected_name = posixpath.basename(path)
        if not isinstance(metadata, Mapping):
            raise SandboxInfrastructureError("Docker returned invalid JUnit metadata")
        metadata_name = metadata.get("name")
        metadata_size = _nonnegative_metadata_integer(metadata.get("size"), "size")
        metadata_mode = _nonnegative_metadata_integer(metadata.get("mode"), "mode")
        if metadata_name != expected_name:
            raise SandboxInfrastructureError("Docker returned mismatched JUnit metadata")
        if not stat.S_ISREG(metadata_mode) or metadata.get("linkTarget") not in (None, ""):
            raise SandboxInfrastructureError("JUnit evidence is not a regular file")
        if metadata_size > self._config.max_junit_bytes:
            raise SandboxInfrastructureError("JUnit evidence exceeds the configured size limit")

        archive_limit = self._config.max_junit_bytes + 1024 * 1024
        buffer = io.BytesIO()
        try:
            for chunk in stream:
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise TypeError("Docker archive stream contained a non-byte chunk")
                if buffer.tell() + len(chunk) > archive_limit:
                    raise SandboxInfrastructureError("JUnit archive exceeds the bounded read limit")
                buffer.write(chunk)
        except SandboxInfrastructureError:
            raise
        except Exception as error:
            raise SandboxInfrastructureError("failed while reading JUnit evidence") from error

        buffer.seek(0)
        try:
            with tarfile.open(fileobj=buffer, mode="r:") as archive:
                members = archive.getmembers()
                if len(members) != 1:
                    raise SandboxInfrastructureError("JUnit archive must contain exactly one entry")
                member = members[0]
                member_path = PurePosixPath(member.name)
                if member_path.parts != (expected_name,) or not member.isreg():
                    raise SandboxInfrastructureError(
                        "JUnit archive entry is not the expected regular file"
                    )
                if member.size != metadata_size or member.size > self._config.max_junit_bytes:
                    raise SandboxInfrastructureError("JUnit archive size does not match metadata")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise SandboxInfrastructureError("JUnit archive payload is unavailable")
                payload = extracted.read(self._config.max_junit_bytes + 1)
                if len(payload) != member.size:
                    raise SandboxInfrastructureError("JUnit archive payload is truncated")
                return payload
        except SandboxInfrastructureError:
            raise
        except Exception as error:
            raise SandboxInfrastructureError("Docker returned malformed JUnit evidence") from error

    @staticmethod
    def _pause_container(container: Any) -> None:
        try:
            container.pause()
        except Exception as error:
            raise SandboxInfrastructureError(
                "failed to freeze suite container before evidence collection"
            ) from error

    def _put_archive(
        self,
        container: Any,
        destination: str,
        name: str,
        content: str,
    ) -> None:
        payload = content.encode("utf-8")
        uid, gid = _numeric_owner(self._config.user)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mode = 0o600
            info.uid = uid
            info.gid = gid
            archive.addfile(info, io.BytesIO(payload))
        self._put_archive_bytes(container, destination, buffer.getvalue())

    def _put_archive_bytes(
        self,
        container: Any,
        destination: str,
        archive: bytes,
    ) -> None:
        try:
            accepted = container.put_archive(destination, archive)
        except Exception as error:
            raise SandboxInfrastructureError("failed to upload archive to verifier") from error
        if not accepted:
            raise SandboxInfrastructureError("verifier rejected archive upload")

    def _exec(
        self,
        container: Any,
        spec: CommandSpec,
        *,
        workdir: str | None = None,
    ) -> ProcessResult:
        if self._closed:
            raise SandboxInfrastructureError("verifier sandbox is closed")
        argv: Sequence[str] = (
            "timeout",
            "--signal=KILL",
            str(spec.timeout_seconds),
            *spec.argv,
        )
        started = monotonic()
        try:
            exit_code, output = _bounded_container_exec(
                container,
                argv,
                environment=dict(spec.environment),
                workdir=workdir or spec.workdir or self._WORKSPACE,
                max_output_bytes=self._config.max_process_output_bytes,
            )
            if isinstance(output, tuple):
                stdout_raw, stderr_raw = output
            elif isinstance(output, bytes):
                stdout_raw, stderr_raw = output, None
            else:
                raise TypeError("unexpected Docker exec output")
        except Exception as error:
            raise SandboxInfrastructureError("Docker verifier exec failed") from error
        return ProcessResult(
            argv=spec.argv,
            exit_code=exit_code,
            stdout=_decode(stdout_raw),
            stderr=_decode(stderr_raw),
            duration_ms=max(0, round((monotonic() - started) * 1_000)),
            timed_out=exit_code == 124,
        )

    def _require_validation_container(self) -> Any:
        if self._closed or self._validation_container is None:
            raise SandboxInfrastructureError("patch-validation container is unavailable")
        return self._validation_container


def _decode(value: bytes | None) -> str:
    return "" if value is None else value.decode("utf-8", errors="replace")


def _readonly_directory_archive(source: Path, root_name: str) -> bytes:
    """Build a root-owned archive that the configured suite UID cannot mutate."""

    root = source.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("hidden-test source must be a directory")
    archive_root = _normalized_archive_root(root_name)
    buffer = io.BytesIO()

    def add_entry(archive: tarfile.TarFile, path: Path, archive_name: str) -> None:
        try:
            metadata = path.lstat()
            info = archive.gettarinfo(str(path), arcname=archive_name)
        except OSError as error:
            raise SandboxInfrastructureError(
                f"failed to read hidden-test entry: {path.name}"
            ) from error
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""

        if stat.S_ISDIR(metadata.st_mode):
            info.mode = 0o555
            archive.addfile(info)
            try:
                children = sorted(path.iterdir(), key=lambda child: child.name)
            except OSError as error:
                raise SandboxInfrastructureError(
                    f"failed to enumerate hidden-test entry: {path.name}"
                ) from error
            for child in children:
                if child.name == ".git":
                    continue
                add_entry(archive, child, f"{archive_name}/{child.name}")
            return
        if stat.S_ISREG(metadata.st_mode):
            info.mode = 0o555 if metadata.st_mode & 0o111 else 0o444
            try:
                with path.open("rb") as payload:
                    archive.addfile(info, payload)
            except OSError as error:
                raise SandboxInfrastructureError(
                    f"failed to archive hidden-test entry: {path.name}"
                ) from error
            return
        raise SandboxInfrastructureError(
            f"hidden-test bundle contains unsupported entry: {path.name}"
        )

    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        add_entry(archive, root, archive_root)
    return buffer.getvalue()


def _suite_staging_archive(root_name: str, user: str) -> bytes:
    """Create an immutable stage root with one suite-writable evidence directory."""

    archive_root = _normalized_archive_root(root_name)
    uid, gid = _numeric_owner(user)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, mode, owner in (
            (archive_root, 0o555, (0, 0)),
            (f"{archive_root}/evidence", 0o700, (uid, gid)),
        ):
            info = tarfile.TarInfo(name=name)
            info.type = tarfile.DIRTYPE
            info.mode = mode
            info.uid, info.gid = owner
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info)
    return buffer.getvalue()


def _normalized_archive_root(root_name: str) -> str:
    root = PurePosixPath(root_name)
    if root.is_absolute() or not root.parts or any(part in {"", ".", ".."} for part in root.parts):
        raise ValueError("archive root must be a normalized relative path")
    return root.as_posix()


def _nonnegative_metadata_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SandboxInfrastructureError(f"Docker returned invalid JUnit {field} metadata")
    return value


def _docker_status_code(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None
