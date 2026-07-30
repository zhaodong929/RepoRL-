"""Docker-backed repeated test evidence for task admission."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field

from reporl.sandbox.base import CommandSpec, SandboxInfrastructureError
from reporl.sandbox.docker import (
    DockerClientLike,
    _directory_archive,
    _docker_client_from_environment,
)
from reporl.schemas import StrictModel
from reporl.tasks.admission import SnapshotKind, SnapshotValidation, TestRunOutcome
from reporl.tasks.manifest import TrustedCommand, VerifierTestSuite
from reporl.verifier.junit import MAX_JUNIT_BYTES, parse_junit_xml


class AdmissionExecutionError(RuntimeError):
    """Admission evidence could not be produced reliably."""


class SnapshotAdmissionExecutor(Protocol):
    def run_snapshot(
        self,
        *,
        kind: SnapshotKind,
        snapshot: Path,
        snapshot_sha256: str,
        hidden_tests: Path,
        image: str,
        suites: tuple[VerifierTestSuite, ...],
        repetitions: int,
    ) -> SnapshotValidation:
        """Execute the requested suites repeatedly in fresh isolated environments."""


class DockerAdmissionConfig(StrictModel):
    user: str = Field(default="1000:1000", pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")
    memory_limit: str = "12g"
    nano_cpus: int = Field(default=4_000_000_000, ge=100_000_000)
    pids_limit: int = Field(default=512, ge=16, le=4_096)
    workspace_size: str = "8g"
    setup_timeout_seconds: int = Field(default=60, ge=1, le=600)


class DockerAdmissionExecutor:
    """Collect JUnit-backed evidence with one fresh container per suite attempt."""

    _REPOSITORY = "/workspace/repo"
    _HIDDEN_TESTS = "/workspace/verifier-tests"

    def __init__(
        self,
        config: DockerAdmissionConfig | None = None,
        *,
        client: DockerClientLike | None = None,
    ) -> None:
        self._config = config or DockerAdmissionConfig()
        self._client = client

    def run_snapshot(
        self,
        *,
        kind: SnapshotKind,
        snapshot: Path,
        snapshot_sha256: str,
        hidden_tests: Path,
        image: str,
        suites: tuple[VerifierTestSuite, ...],
        repetitions: int,
    ) -> SnapshotValidation:
        if repetitions < 3:
            raise ValueError("admission requires at least three independent repetitions")
        snapshot_archive = _directory_archive(
            snapshot.resolve(strict=True),
            "repo",
            self._config.user,
        )
        hidden_archive = _directory_archive(
            hidden_tests.resolve(strict=True),
            "verifier-tests",
            self._config.user,
        )
        client = self._client or _docker_client_from_environment()
        outcomes: list[TestRunOutcome] = []
        for suite in suites:
            for attempt in range(1, repetitions + 1):
                outcomes.append(
                    self._run_attempt(
                        client=client,
                        image=image,
                        snapshot_archive=snapshot_archive,
                        hidden_archive=hidden_archive if suite.name == "target" else None,
                        suite=suite,
                        attempt=attempt,
                    )
                )
        return SnapshotValidation(
            kind=kind,
            snapshot_sha256=snapshot_sha256,
            outcomes=tuple(outcomes),
        )

    def _run_attempt(
        self,
        *,
        client: DockerClientLike,
        image: str,
        snapshot_archive: bytes,
        hidden_archive: bytes | None,
        suite: VerifierTestSuite,
        attempt: int,
    ) -> TestRunOutcome:
        container: Any | None = None
        primary_error: BaseException | None = None
        started = time.monotonic()
        try:
            container = client.containers.run(
                image,
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
                labels={"reporl.role": "task-admission", "reporl.suite": suite.name},
            )
            self._put_archive(container, "/workspace", snapshot_archive)
            if hidden_archive is not None:
                self._put_archive(container, "/workspace", hidden_archive)
            prepared = self._exec(
                container,
                CommandSpec(
                    argv=("mkdir", "-p", "/tmp/reporl-junit"),
                    timeout_seconds=self._config.setup_timeout_seconds,
                ),
            )
            if prepared[0] != 0:
                raise AdmissionExecutionError("failed to prepare JUnit output directory")
            command = self._runtime_command(suite.command)
            exit_code, _, _, timed_out = self._exec(container, command)
            if timed_out:
                raise AdmissionExecutionError(
                    f"admission suite timed out: {suite.name}, attempt {attempt}"
                )
            junit_exit, junit_stdout, _, junit_timed_out = self._exec(
                container,
                CommandSpec(
                    argv=("head", "-c", str(MAX_JUNIT_BYTES + 1), "--", suite.junit_path),
                    timeout_seconds=self._config.setup_timeout_seconds,
                ),
            )
            if junit_exit != 0 or junit_timed_out:
                raise AdmissionExecutionError(
                    f"admission suite did not produce JUnit: {suite.name}, attempt {attempt}"
                )
            junit = junit_stdout.encode("utf-8")
            report = parse_junit_xml(junit, expected_test_ids=suite.expected_test_ids)
            if report.missing_test_ids or report.unexpected_test_ids:
                raise AdmissionExecutionError(
                    f"admission suite collected non-canonical tests: {suite.name}"
                )
            return TestRunOutcome(
                suite=suite.name,
                attempt=attempt,
                exit_code=exit_code,
                passed=report.passed,
                failed=report.failed,
                errors=report.errors,
                skipped=report.skipped,
                duration_ms=max(0, round((time.monotonic() - started) * 1_000)),
                report_sha256=f"sha256:{hashlib.sha256(junit).hexdigest()}",
            )
        except BaseException as error:
            primary_error = error
            if isinstance(error, (AdmissionExecutionError, ValueError)):
                raise
            raise AdmissionExecutionError(
                f"Docker admission failed for {suite.name}, attempt {attempt}"
            ) from error
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception as error:
                    if primary_error is None:
                        raise AdmissionExecutionError(
                            "failed to remove admission container"
                        ) from error

    def _runtime_command(self, command: TrustedCommand) -> CommandSpec:
        def relocate(value: str) -> str:
            return value.replace("/verifier-tests", self._HIDDEN_TESTS)

        workdir = self._REPOSITORY if command.cwd == "." else f"{self._REPOSITORY}/{command.cwd}"
        return CommandSpec(
            argv=tuple(relocate(argument) for argument in command.argv),
            timeout_seconds=min(command.timeout_seconds, 3_600),
            environment={
                variable.name: relocate(variable.value) for variable in command.environment
            },
            workdir=relocate(workdir),
        )

    @staticmethod
    def _put_archive(container: Any, destination: str, archive: bytes) -> None:
        try:
            accepted = container.put_archive(destination, archive)
        except Exception as error:
            raise AdmissionExecutionError("failed to upload admission artifact") from error
        if not accepted:
            raise AdmissionExecutionError("admission container rejected artifact upload")

    @staticmethod
    def _exec(
        container: Any,
        command: CommandSpec,
    ) -> tuple[int, str, str, bool]:
        argv = ("timeout", "--signal=KILL", str(command.timeout_seconds), *command.argv)
        try:
            raw = container.exec_run(
                argv,
                environment=dict(command.environment),
                workdir=command.workdir,
                demux=True,
            )
            exit_code = int(raw.exit_code if hasattr(raw, "exit_code") else raw[0])
            output = raw.output if hasattr(raw, "output") else raw[1]
            if isinstance(output, tuple):
                stdout_raw, stderr_raw = output
            elif isinstance(output, bytes):
                stdout_raw, stderr_raw = output, None
            else:
                raise TypeError("unexpected Docker exec output")
        except Exception as error:
            raise SandboxInfrastructureError("Docker admission exec failed") from error
        stdout = "" if stdout_raw is None else stdout_raw.decode("utf-8", errors="replace")
        stderr = "" if stderr_raw is None else stderr_raw.decode("utf-8", errors="replace")
        return exit_code, stdout, stderr, exit_code == 124
