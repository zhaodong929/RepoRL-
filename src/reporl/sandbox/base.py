"""Sandbox contracts shared by agent tools and trusted verification."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable

from pydantic import Field, field_validator, model_validator

from reporl.schemas import StrictModel


class SandboxError(RuntimeError):
    """Base class for errors raised at the sandbox boundary."""


class SandboxInfrastructureError(SandboxError):
    """The sandbox could not perform an operation for infrastructure reasons."""


class SandboxStateError(SandboxError):
    """An operation was attempted on an unavailable sandbox."""


class CommandSpec(StrictModel):
    """A trusted, shell-free command associated with a named operation."""

    argv: tuple[str, ...] = Field(min_length=1)
    timeout_seconds: int = Field(default=300, ge=1, le=3_600)
    environment: Mapping[str, str] = Field(default_factory=dict)
    workdir: str | None = None

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, argv: tuple[str, ...]) -> tuple[str, ...]:
        if any(not argument or "\x00" in argument for argument in argv):
            raise ValueError("command arguments must be non-empty and contain no null bytes")
        executable = argv[0].replace("\\", "/").rsplit("/", maxsplit=1)[-1].lower()
        if executable.removesuffix(".exe") in {
            "bash",
            "cmd",
            "dash",
            "fish",
            "powershell",
            "pwsh",
            "sh",
            "zsh",
        }:
            raise ValueError("test commands must not invoke a command shell")
        return argv

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, environment: Mapping[str, str]) -> Mapping[str, str]:
        for name, value in environment.items():
            if not name or "=" in name or "\x00" in name or "\x00" in value:
                raise ValueError("invalid command environment entry")
        return dict(environment)

    @field_validator("workdir")
    @classmethod
    def validate_workdir(cls, workdir: str | None) -> str | None:
        if workdir is None:
            return None
        if not workdir.startswith("/") or ".." in PurePosixPath(workdir).parts or "\x00" in workdir:
            raise ValueError("command workdir must be an absolute normalized POSIX path")
        return workdir.rstrip("/") or "/"


class ProcessResult(StrictModel):
    """Structured result from a fixed command inside a sandbox."""

    argv: tuple[str, ...]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = Field(ge=0)
    timed_out: bool = False


class PatchArtifact(StrictModel):
    """Content-addressed patch exported from an untrusted agent workspace."""

    content: str
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def populate_or_check_hash(self) -> PatchArtifact:
        digest = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.sha256 is not None and self.sha256 != digest:
            raise ValueError("patch sha256 does not match patch content")
        if self.sha256 is None:
            object.__setattr__(self, "sha256", digest)
        return self

    @property
    def byte_size(self) -> int:
        return len(self.content.encode("utf-8"))


@runtime_checkable
class AgentSandbox(Protocol):
    """Operations exposed to the trusted tool gateway, never directly to a policy."""

    @property
    def available_suites(self) -> frozenset[str]: ...

    def search_code(
        self,
        query: str,
        path: str,
        max_results: int,
        *,
        timeout_seconds: float | None = None,
    ) -> ProcessResult: ...

    def read_file(
        self,
        path: str,
        start_line: int,
        end_line: int,
        *,
        timeout_seconds: float | None = None,
    ) -> ProcessResult: ...

    def apply_patch(
        self,
        patch: str,
        *,
        timeout_seconds: float | None = None,
    ) -> ProcessResult: ...

    def run_suite(
        self,
        suite: str,
        *,
        timeout_seconds: float | None = None,
    ) -> ProcessResult: ...

    def diff(self, *, timeout_seconds: float | None = None) -> PatchArtifact: ...

    def close(self) -> None: ...
