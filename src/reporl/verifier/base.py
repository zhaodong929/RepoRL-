"""Narrow verifier sandbox interface, separate from agent tool capabilities."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from reporl.sandbox.base import PatchArtifact, ProcessResult
from reporl.verifier.models import (
    RepositoryEntry,
    SuiteExecution,
    VerifierRunSpec,
    VerifierSuiteSpec,
)


@runtime_checkable
class VerifierSandbox(Protocol):
    """A pristine environment that never exposes hidden tests to agent tools."""

    def inspect_entries(self, paths: tuple[str, ...]) -> tuple[RepositoryEntry, ...]: ...

    def apply_patch(self, patch: PatchArtifact) -> ProcessResult: ...

    def changed_paths(self) -> tuple[str, ...]: ...

    def run_suite(self, suite: VerifierSuiteSpec) -> SuiteExecution: ...

    def close(self) -> None: ...


@runtime_checkable
class VerifierSandboxFactory(Protocol):
    def create(self, manifest: VerifierRunSpec) -> VerifierSandbox: ...
