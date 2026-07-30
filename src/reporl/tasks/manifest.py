"""Trusted verifier-only task manifests.

The manifest intentionally embeds, rather than subclasses, :class:`TaskSpec`. Calling
``agent_view`` is therefore an explicit declassification step with a narrow return type.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from reporl.schemas import StrictModel, TaskSpec
from reporl.tasks.canonical import canonical_json, canonical_sha256

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"


def _artifact_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    if (
        not normalized
        or "\x00" in normalized
        or normalized.startswith("/")
        or re.match(r"^[A-Za-z]:", normalized)
        or ".." in PurePosixPath(normalized).parts
    ):
        raise ValueError("artifact path must stay within the trusted artifact root")
    return normalized


class ArtifactReference(StrictModel):
    """Content-addressed artifact available only to trusted infrastructure."""

    path: str = Field(min_length=1, max_length=1_000)
    sha256: str = Field(pattern=_DIGEST_PATTERN)
    size_bytes: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        return _artifact_path(path)


class EnvironmentVariable(StrictModel):
    name: str = Field(pattern=r"^[A-Z_][A-Z0-9_]*$")
    value: str = Field(max_length=10_000)


class TrustedCommand(StrictModel):
    """An argv command executed directly, never through a shell."""

    argv: tuple[str, ...] = Field(min_length=1, max_length=128)
    cwd: str = "."
    timeout_seconds: int = Field(default=1_800, ge=1, le=86_400)
    environment: tuple[EnvironmentVariable, ...] = ()

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, argv: tuple[str, ...]) -> tuple[str, ...]:
        if any(not argument or "\x00" in argument for argument in argv):
            raise ValueError("command arguments must be non-empty and contain no null bytes")
        return argv

    @field_validator("cwd")
    @classmethod
    def validate_cwd(cls, cwd: str) -> str:
        if cwd == ".":
            return cwd
        return _artifact_path(cwd)

    @field_validator("environment")
    @classmethod
    def validate_environment(
        cls, environment: tuple[EnvironmentVariable, ...]
    ) -> tuple[EnvironmentVariable, ...]:
        names = tuple(item.name for item in environment)
        if len(names) != len(set(names)):
            raise ValueError("environment variable names must be unique")
        return tuple(sorted(environment, key=lambda item: item.name))


class AgentTestSuite(StrictModel):
    """Trusted command behind an agent-visible suite alias.

    The command is controller configuration, not prompt content. It must remain independent of
    the verifier-only hidden-test mount so an agent can run it in its isolated sandbox.
    """

    name: Literal["target", "regression"]
    command: TrustedCommand

    @model_validator(mode="after")
    def does_not_reference_hidden_mount(self) -> AgentTestSuite:
        values = (
            *self.command.argv,
            self.command.cwd,
            *(variable.value for variable in self.command.environment),
        )
        if any("/verifier-tests" in value for value in values):
            raise ValueError("agent suite commands must not reference verifier hidden tests")
        return self


class VerifierTestSuite(StrictModel):
    name: Literal["target", "regression"]
    command: TrustedCommand
    junit_path: str
    expected_test_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("junit_path")
    @classmethod
    def validate_junit_path(cls, path: str) -> str:
        if (
            not path.startswith("/tmp/reporl-junit/")
            or not path.endswith(".xml")
            or ".." in PurePosixPath(path).parts
            or "\x00" in path
        ):
            raise ValueError("JUnit output must be an XML file under /tmp/reporl-junit")
        return path

    @model_validator(mode="after")
    def command_writes_junit(self) -> VerifierTestSuite:
        if not any(self.junit_path in argument for argument in self.command.argv):
            raise ValueError("suite command must name its declared JUnit output path")
        if any(not test_id.strip() or "\x00" in test_id for test_id in self.expected_test_ids):
            raise ValueError("expected JUnit test IDs must be non-empty and contain no null bytes")
        if len(self.expected_test_ids) != len(set(self.expected_test_ids)):
            raise ValueError("expected JUnit test IDs must be unique")
        return self


class VerifierManifest(StrictModel):
    """Manifest held behind the verifier trust boundary.

    Hidden tests, the reference repair, commands, and pristine snapshots must never be passed to
    the policy. ``task`` is the complete and only agent-visible projection.
    """

    schema_version: Literal[1] = 1
    task: TaskSpec
    verifier_image_digest: str = Field(pattern=_DIGEST_PATTERN)
    clean_snapshot: ArtifactReference
    buggy_snapshot: ArtifactReference
    reference_snapshot: ArtifactReference
    hidden_tests: ArtifactReference
    reference_patch: ArtifactReference
    agent_test_suites: tuple[AgentTestSuite, ...]
    test_suites: tuple[VerifierTestSuite, ...]
    admission_result_sha256: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator("agent_test_suites")
    @classmethod
    def order_agent_test_suites(
        cls, suites: tuple[AgentTestSuite, ...]
    ) -> tuple[AgentTestSuite, ...]:
        names = tuple(suite.name for suite in suites)
        if len(names) != len(set(names)):
            raise ValueError("test suite names must be unique")
        return tuple(sorted(suites, key=lambda suite: suite.name))

    @field_validator("test_suites")
    @classmethod
    def order_verifier_test_suites(
        cls, suites: tuple[VerifierTestSuite, ...]
    ) -> tuple[VerifierTestSuite, ...]:
        names = tuple(suite.name for suite in suites)
        if len(names) != len(set(names)):
            raise ValueError("test suite names must be unique")
        return tuple(sorted(suites, key=lambda suite: suite.name))

    @model_validator(mode="after")
    def suites_match_agent_contract(self) -> VerifierManifest:
        verifier_names = {suite.name for suite in self.test_suites}
        agent_names = {suite.name for suite in self.agent_test_suites}
        expected_names = set(self.task.available_test_suites)
        if verifier_names != expected_names or agent_names != expected_names:
            raise ValueError("agent and verifier suites must exactly match agent-visible aliases")
        junit_paths = tuple(suite.junit_path for suite in self.test_suites)
        if len(junit_paths) != len(set(junit_paths)):
            raise ValueError("verifier suites must use distinct JUnit output paths")
        return self

    def agent_view(self) -> TaskSpec:
        """Return the sole policy-visible projection of this trusted object."""

        return self.task

    def agent_json(self) -> str:
        """Serialize only the policy-visible projection."""

        return canonical_json(self.task)

    def digest(self) -> str:
        """Return the canonical content digest used by dataset manifests."""

        return canonical_sha256(self)
