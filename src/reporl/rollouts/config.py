"""Strict materialized task and rollout-job configurations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from reporl.agent.models import PolicyIdentity
from reporl.agent.runner import RunnerConfig
from reporl.rewards import RewardConfig
from reporl.sandbox.base import CommandSpec
from reporl.sandbox.docker import DockerSandboxConfig
from reporl.schemas import DatasetSplit, StrictModel, TaskSpec
from reporl.tasks.canonical import artifact_sha256
from reporl.tasks.manifest import TrustedCommand
from reporl.tasks.manifest import VerifierManifest as SealedVerifierManifest
from reporl.verifier.models import VerifierRunSpec


class AgentRunSpec(StrictModel):
    repository_snapshot: Path
    image: str = Field(min_length=1)
    suite_commands: dict[str, CommandSpec]
    user: str = "1000:1000"
    memory_limit: str = "12g"
    nano_cpus: int = Field(default=4_000_000_000, ge=100_000_000)
    pids_limit: int = Field(default=512, ge=16, le=4_096)
    workspace_size: str = "8g"
    default_timeout_seconds: int = Field(default=60, ge=1, le=3_600)

    def docker_config(self) -> DockerSandboxConfig:
        return DockerSandboxConfig(
            image=self.image,
            suite_commands=self.suite_commands,
            user=self.user,
            memory_limit=self.memory_limit,
            nano_cpus=self.nano_cpus,
            pids_limit=self.pids_limit,
            workspace_size=self.workspace_size,
            default_timeout_seconds=self.default_timeout_seconds,
        )


class RolloutTaskSpec(StrictModel):
    """Materialized paths plus the sealed-manifest digest used by one rollout worker."""

    sealed_manifest_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    split_seal_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    split_assignment_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    split_membership_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    repository_records_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    sealed_manifest: SealedVerifierManifest
    artifact_root: Path
    task: TaskSpec
    agent: AgentRunSpec
    verifier: VerifierRunSpec
    baseline_target_pass_fraction: float = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def boundaries_match(self) -> RolloutTaskSpec:
        if self.sealed_manifest.digest() != self.sealed_manifest_sha256:
            raise ValueError("sealed manifest digest does not match its canonical content")
        if self.sealed_manifest.task != self.task:
            raise ValueError("runtime TaskSpec must equal the sealed agent view")
        if self.verifier.task_id != self.task.task_id:
            raise ValueError("agent and verifier task IDs must match")
        if not (
            self.agent.image == self.task.agent_image_digest
            or self.agent.image.endswith(f"@{self.task.agent_image_digest}")
        ):
            raise ValueError("agent image must be pinned to the TaskSpec digest")
        if set(self.agent.suite_commands) != set(self.task.available_test_suites):
            raise ValueError("agent suite commands must match TaskSpec aliases")
        sealed_agent_suites = {
            suite.name: _command_from_trusted(suite.command)
            for suite in self.sealed_manifest.agent_test_suites
        }
        if self.agent.suite_commands != sealed_agent_suites:
            raise ValueError("runtime agent suite commands differ from the sealed manifest")
        verifier_suites = {suite.name for suite in self.verifier.suites}
        if not {"target", "regression"}.issubset(verifier_suites):
            raise ValueError("research rollouts require target and regression verifier suites")
        if self.verifier.allowed_paths != self.task.allowed_paths:
            raise ValueError("agent and verifier allowed paths must match")
        if self.verifier.forbidden_globs != self.task.forbidden_globs:
            raise ValueError("agent and verifier forbidden globs must match")
        if self.verifier.max_patch_bytes != self.task.budgets.max_patch_bytes:
            raise ValueError("agent and verifier patch byte limits must match")
        if self.verifier.image_digest != self.sealed_manifest.verifier_image_digest:
            raise ValueError("verifier image digest must match the sealed manifest")
        sealed_suites = {suite.name: suite for suite in self.sealed_manifest.test_suites}
        runtime_suites = {suite.name: suite for suite in self.verifier.suites}
        if set(sealed_suites) != set(runtime_suites):
            raise ValueError("runtime verifier suites must match the sealed manifest")
        for name, sealed_suite in sealed_suites.items():
            runtime_suite = runtime_suites[name]
            expected_workdir = (
                "/workspace/repo"
                if sealed_suite.command.cwd == "."
                else f"/workspace/repo/{sealed_suite.command.cwd}"
            )
            expected_environment = {
                variable.name: variable.value for variable in sealed_suite.command.environment
            }
            if (
                runtime_suite.command.argv != sealed_suite.command.argv
                or runtime_suite.command.timeout_seconds != sealed_suite.command.timeout_seconds
                or dict(runtime_suite.command.environment) != expected_environment
                or (runtime_suite.command.workdir or "/workspace/repo") != expected_workdir
                or runtime_suite.junit_path != sealed_suite.junit_path
                or runtime_suite.expected_test_ids != sealed_suite.expected_test_ids
            ):
                raise ValueError(f"runtime verifier suite {name!r} differs from its sealed suite")
        if not self.artifact_root.is_absolute():
            expected_snapshot = Path(self.sealed_manifest.buggy_snapshot.path)
            expected_hidden = Path(self.sealed_manifest.hidden_tests.path)
            if self.artifact_root != Path("."):
                raise ValueError("portable runtime artifact_root must be '.'")
            if (
                self.agent.repository_snapshot != expected_snapshot
                or self.verifier.repository_snapshot != expected_snapshot
                or self.verifier.hidden_tests_path != expected_hidden
            ):
                raise ValueError("portable runtime paths must be relative sealed artifact paths")
        return self

    def validate_materialized_paths(self, artifact_root_override: Path | None = None) -> None:
        artifact_root = (artifact_root_override or self.artifact_root).resolve(strict=True)
        references = (
            self.sealed_manifest.clean_snapshot,
            self.sealed_manifest.buggy_snapshot,
            self.sealed_manifest.reference_snapshot,
            self.sealed_manifest.hidden_tests,
            self.sealed_manifest.reference_patch,
        )
        resolved_artifacts: dict[str, Path] = {}
        for reference in references:
            artifact = (artifact_root / reference.path).resolve(strict=True)
            try:
                artifact.relative_to(artifact_root)
            except ValueError as error:
                raise ValueError("sealed artifact escapes artifact_root") from error
            digest, size_bytes = artifact_sha256(artifact)
            if digest != reference.sha256 or size_bytes != reference.size_bytes:
                raise ValueError(f"sealed artifact does not match its digest: {reference.path}")
            resolved_artifacts[reference.path] = artifact

        buggy_snapshot = resolved_artifacts[self.sealed_manifest.buggy_snapshot.path]
        hidden_tests = resolved_artifacts[self.sealed_manifest.hidden_tests.path]
        agent_snapshot = _resolve_runtime_path(artifact_root, self.agent.repository_snapshot)
        verifier_snapshot = _resolve_runtime_path(artifact_root, self.verifier.repository_snapshot)
        if not agent_snapshot.is_dir() or not verifier_snapshot.is_dir():
            raise ValueError("repository snapshots must be directories")
        if agent_snapshot != buggy_snapshot or verifier_snapshot != buggy_snapshot:
            raise ValueError("agent and verifier must start from the sealed buggy snapshot")
        if self.verifier.hidden_tests_path is None:
            raise ValueError("research rollouts require sealed hidden tests")
        hidden = _resolve_runtime_path(artifact_root, self.verifier.hidden_tests_path)
        if hidden != hidden_tests or not hidden.is_dir():
            raise ValueError("verifier hidden tests must equal the sealed hidden-test directory")
        try:
            hidden.relative_to(agent_snapshot)
        except ValueError:
            pass
        else:
            raise ValueError("hidden tests must not be inside the agent snapshot")

    def rebase_artifact_root(self, artifact_root: Path) -> RolloutTaskSpec:
        """Resolve portable artifact-relative paths on the current rollout worker."""

        root = artifact_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("task_artifacts_root must be a directory")
        if self.artifact_root.is_absolute():
            raise ValueError("runtime JSONL must use portable artifact-relative paths")
        assert self.verifier.hidden_tests_path is not None
        return self.model_copy(
            update={
                "artifact_root": root,
                "agent": self.agent.model_copy(
                    update={
                        "repository_snapshot": root / self.agent.repository_snapshot,
                    }
                ),
                "verifier": self.verifier.model_copy(
                    update={
                        "repository_snapshot": root / self.verifier.repository_snapshot,
                        "hidden_tests_path": root / self.verifier.hidden_tests_path,
                    }
                ),
            }
        )


class RolloutPolicyConfig(StrictModel):
    backend: Literal["transformers", "remote_trace", "openai"] = "transformers"
    model_id: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    model_revision: str = "main"
    expected_policy_revision: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    expected_adapter_sha256: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    adapter_path: Path | None = None
    base_url: str | None = None
    token_environment_variable: str = "REPORL_POLICY_SERVER_TOKEN"
    load_in_4bit: bool = True
    max_input_tokens: int = Field(default=16_384, ge=512)
    max_new_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.7, ge=0)
    top_p: float = Field(default=1, gt=0, le=1)
    timeout_seconds: float = Field(default=180, gt=0)

    @model_validator(mode="after")
    def backend_fields_exist(self) -> RolloutPolicyConfig:
        if self.backend in {"remote_trace", "openai"} and self.base_url is None:
            raise ValueError("remote policy backends require base_url")
        return self

    def validate_server_identity(self, identity: PolicyIdentity) -> None:
        expected_quantization = "bnb-nf4-double" if self.load_in_4bit else "none"
        expected_preparation = (
            "peft-kbit-training-v1"
            if self.load_in_4bit and self.adapter_path is not None
            else "inference-only"
        )
        checks = {
            "model_id": (identity.model_id, self.model_id),
            "model_revision": (identity.model_revision, self.model_revision),
            "max_input_tokens": (identity.max_input_tokens, self.max_input_tokens),
            "max_new_tokens": (identity.max_new_tokens, self.max_new_tokens),
            "sampling_temperature": (identity.sampling_temperature, self.temperature),
            "sampling_top_p": (identity.sampling_top_p, self.top_p),
            "quantization": (identity.quantization, expected_quantization),
            "model_preparation": (identity.model_preparation, expected_preparation),
        }
        mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
        if mismatches:
            raise ValueError(
                "trace policy server differs from rollout config: " + ", ".join(mismatches)
            )
        if (
            self.expected_policy_revision is not None
            and identity.digest != self.expected_policy_revision
        ):
            raise ValueError(
                "trace policy server fingerprint differs from expected_policy_revision"
            )
        if (
            self.expected_adapter_sha256 is not None
            and identity.adapter_sha256 != self.expected_adapter_sha256
        ):
            raise ValueError("trace policy server adapter differs from expected_adapter_sha256")

    def secret(self) -> str:
        value = os.environ.get(self.token_environment_variable, "")
        if self.backend in {"remote_trace", "openai"} and not value:
            raise ValueError(f"environment variable {self.token_environment_variable} is required")
        return value


class RolloutCollectionConfig(StrictModel):
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
    method: str = Field(min_length=1)
    tasks_file: Path
    task_artifacts_root: Path
    artifacts_root: Path
    expected_split: DatasetSplit
    expected_dataset_manifest_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    expected_split_seal_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    expected_split_assignment_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    expected_split_membership_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    expected_repository_records_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    expected_tasks_file_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    group_size: int = Field(default=4, ge=1, le=32)
    seed: int = 42
    maximum_tasks: int | None = Field(default=None, ge=1)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    policy: RolloutPolicyConfig = Field(default_factory=RolloutPolicyConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)

    @model_validator(mode="after")
    def grouped_rollouts_are_stochastic(self) -> RolloutCollectionConfig:
        if self.group_size >= 2 and self.policy.temperature <= 0:
            raise ValueError("grouped rollouts require a positive sampling temperature")
        if (
            self.runner.max_conversation_bytes + self.runner.context_token_reserve
            > self.policy.max_input_tokens
        ):
            raise ValueError(
                "runner context bytes plus token reserve exceed policy max_input_tokens"
            )
        return self

    def validate_task_bindings(self, tasks: tuple[RolloutTaskSpec, ...]) -> None:
        """Reject a task file from the wrong sealed dataset or split before rollout."""

        if not tasks:
            raise ValueError("no rollout tasks were loaded")
        checks = {
            "dataset manifest": (
                "dataset_manifest_sha256",
                self.expected_dataset_manifest_sha256,
            ),
            "split seal": ("split_seal_sha256", self.expected_split_seal_sha256),
            "split assignment": (
                "split_assignment_sha256",
                self.expected_split_assignment_sha256,
            ),
            "repository records": (
                "repository_records_sha256",
                self.expected_repository_records_sha256,
            ),
            "split membership": (
                "split_membership_sha256",
                self.expected_split_membership_sha256,
            ),
        }
        for task in tasks:
            if task.task.split != self.expected_split:
                raise ValueError(
                    f"task {task.task.task_id!r} is from {task.task.split.value}, "
                    f"expected {self.expected_split.value}"
                )
            for label, (attribute, expected) in checks.items():
                if getattr(task, attribute) != expected:
                    raise ValueError(f"task {task.task.task_id!r} has the wrong {label} digest")


def _command_from_trusted(command: TrustedCommand) -> CommandSpec:
    """Convert the sealed, shell-free suite contract into its runtime representation."""
    workdir = "/workspace/repo" if command.cwd == "." else f"/workspace/repo/{command.cwd}"
    return CommandSpec(
        argv=command.argv,
        timeout_seconds=min(command.timeout_seconds, 3_600),
        environment={variable.name: variable.value for variable in command.environment},
        workdir=workdir,
    )


def _resolve_runtime_path(artifact_root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else artifact_root / path
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError("runtime artifact path escapes artifact_root") from error
    return resolved
