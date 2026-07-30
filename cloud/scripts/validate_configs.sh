#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_python
cd -- "${REPORL_ROOT}"

"${REPORL_PYTHON}" - <<'PY'
from pathlib import Path

from reporl.rollouts.collector import load_collection_config
from reporl.training.config import GRPOConfig, SFTConfig, load_toml_config

root = Path("configs")
sft = load_toml_config(
    root / "sft_qwen25_coder_3b_4090.toml",
    SFTConfig,
    section="sft",
)
grpo = load_toml_config(
    root / "grpo_qwen25_coder_3b_4090.toml",
    GRPOConfig,
    section="grpo",
)
rollout_paths = (
    root / "rollout_remote_canary_4090.toml",
    root / "rollout_remote_sft_seed_4090.toml",
    root / "rollout_remote_sft_validation_4090.toml",
    root / "rollout_remote_grpo_iteration_001_4090.toml",
    root / "eval_remote_base_4090.toml",
    root / "eval_remote_sft_4090.toml",
    root / "eval_remote_grpo_iteration_001_4090.toml",
)
rollouts = {path.name: load_collection_config(path) for path in rollout_paths}
zero = "sha256:" + "0" * 64

base_revision = sft.model_revision
if base_revision == "main" or grpo.model_revision != base_revision:
    raise SystemExit("SFT and GRPO must use one immutable base revision")
if sft.model_id != grpo.model_id:
    raise SystemExit("SFT and GRPO model IDs differ")

if grpo.expected_policy_revision != "sha256:" + "0" * 64:
    raise SystemExit("the checked-in GRPO template must require manifest materialization")

base_names = (
    "rollout_remote_canary_4090.toml",
    "rollout_remote_sft_seed_4090.toml",
    "rollout_remote_sft_validation_4090.toml",
    "eval_remote_base_4090.toml",
)
for name in base_names:
    policy = rollouts[name].policy
    if policy.model_id != sft.model_id or policy.model_revision != base_revision:
        raise SystemExit(f"base policy identity drift in {name}")
    if policy.adapter_path is not None:
        raise SystemExit(f"base policy unexpectedly declares an adapter in {name}")

for name in (
    "rollout_remote_grpo_iteration_001_4090.toml",
    "eval_remote_sft_4090.toml",
):
    policy = rollouts[name].policy
    if policy.model_revision != base_revision or policy.adapter_path != grpo.initial_adapter:
        raise SystemExit(f"SFT policy identity drift in {name}")

rl_adapter = Path("outputs/grpo-qwen25-coder-3b-4090-iteration-001/adapter-final/policy")
rl_policy = rollouts["eval_remote_grpo_iteration_001_4090.toml"].policy
if rl_policy.model_revision != base_revision or rl_policy.adapter_path != rl_adapter:
    raise SystemExit("RL evaluation does not identify the iteration-001 adapter")

grpo_rollout = rollouts["rollout_remote_grpo_iteration_001_4090.toml"]
expected_groups = grpo_rollout.artifacts_root / grpo_rollout.run_id / "grpo-groups.jsonl"
if grpo.rollout_groups_file != expected_groups:
    raise SystemExit("GRPO input path does not match the rollout collector output path")

for name, config in rollouts.items():
    policy = config.policy
    if policy.backend != "remote_trace" or policy.base_url != "http://127.0.0.1:8010":
        raise SystemExit(f"split-deployment policy settings drift in {name}")
    if policy.model_revision != base_revision:
        raise SystemExit(f"mutable model revision in {name}")
    if policy.expected_policy_revision is not None or policy.expected_adapter_sha256 is not None:
        raise SystemExit(f"checked-in template contains a machine-specific digest in {name}")
    if config.task_artifacts_root != Path("artifacts/sealed"):
        raise SystemExit(f"task artifact root drift in {name}")
    dataset_bindings = (
        config.expected_dataset_manifest_sha256,
        config.expected_split_seal_sha256,
        config.expected_split_assignment_sha256,
        config.expected_split_membership_sha256,
        config.expected_repository_records_sha256,
        config.expected_tasks_file_sha256,
    )
    if dataset_bindings != (zero, zero, zero, zero, zero, zero):
        raise SystemExit(f"checked-in template contains dataset-specific digests in {name}")

runner_configs = {config.runner.model_dump_json() for config in rollouts.values()}
if len(runner_configs) != 1:
    raise SystemExit("rollout/evaluation templates do not use one fixed RunnerConfig")

expected_splits = {
    "rollout_remote_canary_4090.toml": "train",
    "rollout_remote_sft_seed_4090.toml": "train",
    "rollout_remote_sft_validation_4090.toml": "validation",
    "rollout_remote_grpo_iteration_001_4090.toml": "train",
    "eval_remote_base_4090.toml": "test",
    "eval_remote_sft_4090.toml": "test",
    "eval_remote_grpo_iteration_001_4090.toml": "test",
}
for name, split in expected_splits.items():
    if rollouts[name].expected_split.value != split:
        raise SystemExit(f"dataset split drift in {name}")

for name in (
    "eval_remote_base_4090.toml",
    "eval_remote_sft_4090.toml",
    "eval_remote_grpo_iteration_001_4090.toml",
):
    policy = rollouts[name].policy
    if policy.temperature != 0 or policy.top_p != 1:
        raise SystemExit(f"evaluation decoding is not deterministic in {name}")

print(f"Validated {2 + len(rollouts)} cloud configs and their policy lineage.")
PY
