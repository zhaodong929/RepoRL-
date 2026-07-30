#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_python
ensure_policy_server_stopped
config="${1:-configs/grpo_qwen25_coder_3b_4090.toml}"
cd -- "${REPORL_ROOT}"
require_file "${config}"

"${REPORL_PYTHON}" - "${config}" <<'PY'
import json
import sys
from pathlib import Path

from reporl.agent.hf_policy import directory_sha256
from reporl.agent.models import PolicyIdentity
from reporl.training.config import GRPOConfig, load_toml_config
from reporl.training.records import read_grpo_groups_jsonl

config = load_toml_config(Path(sys.argv[1]), GRPOConfig, section="grpo")
if config.model_revision == "main":
    raise SystemExit("model_revision must be an immutable commit, not main")
if not config.initial_adapter.is_dir():
    raise SystemExit(f"initial adapter does not exist: {config.initial_adapter}")
if not config.rollout_groups_file.is_file():
    raise SystemExit(f"rollout group file does not exist: {config.rollout_groups_file}")
if config.expected_policy_revision == "sha256:" + "0" * 64:
    raise SystemExit("materialize expected_policy_revision from the rollout manifest first")
manifest_path = config.rollout_groups_file.parent / "run-manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"rollout manifest does not exist: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
identity_payload = manifest.get("policy_identity")
if not isinstance(identity_payload, dict):
    raise SystemExit("rollout manifest does not contain a full PolicyIdentity")
identity = PolicyIdentity.model_validate(identity_payload)
if identity.digest != config.expected_policy_revision:
    raise SystemExit("GRPO expected revision differs from the rollout PolicyIdentity")
adapter_sha256 = "sha256:" + directory_sha256(config.initial_adapter)
if identity.adapter_sha256 != adapter_sha256:
    raise SystemExit("GRPO initial adapter differs from the behavior-policy adapter")
groups = read_grpo_groups_jsonl(config.rollout_groups_file)
if any(
    episode.policy_revision != identity.digest
    for group in groups
    for episode in group.episodes
):
    raise SystemExit("a GRPO episode has the wrong behavior-policy digest")
if any(
    episode.policy_id != identity.model_id
    or episode.policy_adapter_sha256 != identity.adapter_sha256
    for group in groups
    for episode in group.episodes
):
    raise SystemExit("a GRPO episode has the wrong model or adapter identity")
print(f"Validated GRPO config for behavior policy {config.expected_policy_revision}")
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
note "Starting foreground GRPO job. The input groups must be from one on-policy iteration."
exec "${REPORL_PYTHON}" -m reporl.training.grpo --config "${config}"
