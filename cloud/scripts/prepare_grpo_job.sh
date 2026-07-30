#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_python
(( $# >= 1 && $# <= 2 )) || die "usage: $0 ROLLOUT_RUN_DIR [OUTPUT.toml]"
run_dir="$1"
output="${2:-artifacts/job-configs/grpo-iteration-001.toml}"
template="configs/grpo_qwen25_coder_3b_4090.toml"

cd -- "${REPORL_ROOT}"
manifest="${run_dir}/run-manifest.json"
groups="${run_dir}/grpo-groups.jsonl"
require_file "${template}"
require_file "${manifest}"
require_file "${groups}"
[[ ! -e "${output}" ]] || die "output job config already exists: ${output}"

"${REPORL_PYTHON}" - "${template}" "${manifest}" "${groups}" "${output}" <<'PY'
import json
import sys
from pathlib import Path

from reporl.agent.hf_policy import directory_sha256
from reporl.agent.models import PolicyIdentity
from reporl.training.config import GRPOConfig, load_toml_config
from reporl.training.records import read_grpo_groups_jsonl

template_path, manifest_path, groups_path, output_path = map(Path, sys.argv[1:])
config = load_toml_config(template_path, GRPOConfig, section="grpo")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("kind") != "rollout-collection":
    raise SystemExit("input manifest is not a rollout-collection manifest")
identity_payload = manifest.get("policy_identity")
if not isinstance(identity_payload, dict):
    raise SystemExit("rollout manifest does not contain a full PolicyIdentity")
identity = PolicyIdentity.model_validate(identity_payload)
if manifest.get("policy_revision") != identity.digest:
    raise SystemExit("rollout manifest policy revision does not match PolicyIdentity")
if identity.model_id != config.model_id or identity.model_revision != config.model_revision:
    raise SystemExit("rollout policy base model differs from the GRPO template")
if identity.adapter_sha256 is None:
    raise SystemExit("GRPO rollout policy does not identify an SFT adapter")

local_adapter_sha256 = "sha256:" + directory_sha256(config.initial_adapter)
if local_adapter_sha256 != identity.adapter_sha256:
    raise SystemExit("local GRPO initial adapter differs from the rollout behavior adapter")
if groups_path.resolve() != config.rollout_groups_file.resolve():
    raise SystemExit("rollout group path differs from the GRPO template")
groups = read_grpo_groups_jsonl(groups_path)
if not groups:
    raise SystemExit("rollout group file is empty")
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
if manifest.get("trainable_group_count") != len(groups):
    raise SystemExit("rollout manifest group count differs from the group file")

placeholder = 'expected_policy_revision = "sha256:0000000000000000000000000000000000000000000000000000000000000000"'
template_text = template_path.read_text(encoding="utf-8")
if template_text.count(placeholder) != 1:
    raise SystemExit("GRPO template does not contain exactly one revision placeholder")
materialized = template_text.replace(
    placeholder,
    f'expected_policy_revision = "{identity.digest}"',
)
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("x", encoding="utf-8", newline="\n") as handle:
    handle.write(materialized)
print(f"Materialized {output_path} for {identity.digest}")
PY
