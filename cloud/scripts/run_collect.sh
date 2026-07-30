#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_python
token="${REPORL_POLICY_SERVER_TOKEN:-}"
[[ ${#token} -ge 32 ]] || die "REPORL_POLICY_SERVER_TOKEN must contain at least 32 characters"
config="${1:-configs/rollout_remote_canary_4090.toml}"
collection_timeout_seconds="${REPORL_COLLECTION_TIMEOUT_SECONDS:-86400}"
[[ "${collection_timeout_seconds}" =~ ^[0-9]+$ ]] || \
  die "REPORL_COLLECTION_TIMEOUT_SECONDS must be an integer"
((collection_timeout_seconds >= 300 && collection_timeout_seconds <= 604800)) || \
  die "REPORL_COLLECTION_TIMEOUT_SECONDS must be between 300 and 604800"
require_command timeout
timeout_version="$(timeout --version 2>/dev/null || true)"
[[ "${timeout_version}" == *"GNU coreutils"* ]] || \
  die "run_collect.sh requires GNU coreutils timeout"
cd -- "${REPORL_ROOT}"
require_file "${config}"

"${REPORL_PYTHON}" - "${config}" <<'PY'
import sys
from pathlib import Path

from reporl.agent.remote_policy import fetch_policy_server_info
from reporl.rollouts.collector import load_collection_config
from reporl.tasks.materialize import verify_runtime_splits

config = load_collection_config(Path(sys.argv[1]))
if not config.tasks_file.is_file():
    raise SystemExit(f"materialized task file does not exist: {config.tasks_file}")
if not config.task_artifacts_root.is_dir():
    raise SystemExit(f"task artifact root does not exist: {config.task_artifacts_root}")
zero = "sha256:" + "0" * 64
bindings = (
    config.expected_dataset_manifest_sha256,
    config.expected_split_seal_sha256,
    config.expected_split_assignment_sha256,
    config.expected_split_membership_sha256,
    config.expected_repository_records_sha256,
    config.expected_tasks_file_sha256,
)
if zero in bindings:
    raise SystemExit("bind the rollout template to materialization metadata before collection")
metadata_path = config.tasks_file.parent / "materialization-metadata.json"
if not metadata_path.is_file():
    raise SystemExit(f"materialization metadata does not exist: {metadata_path}")
_, metadata = verify_runtime_splits(
    metadata_path.parent,
    config.task_artifacts_root,
)
runtime_seals = {seal.split: seal for seal in metadata.runtime_files}
try:
    runtime_seal = runtime_seals[config.expected_split]
except KeyError as error:
    raise SystemExit("materialization metadata does not seal the configured split") from error
if config.tasks_file.resolve() != (metadata_path.parent / runtime_seal.path).resolve():
    raise SystemExit("rollout tasks_file does not match the sealed split runtime")
if bindings != (
    metadata.dataset_manifest_sha256,
    metadata.split_seal_sha256,
    metadata.split_assignment_sha256,
    runtime_seal.split_membership_sha256,
    metadata.repository_records_sha256,
    runtime_seal.sha256,
):
    raise SystemExit("rollout config bindings differ from materialization metadata")
if config.policy.backend != "remote_trace":
    raise SystemExit("cloud split collection requires the remote_trace policy backend")
assert config.policy.base_url is not None
info = fetch_policy_server_info(
    config.policy.base_url,
    bearer_token=config.policy.secret(),
    timeout_seconds=10,
)
if info.policy_id != config.policy.model_id:
    raise SystemExit("policy server model ID does not match rollout config")
if info.policy_identity is None:
    raise SystemExit("policy server health response does not contain a full PolicyIdentity")
config.policy.validate_server_identity(info.policy_identity)
expects_adapter = config.policy.adapter_path is not None
has_adapter = info.policy_identity.adapter_sha256 is not None
if expects_adapter != has_adapter:
    raise SystemExit("policy server adapter presence does not match rollout config")
if info.policy_revision != info.policy_identity.digest:
    raise SystemExit("policy server revision is not the PolicyIdentity digest")
run_dir = config.artifacts_root / config.run_id
if run_dir.exists() and any(run_dir.iterdir()):
    raise SystemExit(f"run directory is not empty; choose a new run_id: {run_dir}")
print(
    f"Validated {config.group_size} candidates for "
    f"{config.maximum_tasks or 'all'} task(s) against {info.policy_revision}"
)
PY

note "Starting foreground rollout collection on the Docker worker."
note "Collector process deadline: ${collection_timeout_seconds}s (SIGTERM, then SIGKILL after 30s)."
exec timeout \
  --signal=TERM \
  --kill-after=30s \
  "${collection_timeout_seconds}s" \
  "${REPORL_PYTHON}" -m reporl.rollouts.collector --config "${config}"
