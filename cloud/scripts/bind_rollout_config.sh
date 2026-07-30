#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_python
(( $# == 3 )) || die "usage: $0 TEMPLATE.toml MATERIALIZATION_METADATA.json OUTPUT.toml"
template="$1"
metadata="$2"
output="$3"

cd -- "${REPORL_ROOT}"
require_file "${template}"
require_file "${metadata}"
[[ ! -e "${output}" ]] || die "bound rollout config already exists: ${output}"

"${REPORL_PYTHON}" - "${template}" "${metadata}" "${output}" <<'PY'
import sys
from pathlib import Path

from reporl.rollouts.collector import load_collection_config
from reporl.rollouts.config import RolloutTaskSpec
from reporl.tasks.loader import load_jsonl
from reporl.tasks.materialize import verify_runtime_splits

template_path, metadata_path, output_path = map(Path, sys.argv[1:])
config = load_collection_config(template_path)
if not config.task_artifacts_root.is_dir():
    raise SystemExit(f"task artifact root does not exist: {config.task_artifacts_root}")
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
    raise SystemExit("rollout tasks_file does not match the materialized split file")

bindings = {
    "expected_dataset_manifest_sha256": metadata.dataset_manifest_sha256,
    "expected_split_seal_sha256": metadata.split_seal_sha256,
    "expected_split_assignment_sha256": metadata.split_assignment_sha256,
    "expected_split_membership_sha256": runtime_seal.split_membership_sha256,
    "expected_repository_records_sha256": metadata.repository_records_sha256,
    "expected_tasks_file_sha256": runtime_seal.sha256,
}
text = template_path.read_text(encoding="utf-8")
zero = "sha256:" + "0" * 64
for key, digest in bindings.items():
    placeholder = f'{key} = "{zero}"'
    if text.count(placeholder) != 1:
        raise SystemExit(f"template does not contain exactly one placeholder for {key}")
    text = text.replace(placeholder, f'{key} = "{digest}"')
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("x", encoding="utf-8", newline="\n") as handle:
    handle.write(text)
bound = load_collection_config(output_path)
bound.validate_task_bindings(load_jsonl(bound.tasks_file, RolloutTaskSpec))
print(f"Bound {output_path} to materialization {metadata.digest()}")
PY
