#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_python
(( $# >= 1 && $# <= 3 )) || \
  die "usage: $0 ROLLOUT_RUN_DIR [train|validation|test] [OUTPUT.jsonl]"
run_dir="$1"
split="${2:-train}"
output="${3:-artifacts/datasets/sft/${split}.jsonl}"
tasks_file="${REPORL_TASKS_FILE:-artifacts/tasks/train-runtimes.jsonl}"

case "${split}" in
  train | validation | test) ;;
  *) die "split must be train, validation, or test" ;;
esac

cd -- "${REPORL_ROOT}"
require_file "${tasks_file}"
[[ -d "${run_dir}/trajectories" ]] || die "trajectory directory not found: ${run_dir}/trajectories"
[[ -d "${run_dir}/verifications" ]] || die "verification directory not found: ${run_dir}/verifications"

exec "${REPORL_PYTHON}" -m reporl.training.prepare_sft \
  --tasks "${tasks_file}" \
  --trajectories "${run_dir}/trajectories" \
  --verifications "${run_dir}/verifications" \
  --output "${output}" \
  --split "${split}"
