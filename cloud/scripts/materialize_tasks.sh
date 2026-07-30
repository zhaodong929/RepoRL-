#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_python
(( $# <= 3 )) || die "usage: $0 [MANIFESTS.jsonl] [ARTIFACT_ROOT] [OUTPUT_DIR]"
manifests="${1:-artifacts/sealed/verifier-manifests.jsonl}"
artifact_root="${2:-artifacts/sealed}"
output_dir="${3:-artifacts/tasks}"
baseline_fraction="${REPORL_BASELINE_TARGET_PASS_FRACTION:-0}"
dataset_manifest="${REPORL_DATASET_MANIFEST:-artifacts/sealed/dataset-manifest.json}"
split_seal="${REPORL_SPLIT_SEAL:-artifacts/sealed/split-seal.json}"
repositories="${REPORL_REPOSITORIES:-artifacts/sealed/repositories.jsonl}"
admission_evidence="${REPORL_ADMISSION_EVIDENCE:-artifacts/sealed/admission-evidence.jsonl}"
admission_results="${REPORL_ADMISSION_RESULTS:-artifacts/sealed/admission-results.jsonl}"

cd -- "${REPORL_ROOT}"
require_file "${manifests}"
require_file "${dataset_manifest}"
require_file "${split_seal}"
require_file "${repositories}"
require_file "${admission_evidence}"
require_file "${admission_results}"
[[ -d "${artifact_root}" ]] || die "sealed artifact root not found: ${artifact_root}"

args=(
  -m reporl.tasks.materialize materialize
  --manifests "${manifests}"
  --artifact-root "${artifact_root}"
  --dataset-manifest "${dataset_manifest}"
  --split-seal "${split_seal}"
  --repositories "${repositories}"
  --admission-evidence "${admission_evidence}"
  --admission-results "${admission_results}"
  --output-dir "${output_dir}"
  --baseline-target-pass-fraction "${baseline_fraction}"
)
if [[ -n "${REPORL_AGENT_IMAGE_REPOSITORY:-}" ]]; then
  args+=(--agent-image-repository "${REPORL_AGENT_IMAGE_REPOSITORY}")
fi
if [[ -n "${REPORL_VERIFIER_IMAGE_REPOSITORY:-}" ]]; then
  args+=(--verifier-image-repository "${REPORL_VERIFIER_IMAGE_REPOSITORY}")
fi

"${REPORL_PYTHON}" "${args[@]}"
"${REPORL_PYTHON}" -m reporl.tasks.materialize verify \
  --runtime-dir "${output_dir}" \
  --artifact-root "${artifact_root}"
note "Materialized and revalidated all task splits under ${output_dir}"
