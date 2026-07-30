#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

mode="${1:-split}"
case "${mode}" in
  split | single-node) ;;
  *) die "usage: $0 [split|single-node]" ;;
esac

[[ "$(uname -s)" == "Linux" ]] || die "cloud execution requires Linux"
require_command git
require_command nvidia-smi
require_python

gpu_rows="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)"
[[ -n "${gpu_rows}" ]] || die "nvidia-smi did not report a GPU"
first_row="${gpu_rows%%$'\n'*}"
vram_mib="${first_row##*, }"
[[ "${vram_mib}" =~ ^[0-9]+$ ]] || die "could not parse GPU memory from: ${first_row}"
((vram_mib >= 22000)) || die "at least 22000 MiB VRAM is required for the provided 4090 profile"

storage_root="${REPORL_STORAGE_ROOT:-${REPORL_ROOT}}"
[[ -d "${storage_root}" ]] || die "storage root does not exist: ${storage_root}"
available_kib="$(df -Pk "${storage_root}" | awk 'NR == 2 {print $4}')"
[[ "${available_kib}" =~ ^[0-9]+$ ]] || die "could not determine free disk space"
minimum_kib=$((80 * 1024 * 1024))
recommended_kib=$((150 * 1024 * 1024))
((available_kib >= minimum_kib)) || die "GPU storage has less than the 80 GiB hard minimum"
if ((available_kib < recommended_kib)); then
  note "WARNING: less than 150 GiB is free; cache and checkpoint headroom is limited"
fi

"${REPORL_PYTHON}" - <<'PY'
import importlib
import sys

if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit("RepoRL requires Python 3.11 or 3.12")
for name in (
    "accelerate",
    "bitsandbytes",
    "datasets",
    "fastapi",
    "peft",
    "reporl",
    "safetensors",
    "sentencepiece",
    "torch",
    "transformers",
    "uvicorn",
):
    importlib.import_module(name)
import torch
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("the configured training profile requires CUDA bf16 support")
free_bytes, total_bytes = torch.cuda.mem_get_info()
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"CUDA memory: {free_bytes / 2**30:.1f} GiB free / {total_bytes / 2**30:.1f} GiB total")
PY

if [[ "${mode}" == "single-node" ]]; then
  note "Testing the actual Docker daemon; platform documentation alone is not sufficient."
  docker_hardening_smoke
  note "Docker daemon and hardened-container smoke test passed on this GPU host."
fi

note "GPU preflight passed for mode: ${mode}"
