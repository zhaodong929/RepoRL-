#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_python
require_command git
require_command sha256sum
output_dir="${1:-${REPORL_CLOUD_STATE_DIR}/environment}"
mkdir -p -- "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd)"
umask 077

git -C "${REPORL_ROOT}" rev-parse HEAD >"${output_dir}/git-commit.txt"
git -C "${REPORL_ROOT}" status --short >"${output_dir}/git-status.txt"
"${REPORL_PYTHON}" -VV >"${output_dir}/python-version.txt" 2>&1
"${REPORL_PYTHON}" -m pip freeze >"${output_dir}/pip-freeze.txt"
uname -a >"${output_dir}/uname.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q >"${output_dir}/nvidia-smi.txt"
fi
if command -v docker >/dev/null 2>&1; then
  docker version >"${output_dir}/docker-version.txt" 2>&1 || true
  docker info >"${output_dir}/docker-info.txt" 2>&1 || true
fi

(
  cd -- "${REPORL_ROOT}"
  sha256sum -- configs/*.toml >"${output_dir}/config-sha256.txt"
)
note "Captured a secret-free environment record in ${output_dir}"
