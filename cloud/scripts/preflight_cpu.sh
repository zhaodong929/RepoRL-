#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

[[ "$(uname -s)" == "Linux" ]] || die "cloud execution requires Linux"
require_command git
require_python

cpu_count="$(getconf _NPROCESSORS_ONLN)"
memory_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
((cpu_count >= 4)) || die "the CPU worker requires at least 4 logical CPUs"
((memory_kib >= 16 * 1024 * 1024)) || die "the CPU worker requires at least 16 GiB RAM"
if ((cpu_count < 8)); then
  note "WARNING: fewer than 8 logical CPUs will limit parallel sandbox throughput"
fi
if ((memory_kib < 32 * 1024 * 1024)); then
  note "WARNING: less than 32 GiB RAM will limit parallel sandbox throughput"
fi

storage_root="${REPORL_STORAGE_ROOT:-${REPORL_ROOT}}"
[[ -d "${storage_root}" ]] || die "storage root does not exist: ${storage_root}"
available_kib="$(df -Pk "${storage_root}" | awk 'NR == 2 {print $4}')"
[[ "${available_kib}" =~ ^[0-9]+$ ]] || die "could not determine free disk space"
((available_kib >= 100 * 1024 * 1024)) || die "CPU storage has less than the 100 GiB hard minimum"
if ((available_kib < 250 * 1024 * 1024)); then
  note "WARNING: less than 250 GiB is free; Docker image and task cache headroom is limited"
fi

"${REPORL_PYTHON}" - <<'PY'
import importlib
import sys

if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit("RepoRL requires Python 3.11 or 3.12")
for name in ("docker", "reporl", "unidiff"):
    importlib.import_module(name)
PY

note "Testing a network-disabled, read-only, unprivileged container."
docker_hardening_smoke
note "CPU worker preflight passed."
