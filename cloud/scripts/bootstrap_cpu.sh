#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

[[ "$(uname -s)" == "Linux" ]] || die "CPU bootstrap supports Linux hosts only"
base_python="${REPORL_BOOTSTRAP_PYTHON:-python3}"
require_command "${base_python}"

"${base_python}" - <<'PY'
import sys

if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit("RepoRL requires Python 3.11 or 3.12")
PY

if [[ ! -x "${REPORL_PYTHON}" ]]; then
  note "Creating ${REPORL_VENV_DIR}"
  "${base_python}" -m venv "${REPORL_VENV_DIR}"
fi

"${REPORL_PYTHON}" -m pip install --upgrade pip setuptools wheel
mkdir -p "${REPORL_CLOUD_STATE_DIR}"
constraints="${REPORL_CLOUD_STATE_DIR}/constraints-cpu.txt"
"${REPORL_PYTHON}" - "${REPORL_ROOT}/uv.lock" "${constraints}" <<'PY'
import sys
import tomllib
from pathlib import Path

wanted = {"docker", "pydantic", "unidiff"}
payload = tomllib.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
versions = {
    package["name"]: package["version"]
    for package in payload["package"]
    if package.get("name") in wanted and "version" in package
}
missing = sorted(wanted - versions.keys())
if missing:
    raise SystemExit("uv.lock is missing cloud dependencies: " + ", ".join(missing))
Path(sys.argv[2]).write_text(
    "".join(f"{name}=={versions[name]}\n" for name in sorted(wanted)),
    encoding="utf-8",
)
PY
"${REPORL_PYTHON}" -m pip install -c "${constraints}" -e "${REPORL_ROOT}[sandbox]"

"${REPORL_PYTHON}" -m pip freeze >"${REPORL_CLOUD_STATE_DIR}/pip-freeze-cpu.txt"
note "CPU dependencies installed. Docker Engine must be installed separately."
note "Run preflight_cpu.sh next."
