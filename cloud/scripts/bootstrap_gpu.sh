#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

[[ "$(uname -s)" == "Linux" ]] || die "GPU bootstrap supports Linux hosts only"
base_python="${REPORL_BOOTSTRAP_PYTHON:-python3}"
require_command "${base_python}"

"${base_python}" - <<'PY'
import sys

if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit("RepoRL requires Python 3.11 or 3.12")
PY

if [[ ! -x "${REPORL_PYTHON}" ]]; then
  note "Creating ${REPORL_VENV_DIR} with access to the provider CUDA PyTorch image"
  "${base_python}" -m venv --system-site-packages "${REPORL_VENV_DIR}"
fi

"${REPORL_PYTHON}" -m pip install --upgrade pip setuptools wheel
"${REPORL_PYTHON}" - <<'PY'
import re

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit(
        "The provider image must supply CUDA PyTorch before RepoRL dependencies are installed."
    ) from error

match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
version = tuple(map(int, match.groups())) if match is not None else (0, 0)
if not ((2, 6) <= version < (3, 0)):
    raise SystemExit("The provider CUDA image must supply PyTorch >=2.6,<3.")
if not torch.cuda.is_available():
    raise SystemExit("The provider PyTorch build cannot access CUDA.")
PY
mkdir -p "${REPORL_CLOUD_STATE_DIR}"
constraints="${REPORL_CLOUD_STATE_DIR}/constraints-gpu.txt"
"${REPORL_PYTHON}" - "${REPORL_ROOT}/uv.lock" "${constraints}" <<'PY'
import sys
import tomllib
from pathlib import Path

wanted = {
    "accelerate",
    "bitsandbytes",
    "datasets",
    "fastapi",
    "peft",
    "pydantic",
    "safetensors",
    "sentencepiece",
    "transformers",
    "uvicorn",
}
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
"${REPORL_PYTHON}" -m pip install -c "${constraints}" -e "${REPORL_ROOT}[training,server]"

"${REPORL_PYTHON}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA PyTorch is not usable. Select a provider image with a driver-compatible "
        "PyTorch build; this script intentionally does not guess a CUDA wheel."
    )
print(f"PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}")
PY

"${REPORL_PYTHON}" -m pip freeze >"${REPORL_CLOUD_STATE_DIR}/pip-freeze-gpu.txt"
note "GPU dependencies installed. Run preflight_gpu.sh next."
