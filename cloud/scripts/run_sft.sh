#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_python
ensure_policy_server_stopped
config="${1:-configs/sft_qwen25_coder_3b_4090.toml}"
cd -- "${REPORL_ROOT}"
require_file "${config}"

"${REPORL_PYTHON}" - "${config}" <<'PY'
import sys
from pathlib import Path

from reporl.training.config import SFTConfig, load_toml_config

config = load_toml_config(Path(sys.argv[1]), SFTConfig, section="sft")
if config.model_revision == "main":
    raise SystemExit("model_revision must be an immutable commit, not main")
for path in (config.train_file, config.eval_file):
    if path is not None and not path.is_file():
        raise SystemExit(f"training data file does not exist: {path}")
if config.resume_from_checkpoint is not None and not config.resume_from_checkpoint.is_dir():
    raise SystemExit(f"resume checkpoint does not exist: {config.resume_from_checkpoint}")
print(f"Validated SFT config for {config.model_id} @ {config.model_revision}")
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
note "Starting foreground SFT job. Run this script inside tmux for disconnect tolerance."
exec "${REPORL_PYTHON}" -m reporl.training.sft --config "${config}"
