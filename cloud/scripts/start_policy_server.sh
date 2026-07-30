#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_python
token="${REPORL_POLICY_SERVER_TOKEN:-}"
[[ ${#token} -ge 32 ]] || die "REPORL_POLICY_SERVER_TOKEN must contain at least 32 characters"

rollout_config="${1:-}"
if [[ -n "${rollout_config}" ]]; then
  cd -- "${REPORL_ROOT}"
  require_file "${rollout_config}"
  policy_output="$("${REPORL_PYTHON}" - "${rollout_config}" <<'PY'
import sys
from pathlib import Path

from reporl.rollouts.collector import load_collection_config

config = load_collection_config(Path(sys.argv[1]))
policy = config.policy
if policy.backend != "remote_trace":
    raise SystemExit("policy server launch requires a remote_trace rollout config")
adapter_path = policy.adapter_path.as_posix() if policy.adapter_path is not None else ""
values = (
    policy.model_id,
    policy.model_revision,
    adapter_path,
    str(policy.max_input_tokens),
    str(policy.max_new_tokens),
    str(policy.temperature),
    str(policy.top_p),
    "true" if policy.load_in_4bit else "false",
)
if any("\n" in value or "\r" in value for value in values):
    raise SystemExit("policy config values must be single-line strings")
print("\n".join(values))
PY
)"
  mapfile -t policy_values <<<"${policy_output}"
  (( ${#policy_values[@]} == 8 )) || die "could not parse rollout policy config"
  model_id="${policy_values[0]}"
  model_revision="${policy_values[1]}"
  adapter_path="${policy_values[2]}"
  max_input_tokens="${policy_values[3]}"
  max_new_tokens="${policy_values[4]}"
  temperature="${policy_values[5]}"
  top_p="${policy_values[6]}"
  load_in_4bit="${policy_values[7]}"
else
  model_id="${REPORL_MODEL_ID:-Qwen/Qwen2.5-Coder-3B-Instruct}"
  model_revision="${REPORL_MODEL_REVISION:-488639f1ff808d1d3d0ba301aef8c11461451ec5}"
  adapter_path="${REPORL_ADAPTER_PATH:-}"
  max_input_tokens="${REPORL_MAX_INPUT_TOKENS:-4096}"
  max_new_tokens="${REPORL_MAX_NEW_TOKENS:-256}"
  temperature="${REPORL_TEMPERATURE:-0.7}"
  top_p="${REPORL_TOP_P:-0.95}"
  load_in_4bit="true"
fi
port="${REPORL_POLICY_PORT:-8010}"
startup_timeout="${REPORL_POLICY_STARTUP_TIMEOUT:-600}"

validate_port "${port}"
[[ "${startup_timeout}" =~ ^[0-9]+$ ]] || die "REPORL_POLICY_STARTUP_TIMEOUT must be numeric"
((startup_timeout >= 30 && startup_timeout <= 1800)) || \
  die "REPORL_POLICY_STARTUP_TIMEOUT must be between 30 and 1800 seconds"
[[ "${model_revision}" != "main" ]] || die "pin REPORL_MODEL_REVISION to an immutable commit"
[[ "${max_input_tokens}" =~ ^[0-9]+$ ]] || die "REPORL_MAX_INPUT_TOKENS must be numeric"
[[ "${max_new_tokens}" =~ ^[0-9]+$ ]] || die "REPORL_MAX_NEW_TOKENS must be numeric"

cd -- "${REPORL_ROOT}"
if [[ -n "${adapter_path}" ]]; then
  [[ -d "${adapter_path}" ]] || die "adapter directory not found: ${adapter_path}"
fi

mkdir -p "${REPORL_CLOUD_STATE_DIR}"
umask 077
pid_file="$(policy_pid_file)"
log_file="${REPORL_CLOUD_STATE_DIR}/policy-server.log"
if [[ -f "${pid_file}" ]]; then
  old_pid="$(<"${pid_file}")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    die "policy server is already running as PID ${old_pid}"
  fi
  rm -f -- "${pid_file}"
fi

args=(
  -m reporl.agent.policy_server
  --model "${model_id}"
  --revision "${model_revision}"
  --host 127.0.0.1
  --port "${port}"
  --max-input-tokens "${max_input_tokens}"
  --max-new-tokens "${max_new_tokens}"
  --temperature "${temperature}"
  --top-p "${top_p}"
)
if [[ -n "${adapter_path}" ]]; then
  args+=(--adapter "${adapter_path}")
fi
if [[ "${load_in_4bit}" == "false" ]]; then
  args+=(--no-4bit)
fi

printf '\n=== policy server start %s ===\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${log_file}"
nohup "${REPORL_PYTHON}" "${args[@]}" >>"${log_file}" 2>&1 </dev/null &
pid=$!
printf '%s\n' "${pid}" >"${pid_file}"

health_url="http://127.0.0.1:${port}/health"
for _ in $(seq 1 "${startup_timeout}"); do
  if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f -- "${pid_file}"
    tail -n 50 -- "${log_file}" >&2 || true
    die "policy server exited during startup"
  fi
  if "${REPORL_PYTHON}" -c \
    'import os,sys,urllib.request; request=urllib.request.Request(sys.argv[1], headers={"Authorization": "Bearer " + os.environ["REPORL_POLICY_SERVER_TOKEN"]}); urllib.request.urlopen(request, timeout=2).read()' \
    "${health_url}" >/dev/null 2>&1; then
    note "Policy server is ready at ${health_url} (PID ${pid})."
    note "Policy endpoints require the bearer token and the server listens on loopback only."
    exit 0
  fi
  sleep 1
done

note "Policy server is still running but did not become healthy within ${startup_timeout} seconds."
note "Inspect ${log_file}; stop it explicitly if model loading cannot complete."
exit 1
