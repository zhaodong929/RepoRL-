#!/usr/bin/env bash
set -euo pipefail

CLOUD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPORL_ROOT="$(cd -- "${CLOUD_DIR}/.." && pwd)"
REPORL_VENV_DIR="${REPORL_VENV_DIR:-${REPORL_ROOT}/.venv-cloud}"
REPORL_PYTHON="${REPORL_PYTHON:-${REPORL_VENV_DIR}/bin/python}"
REPORL_CLOUD_STATE_DIR="${REPORL_CLOUD_STATE_DIR:-${REPORL_ROOT}/.reporl/cloud}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

note() {
  printf '%s\n' "$*"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command is unavailable: $1"
}

require_python() {
  [[ -x "${REPORL_PYTHON}" ]] || die "Python environment not found: ${REPORL_PYTHON}"
}

require_file() {
  [[ -f "$1" ]] || die "required file not found: $1"
}

validate_port() {
  local port="$1"
  [[ "${port}" =~ ^[0-9]+$ ]] || die "port must be numeric: ${port}"
  ((port >= 1 && port <= 65535)) || die "port is outside 1-65535: ${port}"
}

policy_pid_file() {
  printf '%s/policy-server.pid\n' "${REPORL_CLOUD_STATE_DIR}"
}

ensure_policy_server_stopped() {
  local pid_file
  pid_file="$(policy_pid_file)"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(<"${pid_file}")"
    if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
      die "policy server PID ${pid} is active; stop it before starting a training job"
    fi
  fi
}

docker_hardening_smoke() {
  local image="${REPORL_PREFLIGHT_IMAGE:-alpine:3.20}"
  require_command docker
  docker info >/dev/null 2>&1 || die "Docker daemon is not reachable by the current user"
  docker run --rm \
    --network none \
    --read-only \
    --user 65534:65534 \
    --cap-drop ALL \
    --security-opt no-new-privileges:true \
    --pids-limit 64 \
    --memory 128m \
    --tmpfs /tmp:rw,noexec,nosuid,size=16m \
    "${image}" true >/dev/null
}
