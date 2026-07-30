#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_command ssh
gpu_host="${REPORL_GPU_HOST:-}"
gpu_user="${REPORL_GPU_SSH_USER:-root}"
ssh_port="${REPORL_GPU_SSH_PORT:-22}"
local_port="${REPORL_TUNNEL_LOCAL_PORT:-8010}"
remote_port="${REPORL_TUNNEL_REMOTE_PORT:-8010}"

[[ "${gpu_host}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
  die "REPORL_GPU_HOST must be a DNS name or IPv4 address"
[[ "${gpu_user}" =~ ^[A-Za-z_][A-Za-z0-9._-]*$ ]] || die "invalid REPORL_GPU_SSH_USER"
validate_port "${ssh_port}"
validate_port "${local_port}"
validate_port "${remote_port}"

note "Opening 127.0.0.1:${local_port} -> GPU 127.0.0.1:${remote_port}."
note "Keep this foreground process running; stop the tunnel with Ctrl-C."
exec ssh \
  -N \
  -T \
  -p "${ssh_port}" \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L "127.0.0.1:${local_port}:127.0.0.1:${remote_port}" \
  "${gpu_user}@${gpu_host}"
