#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

pid_file="$(policy_pid_file)"
[[ -f "${pid_file}" ]] || die "policy server PID file does not exist: ${pid_file}"
pid="$(<"${pid_file}")"
[[ "${pid}" =~ ^[0-9]+$ ]] || die "policy server PID file is invalid"

if ! kill -0 "${pid}" 2>/dev/null; then
  rm -f -- "${pid_file}"
  note "Removed a stale policy server PID file."
  exit 0
fi

proc_cmdline="/proc/${pid}/cmdline"
[[ -r "${proc_cmdline}" ]] || die "cannot verify process identity for PID ${pid}"
cmdline="$(tr '\0' ' ' <"${proc_cmdline}")"
[[ "${cmdline}" == *"reporl.agent.policy_server"* ]] || \
  die "PID ${pid} is not a RepoRL policy server; refusing to signal it"

kill -TERM "${pid}"
for _ in $(seq 1 30); do
  if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f -- "${pid_file}"
    note "Policy server stopped."
    exit 0
  fi
  sleep 1
done

die "policy server did not stop after SIGTERM; inspect PID ${pid} before taking further action"
