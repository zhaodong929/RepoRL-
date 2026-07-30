#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_python
token="${REPORL_POLICY_SERVER_TOKEN:-}"
[[ ${#token} -ge 32 ]] || die "REPORL_POLICY_SERVER_TOKEN must contain at least 32 characters"
export REPORL_POLICY_URL="${REPORL_POLICY_URL:-http://127.0.0.1:8010}"

"${REPORL_PYTHON}" - <<'PY'
import hashlib
import json
import os
import urllib.request

base_url = os.environ["REPORL_POLICY_URL"].rstrip("/")
token = os.environ["REPORL_POLICY_SERVER_TOKEN"]
health_request = urllib.request.Request(
    base_url + "/health",
    method="GET",
    headers={"Authorization": f"Bearer {token}"},
)
with urllib.request.urlopen(health_request, timeout=10) as response:
    info = json.load(response)

identity = info.get("policy_identity")
if not isinstance(identity, dict):
    raise SystemExit("policy health response has no full PolicyIdentity")
identity_payload = json.dumps(
    identity,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")
identity_digest = "sha256:" + hashlib.sha256(identity_payload).hexdigest()
if info.get("policy_revision") != identity_digest:
    raise SystemExit("health policy revision does not match its PolicyIdentity digest")

payload = json.dumps(
    {
        "messages": [
            {
                "role": "system",
                "content": "Return one concise coding-agent action and no prose.",
            },
            {
                "role": "user",
                "content": '{"action":"finish","summary":"connectivity check"}',
            },
        ],
        "seed": 42,
    }
).encode("utf-8")
request = urllib.request.Request(
    base_url + "/action",
    data=payload,
    method="POST",
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    },
)
with urllib.request.urlopen(request, timeout=180) as response:
    envelope = json.load(response)

if envelope.get("policy_id") != info.get("policy_id"):
    raise SystemExit("policy identity changed between health and action responses")
if envelope.get("policy_revision") != info.get("policy_revision"):
    raise SystemExit("policy revision changed between health and action responses")
if envelope.get("adapter_sha256") != identity.get("adapter_sha256"):
    raise SystemExit("adapter digest changed between health and action responses")
step = envelope.get("step") or {}
usage = step.get("token_usage") or {}
trace = step.get("generation_trace") or {}
prompt_ids = trace.get("prompt_input_ids") or []
generated_ids = trace.get("generated_token_ids") or []
old_logprobs = trace.get("old_logprobs") or []
if not generated_ids:
    raise SystemExit("policy response has no generated-token trace")
if len(generated_ids) != len(old_logprobs):
    raise SystemExit("generated token and behavior-logprob lengths differ")
if usage.get("input_tokens") != len(prompt_ids):
    raise SystemExit("prompt trace length does not match token usage")
if usage.get("output_tokens") != len(generated_ids):
    raise SystemExit("generated trace length does not match token usage")
print(
    "Policy smoke test passed: "
    f"{info['policy_id']} @ {info['policy_revision']}; "
    f"{len(prompt_ids)} input and {len(generated_ids)} output tokens."
)
PY
