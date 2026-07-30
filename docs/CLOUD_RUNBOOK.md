# RepoRL Cloud Runbook

This runbook takes RepoRL from a prepared task bundle to one 24 GiB NVIDIA GPU training
run. It is intentionally provider-neutral. An AutoDL image may or may not expose a usable
Docker daemon. RepoRL never assumes that nested Docker is available.

## Current status

As of 2026-07-30, the AutoDL pre-rental implementation is ready, but the paid-cloud workflow
in this document has not been executed:

- CPU-side contracts, deterministic tests, strict configs, and orchestration scripts are ready.
- No real Docker agent/verifier rollout has run on the target cloud worker.
- No GPU policy smoke test, two-task canary, SFT, GRPO, or held-out evaluation has run.
- Success, efficiency, VRAM, throughput, duration, and cost metrics are all **Not measured**.
- Agent Lightning credit assignment and the dashboard are deferred until terminal GRPO and
  frozen evaluation produce real artifacts.

This is an execution runbook, not a completed run log. Before renting, finish and freeze the
inputs in the `Before renting a GPU` section. After renting, the only first end-to-end path is
GPU preflight, authenticated policy smoke, and the two-task canary. SFT collection, training,
GRPO, and evaluation remain locked until that canary passes.

## Deployment decision

Use the split deployment unless the rented GPU host passes the actual Docker smoke test.

### Recommended: split GPU and CPU workers

```text
CPU Docker worker                         GPU worker (one RTX 4090)
-----------------                         -------------------------
task snapshots                            Qwen policy + adapter
agent containers                          authenticated /action API
hidden tests                              SFT and GRPO training
verifier containers       SSH tunnel      model cache and checkpoints
rollout collector  -------------------->  127.0.0.1:8010
```

The CPU worker initiates an SSH local forward. The GPU policy server binds only to
`127.0.0.1`; do not expose port 8010 to the public Internet. The bearer token authenticates
`/action`, while SSH provides transport encryption. Hidden tests remain on the CPU worker.

### Conditional: one GPU node

This mode is allowed only after this command succeeds on the rented image:

```bash
bash cloud/scripts/preflight_gpu.sh single-node
```

The test reaches the Docker daemon and runs a network-disabled, read-only, unprivileged
container with dropped capabilities. A provider statement that Docker is installed is not
enough. If the command fails, use a separate CPU Docker worker.

## Capacity plan

The checked-in profile uses Qwen2.5-Coder-3B-Instruct, 4-bit base weights, bf16 compute,
LoRA rank 32, batch size 1, gradient checkpointing, and a 4096-token SFT limit. It is a
conservative first profile for a 24 GiB RTX 4090. Do not start with 7B or 8192-token training
until the 3B canary has measured memory headroom.

Approximate planning ranges are below. Every value is an unmeasured budget, not a result or
guarantee; record actual peak memory and disk use for every run.

| Workload | Planning VRAM range (not measured) | Notes |
| --- | ---: | --- |
| 3B 4-bit policy generation | 6-12 GiB | Context length and KV cache dominate the range. |
| 3B QLoRA SFT at 4096 tokens | 14-21 GiB | Version and sample length dependent. |
| RepoRL GRPO canary | 18-24 GiB | Start with group size 2 and short traces. |

Recommended host capacity:

| Node | CPU/RAM | Persistent fast disk | Main disk consumers |
| --- | --- | ---: | --- |
| GPU | 8 CPU, 32 GiB RAM | 150 GiB | Python/CUDA env, HF cache, adapters, checkpoints |
| CPU Docker | 8-16 CPU, 32-64 GiB RAM | 250 GiB | Images, repository snapshots, hidden tests, trajectories |

Hard preflight floors are 80 GiB free on the GPU node and 100 GiB on the CPU node. Typical
planning allowances are 10-25 GiB for the Python and model cache, 50-150 GiB for task images
and repository snapshots, 5-30 GiB for trajectories, and 5-20 GiB for checkpoint headroom.

## Files and invariants

The cloud assets are:

```text
cloud/.env.example
cloud/scripts/bootstrap_gpu.sh
cloud/scripts/bootstrap_cpu.sh
cloud/scripts/preflight_gpu.sh
cloud/scripts/preflight_cpu.sh
cloud/scripts/start_policy_server.sh
cloud/scripts/stop_policy_server.sh
cloud/scripts/open_policy_tunnel.sh
cloud/scripts/smoke_policy.sh
cloud/scripts/run_collect.sh
cloud/scripts/materialize_tasks.sh
cloud/scripts/bind_rollout_config.sh
cloud/scripts/prepare_sft_data.sh
cloud/scripts/prepare_grpo_job.sh
cloud/scripts/run_sft.sh
cloud/scripts/run_grpo.sh
cloud/scripts/create_transfer_bundle.sh
cloud/scripts/verify_transfer_bundle.sh
cloud/scripts/capture_environment.sh
cloud/scripts/validate_configs.sh
configs/*.toml
```

Maintain these invariants:

1. Both nodes run the same clean Git commit.
2. The Hugging Face revision is an immutable commit, never `main`.
3. The collector validates the full `/health` `PolicyIdentity` against the rollout config.
4. The behavior-policy revision is the SHA-256 digest of that full identity, not a path label.
5. GRPO consumes groups produced by that exact identity digest and adapter content digest.
6. Hidden tests and verifier manifests never enter the agent snapshot or GPU prompt context.
7. Every transferred bundle is checked with its adjacent `.sha256` file before extraction.
8. A run ID and output directory are immutable. A retry gets a new name.

The included model revision is:

```text
Qwen/Qwen2.5-Coder-3B-Instruct
488639f1ff808d1d3d0ba301aef8c11461451ec5
```

Update the revision in all related configs together if it is deliberately changed.

## Runtime limits

RepoRL has distinct runner, trajectory, backend, Docker, and process limits. Do not treat one
as a substitute for another:

| Layer | Current source | Enforcement |
| --- | --- | --- |
| Runner limits | `[rollout.runner]` | Three consecutive invalid actions, 100,000 policy-output characters, 3,000 conversation bytes, and 768 reserved context tokens. These are not timeouts. |
| Trajectory budget | Sealed `TaskSpec.budgets` | Steps, policy tokens, patch bytes, tool output, and total wall time are task-specific. The wall-time schema default is 1,800 seconds. |
| Remote policy call | `[rollout.policy].timeout_seconds = 180.0` | Each HTTP call uses the smaller of this backend cap and the remaining trajectory wall time. |
| Agent Docker operation | Remaining trajectory time plus sealed command limits | `AgentRunner` passes remaining wall time into tools and final diff export; named test commands cannot exceed their sealed timeout. |
| Independent verifier | Sealed verifier suite timeouts | Verification starts in a pristine container after the trajectory and uses fixed, model-inaccessible commands. |
| Collector process | `REPORL_COLLECTION_TIMEOUT_SECONDS=86400` | GNU `timeout` sends `SIGTERM` at expiry and `SIGKILL` 30 seconds later if needed. This is only an outer watchdog. |

The checked-in rollout templates use identical `[rollout.runner]` limits. Their task-file
digests are zero sentinels until binding, so the materialized `TaskSpec` remains authoritative
for each trajectory's wall-clock and execution budgets.

## Before renting a GPU

### 1. Finish and freeze local inputs

Run the repository quality checks, commit the code, and record the commit:

```bash
git status --short
git rev-parse HEAD
```

Do not rent the GPU until the working tree used for the run is clean and pushed. The CPU and
GPU clones must checkout that exact commit.

After installing the project in any Linux Python environment, validate the strict schemas and
cross-config policy lineage:

```bash
bash cloud/scripts/validate_configs.sh
```

Freeze the sealed verifier manifests and every content-addressed artifact they reference. The
recommended source layout is:

```text
artifacts/sealed/verifier-manifests.jsonl
artifacts/sealed/dataset-manifest.json
artifacts/sealed/split-seal.json
artifacts/sealed/repositories.jsonl
artifacts/sealed/admission-evidence.jsonl
artifacts/sealed/admission-results.jsonl
artifacts/sealed/... pinned clean, buggy, reference, hidden-test, and patch artifacts ...
```

The sealed manifests must use pinned agent and verifier image digests. Confirm that train,
validation, and test repositories have disjoint lineage groups before collection. Runtime
JSONL records retain artifact-root-relative paths and are rebound through
`task_artifacts_root` on the CPU worker. Materialize them after transfer so the same command
also validates the destination's sealed artifacts.

### 2. Build and verify the task bundle

Run the bundler from Linux, WSL, or a Linux CI job. It accepts only regular files and
directories, refuses to overwrite an existing output, creates a deterministic tar stream, and
writes a SHA-256 file.

```bash
bash cloud/scripts/create_transfer_bundle.sh \
  artifacts/transfers/tasks-v1.tar.gz \
  artifacts/sealed

bash cloud/scripts/verify_transfer_bundle.sh \
  artifacts/transfers/tasks-v1.tar.gz
```

Every verifier artifact referenced by a sealed manifest must be under the bundled artifact
root. Never put a real bearer token in this bundle.

### 3. Provision the CPU Docker worker

The CPU worker can be an existing Linux workstation or a CPU cloud VM. Install Docker Engine
using the host vendor's supported procedure, then clone and checkout the frozen RepoRL commit:

```bash
git clone https://github.com/zhaodong929/RepoRL-.git
cd RepoRL-
git checkout YOUR_FROZEN_COMMIT
bash cloud/scripts/bootstrap_cpu.sh
export REPORL_STORAGE_ROOT="$PWD"
bash cloud/scripts/preflight_cpu.sh
```

The preflight pulls `alpine:3.20` by default for a real container check. Set
`REPORL_PREFLIGHT_IMAGE` to an already approved equivalent if registry policy forbids that
image. Pre-pull all task images by immutable digest before the paid rollout window.

Copy both the task bundle and checksum to the CPU worker, verify them, then extract only after
verification:

```bash
bash cloud/scripts/verify_transfer_bundle.sh artifacts/transfers/tasks-v1.tar.gz
tar -xzf artifacts/transfers/tasks-v1.tar.gz -C .
```

If the Docker images are addressed through repositories, set
`REPORL_AGENT_IMAGE_REPOSITORY` and `REPORL_VERIFIER_IMAGE_REPOSITORY` to repository names
without an `@` suffix. After pulling or loading the exact image digests, materialize and
revalidate all split runtimes on this worker:

```bash
bash cloud/scripts/materialize_tasks.sh \
  artifacts/sealed/verifier-manifests.jsonl \
  artifacts/sealed \
  artifacts/tasks
```

The command publishes these files without overwriting prior evidence:

```text
artifacts/tasks/train-runtimes.jsonl
artifacts/tasks/validation-runtimes.jsonl
artifacts/tasks/test-runtimes.jsonl
```

It then re-hashes every sealed artifact and validates every runtime path. Run the CPU preflight
again after loading task images. Do not start the GPU rental if materialization or verification
fails.

Checked-in rollout TOMLs contain all-zero digest sentinels because dataset hashes do not exist
until materialization. They are templates and `run_collect.sh` rejects them. Bind each required
job config to the immutable metadata on the CPU worker. For the canary:

```bash
bash cloud/scripts/bind_rollout_config.sh \
  configs/rollout_remote_canary_4090.toml \
  artifacts/tasks/materialization-metadata.json \
  artifacts/job-configs/rollout-canary.toml
```

The binder validates the clean lineage audit, selected split, sealed runtime path, dataset and
split digests, selected runtime-file hash, and bound task records. It writes a new file
exclusively and never changes the checked-in template.

### 4. Prepare provider choices

Before payment, confirm the selected GPU offer has:

- one RTX 4090 with approximately 24 GiB VRAM;
- Ubuntu, Python 3.11 or 3.12, and a CUDA-enabled PyTorch image;
- at least 150 GiB of persistent fast disk or a deliberate external cache plan;
- outbound access to GitHub and Hugging Face, or pre-staged model files;
- inbound SSH access suitable for local port forwarding;
- a documented shutdown and persistent-volume policy.

Do not select the instance based on an assumption about nested Docker. The split design does
not require Docker on the GPU host.

For an AutoDL-like rental, verify the data-disk mount with `df -h` after login instead of
assuming a path from an older image. `/root/autodl-tmp` is common but is not guaranteed here.
Record the provider's external SSH host and mapped SSH port for the CPU-side tunnel. Keep the
policy HTTP port closed in the provider firewall; only SSH needs to be reachable. Do not try to
install or start Docker on the GPU instance unless the provider explicitly permits it and the
single-node preflight passes.

## GPU node setup

The following path is an example only. Use the provider's actual persistent disk mount.

```bash
cd /persistent
git clone https://github.com/zhaodong929/RepoRL-.git
cd RepoRL-
git checkout YOUR_FROZEN_COMMIT
export REPORL_STORAGE_ROOT=/persistent
bash cloud/scripts/bootstrap_gpu.sh
bash cloud/scripts/preflight_gpu.sh split
```

`bootstrap_gpu.sh` reuses the CUDA PyTorch supplied by the provider image through a dedicated
virtual environment. It deliberately refuses to guess and install a CUDA wheel if the
provider image has no usable CUDA PyTorch. Other direct GPU dependencies are constrained to
the versions in `uv.lock`; the provider PyTorch build and the full resolved environment are
captured in `.reporl/cloud/pip-freeze-gpu.txt`.

Set persistent Hugging Face cache paths before the first model load. Create `cloud/.env` from
the example, fill only non-secret values, and load it with export semantics:

```bash
cp cloud/.env.example cloud/.env
chmod 600 cloud/.env
set -a
source cloud/.env
set +a
```

Create a token without printing it or placing it in shell history:

```bash
umask 077
mkdir -p .reporl/cloud
openssl rand -hex 32 >.reporl/cloud/policy-token
export REPORL_POLICY_SERVER_TOKEN="$(<.reporl/cloud/policy-token)"
```

Provision the same token on the CPU worker through SSH file transfer, a cloud secret manager,
or another authenticated secret channel. Store it with mode 600 outside Git and export it in
the CPU shell. Do not put it in command-line arguments, TOML, logs, bundles, or commits.

Capture the initial environment:

```bash
bash cloud/scripts/capture_environment.sh .reporl/cloud/environment-gpu-before
```

## First paid-cloud gate: two-task canary

Do not start from an SFT seed, GRPO, or evaluation config. For the first paid execution, start
the GPU server from the exact checked-in canary policy config:

```bash
bash cloud/scripts/start_policy_server.sh \
  configs/rollout_remote_canary_4090.toml
```

The process runs in the background, writes a protected PID file and log under
`.reporl/cloud`, and only listens on `127.0.0.1:8010`. Inspect its log without printing
secrets:

```bash
tail -n 50 .reporl/cloud/policy-server.log
```

On the CPU worker, load non-secret connection settings from `cloud/.env`, export the token,
and open the tunnel in a persistent terminal:

```bash
set -a
source cloud/.env
set +a
export REPORL_POLICY_SERVER_TOKEN="$(<.reporl/cloud/policy-token)"
tmux new -s reporl-tunnel
bash cloud/scripts/open_policy_tunnel.sh
```

Detach from tmux with its normal prefix and leave the tunnel process running. In another CPU
shell, load the same environment and run an authenticated generation smoke test:

```bash
bash cloud/scripts/smoke_policy.sh
```

The smoke test verifies the model identity, revision, prompt-token trace, generated-token
trace, and one old log probability per generated token. It does not require Docker.

The server revision is a digest of the complete `PolicyIdentity`: resolved model and tokenizer
revisions, adapter content hash, quantization, dtype, Transformers version, chat-template hash,
special token IDs, context limits, and sampling settings. It cannot be predicted from an
adapter path. The collector obtains it through the authenticated health handshake and records
the full identity and digest in the immutable run manifest.

Passing a rollout config to `start_policy_server.sh` makes the server use that config's model,
revision, adapter path, context limit, generation limit, sampling parameters, and quantization
setting. This avoids an undetectable mismatch between nominal rollout settings and server-side
generation settings. Environment model settings are only a manual fallback when no config is
passed.

### Run the two-task canary

Run this on the CPU Docker worker:

```bash
bash cloud/scripts/run_collect.sh artifacts/job-configs/rollout-canary.toml
```

`run_collect.sh` wraps the complete collector process with GNU `timeout`. The default outer
watchdog is 86,400 seconds (24 hours); set `REPORL_COLLECTION_TIMEOUT_SECONDS` to an integer
from 300 through 604,800 to override it. At expiry, the supervisor sends `SIGTERM`, waits 30
seconds, and then sends `SIGKILL` if the process is still running. This is only a last-resort
process watchdog. It does not replace the task wall-time budget, the 180-second remote HTTP
cap, or the code-enforced Docker command deadlines described under `Runtime limits`.

The canary executes two candidates for each of two tasks. It must produce:

```text
artifacts/rollouts/remote-canary-001/run-manifest.json
artifacts/rollouts/remote-canary-001/evaluation.jsonl
artifacts/rollouts/remote-canary-001/trajectories/
artifacts/rollouts/remote-canary-001/verifications/
```

Review the manifest before scaling. Treat infrastructure errors, identity mismatches, missing
generation traces, verifier configuration failures, leaked hidden-test paths, or leftover
containers as stop conditions. Reward variance may be zero in a tiny canary; that is not by
itself an infrastructure failure.

Each config has a fixed run ID. The collector refuses to append to a non-empty run directory.
For another canary, copy the config to an ignored job-config directory and assign a new run ID.
Do not continue to the sections below until all four candidate trajectories and both independent
verifications are accounted for and every stop condition has been cleared.

## SFT data and training

**Status: not run.** This section is unlocked only by a successful two-task canary.

### 1. Collect seed trajectories

With the base policy server still running, collect train and validation trajectories on the
CPU worker:

```bash
bash cloud/scripts/bind_rollout_config.sh \
  configs/rollout_remote_sft_seed_4090.toml \
  artifacts/tasks/materialization-metadata.json \
  artifacts/job-configs/rollout-sft-seed.toml

bash cloud/scripts/bind_rollout_config.sh \
  configs/rollout_remote_sft_validation_4090.toml \
  artifacts/tasks/materialization-metadata.json \
  artifacts/job-configs/rollout-sft-validation.toml

bash cloud/scripts/run_collect.sh artifacts/job-configs/rollout-sft-seed.toml
bash cloud/scripts/run_collect.sh artifacts/job-configs/rollout-sft-validation.toml
```

Convert independently verified successes into assistant-masked SFT records:

```bash
bash cloud/scripts/prepare_sft_data.sh \
  artifacts/rollouts/sft-seed-001 \
  train \
  artifacts/datasets/sft/train.jsonl

REPORL_TASKS_FILE=artifacts/tasks/validation-runtimes.jsonl \
bash cloud/scripts/prepare_sft_data.sh \
  artifacts/rollouts/sft-validation-001 \
  validation \
  artifacts/datasets/sft/validation.jsonl
```

The preparation command fails rather than create an empty dataset. Check record counts,
trajectory IDs, verification hashes, repository splits, and rejection reports before training.

### 2. Transfer SFT records to the GPU

On the CPU worker:

```bash
bash cloud/scripts/create_transfer_bundle.sh \
  artifacts/transfers/sft-data-v1.tar.gz \
  artifacts/datasets/sft
```

Transfer both files:

```text
artifacts/transfers/sft-data-v1.tar.gz
artifacts/transfers/sft-data-v1.tar.gz.sha256
```

On the GPU worker:

```bash
bash cloud/scripts/verify_transfer_bundle.sh artifacts/transfers/sft-data-v1.tar.gz
tar -xzf artifacts/transfers/sft-data-v1.tar.gz -C .
```

### 3. Run SFT

Stop the policy server first so it does not reserve GPU memory:

```bash
bash cloud/scripts/stop_policy_server.sh
tmux new -s reporl-sft
bash cloud/scripts/run_sft.sh configs/sft_qwen25_coder_3b_4090.toml
```

The expected final adapter is:

```text
outputs/sft-qwen25-coder-3b-4090/adapter-final
```

The run also writes `run-manifest.json` and rejected-record reports. Record peak memory with
`nvidia-smi` in a second shell. If the first batch exceeds memory, reduce `max_length` to 2048
in a copied run config and assign a new output directory. Do not silently alter the canonical
run after results have been recorded.

## GRPO iteration 001

**Status: not run.** It additionally requires a completed, audited SFT adapter and fresh
on-policy groups from that exact adapter identity.

### 1. Serve the SFT adapter

On the GPU worker:

```bash
bash cloud/scripts/start_policy_server.sh \
  configs/rollout_remote_grpo_iteration_001_4090.toml
```

The checked-in rollout config pins the base revision, adapter path, and generation settings.
The runtime identity adds the adapter content digest and environment-sensitive tokenizer and
Transformers fields. `run_collect.sh` rejects the server before task execution if those fields
do not match the config or if an expected adapter is absent.

On the CPU worker, verify the tunnel and trace again, then collect:

```bash
bash cloud/scripts/smoke_policy.sh
bash cloud/scripts/bind_rollout_config.sh \
  configs/rollout_remote_grpo_iteration_001_4090.toml \
  artifacts/tasks/materialization-metadata.json \
  artifacts/job-configs/rollout-grpo-iteration-001.toml
bash cloud/scripts/run_collect.sh artifacts/job-configs/rollout-grpo-iteration-001.toml
```

The collector retains only complete trainable groups. Infrastructure-failed candidates cause
their group to be listed under `dropped_groups`. Zero-variance groups are recorded and skipped
by GRPO.

### 2. Transfer on-policy groups

On the CPU worker:

```bash
bash cloud/scripts/create_transfer_bundle.sh \
  artifacts/transfers/grpo-iteration-001.tar.gz \
  artifacts/rollouts/grpo-iteration-001/grpo-groups.jsonl \
  artifacts/rollouts/grpo-iteration-001/run-manifest.json
```

Transfer the bundle and checksum to the GPU, verify, and extract at the repository root. Then
materialize a per-run GRPO config from the verified rollout evidence:

```bash
bash cloud/scripts/prepare_grpo_job.sh \
  artifacts/rollouts/grpo-iteration-001 \
  artifacts/job-configs/grpo-iteration-001.toml
```

This command recomputes the local SFT adapter directory hash, validates the full identity and
its digest, checks every episode revision and the group count, and replaces the checked-in
template placeholder with the exact behavior-policy digest. It refuses to overwrite an
existing job config.

### 3. Train one GRPO epoch

Stop the policy server to release VRAM, then run the configured single epoch:

```bash
bash cloud/scripts/stop_policy_server.sh
tmux new -s reporl-grpo-001
bash cloud/scripts/run_grpo.sh artifacts/job-configs/grpo-iteration-001.toml
```

Expected outputs include:

```text
outputs/grpo-qwen25-coder-3b-4090-iteration-001/metrics.jsonl
outputs/grpo-qwen25-coder-3b-4090-iteration-001/checkpoint-epoch-1/
outputs/grpo-qwen25-coder-3b-4090-iteration-001/adapter-final/policy/
outputs/grpo-qwen25-coder-3b-4090-iteration-001/run-manifest.json
```

PEFT saves the named `policy` adapter under the nested directory shown above. Treat the
`output_adapter` path in `run-manifest.json` as authoritative if a future PEFT version writes a
different compatible layout.

For iteration 002, serve the iteration-001 adapter, create a new rollout config and run ID,
collect fresh groups, then create a new GRPO config and output directory. Never train an
updated policy on rollout groups collected from an older revision while calling them on-policy.

## Evaluation

**Status: not run.** All metric values remain `Not measured` until the repository-held-out
evaluation artifacts and paired reports in this section exist.

Use identical test tasks, seeds, budgets, and deterministic decoding for all methods. The
checked-in evaluation configs use seed 12042 and temperature 0.

Serve each adapter in turn and collect on the CPU worker:

```text
configs/eval_remote_base_4090.toml
configs/eval_remote_sft_4090.toml
configs/eval_remote_grpo_iteration_001_4090.toml
```

Start the policy server with the evaluation config itself so its adapter and deterministic
decoding settings cannot drift. Stop the preceding policy server before changing configs.
First bind each evaluation template to the test split metadata on the CPU worker. For example:

```bash
bash cloud/scripts/bind_rollout_config.sh \
  configs/eval_remote_base_4090.toml \
  artifacts/tasks/materialization-metadata.json \
  artifacts/job-configs/eval-base.toml
```

Repeat for SFT and RL evaluation. Run the first command below on the GPU worker with the
checked-in template, and the second on the CPU worker with its bound counterpart:

```bash
bash cloud/scripts/start_policy_server.sh TEMPLATE_CONFIG_PATH
bash cloud/scripts/run_collect.sh BOUND_CONFIG_PATH
```

Combine the three `evaluation.jsonl` files without changing line contents, then generate paired
reports. For example:

```bash
mkdir -p artifacts/evaluations/reports
cat \
  artifacts/evaluations/eval-base-001/evaluation.jsonl \
  artifacts/evaluations/eval-sft-001/evaluation.jsonl \
  artifacts/evaluations/eval-grpo-iteration-001/evaluation.jsonl \
  >artifacts/evaluations/reports/all-records.jsonl

.venv-cloud/bin/python -m reporl.evaluation.report \
  --records artifacts/evaluations/reports/all-records.jsonl \
  --baseline base-prompt-agent \
  --candidate rl-agent-iteration-001 \
  --resamples 10000 \
  --seed 42 \
  --output artifacts/evaluations/reports/base-vs-rl.json
```

Keep SFT-vs-base and RL-vs-SFT comparisons as separate reports. Report task success, target and
regression pass fractions, tool calls, tokens, wall time, policy violations, and infrastructure
errors. Do not count infrastructure failures as agent failures without a separate sensitivity
analysis.

## Stop and resume behavior

### Policy service and tunnel

Stop the policy service with the PID-verified helper:

```bash
bash cloud/scripts/stop_policy_server.sh
```

The helper sends SIGTERM only after verifying the process command line. It does not force-kill
an unresponsive process. Stop the SSH tunnel with Ctrl-C in its tmux session.

### Rollout collection

Rollout directories are immutable and collection is not resumable in place. If interrupted,
keep the partial directory for diagnosis, assign a new `run_id`, and restart. Do not merge
partial groups unless an explicit audited merge tool verifies task, policy revision, candidate
count, seeds, and unique trajectory IDs.

An outer watchdog expiry normally returns status 124 from GNU `timeout`; status 137 indicates
the collector did not exit during the 30-second `SIGTERM` grace period and was killed. Diagnose
the partial run before retrying. Increasing the outer watchdog changes neither the sealed task
wall-time budget nor the remote HTTP and Docker command deadlines enforced inside RepoRL.

### SFT

SFT writes Transformers checkpoints at `save_steps`. To resume, copy the SFT TOML to a new
job config and insert this key inside `[sft]`, before `[sft.lora]`:

```toml
resume_from_checkpoint = "outputs/sft-qwen25-coder-3b-4090/checkpoint-100"
```

Keep the same data, seed, model revision, and optimizer settings. Preserve the original config
and environment record for audit.

### GRPO

The current GRPO implementation saves adapter weights after each complete epoch but does not
save optimizer and scheduler state. It cannot resume exactly in the middle of an epoch. The
provided config uses one epoch. If interrupted, keep the incomplete output, select a new output
directory, and restart from the same initial adapter and same on-policy group file. If a complete
new adapter is adopted as the behavior policy, recollect rollout groups before the next update.

## End-of-rental procedure

Do this before shutting down a non-persistent instance:

1. Stop the policy server and any training process cleanly.
2. Capture the final environment and Git state.
3. Bundle adapters, manifests, metrics, configs, and evaluation records.
4. Transfer both bundle and `.sha256` file off the instance.
5. Verify the received bundle on the destination before releasing the instance.

Example on the GPU worker:

```bash
bash cloud/scripts/capture_environment.sh .reporl/cloud/environment-gpu-after
bash cloud/scripts/create_transfer_bundle.sh \
  artifacts/transfers/gpu-results-iteration-001.tar.gz \
  outputs/sft-qwen25-coder-3b-4090 \
  outputs/grpo-qwen25-coder-3b-4090-iteration-001 \
  .reporl/cloud/environment-gpu-after \
  artifacts/job-configs/grpo-iteration-001.toml \
  configs
```

On the CPU worker, list any interrupted containers before shutdown:

```bash
docker ps -a --filter label=reporl.role
```

Inspect container IDs before removing leftovers. Do not run broad cleanup commands on a shared
Docker host.

## Paid-run gate

Rent the 4090 only when every applicable item is true:

- Code is pushed at a clean, recorded commit and both nodes will use it.
- Train, validation, and test lineage splits have been audited.
- Task images are pinned by digest and available on the CPU worker.
- Task bundles verify against their SHA-256 records.
- CPU Docker preflight passes with the intended daemon and user.
- GPU offer provides 24 GiB VRAM, persistent disk, SSH, and a compatible CUDA PyTorch image.
- The split tunnel and token provisioning procedure is prepared.
- Canary limits, stop conditions, run IDs, output paths, and cost ceiling are written down.
- Result transfer and checksum verification are tested before rental.

After rental, do not start the full job until GPU preflight, policy smoke, and the two-task
rollout canary all pass.
