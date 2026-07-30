# RepoRL

Executable-reward post-training for repository-level coding agents.

> **Current status (2026-07-30): AutoDL pre-rental implementation is ready.** CPU-side
> contracts, deterministic harnesses, task materialization, rollout accounting, training
> entrypoints, evaluation code, and cloud orchestration are implemented and covered by local
> non-Docker checks. A real Docker rollout, GPU policy canary, SFT run, GRPO run, and held-out
> evaluation have **not** been executed. All research metrics are **Not measured**.

## Status

| Area | State | Evidence or next gate |
| --- | --- | --- |
| CPU contracts | Ready before rental | Typed agent, tools, sealed tasks, verifier, rewards, rollout store, and provenance checks are implemented. |
| Cloud orchestration | Ready before rental | Split-node scripts, strict 4090 configs, authenticated policy protocol, transfer checksums, and recovery steps are present. |
| Real Docker execution | Not run | Must pass the CPU preflight and hardened two-task rollout canary on a real Docker daemon. |
| GPU inference and training | Not run | Must pass the RTX 4090 preflight, authenticated policy smoke test, and two-task canary before any SFT or GRPO job. |
| Experimental results | Not measured | No success-rate, efficiency, memory, throughput, or cost result is claimed. |
| Agent Lightning and dashboard | Deferred | Revisit only after terminal-reward GRPO and frozen evaluation produce real artifacts. |

The source tree is therefore a runnable experimental system, not evidence that the research
hypothesis is true. The first paid-cloud evidence must come from the canary described in
[the cloud runbook](docs/CLOUD_RUNBOOK.md).

## Research question

Under a fixed inference budget, can supervised fine-tuning and executable-reward
reinforcement learning improve a small open coding model's success rate and behavioral
efficiency on repositories held out from post-training?

RepoRL turns an issue into a verified patch through a bounded loop:

```text
Issue + repository
        |
        v
Policy -> search -> read -> patch -> run named test suite
        |                              |
        +---------- trajectory <-------+
                       |
                       v
             isolated verifier sandbox
                       |
                       v
       reward vector + patch + reproducible artifacts
```

The MVP exposes five typed actions: `search_code`, `read_file`, `apply_patch`, `run_tests`,
and `finish`. The policy never receives arbitrary host shell access.

## Implemented scope

- A bounded `AgentRunner`, strict action parser, compact conversation policy, and fake/remote/HF
  policy backends.
- Network-disabled Docker agent and pristine verifier adapters with no Docker socket, bounded
  resources, path controls, named command aliases, and JUnit-based executable evidence.
- Patch-policy checks for traversal, forbidden files, tests, skips, modes, symlinks, binaries,
  and related bypasses.
- Sealed task manifests, repository-lineage splits, SWE-smith import preparation, admission
  evidence, content hashing, portable materialization, and split-bound rollout configs.
- Immutable trajectory and verification artifacts, grouped rollout accounting, terminal and
  cost-aware rewards, assistant-masked SFT records, QLoRA SFT, offline grouped GRPO, and paired
  bootstrap evaluation.
- A recommended CPU-Docker plus GPU-policy deployment with an authenticated loopback service,
  SSH tunnel, pinned model revision, transfer checksums, preflight gates, and recovery scripts.

These are implemented contracts and entrypoints. Their real Docker, CUDA, performance, and
research behavior remains unverified until the paid canary and subsequent experiments run.

## Design invariants

- Agent and verifier run in separate, network-disabled sandboxes.
- Hidden tests, reference patches, and reward code are never mounted into the agent sandbox.
- The controller, not the model, owns tool schemas, budgets, and termination.
- Test commands are trusted manifest aliases rather than model-authored shell commands.
- Every task is pinned by repository commit and container image digest.
- Every result retains the full reward vector; infrastructure failures are not training samples.
- Train, validation, and test repository lineages are disjoint.
- Headline comparisons use the same model family, harness, sampling settings, and budgets.

## Planned comparison

| Method | Policy initialization | Training signal | Current state |
| --- | --- | --- | --- |
| Prompt Agent | Base instruct model | None | Implemented, not run on the frozen benchmark |
| SFT Agent | Base instruct model | Verified successful trajectories | Entrypoint ready, not trained |
| Outcome RL | SFT checkpoint | Final executable outcome | Entrypoint ready, not trained |
| Cost-aware RL | SFT checkpoint | Outcome, safety, and capped cost terms | Entrypoint ready, not trained |
| Step-credit RL | Best RL checkpoint | Agent Lightning ablation | Deferred |

A direct, tool-free base model is retained only as a secondary reference because it does not
have the same information access as an agent.

## Results

| Metric | Prompt Agent | SFT Agent | RL Agent |
| --- | ---: | ---: | ---: |
| Held-out task success | Not measured | Not measured | Not measured |
| Pass@1 / Pass@k | Not measured | Not measured | Not measured |
| Mean tool calls | Not measured | Not measured | Not measured |
| Mean policy tokens | Not measured | Not measured | Not measured |
| Invalid action rate | Not measured | Not measured | Not measured |
| Regression break rate | Not measured | Not measured | Not measured |
| Wall time and cloud cost | Not measured | Not measured | Not measured |

Results will be added only after the test repositories and evaluation protocol are frozen.
Uncertainty will use repository-cluster or hierarchical bootstrap intervals, not a naive
task-only bootstrap. Planning ranges in the runbook are not measured results.

## Validate before renting

Use Linux or WSL with Python 3.11 or 3.12 and
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev --extra sandbox
uv run ruff check .
uv run mypy
uv run pytest
bash -n cloud/scripts/*.sh
bash cloud/scripts/validate_configs.sh
```

Tests that require a live Docker daemon may skip when Docker is unavailable. A skipped Docker
test is not cloud evidence. Before renting a GPU, freeze and push one clean commit, prepare the
sealed task bundle, materialize it on the CPU Docker worker, bind the canary template, and pass
the CPU preflight. Follow [the cloud runbook](docs/CLOUD_RUNBOOK.md) rather than improvising
commands from this summary.

## First paid-cloud gate

The only first end-to-end entry after renting the RTX 4090 is the two-task canary:

1. Run the GPU split-node preflight.
2. Start the base policy from `configs/rollout_remote_canary_4090.toml`.
3. Open the authenticated SSH tunnel and pass `cloud/scripts/smoke_policy.sh`.
4. Run the already bound `artifacts/job-configs/rollout-canary.toml` on the CPU Docker worker.
5. Audit both tasks, all four candidate trajectories, verifier evidence, identity traces, and
   leftover containers.

Do not start SFT collection, SFT, GRPO, or held-out evaluation until this gate passes. The
canary is an infrastructure acceptance test, not a research result.

## Runtime limits

The timeout layers are deliberately separate:

| Layer | Current source | Behavior |
| --- | --- | --- |
| Runner limits | `[rollout.runner]` | Three consecutive invalid actions, 100,000 output characters, 3,000 conversation bytes, and 768 reserved context tokens; these are not clock deadlines. |
| Trajectory deadline | Sealed `TaskSpec.budgets.max_wall_time_seconds` | `AgentRunner` passes the remaining trajectory budget into policy and Docker operations; the schema default is 1,800 seconds. |
| Remote policy call | `[rollout.policy].timeout_seconds` | Current cloud templates cap each remote HTTP call at 180 seconds, further bounded by remaining trajectory time. |
| Docker command | Sealed suite command plus remaining trajectory time | Tool and verifier execution use fixed command deadlines; model output cannot extend them. |
| Collector watchdog | `REPORL_COLLECTION_TIMEOUT_SECONDS` | GNU `timeout` defaults to 86,400 seconds, sends `SIGTERM`, then `SIGKILL` after 30 seconds. It is only a process-level fallback. |

## Repository layout

```text
src/reporl/
  agent/        # Policy backends, parser, prompts, runner, and policy service
  sandbox/      # Agent sandbox contract and hardened Docker adapter
  tools/        # Search, bounded read, patch, named tests, and path controls
  tasks/        # Provenance, admission, lineage, SWE-smith, and materialization
  verifier/     # Pristine Docker verification, JUnit evidence, and policy gate
  rewards/      # Executable outcome, safety, progress, and capped cost rewards
  rollouts/     # Strict configs, collection, accounting, and immutable storage
  training/     # SFT preparation/training and grouped offline GRPO
  evaluation/   # Metrics and repository-aware paired bootstrap reports
cloud/scripts/  # Linux CPU/GPU bootstrap, preflight, transfer, run, and recovery helpers
configs/        # Pinned 3B/RTX 4090 rollout, SFT, GRPO, and evaluation templates
tests/
  unit/         # Deterministic CPU contract tests
  contract/     # Docker isolation and adversarial contract tests
```

The dashboard and Agent Lightning integration are intentionally absent from the current
critical path. They remain deferred until real evaluation artifacts exist.

See [the project plan](docs/PROJECT_PLAN.md), [architecture](docs/ARCHITECTURE.md),
[experiment protocol](docs/EXPERIMENT_PROTOCOL.md),
[task materialization guide](docs/TASK_MATERIALIZATION.md), and
[cloud runbook](docs/CLOUD_RUNBOOK.md).

## License

MIT. Dataset records also preserve each source repository's license and provenance;
source-code licenses remain in force for generated tasks and patches.
