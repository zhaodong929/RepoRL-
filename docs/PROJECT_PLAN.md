# RepoRL Feasible Project Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-29
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## Experiment overview

- **Title:** RepoRL: Executable-Reward Post-Training for Repository-Level Coding Agents
- **Objective:** Build and evaluate an auditable coding-agent training system from issue to patch.
- **Primary hypothesis:** Under a fixed harness and inference budget, SFT plus cost-aware RL
  improves regression-safe patch success over SFT on repository lineages held out from project
  post-training.
- **Type:** Agent system engineering, post-training, and controlled evaluation.
- **Estimated core duration:** 10-12 full-time weeks or 16-20 part-time weeks.

## Deliverable definition

The hiring-ready project is complete when another engineer can:

1. install the pinned environment from a clean Linux machine;
2. validate task provenance and repository-lineage isolation;
3. reproduce a prompt baseline, SFT checkpoint, and at least one RL checkpoint;
4. inspect any trajectory from issue through tools, patch, and verifier evidence;
5. reproduce the primary table and repository-aware confidence intervals with one documented
   workflow;
6. view a real successful and failed trajectory in a read-only demo;
7. distinguish measured results from targets, examples, and exploratory findings.

The project is not complete merely because one training loss curve runs or a demo patch passes a
model-visible test.

## Scope decisions

### Core scope

- Python repositories and pytest-compatible tasks only.
- Five structured tools: search, bounded read, unified-diff patch, named test suite, finish.
- Docker agent/verifier isolation with executable target and regression rewards.
- Prompt baseline, verified-trajectory QLoRA SFT, and small-model GRPO pilot.
- Repository-lineage split, immutable trajectories, reproducible evaluation, and a technical
  report.

### Deferred scope

- Agent Lightning step-level credit assignment, until terminal RL is stable.
- 7B online RL, until a 3B memory and throughput profile justifies it.
- General shell, browser, package installation, multi-language repositories, and distributed
  sandbox scheduling.
- FastAPI/Streamlit demo, until real evaluation artifacts exist.
- LangGraph, RAG frameworks, and production OpenHands integration.

Deferral keeps the research contribution centered on environments, verifiers, rewards, and
reproducible post-training rather than framework assembly.

## Target system

```text
Task manifest
    -> AgentRunner
       -> PolicyBackend
       -> ToolGateway
       -> isolated AgentSandbox
    -> patch artifact
    -> policy gate
    -> pristine VerifierSandbox + hidden tests
    -> VerificationResult + RewardVector
    -> immutable TrajectoryStore
    -> SFT / GRPO adapters
    -> frozen evaluation and report
```

See `ARCHITECTURE.md` for component contracts and trust boundaries.

## Setup

- **Development OS:** WSL2 Ubuntu or native Linux.
- **Python:** 3.11 for training; 3.12 is supported for core development.
- **Packaging:** `uv` with committed lock file.
- **Core:** Pydantic, pytest, Ruff, mypy.
- **Sandbox milestone:** Docker Engine/Desktop with Linux containers, Docker SDK, unidiff.
- **Training milestone:** PyTorch, Transformers, Accelerate, PEFT, Datasets, TRL after a versioned
  integration spike; vLLM is Linux-only and optional.
- **Storage:** append-only JSONL plus content-addressed files; Parquet/DuckDB for derived analysis.
- **Tracking:** local artifacts are canonical; W&B or MLflow is optional.

### Current-machine constraint

The audited development machine has an RTX 5070 Laptop GPU with about 8 GB VRAM. Python 3.13 is
installed on Windows, while WSL has Python 3.12 and `uv`. Docker is not currently installed.
Therefore:

- core contracts, task fixtures, CI, analysis, and API-model baselines can run locally;
- Docker installation and WSL integration are a blocker for M1, not for repository initialization;
- local training is limited to small quantized inference and experimental 1.5B/3B adapters;
- credible multi-rollout RL is budgeted on cloud GPUs after a measured smoke test.

## Workstreams

### A. Environment and verifier

Owns task snapshots, containers, tools, patch policy, JUnit parsing, hidden tests, and result
classification. This is the highest-priority workstream because a weak verifier invalidates every
later result.

### B. Agent and trajectory system

Owns typed actions, policy adapters, state machine, budgets, context policy, replay, token masks,
and immutable traces.

### C. Data and provenance

Owns repository selection, license review, lineage grouping, SWE-smith adapters, task validation,
split sealing, natural issue curation, and leak scans.

### D. Post-training

Owns trajectory filtering, chat-template conversion, QLoRA SFT, compute-matched continued SFT,
interactive rollouts, reward integration, GRPO, checkpoints, and run manifests.

### E. Evaluation and communication

Owns fixed-budget evaluation, hierarchical bootstrap, sensitivity analyses, failure taxonomy,
technical report, README results, and artifact-only demo.

## Milestones and gates

### M0: Repository foundation, 1-2 days

Deliverables:

- source layout, Python package, `uv.lock`, lint, type checks, unit tests, and CI;
- research question, architecture, experiment protocol, and risk register;
- `main` pushed to GitHub with no fabricated results.

Gate:

- `uv sync --extra dev`, `ruff`, `mypy`, and `pytest` pass on Python 3.11/3.12;
- all public links resolve and CI is green.

### M1: Trusted task and sandbox vertical slice, week 1

Deliverables:

- two tiny fixture repositories with local and cross-file bugs;
- agent and verifier container builders pinned by digest;
- task manifest and verifier-only manifest schemas;
- tool gateway with bounded search/read, structured patch application, and named test suites;
- JUnit XML parser and patch-policy checks.

Gate:

- clean, buggy, and reference states reproduce at least three times;
- hidden assets and `.git` history are absent from the agent container;
- network, Docker socket, traversal, symlink, forbidden-file, fork-bomb, memory, output, and timeout
  probes fail closed;
- infrastructure failure is distinguishable from agent failure.

### M2: Deterministic agent harness, week 2

Deliverables:

- `PolicyBackend`, parser, runner, budget manager, trajectory writer, and replay reader;
- deterministic fake policies for success, invalid action, timeout, and budget exhaustion;
- one real model adapter and a compact ReAct/tool-use prompt.

Gate:

- one command runs at least 20 fixture/pilot tasks end to end;
- fake-policy traces replay with identical action/result hashes;
- every terminal state is classified, sandbox cleanup is guaranteed, and artifact completeness is
  at least 99%.

### M3: Data pilot and lineage audit, weeks 3-4

Deliverables:

- repository intake checklist covering license, tests, build time, and stability;
- fork/shared-commit/clone lineage grouper;
- SWE-smith task adapter and clean-buggy-reference admission validator;
- initial split of roughly 60-100 / 20-30 / 40-60 tasks over 3 / 1 / 2 lineages;
- provenance ledger and leak scanner.

Gate:

- repository lineages are disjoint by automated assertions and manual audit;
- every admitted task passes repeated three-state validation;
- issue text and agent images pass gold/mutation/hidden-test leak checks;
- test tasks are sealed before prompt or reward tuning uses them.

### M4: Prompt baseline, week 5

Deliverables:

- frozen base model revision, chat template, prompt, decoding settings, and inference budgets;
- P0 runs on pilot train/validation data and a pre-seal smoke set;
- failure taxonomy and throughput/cost profile.

Gate:

- the entire run is reproducible from a run manifest;
- infrastructure errors are below a frozen operational threshold, initially 2%;
- invalid actions and context truncation are measured, not silently repaired;
- environment/test execution cost is known well enough to budget data generation and RL.

### M5: Dataset expansion and SFT, weeks 6-7

Deliverables:

- core train/validation/test target of 300-600 / 60-120 / 150-320 validated synthetic tasks;
- verified successful trajectories from train lineages only;
- S0 QLoRA training, checkpoint resumption, validation selection, and SC compute control;
- assistant-only loss masks and exact chat-template tests.

Gate:

- no test-lineage trajectory enters training or checkpoint selection;
- tool-call parse rate and invalid-action rate improve on validation;
- S0 is reproducible for at least two smoke seeds before the final three-seed run;
- SC has a documented compute/data matching rule.

### M6: Interactive RL pilot, weeks 8-10

Deliverables:

- version-pinned TRL integration spike and `RolloutBackend` contract test;
- R0, R1, and R2 with 1.5B/3B policy, group size 4, 8-12 steps, bounded episode tokens;
- R3 no-SFT warm-start ablation at reduced scale;
- reward curves, KL, length, zero-variance group rate, verifier throughput, GPU hours, and peak VRAM.

Gate:

- generated token IDs, old-policy log probabilities, masks, group IDs, and policy versions are
  retained exactly;
- tool observations receive zero policy loss;
- at least one training run completes without NaN, unhandled OOM, stale-rollout ambiguity, or
  unverifiable samples;
- success-dominance reward invariant passes for the frozen weights;
- the cost estimate justifies or rejects scale-up before additional cloud spend.

### M7: Frozen evaluation and ablations, weeks 11-12

Deliverables:

- at least three independent training seeds for headline conditions;
- one sealed evaluation across P0, S0, SC, R0, R1, R2, and R3 under identical budgets;
- paired two-level bootstrap, leave-one-repository-out analysis, and failure review;
- optional natural issue external set; R4 only if core gates are already met.

Gate:

- the primary comparison, exclusions, reruns, and confidence intervals are reproducible;
- non-inferiority and cost criteria use validation-frozen thresholds;
- no result is promoted from exploratory to confirmatory after viewing test outcomes.

### M8: Portfolio release, 3-5 days

Deliverables:

- README populated with measured values and uncertainty;
- technical report PDF, model/data cards, reproducibility commands, and limitations;
- read-only trajectory demo showing at least one first-patch failure followed by a successful
  correction, plus a representative failure;
- tagged release with manifests and artifact index.

Gate:

- every README number links to a machine-readable result artifact;
- the demo renders recorded artifacts and cannot alter the evaluated repository;
- a clean-machine reproduction dry run succeeds.

## Initial issue backlog

Create these as GitHub milestones/issues after M0:

| Priority | Issue | Milestone | Acceptance summary |
|---|---|---|---|
| P0 | Define verifier-only manifest | M1 | Hidden fields cannot serialize into agent view |
| P0 | Build fixture repositories | M1 | Clean/buggy/reference states deterministic |
| P0 | Harden Docker sandbox | M1 | Security probe suite fails closed |
| P0 | Implement patch policy | M1 | Traversal/test/config/symlink edits rejected |
| P0 | Parse pytest JUnit output | M1 | Canonical IDs and infra status are structured |
| P0 | Implement AgentRunner | M2 | Budgets and all terminal states covered |
| P0 | Add immutable trajectory store | M2 | Replay and artifact hashes verified |
| P0 | Validate lineage split | M3 | Fork/shared/clone leakage assertions pass |
| P1 | Add SWE-smith task adapter | M3 | Admission checks produce signed manifest |
| P1 | Run prompt baseline | M4 | Frozen config and complete artifact set |
| P1 | Build SFT dataset converter | M5 | Assistant-only masks unit-tested |
| P1 | Train S0 and SC | M5 | Resume and validation reports reproducible |
| P1 | Spike interactive TRL rollouts | M6 | Logprob/mask/group contract demonstrated |
| P1 | Train R0-R3 | M6 | Reward and health metrics complete |
| P1 | Implement hierarchical bootstrap | M7 | Simulation and paired tests pass |
| P2 | Curate natural issue set | M7 | Post-release provenance and no leakage |
| P2 | Add Agent Lightning adapter | M7 | Only credit assignment differs from R2 |
| P2 | Build artifact viewer | M8 | Read-only, mobile/desktop usable, no live execution |

## Compute plan

### Required measurement before renting GPUs

Run a 10-task benchmark and record:

- generated tokens per step and per episode;
- policy tokens per second at the chosen context length;
- average target and regression test seconds;
- peak inference, optimizer, and verifier memory;
- valid episodes per wall-clock hour and zero-variance group rate.

Estimate rollout generation before every scale decision:

```text
generated_tokens = tasks * epochs * group_size * mean_steps * tokens_per_step
```

For example, `1000 * 1 * 8 * 20 * 256` is about 41 million generated tokens before training
and test execution. That is not a laptop-scale pilot.

### Practical tiers

| Tier | Intended work | Rough hardware expectation |
|---|---|---|
| Local 8 GB | Core code, Docker CPU tests, API baseline, short quantized inference | Current laptop |
| Pilot SFT | 1.5B/3B QLoRA, short contexts | One 24 GB GPU after profiling |
| Pilot RL | 1.5B/3B, group 4, sequential rollout/train | One or two 48 GB GPUs, workload-dependent |
| 7B RL | Only after measured 3B result and funding gate | Multi-GPU or 80 GB-class capacity |

These are planning ranges, not guarantees. Sequence length, attention implementation, optimizer,
LoRA rank, vLLM topology, and concurrent rollouts determine actual memory.

### Cost controls

- prebuild immutable repository images and cache dependencies outside timed rollouts;
- run focused public tests during the trajectory and the complete hidden/regression suite only in
  terminal verification;
- use copy-on-write workspaces and asynchronous CPU verifier workers;
- stop scale-up when environment throughput, not policy inference, is the bottleneck;
- retain failed pilot measurements to prevent repeated unproductive configurations.

## Quality gates

| Gate | Required threshold before next phase |
|---|---|
| Task determinism | Clean/buggy/reference outcomes stable across at least 3 repeats |
| Artifact completeness | At least 99% of non-infrastructure pilot episodes fully reconstructable |
| Split integrity | Zero lineage overlap and zero known gold/hidden-test leakage |
| Sandbox security | All required adversarial probes fail closed |
| Baseline stability | Same config rerun is within preregistered stochastic tolerance |
| Trainer integrity | Assistant-only masks/logprobs verified; infra episodes excluded |
| Evaluation integrity | Test sealed, paired budgets equal, cluster-aware CI reproducible |

## Risks and mitigations

| Risk | Impact | Mitigation / stop rule |
|---|---|---|
| Hidden-test or gold leakage | Invalidates reward and claims | Separate images/manifests, byte scans, red-team tests; stop all training on detection |
| Sandbox escape or arbitrary shell | Host/security compromise | Named suites, no shell, non-root/no-network/no-socket, resource limits |
| Synthetic tasks do not represent issues | Weak external validity | Natural issue set; limit claim to measured distributions |
| Fork/clone leakage | Inflated generalization | Split by lineage before generation; commit and MinHash audit |
| Flaky tests | Noisy or wrong reward | Repeated admission tests, quarantine, fixed images |
| Too few test repositories | False precision | Preserve lineage count, cluster bootstrap, leave-one-out |
| TRL cannot preserve interactive rollout math | Incorrect RL implementation | Integration contract spike; keep backend abstraction; do not label failed bridge as GRPO |
| Early all-fail/all-pass groups | Zero GRPO signal | Log/skip zero-variance groups, curriculum by validation difficulty |
| GPU or generation cost exceeds budget | Incomplete experiment | Start 1.5B/3B, group 4, short horizon; profile before scale |
| pytest becomes throughput bottleneck | Expensive idle GPU | Prebuilt images, focused suites, parallel CPU verifier pool |
| Reward weights reduce success | Optimizes cost at wrong tradeoff | Success-dominance invariant and validation-frozen non-inferiority rule |
| Base-model pretraining contamination | Overstated novelty | Precise held-out wording and post-release natural set |
| Licensing blocks redistribution | Dataset cannot ship | License intake before generation; publish metadata/scripts where tasks cannot be redistributed |

## Expected outputs

| Output | Path | Format | Success criterion |
|---|---|---|---|
| Task manifests | `data/manifests/` | JSONL | Provenance, lineage, hashes, and validation state complete |
| Trajectories | `artifacts/<run>/trajectories/` | JSONL | Immutable and replayable |
| Verifier evidence | `artifacts/<run>/verifier/` | JSON + JUnit XML | Patch and environment hashes match manifest |
| SFT/RL checkpoints | `artifacts/<run>/checkpoints/` | Adapter weights | Resumable with exact config |
| Evaluation summary | `artifacts/<run>/summaries/` | Parquet + JSON | Rebuilds every table value |
| Technical report | `report.pdf` | PDF | Methods, uncertainty, ablations, failures, limitations |
| Demo | `dashboard/` | Read-only app | Renders recorded trajectories without live patch execution |

Large outputs stay outside Git and are published through a versioned artifact release or model
registry. The repository retains small manifests, schemas, scripts, and checksums.

## First seven working days

1. Install Docker Desktop/Engine with WSL2 Linux integration and verify a non-root, no-network
   container.
2. Finish verifier-only schemas and two fixture repositories.
3. Implement bounded search/read/patch/test tools without a generic shell.
4. Build pristine agent/verifier images and the clean-buggy-reference validator.
5. Add security and determinism contract tests.
6. Implement a fake-policy end-to-end episode and immutable trace.
7. Run the first 20-task pilot, inspect every failure manually, and revise the manifest before any
   model training.

## Decision log

- Use `src/reporl` package layout while preserving the conceptual agent/environment/reward/training
  boundaries from the proposal.
- Keep the remote repository slug `RepoRL-` because it was explicitly supplied; use `RepoRL` and
  `reporl` for the project and Python package. Renaming the GitHub repository to `RepoRL` later
  would improve presentation but is not required technically.
- Use MIT for RepoRL-owned code; source repository licenses remain independently binding.
- Do not publish placeholder performance gains. README results remain `Not measured` until M7.
- Treat Agent Lightning and the dashboard as extensions, not prerequisites for a defensible core
  result.
