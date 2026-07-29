# RepoRL

Executable-reward post-training for repository-level coding agents.

> Status: foundation phase. The repository contains the research plan and initial
> typed contracts. No model-training result is claimed yet.

## Research question

Under a fixed inference budget, can supervised fine-tuning and executable-reward
reinforcement learning improve a small open coding model's success rate and behavioral
efficiency on repositories held out from post-training?

RepoRL trains an agent to turn an issue into a verified patch through a bounded loop:

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

The MVP exposes five typed actions: `search_code`, `read_file`, `apply_patch`,
`run_tests`, and `finish`. The policy never receives arbitrary host shell access.

## Design invariants

- Agent and verifier run in separate, network-disabled sandboxes.
- Hidden tests, reference patches, and reward code are never mounted into the agent sandbox.
- The controller, not the model, owns tool schemas, budgets, and termination.
- Test commands are trusted manifest aliases rather than model-authored shell commands.
- Every task is pinned by repository commit and container image digest.
- Every result retains the full reward vector; infrastructure failures are not training samples.
- Train, validation, and test repositories are disjoint.
- Headline comparisons use the same model family, harness, sampling settings, and budgets.

## Planned comparison

| Method | Policy initialization | Training signal |
|---|---|---|
| Prompt Agent | Base instruct model | None |
| SFT Agent | Base instruct model | Verified successful trajectories |
| Outcome RL | SFT checkpoint | Final executable outcome |
| Cost-aware RL | SFT checkpoint | Outcome, safety, and capped cost terms |
| Step-credit RL | Best RL checkpoint | Optional long-trajectory ablation |

A direct, tool-free base model is retained only as a secondary reference because it does not
have the same information access as an agent.

## Results

| Metric | Prompt Agent | SFT Agent | RL Agent |
|---|---:|---:|---:|
| Held-out task success | Not measured | Not measured | Not measured |
| Mean tool calls | Not measured | Not measured | Not measured |
| Invalid action rate | Not measured | Not measured | Not measured |
| Regression break rate | Not measured | Not measured | Not measured |

Results will be added only after the test repositories and evaluation protocol are frozen.
Uncertainty will use repository-cluster or hierarchical bootstrap intervals, not a naive
task-only bootstrap.

## Quick start

Prerequisites for the current foundation phase are Python 3.11 or 3.12 and
[`uv`](https://docs.astral.sh/uv/). Docker is required from the sandbox milestone onward.

```bash
uv sync --extra dev
uv run ruff check .
uv run mypy
uv run pytest
```

## Repository layout

```text
src/reporl/
  schemas.py             # Task, action, trajectory, and verifier contracts
  rewards/terminal.py    # Bounded executable-reward composition
docs/
  ARCHITECTURE.md         # Trust boundaries and component contracts
  EXPERIMENT_PROTOCOL.md  # Hypotheses, comparisons, metrics, and statistics
  PROJECT_PLAN.md         # Milestones, gates, resources, and risks
tests/
  unit/                   # Fast contract and reward tests
```

The agent harness, Docker sandbox, task builder, trainers, evaluator, and read-only demo are
added in that order. UI code is intentionally outside the critical path until evaluation
artifacts exist.

## Near-term milestones

1. Harden typed contracts and implement two deterministic fixture repositories.
2. Build separate agent and verifier Docker sandboxes with path, network, and resource limits.
3. Run a fixed-budget prompt-only baseline and audit its trajectories.
4. Generate validated tasks with repository-level splits and full provenance.
5. Train QLoRA SFT, then a 1.5B/3B cost-aware RL pilot.
6. Freeze evaluation, run ablations with at least three seeds, and publish the report.

See [the full project plan](docs/PROJECT_PLAN.md), [the architecture](docs/ARCHITECTURE.md),
and [the experiment protocol](docs/EXPERIMENT_PROTOCOL.md).

## Hardware scope

The current 8 GB laptop GPU is suitable for unit tests, sandbox development, API-backed
baselines, and limited quantized small-model inference. It is not treated as a credible 7B
online-GRPO machine. The first RL target is 1.5B/3B with short bounded trajectories; larger
runs require a measured cloud-GPU budget.

## License

MIT. Dataset records also preserve each source repository's license and provenance;
source-code licenses remain in force for generated tasks and patches.
