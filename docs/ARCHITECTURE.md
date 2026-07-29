# RepoRL Architecture

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-29
- Verification Status: UNVERIFIED
- Version Label: architecture_v1

`UNVERIFIED` means the design has not yet passed the sandbox and end-to-end acceptance tests
defined below. It does not mean the document is speculative about its trust boundaries.

## System boundary

```text
                         trusted control plane
  +---------------------------------------------------------------+
  | Task loader -> AgentRunner -> PolicyBackend                   |
  |                    |                                          |
  |                    v                                          |
  |              ToolGateway + budgets                            |
  |                    |                                          |
  |                    v                                          |
  |             Agent sandbox (untrusted)                          |
  |                    | patch only                                |
  |                    v                                          |
  |             patch policy validation                           |
  |                    |                                          |
  |                    v                                          |
  |          pristine verifier sandbox + hidden tests              |
  |                    |                                          |
  |                    v                                          |
  | VerificationResult -> RewardVector -> TrajectoryStore          |
  +---------------------------------------------------------------+
```

The policy is untrusted. Repository contents, issue text, tool output, generated patches, and
test output are also untrusted. The controller validates every action and is the only component
that may advance the state machine.

## Core contracts

### TaskSpec

Agent-visible metadata contains:

- task ID, issue text, and post-training split;
- source repository, license, lineage group, and pinned base commit;
- exact agent image digest and task-generator version;
- allowed paths, forbidden globs, named test suites, and hard budgets.

The agent-visible task never contains hidden test paths, hidden test IDs, a reference patch,
the pristine source tree, or reward weights. Those fields live in a verifier-only manifest with
separate access controls.

### Action

Actions are a Pydantic discriminated union:

```text
SearchCode(query, path=".", max_results=50)
ReadFile(path, start_line=1, end_line=200)
ApplyPatch(unified_diff)
RunTests(suite="target" | "regression")
Finish(summary="")
```

`RunTests` accepts a suite alias, not shell text. The trusted task manifest maps the alias to an
argument vector. Commands execute without a shell. Output, runtime, file reads, patch bytes, and
result counts are bounded.

### PolicyBackend

```python
class PolicyBackend(Protocol):
    def act(self, context: AgentContext, tools: ToolSchemaSet) -> PolicyStep: ...
```

`PolicyStep` retains the raw assistant output, validated action, token IDs when available,
assistant-token mask, policy revision, token usage, latency, and optional old-policy log
probabilities. Tool observations are context but are never treated as policy-generated tokens.

### Sandbox

```python
class Sandbox(Protocol):
    def reset(self, task: TaskSpec) -> WorkspaceHandle: ...
    def execute(self, action: Action) -> ToolResult: ...
    def diff(self) -> PatchArtifact: ...
    def close(self) -> None: ...
```

The sandbox implementation may change, but the contract may not expose a generic host command.

### AgentRunner

```python
class AgentRunner:
    def run(
        self,
        task: TaskSpec,
        policy: PolicyBackend,
        sandbox: Sandbox,
        budget: TaskBudgets,
    ) -> Trajectory: ...
```

The runner owns step, token, tool-output, wall-clock, and patch-size budgets. It classifies all
terminal states and always closes the sandbox.

### Verifier

```python
class Verifier(Protocol):
    def verify(self, task_id: str, patch: PatchArtifact) -> VerificationResult: ...
```

Verification starts from a pristine snapshot, never from the agent workspace. The verifier:

1. verifies patch hash, size, file modes, paths, and forbidden files;
2. rejects absolute paths, traversal, symlink escape, submodules, and unexpected binaries;
3. applies the patch with a structured patch library or `git apply --check`;
4. mounts hidden tests read-only only after the patch is applied;
5. runs fixed target and regression argv under hard resource limits;
6. parses JUnit XML and fixed canonical test IDs, not model-visible stdout counts;
7. emits `pass`, `agent_failure`, or `infrastructure_error` with component evidence.

## Agent loop

```python
context = runner.start(task)
for step in range(task.budgets.max_steps):
    policy_step = policy.act(context, TOOL_SCHEMAS)
    action = parser.validate(policy_step.raw_action)
    result = tool_gateway.execute(action, sandbox, task)
    trajectory.append(policy_step, result)
    context = runner.advance(context, action, result)
    if action.kind == "finish":
        break
patch = sandbox.diff()
return runner.finalize(trajectory, patch)
```

An invalid action consumes a step and receives a structured error. Repeated invalid actions,
budget exhaustion, policy exceptions, and infrastructure exceptions have distinct terminal
reasons.

## Sandbox security profile

Both sandboxes require:

- Linux container, non-root user, read-only root filesystem where practical;
- no network, no Docker socket, no host PID or IPC namespace;
- explicit CPU, memory, process, disk, and wall-time limits;
- a disposable copy-on-write workspace with only the required repository snapshot;
- prebuilt dependencies addressed by immutable image digest;
- no credentials, host home directory, Git credential store, or cloud metadata access;
- capped stdout/stderr with complete overflow logs stored outside model context.

The agent image must additionally exclude `.git` history, hidden tests, reference patches,
mutation metadata, and pristine files that make a synthetic defect trivially reversible.

The verifier image must additionally use fixed commands, read-only tests, and a clean workspace.
No verifier result is accepted from an agent-created executable named `pytest` or an altered
test configuration.

## Patch policy

The first version rejects changes to the following unless a task explicitly grants an exception:

- tests, `conftest.py`, `pytest.ini`, and test-related `pyproject.toml` settings;
- dependency manifests and lock files;
- CI workflows, test entry points, `sitecustomize.py`, and startup hooks;
- symlinks, submodules, binaries, file-mode changes, and generated artifacts;
- `skip`, `xfail`, or collection-filtering changes.

Static hard-code heuristics are diagnostics, not decisive reward signals. Hidden tests, property
tests, input perturbations, and regression tests are the primary defenses.

## Trajectory and artifact model

The event log is append-only JSONL. Each event records:

- trajectory, task, policy, policy revision, config digest, and seed;
- validated action, bounded observation, timings, exit status, and token usage;
- assistant-token mask and old-policy log probability when used for RL;
- hashes for task image, repository snapshot, patch, and large external logs.

Large logs, patches, JUnit XML, and model checkpoints are content-addressed artifacts. A Parquet
summary supports analysis; it is derived and can be rebuilt from JSONL. Recommended layout:

```text
artifacts/<run_id>/
  run_manifest.json
  trajectories/events-*.jsonl
  summaries/episodes.parquet
  patches/<sha256>.diff
  verifier/<trajectory_id>/result.json
  verifier/<trajectory_id>/junit.xml
  logs/<sha256>.txt
```

Weights & Biases or MLflow may mirror metrics, but no external service is required to reproduce
an experiment.

## Failure taxonomy

| Class | Examples | Training treatment |
|---|---|---|
| Agent failure | Invalid action, wrong patch, test failure, agent-caused timeout | Valid negative signal |
| Policy failure | Invalid JSON, context overflow, model exception | Negative if attributable to policy |
| Safety violation | Forbidden file, traversal, test tampering | Strong negative signal |
| Infrastructure error | Container daemon failure, corrupted image, host outage | Exclude from training |
| Flaky task | Baseline or reference outcome changes across repeats | Quarantine task |

Agent-caused resource exhaustion is not reclassified as infrastructure failure.

## Module map

```text
src/reporl/
  schemas.py
  agent/{runner,policy,parser,prompts}.py
  tools/{gateway,search,read,patch,test}.py
  sandbox/{base,docker}.py
  tasks/{loader,validator,builder,lineage}.py
  verifier/{pipeline,policy,reward,junit}.py
  rollouts/{backend,collector,store}.py
  training/{sft,grpo}.py
  evaluation/{runner,metrics,bootstrap}.py
  cli.py
configs/{agent,data,reward,training,evaluation}/
fixtures/
tests/{unit,contract,integration,e2e}/
```

Only implemented modules are added. Placeholder modules must not imply a working capability.

## Training boundary

`RolloutBackend` decouples environment execution from optimization. TRL's standard
`GRPOTrainer` is not assumed to provide a multi-turn tool environment. The integration spike
must demonstrate that exact generated token IDs, old-policy log probabilities, policy versions,
and assistant-only loss masks survive the tool loop. Zero-variance reward groups are skipped and
reported.

Agent Lightning is a later `CreditAssigner` implementation. It is admitted only after terminal
reward RL is stable, so a step-credit comparison changes credit assignment rather than the
environment, policy, verifier, or task distribution.

## Architecture acceptance tests

- Hidden test and gold-patch byte signatures cannot be found from the agent container.
- Absolute, traversal, symlink, submodule, test, config, and oversized patch attacks are rejected.
- Network, Docker socket, host home, credential, fork bomb, memory bomb, and timeout probes fail
  closed.
- Clean, buggy, and reference states are reproduced at least three times for fixture tasks.
- A fake deterministic policy produces byte-identical replay metadata apart from timestamps.
- Infrastructure failures never enter a trainer batch.
- The same patch verified twice in the same image has the same structured result.
