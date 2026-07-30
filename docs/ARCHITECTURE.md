# RepoRL Architecture

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-29
- Verification Status: LOCAL CONTRACTS VERIFIED; REAL DOCKER AND GPU E2E PENDING
- Version Label: architecture_v2

The core Python components and fake-Docker contract tests exist. This status does not mean the
system has completed a real Linux Docker adversarial run, an AutoDL rollout, model training, or a
sealed evaluation. Those execution gates remain open.

## Implementation status (2026-07-30)

Implemented in the repository:

- typed task, action, trajectory, verifier, reward, and training records;
- bounded agent tools, runner budgets and context compaction, local/remote policy adapters, and an
  authenticated policy server;
- Docker agent, admission, and verifier implementations with unit and injected-client contract
  tests;
- SWE-smith preparation/import, lineage sealing, task admission/materialization, rollout
  collection, immutable artifact storage, SFT preparation/training, external-rollout LoRA GRPO,
  and evaluation/bootstrap entry points;
- cloud preflight, transfer, policy-service, collection, SFT, GRPO, and evaluation configurations
  intended for a split CPU-Docker/GPU deployment.

Not yet demonstrated:

- real Docker admission and adversarial probes on the intended Linux worker;
- a model-driven repository rollout, prompt baseline, QLoRA checkpoint, GRPO update, or sealed
  evaluation on AutoDL or another GPU host;
- reported performance, throughput, VRAM, or cost measurements.

Agent Lightning and a dashboard are deliberately absent. They are deferred extensions, not
implemented capabilities.

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
    def act(
        self,
        messages: tuple[ChatMessage, ...],
        *,
        seed: int,
        timeout_seconds: float | None = None,
    ) -> PolicyStep: ...
```

`PolicyStep` retains raw assistant output, token usage, latency, and an optional generation trace
containing prompt IDs, generated token IDs, sampling settings, and old-policy log probabilities.
The runner records the separately validated action and tool result in `TrajectoryEvent`; SFT
conversion later derives assistant-only loss masks from the exact replayed conversation. Tool
observations are context but are never treated as policy-generated tokens.

### Sandbox

```python
class SandboxProtocol(Protocol):
    def reset(self, task: TaskSpec) -> None: ...
    def execute(
        self,
        action: Action,
        *,
        call_id: str,
        timeout_seconds: float | None = None,
    ) -> ToolResult: ...
    def diff(self, *, timeout_seconds: float | None = None) -> str: ...
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
        sandbox: SandboxProtocol,
        *,
        seed: int,
        trajectory_id: str | None = None,
    ) -> Trajectory: ...
```

The runner owns step, token, tool-output, wall-clock, and patch-size budgets. It classifies all
terminal states and always closes the sandbox.

### Verifier

```python
class Verifier:
    def verify(
        self,
        manifest: VerifierRunSpec,
        patch: PatchArtifact,
    ) -> VerificationResult: ...
```

Verification starts from a pristine snapshot, never from the agent workspace. The verifier:

1. verifies patch hash, size, file modes, paths, and forbidden files;
2. rejects absolute paths, traversal, symlink escape, submodules, and unexpected binaries;
3. applies the patch with `git apply --check`, records Git's resulting paths, and disposes the
   patch-validation container before any hidden tests are introduced;
4. creates a fresh suite container, reconstructs the pristine repository, and reapplies the exact
   approved patch;
5. uploads hidden tests through the Docker archive API into a random staging root under `/tmp`;
6. runs fixed target and regression argv under hard resource limits;
7. pauses the suite container, obtains the declared JUnit path with Docker `get_archive`, and
   validates Docker stat metadata plus a bounded one-member regular-file tar before parsing XML;
8. checks fixed canonical test IDs rather than trusting stdout counters, then emits `pass`,
   `agent_failure`, or `infrastructure_error` with structured evidence.

The staging root and hidden-test directories are owned by UID/GID 0 with mode `0555`; hidden
regular files are `0444`, or `0555` when source executable bits must be preserved. The sole
`evidence/` directory is owned by the configured non-root suite UID/GID with mode `0700`, so the
test process can create JUnit output without being able to rename or modify the hidden-test tree.
The container is paused before the daemon reads evidence, and JUnit size is bounded independently
of process stdout.

This is an integrity improvement, not a secrecy or authenticity boundary. Hidden tests and the
patched repository execute in the same suite container and normally in the same Python process.
The tested code can therefore read hidden-test files and can theoretically write forged JUnit XML
to the writable evidence directory. The random staging path reduces accidental collisions and
simple precomputed paths; it is not a cryptographic secret. Canonical test IDs, fresh containers,
patch policy, and regression checks raise the cost of forgery but do not prove JUnit provenance.

## Agent loop

```python
sandbox.reset(task)
try:
    messages = initial_messages(task)
    for step in range(task.budgets.max_steps):
        policy_step = policy.act(messages, seed=seed + step, timeout_seconds=remaining)
        action = parse_policy_action(policy_step.raw_output)
        result = sandbox.execute(action, call_id=call_id, timeout_seconds=remaining)
        events.append(TrajectoryEvent(policy_step, action, result))
        messages = append_tool_observation(messages, result)
        if action.kind == "finish":
            break
    patch = sandbox.diff(timeout_seconds=remaining)
finally:
    sandbox.close()
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
- capped stdout/stderr and bounded observations stored in trajectory artifacts.

The agent image must additionally exclude `.git` history, hidden tests, reference patches,
mutation metadata, and pristine files that make a synthetic defect trivially reversible.

The verifier uses fixed commands and a clean workspace. Hidden tests are absent from the
patch-validation container and are daemon-injected as root-owned, non-writable files only in the
fresh suite container. Patch policy rejects test configuration and startup-hook changes, but the
same-process limitation above still applies.

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
  agent/{environment,hf_policy,models,parser,policy,policy_server,prompts,remote_policy,runner}.py
  tools/{gateway,output,patch,paths}.py
  sandbox/{base,docker}.py
  tasks/{adapters,admission,admission_docker,canonical,dataset,fixture,lineage,loader,
         manifest,materialize,swe_smith_prepare}.py
  verifier/{base,docker,junit,models,pipeline}.py
  rewards/terminal.py
  rollouts/{collector,config,store}.py
  training/{config,grpo,math,prepare_sft,provenance,records,sft}.py
  evaluation/{bootstrap,metrics,report}.py
  cloud/preflight.py
cloud/scripts/
configs/*.toml
tests/{unit,contract}/
```

This map lists current files, not a target layout. There is no dashboard package, Agent Lightning
adapter, distributed scheduler, or real-Docker integration/e2e test suite in the repository.

## Training boundary

The rollout collector runs the interactive environment separately from optimization. A remote
policy endpoint returns prompt IDs, generated IDs, old-policy log probabilities, sampling
parameters, and a full policy identity; immutable GRPO groups bind those traces to the exact
behavior-policy adapter. The current trainer is a custom single-GPU LoRA GRPO update over these
externally collected groups, not TRL `GRPOTrainer` and not Agent Lightning. It skips and reports
zero-variance groups and rejects stale policy identities. Unit tests cover the record and math
contracts, but no real GPU update has run yet.

Agent Lightning remains a future credit-assignment experiment. It is admitted only after terminal
reward RL is stable, so a later comparison changes credit assignment rather than the environment,
policy, verifier, or task distribution.

## Architecture acceptance tests

The following remain release gates, not claims that all have passed:

- Hidden test and gold-patch byte signatures cannot be found from the agent container.
- Absolute, traversal, symlink, submodule, test, config, and oversized patch attacks are rejected.
- Network, Docker socket, host home, credential, fork bomb, memory bomb, and timeout probes fail
  closed.
- Clean, buggy, and reference states are reproduced at least three times for fixture tasks.
- A fake deterministic policy produces byte-identical replay metadata apart from timestamps.
- Infrastructure failures never enter a trainer batch.
- The same patch verified twice in the same image has the same structured result.

Current local coverage exercises schemas, policies, tools, reward math, data sealing,
materialization, collector accounting, training records, JUnit parsing, verifier classification,
and injected-client Docker contracts. The verifier contract asserts root-owned staging,
pause-before-archive ordering, and rejection of symlink or oversized JUnit evidence. A real Docker
adversarial canary exists behind `REPORL_RUN_DOCKER_TESTS=1`, but it has not been run on this
machine. Linux Docker, real task images, repeated admission, remote-policy rollout, and GPU
training/evaluation gates remain pending.
