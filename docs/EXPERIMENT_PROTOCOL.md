# RepoRL Experiment Protocol

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-29
- Verification Status: IMPLEMENTED PROTOCOL; REAL EXPERIMENT UNRUN
- Version Label: experiment_protocol_v2

This protocol is a preregistration target. Values marked `pilot-estimated` must be fixed from
training and validation data before the sealed test evaluation. No Docker/GPU training run or
research metric has been completed as of 2026-07-30.

## Research question

Under fixed agent and inference budgets, does executable-reward post-training improve a small
open coding model's repository-level patch success and cost efficiency on repository lineages
held out from RepoRL post-training?

The defensible scope is deliberately narrower than "the model has never seen the repository."
Public code may have appeared in base-model pretraining. A separate post-model-release natural
issue set is required for a stronger contamination-resistant external-validity claim.

## Hypotheses

- **H1, SFT:** SFT improves valid tool use and held-out success over the prompt-only agent.
- **H2, RL:** Cost-aware RL improves `pass@1` over SFT and compute-matched continued SFT.
- **H3, cost:** Cost-aware RL reduces tools and generated tokens relative to shaped RL without
  cost, while meeting a preregistered success non-inferiority margin.
- **H4, shaping:** Potential-based progress shaping improves learning-curve area over outcome-only
  RL under the same environment-step and GPU-hour budget.
- **H5, warm start:** SFT plus RL outperforms direct RL from the base instruct model.
- **H6, credit assignment:** On long-horizon tasks, step-level credit assignment improves
  success-versus-environment-step area over terminal credit assignment. H6 is an optional v2
  hypothesis and does not block the core project.

## Primary estimand

The primary estimand is the paired absolute percentage-point difference in `pass@1` between the
main cost-aware RL condition and SFT on the same sealed synthetic tasks:

```text
E_repo,task[success(RL) - success(SFT)]
```

Success means one candidate patch, applied to a pristine task snapshot, passes all hidden target
tests, all fixed regression tests, and all patch-policy checks in a fresh verifier container.
Training reward is never an evaluation metric.

## Conditions

| ID | Condition | Purpose |
|---|---|---|
| D0 | Direct patch with fixed retrieval context | Descriptive, tool-free reference only |
| P0 | Base model plus fixed RepoRL harness | Primary prompt-only baseline |
| S0 | P0 model plus verified-trajectory LoRA SFT | Tool-learning baseline |
| SC | Continued/rejection-sampling SFT matched to RL compute or new data | Extra-compute control |
| R0 | S0 plus outcome-only GRPO | Sparse-reward baseline |
| R1 | S0 plus outcome and potential progress | Shaping ablation |
| R2 | S0 plus outcome, progress, and capped cost | Main method |
| R3 | Base model plus R2 reward, without SFT | Warm-start ablation |
| R4 | R2 with step-level credit assignment | Deferred, unimplemented Agent Lightning extension |

Headline comparisons are `R2-S0`, `R2-SC`, `R2-R1`, `R2-R0`, and `R2-R3`. `R4-R2` is reported
only if both use identical rollouts, reward components, and training compute except for credit
assignment.

## Fair-comparison budget

The following are identical across methods during evaluation:

- base model family and tokenizer, except for learned adapters/checkpoints;
- agent prompt, action schema, parser, tool gateway, sandbox, and verifier;
- maximum tool steps, generated policy tokens, tool-output characters, and wall time;
- model context limit, context truncation policy, temperature, top-p, and stop rules;
- task order, candidate count, seed schedule, and container resource limits.

`pass@k` is reported only when every condition samples at least `k` independent candidates under
the same total candidate and token budget. Otherwise the project reports `pass@1` only.

## Dataset design

### Split unit

Repositories are grouped by lineage before task generation. Fork ancestry, shared Git history,
mirrors, vendored copies, and high code-clone similarity are considered the same lineage. No
lineage may cross train, validation, synthetic test, or natural external test.

### Scale

| Stage | Train lineages/tasks | Validation lineages/tasks | Test lineages/tasks |
|---|---:|---:|---:|
| Infrastructure pilot | 3 / 60-100 | 1 / 20-30 | 2 / 40-60 |
| Core experiment | 8-12 / 300-600 | 2-4 / 60-120 | 5-8 / 150-320 |
| Strong final claim | 12+ / 600+ | 4+ / 120+ | 8+ / 320-600 |

Task count may be reduced when compute is constrained, but the number of independent test
lineages is preserved. The final test size is chosen by simulation using pilot paired
discordance and intra-repository correlation, not by a task-only power calculation.

Add a separate external-validity set of 50-100 natural issues from at least five repository
lineages. Prefer repositories or issue commits created after the selected base model's release.
Synthetic and natural results are always reported separately.

### Task classes

- local bug repair, normally one file and one function;
- failure-guided repair from public/focused test output;
- cross-file repair requiring coordinated changes;
- long-horizon subset defined before evaluation, for example at least two files or a minimum
  reference edit graph, used for H6 interaction analysis.

Difficulty labels derive from task construction and reference-patch structure, not the tested
model's outcome.

### Admission checks

Every generated task must satisfy all checks before it receives a split manifest:

1. The clean parent passes the fixed regression suite on at least three repeated runs.
2. The buggy snapshot reproducibly fails at least one canonical target test.
3. Non-target regression status is known and matches the task contract.
4. The reference repair passes target and regression suites on at least three repeated runs.
5. The issue text does not reveal the mutation operator, changed line, gold patch, or hidden test.
6. Agent image inspection finds no hidden test, `.git` history, gold patch, mutation metadata,
   pristine source copy, or informative Docker layer cache.
7. Repository license and task redistribution policy are recorded and approved.

Flaky or ambiguous tasks are quarantined rather than retried until they happen to pass.

### Sealing

Prompt wording, reward weights, checkpoint selection, early stopping, context policy, and all
hyperparameters use train/validation only. Test manifests are encrypted or access-controlled,
their aggregate hashes are published before evaluation, and the test is run once after the
configuration is frozen. Any rerun and reason are disclosed.

## Reward protocol

### Terminal success

```text
S = all_hidden_target_tests_pass
    and all_regression_tests_pass
    and patch_policy_passes
```

### Main scalarization

The initial validation candidate is:

```text
R = 1.00 S
  + 0.05 progress_potential_delta
  + 0.03 regression_pass_fraction
  + 0.02 valid_patch
  - 0.05 normalized_cost
  - 0.02 invalid_action_fraction
  - 0.02 budget_exhausted
  - 1.00 policy_violation
```

All fractions and costs are clipped. The exact weights are selected on validation and then
frozen. A programmatic invariant verifies that the minimum score for a policy-compliant success
is greater than the maximum score for a failure.

For intermediate progress, use a fixed canonical test-set potential:

```text
r_progress,t = alpha * (Phi(s_t) - Phi(s_t-1))
```

The sum telescopes, so repeatedly breaking and repairing the same tests cannot accumulate extra
reward. The agent never sees hidden test code or per-test hidden feedback. Public/focused tests
may provide observations; hidden tests remain terminal verifier signals.

Every trajectory stores both scalar reward and components. Infrastructure errors are excluded
from training. Agent-caused timeouts and resource exhaustion remain agent failures.

## Training protocol

### P0: prompt-only baseline

- Freeze the harness and inference budget before generating training trajectories.
- Run the base instruct model on the pilot tasks with deterministic seeds.
- Record parse success, invalid actions, test usage, patch validity, success, cost, and failures.
- Do not start SFT until end-to-end artifact completeness exceeds 99% and infrastructure-error
  classification has been manually audited on a sample.

### S0: SFT

- Generate candidates with a stronger teacher or repeated base-model sampling.
- Admit only verifier-passed, policy-compliant trajectories from train lineages.
- Deduplicate near-identical trajectories and cap overrepresented repositories/operators.
- Use the model's exact chat template and train only on assistant message and tool-call tokens.
- Do not compute loss on issue text, repository observations, or tool results.
- Select the checkpoint on validation success, then invalid-action rate, then cost.

### SC: compute-matched continued SFT

Use additional verified/rejection-sampled trajectories so the optimizer token count or GPU-hour
budget approximately matches R2. This tests whether RL gains come from reward optimization rather
than merely more post-training.

### R0-R3: GRPO pilot

- Start with a 1.5B/3B model, group size 4, 8-12 tool steps, and an 8k-16k episode cap.
- Collect full interactive episodes with the split CPU-sandbox/GPU-policy collector.
- Store exact prompt and generated token IDs, old-policy log probabilities, sampling parameters,
  policy identity and adapter hashes, group ID, reward components, and verifier hashes.
- Skip and report zero-variance groups; monitor their rate as a training-health metric.
- Run rollout and optimization sequentially first. Add asynchronous workers only after policy
  staleness is measured and bounded.
- Use the implemented external-rollout LoRA GRPO update only after the policy-identity checks,
  initial log-probability-ratio tolerance, and a real GPU smoke update pass. TRL is not a current
  dependency, and a text completion with one final reward is not treated as an interactive
  environment implementation.

### R4: step-level credit

This unimplemented extension begins only after R2 is stable. It reuses the same episodes and
reward components, changes only the credit allocator, and evaluates learning-curve area against
environment steps and GPU hours.

## Metrics

### Primary

- verifier-defined regression-safe `pass@1` on sealed held-out lineages;
- paired absolute percentage-point difference between R2 and S0.

### Secondary effectiveness

- per-repository macro success and task-level micro success;
- target-test pass fraction, regression break rate, and valid-patch rate;
- `pass@k` when sampling requirements are met;
- synthetic versus natural-issue success, reported separately.

### Efficiency

- tool calls, generated tokens, input tokens, wall time, test CPU seconds, and peak GPU memory;
- metrics across all tasks and separately among successful tasks;
- success-versus-environment-step and success-versus-GPU-hour learning-curve area.

### Reliability and safety

- invalid action rate, parser failures, policy violations, budget exhaustion;
- infrastructure error and quarantined/flaky task rate;
- GRPO zero-variance group rate and stale-policy rollout rate;
- forbidden-change and regression-break rate.

## Statistical analysis

- Run at least three independent training seeds for headline trained conditions.
- Keep evaluation paired: each method sees the same task and seed schedule.
- Compute two-level paired bootstrap intervals with at least 10,000 resamples: sample repository
  lineages with replacement, then tasks within sampled lineages, preserving method pairs.
- Report both micro and repository-macro estimates with 95% intervals.
- Report leave-one-repository-out sensitivity for the primary comparison.
- Optionally fit a logistic mixed model with method as a fixed effect and repository/task random
  intercepts as a robustness analysis.
- Report absolute percentage points and raw counts, not only relative improvement.
- Apply Holm correction to the preregistered family of secondary hypothesis tests; mark all other
  analyses exploratory.

For H3, predeclare a success non-inferiority margin from validation, initially no wider than five
percentage points, and require cost superiority. "No significant success decrease" is not
evidence of non-inferiority.

## Claim gates

The project may claim an RL success improvement only if:

- R2 exceeds both S0 and SC on the preregistered primary comparison;
- the paired repository-aware 95% interval excludes zero in the favorable direction;
- the result is not driven by one repository in leave-one-out analysis;
- all compared conditions have matching harness and inference budgets;
- infrastructure failures and test reruns are fully disclosed.

The project may claim a better success-cost tradeoff only if R2 meets the frozen success
non-inferiority margin against R1 and has a favorable repository-aware cost interval.

Acceptable final language is:

> Under the fixed RepoRL harness, budget, and task distribution, post-training improved hidden-test
> success on repository lineages held out from RepoRL training. Cost-aware reward improved the
> measured success-cost tradeoff.

The evidence does not by itself prove universal superiority of RL, absence of base-pretraining
contamination, or generalization to arbitrary GitHub issues.

## Reproducibility manifest

Each run publishes:

- Git commit and dirty-worktree status;
- model ID, exact revision, tokenizer revision, adapter hash, and chat template hash;
- container image digests, repository commits, task-manifest hash, and split-lineage hash;
- prompt, tool schemas, reward config, budgets, decoding config, and all random seeds;
- package lock, CUDA/driver/GPU details, CPU/RAM limits, and environment variables minus secrets;
- immutable trajectories, patches, verifier JSON, JUnit XML, training curves, and exclusions;
- bootstrap code, resample seed, result tables, and failure taxonomy.

## Planned result table

No value below is populated until the sealed evaluation.

| Condition | Micro pass@1 | Repo-macro pass@1 | Calls/task | Tokens/task | Regression breaks |
|---|---:|---:|---:|---:|---:|
| P0 | Not measured | Not measured | Not measured | Not measured | Not measured |
| S0 | Not measured | Not measured | Not measured | Not measured | Not measured |
| SC | Not measured | Not measured | Not measured | Not measured | Not measured |
| R0 | Not measured | Not measured | Not measured | Not measured | Not measured |
| R1 | Not measured | Not measured | Not measured | Not measured | Not measured |
| R2 | Not measured | Not measured | Not measured | Not measured | Not measured |
| R3 | Not measured | Not measured | Not measured | Not measured | Not measured |
