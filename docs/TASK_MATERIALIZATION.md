# Task materialization

RepoRL materializes trusted task data on the CPU sandbox worker. Runtime JSONL files contain
portable artifact-relative paths and can be transferred between hosts; the destination worker
selects the local artifact location through `task_artifacts_root`.

## SWE-smith boundary

SWE-smith's public `swesmith.harness.gather` output contains `instance_id`, `repo`, the mutation
`patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`, and `image_name`. An issue-generation pass may also add
`problem_statement`. This is not enough to create an admitted RepoRL task: it does not contain
content-addressed clean/buggy/reference snapshots, separated hidden tests, a repair patch, pinned
image digests, repeated executable admission evidence, or license and leakage review.

`reporl-materialize import-swe-smith` is therefore a strict importer, not a task generator. It
accepts only `reporl.swe-smith-export/v1` JSONL. Each row preserves the official gathered fields
and adds:

- `task` and repository-level split/lineage evidence;
- artifact-root-relative paths for the mutation patch, three snapshots, hidden tests, and repair
  patch;
- pinned agent and verifier image digests;
- separate public agent suites and hidden verifier suites with canonical expected test IDs;
- complete `AdmissionEvidence`, including three repeated executable outcomes, license approval,
  and leakage review.

Unknown schema versions and fields are rejected. The importer recomputes artifact hashes and the
admission result, checks the official mutation patch byte-for-byte, verifies `FAIL_TO_PASS` and
`PASS_TO_PASS` against canonical verifier IDs, audits repository-level split isolation, and emits
the generic sealed inputs. It does not claim to generate or execute the enrichment evidence.

The public source schema used for the adapter is documented in the
[SWE-smith harness guide](https://github.com/SWE-bench/SWE-smith/blob/9b74ac08118a85c39c356802f7961893af73e07f/docs/guides/harnesses.md).

## Real SWE-smith preparation on a CPU worker

`reporl-prepare-swe-smith` is the fail-closed bridge between official gather output and the
enriched v1 export. Run it on a Docker-capable CPU worker before renting a GPU. The command does
not run SWE-smith itself and does not invent missing trust evidence.

Prepare these inputs first:

- official gather JSON or JSONL produced from SWE-smith commit
  `9b74ac08118a85c39c356802f7961893af73e07f`;
- one clean checkout per task repository, with `HEAD` equal to the full 40-character base commit;
- strict `reporl.swe-smith-prepare/v1` JSONL with the agent-visible `TaskSpec`, repository identity,
  hidden-test paths, test-ID mappings, public and verifier commands, and pinned image references;
- a human `license_review` that records the reviewed license-file digest and explicit use and
  redistribution decisions;
- a completed `leak_scan` for the issue text and exact pinned agent image. The preparation command
  validates this declaration but cannot infer that the review happened;
- a verifier image that already contains the repository's test dependencies and GNU `timeout`.
  Pull and resolve the image to an immutable `name@sha256:...` reference before preparation.

Every verifier command must write its declared JUnit XML file below `/tmp/reporl-junit/`.
`target_test_id_map` and `regression_test_id_map` must provide a one-to-one mapping from every
official SWE-smith ID to every exact JUnit ID collected by the corresponding command.

```bash
reporl-prepare-swe-smith \
  --instances data/raw/swe-smith-gather.json \
  --preparation-specs data/trusted/preparation-specs.jsonl \
  --repositories-root /srv/reporl/checkouts \
  --output-root /srv/reporl/prepared/swe-smith-v1 \
  --repetitions 3
```

For each task, preparation performs the following checks and actions:

1. Reject a dirty checkout or a checkout at any commit other than the declared base commit.
2. Extract a sanitized Git archive, apply the official mutation, isolate the declared hidden tests,
   and remove those tests from clean, buggy, and reference snapshots.
3. Generate the reverse repair patch, enforce the task patch policy, and prove that applying it to
   the buggy snapshot reconstructs the reference snapshot byte-for-byte.
4. Run clean, buggy, and reference admission in fresh containers with networking disabled,
   read-only roots, writable tmpfs workspaces, dropped capabilities, resource limits, and exact
   JUnit collection checks. At least three independent repetitions are required.
5. Publish only after all tasks pass admission. The destination must not already exist; failures
   leave no partial output at that path.

The output contains `artifacts/`, `swe-smith-export-v1.jsonl`, and
`preparation-metadata.json`. Import it through the generic trust boundary:

```bash
reporl-materialize import-swe-smith \
  --export-jsonl /srv/reporl/prepared/swe-smith-v1/swe-smith-export-v1.jsonl \
  --artifact-root /srv/reporl/prepared/swe-smith-v1/artifacts \
  --output-dir /srv/reporl/prepared/imported \
  --dataset-id reporl-swe-smith \
  --dataset-version 1
```

Preparation is intentionally conservative. Unsupported repository objects, mutations that touch
hidden tests, non-reproducible failures, unexpected JUnit IDs, rejected licenses, or detected leaks
stop the whole publication instead of being downgraded to warnings.

## CPU-only contract check

The checked-in fixture builder creates three tiny Python repositories and a v1 enriched export.
It runs no Docker container and downloads no model:

```bash
reporl-materialize build-fixture --output-root .reporl/task-fixture

reporl-materialize import-swe-smith \
  --export-jsonl .reporl/task-fixture/swe-smith-export-v1.jsonl \
  --artifact-root .reporl/task-fixture/artifacts \
  --output-dir .reporl/task-fixture/imported \
  --dataset-id reporl-fixture \
  --dataset-version 1
```

For a real enriched export, replace only the paths and dataset identity. Materialize the imported
trust evidence:

```bash
reporl-materialize materialize \
  --manifests .reporl/task-fixture/imported/verifier-manifests.jsonl \
  --artifact-root .reporl/task-fixture/artifacts \
  --dataset-manifest .reporl/task-fixture/imported/dataset-manifest.json \
  --split-seal .reporl/task-fixture/imported/split-seal.json \
  --repositories .reporl/task-fixture/imported/repositories.jsonl \
  --admission-evidence .reporl/task-fixture/imported/admission-evidence.jsonl \
  --admission-results .reporl/task-fixture/imported/admission-results.jsonl \
  --output-dir artifacts/tasks

reporl-materialize verify \
  --runtime-dir artifacts/tasks \
  --artifact-root .reporl/task-fixture/artifacts
```

The output directory contains:

```text
train-runtimes.jsonl
validation-runtimes.jsonl
test-runtimes.jsonl
materialization-metadata.json
```

`materialization-metadata.json` binds the dataset manifest, global split seal and assignment,
repository record audit, per-split membership, task count, runtime file size, and runtime file
SHA-256. Save the metadata digest printed by the CLI.

## Rollout binding

Every rollout config must copy the following values from the materialization metadata:

```toml
task_artifacts_root = "/srv/reporl/task-bundle/artifacts"
expected_split = "train"
expected_dataset_manifest_sha256 = "sha256:..."
expected_split_seal_sha256 = "sha256:..."
expected_split_assignment_sha256 = "sha256:..."
expected_split_membership_sha256 = "sha256:..."
expected_repository_records_sha256 = "sha256:..."
expected_tasks_file_sha256 = "sha256:..."
```

The collector verifies these values and the selected runtime file before creating a run
directory. It then rebases snapshot and hidden-test paths beneath `task_artifacts_root` and checks
all artifact hashes again.

Do not rent a GPU while any real v1 enrichment is missing, any admission evidence was not produced
by actual repeated test execution, the lineage audit is not clean, or materialization verification
fails on the destination CPU worker. The fixture proves only the software contract, not real task
admission or research validity.
