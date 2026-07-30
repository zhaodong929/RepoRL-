from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reporl.schemas import DatasetSplit, TaskProvenance, TaskSpec
from reporl.tasks import (
    AdmissionEvidence,
    AdmissionFailure,
    AgentTestSuite,
    ArtifactReference,
    DatasetManifest,
    DatasetTaskEntry,
    LeakScan,
    LicenseReview,
    LineageConflictKind,
    RegressionExpectation,
    RepositoryRecord,
    SealedSplitIssue,
    SnapshotKind,
    SnapshotValidation,
    SplitSeal,
    TaskDataError,
    TrustedCommand,
    VerifierManifest,
    VerifierTestSuite,
    audit_repository_splits,
    audit_sealed_split,
    canonical_json,
    load_task_specs_json,
    load_task_specs_jsonl,
    load_verifier_manifests_json,
    normalize_source_url,
    seal_dataset_manifest,
    validate_task_admission,
)
from reporl.tasks import TestRunOutcome as RunOutcome
from reporl.tasks.canonical import artifact_sha256


def digest(character: str) -> str:
    return f"sha256:{character * 64}"


def task_spec(*, task_id: str = "task-001", split: DatasetSplit = DatasetSplit.TRAIN) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        issue="Return the expected value for negative inputs.",
        split=split,
        agent_image_digest=digest("a"),
        provenance=TaskProvenance(
            source_repository="https://github.com/Example/Widget.git",
            source_license="MIT",
            base_commit="abcdef0",
            lineage_group="widget-family",
            generator="fixture",
            generator_version="1.0",
        ),
    )


def artifact(path: str, character: str) -> ArtifactReference:
    return ArtifactReference(path=path, sha256=digest(character), size_bytes=10)


def verifier_manifest() -> VerifierManifest:
    agent_suites = tuple(
        AgentTestSuite(
            name=name,
            command=TrustedCommand(argv=("python", "-m", "pytest", f"tests/{name}")),
        )
        for name in ("target", "regression")
    )
    suites = tuple(
        VerifierTestSuite(
            name=name,
            command=TrustedCommand(
                argv=(
                    "python",
                    "-m",
                    "pytest",
                    f"tests/{name}",
                    f"--junitxml=/tmp/reporl-junit/{name}.xml",
                )
            ),
            junit_path=f"/tmp/reporl-junit/{name}.xml",
            expected_test_ids=(f"tests.{name}::test_case",),
        )
        for name in ("target", "regression")
    )
    return VerifierManifest(
        task=task_spec(),
        verifier_image_digest=digest("b"),
        clean_snapshot=artifact("states/clean.tar.zst", "c"),
        buggy_snapshot=artifact("states/buggy.tar.zst", "d"),
        reference_snapshot=artifact("states/reference.tar.zst", "e"),
        hidden_tests=artifact("sealed/hidden-tests.tar.zst", "f"),
        reference_patch=artifact("sealed/reference.patch", "1"),
        agent_test_suites=agent_suites,
        test_suites=suites,
        admission_result_sha256=digest("2"),
    )


def outcome(suite: str, attempt: int, *, passes: bool) -> RunOutcome:
    return RunOutcome(
        suite=suite,  # type: ignore[arg-type]
        attempt=attempt,
        exit_code=0 if passes else 1,
        passed=3 if passes else 2,
        failed=0 if passes else 1,
        errors=0,
        duration_ms=10,
        report_sha256=digest(str(attempt)),
    )


def repeated(suite: str, *, passes: bool, count: int = 3) -> tuple[RunOutcome, ...]:
    return tuple(outcome(suite, attempt, passes=passes) for attempt in range(1, count + 1))


def admission_evidence() -> AdmissionEvidence:
    return AdmissionEvidence(
        task=task_spec(),
        clean=SnapshotValidation(
            kind=SnapshotKind.CLEAN,
            snapshot_sha256=digest("c"),
            outcomes=repeated("regression", passes=True),
        ),
        buggy=SnapshotValidation(
            kind=SnapshotKind.BUGGY,
            snapshot_sha256=digest("d"),
            outcomes=repeated("target", passes=False) + repeated("regression", passes=True),
        ),
        reference=SnapshotValidation(
            kind=SnapshotKind.REFERENCE,
            snapshot_sha256=digest("c"),
            outcomes=repeated("target", passes=True) + repeated("regression", passes=True),
        ),
        buggy_regression_expectation=RegressionExpectation.PASSES,
        license_review=LicenseReview(
            repository_url="git@github.com:example/widget.git",
            spdx_identifier="MIT",
            license_file_sha256=digest("9"),
            use_approved=True,
            redistribution_approved=True,
            reviewed_by="dataset-owner",
        ),
        leak_scan=LeakScan(
            issue_text_clean=True,
            agent_image_clean=True,
            agent_image_digest=digest("a"),
        ),
    )


def repository(
    repository_id: str,
    split: DatasetSplit,
    *,
    url: str,
    lineage: str,
    commits: tuple[str, ...] = (),
    fingerprints: tuple[str, ...] = (),
) -> RepositoryRecord:
    return RepositoryRecord(
        repository_id=repository_id,
        source_url=url,
        lineage_group=lineage,
        split=split,
        commits=commits,
        content_fingerprints=fingerprints,
    )


def dataset_entry(task_id: str, repository_record: RepositoryRecord) -> DatasetTaskEntry:
    return DatasetTaskEntry(
        task_id=task_id,
        split=repository_record.split,
        repository_id=repository_record.repository_id,
        source_url=repository_record.source_url,
        lineage_group=repository_record.lineage_group,
        verifier_manifest_sha256=digest("4"),
        admission_result_sha256=digest("5"),
    )


def test_verifier_manifest_declassifies_only_task_spec() -> None:
    manifest = verifier_manifest()

    assert manifest.agent_view() == manifest.task
    payload = json.loads(manifest.agent_json())
    assert payload["task_id"] == "task-001"
    assert "hidden_tests" not in payload
    assert "reference_patch" not in payload
    assert "test_suites" not in payload


def test_verifier_manifest_requires_exact_named_suites() -> None:
    payload = verifier_manifest().model_dump()
    payload["test_suites"] = payload["test_suites"][:1]

    with pytest.raises(ValidationError, match="exactly match"):
        VerifierManifest.model_validate(payload)


def test_trusted_command_rejects_shell_like_empty_or_escaped_paths() -> None:
    with pytest.raises(ValidationError):
        TrustedCommand(argv=("",))
    with pytest.raises(ValidationError):
        TrustedCommand(argv=("pytest",), cwd="../outside")


def test_canonical_json_and_manifest_hash_are_order_independent_for_keys() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert verifier_manifest().digest() == verifier_manifest().digest()


def test_json_loader_rejects_duplicate_and_unknown_fields(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"task_id":"one","task_id":"two"}', encoding="utf-8")
    with pytest.raises(TaskDataError, match="duplicate JSON object key"):
        load_task_specs_json(duplicate)

    unknown = tmp_path / "unknown.json"
    payload = task_spec().model_dump(mode="json")
    payload["unexpected"] = True
    unknown.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TaskDataError, match="schema validation failed"):
        load_task_specs_json(unknown)


def test_json_loader_rejects_type_coercion(tmp_path: Path) -> None:
    path = tmp_path / "coercion.json"
    payload = task_spec().model_dump(mode="json")
    payload["budgets"]["max_steps"] = "20"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TaskDataError, match="schema validation failed"):
        load_task_specs_json(path)


def test_json_loader_rejects_nonstandard_constants(tmp_path: Path) -> None:
    path = tmp_path / "nan.json"
    path.write_text('{"value":NaN}', encoding="utf-8")

    with pytest.raises(TaskDataError, match="non-standard JSON constant"):
        load_task_specs_json(path)


def test_verifier_manifest_json_loader_round_trips_strict_model(tmp_path: Path) -> None:
    path = tmp_path / "verifier.json"
    manifest = verifier_manifest()
    path.write_text(manifest.model_dump_json(), encoding="utf-8")

    assert load_verifier_manifests_json(path) == manifest


def test_jsonl_loader_rejects_blank_lines_and_duplicate_task_ids(tmp_path: Path) -> None:
    record = task_spec().model_dump_json()
    blank = tmp_path / "blank.jsonl"
    blank.write_text(f"{record}\n\n{record}\n", encoding="utf-8")
    with pytest.raises(TaskDataError, match="blank JSONL record"):
        load_task_specs_jsonl(blank)

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(f"{record}\n{record}\n", encoding="utf-8")
    with pytest.raises(TaskDataError, match="duplicate task_id"):
        load_task_specs_jsonl(duplicate)


def test_admission_accepts_reproducible_three_state_evidence() -> None:
    result = validate_task_admission(admission_evidence())

    assert result.admitted is True
    assert result.failures == ()
    assert result.evidence_sha256.startswith("sha256:")


def test_admission_reports_independent_failures_without_retrying() -> None:
    evidence = admission_evidence()
    bad_reference = evidence.reference.model_copy(
        update={"outcomes": repeated("target", passes=False, count=2)}
    )
    bad_license = evidence.license_review.model_copy(
        update={"use_approved": False, "redistribution_approved": False}
    )
    bad_scan = LeakScan(
        issue_text_clean=False,
        agent_image_clean=True,
        agent_image_digest=digest("b"),
        findings=("issue reveals mutation",),
    )
    evidence = evidence.model_copy(
        update={
            "reference": bad_reference,
            "license_review": bad_license,
            "leak_scan": bad_scan,
        }
    )

    result = validate_task_admission(evidence)

    assert result.admitted is False
    assert AdmissionFailure.INSUFFICIENT_REPETITIONS in result.failures
    assert AdmissionFailure.REFERENCE_TARGET_FAILED in result.failures
    assert AdmissionFailure.REFERENCE_REGRESSION_FAILED in result.failures
    assert AdmissionFailure.LICENSE_NOT_APPROVED in result.failures
    assert AdmissionFailure.REDISTRIBUTION_NOT_APPROVED in result.failures
    assert AdmissionFailure.LEAK_DETECTED in result.failures
    assert AdmissionFailure.AGENT_IMAGE_MISMATCH in result.failures


def test_admission_rejects_flaky_buggy_target_and_infrastructure_errors() -> None:
    evidence = admission_evidence()
    flaky = repeated("target", passes=False)[:2] + (outcome("target", 3, passes=True),)
    regression = list(repeated("regression", passes=True))
    regression[1] = regression[1].model_copy(update={"infrastructure_error": True})
    buggy = evidence.buggy.model_copy(update={"outcomes": flaky + tuple(regression)})

    result = validate_task_admission(evidence.model_copy(update={"buggy": buggy}))

    assert AdmissionFailure.BUGGY_TARGET_NOT_REPRODUCIBLE in result.failures
    assert AdmissionFailure.INFRASTRUCTURE_ERROR in result.failures
    assert AdmissionFailure.BUGGY_REGRESSION_MISMATCH in result.failures


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/Example/Widget.git/",
        "ssh://git@github.com/example/widget.git",
        "git@github.com:EXAMPLE/WIDGET.git",
        "github.com/example/widget",
    ],
)
def test_source_url_normalization_collapses_transport_and_case(url: str) -> None:
    assert normalize_source_url(url) == "github.com/example/widget"


def test_lineage_audit_finds_direct_url_commit_and_fingerprint_overlap() -> None:
    shared_commit = "a" * 40
    shared_fingerprint = digest("f")
    records = (
        repository(
            "repo-a",
            DatasetSplit.TRAIN,
            url="https://github.com/org/project.git",
            lineage="family-one",
            commits=(shared_commit,),
            fingerprints=(shared_fingerprint,),
        ),
        repository(
            "repo-b",
            DatasetSplit.TEST,
            url="git@github.com:ORG/PROJECT.git",
            lineage="family-two",
            commits=(shared_commit,),
            fingerprints=(shared_fingerprint,),
        ),
        repository(
            "repo-c",
            DatasetSplit.VALIDATION,
            url="https://gitlab.com/org/fork",
            lineage="family-one",
        ),
    )

    report = audit_repository_splits(records)

    assert report.is_clean is False
    assert {conflict.kind for conflict in report.conflicts} == set(LineageConflictKind)


def test_lineage_fingerprints_can_be_disabled_explicitly() -> None:
    records = (
        repository(
            "repo-a",
            DatasetSplit.TRAIN,
            url="https://github.com/org/a",
            lineage="a",
            fingerprints=(digest("f"),),
        ),
        repository(
            "repo-b",
            DatasetSplit.TEST,
            url="https://github.com/org/b",
            lineage="b",
            fingerprints=(digest("f"),),
        ),
    )

    report = audit_repository_splits(records, check_fingerprints=False)

    assert report.is_clean is True
    assert report.fingerprints_checked is False


def test_lineage_audit_handles_duplicate_repository_id_across_splits() -> None:
    records = (
        repository(
            "repo-a",
            DatasetSplit.TRAIN,
            url="https://github.com/org/a",
            lineage="a",
        ),
        repository(
            "repo-a",
            DatasetSplit.TEST,
            url="https://github.com/org/a",
            lineage="a",
        ),
    )

    report = audit_repository_splits(records)

    assert report.is_clean is False
    assert report.conflicts[0].repository_ids == ("repo-a",)
    assert report.records_sha256 == audit_repository_splits(records[::-1]).records_sha256


def test_dataset_manifest_hash_is_stable_under_input_order() -> None:
    train = repository(
        "repo-train",
        DatasetSplit.TRAIN,
        url="https://github.com/org/train",
        lineage="train-family",
    )
    test = repository(
        "repo-test",
        DatasetSplit.TEST,
        url="https://github.com/org/test",
        lineage="test-family",
    )
    entries = (dataset_entry("task-002", test), dataset_entry("task-001", train))

    first = DatasetManifest(dataset_id="reporl-pilot", version="1", tasks=entries)
    second = DatasetManifest(dataset_id="reporl-pilot", version="1", tasks=entries[::-1])

    assert tuple(task.task_id for task in first.tasks) == ("task-001", "task-002")
    assert first.digest() == second.digest()
    assert seal_dataset_manifest(first) == seal_dataset_manifest(second)


def test_sealed_split_audit_accepts_clean_untampered_manifest() -> None:
    train = repository(
        "repo-train",
        DatasetSplit.TRAIN,
        url="https://github.com/org/train",
        lineage="train-family",
        commits=("a" * 40,),
    )
    test = repository(
        "repo-test",
        DatasetSplit.TEST,
        url="https://github.com/org/test",
        lineage="test-family",
        commits=("b" * 40,),
    )
    manifest = DatasetManifest(
        dataset_id="reporl-pilot",
        version="1",
        tasks=(dataset_entry("task-001", train), dataset_entry("task-002", test)),
    )

    audit = audit_sealed_split(manifest, seal_dataset_manifest(manifest), (train, test))

    assert audit.is_valid is True
    assert audit.issues == ()
    assert audit.lineage_audit.is_clean is True


def test_sealed_split_audit_detects_tamper_missing_repository_and_lineage_leak() -> None:
    train = repository(
        "repo-train",
        DatasetSplit.TRAIN,
        url="https://github.com/org/train",
        lineage="shared-family",
    )
    test = repository(
        "repo-test",
        DatasetSplit.TEST,
        url="https://github.com/org/test",
        lineage="shared-family",
    )
    manifest = DatasetManifest(
        dataset_id="reporl-pilot",
        version="1",
        tasks=(dataset_entry("task-001", train), dataset_entry("task-002", test)),
    )
    seal = seal_dataset_manifest(manifest)
    tampered = SplitSeal(
        dataset_id=seal.dataset_id,
        dataset_version=seal.dataset_version,
        manifest_sha256=digest("0"),
        assignment_sha256=seal.assignment_sha256,
        split_digests=seal.split_digests,
    )

    audit = audit_sealed_split(manifest, tampered, (train,))

    assert audit.is_valid is False
    assert SealedSplitIssue.MANIFEST_HASH_MISMATCH in audit.issues
    assert SealedSplitIssue.REPOSITORY_MISSING in audit.issues

    leaked = audit_sealed_split(manifest, seal, (train, test))
    assert SealedSplitIssue.LINEAGE_LEAKAGE in leaked.issues


def test_models_are_frozen_and_reject_unknown_fields() -> None:
    record = repository(
        "repo-a",
        DatasetSplit.TRAIN,
        url="https://github.com/org/a",
        lineage="a",
    )
    with pytest.raises(ValidationError):
        RepositoryRecord.model_validate({**record.model_dump(), "unknown": True})
    with pytest.raises(ValidationError):
        record.split = DatasetSplit.TEST


def test_directory_artifact_hash_frames_file_content(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    embedded_header = b"f" + (1).to_bytes(8, "big") + b"b"
    (first / "a").write_bytes(b"X" + embedded_header + b"Y")
    (second / "a").write_bytes(b"X")
    (second / "b").write_bytes(b"Y")

    first_digest, _ = artifact_sha256(first)
    second_digest, _ = artifact_sha256(second)

    assert first_digest != second_digest
