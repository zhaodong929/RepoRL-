from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from reporl.rollouts.config import RolloutCollectionConfig, RolloutTaskSpec
from reporl.schemas import DatasetSplit
from reporl.tasks.adapters import JsonlSealedManifestSource
from reporl.tasks.admission import AdmissionEvidence, TaskAdmissionResult
from reporl.tasks.dataset import DatasetManifest, SplitSeal
from reporl.tasks.fixture import build_materialization_fixture
from reporl.tasks.lineage import RepositoryRecord
from reporl.tasks.loader import load_json, load_jsonl
from reporl.tasks.materialize import (
    MaterializationPlan,
    main,
    materialize_source,
    verify_runtime_splits,
    write_runtime_splits,
)


def _plan(paths: dict[str, Path]) -> MaterializationPlan:
    return materialize_source(
        JsonlSealedManifestSource(paths["manifests"]),
        paths["artifact_root"],
        dataset_manifest=load_json(paths["dataset_manifest"], DatasetManifest),
        split_seal=load_json(paths["split_seal"], SplitSeal),
        repositories=load_jsonl(paths["repositories"], RepositoryRecord),
        admission_evidence=load_jsonl(paths["admission_evidence"], AdmissionEvidence),
        admission_results=load_jsonl(paths["admission_results"], TaskAdmissionResult),
    )


def test_cpu_fixture_materializes_and_verifies_portably(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture"
    paths = build_materialization_fixture(fixture_root)
    plan = _plan(paths)
    published = write_runtime_splits(tmp_path / "runtimes", plan)

    train_path = published.runtime_paths[DatasetSplit.TRAIN]
    train = load_jsonl(train_path, RolloutTaskSpec)
    assert len(train) == 1
    assert train[0].artifact_root == Path(".")
    assert train[0].agent.repository_snapshot == Path(train[0].sealed_manifest.buggy_snapshot.path)
    task_payload = json.loads(train[0].task.model_dump_json())
    assert not {"hidden_tests", "reference_patch", "test_suites"} & set(task_payload)

    copied_artifacts = tmp_path / "transferred" / "artifacts"
    shutil.copytree(paths["artifact_root"], copied_artifacts)
    counts, metadata = verify_runtime_splits(tmp_path / "runtimes", copied_artifacts)

    assert counts == {
        DatasetSplit.TRAIN: 1,
        DatasetSplit.VALIDATION: 1,
        DatasetSplit.TEST: 1,
    }
    assert metadata.lineage_audit.is_valid
    serialized = train_path.read_text(encoding="utf-8")
    assert str(paths["artifact_root"]) not in serialized
    assert str(copied_artifacts) not in serialized


def test_materialization_rejects_artifact_tampering(tmp_path: Path) -> None:
    paths = build_materialization_fixture(tmp_path / "fixture")
    buggy_file = (
        paths["artifact_root"]
        / "tasks"
        / "fixture-train-001"
        / "states"
        / "buggy"
        / "src"
        / "calculator.py"
    )
    buggy_file.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its digest"):
        _plan(paths)


def test_materialization_recomputes_admission_and_lineage_audit(tmp_path: Path) -> None:
    paths = build_materialization_fixture(tmp_path / "fixture")
    evidence = list(load_jsonl(paths["admission_evidence"], AdmissionEvidence))
    evidence[0] = evidence[0].model_copy(update={"required_repetitions": 4})

    with pytest.raises(ValueError, match="admission result does not match recomputed evidence"):
        materialize_source(
            JsonlSealedManifestSource(paths["manifests"]),
            paths["artifact_root"],
            dataset_manifest=load_json(paths["dataset_manifest"], DatasetManifest),
            split_seal=load_json(paths["split_seal"], SplitSeal),
            repositories=load_jsonl(paths["repositories"], RepositoryRecord),
            admission_evidence=tuple(evidence),
            admission_results=load_jsonl(paths["admission_results"], TaskAdmissionResult),
        )

    repositories = list(load_jsonl(paths["repositories"], RepositoryRecord))
    repositories[1] = repositories[1].model_copy(
        update={"lineage_group": repositories[0].lineage_group}
    )
    with pytest.raises(ValueError, match="sealed dataset audit failed"):
        materialize_source(
            JsonlSealedManifestSource(paths["manifests"]),
            paths["artifact_root"],
            dataset_manifest=load_json(paths["dataset_manifest"], DatasetManifest),
            split_seal=load_json(paths["split_seal"], SplitSeal),
            repositories=tuple(repositories),
            admission_evidence=load_jsonl(paths["admission_evidence"], AdmissionEvidence),
            admission_results=load_jsonl(paths["admission_results"], TaskAdmissionResult),
        )


def test_runtime_file_seal_detects_record_tampering(tmp_path: Path) -> None:
    paths = build_materialization_fixture(tmp_path / "fixture")
    published = write_runtime_splits(tmp_path / "runtimes", _plan(paths))
    train_path = published.runtime_paths[DatasetSplit.TRAIN]
    train_path.write_text(train_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from materialization metadata"):
        verify_runtime_splits(tmp_path / "runtimes", paths["artifact_root"])


def test_collection_config_rejects_wrong_split_runtime(tmp_path: Path) -> None:
    paths = build_materialization_fixture(tmp_path / "fixture")
    plan = _plan(paths)
    config = RolloutCollectionConfig(
        run_id="fixture-run",
        method="fixture",
        tasks_file=tmp_path / "validation-runtimes.jsonl",
        task_artifacts_root=paths["artifact_root"],
        artifacts_root=tmp_path / "rollouts",
        expected_split=DatasetSplit.TRAIN,
        expected_dataset_manifest_sha256=plan.context.manifest.digest(),
        expected_split_seal_sha256=plan.context.split_seal_sha256,
        expected_split_assignment_sha256=plan.context.seal.assignment_sha256,
        expected_split_membership_sha256=plan.context.membership_sha256(DatasetSplit.TRAIN),
        expected_repository_records_sha256=plan.context.repository_records_sha256,
        expected_tasks_file_sha256="sha256:" + "0" * 64,
    )

    with pytest.raises(ValueError, match="expected train"):
        config.validate_task_bindings(plan.splits[DatasetSplit.VALIDATION])


def test_materialization_cli_contract_without_docker(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture"
    imported_root = tmp_path / "imported"
    runtime_root = tmp_path / "runtimes"
    assert main(["build-fixture", "--output-root", str(fixture_root)]) == 0
    assert (
        main(
            [
                "import-swe-smith",
                "--export-jsonl",
                str(fixture_root / "swe-smith-export-v1.jsonl"),
                "--artifact-root",
                str(fixture_root / "artifacts"),
                "--output-dir",
                str(imported_root),
                "--dataset-id",
                "reporl-swe-smith-fixture",
                "--dataset-version",
                "1",
            ]
        )
        == 0
    )
    arguments = [
        "materialize",
        "--manifests",
        str(imported_root / "verifier-manifests.jsonl"),
        "--artifact-root",
        str(fixture_root / "artifacts"),
        "--dataset-manifest",
        str(imported_root / "dataset-manifest.json"),
        "--split-seal",
        str(imported_root / "split-seal.json"),
        "--repositories",
        str(imported_root / "repositories.jsonl"),
        "--admission-evidence",
        str(imported_root / "admission-evidence.jsonl"),
        "--admission-results",
        str(imported_root / "admission-results.jsonl"),
        "--output-dir",
        str(runtime_root),
    ]
    assert main(arguments) == 0
    assert (
        main(
            [
                "verify",
                "--runtime-dir",
                str(runtime_root),
                "--artifact-root",
                str(fixture_root / "artifacts"),
            ]
        )
        == 0
    )


def test_swe_smith_import_rejects_unknown_export_version(tmp_path: Path) -> None:
    paths = build_materialization_fixture(tmp_path / "fixture")
    export_path = paths["swe_smith_export"]
    payload = json.loads(export_path.read_text(encoding="utf-8").splitlines()[0])
    payload["schema_version"] = "reporl.swe-smith-export/v2"
    invalid = tmp_path / "unknown-version.jsonl"
    invalid.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    assert (
        main(
            [
                "import-swe-smith",
                "--export-jsonl",
                str(invalid),
                "--artifact-root",
                str(paths["artifact_root"]),
                "--output-dir",
                str(tmp_path / "imported"),
                "--dataset-id",
                "reporl-swe-smith-fixture",
                "--dataset-version",
                "1",
            ]
        )
        == 2
    )
