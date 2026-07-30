from __future__ import annotations

from pathlib import Path

import pytest

from reporl.training.provenance import artifact_evidence, prepare_output_directory


def test_training_output_directory_refuses_existing_content(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "old-run.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        prepare_output_directory(output)


def test_training_output_directory_allows_explicit_resume(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "checkpoint").mkdir()

    prepare_output_directory(output, allow_nonempty=True)


def test_artifact_evidence_changes_with_input_content(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    first = artifact_evidence(source)

    source.write_text('{"changed":true}\n', encoding="utf-8")
    second = artifact_evidence(source)

    assert first.sha256 != second.sha256
    assert second.size_bytes == len(source.read_bytes())
