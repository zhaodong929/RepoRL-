"""Strict JSON and JSONL loading for task trust boundaries."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from reporl.schemas import TaskSpec
from reporl.tasks.manifest import VerifierManifest

ModelT = TypeVar("ModelT", bound=BaseModel)


class TaskDataError(ValueError):
    """Raised when a task data file is malformed or violates its schema."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TaskDataError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise TaskDataError(f"non-standard JSON constant: {value}")


def _decode(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise TaskDataError(f"cannot read UTF-8 task data from {path}: {error}") from error


def _preflight_json(raw: str, source: str) -> None:
    try:
        json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except TaskDataError:
        raise
    except (json.JSONDecodeError, ValueError) as error:
        raise TaskDataError(f"invalid JSON in {source}: {error}") from error


def _validate_json(raw: str, model_type: type[ModelT], source: str) -> ModelT:
    _preflight_json(raw, source)
    try:
        return model_type.model_validate_json(raw, strict=True)
    except ValidationError as error:
        raise TaskDataError(f"schema validation failed for {source}: {error}") from error


def load_json(path: str | Path, model_type: type[ModelT]) -> ModelT:
    """Load exactly one strict Pydantic model from a JSON file."""

    source_path = Path(path)
    return _validate_json(_decode(source_path), model_type, str(source_path))


def load_jsonl(path: str | Path, model_type: type[ModelT]) -> tuple[ModelT, ...]:
    """Load strict models from non-empty JSONL records.

    Blank records are rejected so accidental truncation or line insertion changes cannot be
    silently ignored. A final newline is accepted by ``splitlines``.
    """

    source_path = Path(path)
    raw = _decode(source_path)
    if not raw:
        raise TaskDataError(f"JSONL file is empty: {source_path}")

    records: list[ModelT] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise TaskDataError(f"blank JSONL record at {source_path}:{line_number}")
        records.append(_validate_json(line, model_type, f"{source_path}:{line_number}"))
    return tuple(records)


def _unique_by(
    records: tuple[ModelT, ...], key: Callable[[ModelT], str], label: str
) -> tuple[ModelT, ...]:
    values = tuple(key(record) for record in records)
    if len(values) != len(set(values)):
        raise TaskDataError(f"duplicate {label} in JSONL data")
    return records


def load_task_specs_json(path: str | Path) -> TaskSpec:
    return load_json(path, TaskSpec)


def load_task_specs_jsonl(path: str | Path) -> tuple[TaskSpec, ...]:
    return _unique_by(load_jsonl(path, TaskSpec), lambda task: task.task_id, "task_id")


def load_verifier_manifests_json(path: str | Path) -> VerifierManifest:
    return load_json(path, VerifierManifest)


def load_verifier_manifests_jsonl(path: str | Path) -> tuple[VerifierManifest, ...]:
    return _unique_by(
        load_jsonl(path, VerifierManifest), lambda manifest: manifest.task.task_id, "task_id"
    )
