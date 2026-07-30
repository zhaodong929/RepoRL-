"""Canonical serialization primitives for signed and sealed task metadata."""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json(value: object) -> str:
    """Return a stable UTF-8-independent JSON representation.

    Model fields are serialized in JSON mode before object keys are sorted. Lists and tuples
    retain their semantic order; models that represent sets sort those fields during validation.
    NaN and infinity are rejected because they are not portable JSON values.
    """

    payload = _jsonable(value)
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: object) -> str:
    """Hash canonical JSON and return a tagged digest."""

    encoded = canonical_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def artifact_sha256(path: Path) -> tuple[str, int]:
    """Hash a regular file or a symlink-free directory tree deterministically."""

    resolved = path.resolve(strict=True)
    if resolved.is_file():
        digest = hashlib.sha256()
        size = 0
        with resolved.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
                size += len(block)
        return f"sha256:{digest.hexdigest()}", size
    if not resolved.is_dir():
        raise ValueError(f"artifact is neither a regular file nor a directory: {resolved}")
    digest = hashlib.sha256()
    size = 0
    for entry in sorted(resolved.rglob("*")):
        if entry.is_symlink():
            raise ValueError(f"artifact trees must not contain symlinks: {entry}")
        relative = entry.relative_to(resolved).as_posix().encode("utf-8")
        kind = b"d" if entry.is_dir() else b"f"
        raw_mode = stat.S_IMODE(entry.stat().st_mode)
        normalized_mode = (
            (raw_mode & 0o077) | 0o700 if entry.is_dir() else (raw_mode & 0o111) | 0o600
        )
        digest.update(kind)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(normalized_mode.to_bytes(4, "big"))
        if entry.is_file():
            content_digest = hashlib.sha256()
            content_size = 0
            with entry.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    content_digest.update(block)
                    content_size += len(block)
            digest.update(content_size.to_bytes(8, "big"))
            digest.update(content_digest.digest())
            size += content_size
    return f"sha256:{digest.hexdigest()}", size
