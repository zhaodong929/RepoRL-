"""Repository-relative path validation and host containment helpers."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath


class UnsafePathError(ValueError):
    """A path is not safely contained by the repository root."""


def normalize_repo_path(path: str, *, allow_dot: bool = False) -> str:
    """Return a canonical POSIX repository path or reject it."""

    normalized = path.replace("\\", "/")
    if "\x00" in normalized:
        raise UnsafePathError("path contains a null byte")
    if normalized == ".":
        if allow_dot:
            return normalized
        raise UnsafePathError("path must identify a repository entry")
    if (
        not normalized
        or normalized.startswith("/")
        or normalized.startswith("//")
        or re.match(r"^[A-Za-z]:", normalized)
    ):
        raise UnsafePathError("path must be relative to the repository root")
    pure_path = PurePosixPath(normalized)
    if any(part in {"", ".", ".."} for part in pure_path.parts):
        raise UnsafePathError("path must be normalized and cannot traverse directories")
    return pure_path.as_posix()


def resolve_contained_path(
    repository_root: Path,
    path: str,
    *,
    allow_dot: bool = False,
    must_exist: bool = True,
) -> Path:
    """Resolve a path and reject symlinks that escape the repository root."""

    normalized = normalize_repo_path(path, allow_dot=allow_dot)
    root = repository_root.resolve(strict=True)
    candidate = (root / normalized).resolve(strict=must_exist)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise UnsafePathError("resolved path escapes the repository root") from error
    return candidate
