"""Fail-closed output handling and compact provenance evidence for training jobs."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import Field

from reporl.schemas import StrictModel
from reporl.tasks.canonical import artifact_sha256


class ArtifactEvidence(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


def artifact_evidence(path: Path) -> ArtifactEvidence:
    resolved = path.resolve(strict=True)
    digest, size_bytes = artifact_sha256(resolved)
    return ArtifactEvidence(path=str(path), sha256=digest, size_bytes=size_bytes)


def prepare_output_directory(path: Path, *, allow_nonempty: bool = False) -> None:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"training output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()) and not allow_nonempty:
        raise FileExistsError(f"training output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def git_state() -> dict[str, object]:
    def command(*args: str) -> str:
        result = subprocess.run(
            ("git", *args),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": command("rev-parse", "HEAD"),
            "dirty": bool(command("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": True}
