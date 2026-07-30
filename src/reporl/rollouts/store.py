"""Content-addressed artifacts and immutable one-file-per-trajectory storage."""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

from reporl.schemas import Trajectory


class ArtifactCollisionError(RuntimeError):
    pass


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, payload: bytes, *, suffix: str = "") -> tuple[str, Path]:
        if suffix and (not suffix.startswith(".") or "/" in suffix or "\\" in suffix):
            raise ValueError("artifact suffix must be empty or a simple extension")
        digest_value = hashlib.sha256(payload).hexdigest()
        digest = f"sha256:{digest_value}"
        target = self.root / digest_value[:2] / f"{digest_value}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != payload:
                raise ArtifactCollisionError(f"artifact digest collision at {target}")
            return digest, target
        temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                if target.read_bytes() != payload:
                    raise ArtifactCollisionError(
                        f"artifact digest collision at {target}"
                    ) from error
        finally:
            temporary.unlink(missing_ok=True)
        return digest, target

    def put_text(self, payload: str, *, suffix: str = ".txt") -> tuple[str, Path]:
        return self.put_bytes(payload.encode("utf-8"), suffix=suffix)

    def read_bytes(self, digest: str, *, suffix: str = "") -> bytes:
        digest_value = _parse_digest(digest)
        return (self.root / digest_value[:2] / f"{digest_value}{suffix}").read_bytes()


class TrajectoryStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, trajectory: Trajectory) -> Path:
        target = self.root / f"{trajectory.trajectory_id}.json"
        payload = (trajectory.model_dump_json() + "\n").encode("utf-8")
        if target.exists():
            if target.read_bytes() == payload:
                return target
            raise ArtifactCollisionError(
                f"trajectory ID {trajectory.trajectory_id!r} already stores different content"
            )
        temporary = self.root / (
            f".{trajectory.trajectory_id}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                raise ArtifactCollisionError(
                    f"trajectory ID {trajectory.trajectory_id!r} was written concurrently"
                ) from error
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def get(self, trajectory_id: str) -> Trajectory:
        if not trajectory_id or any(character in trajectory_id for character in "/\\\x00"):
            raise ValueError("invalid trajectory ID")
        return Trajectory.model_validate_json(
            (self.root / f"{trajectory_id}.json").read_text(encoding="utf-8")
        )

    def iter_all(self) -> tuple[Trajectory, ...]:
        return tuple(
            Trajectory.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.root.glob("*.json"))
        )


def _parse_digest(digest: str) -> str:
    if not digest.startswith("sha256:"):
        raise ValueError("artifact digest must use sha256")
    value = digest.removeprefix("sha256:")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("invalid sha256 digest")
    return value
