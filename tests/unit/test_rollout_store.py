from __future__ import annotations

from pathlib import Path

import pytest

from reporl.rollouts.store import ArtifactCollisionError, ArtifactStore, TrajectoryStore
from reporl.schemas import TerminationReason, Trajectory


def trajectory(patch: str = "") -> Trajectory:
    return Trajectory(
        trajectory_id="trajectory-1",
        task_id="task-1",
        policy_id="policy",
        policy_revision="revision",
        config_digest=f"sha256:{'a' * 64}",
        seed=0,
        events=(),
        termination_reason=TerminationReason.POLICY_ERROR,
        patch=patch,
    )


def test_artifact_store_is_content_addressed(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    first_digest, first_path = store.put_text("hello")
    second_digest, second_path = store.put_text("hello")
    assert first_digest == second_digest
    assert first_path == second_path
    assert store.read_bytes(first_digest, suffix=".txt") == b"hello"


def test_trajectory_store_is_idempotent_but_immutable(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path / "trajectories")
    original = trajectory()
    store.append(original)
    store.append(original)
    assert store.get("trajectory-1") == original

    with pytest.raises(ArtifactCollisionError):
        store.append(trajectory("changed"))
