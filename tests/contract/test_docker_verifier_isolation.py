from __future__ import annotations

import io
import re
import stat
import tarfile
from pathlib import Path
from typing import Any

import pytest

from reporl.sandbox.base import CommandSpec, PatchArtifact
from reporl.verifier.docker import DockerVerifierConfig, DockerVerifierFactory
from reporl.verifier.models import (
    FailureKind,
    VerifierRunSpec,
    VerifierStatus,
    VerifierSuiteSpec,
)
from reporl.verifier.pipeline import Verifier

PATCH = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -1 +1 @@
-a = 1
+a = 2
"""
XML = b"<testsuite><testcase classname='hidden.test_a' name='test_fix'/></testsuite>"


class ExecResult:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", code: int = 0) -> None:
        self.exit_code = code
        self.output = (stdout, stderr)


class FakeNotFoundError(Exception):
    status_code = 404


class FakeContainer:
    def __init__(
        self,
        role: str,
        *,
        evidence_payload: bytes = XML,
        evidence_member_type: bytes = tarfile.REGTYPE,
        evidence_stat_mode: int = stat.S_IFREG | 0o600,
        evidence_missing: bool = False,
    ) -> None:
        self.role = role
        self.removed = False
        self.paused = False
        self.commands: list[tuple[str, ...]] = []
        self.archives: list[tuple[str, bytes]] = []
        self.events: list[str] = []
        self.evidence_payload = evidence_payload
        self.evidence_member_type = evidence_member_type
        self.evidence_stat_mode = evidence_stat_mode
        self.evidence_missing = evidence_missing

    def exec_run(self, argv: tuple[str, ...], **_: Any) -> ExecResult:
        command = tuple(argv[3:])
        self.commands.append(command)
        if command[:3] == ("realpath", "-m", "--"):
            return ExecResult(b"/workspace/repo/src/a.py\n")
        if command[:3] == ("git", "ls-files", "--stage"):
            return ExecResult(b"100644 deadbeef 0\tsrc/a.py\n")
        if command[:3] == ("git", "diff", "--name-only"):
            return ExecResult(b"src/a.py\0")
        if command[:3] == ("stat", "-c", "%F"):
            return ExecResult(b"regular file\n")
        if command and command[0] == "pytest":
            return ExecResult(b"1 passed\n")
        return ExecResult()

    def put_archive(self, destination: str, archive: bytes) -> bool:
        self.archives.append((destination, archive))
        return bool(destination and archive)

    def pause(self) -> None:
        self.paused = True
        self.events.append("pause")

    def get_archive(
        self,
        path: str,
        *,
        chunk_size: int,
    ) -> tuple[tuple[bytes, ...], dict[str, Any]]:
        assert chunk_size > 0
        self.events.append("get_archive")
        if self.evidence_missing:
            raise FakeNotFoundError
        name = path.rsplit("/", maxsplit=1)[-1]
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            info = tarfile.TarInfo(name=name)
            info.type = self.evidence_member_type
            info.mode = 0o600
            if info.isreg():
                info.size = len(self.evidence_payload)
                archive.addfile(info, io.BytesIO(self.evidence_payload))
            else:
                info.linkname = "forged.xml"
                archive.addfile(info)
        metadata = {
            "name": name,
            "size": len(self.evidence_payload),
            "mode": self.evidence_stat_mode,
            "linkTarget": "",
        }
        return (buffer.getvalue(),), metadata

    def remove(self, *, force: bool) -> None:
        self.removed = force


class FakeContainers:
    def __init__(
        self,
        *,
        evidence_payload: bytes = XML,
        evidence_member_type: bytes = tarfile.REGTYPE,
        evidence_stat_mode: int = stat.S_IFREG | 0o600,
        evidence_missing: bool = False,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.created: list[FakeContainer] = []
        self.evidence_payload = evidence_payload
        self.evidence_member_type = evidence_member_type
        self.evidence_stat_mode = evidence_stat_mode
        self.evidence_missing = evidence_missing

    def run(self, image: str, command: tuple[str, ...], **kwargs: Any) -> FakeContainer:
        call = {"image": image, "command": command, **kwargs}
        self.calls.append(call)
        role = kwargs["labels"]["reporl.role"]
        container = FakeContainer(
            role,
            evidence_payload=self.evidence_payload,
            evidence_member_type=self.evidence_member_type,
            evidence_stat_mode=self.evidence_stat_mode,
            evidence_missing=self.evidence_missing,
        )
        self.created.append(container)
        return container


class FakeClient:
    def __init__(
        self,
        *,
        evidence_payload: bytes = XML,
        evidence_member_type: bytes = tarfile.REGTYPE,
        evidence_stat_mode: int = stat.S_IFREG | 0o600,
        evidence_missing: bool = False,
    ) -> None:
        self.containers = FakeContainers(
            evidence_payload=evidence_payload,
            evidence_member_type=evidence_member_type,
            evidence_stat_mode=evidence_stat_mode,
            evidence_missing=evidence_missing,
        )


def _archive_names(payload: bytes) -> tuple[str, ...]:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        return tuple(archive.getnames())


def _archive_members(payload: bytes) -> dict[str, tarfile.TarInfo]:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        return {member.name: member for member in archive.getmembers()}


def _single_suite_manifest(tmp_path: Path) -> VerifierRunSpec:
    snapshot = tmp_path / "snapshot"
    hidden = tmp_path / "hidden"
    snapshot.mkdir()
    hidden.mkdir()
    (snapshot / "src").mkdir()
    (snapshot / "src" / "a.py").write_text("a = 1\n", encoding="utf-8")
    (hidden / "test_hidden.py").write_text("def test_fix(): pass\n", encoding="utf-8")
    digest = f"sha256:{'a' * 64}"
    manifest = VerifierRunSpec(
        task_id="task-001",
        image=f"verifier@{digest}",
        image_digest=digest,
        repository_snapshot=snapshot,
        hidden_tests_path=hidden,
        suites=(
            VerifierSuiteSpec(
                name="target",
                command=CommandSpec(
                    argv=(
                        "pytest",
                        "-q",
                        "/verifier-tests",
                        "--junitxml=/tmp/reporl-junit/target.xml",
                    )
                ),
                junit_path="/tmp/reporl-junit/target.xml",
                expected_test_ids=("hidden.test_a::test_fix",),
            ),
        ),
        allowed_paths=("src",),
    )
    return manifest


def test_hidden_tests_are_absent_during_patch_validation(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    hidden = tmp_path / "hidden"
    snapshot.mkdir()
    hidden.mkdir()
    (snapshot / ".git").mkdir()
    (snapshot / ".git" / "config").write_text("secret history", encoding="utf-8")
    (snapshot / "src").mkdir()
    (snapshot / "src" / "a.py").write_text("a = 1\n", encoding="utf-8")
    (hidden / "test_hidden.py").write_text("def test_fix(): pass\n", encoding="utf-8")
    client = FakeClient()
    digest = f"sha256:{'a' * 64}"
    manifest = VerifierRunSpec(
        task_id="task-001",
        image=f"verifier@{digest}",
        image_digest=digest,
        repository_snapshot=snapshot,
        hidden_tests_path=hidden,
        suites=(
            VerifierSuiteSpec(
                name="target",
                command=CommandSpec(
                    argv=(
                        "pytest",
                        "-q",
                        "/verifier-tests",
                        "--junitxml=/tmp/reporl-junit/target.xml",
                    )
                ),
                junit_path="/tmp/reporl-junit/target.xml",
                expected_test_ids=("hidden.test_a::test_fix",),
            ),
        ),
        allowed_paths=("src",),
    )

    result = Verifier(DockerVerifierFactory(client=client)).verify(
        manifest, PatchArtifact(content=PATCH)
    )

    assert result.status == VerifierStatus.PASSED
    assert len(client.containers.calls) == 2
    patch_mounts = client.containers.calls[0]["volumes"]
    test_mounts = client.containers.calls[1]["volumes"]
    assert patch_mounts == {}
    assert test_mounts == {}
    validation_names = {
        name
        for _, payload in client.containers.created[0].archives
        for name in _archive_names(payload)
    }
    test_names = {
        name
        for _, payload in client.containers.created[1].archives
        for name in _archive_names(payload)
    }
    suite_container = client.containers.created[1]
    stage_payload = next(
        payload
        for destination, payload in suite_container.archives
        if destination == "/tmp"
        and any(
            re.fullmatch(r"\.reporl-suite-[0-9a-f]{32}", name) for name in _archive_names(payload)
        )
    )
    stage_members = _archive_members(stage_payload)
    stage_root = next(
        name for name in stage_members if re.fullmatch(r"\.reporl-suite-[0-9a-f]{32}", name)
    )
    hidden_payload = next(
        payload
        for destination, payload in suite_container.archives
        if destination == f"/tmp/{stage_root}" and "tests/test_hidden.py" in _archive_names(payload)
    )
    hidden_members = _archive_members(hidden_payload)
    assert "repo/src/a.py" in validation_names
    assert all(".git" not in name.split("/") for name in validation_names | test_names)
    assert "tests/test_hidden.py" not in validation_names
    assert "tests/test_hidden.py" in test_names
    assert (stage_members[stage_root].uid, stage_members[stage_root].gid) == (0, 0)
    assert stage_members[stage_root].mode & 0o777 == 0o555
    evidence = stage_members[f"{stage_root}/evidence"]
    assert (evidence.uid, evidence.gid) == (1000, 1000)
    assert evidence.mode & 0o777 == 0o700
    for name, expected_mode in (("tests", 0o555), ("tests/test_hidden.py", 0o444)):
        member = hidden_members[name]
        assert (member.uid, member.gid) == (0, 0)
        assert member.mode & 0o777 == expected_mode
    pytest_command = next(command for command in suite_container.commands if command[0] == "pytest")
    assert "/verifier-tests" not in " ".join(pytest_command)
    assert "/tmp/reporl-junit/target.xml" not in " ".join(pytest_command)
    assert "/tmp/.reporl-suite-" in " ".join(pytest_command)
    assert all(command[0] not in {"head", "mkdir"} for command in suite_container.commands)
    assert suite_container.paused
    assert suite_container.events == ["pause", "get_archive"]
    assert client.containers.created[0].removed
    assert suite_container.removed


def test_each_suite_uses_a_fresh_container_and_evidence_path(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    hidden = tmp_path / "hidden"
    snapshot.mkdir()
    hidden.mkdir()
    (snapshot / "src").mkdir()
    (snapshot / "src" / "a.py").write_text("a = 1\n", encoding="utf-8")
    (hidden / "test_hidden.py").write_text("def test_fix(): pass\n", encoding="utf-8")
    client = FakeClient()
    digest = f"sha256:{'a' * 64}"
    suites = tuple(
        VerifierSuiteSpec(
            name=name,
            command=CommandSpec(
                argv=(
                    "pytest",
                    "-q",
                    "/verifier-tests",
                    f"--junitxml=/tmp/reporl-junit/{name}.xml",
                )
            ),
            junit_path=f"/tmp/reporl-junit/{name}.xml",
            expected_test_ids=("hidden.test_a::test_fix",),
        )
        for name in ("target", "regression")
    )
    manifest = VerifierRunSpec(
        task_id="task-001",
        image=f"verifier@{digest}",
        image_digest=digest,
        repository_snapshot=snapshot,
        hidden_tests_path=hidden,
        suites=suites,
        allowed_paths=("src",),
    )

    result = Verifier(DockerVerifierFactory(client=client)).verify(
        manifest, PatchArtifact(content=PATCH)
    )

    assert result.status == VerifierStatus.PASSED
    assert len(client.containers.created) == 3
    target_container, regression_container = client.containers.created[1:]
    assert target_container is not regression_container
    assert target_container.removed and regression_container.removed
    target_command = next(
        command for command in target_container.commands if command[0] == "pytest"
    )
    regression_command = next(
        command for command in regression_container.commands if command[0] == "pytest"
    )
    target_stage = re.search(r"/tmp/\.reporl-suite-[0-9a-f]{32}", " ".join(target_command))
    regression_stage = re.search(r"/tmp/\.reporl-suite-[0-9a-f]{32}", " ".join(regression_command))
    assert target_stage is not None and regression_stage is not None
    assert target_stage.group() != regression_stage.group()
    assert "/tmp/reporl-junit/target.xml" not in " ".join(target_command)
    assert "/tmp/reporl-junit/regression.xml" not in " ".join(regression_command)
    assert all(call["volumes"] == {} for call in client.containers.calls)
    assert target_container.events == ["pause", "get_archive"]
    assert regression_container.events == ["pause", "get_archive"]


@pytest.mark.parametrize(
    ("evidence_kwargs", "max_junit_bytes", "detail"),
    [
        (
            {
                "evidence_member_type": tarfile.SYMTYPE,
                "evidence_stat_mode": stat.S_IFREG | 0o600,
            },
            10_000_000,
            "JUnit archive entry is not the expected regular file",
        ),
        (
            {"evidence_payload": b"x" * 65},
            64,
            "JUnit evidence exceeds the configured size limit",
        ),
    ],
)
def test_junit_archive_rejects_special_or_oversized_evidence(
    tmp_path: Path,
    evidence_kwargs: dict[str, Any],
    max_junit_bytes: int,
    detail: str,
) -> None:
    manifest = _single_suite_manifest(tmp_path)
    client = FakeClient(**evidence_kwargs)

    result = Verifier(
        DockerVerifierFactory(
            config=DockerVerifierConfig(max_junit_bytes=max_junit_bytes),
            client=client,
        )
    ).verify(manifest, PatchArtifact(content=PATCH))

    assert result.status == VerifierStatus.INFRASTRUCTURE_ERROR
    assert result.failure_kind == FailureKind.SANDBOX_INFRASTRUCTURE
    assert detail in result.detail
    assert client.containers.created[1].removed


def test_missing_junit_archive_is_reported_as_missing_evidence(tmp_path: Path) -> None:
    manifest = _single_suite_manifest(tmp_path)
    client = FakeClient(evidence_missing=True)

    result = Verifier(DockerVerifierFactory(client=client)).verify(
        manifest, PatchArtifact(content=PATCH)
    )

    assert result.status == VerifierStatus.INFRASTRUCTURE_ERROR
    assert result.failure_kind == FailureKind.VERIFIER_CONFIGURATION
    assert result.detail == "suite did not produce JUnit XML"
    assert client.containers.created[1].events == ["pause", "get_archive"]
    assert client.containers.created[1].removed
