from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path
from typing import Any

import pytest

from reporl.verifier.docker import _readonly_directory_archive, _suite_staging_archive

pytestmark = pytest.mark.skipif(
    os.environ.get("REPORL_RUN_DOCKER_TESTS") != "1",
    reason="set REPORL_RUN_DOCKER_TESTS=1 to run the real Docker adversarial canary",
)


def _archive_file(container: Any, path: str) -> tuple[bytes, dict[str, Any]]:
    stream, metadata = container.get_archive(path, chunk_size=1024 * 1024)
    payload = b"".join(stream)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        members = archive.getmembers()
        assert len(members) == 1
        member = members[0]
        assert member.isreg()
        extracted = archive.extractfile(member)
        assert extracted is not None
        return extracted.read(), metadata


def test_non_root_suite_cannot_mutate_root_owned_hidden_tests(tmp_path: Path) -> None:
    docker = pytest.importorskip("docker")
    client = None
    try:
        client = docker.from_env()
        client.ping()
    except Exception as error:
        if client is not None:
            client.close()
        pytest.skip(f"Docker daemon is unavailable: {type(error).__name__}")

    image = os.environ.get("REPORL_DOCKER_CANARY_IMAGE", "alpine:3.20")
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    original = b"SECRET = 41\n"
    (hidden / "test_hidden.py").write_bytes(original)
    stage_name = ".reporl-canary"
    stage_root = f"/tmp/{stage_name}"
    hidden_path = f"{stage_root}/tests/test_hidden.py"
    container = None
    try:
        container = client.containers.run(
            image,
            ("sleep", "86400"),
            detach=True,
            network_disabled=True,
            read_only=True,
            user="1000:1000",
            working_dir="/tmp",
            cap_drop=("ALL",),
            security_opt=("no-new-privileges:true",),
            tmpfs={"/tmp": "rw,nosuid,nodev,size=64m,mode=1777"},
            labels={"reporl.role": "verifier-adversarial-canary"},
        )
        assert container.put_archive("/tmp", _suite_staging_archive(stage_name, "1000:1000"))
        assert container.put_archive(stage_root, _readonly_directory_archive(hidden, "tests"))

        script = """
set -eu
stage="$1"
hidden="$stage/tests/test_hidden.py"
evidence="$stage/evidence/canary.xml"
[ "$(stat -c '%u:%g:%a' "$stage")" = "0:0:555" ]
[ "$(stat -c '%u:%g:%a' "$stage/tests")" = "0:0:555" ]
[ "$(stat -c '%u:%g:%a' "$hidden")" = "0:0:444" ]
[ "$(stat -c '%u:%g:%a' "$stage/evidence")" = "1000:1000:700" ]
[ "$(cat "$hidden")" = "SECRET = 41" ]
if printf 'FORGED\n' >"$hidden" 2>/dev/null; then exit 20; fi
if rm -f "$hidden" 2>/dev/null; then exit 21; fi
if mv "$stage" "${stage}-moved" 2>/dev/null; then exit 22; fi
printf '<testsuite />\n' >"$evidence"
[ "$(cat "$hidden")" = "SECRET = 41" ]
"""
        result = container.exec_run(("sh", "-c", script, "reporl-canary", stage_root), demux=True)
        assert result.exit_code == 0, result.output

        archived, metadata = _archive_file(container, hidden_path)
        assert archived == original
        assert metadata["name"] == "test_hidden.py"
        assert metadata["size"] == len(original)
    finally:
        try:
            if container is not None:
                container.remove(force=True)
        finally:
            client.close()
