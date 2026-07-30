from __future__ import annotations

import pytest
from pydantic import ValidationError

from reporl.sandbox.base import CommandSpec
from reporl.tools.output import truncate_output
from reporl.tools.patch import PatchPolicy, PatchViolationCode
from reporl.tools.paths import UnsafePathError, normalize_repo_path

VALID_PATCH = """diff --git a/src/math.py b/src/math.py
--- a/src/math.py
+++ b/src/math.py
@@ -1 +1 @@
-return 1
+return 2
"""


def test_patch_policy_accepts_scoped_text_patch() -> None:
    inspection = PatchPolicy(
        allowed_paths=("src",),
        forbidden_globs=("tests/**",),
    ).inspect(VALID_PATCH)

    assert inspection.accepted
    assert inspection.paths == ("src/math.py",)
    assert inspection.hunk_count == 1


@pytest.mark.parametrize(
    "patch, code",
    [
        (
            VALID_PATCH.replace("src/math.py", "tests/test_math.py"),
            PatchViolationCode.FORBIDDEN_PATH,
        ),
        (
            VALID_PATCH.replace("a/src/math.py", "a/../../outside.py", 1),
            PatchViolationCode.PATH_ESCAPE,
        ),
        (
            VALID_PATCH.replace("diff --git", "old mode 100644\nnew mode 100755\ndiff --git"),
            PatchViolationCode.MODE_CHANGE,
        ),
        (
            VALID_PATCH.replace("--- a/src/math.py", "new file mode 120000\n--- /dev/null"),
            PatchViolationCode.SYMLINK_OR_SUBMODULE,
        ),
        (
            VALID_PATCH.replace("@@ -1 +1 @@", "GIT binary patch"),
            PatchViolationCode.BINARY_PATCH,
        ),
        (
            VALID_PATCH.replace("+return 2", "+pytest.skip('not today')"),
            PatchViolationCode.TEST_BYPASS,
        ),
        (
            VALID_PATCH.replace("src/math.py", "docs/math.py"),
            PatchViolationCode.PATH_NOT_ALLOWED,
        ),
    ],
)
def test_patch_policy_rejects_unsafe_changes(
    patch: str,
    code: PatchViolationCode,
) -> None:
    inspection = PatchPolicy(
        allowed_paths=("src",),
        forbidden_globs=("tests/**",),
    ).inspect(patch)

    assert not inspection.accepted
    assert code in {violation.code for violation in inspection.violations}


def test_patch_policy_enforces_utf8_byte_limit() -> None:
    inspection = PatchPolicy(max_patch_bytes=10).inspect(VALID_PATCH)

    assert PatchViolationCode.OVERSIZED in {violation.code for violation in inspection.violations}


def test_patch_policy_rejects_git_quoted_octal_path() -> None:
    patch = r"""diff --git "a/src/\164est_math.py" "b/src/\164est_math.py"
--- "a/src/\164est_math.py"
+++ "b/src/\164est_math.py"
@@ -1 +1 @@
-return 1
+return 2
"""
    inspection = PatchPolicy(
        allowed_paths=("src",),
        forbidden_globs=("**/test*.py",),
    ).inspect(patch)

    assert not inspection.accepted
    assert PatchViolationCode.MALFORMED in {violation.code for violation in inspection.violations}


@pytest.mark.parametrize("shell", ["sh", "bash", "/bin/dash", "pwsh.exe", "cmd.exe"])
def test_command_spec_rejects_shell_executables(shell: str) -> None:
    with pytest.raises(ValidationError, match="command shell"):
        CommandSpec(argv=(shell, "-c", "pytest"))


def test_output_truncation_respects_exact_bound_and_keeps_tail() -> None:
    output, truncated = truncate_output("a" * 100 + "TAIL", 40)

    assert truncated
    assert len(output) == 40
    assert output.endswith("TAIL")


@pytest.mark.parametrize(
    "path",
    ["../secret", "/etc/passwd", "C:/Windows/win.ini", "src/../secret", "x\x00y"],
)
def test_path_normalizer_rejects_escape(path: str) -> None:
    with pytest.raises(UnsafePathError):
        normalize_repo_path(path)
