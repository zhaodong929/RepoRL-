"""Static patch inspection performed before untrusted code is executed."""

from __future__ import annotations

import fnmatch
from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import Field

from reporl.schemas import StrictModel
from reporl.tools.paths import UnsafePathError, normalize_repo_path


class PatchViolationCode(StrEnum):
    MALFORMED = "malformed"
    OVERSIZED = "oversized"
    PATH_ESCAPE = "path_escape"
    PATH_NOT_ALLOWED = "path_not_allowed"
    FORBIDDEN_PATH = "forbidden_path"
    BINARY_PATCH = "binary_patch"
    MODE_CHANGE = "mode_change"
    SYMLINK_OR_SUBMODULE = "symlink_or_submodule"
    RENAME_OR_COPY = "rename_or_copy"
    TEST_BYPASS = "test_bypass"


class PatchViolation(StrictModel):
    code: PatchViolationCode
    message: str = Field(min_length=1, max_length=1_000)
    path: str | None = None


class PatchInspection(StrictModel):
    paths: tuple[str, ...]
    byte_size: int = Field(ge=0)
    hunk_count: int = Field(ge=0)
    violations: tuple[PatchViolation, ...] = ()

    @property
    def accepted(self) -> bool:
        return not self.violations


def _matches_glob(path: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(path, pattern) or PurePosixPath(path).match(pattern)


def _strip_git_prefix(raw_path: str) -> str | None:
    if raw_path == "/dev/null":
        return None
    if "\\" in raw_path or '"' in raw_path:
        raise UnsafePathError("quoted or escaped Git paths are not allowed")
    if raw_path.startswith("a/") or raw_path.startswith("b/"):
        raw_path = raw_path[2:]
    return normalize_repo_path(raw_path)


def _parse_diff_git_paths(line: str) -> tuple[str, str]:
    if '"' in line or "\\" in line:
        raise UnsafePathError("quoted or escaped Git paths are not allowed")
    fields = line.split()
    if len(fields) != 4 or fields[:2] != ["diff", "--git"]:
        raise UnsafePathError("malformed diff --git header")
    return fields[2], fields[3]


def _parse_file_header(line: str, marker: str) -> str:
    value = line[len(marker) :]
    if "\t" in value:
        value = value.split("\t", maxsplit=1)[0]
    if value.startswith('"'):
        raise UnsafePathError("quoted Git paths are not allowed")
    if "\\" in value:
        raise UnsafePathError("escaped Git paths are not allowed")
    return value


class PatchPolicy:
    """Reject unsafe or out-of-scope unified diffs without applying them."""

    _BYPASS_MARKERS = (
        "pytest.skip(",
        "pytest.xfail(",
        "pytestmark = pytest.mark.skip",
        "@pytest.mark.skip",
        "@pytest.mark.xfail",
        "@unittest.skip",
        "__test__ = false",
    )

    def __init__(
        self,
        *,
        allowed_paths: tuple[str, ...] = (),
        forbidden_globs: tuple[str, ...] = (),
        max_patch_bytes: int = 100_000,
    ) -> None:
        self._allowed_paths = tuple(
            normalize_repo_path(path, allow_dot=True) for path in allowed_paths
        )
        self._forbidden_globs = forbidden_globs
        self._max_patch_bytes = max_patch_bytes

    def inspect(self, patch: str) -> PatchInspection:
        byte_size = len(patch.encode("utf-8"))
        violations: list[PatchViolation] = []
        paths: set[str] = set()
        hunk_count = 0
        saw_file_header = False

        if byte_size > self._max_patch_bytes:
            violations.append(
                PatchViolation(
                    code=PatchViolationCode.OVERSIZED,
                    message=(
                        f"patch is {byte_size} bytes; maximum is {self._max_patch_bytes} bytes"
                    ),
                )
            )

        for line in patch.splitlines():
            candidate_paths: tuple[str, ...] = ()
            try:
                if line.startswith("diff --git "):
                    candidate_paths = _parse_diff_git_paths(line)
                    saw_file_header = True
                elif line.startswith("--- "):
                    candidate_paths = (_parse_file_header(line, "--- "),)
                    saw_file_header = True
                elif line.startswith("+++ "):
                    candidate_paths = (_parse_file_header(line, "+++ "),)
                    saw_file_header = True
            except UnsafePathError as error:
                violations.append(
                    PatchViolation(
                        code=PatchViolationCode.MALFORMED,
                        message=str(error),
                    )
                )

            for candidate in candidate_paths:
                try:
                    normalized = _strip_git_prefix(candidate)
                except UnsafePathError as error:
                    violations.append(
                        PatchViolation(
                            code=PatchViolationCode.PATH_ESCAPE,
                            message=str(error),
                            path=candidate,
                        )
                    )
                    continue
                if normalized is not None:
                    paths.add(normalized)

            if line.startswith("@@ ") or line == "@@":
                hunk_count += 1
            if line == "GIT binary patch" or line.startswith("Binary files "):
                violations.append(
                    PatchViolation(
                        code=PatchViolationCode.BINARY_PATCH,
                        message="binary patches are not allowed",
                    )
                )
            if line.startswith(("old mode ", "new mode ")):
                violations.append(
                    PatchViolation(
                        code=PatchViolationCode.MODE_CHANGE,
                        message="file mode changes are not allowed",
                    )
                )
            if line.startswith(("new file mode ", "deleted file mode ")):
                mode = line.rsplit(" ", maxsplit=1)[-1]
                if mode in {"120000", "160000"}:
                    violations.append(
                        PatchViolation(
                            code=PatchViolationCode.SYMLINK_OR_SUBMODULE,
                            message=f"unsafe git object mode {mode} is not allowed",
                        )
                    )
                elif mode != "100644":
                    violations.append(
                        PatchViolation(
                            code=PatchViolationCode.MODE_CHANGE,
                            message=f"file mode {mode} is not allowed",
                        )
                    )
            if line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
                violations.append(
                    PatchViolation(
                        code=PatchViolationCode.RENAME_OR_COPY,
                        message="rename and copy patches are not allowed",
                    )
                )
            if line.startswith("+") and not line.startswith("+++"):
                normalized_addition = line[1:].strip().lower()
                if any(marker in normalized_addition for marker in self._BYPASS_MARKERS):
                    violations.append(
                        PatchViolation(
                            code=PatchViolationCode.TEST_BYPASS,
                            message="patch adds a test skipping or collection bypass",
                        )
                    )

        if not saw_file_header or not paths:
            violations.append(
                PatchViolation(
                    code=PatchViolationCode.MALFORMED,
                    message="patch does not contain a changed repository path",
                )
            )
        if hunk_count == 0 and not any(
            violation.code == PatchViolationCode.BINARY_PATCH for violation in violations
        ):
            violations.append(
                PatchViolation(
                    code=PatchViolationCode.MALFORMED,
                    message="patch does not contain a unified-diff hunk",
                )
            )

        violations.extend(self._path_violations(paths))

        return self._inspection(
            paths=paths,
            byte_size=byte_size,
            hunk_count=hunk_count,
            violations=violations,
        )

    def inspect_applied_paths(
        self,
        inspection: PatchInspection,
        applied_paths: tuple[str, ...],
    ) -> PatchInspection:
        """Re-check Git's NUL-delimited view of paths after applying a patch."""

        paths: set[str] = set()
        violations = list(inspection.violations)
        for path in applied_paths:
            try:
                paths.add(normalize_repo_path(path))
            except UnsafePathError as error:
                violations.append(
                    PatchViolation(
                        code=PatchViolationCode.PATH_ESCAPE,
                        message=str(error),
                        path=path,
                    )
                )
        if paths != set(inspection.paths):
            violations.append(
                PatchViolation(
                    code=PatchViolationCode.MALFORMED,
                    message="patch headers do not match Git's applied paths",
                )
            )
        violations.extend(self._path_violations(paths))
        return self._inspection(
            paths=paths,
            byte_size=inspection.byte_size,
            hunk_count=inspection.hunk_count,
            violations=violations,
        )

    def _path_violations(self, paths: set[str]) -> list[PatchViolation]:
        violations: list[PatchViolation] = []
        for path in sorted(paths):
            if self._allowed_paths and not self._is_allowed(path):
                violations.append(
                    PatchViolation(
                        code=PatchViolationCode.PATH_NOT_ALLOWED,
                        message="path is outside the task allowlist",
                        path=path,
                    )
                )
            matching_glob = next(
                (pattern for pattern in self._forbidden_globs if _matches_glob(path, pattern)),
                None,
            )
            if matching_glob is not None:
                violations.append(
                    PatchViolation(
                        code=PatchViolationCode.FORBIDDEN_PATH,
                        message=f"path matches forbidden pattern {matching_glob!r}",
                        path=path,
                    )
                )
        return violations

    @staticmethod
    def _inspection(
        *,
        paths: set[str],
        byte_size: int,
        hunk_count: int,
        violations: list[PatchViolation],
    ) -> PatchInspection:
        unique_violations = tuple(
            dict.fromkeys(
                (violation.code, violation.message, violation.path) for violation in violations
            )
        )
        return PatchInspection(
            paths=tuple(sorted(paths)),
            byte_size=byte_size,
            hunk_count=hunk_count,
            violations=tuple(
                PatchViolation(code=code, message=message, path=path)
                for code, message, path in unique_violations
            ),
        )

    def _is_allowed(self, path: str) -> bool:
        for allowed in self._allowed_paths:
            if allowed == "." or path == allowed or path.startswith(f"{allowed.rstrip('/')}/"):
                return True
        return False
