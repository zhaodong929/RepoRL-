"""Fail-fast preflight checks for GPU, sandbox, or single-node RepoRL workers."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from reporl.schemas import StrictModel


class CheckResult(StrictModel):
    name: str
    passed: bool
    detail: str


class PreflightReport(StrictModel):
    mode: Literal["gpu", "sandbox", "single"]
    passed: bool
    checks: tuple[CheckResult, ...]


def _command(argv: Sequence[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def parse_nvidia_smi_line(line: str) -> tuple[str, int, str]:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        raise ValueError("unexpected nvidia-smi CSV output")
    name, memory, driver = parts
    return name, int(memory), driver


def run_preflight(
    *,
    mode: Literal["gpu", "sandbox", "single"],
    workspace: Path,
    minimum_vram_mb: int = 20_000,
    minimum_disk_gb: int = 80,
) -> PreflightReport:
    checks: list[CheckResult] = []
    checks.append(
        CheckResult(
            name="python",
            passed=(3, 11) <= sys.version_info[:2] < (3, 13),
            detail=platform.python_version(),
        )
    )
    checks.append(
        CheckResult(
            name="linux",
            passed=platform.system() == "Linux",
            detail=platform.platform(),
        )
    )
    free_gb = shutil.disk_usage(workspace).free // (1024**3)
    checks.append(
        CheckResult(
            name="disk",
            passed=free_gb >= minimum_disk_gb,
            detail=f"{free_gb} GiB free; require {minimum_disk_gb} GiB",
        )
    )
    for executable in ("git", "rg"):
        location = shutil.which(executable)
        checks.append(
            CheckResult(
                name=f"executable:{executable}",
                passed=location is not None,
                detail=location or "not found",
            )
        )
    if mode in {"gpu", "single"}:
        checks.extend(_gpu_checks(minimum_vram_mb))
    if mode in {"sandbox", "single"}:
        checks.extend(_docker_checks())
    return PreflightReport(
        mode=mode,
        passed=all(check.passed for check in checks),
        checks=tuple(checks),
    )


def _gpu_checks(minimum_vram_mb: int) -> list[CheckResult]:
    checks: list[CheckResult] = []
    result = _command(
        (
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        )
    )
    if result.returncode != 0 or not result.stdout.strip():
        return [CheckResult(name="nvidia-smi", passed=False, detail=result.stderr.strip())]
    try:
        name, memory_mb, driver = parse_nvidia_smi_line(result.stdout.splitlines()[0])
    except (ValueError, IndexError) as error:
        return [CheckResult(name="nvidia-smi", passed=False, detail=str(error))]
    checks.append(
        CheckResult(
            name="gpu-memory",
            passed=memory_mb >= minimum_vram_mb,
            detail=f"{name}; {memory_mb} MiB; driver {driver}",
        )
    )
    try:
        torch = importlib.import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
        detail = f"torch={torch.__version__}; cuda={torch.version.cuda}"
    except (ModuleNotFoundError, AttributeError) as error:
        cuda_available = False
        detail = str(error)
    checks.append(CheckResult(name="torch-cuda", passed=cuda_available, detail=detail))
    for module in ("transformers", "peft", "datasets", "bitsandbytes"):
        try:
            imported = importlib.import_module(module)
            version = str(getattr(imported, "__version__", "unknown"))
            checks.append(CheckResult(name=f"module:{module}", passed=True, detail=version))
        except (ModuleNotFoundError, OSError) as error:
            checks.append(CheckResult(name=f"module:{module}", passed=False, detail=str(error)))
    return checks


def _docker_checks() -> list[CheckResult]:
    docker = shutil.which("docker")
    if docker is None:
        return [CheckResult(name="docker", passed=False, detail="docker executable not found")]
    result = _command((docker, "info", "--format", "{{json .ServerVersion}}"))
    daemon_ok = result.returncode == 0
    return [
        CheckResult(
            name="docker-daemon",
            passed=daemon_ok,
            detail=result.stdout.strip() if daemon_ok else result.stderr.strip(),
        )
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gpu", "sandbox", "single"), required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--minimum-vram-mb", type=int, default=20_000)
    parser.add_argument("--minimum-disk-gb", type=int, default=80)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_preflight(
        mode=args.mode,
        workspace=args.workspace,
        minimum_vram_mb=args.minimum_vram_mb,
        minimum_disk_gb=args.minimum_disk_gb,
    )
    rendered = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
