from __future__ import annotations

import pytest

from reporl.cloud.preflight import parse_nvidia_smi_line


def test_parse_nvidia_smi_line() -> None:
    assert parse_nvidia_smi_line("NVIDIA GeForce RTX 4090, 24564, 575.57") == (
        "NVIDIA GeForce RTX 4090",
        24_564,
        "575.57",
    )


def test_parse_nvidia_smi_line_rejects_invalid_csv() -> None:
    with pytest.raises(ValueError, match="unexpected"):
        parse_nvidia_smi_line("not,csv")
