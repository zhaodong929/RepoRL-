"""Bound observations before returning them to an untrusted policy."""

from __future__ import annotations


def truncate_output(value: str, max_chars: int) -> tuple[str, bool]:
    """Keep deterministic head and tail context within an exact character bound."""

    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if len(value) <= max_chars:
        return value, False

    marker = "\n...[output truncated]...\n"
    if max_chars <= len(marker):
        return marker[:max_chars], True
    available = max_chars - len(marker)
    head_size = (available + 1) // 2
    tail_size = available - head_size
    tail = value[-tail_size:] if tail_size else ""
    return f"{value[:head_size]}{marker}{tail}", True


def format_process_output(stdout: str, stderr: str) -> str:
    """Create an unambiguous observation without interpreting command output."""

    if stdout and stderr:
        return f"{stdout}\n[stderr]\n{stderr}"
    return stdout or stderr
