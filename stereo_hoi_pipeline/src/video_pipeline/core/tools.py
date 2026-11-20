"""Resolution and safe execution of external command-line tools.

Commands are always executed with argument lists (never a shell string).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass

from video_pipeline.core.errors import ToolExecutionError, ToolNotFoundError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float

    def to_record(self) -> dict:
        """Serializable record of the invocation (output omitted)."""
        return {"args": list(self.args), "returncode": self.returncode, "elapsed_seconds": round(self.elapsed_seconds, 3)}


def resolve_executable(name_or_path: str) -> str:
    """Return an absolute path for an executable name (via PATH) or an explicit path."""
    if os.path.isfile(name_or_path):
        return os.path.abspath(name_or_path)
    found = shutil.which(name_or_path)
    if not found:
        raise ToolNotFoundError(
            f"Required executable '{name_or_path}' was not found. Install it or set its path in the 'tools' configuration section."
        )
    return found


def run_command(args: Sequence[str], *, timeout_seconds: float) -> CommandResult:
    """Run a command and capture its output. Raises ToolExecutionError on timeout/launch failure.

    A non-zero exit status is returned to the caller, which decides how to handle it.
    """
    argv = tuple(str(a) for a in args)
    logger.debug("Executing: %s", argv)
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ToolNotFoundError(f"Executable not found: '{argv[0]}'.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolExecutionError(f"Command timed out after {timeout_seconds}s: {argv}") from exc
    except OSError as exc:
        raise ToolExecutionError(f"Could not execute {argv}: {exc}") from exc
    return CommandResult(argv, proc.returncode, proc.stdout, proc.stderr, time.perf_counter() - started)


def tool_version(name_or_path: str, *, timeout_seconds: float) -> str | None:
    """First line of ``<tool> -version`` (FFmpeg-style), or None if unavailable."""
    try:
        result = run_command([resolve_executable(name_or_path), "-version"], timeout_seconds=timeout_seconds)
    except (ToolNotFoundError, ToolExecutionError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    first_line = result.stdout.strip().splitlines()[0]
    # "ffmpeg version 8.0.1-essentials_build-www.gyan.dev Copyright ..." -> "8.0.1-essentials_build-www.gyan.dev"
    parts = first_line.split()
    if len(parts) >= 3 and parts[1] == "version":
        return parts[2]
    return first_line
