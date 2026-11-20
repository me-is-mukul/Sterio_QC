"""Logging configuration.

Design:
* One package logger (``video_pipeline``); modules use ``logging.getLogger(__name__)``.
* Console output for progress messages; a per-run file log inside the workspace.
* Records emitted before the workspace exists (e.g. during download) are buffered
  in memory and flushed into the file log once it is attached, so the log is complete.
* Every record carries the run ID and the current processing stage.
"""

from __future__ import annotations

import contextvars
import logging
import logging.handlers
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOGGER_NAME = "video_pipeline"

_STAGE: contextvars.ContextVar[str] = contextvars.ContextVar("pipeline_stage", default="setup")
_RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar("pipeline_run_id", default="pending")

_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | run=%(run_id)s | stage=%(stage)s | %(name)s | %(message)s"
_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(stage)-10s | %(message)s"


@contextmanager
def stage_scope(stage_name: str) -> Iterator[None]:
    """Tag every log record emitted inside the block with ``stage_name``."""
    token = _STAGE.set(stage_name)
    try:
        yield
    finally:
        _STAGE.reset(token)


def set_run_id(run_id: str) -> None:
    _RUN_ID.set(run_id)


class _StageFilter(logging.Filter):
    """Captures the stage at the time the record is created (first handler to see it)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "stage"):
            record.stage = _STAGE.get()
        return True


class _ContextFilter(_StageFilter):
    """Adds the stage (if not already captured) and the current run ID at emit time."""

    def filter(self, record: logging.LogRecord) -> bool:
        super().filter(record)
        record.run_id = _RUN_ID.get()
        return True


class LoggingController:
    """Owns the handlers of the package logger for the lifetime of one CLI invocation."""

    def __init__(self, console_level: str = "INFO") -> None:
        self.logger = logging.getLogger(LOGGER_NAME)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()

        self._console = logging.StreamHandler(sys.stderr)
        self._console.setLevel(console_level.upper())
        self._console.addFilter(_ContextFilter())
        self._console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt="%H:%M:%S"))
        self.logger.addHandler(self._console)

        self._buffer: logging.handlers.MemoryHandler | None = logging.handlers.MemoryHandler(
            capacity=100_000, flushLevel=logging.CRITICAL + 1, target=None, flushOnClose=False
        )
        self._buffer.setLevel(logging.DEBUG)
        self._buffer.addFilter(_StageFilter())
        self.logger.addHandler(self._buffer)
        self._file: logging.FileHandler | None = None

    def set_console_level(self, level: str) -> None:
        self._console.setLevel(level.upper())

    def attach_file(self, path: Path, level: str = "DEBUG") -> None:
        """Start writing to ``path`` (append mode) and flush buffered early records into it."""
        if self._file is not None:
            raise RuntimeError("A log file is already attached.")
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setLevel(level.upper())
        file_handler.addFilter(_ContextFilter())
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        if self._buffer is not None:
            self._buffer.setTarget(file_handler)
            self._buffer.flush()
            self.logger.removeHandler(self._buffer)
            self._buffer.close()
            self._buffer = None
        self.logger.addHandler(file_handler)
        self._file = file_handler

    def detach_file(self) -> None:
        if self._file is not None:
            self.logger.removeHandler(self._file)
            self._file.close()
            self._file = None

    def shutdown(self) -> None:
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()
        self._buffer = None
        self._file = None
