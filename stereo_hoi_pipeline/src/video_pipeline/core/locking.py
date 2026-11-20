"""Exclusive per-workspace run lock.

Two runs on the same video would interleave writes to the same files (this
happened once: two concurrently appended JSONL streams corrupted each other). The
lock file is created atomically (O_EXCL) and holds the owner's PID; a lock whose
process no longer exists is treated as stale and replaced.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from video_pipeline.core.errors import WorkspaceError
from video_pipeline.core.io import utc_now_iso


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # Never use os.kill on Windows: it terminates the target process.
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunLock:
    def __init__(self, path: Path, description: str) -> None:
        self.path = path
        self._description = description
        self._held = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"pid": os.getpid(), "started_at": utc_now_iso(), "command": self._description})
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                owner = self._read_owner()
                if owner and _pid_alive(int(owner.get("pid", -1))):
                    raise WorkspaceError(
                        f"Another run is already processing this video (pid {owner.get('pid')}, started "
                        f"{owner.get('started_at')}). Wait for it to finish or stop it, then try again."
                    ) from None
                self.path.unlink(missing_ok=True)  # stale lock from a crashed/killed run
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            self._held = True
            return
        raise WorkspaceError(f"Could not acquire run lock '{self.path}'.")

    def _read_owner(self) -> dict | None:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def release(self) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False
