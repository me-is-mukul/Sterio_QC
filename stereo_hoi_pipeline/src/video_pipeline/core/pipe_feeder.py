"""Feed a subprocess's stdin from a bounded in-memory queue on its own thread.

Lets one reader (a Cloud Storage stream) serve several FFmpeg processes at once: a slow consumer
only blocks the producer when its queue is full, and memory stays bounded (max_bytes per process).
If the process exits early (it failed), the rest of the data is discarded so the producer never
blocks; the process's exit code reports the failure.
"""

from __future__ import annotations

import queue
import subprocess
import threading

_END = object()


class PipeFeeder:
    def __init__(self, proc: subprocess.Popen, *, max_bytes: int = 64 << 20) -> None:
        self._proc = proc
        self._queue: queue.Queue = queue.Queue()
        self._total_units = max(1, max_bytes // (1 << 16))                       # 64 KiB units
        self._budget = threading.BoundedSemaphore(self._total_units)
        self._dead = False
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _units(self, data: bytes) -> int:
        return min(self._total_units, max(1, len(data) >> 16))   # one item never needs more than the budget

    def write(self, data: bytes) -> None:
        if self._dead:
            return
        for _ in range(self._units(data)):
            self._budget.acquire()
        self._queue.put(data)

    def _run(self) -> None:
        stdin = self._proc.stdin
        while True:
            item = self._queue.get()
            if item is _END:
                break
            try:
                if not self._dead:
                    stdin.write(item)
            except (BrokenPipeError, OSError) as exc:      # the process exited; its exit code tells why
                self._dead = True
                self.error = exc
            finally:
                for _ in range(self._units(item)):
                    self._budget.release()
        try:
            stdin.close()
        except OSError:
            pass

    def close(self) -> None:
        """Signal end of data and wait until everything queued was written (or discarded)."""
        self._queue.put(_END)
        self._thread.join()

    @property
    def broken(self) -> bool:
        return self._dead
