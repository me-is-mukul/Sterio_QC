"""One FFmpeg decode pass per clip.

The single pass yields, at the same time:
  * every `stride`-th frame, downscaled so each camera view is <= max_view_width, as RGB
    (for hand tracking and the per-frame picture metrics), streamed through a pipe;
  * FFmpeg's `blockdetect` score on every `block_stride`-th frame at FULL resolution
    (block artefacts are only measurable before any scaling), written to a text file;
  * every decoder error message (stderr at log level "error") for the decode-error test.
Rotation metadata is applied by FFmpeg (autorotate), so frames are as a player shows them.
"""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .manifest import LAYOUT_GRID

BLOCK_FILE = "blockiness_raw.txt"


def _even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


@dataclass(frozen=True)
class Geometry:
    columns: int
    rows: int
    view_w: float          # original view size (display orientation), pixels
    view_h: float
    out_view_w: int        # analysed view size
    out_view_h: int

    @property
    def out_w(self) -> int:
        return self.out_view_w * self.columns

    @property
    def out_h(self) -> int:
        return self.out_view_h * self.rows

    def split(self, frame: np.ndarray) -> list[np.ndarray]:
        """Views in reading order (left->right, top->bottom)."""
        views = []
        for r in range(self.rows):
            for c in range(self.columns):
                y0, x0 = r * self.out_view_h, c * self.out_view_w
                views.append(frame[y0:y0 + self.out_view_h, x0:x0 + self.out_view_w])
        return views


def geometry_for(layout: str, display_w: int, display_h: int, max_view_width: int) -> Geometry:
    cols, rows = LAYOUT_GRID[layout]
    view_w, view_h = display_w / cols, display_h / rows
    scale = min(1.0, max_view_width / view_w)
    return Geometry(cols, rows, view_w, view_h, _even(view_w * scale), _even(view_h * scale))


def stride_for(fps: float, bands: list[dict]) -> int:
    for band in bands:
        if fps <= float(band["max_fps"]):
            return int(band["stride"])
    return int(bands[-1]["stride"])


class FrameReader:
    """Iterate over (frame_number, time_s, rgb_frame) while FFmpeg decodes the whole clip once."""

    def __init__(self, ffmpeg: str, video: Path, *, fps: float, geometry: Geometry, stride: int,
                 block_stride: int, work_dir: Path):
        self.video, self.fps, self.geometry = video, fps, geometry
        self.stride, self.block_stride, self.work_dir = stride, block_stride, work_dir
        self.errors: list[str] = []
        self.returncode: int | None = None
        graph = (
            f"[0:v:0]split=2[h][q];"
            f"[h]select=not(mod(n\\,{stride})),scale={geometry.out_w}:{geometry.out_h}:flags=area,format=rgb24[hout];"
            f"[q]select=not(mod(n\\,{block_stride})),blockdetect,"
            f"metadata=mode=print:key=lavfi.block:file={BLOCK_FILE}[qout]"
        )
        self.cmd = [ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-i", str(video.resolve()),
                    "-filter_complex", graph,
                    "-map", "[hout]", "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1",
                    "-map", "[qout]", "-fps_mode", "passthrough", "-f", "null", "-"]

    def _drain_stderr(self, stream) -> None:
        for line in iter(stream.readline, b""):
            text = line.decode("utf-8", "replace").rstrip()
            if text:
                self.errors.append(text)

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / BLOCK_FILE).unlink(missing_ok=True)
        frame_bytes = self.geometry.out_w * self.geometry.out_h * 3
        proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.work_dir,
                                bufsize=frame_bytes)
        drain = threading.Thread(target=self._drain_stderr, args=(proc.stderr,), daemon=True)
        drain.start()
        index = 0
        completed = False
        try:
            buf = bytearray(frame_bytes)
            view = memoryview(buf)
            while True:
                got = 0
                while got < frame_bytes:
                    n = proc.stdout.readinto(view[got:])
                    if not n:
                        break
                    got += n
                if got == 0:
                    break
                if got < frame_bytes:
                    self.errors.append(f"[handqc] truncated frame from decoder ({got} of {frame_bytes} bytes)")
                    break
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(self.geometry.out_h, self.geometry.out_w, 3).copy()
                n_frame = index * self.stride
                yield n_frame, n_frame / self.fps, frame
                index += 1
            completed = True
        finally:
            self.frames_read = index
            if not completed and proc.poll() is None:
                proc.kill()                      # consumer stopped early (error / interrupt)
            try:
                proc.stdout.close()
            except OSError:
                pass
            try:
                self.returncode = proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                proc.kill()
                self.returncode = proc.wait()
            drain.join(timeout=10)

    def blockiness(self) -> list[tuple[int, float]]:
        """(frame_number, score) pairs from the blockdetect pass."""
        path = self.work_dir / BLOCK_FILE
        if not path.is_file():
            return []
        out, k = [], 0
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("lavfi.block="):
                try:
                    out.append((k * self.block_stride, float(line.split("=", 1)[1])))
                except ValueError:
                    pass
                k += 1
        return out
