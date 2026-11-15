"""Blockiness: visible compression-block edges, measured on full-resolution luma.

Block artefacts are steps on a fixed grid (8-64 px coding blocks, or e.g. 24/48 px when
low-resolution video was upscaled), and they are visible where the picture is otherwise
smooth. For every sampled frame and every column boundary we count:

    flat pixels   : the neighbouring differences on both sides are <= FLAT grey levels
    visible steps : flat pixels whose own difference is >= STEP grey levels

The step rate per column (steps / flat pixels) is accumulated over the clip; scene content
washes out, a block grid stays. The strongest grid (period P in PERIODS, any phase, chosen
on medians so a few strong scene edges cannot fake it) gives

    blockiness = step rate on the grid columns - step rate on all columns     (percentage points)

computed for columns and rows and averaged. 0 = no visible block edges.
Calibration (docs/METHODOLOGY.md): clean footage 0.5-2.5; visibly blocky footage 3.4 and up;
MPEG-2 with clear blocking 8-11.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

BLOCKING_VERSION = 2
PERIODS = (8, 16, 24, 32, 48, 64)
FLAT = 1          # grey levels: neighbouring differences at most this = flat surroundings
STEP = 3          # grey levels: a step of at least this is visible


def grid_strength(profile: np.ndarray) -> tuple[float, int | None, int | None]:
    """Strongest periodic grid in a per-column profile: (median ratio, period, phase)."""
    base = float(np.median(profile))
    best = (1.0, None, None)
    for period in PERIODS:
        if len(profile) < period * 8:
            continue
        for phase in range(period):
            on_grid = float(np.median(profile[phase::period]))
            ratio = on_grid / base if base > 0 else (np.inf if on_grid > 0 else 1.0)
            if ratio > best[0]:
                best = (ratio, period, phase)
    return best


def _accumulate(diff: np.ndarray, flat_count: np.ndarray, step_count: np.ndarray) -> None:
    """diff: |neighbour difference| along axis 1 (uint8). Updates per-position counts in place."""
    flat = (diff[:, :-2] <= FLAT) & (diff[:, 2:] <= FLAT)
    flat_count += flat.sum(axis=0)
    step_count += (flat & (diff[:, 1:-1] >= STEP)).sum(axis=0)


def _direction(flat_count: np.ndarray, step_count: np.ndarray) -> dict:
    rate = step_count / np.maximum(flat_count, 1)
    ratio, period, phase = grid_strength(rate)
    base = float(np.median(rate))
    grid = float(np.median(rate[phase::period])) if period else base
    return {"excess_pct": round(100.0 * (grid - base), 3), "grid_step_rate_pct": round(100.0 * grid, 3),
            "base_step_rate_pct": round(100.0 * base, 3), "period_px": period,
            # +1: the profile starts at the boundary between pixel 1 and 2
            "boundary_offset_px": (phase + 2) % period if period else None,
            "ratio": round(float(ratio), 3) if np.isfinite(ratio) else None}


def measure_blockiness(ffmpeg: str, video: Path, *, width: int, height: int, frame_stride: int) -> dict:
    """Decode every `frame_stride`-th frame at full resolution (grey) and measure blockiness.

    width/height: decoded (display) frame size, i.e. after rotation metadata is applied.
    """
    started = time.monotonic()
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-i", str(video.resolve()), "-map", "0:v:0",
           "-vf", f"select=not(mod(n\\,{frame_stride})),format=gray", "-fps_mode", "passthrough",
           "-f", "rawvideo", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=width * height)
    errors: list[str] = []
    drain = threading.Thread(target=lambda: errors.extend(
        x.decode("utf-8", "replace").rstrip() for x in proc.stderr), daemon=True)
    drain.start()
    col_flat, col_step = np.zeros(width - 3, np.int64), np.zeros(width - 3, np.int64)
    row_flat, row_step = np.zeros(height - 3, np.int64), np.zeros(height - 3, np.int64)
    frames = 0
    size = width * height
    buf = bytearray(size)
    view = memoryview(buf)
    try:
        while True:
            got = 0
            while got < size:
                n = proc.stdout.readinto(view[got:])
                if not n:
                    break
                got += n
            if got < size:
                break
            f = np.frombuffer(buf, dtype=np.uint8).reshape(height, width)
            _accumulate(cv2.absdiff(f[:, 1:], f[:, :-1]), col_flat, col_step)
            _accumulate(np.ascontiguousarray(cv2.absdiff(f[1:, :], f[:-1, :]).T), row_flat, row_step)
            frames += 1
    finally:
        proc.stdout.close()
        rc = proc.wait()
        drain.join(timeout=10)
    if frames == 0:
        return {"status": "failed", "error": "no frames decoded: " + "; ".join(errors[:3]), "decoder_exit_code": rc}
    h, v = _direction(col_flat, col_step), _direction(row_flat, row_step)
    return {
        "status": "ok",
        "blocking_version": BLOCKING_VERSION,
        "blockiness": round((h["excess_pct"] + v["excess_pct"]) / 2, 3),
        "columns": h,
        "rows": v,
        "frames": frames,
        "frame_stride": frame_stride,
        "flat_level": FLAT,
        "step_level": STEP,
        "decoder_exit_code": rc,
        "decoder_messages": len(errors),
        "elapsed_s": round(time.monotonic() - started, 1),
    }
