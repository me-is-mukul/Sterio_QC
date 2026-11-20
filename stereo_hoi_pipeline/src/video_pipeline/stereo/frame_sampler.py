"""Deterministic, timestamp-based stereo frame sampling.

Rule: inside a window ``[window_start, window_end)`` (seconds from the video start),
sample ``k`` is the **first decoded frame whose timestamp is >= window_start + k * interval**.
Selection happens on real timestamps inside one FFmpeg decode pass (``select``
filter), so it is correct for variable-frame-rate video and never uses frame-number
arithmetic. A window that starts after 0 is reached with an accurate input seek.

The same decoded frame is cropped into left and right, so both images of a pair
share one timestamp. The actual PTS of every selected frame is captured with
``showinfo``. If an interval contains no frame, that sample is reported missing
rather than invented.

Frames are written to the caller's directories; the caller decides when to delete
them (the pipeline deletes each pair as soon as it has been classified).
"""

from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from video_pipeline.core.errors import SamplingError
from video_pipeline.core.tools import CommandResult, run_command
from video_pipeline.stereo.stereo_splitter import StereoGeometry

_SHOWINFO = re.compile(r"Parsed_showinfo.*?\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\s*(-?[\d.]+(?:e-?\d+)?)")
_EPS = 1e-6


@dataclass(frozen=True)
class SampledFrame:
    sample_index: int                 # k within the window
    target_timestamp_seconds: float   # window_start + k * interval (from video start)
    frame_pts_seconds: float          # actual frame time, from video start
    container_pts_seconds: float      # actual frame time on the container timeline
    left_path: Path
    right_path: Path


def jpeg_qscale(quality: int) -> int:
    """Map a 1-100 JPEG quality to FFmpeg's mjpeg -q:v scale (2 best .. 31 worst). Approximate."""
    return max(2, min(31, round(2 + (100 - quality) * 29 / 99)))


def expected_samples(window_start: float, window_end: float, interval: float) -> int:
    span = window_end - window_start
    return max(0, math.ceil(span / interval - _EPS))


def build_command(ffmpeg: str, video: Path, geometry: StereoGeometry, *, interval: float, video_start: float,
                  window_start: float, window_end: float, max_width: int, jpeg_quality: int,
                  left_pattern: Path, right_pattern: Path) -> tuple[list[str], float]:
    """Return (command, pts_offset) where container_time = frame pts + pts_offset."""
    seek = video_start + window_start if window_start > 0 else None
    pts_offset = seek or 0.0
    # Frame time relative to the window start, expressed in filter-graph pts seconds.
    c = pts_offset - video_start - window_start
    rel = f"(t+{c:.6f})"
    rel_prev = f"(prev_selected_t+{c:.6f})"
    bucket = f"floor({rel}/{interval}+{_EPS})"
    prev_bucket = f"floor({rel_prev}/{interval}+{_EPS})"
    trim_end = video_start + window_end - pts_offset
    select = f"select='lt({rel},{window_end - window_start:.6f})*(isnan(prev_selected_t)+gt({bucket},{prev_bucket}))'"
    scale = f"scale=w='min({max_width},iw)':h=-2"
    graph = (
        f"[0:v:0]trim=end={trim_end:.6f},{select},showinfo,split=2[l0][r0];"
        f"[l0]{geometry.left.ffmpeg_filter()},{scale}[left];"
        f"[r0]{geometry.right.ffmpeg_filter()},{scale}[right]"
    )
    q = str(jpeg_qscale(jpeg_quality))
    cmd = [ffmpeg, "-hide_banner", "-nostats", "-loglevel", "info", "-y"]
    if seek is not None:
        cmd += ["-ss", f"{seek:.6f}"]
    cmd += [
        "-i", str(video),
        "-filter_complex", graph,
        "-map", "[left]", "-fps_mode", "passthrough", "-q:v", q, "-start_number", "0", str(left_pattern),
        "-map", "[right]", "-fps_mode", "passthrough", "-q:v", q, "-start_number", "0", str(right_pattern),
    ]
    return cmd, pts_offset


def sample_stereo_frames(ffmpeg: str, video: Path, geometry: StereoGeometry, *, interval: float,
                         video_start: float, window_start: float, window_end: float, max_width: int,
                         jpeg_quality: int, left_dir: Path, right_dir: Path, timeout_seconds: float,
                         name_prefix: str = "") -> tuple[list[SampledFrame], list[int], CommandResult]:
    """Extract stereo pairs for one window into ``left_dir``/``right_dir``.

    Files are named ``<prefix><k:06d>.jpg``. Returns (frames, missing_sample_indices, command).
    """
    staging = left_dir.parent / f"_staging_{name_prefix or 'main'}"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "left").mkdir(parents=True)
    (staging / "right").mkdir(parents=True)
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    try:
        cmd, pts_offset = build_command(
            ffmpeg, video, geometry, interval=interval, video_start=video_start, window_start=window_start,
            window_end=window_end, max_width=max_width, jpeg_quality=jpeg_quality,
            left_pattern=staging / "left" / "%06d.jpg", right_pattern=staging / "right" / "%06d.jpg")
        result = run_command(cmd, timeout_seconds=timeout_seconds)
        if result.returncode != 0:
            tail = "\n".join(result.stderr.strip().splitlines()[-5:])
            raise SamplingError(f"FFmpeg frame sampling failed (exit {result.returncode}): {tail}")

        selections = [float(t) for _n, _pts, t in _SHOWINFO.findall(result.stderr)]
        left_files = sorted((staging / "left").glob("*.jpg"))
        right_files = sorted((staging / "right").glob("*.jpg"))
        if not (len(selections) == len(left_files) == len(right_files)):
            raise SamplingError(
                f"Sampling mismatch: {len(selections)} selected frames, {len(left_files)} left, {len(right_files)} right images."
            )

        frames: list[SampledFrame] = []
        seen: set[int] = set()
        for pts, left_src, right_src in zip(selections, left_files, right_files):
            container = pts + pts_offset
            from_video_start = container - video_start
            k = int(math.floor((from_video_start - window_start) / interval + _EPS))
            if k in seen or k < 0:
                raise SamplingError(f"Unexpected selection for sample {k}; sampling is not deterministic.")
            seen.add(k)
            left_dst = left_dir / f"{name_prefix}{k:06d}.jpg"
            right_dst = right_dir / f"{name_prefix}{k:06d}.jpg"
            left_src.replace(left_dst)
            right_src.replace(right_dst)
            frames.append(SampledFrame(k, round(window_start + k * interval, 6), round(from_video_start, 6),
                                       round(container, 6), left_dst, right_dst))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    missing = sorted(set(range(expected_samples(window_start, window_end, interval))) - seen)
    return frames, missing, result
