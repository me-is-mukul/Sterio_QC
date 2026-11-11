"""FFmpeg-based checks of the processed video against the original."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from video_pipeline.core.tools import run_command

_PSNR = re.compile(r"average:(inf|[\d.]+)")


def probe_stream(ffprobe: str, path: Path, timeout: float) -> dict[str, Any]:
    result = run_command([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,codec_name,pix_fmt,avg_frame_rate:format=duration",
                          "-of", "json", str(path)], timeout_seconds=timeout)
    data = json.loads(result.stdout or "{}") if result.returncode == 0 else {}
    stream = (data.get("streams") or [{}])[0]
    return {**stream, "duration": float((data.get("format") or {}).get("duration", 0) or 0)}


def decode_errors(ffmpeg: str, path: Path, timeout: float) -> tuple[list[str], float]:
    """Fully decode the first video stream; returns (error lines, seconds taken)."""
    result = run_command([ffmpeg, "-hide_banner", "-nostats", "-v", "error", "-i", str(path),
                          "-map", "0:v:0", "-f", "null", "-"], timeout_seconds=timeout)
    lines = [l for l in result.stderr.splitlines() if l.strip()]
    if result.returncode != 0 and not lines:
        lines = [f"ffmpeg exited with {result.returncode}"]
    return lines, result.elapsed_seconds


def frame_psnr(ffmpeg: str, a: Path, ta: float, b: Path, tb: float, timeout: float) -> float | None:
    """PSNR (dB) between frame of ``a`` at ``ta`` and frame of ``b`` at ``tb``. 100 = identical."""
    # psnr pairs frames by timestamp: take exactly one frame from each input and give both
    # the same timebase and pts, otherwise frames from different files may not be compared.
    norm = "trim=end_frame=1,settb=AVTB,setpts=0,format=yuv420p"
    graph = f"[0:v]{norm}[x];[1:v]{norm}[y];[x][y]psnr"
    result = run_command([ffmpeg, "-hide_banner", "-nostats", "-ss", f"{ta:.6f}", "-i", str(a),
                          "-ss", f"{tb:.6f}", "-i", str(b), "-frames:v", "1", "-filter_complex", graph,
                          "-f", "null", "-"], timeout_seconds=timeout)
    match = _PSNR.search(result.stderr)
    if not match:
        return None
    return 100.0 if match.group(1) == "inf" else float(match.group(1))


def alignment_checkpoints(segments: Sequence[Mapping[str, Any]], max_points: int) -> list[tuple[float, float]]:
    """(processed_time, original_time) pairs near the start, middle and end of each retained segment."""
    points: list[tuple[float, float]] = []
    for seg in segments:
        d = seg["duration"]
        edge = min(0.2, d / 4)
        for offset in sorted({edge, d / 2, d - edge}):
            points.append((seg["processed_start"] + offset, seg["original_start"] + offset))
    if len(points) <= max_points:
        return points
    step = (len(points) - 1) / (max_points - 1)
    return [points[round(i * step)] for i in range(max_points)]
