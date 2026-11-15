"""Container / stream metadata via ffprobe."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path


class ProbeError(Exception):
    pass


def _fraction(text: str | None) -> float | None:
    if not text or text in ("0/0", "N/A"):
        return None
    try:
        value = float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def _num(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rotation(stream: dict) -> int:
    for item in stream.get("side_data_list", []) or []:
        if "rotation" in item:
            return int(round(float(item["rotation"])))
    tag = (stream.get("tags") or {}).get("rotate")
    return int(tag) if tag not in (None, "") else 0


def probe(ffprobe: str, video: Path) -> dict:
    cmd = [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(video)]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    if result.returncode != 0:
        raise ProbeError(f"ffprobe failed: {result.stderr.strip()[:500]}")
    raw = json.loads(result.stdout)
    streams = raw.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")]
    if not videos:
        raise ProbeError("no video stream")
    v = videos[0]
    fmt = raw.get("format", {})
    rotation = _rotation(v)
    width, height = int(v["width"]), int(v["height"])
    display_w, display_h = (height, width) if abs(rotation) % 180 == 90 else (width, height)
    fps = _fraction(v.get("avg_frame_rate")) or _fraction(v.get("r_frame_rate"))
    duration = _num(v.get("duration")) or _num(fmt.get("duration"))
    nb_frames = int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else None
    return {
        "container": fmt.get("format_name"),
        "file_size_bytes": int(fmt["size"]) if fmt.get("size") else video.stat().st_size,
        "duration_s": duration,
        "container_bitrate_bps": _num(fmt.get("bit_rate")),
        "video_bitrate_bps": _num(v.get("bit_rate")),
        "codec": v.get("codec_name"),
        "codec_profile": v.get("profile"),
        "pix_fmt": v.get("pix_fmt"),
        "width": width,
        "height": height,
        "rotation_deg": rotation,
        "display_width": display_w,
        "display_height": display_h,
        "fps": fps,
        "r_frame_rate": v.get("r_frame_rate"),
        "avg_frame_rate": v.get("avg_frame_rate"),
        "nb_frames": nb_frames,
        "video_stream_index": v.get("index"),
        "stream_types": [s.get("codec_type") for s in streams],
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "creation_time": (fmt.get("tags") or {}).get("creation_time"),
        "encoder": (fmt.get("tags") or {}).get("encoder"),
    }
