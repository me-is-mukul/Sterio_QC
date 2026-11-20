"""FFprobe metadata extraction.

``probe_video`` runs FFprobe and returns its complete JSON output unchanged (the
authoritative record). ``summarize_probe`` derives a normalised, typed summary
used by later stages. The raw output is always persisted alongside the summary.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from video_pipeline.core.errors import MetadataExtractionError
from video_pipeline.core.tools import CommandResult, run_command

logger = logging.getLogger(__name__)

SUMMARY_SCHEMA_VERSION = "1.0.0"

_BASE_ARGS: tuple[str, ...] = (
    "-v", "error",
    "-hide_banner",
    "-print_format", "json",
    "-show_format",
    "-show_streams",
    "-show_chapters",
    "-show_programs",
)


@dataclass(frozen=True)
class ProbeResult:
    raw: dict[str, Any]
    command: CommandResult


def probe_video(ffprobe_executable: str, video_path: Path, *, count_frames: bool, timeout_seconds: float) -> ProbeResult:
    args = [ffprobe_executable, *_BASE_ARGS]
    if count_frames:
        args.append("-count_frames")
    args.append(str(video_path))
    result = run_command(args, timeout_seconds=timeout_seconds)
    if result.returncode != 0:
        detail = result.stderr.strip() or "no error output"
        raise MetadataExtractionError(f"FFprobe failed (exit {result.returncode}) for '{video_path}': {detail}")
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MetadataExtractionError(f"FFprobe returned invalid JSON for '{video_path}': {exc}") from exc
    if not isinstance(raw, dict):
        raise MetadataExtractionError("FFprobe JSON output is not an object.")
    if result.stderr.strip():
        logger.warning("FFprobe reported: %s", result.stderr.strip())
    return ProbeResult(raw=raw, command=result)


# --- value parsing -----------------------------------------------------------

def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rational(value: Any) -> float | None:
    """Parse FFprobe rationals like '30000/1001'. '0/0' and invalid values -> None."""
    if not isinstance(value, str) or "/" not in value:
        return _float(value)
    try:
        num, den = value.split("/", 1)
        if int(den) == 0:
            return None
        return float(Fraction(int(num), int(den)))
    except (ValueError, ZeroDivisionError):
        return None


def _rotation(stream: Mapping[str, Any]) -> float | None:
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            return _float(side_data["rotation"])
    return _float((stream.get("tags") or {}).get("rotate"))


def _is_gpmf(stream: Mapping[str, Any]) -> bool:
    handler = str((stream.get("tags") or {}).get("handler_name", "")).lower()
    return stream.get("codec_tag_string") == "gpmd" or "gopro met" in handler


# --- summary -------------------------------------------------------------------

def _video_summary(stream: Mapping[str, Any]) -> dict[str, Any]:
    nominal = _rational(stream.get("r_frame_rate"))
    average = _rational(stream.get("avg_frame_rate"))
    # r_frame_rate is the lowest rate that represents all timestamps; a different
    # avg_frame_rate is a strong hint of variable frame rate. Heuristic only.
    vfr = None
    if nominal is not None and average is not None:
        vfr = abs(nominal - average) > 0.01
    return {
        "index": stream.get("index"),
        "codec_name": stream.get("codec_name"),
        "codec_long_name": stream.get("codec_long_name"),
        "profile": stream.get("profile"),
        "codec_tag_string": stream.get("codec_tag_string"),
        "width": _int(stream.get("width")),
        "height": _int(stream.get("height")),
        "pix_fmt": stream.get("pix_fmt"),
        "field_order": stream.get("field_order"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "fps_nominal": nominal,
        "fps_average": average,
        "possibly_variable_frame_rate": vfr,
        "time_base": stream.get("time_base"),
        "start_pts": _int(stream.get("start_pts")),
        "start_time_seconds": _float(stream.get("start_time")),
        "duration_seconds": _float(stream.get("duration")),
        "bit_rate": _int(stream.get("bit_rate")),
        "nb_frames": _int(stream.get("nb_frames")),
        "nb_read_frames": _int(stream.get("nb_read_frames")),
        "rotation_degrees": _rotation(stream),
        "is_attached_picture": bool((stream.get("disposition") or {}).get("attached_pic")),
    }


def _audio_summary(stream: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "index": stream.get("index"),
        "codec_name": stream.get("codec_name"),
        "sample_rate": _int(stream.get("sample_rate")),
        "channels": _int(stream.get("channels")),
        "channel_layout": stream.get("channel_layout"),
        "time_base": stream.get("time_base"),
        "start_time_seconds": _float(stream.get("start_time")),
        "duration_seconds": _float(stream.get("duration")),
        "bit_rate": _int(stream.get("bit_rate")),
    }


def _data_summary(stream: Mapping[str, Any]) -> dict[str, Any]:
    tags = stream.get("tags") or {}
    return {
        "index": stream.get("index"),
        "codec_type": stream.get("codec_type"),
        "codec_name": stream.get("codec_name"),
        "codec_tag_string": stream.get("codec_tag_string"),
        "handler_name": tags.get("handler_name"),
        "time_base": stream.get("time_base"),
        "start_time_seconds": _float(stream.get("start_time")),
        "duration_seconds": _float(stream.get("duration")),
        "nb_frames": _int(stream.get("nb_frames")),
        "is_gpmf": _is_gpmf(stream),
    }


def summarize_probe(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalised, typed view of the FFprobe output (raw output remains authoritative)."""
    fmt = raw.get("format") or {}
    streams = raw.get("streams") or []

    video = [_video_summary(s) for s in streams if s.get("codec_type") == "video"]
    audio = [_audio_summary(s) for s in streams if s.get("codec_type") == "audio"]
    other = [_data_summary(s) for s in streams if s.get("codec_type") not in {"video", "audio"}]

    primary = next((v for v in video if not v["is_attached_picture"]), None)
    duration = _float(fmt.get("duration"))
    if duration is None and primary is not None:
        duration = primary["duration_seconds"]

    gpmf_indices = [d["index"] for d in other if d["is_gpmf"]]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "duration_seconds": duration,
        "primary_video_stream_index": primary["index"] if primary else None,
        "format": {
            "filename": fmt.get("filename"),
            "format_name": fmt.get("format_name"),
            "format_long_name": fmt.get("format_long_name"),
            "duration_seconds": _float(fmt.get("duration")),
            "start_time_seconds": _float(fmt.get("start_time")),
            "size_bytes": _int(fmt.get("size")),
            "bit_rate": _int(fmt.get("bit_rate")),
            "nb_streams": _int(fmt.get("nb_streams")),
            "tags": dict(fmt.get("tags") or {}),
        },
        "video_streams": video,
        "audio_streams": audio,
        "other_streams": other,
        "chapters_count": len(raw.get("chapters") or []),
        "telemetry": {
            "gpmf_stream_present": bool(gpmf_indices),
            "gpmf_stream_indices": gpmf_indices,
        },
    }
