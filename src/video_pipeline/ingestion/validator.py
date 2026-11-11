"""Input validation for source videos.

Two levels:
* ``validate_source_file`` — filesystem checks (exists, regular file, non-empty, extension).
* ``validate_probe`` — container/stream checks on raw FFprobe JSON (passed in as data,
  so this module does not depend on how FFprobe is executed).

Fatal problems raise InputValidationError; non-fatal observations are returned as warnings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from video_pipeline.core.errors import InputValidationError


def validate_source_file(path: Path, supported_formats: Sequence[str]) -> dict[str, Any]:
    if not path.exists():
        raise InputValidationError(f"Source video does not exist: '{path}'.")
    if not path.is_file():
        raise InputValidationError(f"Source path is not a regular file: '{path}'.")
    size = path.stat().st_size
    if size == 0:
        raise InputValidationError(f"Source video is empty: '{path}'.")
    extension = path.suffix.lstrip(".").lower()
    allowed = {fmt.lower().lstrip(".") for fmt in supported_formats}
    if extension not in allowed:
        raise InputValidationError(
            f"Unsupported file extension '.{extension}' for '{path.name}'. Supported: {sorted(allowed)}."
        )
    return {"exists": True, "is_file": True, "size_bytes": size, "extension": extension}


def _is_real_video_stream(stream: Mapping[str, Any]) -> bool:
    if stream.get("codec_type") != "video":
        return False
    # Cover art / thumbnails are exposed as video streams with attached_pic set.
    return not (stream.get("disposition") or {}).get("attached_pic")


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def validate_probe(raw_probe: Mapping[str, Any]) -> list[str]:
    """Validate FFprobe output. Returns warnings; raises on fatal problems."""
    streams = raw_probe.get("streams") or []
    if not streams:
        raise InputValidationError("FFprobe reported no streams in the source file.")

    video_streams = [s for s in streams if _is_real_video_stream(s)]
    if not video_streams:
        raise InputValidationError("The source file contains no video stream.")

    primary = video_streams[0]
    if not (_positive_float(primary.get("width")) and _positive_float(primary.get("height"))):
        raise InputValidationError("The primary video stream has no valid width/height.")

    duration = _positive_float((raw_probe.get("format") or {}).get("duration")) or _positive_float(primary.get("duration"))
    if duration is None:
        raise InputValidationError("Could not determine a positive duration for the source video.")

    warnings: list[str] = []
    if len(video_streams) > 1:
        warnings.append(f"{len(video_streams)} video streams found; stream index {primary.get('index')} is treated as primary.")
    if not any(s.get("codec_type") == "audio" for s in streams):
        warnings.append("No audio stream present.")
    has_gpmf = any(
        s.get("codec_tag_string") == "gpmd" or "gopro met" in str((s.get("tags") or {}).get("handler_name", "")).lower()
        for s in streams
    )
    if not has_gpmf:
        warnings.append("No GoPro GPMF telemetry stream detected; IMU stages will have no input.")
    return warnings
