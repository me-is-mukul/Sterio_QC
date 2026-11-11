"""Frame-accurate removal of intervals from the stereo video with FFmpeg.

The full side-by-side frame is cut (left and right are never cut independently).
Retained intervals are selected with ``trim``/``atrim`` on real timestamps,
re-timed with ``setpts``, and concatenated in chronological order, then re-encoded.
Re-encoding is deliberate: stream-copy cuts snap to keyframes and are not accurate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_pipeline.analysis.temporal_filter import Interval
from video_pipeline.core.errors import SegmentationError
from video_pipeline.core.tools import CommandResult, run_command


@dataclass(frozen=True)
class EncoderSettings:
    video_encoder: str
    video_encoder_args: tuple[str, ...]
    pixel_format: str
    audio_encoder: str


def build_processing_timeline(retained: Sequence[Interval]) -> list[dict[str, Any]]:
    """Original -> processed time mapping for each retained segment."""
    segments = []
    cursor = 0.0
    for i, iv in enumerate(retained):
        segments.append({
            "segment_index": i,
            "original_start": round(iv.start, 6),
            "original_end": round(iv.end, 6),
            "processed_start": round(cursor, 6),
            "processed_end": round(cursor + iv.duration, 6),
            "duration": round(iv.duration, 6),
        })
        cursor += iv.duration
    return segments


def build_cut_command(ffmpeg: str, video: Path, retained: Sequence[Interval], output: Path, *,
                      start_offset: float, include_audio: bool, settings: EncoderSettings) -> list[str]:
    parts: list[str] = []
    concat_inputs: list[str] = []
    for i, iv in enumerate(retained):
        s, e = iv.start + start_offset, iv.end + start_offset
        parts.append(f"[0:v:0]trim=start={s:.6f}:end={e:.6f},setpts=PTS-STARTPTS[v{i}]")
        concat_inputs.append(f"[v{i}]")
        if include_audio:
            parts.append(f"[0:a:0]atrim=start={s:.6f}:end={e:.6f},asetpts=PTS-STARTPTS[a{i}]")
            concat_inputs.append(f"[a{i}]")
    n = len(retained)
    outputs = "[outv][outa]" if include_audio else "[outv]"
    parts.append(f"{''.join(concat_inputs)}concat=n={n}:v=1:a={1 if include_audio else 0}{outputs}")

    cmd = [ffmpeg, "-hide_banner", "-nostats", "-loglevel", "error", "-y", "-i", str(video),
           "-filter_complex", ";".join(parts), "-map", "[outv]"]
    if include_audio:
        cmd += ["-map", "[outa]", "-c:a", settings.audio_encoder]
    cmd += ["-c:v", settings.video_encoder, *settings.video_encoder_args, "-pix_fmt", settings.pixel_format,
            "-map_metadata", "0", "-map_chapters", "-1", "-movflags", "+faststart", str(output)]
    return cmd


def cut_video(ffmpeg: str, video: Path, retained: Sequence[Interval], output: Path, *, start_offset: float,
              include_audio: bool, settings: EncoderSettings, timeout_seconds: float) -> CommandResult:
    if not retained:
        raise SegmentationError("Nothing to keep: every interval was removed.")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.stem + ".partial" + output.suffix)
    cmd = build_cut_command(ffmpeg, video, retained, tmp, start_offset=start_offset,
                            include_audio=include_audio, settings=settings)
    result = run_command(cmd, timeout_seconds=timeout_seconds)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        tail = "\n".join(result.stderr.strip().splitlines()[-5:])
        raise SegmentationError(f"FFmpeg cutting failed (exit {result.returncode}): {tail}")
    tmp.replace(output)
    return result
