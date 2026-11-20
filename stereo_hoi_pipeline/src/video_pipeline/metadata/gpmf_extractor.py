"""GoPro GPMF telemetry extraction.

Copies the GPMF data stream out of the container byte-for-byte (``-c copy``) and
records each packet's container timing and byte range, so the raw payload can be
re-parsed at any time. The raw file is the authoritative telemetry record.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from video_pipeline.core.errors import TelemetryError
from video_pipeline.core.tools import CommandResult, run_command


@dataclass(frozen=True)
class GpmfPacket:
    index: int
    pts_seconds: float
    duration_seconds: float | None
    offset: int  # byte offset inside the extracted raw file
    size: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GpmfExtraction:
    raw_path: Path
    stream_index: int
    packets: tuple[GpmfPacket, ...]
    commands: tuple[CommandResult, ...]


def _float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def list_packets(ffprobe: str, video_path: Path, stream_index: int, *, timeout_seconds: float) -> tuple[list[GpmfPacket], CommandResult]:
    result = run_command(
        [ffprobe, "-v", "error", "-select_streams", str(stream_index), "-show_packets",
         "-show_entries", "packet=pts_time,duration_time,size", "-of", "json", str(video_path)],
        timeout_seconds=timeout_seconds,
    )
    if result.returncode != 0:
        raise TelemetryError(f"FFprobe could not list telemetry packets: {result.stderr.strip()}")
    try:
        raw_packets = json.loads(result.stdout).get("packets", [])
    except json.JSONDecodeError as exc:
        raise TelemetryError(f"Invalid FFprobe packet JSON: {exc}") from exc

    packets: list[GpmfPacket] = []
    offset = 0
    for i, pkt in enumerate(raw_packets):
        size = int(pkt.get("size", 0))
        pts = _float(pkt.get("pts_time"))
        if pts is None:
            raise TelemetryError(f"Telemetry packet {i} has no timestamp; cannot place samples in time.")
        packets.append(GpmfPacket(i, pts, _float(pkt.get("duration_time")), offset, size))
        offset += size
    return packets, result


def extract_gpmf(ffmpeg: str, ffprobe: str, video_path: Path, stream_index: int, raw_path: Path,
                 *, timeout_seconds: float) -> GpmfExtraction:
    """Copy stream ``stream_index`` to ``raw_path`` and verify it matches the packet table."""
    packets, probe_cmd = list_packets(ffprobe, video_path, stream_index, timeout_seconds=timeout_seconds)
    if not packets:
        raise TelemetryError(f"Telemetry stream {stream_index} contains no packets.")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = raw_path.with_name(raw_path.name + ".tmp")
    copy_cmd = run_command(
        [ffmpeg, "-v", "error", "-y", "-i", str(video_path), "-map", f"0:{stream_index}",
         "-c", "copy", "-f", "data", str(tmp)],
        timeout_seconds=timeout_seconds,
    )
    if copy_cmd.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise TelemetryError(f"FFmpeg could not extract telemetry: {copy_cmd.stderr.strip()}")

    expected = sum(p.size for p in packets)
    actual = tmp.stat().st_size
    if actual != expected:
        tmp.unlink(missing_ok=True)
        raise TelemetryError(f"Extracted telemetry is {actual} bytes but packets total {expected} bytes.")
    tmp.replace(raw_path)
    return GpmfExtraction(raw_path, stream_index, tuple(packets), (probe_cmd, copy_cmd))
