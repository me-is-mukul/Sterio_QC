"""Parse IMU samples (ACCL, GYRO, ...) from raw GoPro GPMF telemetry.

GPMF is a nested KLV format: 4-byte FourCC key, 1-byte type, 1-byte struct size,
2-byte big-endian repeat count, then ``size * repeat`` bytes padded to 4 bytes.
Type ``\\0`` means the payload contains nested KLVs (DEVC -> STRM -> data).

Timing: each GPMF packet carries the samples for one container packet. Sample ``i``
of ``n`` in a packet with container time ``pts`` and duration ``d`` is placed at
``pts + d * i / n``. The sensor's own STMP (microseconds) is preserved per packet.

Axis order is camera-dependent (see ORIN), so channels are exported as c0/c1/c2
with the declared order recorded in the metadata; no axis names are assumed.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from video_pipeline.metadata.gpmf_extractor import GpmfPacket

PARSER_VERSION = "1.0.0"

# GPMF type char -> struct format char (big-endian).
_TYPE_FORMATS = {
    "b": "b", "B": "B", "s": "h", "S": "H", "l": "i", "L": "I",
    "f": "f", "d": "d", "j": "q", "J": "Q",
}
_STICKY_KEYS = ("SCAL", "SIUN", "UNIT", "ORIN", "STNM", "STMP", "TSMP", "TMPC")


@dataclass
class Klv:
    key: str
    type: str
    struct_size: int
    repeat: int
    payload: bytes
    children: list[Klv] = field(default_factory=list)


def parse_klv(buffer: bytes, start: int = 0, end: int | None = None) -> list[Klv]:
    """Parse a sequence of KLVs. Truncated trailing data is ignored, not fabricated."""
    end = len(buffer) if end is None else end
    items: list[Klv] = []
    pos = start
    while pos + 8 <= end:
        key = buffer[pos:pos + 4].decode("latin-1")
        type_char = chr(buffer[pos + 4])
        struct_size = buffer[pos + 5]
        repeat = int.from_bytes(buffer[pos + 6:pos + 8], "big")
        length = struct_size * repeat
        data_start = pos + 8
        data_end = data_start + length
        if key == "\x00\x00\x00\x00":  # padding
            pos += 4
            continue
        if data_end > end:
            break
        item = Klv(key, type_char, struct_size, repeat, buffer[data_start:data_end])
        if type_char == "\x00":
            item.children = parse_klv(buffer, data_start, data_end)
        items.append(item)
        pos = data_start + ((length + 3) & ~3)
    return items


def decode_rows(item: Klv) -> list[tuple[Any, ...]]:
    """Decode numeric KLV payload into rows of values (one row per repeat)."""
    fmt = _TYPE_FORMATS.get(item.type)
    if fmt is None:
        raise ValueError(f"Unsupported GPMF numeric type '{item.type}' for {item.key}")
    element = struct.calcsize(fmt)
    if item.struct_size % element:
        raise ValueError(f"{item.key}: struct size {item.struct_size} is not a multiple of {element}")
    per_row = item.struct_size // element
    values = struct.unpack(f">{per_row * item.repeat}{fmt}", item.payload)
    return [values[i:i + per_row] for i in range(0, len(values), per_row)]


def decode_text(item: Klv) -> str:
    return item.payload.decode("latin-1").replace("\x00", "").strip()


@dataclass
class SensorSeries:
    fourcc: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    scale: list[float] | None = None
    units: str | None = None
    axis_order: str | None = None
    name: str | None = None
    channels: int | None = None
    warnings: list[str] = field(default_factory=list)


def _packet_durations(packets: Sequence[GpmfPacket]) -> list[float]:
    """Packet durations; missing values are filled from the next pts, else the median."""
    known = [p.duration_seconds for p in packets if p.duration_seconds]
    fallback = median(known) if known else 1.0
    durations = []
    for i, p in enumerate(packets):
        if p.duration_seconds:
            durations.append(p.duration_seconds)
        elif i + 1 < len(packets):
            durations.append(packets[i + 1].pts_seconds - p.pts_seconds)
        else:
            durations.append(fallback)
    return durations


def parse_imu(raw: bytes, packets: Sequence[GpmfPacket], fourccs: Sequence[str], *,
              video_start_seconds: float) -> dict[str, SensorSeries]:
    series = {code: SensorSeries(code) for code in fourccs}
    durations = _packet_durations(packets)

    for packet, duration in zip(packets, durations):
        payload = raw[packet.offset:packet.offset + packet.size]
        for devc in parse_klv(payload):
            if devc.key != "DEVC":
                continue
            for strm in devc.children:
                if strm.key != "STRM":
                    continue
                sticky = {c.key: c for c in strm.children if c.key in _STICKY_KEYS}
                for item in strm.children:
                    if item.key in series:
                        _append_samples(series[item.key], item, sticky, packet, duration, video_start_seconds)
    return series


def _append_samples(target: SensorSeries, item: Klv, sticky: dict[str, Klv], packet: GpmfPacket,
                    duration: float, video_start: float) -> None:
    try:
        rows = decode_rows(item)
    except (ValueError, struct.error) as exc:
        target.warnings.append(f"packet {packet.index}: {exc}")
        return
    if not rows:
        return
    channels = len(rows[0])
    target.channels = target.channels or channels

    scale = [1.0] * channels
    if "SCAL" in sticky:
        try:
            flat = [float(v) for row in decode_rows(sticky["SCAL"]) for v in row]
            if len(flat) == 1:
                scale = flat * channels
            elif len(flat) == channels:
                scale = flat
            else:
                target.warnings.append(f"packet {packet.index}: SCAL has {len(flat)} values for {channels} channels; unscaled")
        except (ValueError, struct.error) as exc:
            target.warnings.append(f"packet {packet.index}: SCAL unreadable ({exc}); unscaled")
    if any(s == 0 for s in scale):
        target.warnings.append(f"packet {packet.index}: zero SCAL; unscaled")
        scale = [1.0] * channels
    target.scale = scale

    for key, attr in (("SIUN", "units"), ("UNIT", "units"), ("ORIN", "axis_order"), ("STNM", "name")):
        if key in sticky and getattr(target, attr) is None:
            setattr(target, attr, decode_text(sticky[key]))

    stmp_us = None
    if "STMP" in sticky:
        try:
            stmp_us = int(decode_rows(sticky["STMP"])[0][0])
        except (ValueError, struct.error, IndexError):
            stmp_us = None

    n = len(rows)
    base_index = len(target.rows)
    for i, row in enumerate(rows):
        t_container = packet.pts_seconds + duration * i / n
        record: dict[str, Any] = {
            "sample_index": base_index + i,
            "packet_index": packet.index,
            "sample_in_packet": i,
            "packet_pts_seconds": round(packet.pts_seconds, 6),
            "packet_stmp_us": stmp_us,
            "time_container_seconds": round(t_container, 6),
            "time_video_seconds": round(t_container - video_start, 6),
        }
        for c, value in enumerate(row):
            record[f"c{c}"] = value / scale[c]
        for c, value in enumerate(row):
            record[f"raw_c{c}"] = value
        target.rows.append(record)


def series_metadata(s: SensorSeries) -> dict[str, Any]:
    times = [r["time_video_seconds"] for r in s.rows]
    span = (times[-1] - times[0]) if len(times) > 1 else 0.0
    return {
        "fourcc": s.fourcc,
        "name": s.name,
        "samples": len(s.rows),
        "channels": s.channels,
        "axis_order": s.axis_order,
        "units": s.units,
        "scale": s.scale,
        "first_time_video_seconds": times[0] if times else None,
        "last_time_video_seconds": times[-1] if times else None,
        "approx_sampling_rate_hz": round((len(times) - 1) / span, 3) if span > 0 else None,
        "warnings": s.warnings,
    }
