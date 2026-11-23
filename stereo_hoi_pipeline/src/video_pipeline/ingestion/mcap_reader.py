"""Read MCAP segments as streams: index them, and later stream exactly the video packets needed.

Nothing video-sized is written to disk here. A segment is read straight from its source (a Cloud
Storage object, or a local file in tests) three times at most:
  1. ``extract_segment``  index: per frame its capture time, keyframe flag, size and ORDINAL (its
                          position among that camera's messages in the segment), plus IMU and
                          calibration. Written as small .npy/.json files.
  2. ``stream_packets``   the decode pass: the packets of the requested frames, in order.
  3. ``stream_packets``   the clip pass: only the segments a clip needs.
A frame is found again by (segment, camera, ordinal); every streamed packet is checked against the
size recorded in pass 1, so a mismatch fails loudly instead of producing a wrong video.

Facts this relies on (measured on the op-datoric recordings, see talking_to_mukul.md):
* Channels are looked up by topic NAME: channel IDs differ between files.
* Cameras are foxglove.CompressedVideo, format h265, Annex-B. Messages are read in FILE order (a
  non-seeking reader), which is the encoder's decode order.
* Every keyframe observed carries VPS/SPS/PPS, so a piece starting on a keyframe decodes alone.
* IMU (visio ImuRaw): first_sample_time + samples[i].t_offset_ns, all on the device clock.
MCAP chunk CRCs are validated while reading.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import IO, Any

import numpy as np

from video_pipeline.core.errors import IngestionError
from video_pipeline.core.io import write_json_atomic

logger = logging.getLogger(__name__)

CAMERA_KEYS = ("cam0", "cam1")
# ordinal: index of the message among this camera's messages in the segment (dropped duplicates counted)
FRAME_DTYPE = np.dtype([("t_ns", "<i8"), ("key", "?"), ("params", "?"), ("segment", "<i4"),
                        ("ordinal", "<i4"), ("size", "<i4")])
IMU_COLUMNS = ("gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z", "mag_x", "mag_y", "mag_z",
               "temperature_c")
_IRAP = range(16, 22)          # HEVC NAL types BLA/IDR/CRA: random access points (keyframes)
_VPS, _SPS, _PPS = 32, 33, 34
_SCAN_BYTES = 4096             # parameter sets + first slice header are at the start of a packet

# number -> context manager yielding a readable binary stream of that segment
SegmentOpener = Callable[[int], AbstractContextManager[IO[bytes]]]


def nal_units(data: bytes, limit: int = _SCAN_BYTES) -> list[tuple[int, int]]:
    """(start offset of the NAL header, NAL type) for Annex-B start codes in the first `limit` bytes."""
    found = []
    pos = 0
    end = min(len(data), limit)
    while True:
        i = data.find(b"\x00\x00\x01", pos, end)
        if i < 0 or i + 3 >= len(data):
            return found
        found.append((i + 3, (data[i + 3] >> 1) & 0x3F))
        pos = i + 3


def frame_flags(data: bytes) -> tuple[bool, bool]:
    """(is keyframe, carries VPS+SPS+PPS)."""
    types = {t for _, t in nal_units(data)}
    return any(t in _IRAP for t in types), {_VPS, _SPS, _PPS} <= types


def parameter_set_prefix(data: bytes) -> bytes:
    """The bytes before the first VCL NAL (type < 32) of a packet, i.e. its VPS/SPS/PPS/SEI header."""
    for header_offset, nal_type in nal_units(data):
        if nal_type < 32:
            start = header_offset - 3
            if start > 0 and data[start - 1] == 0:     # 4-byte start code 00 00 00 01
                start -= 1
            return data[:start]
    return b""


def _ts_ns(ts) -> int:
    return int(ts.seconds) * 1_000_000_000 + int(ts.nanos)


def _message_dict(message) -> dict[str, Any]:
    from google.protobuf.json_format import MessageToDict  # noqa: PLC0415

    return MessageToDict(message, preserving_proto_field_name=True)


def _without_timestamp(d: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != "timestamp"}


def _messages(stream: IO[bytes], topics: Sequence[str]):
    """Decoded messages of `topics` in file order, chunk CRCs validated."""
    from mcap.reader import NonSeekingReader  # noqa: PLC0415
    from mcap_protobuf.decoder import DecoderFactory  # noqa: PLC0415

    reader = NonSeekingReader(stream, validate_crcs=True, decoder_factories=[DecoderFactory()])
    return reader.iter_decoded_messages(topics=list(topics), log_time_order=False)


def segment_dir(stream_dir: Path, number: int) -> Path:
    return stream_dir / f"seg_{number:05d}"


def extract_segment(stream: IO[bytes], number: int, name: str, stream_dir: Path, topics: Mapping[str, Any], *,
                    video_format: str, last_times: Mapping[str, int | None]) -> dict[str, Any]:
    """Index one segment into ``seg_<number>/`` (atomically). ``last_times``: last indexed frame/IMU time
    of the previous segment per key ('cam0', 'cam1', 'imu'), used to drop overlapping duplicates."""
    final = segment_dir(stream_dir, number)
    tmp = stream_dir / f".tmp_seg_{number:05d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    camera_topic = dict(zip(topics["cameras"], CAMERA_KEYS))
    calibration_topics = {topics["intrinsics"][0]: "cam0_intrinsics", topics["intrinsics"][1]: "cam1_intrinsics",
                          topics["stereo_extrinsics"]: "stereo_extrinsics", topics["imu_info"]: "imu_info",
                          topics["imu_extrinsics"]: "imu_extrinsics"}
    wanted = list(camera_topic) + [topics["imu_raw"]] + list(calibration_topics)

    frames: dict[str, list[tuple]] = {k: [] for k in CAMERA_KEYS}
    ordinal = {k: 0 for k in CAMERA_KEYS}
    previous = {k: last_times.get(k) for k in CAMERA_KEYS}
    stats: dict[str, Any] = {k: {"messages": 0, "duplicates_dropped": 0, "non_monotonic": 0, "wrong_format": 0,
                                 "keyframes": 0, "bytes": 0} for k in CAMERA_KEYS}
    imu_times: list[int] = []
    imu_rows: list[tuple] = []
    imu_prev = last_times.get("imu")
    imu_stats = {"messages": 0, "samples": 0, "duplicates_dropped": 0, "non_monotonic": 0}
    calibration: dict[str, dict] = {}
    calibration_changes: dict[str, int] = {}
    try:
        for _schema, channel, message, decoded in _messages(stream, wanted):
            topic = channel.topic
            if topic in camera_topic:
                cam = camera_topic[topic]
                st = stats[cam]
                st["messages"] += 1
                index = ordinal[cam]
                ordinal[cam] += 1
                t = _ts_ns(decoded.timestamp) or int(message.log_time)
                if decoded.format != video_format:
                    st["wrong_format"] += 1
                data = decoded.data
                if previous[cam] is not None and t <= previous[cam]:
                    if not frames[cam]:
                        st["duplicates_dropped"] += 1      # overlap with the previous segment
                        continue
                    st["non_monotonic"] += 1
                key, params = frame_flags(data)
                st["keyframes"] += int(key)
                st["bytes"] += len(data)
                frames[cam].append((t, key, params, number, index, len(data)))
                previous[cam] = t
            elif topic == topics["imu_raw"]:
                imu_stats["messages"] += 1
                base = _ts_ns(decoded.first_sample_time)
                for s in decoded.samples:
                    t = base + int(s.t_offset_ns)
                    if imu_prev is not None and t <= imu_prev:
                        if not imu_times:
                            imu_stats["duplicates_dropped"] += 1
                            continue
                        imu_stats["non_monotonic"] += 1
                    g, a, m = s.angular_velocity, s.linear_acceleration, s.magnetic_field
                    temp = float(s.temperature_c) if s.HasField("temperature_c") else math.nan
                    imu_times.append(t)
                    imu_rows.append((g.x, g.y, g.z, a.x, a.y, a.z, m.x, m.y, m.z, temp))
                    imu_prev = t
                imu_stats["samples"] = len(imu_times)
            else:
                cal = calibration_topics[topic]
                value = _message_dict(decoded)
                if cal not in calibration:
                    calibration[cal] = value
                    calibration_changes[cal] = 0
                elif _without_timestamp(value) != _without_timestamp(calibration[cal]):
                    calibration_changes[cal] += 1
    except IngestionError:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001 (corrupt file, CRC mismatch, unknown schema, network)
        shutil.rmtree(tmp, ignore_errors=True)
        raise IngestionError(f"Cannot read MCAP segment {name}: {type(exc).__name__}: {exc}") from exc

    summary: dict[str, Any] = {"number": number, "file": name, "calibration": calibration,
                               "calibration_changes": calibration_changes, "imu": imu_stats}
    for cam in CAMERA_KEYS:
        table = np.array(frames[cam], dtype=FRAME_DTYPE)
        np.save(tmp / f"{cam}.npy", table)
        summary[cam] = {**stats[cam], "frames": int(len(table)),
                        "first_t_ns": int(table["t_ns"][0]) if len(table) else None,
                        "last_t_ns": int(table["t_ns"][-1]) if len(table) else None}
    np.save(tmp / "imu_t.npy", np.array(imu_times, dtype="<i8"))
    np.save(tmp / "imu.npy", np.array(imu_rows, dtype="<f4").reshape(-1, len(IMU_COLUMNS)))
    summary["imu"].update(first_t_ns=imu_times[0] if imu_times else None, last_t_ns=imu_times[-1] if imu_times else None)
    summary["missing_topics"] = [t for t, present in (
        (topics["cameras"][0], stats["cam0"]["messages"] > 0),
        (topics["cameras"][1], stats["cam1"]["messages"] > 0),
        (topics["imu_raw"], imu_stats["messages"] > 0)) if not present]
    write_json_atomic(tmp / "segment.json", summary)
    if final.exists():
        shutil.rmtree(final)
    tmp.replace(final)
    return summary


def load_segment_summary(stream_dir: Path, number: int) -> dict[str, Any] | None:
    path = segment_dir(stream_dir, number) / "segment.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def last_times_after(summary: Mapping[str, Any] | None, previous: Mapping[str, int | None]) -> dict[str, int | None]:
    """Carry the last indexed times forward past one segment."""
    if summary is None:
        return dict(previous)
    result = dict(previous)
    for cam in CAMERA_KEYS:
        if summary[cam]["last_t_ns"] is not None:
            result[cam] = summary[cam]["last_t_ns"]
    if summary["imu"].get("last_t_ns") is not None:
        result["imu"] = summary["imu"]["last_t_ns"]
    return result


def assemble_session(stream_dir: Path, numbers: Sequence[int]) -> dict[str, Any]:
    """Concatenate per-segment tables into session tables (cam0_all.npy, cam1_all.npy, imu_t.npy, imu.npy).
    The technical step pairs the eyes and writes the paired rows as cam0.npy / cam1.npy."""
    out: dict[str, Any] = {}
    for cam in CAMERA_KEYS:
        parts = [np.load(segment_dir(stream_dir, n) / f"{cam}.npy") for n in numbers]
        table = np.concatenate(parts) if parts else np.zeros(0, dtype=FRAME_DTYPE)
        np.save(stream_dir / f"{cam}_all.npy", table)
        out[cam] = int(len(table))
    times = [np.load(segment_dir(stream_dir, n) / "imu_t.npy") for n in numbers]
    rows = [np.load(segment_dir(stream_dir, n) / "imu.npy") for n in numbers]
    np.save(stream_dir / "imu_t.npy", np.concatenate(times) if times else np.zeros(0, dtype="<i8"))
    np.save(stream_dir / "imu.npy", np.concatenate(rows) if rows else np.zeros((0, len(IMU_COLUMNS)), dtype="<f4"))
    out["imu_samples"] = int(sum(len(t) for t in times))
    return out


def stream_packets(open_segment: SegmentOpener, camera_topics: Sequence[str],
                   requests: Mapping[str, tuple[np.ndarray, int, int]],
                   sink: Callable[[str, int, bytes], None]) -> dict[str, int]:
    """Stream the packets of table rows [start, stop) for each requested camera, in table order.

    ``requests``: cam -> (frame table, start row, stop row). ``sink(cam, row, data)`` receives every
    packet. Only the segments holding requested rows are opened, in order. Every packet's size must
    match the table. Returns the number of packets delivered per camera.
    """
    topic_cam = dict(zip(camera_topics, CAMERA_KEYS))
    wanted: dict[int, dict[str, dict[int, int]]] = {}       # segment -> cam -> ordinal -> row
    for cam, (table, start, stop) in requests.items():
        rows = table[start:stop]
        for row, (segment, ordinal) in enumerate(zip(rows["segment"].tolist(), rows["ordinal"].tolist()), start=start):
            wanted.setdefault(segment, {}).setdefault(cam, {})[ordinal] = row
    delivered = {cam: 0 for cam in requests}
    expected_next = {cam: start for cam, (_t, start, _s) in requests.items()}
    for segment in sorted(wanted):
        per_cam = wanted[segment]
        topics = [t for t, cam in topic_cam.items() if cam in per_cam]
        seen = {cam: 0 for cam in per_cam}
        remaining = sum(len(v) for v in per_cam.values())
        try:
            with open_segment(segment) as stream:
                for _schema, channel, _message, decoded in _messages(stream, topics):
                    cam = topic_cam[channel.topic]
                    ordinal = seen[cam]
                    seen[cam] += 1
                    row = per_cam[cam].get(ordinal)
                    if row is None:
                        continue
                    table = requests[cam][0]
                    data = decoded.data
                    if row != expected_next[cam] or len(data) != int(table["size"][row]):
                        raise IngestionError(
                            f"segment {segment} {cam}: packet for row {row} does not match the index "
                            f"(expected row {expected_next[cam]}, {int(table['size'][row])} bytes; got {len(data)})")
                    sink(cam, row, data)
                    expected_next[cam] = row + 1
                    delivered[cam] += 1
                    remaining -= 1
                    if remaining == 0:
                        break
        except IngestionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Reading segment {segment} failed: {type(exc).__name__}: {exc}") from exc
        if remaining:
            raise IngestionError(f"segment {segment}: {remaining} indexed packets were not found when streaming it")
    for cam, (_table, start, stop) in requests.items():
        if delivered[cam] != stop - start:
            raise IngestionError(f"{cam}: streamed {delivered[cam]} packets, expected {stop - start}")
    return delivered
