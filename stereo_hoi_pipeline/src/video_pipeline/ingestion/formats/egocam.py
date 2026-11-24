"""EGO-CAM sessions: one side-by-side picture (4000x1200) holds both eyes; calibration.json gives each
eye's x range ('eye_crop_x', cam0 = 160-2080, cam1 = 2080-4000; the strip at x < 160 carries the
frame's barcode with the embedded IMU samples).

Two containers:
* egocam_live: video_live_NNNN.mp4, 30 s HEVC MP4 segments; frame_map.csv gives every frame's
  host receive time (segment, frame_in_segment, capture_seq, host_recv_us).
* egocam_egoc: capture.egoc, one file of records: 'EGF2', u32 sequence, u64 host time (us),
  u32 length, u32 checksum (algorithm unknown, not verified), then one JPEG (little-endian).

The IMU is only in the barcode of every frame; decoding it needs the device team's tool
(tools/imu_from_video.py), which the pipeline does not have yet: IMU checks are skipped and the
report says so.
"""

from __future__ import annotations

import csv
import io
import logging
import struct
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from video_pipeline.core.errors import IngestionError
from video_pipeline.ingestion.formats.base import (
    IMU_IN_VIDEO,
    RecordingReader,
    check,
    contiguous_checks,
    frame_table,
    load_summary,
    write_segment,
)
from video_pipeline.ingestion.formats.mp4_common import HEVC_SAMPLE_ENTRIES, index_mp4, mp4_info, stream_mp4_segment
from video_pipeline.ingestion.mcap_reader import IMU_COLUMNS, segment_dir

logger = logging.getLogger("video_pipeline.ego")
_EGOC_HEADER = struct.Struct("<4sIQII")
_NO_IMU = (np.zeros(0, dtype=np.int64), np.zeros((0, len(IMU_COLUMNS)), dtype=np.float32))


def rotation_to_quaternion(r: np.ndarray) -> dict[str, float]:
    r = np.asarray(r, dtype=np.float64).reshape(3, 3)
    trace = np.trace(r)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    return {"x": float(x), "y": float(y), "z": float(z), "w": float(w)}


def calibration_messages(cal: Mapping[str, Any] | None) -> dict[str, Any]:
    """calibration.json (OpenCV stereo, Kannala-Brandt) -> the intrinsics/extrinsics messages the
    calibration check reads (same shape as the MCAP calibration topics)."""
    if not cal or "cam0" not in cal or "cam1" not in cal or "stereo" not in cal:
        return {}
    width, height = (int(v) for v in cal.get("image_size", [0, 0]))
    out: dict[str, Any] = {}
    for cam in ("cam0", "cam1"):
        c = cal[cam]
        out[f"{cam}_intrinsics"] = {"K": [float(v) for row in c["K"] for v in row], "D": [float(v) for v in c["dist"]],
                                    "width": width, "height": height,
                                    "distortion_model": cal.get("distortion_model", "kannala_brandt"),
                                    "position": c.get("position")}
    t = [float(v) for v in cal["stereo"]["T"]]
    out["stereo_extrinsics"] = {"translation": {"x": t[0], "y": t[1], "z": t[2]},
                                "rotation": rotation_to_quaternion(np.array(cal["stereo"]["R"])),
                                "source": "calibration.json stereo R/T (OpenCV: X_cam1 = R X_cam0 + T)"}
    return out


class _EgoCam(RecordingReader):
    layout = "sbs"
    imu_unavailable = IMU_IN_VIDEO
    host_clock = True                        # frame times are host receive times
    metadata_files = ("calibration.json", "metadata.json", "encode_gaps.json")

    def default_crops(self, width: int) -> dict[str, list[int]]:
        default = self.cfg.get("formats.egocam.default_eye_crop_x")
        if width == int(default[1][1]):
            return {"cam0": list(default[0]), "cam1": list(default[1])}
        return {"cam0": [0, width // 2], "cam1": [width // 2, width]}

    def eye_crops(self, stream_dir: Path) -> dict[str, list[int]] | None:
        cal = self.kept_json("calibration.json") or {}
        crops = cal.get("eye_crop_x")
        if crops and "cam0" in crops and "cam1" in crops:
            return {"cam0": [int(v) for v in crops["cam0"]], "cam1": [int(v) for v in crops["cam1"]]}
        first = load_summary(stream_dir, 0) or {}
        return self.default_crops(int(first.get("width") or self.cfg.get("formats.egocam.default_eye_crop_x")[1][1]))

    def calibration(self) -> dict[str, Any]:
        return calibration_messages(self.kept_json("calibration.json"))

    def video_missing_check(self) -> list[dict[str, Any]]:
        if self.listing.segments:
            return []
        return [check("video_present", False, "no video file (capture.egoc or video_live_*.mp4) in the session")]


class EgoCamLiveReader(_EgoCam):
    format = "egocam_live"
    codec = "hevc"
    frame_counter = True                     # frame_map.csv capture_seq counts every captured frame
    description = "EGO-CAM live (side-by-side HEVC video_live_NNNN.mp4 + frame_map.csv)"

    def required_files(self) -> list[dict[str, Any]]:
        missing_video = self.video_missing_check()
        fm = "frame_map.csv" in self.listing.files
        return missing_video + [check("frame_map_present", fm, "frame_map.csv present" if fm else "frame_map.csv missing")] + \
            ([] if missing_video else contiguous_checks([s.number for s in self.segments()], 0, "video_live_0000.mp4"))

    def _frame_map(self) -> dict[int, np.ndarray]:
        """segment -> (n, 2) array [host receive time (us), capture_seq] by frame_in_segment (-1 where a
        row is missing), parsed once and cached."""
        cache = self.work_dir / "frame_map.npz"
        if cache.is_file():
            data = np.load(cache)
            return {int(k[4:]): data[k] for k in data.files}
        obj = self.listing.files["frame_map.csv"]
        rows: dict[int, list[tuple[int, int, int]]] = {}
        with self.open(obj) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
            header = next(reader)
            i_seg, i_frame = header.index("segment"), header.index("frame_in_segment")
            i_t, i_seq = header.index("host_recv_us"), header.index("capture_seq")
            for r in reader:
                if r:
                    rows.setdefault(int(r[i_seg]), []).append((int(r[i_frame]), int(r[i_t]), int(r[i_seq])))
        out = {}
        for seg, items in rows.items():
            frames = np.array([f for f, _, _ in items], dtype=np.int64)
            table = np.full((int(frames.max()) + 1 if len(frames) else 0, 2), -1, dtype=np.int64)
            table[frames, 0] = [t for _, t, _ in items]
            table[frames, 1] = [q for _, _, q in items]
            out[seg] = table
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **{f"seg_{k}": v for k, v in out.items()})
        return out

    def extract(self, stream_dir: Path) -> list[int]:
        frame_map = self._frame_map()
        numbers = []
        segments = self.segments()
        last_t: int | None = None
        for position, seg in enumerate(segments, start=1):
            numbers.append(seg.number)
            obj = seg.files["video"]
            identity = [obj.name, obj.generation]
            summary = load_summary(stream_dir, seg.number)
            if summary is None or summary.get("objects") != identity:
                t0 = time.perf_counter()
                summary = self._extract_segment(stream_dir, seg, obj, frame_map.get(seg.number), last_t, identity)
                if position == 1 or position % 20 == 0 or position == len(segments):
                    logger.info("Segment %d/%d (%s, %.0f MB): %d frames%s | %.1f s", position, len(segments), seg.name,
                                obj.size / 1e6, summary["cam0"]["frames"],
                                f" | {summary['missing_topics']}" if summary["missing_topics"] else "",
                                time.perf_counter() - t0)
            if summary["cam0"]["last_t_ns"] is not None:
                last_t = summary["cam0"]["last_t_ns"]
        return numbers

    def _extract_segment(self, stream_dir: Path, seg, obj, frame_map: np.ndarray | None, last_t: int | None,
                         identity) -> dict[str, Any]:
        missing: list[str] = []
        track = index_mp4(self, obj)
        n = track.samples
        fps = float(self.cfg.get("technical.expected_fps"))
        rel_ns = np.round((track.pts - track.pts[0]) * (1e9 / track.timescale)).astype(np.int64)
        timing = "frame_map.csv host receive time"
        host_us = frame_map[:, 0] if frame_map is not None else None
        if host_us is not None and len(host_us) == n and np.all(frame_map >= 0):
            t_ns = host_us * 1000
            seq = frame_map[:, 1].copy()
        else:
            got = 0 if host_us is None else int(np.sum(host_us >= 0))
            missing.append(f"frame_map.csv has {got} rows for this segment, the video {n} frames")
            anchor = int(host_us[host_us >= 0][0]) * 1000 if got else ((last_t + int(1e9 / fps)) if last_t else 0)
            t_ns = anchor + rel_ns
            first_seq = int(frame_map[frame_map[:, 1] >= 0][0, 1]) if got else 0
            seq = np.round(rel_ns * fps / 1e9).astype(np.int64) + first_seq
            timing = "MP4 presentation times (frame_map.csv incomplete)"
        table = frame_table(t_ns, track.keys, track.keys, seg.number, track.sizes)
        extra = {"wrong_format": 0 if track.codec in HEVC_SAMPLE_ENTRIES else n}
        return write_segment(stream_dir, seg.number, seg.name, {"cam0": table, "cam1": table.copy()}, *_NO_IMU,
                             calibration=self.calibration() if seg.number == self.segments()[0].number else None,
                             missing=missing, extra_arrays={"video_offsets": track.offsets, "seq": seq},
                             camera_extra={"cam0": extra, "cam1": extra},
                             info={"objects": identity, "mp4": {"video": mp4_info(track, obj)}, "width": track.width,
                                   "height": track.height, "timing": timing})

    def _segment_packets(self, stream_dir: Path, segment: int, rows: Mapping[str, Sequence[tuple[int, int]]]
                         ) -> Iterator[tuple[str, int, bytes]]:
        return stream_mp4_segment(self, stream_dir, segment, rows, {cam: "video" for cam in rows})


class EgocReader(_EgoCam):
    format = "egocam_egoc"
    codec = "mjpeg"
    # The camera's JPEGs carry an 'AVI1' APP0 block that FFmpeg cannot parse; it logs this at error
    # level for every frame although the picture decodes normally (checked on real frames).
    harmless_decode_errors = ("unable to decode APP fields",)
    description = "EGO-CAM capture.egoc (side-by-side MJPEG records)"

    def fingerprint_extra(self) -> Any:
        return {"egocam": self.cfg.get("formats.egocam")}

    def required_files(self) -> list[dict[str, Any]]:
        return self.video_missing_check()

    def extract(self, stream_dir: Path) -> list[int]:
        """Scan capture.egoc once, record by record, into chunks of `chunk_frames` frames. Each chunk is
        written as a segment with the byte offset where the next one starts, so an interrupted scan
        continues from the last finished chunk."""
        if not self.listing.segments:
            return []
        obj = self.listing.segments[0].files["video"]
        chunk = int(self.cfg.get("formats.egocam.chunk_frames"))
        identity = [obj.name, obj.generation]
        numbers: list[int] = []
        offset = 0
        number = 0
        while True:
            summary = load_summary(stream_dir, number)
            if summary is None or summary.get("objects") != identity:
                break
            numbers.append(number)
            offset = summary["end_offset"]
            number += 1
            if summary.get("end_of_file"):
                return numbers
        started = time.perf_counter()
        with self.open(obj) as stream:
            stream.seek(offset)
            while offset < obj.size:
                if self.max_segments is not None and number >= self.max_segments:
                    break
                t0 = time.perf_counter()
                summary, offset = self._scan_chunk(stream_dir, stream, number, offset, obj.size, chunk, identity)
                numbers.append(number)
                logger.info("capture.egoc chunk %d: %d frames, %.1f of %.1f GB scanned (%.0f%%)%s | %.1f s",
                            number, summary["cam0"]["frames"], offset / 1e9, obj.size / 1e9, 100 * offset / obj.size,
                            "".join(f" | {m}" for m in summary["missing_topics"] + summary.get("notes", [])),
                            time.perf_counter() - t0)
                number += 1
                if summary.get("end_of_file"):
                    break
        logger.info("capture.egoc scanned in %.1f s", time.perf_counter() - started)
        return numbers

    def _scan_chunk(self, stream_dir: Path, stream, number: int, offset: int, size: int, chunk: int, identity
                    ) -> tuple[dict[str, Any], int]:
        t_us, body_offsets, lengths, seqs = [], [], [], []
        problems: list[str] = []            # fail the recording (corrupt data)
        notes: list[str] = []               # reported only (a recording stopped while writing its last frame)
        end_of_file = False
        bad_jpeg = 0
        while len(lengths) < chunk:
            if offset + _EGOC_HEADER.size > size:
                if offset < size:
                    notes.append(f"{size - offset} trailing bytes after the last complete frame (recording cut off)")
                end_of_file = True
                break
            header = stream.read(_EGOC_HEADER.size)
            magic, seq, ts, length, _checksum = _EGOC_HEADER.unpack(header)
            if magic != b"EGF2":
                problems.append(f"capture.egoc is corrupt at byte {offset} (no frame header); the rest is not readable")
                end_of_file = True
                break
            if offset + _EGOC_HEADER.size + length > size:
                notes.append(f"last frame cut off at byte {offset} (recording stopped while writing); "
                             "the frames before it are complete")
                end_of_file = True
                break
            body = stream.read(length)
            if len(body) != length:
                raise IngestionError(f"capture.egoc: short read at byte {offset}")
            if not (body[:2] == b"\xff\xd8" and body[-2:] == b"\xff\xd9"):
                bad_jpeg += 1
            t_us.append(ts)
            body_offsets.append(offset + _EGOC_HEADER.size)
            lengths.append(length)
            seqs.append(seq)
            offset += _EGOC_HEADER.size + length
        if offset >= size:
            end_of_file = True
        n = len(lengths)
        t_ns = np.array(t_us, dtype=np.int64) * 1000
        keys = np.ones(n, dtype=bool)
        table = frame_table(t_ns, keys, keys, number, np.array(lengths, dtype=np.int64))
        seq_arr = np.array(seqs, dtype=np.int64)
        gaps = int(np.sum(np.diff(seq_arr) - 1)) if n > 1 else 0
        if bad_jpeg:
            problems.append(f"{bad_jpeg} records are not complete JPEG pictures")
        width = None
        if number == 0 and n:
            width = 4000                                   # 'GST MJPEG 4000x1200'; crops come from calibration.json
        summary = write_segment(stream_dir, number, f"capture.egoc chunk {number}", {"cam0": table, "cam1": table.copy()},
                                *_NO_IMU, calibration=self.calibration() if number == 0 else None,
                                missing=problems, extra_arrays={"video_offsets": np.array(body_offsets, dtype=np.int64),
                                                                "write_seq": seq_arr},
                                info={"objects": identity, "end_offset": offset, "end_of_file": end_of_file,
                                      "notes": notes,
                                      "sequence_gaps": gaps, "width": width,
                                      "first_sequence": int(seq_arr[0]) if n else None})
        return summary, offset

    def _segment_packets(self, stream_dir: Path, segment: int, rows: Mapping[str, Sequence[tuple[int, int]]]
                         ) -> Iterator[tuple[str, int, bytes]]:
        seg_dir = segment_dir(stream_dir, segment)
        offsets = np.load(seg_dir / "video_offsets.npy")
        obj = self.listing.segments[0].files["video"]
        tables = {cam: np.load(seg_dir / f"{cam}.npy") for cam in rows}
        order = sorted(((ordinal, cam, row) for cam, items in rows.items() for row, ordinal in items))
        with self.open(obj) as stream:
            position = None
            for ordinal, cam, row in order:
                offset, size = int(offsets[ordinal]), int(tables[cam]["size"][ordinal])
                if position != offset:
                    stream.seek(offset)
                data = stream.read(size)
                if len(data) != size:
                    raise IngestionError(f"capture.egoc chunk {segment}: frame {ordinal} is {len(data)} bytes, "
                                         f"index says {size}")
                position = offset + size
                yield cam, row, data
