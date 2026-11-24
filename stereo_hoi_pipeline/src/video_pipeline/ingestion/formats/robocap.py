"""Robocap sessions: robocap_segmentN_<file> for N = 1, 2, ... (about 10 minutes each).

Per segment: six HEVC MP4 cameras (video_left/right, video_left/right_eye, video_left/right_front),
SQLite IMU files (imu_left.db, imu_right.db: gyro_data + acc_data in raw sensor units with
nanosecond device timestamps) and a magnetometer (mag_middle.db). The QA uses one camera pair
(formats.robocap.cameras, default the _front pair) and one IMU file.

Timing: every MP4 stores its first frame's device time in its 'comment' tag, in microseconds, on
the same clock as the IMU timestamps. Frame times = that start + the MP4 presentation times.

IMU units (ICM-42688-P): accelerometer 8192 LSB/g (+-4 g, checked: 1 g at rest), gyroscope
65.5 LSB/(deg/s) (+-500 deg/s; estimated from the recordings by fitting the rotation of the
gravity vector, 58-71 LSB/(deg/s), and matching the sensor's standard ranges).
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from video_pipeline.core.errors import IngestionError
from video_pipeline.ingestion.formats.base import (
    RecordingReader,
    contiguous_checks,
    empty_table,
    frame_table,
    imu_rows,
    interpolate_onto,
    load_summary,
    write_segment,
)
from video_pipeline.ingestion.formats.mp4_common import HEVC_SAMPLE_ENTRIES, index_mp4, mp4_info, stream_mp4_segment
from video_pipeline.ingestion.mcap_reader import CAMERA_KEYS, IMU_COLUMNS

logger = logging.getLogger("video_pipeline.ego")
G = 9.80665


def read_imu_db(path: Path, gyro_lsb_per_dps: float, accel_lsb_per_g: float) -> tuple[np.ndarray, np.ndarray]:
    """(t_ns, IMU_COLUMNS rows) from a robocap IMU database; accel is interpolated onto the gyro times."""
    con = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        gyro = np.array(con.execute("SELECT timestamp, x, y, z FROM gyro_data ORDER BY timestamp").fetchall(),
                        dtype=np.float64).reshape(-1, 4)
        acc = np.array(con.execute("SELECT timestamp, x, y, z FROM acc_data ORDER BY timestamp").fetchall(),
                       dtype=np.float64).reshape(-1, 4)
    finally:
        con.close()
    if len(gyro) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, len(IMU_COLUMNS)), dtype=np.float32)
    t = gyro[:, 0].astype(np.int64)
    keep = np.concatenate([[True], np.diff(t) > 0])            # drop repeated timestamps
    t, gyro = t[keep], gyro[keep]
    gyro_rad = gyro[:, 1:4] / gyro_lsb_per_dps * (math.pi / 180.0)
    accel = interpolate_onto(t.astype(np.float64), acc[:, 0], acc[:, 1:4]) / accel_lsb_per_g * G if len(acc) else None
    return t, imu_rows(gyro_rad, accel)


class RobocapReader(RecordingReader):
    format = "robocap"
    layout = "pair"
    description = "robocap (six MP4 cameras + SQLite IMU per segment)"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        s = self.cfg.get("formats.robocap")
        self.cameras: list[str] = list(s["cameras"])
        self.imu_file: str = s["imu_file"]
        self.gyro_lsb_per_dps = float(s["gyro_lsb_per_dps"])
        self.accel_lsb_per_g = float(s["accel_lsb_per_g"])

    def fingerprint_extra(self) -> Any:
        return {"robocap": self.cfg.get("formats.robocap")}

    def required_files(self) -> list[dict[str, Any]]:
        return contiguous_checks([s.number for s in self.segments()], 1, "robocap_segment1")

    def extract(self, stream_dir: Path) -> list[int]:
        numbers = []
        segments = self.segments()
        last_imu: int | None = None
        for position, seg in enumerate(segments, start=1):
            numbers.append(seg.number)
            name = seg.name
            summary = load_summary(stream_dir, seg.number)
            identity = {k: [o.name, o.generation] for k, o in sorted(seg.files.items())}
            if summary is None or summary.get("objects") != identity:
                t0 = time.perf_counter()
                summary = self._extract_segment(stream_dir, seg, last_imu, identity)
                logger.info("Segment %d/%d (robocap_segment%d, %.0f MB): %d+%d frames, %d IMU samples%s | %.1f s",
                            position, len(segments), seg.number, seg.size / 1e6, summary["cam0"]["frames"],
                            summary["cam1"]["frames"], summary["imu"]["samples"],
                            f" | missing: {summary['missing_topics']}" if summary["missing_topics"] else "",
                            time.perf_counter() - t0)
            if summary["imu"]["last_t_ns"] is not None:
                last_imu = summary["imu"]["last_t_ns"]
        return numbers

    def _extract_segment(self, stream_dir: Path, seg, last_imu: int | None, identity) -> dict[str, Any]:
        tables: dict[str, np.ndarray] = {}
        missing: list[str] = []
        arrays: dict[str, np.ndarray] = {}
        mp4: dict[str, Any] = {}
        camera_extra: dict[str, dict[str, Any]] = {}
        device: dict[str, str] = {}
        for cam, role in zip(CAMERA_KEYS, self.cameras):
            obj = seg.files.get(role)
            if obj is None:
                missing.append(role)
                tables[cam] = empty_table()
                continue
            try:
                track = index_mp4(self, obj)
            except IngestionError as exc:
                missing.append(f"{role} ({exc})")
                tables[cam] = empty_table()
                continue
            start = track.tags.get("comment", "").strip()
            if not start.isdigit():
                missing.append(f"{role} (no start time in its 'comment' tag)")
                tables[cam] = empty_table()
                continue
            device = {k: track.tags[k] for k in ("product", "version", "deviceid", "camera", "imu") if k in track.tags}
            rel = track.pts - track.pts[0]
            t_ns = int(start) * 1000 + np.round(rel * (1e9 / track.timescale)).astype(np.int64)
            tables[cam] = frame_table(t_ns, track.keys, track.keys, seg.number, track.sizes)
            arrays[f"{cam}_offsets"] = track.offsets
            mp4[cam] = mp4_info(track, obj)
            camera_extra[cam] = {"wrong_format": 0 if track.codec in HEVC_SAMPLE_ENTRIES else track.samples}
        imu_t = np.zeros(0, dtype=np.int64)
        imu = np.zeros((0, len(IMU_COLUMNS)), dtype=np.float32)
        imu_obj = seg.files.get(self.imu_file)
        if imu_obj is None:
            missing.append(self.imu_file)
        else:
            local = self.work_dir / "tmp" / f"seg{seg.number}_{self.imu_file}"
            try:
                self.storage.download(self.bucket, imu_obj, local)
                imu_t, imu = read_imu_db(local, self.gyro_lsb_per_dps, self.accel_lsb_per_g)
            except (sqlite3.Error, IngestionError) as exc:
                missing.append(f"{self.imu_file} ({type(exc).__name__}: {exc})")
            finally:
                local.unlink(missing_ok=True)
            if last_imu is not None and len(imu_t):
                keep = imu_t > last_imu
                imu_t, imu = imu_t[keep], imu[keep]
            if len(imu_t) == 0 and self.imu_file not in " ".join(missing):
                missing.append(f"{self.imu_file} (no samples)")
        return write_segment(stream_dir, seg.number, seg.name, tables, imu_t, imu, missing=missing,
                             extra_arrays=arrays, camera_extra=camera_extra,
                             info={"objects": identity, "mp4": mp4, "device": device})

    def _segment_packets(self, stream_dir: Path, segment: int, rows: Mapping[str, Sequence[tuple[int, int]]]
                         ) -> Iterator[tuple[str, int, bytes]]:
        return stream_mp4_segment(self, stream_dir, segment, rows, {cam: cam for cam in rows})
