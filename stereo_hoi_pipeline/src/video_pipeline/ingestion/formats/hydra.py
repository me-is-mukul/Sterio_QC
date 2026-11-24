"""Hydra sessions (S_YYYYMMDD_HHMMSS/): one side-by-side HEVC elementary stream for both eyes.

  video.h265            Annex-B HEVC, 3840x1200 (left eye x 0-1920, right eye x 1920-3840)
  frames.idx            CSV: frame, host_ns, pts_raw, offset, len  (byte range of every frame)
  h265_frame_meta.csv   per frame and eye: frame_pts_us = sensor capture time (us)
  h265_imu.csv          gyro (deg/s) and accel (g) samples, pts_us on the same sensor clock
  meta.json, session.json, stereo_calibration.bin (binary, format unknown: not used)

The single video file can be 100 GB, so it is indexed in chunks of `chunk_frames` frames (each a
segment of the index, resumable). The CSV files are parsed while streaming; nothing big is stored.
"""

from __future__ import annotations

import csv
import io
import logging
import math
import time
from array import array
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from video_pipeline.core.errors import IngestionError
from video_pipeline.ingestion.formats.base import (
    RecordingReader,
    file_checks,
    frame_table,
    imu_rows,
    interpolate_onto,
    load_summary,
    write_segment,
)
from video_pipeline.ingestion.mcap_reader import IMU_COLUMNS, frame_flags, segment_dir

logger = logging.getLogger("video_pipeline.ego")
G = 9.80665


class HydraReader(RecordingReader):
    format = "hydra"
    layout = "sbs"
    codec = "hevc"
    description = "Hydra (side-by-side video.h265 + frames.idx + CSV sensor files)"
    metadata_files = ("session.json", "meta.json")

    def fingerprint_extra(self) -> Any:
        return {"hydra": self.cfg.get("formats.hydra")}

    def required_files(self) -> list[dict[str, Any]]:
        video = bool(self.listing.segments)
        return [{"check": "file_present:video.h265", "passed": video,
                 "message": "video.h265 present" if video else "video.h265 missing"}] + \
            file_checks(self.listing.files, ["frames.idx", "h265_frame_meta.csv"])

    # --- session-level index (parsed once, cached in work_dir) --------------------------------------
    def _csv(self, name: str):
        raw = self.open(self.listing.files[name])
        return raw, csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))

    def _session_index(self) -> dict[str, np.ndarray]:
        cache = self.work_dir / "hydra_index.npz"
        if cache.is_file():
            return dict(np.load(cache))
        t0 = time.perf_counter()
        raw, reader = self._csv("frames.idx")
        values = array("q")
        with raw:
            header = next(reader)
            cols = [header.index(c) for c in ("frame", "pts_raw", "offset", "len")]
            for r in reader:
                if r:
                    values.extend(int(r[c]) for c in cols)
        idx = np.frombuffer(values, dtype=np.int64).reshape(-1, 4).copy()
        order = np.argsort(idx[:, 0], kind="stable")
        idx = idx[order]
        n = len(idx)
        left = np.full(n, -1, dtype=np.int64)
        right = np.full(n, -1, dtype=np.int64)
        frames = idx[:, 0]
        dense = bool(n) and np.array_equal(frames, np.arange(n))      # frame numbers 0..n-1 (the usual case)
        position = None if dense else {int(f): i for i, f in enumerate(frames)}
        raw, reader = self._csv("h265_frame_meta.csv")
        with raw:
            header = next(reader)
            i_f, i_side, i_pts = header.index("video_frame"), header.index("side"), header.index("frame_pts_us")
            for r in reader:
                if not r:
                    continue
                f = int(r[i_f])
                i = (f if 0 <= f < n else None) if dense else position.get(f)
                if i is not None:
                    (left if r[i_side] == "left" else right)[i] = int(r[i_pts])
        # typed arrays, not lists of tuples: a long recording has tens of millions of IMU rows
        gyro_t, gyro, acc_t, acc = array("q"), array("d"), array("q"), array("d")
        expected_units = {"gyro": "deg/s", "accel": "g"}
        if "h265_imu.csv" in self.listing.files:
            raw, reader = self._csv("h265_imu.csv")
            with raw:
                header = next(reader)
                i_s, i_t = header.index("sensor"), header.index("pts_us")
                i_x, i_y, i_z, i_u = (header.index(c) for c in ("x_scaled", "y_scaled", "z_scaled", "units"))
                for r in reader:
                    if len(r) <= i_u or not r[i_x]:
                        continue
                    sensor = r[i_s]
                    if sensor == "gyro":
                        times, values = gyro_t, gyro
                    elif sensor == "accel":
                        times, values = acc_t, acc
                    else:
                        continue
                    if r[i_u] != expected_units[sensor]:
                        raise IngestionError(f"h265_imu.csv: unexpected {sensor} units {r[i_u]!r}")
                    times.append(int(r[i_t]))
                    values.extend((float(r[i_x]), float(r[i_y]), float(r[i_z])))
        data = {"frame": idx[:, 0], "pts_raw": idx[:, 1], "offset": idx[:, 2], "len": idx[:, 3],
                "left_us": left, "right_us": right,
                "gyro_t_us": np.frombuffer(gyro_t, dtype=np.int64).copy(),
                "gyro": np.frombuffer(gyro, dtype=np.float64).reshape(-1, 3).copy(),
                "acc_t_us": np.frombuffer(acc_t, dtype=np.int64).copy(),
                "acc": np.frombuffer(acc, dtype=np.float64).reshape(-1, 3).copy()}
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **data)
        logger.info("Hydra index: %d frames, %d gyro + %d accel samples (parsed in %.1f s)", n, len(gyro_t), len(acc_t),
                    time.perf_counter() - t0)
        return data

    @staticmethod
    def _eye_times(index: Mapping[str, np.ndarray], key: str) -> tuple[np.ndarray, int]:
        """Sensor capture time (ns) per frame; frames without a frame_meta row get pts_raw + the median
        offset between the two clocks (counted and reported)."""
        us = index[key].copy()
        known = us >= 0
        missing = int((~known).sum())
        if missing and known.any():
            offset = int(np.median(us[known] - index["pts_raw"][known]))
            us[~known] = index["pts_raw"][~known] + offset
        elif missing:
            us = index["pts_raw"].copy()
        return us * 1000, missing

    def _imu(self, index: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        t_us, gyro = index["gyro_t_us"], index["gyro"]
        if len(t_us) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros((0, len(IMU_COLUMNS)), dtype=np.float32)
        order = np.argsort(t_us, kind="stable")
        t_us, gyro = t_us[order], gyro[order]
        keep = np.concatenate([[True], np.diff(t_us) > 0])
        t_us, gyro = t_us[keep], gyro[keep]
        acc = None
        if len(index["acc_t_us"]):
            o = np.argsort(index["acc_t_us"], kind="stable")
            acc = interpolate_onto(t_us.astype(np.float64), index["acc_t_us"][o].astype(np.float64), index["acc"][o]) * G
        return t_us * 1000, imu_rows(gyro * (math.pi / 180.0), acc)

    def extract(self, stream_dir: Path) -> list[int]:
        if not self.listing.segments:
            return []
        obj = self.listing.segments[0].files["video"]
        index = self._session_index()
        n = len(index["frame"])
        if n == 0:
            raise IngestionError("frames.idx lists no frames")
        end = int(index["offset"][-1] + index["len"][-1])
        if end > obj.size:
            raise IngestionError(f"frames.idx points beyond the end of video.h265 ({end} > {obj.size} bytes)")
        t0_ns, missing0 = self._eye_times(index, "left_us")
        t1_ns, missing1 = self._eye_times(index, "right_us")
        imu_t, imu = self._imu(index)
        chunk = int(self.cfg.get("formats.hydra.chunk_frames"))
        bounds = list(range(0, n, chunk))
        if self.max_segments is not None:
            bounds = bounds[:self.max_segments]
        identity = [obj.name, obj.generation, n]
        numbers = []
        for number, a in enumerate(bounds):
            b = min(a + chunk, n)
            numbers.append(number)
            summary = load_summary(stream_dir, number)
            if summary is not None and summary.get("objects") == identity:
                continue
            started = time.perf_counter()
            keys, params, bad = self._flags(obj, index["offset"][a:b], index["len"][a:b])
            lo = 0 if number == 0 else int(np.searchsorted(imu_t, t0_ns[a]))
            hi = len(imu_t) if b >= n else int(np.searchsorted(imu_t, t0_ns[b]))
            missing = []
            if bad:
                missing.append(f"{bad} frames of video.h265 do not start with an Annex-B start code")
            if number == 0 and (missing0 or missing1):
                missing.append(f"h265_frame_meta.csv lacks {missing0} left / {missing1} right frame times "
                               "(taken from frames.idx pts_raw)")
            if number == 0 and len(imu_t) == 0:
                missing.append("h265_imu.csv (no IMU samples)")
            sizes = index["len"][a:b]
            summary = write_segment(
                stream_dir, number, f"video.h265 frames {a}-{b - 1}",
                {"cam0": frame_table(t0_ns[a:b], keys, params, number, sizes),
                 "cam1": frame_table(t1_ns[a:b], keys, params, number, sizes)},
                imu_t[lo:hi], imu[lo:hi], missing=missing, extra_arrays={"video_offsets": index["offset"][a:b]},
                info={"objects": identity, "frames": [int(a), int(b)], "width": self._width()})
            logger.info("video.h265 chunk %d/%d: frames %d-%d, %.0f MB, %d keyframes | %.1f s", number + 1, len(bounds),
                        a, b - 1, sizes.sum() / 1e6, int(keys.sum()), time.perf_counter() - started)
        return numbers

    def _flags(self, obj, offsets: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        keys = np.zeros(len(offsets), dtype=bool)
        params = np.zeros(len(offsets), dtype=bool)
        bad = 0
        with self.open(obj) as stream:
            position = None
            for i, (off, ln) in enumerate(zip(offsets.tolist(), lengths.tolist())):
                if position != off:
                    stream.seek(off)
                data = stream.read(ln)
                if len(data) != ln:
                    raise IngestionError(f"video.h265: short read at byte {off}")
                position = off + ln
                if not (data.startswith(b"\x00\x00\x01") or data.startswith(b"\x00\x00\x00\x01")):
                    bad += 1
                keys[i], params[i] = frame_flags(data)
        return keys, params, bad

    def _width(self) -> int:
        meta = self.kept_json("meta.json") or {}
        return int((meta.get("video") or {}).get("width") or self.cfg.get("formats.hydra.default_width"))

    def eye_crops(self, stream_dir: Path) -> dict[str, list[int]]:
        width = self._width()
        return {"cam0": [0, width // 2], "cam1": [width // 2, width]}

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
                    raise IngestionError(f"video.h265 chunk {segment}: frame {ordinal} is {len(data)} bytes, index says {size}")
                position = offset + size
                yield cam, row, data
