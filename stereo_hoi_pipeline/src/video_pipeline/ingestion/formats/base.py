"""What every recording format provides to the pipeline.

A reader turns one session folder into the same small index tables, whatever the format:

  stream_dir/seg_NNNNN/cam0.npy, cam1.npy   one row per frame (FRAME_DTYPE): capture time, keyframe
                                            flag, parameter sets present, segment, ordinal, size
                    imu_t.npy, imu.npy      IMU samples (IMU_COLUMNS, SI units), may be empty
                    segment.json            counts, calibration, missing parts

and streams the exact packets of any frame range again later (``stream_packets``). The rest of
the pipeline (decode, checks, labelling, cropping, clips) only uses these.

Layouts:
* ``pair``: cam0 and cam1 are separate video streams (MCAP, robocap).
* ``sbs`` : one video stream holds both eyes side by side (Hydra, EGO-CAM). Both tables then point
  at the same packets (same segment/ordinal); only cam0's rows are streamed, and each eye is cut
  out of the picture with ``eye_crops`` (x range per camera).
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from video_pipeline.core.errors import IngestionError
from video_pipeline.core.io import write_json_atomic
from video_pipeline.ingestion.mcap_reader import CAMERA_KEYS, FRAME_DTYPE, IMU_COLUMNS, segment_dir

logger = logging.getLogger("video_pipeline.ego")

PacketSink = Callable[[str, int, bytes], None]
# cam -> (frame table, first row, stop row)
PacketRequests = Mapping[str, tuple[np.ndarray, int, int]]

IMU_IN_VIDEO = ("IMU is embedded in the video frames (barcode) and no decoder for it is available yet; "
                "IMU checks skipped")


def check(name: str, passed: bool, message: str, **details: Any) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "message": message, **details}


def contiguous_checks(numbers: Sequence[int], first: int, first_name: str) -> list[dict[str, Any]]:
    """Segment numbering starts at `first` and has no gaps."""
    checks = [check("first_segment_present", first in numbers,
                    f"first video segment ({first_name}) present" if first in numbers
                    else f"first video segment {first_name} missing")]
    missing = sorted(set(range(first, max(numbers) + 1)) - set(numbers)) if numbers else []
    checks.append(check("segments_contiguous", bool(numbers) and not missing,
                        "segment numbers are contiguous" if numbers and not missing else
                        ("no segments" if not numbers else f"missing segment numbers {missing[:20]}"),
                        segments=len(numbers), missing_numbers=missing))
    return checks


def file_checks(present: Mapping[str, Any], names: Iterable[str]) -> list[dict[str, Any]]:
    return [check(f"file_present:{n}", n in present, f"{n} present" if n in present else f"{n} missing")
            for n in names]


def empty_table() -> np.ndarray:
    return np.zeros(0, dtype=FRAME_DTYPE)


def frame_table(t_ns: np.ndarray, keys: np.ndarray, params: np.ndarray, segment: int, sizes: np.ndarray) -> np.ndarray:
    table = np.zeros(len(t_ns), dtype=FRAME_DTYPE)
    table["t_ns"] = t_ns
    table["key"] = keys
    table["params"] = params
    table["segment"] = segment
    table["ordinal"] = np.arange(len(t_ns))
    table["size"] = sizes
    return table


def camera_stats(table: np.ndarray, **extra: Any) -> dict[str, Any]:
    return {"messages": int(len(table)), "duplicates_dropped": 0, "non_monotonic": int(np.sum(np.diff(table["t_ns"]) <= 0)),
            "wrong_format": 0, "keyframes": int(table["key"].sum()), "bytes": int(table["size"].sum()),
            "frames": int(len(table)),
            "first_t_ns": int(table["t_ns"][0]) if len(table) else None,
            "last_t_ns": int(table["t_ns"][-1]) if len(table) else None, **extra}


def write_segment(stream_dir: Path, number: int, name: str, tables: Mapping[str, np.ndarray], imu_t: np.ndarray,
                  imu: np.ndarray, *, calibration: Mapping[str, Any] | None = None, missing: Sequence[str] = (),
                  extra_arrays: Mapping[str, np.ndarray] | None = None, info: Mapping[str, Any] | None = None,
                  camera_extra: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Write one segment's index (atomically, like the MCAP reader) and return its summary."""
    final = segment_dir(stream_dir, number)
    tmp = stream_dir / f".tmp_seg_{number:05d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    imu = np.asarray(imu, dtype="<f4").reshape(-1, len(IMU_COLUMNS))
    imu_t = np.asarray(imu_t, dtype="<i8")
    summary: dict[str, Any] = {"number": number, "file": name, "calibration": dict(calibration or {}),
                               "calibration_changes": {}, "missing_topics": list(missing),
                               "imu": {"messages": int(len(imu_t)), "samples": int(len(imu_t)), "duplicates_dropped": 0,
                                       "non_monotonic": int(np.sum(np.diff(imu_t) <= 0)),
                                       "first_t_ns": int(imu_t[0]) if len(imu_t) else None,
                                       "last_t_ns": int(imu_t[-1]) if len(imu_t) else None},
                               **dict(info or {})}
    for cam in CAMERA_KEYS:
        table = tables.get(cam, empty_table())
        np.save(tmp / f"{cam}.npy", table)
        summary[cam] = camera_stats(table, **dict((camera_extra or {}).get(cam, {})))
    np.save(tmp / "imu_t.npy", imu_t)
    np.save(tmp / "imu.npy", imu)
    for key, array in (extra_arrays or {}).items():
        np.save(tmp / f"{key}.npy", array)
    write_json_atomic(tmp / "segment.json", summary)
    if final.exists():
        shutil.rmtree(final)
    tmp.replace(final)
    return summary


def load_summary(stream_dir: Path, number: int) -> dict[str, Any] | None:
    path = segment_dir(stream_dir, number) / "segment.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def imu_rows(gyro_rad_s: np.ndarray, accel_m_s2: np.ndarray | None = None) -> np.ndarray:
    """IMU_COLUMNS rows: gyro xyz (rad/s), accel xyz (m/s^2), magnetometer and temperature unknown (NaN)."""
    n = len(gyro_rad_s)
    rows = np.full((n, len(IMU_COLUMNS)), np.nan, dtype=np.float32)
    rows[:, 0:3] = gyro_rad_s
    if accel_m_s2 is not None:
        rows[:, 3:6] = accel_m_s2
    return rows


def interpolate_onto(t_target: np.ndarray, t_source: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(t_source) == 0:
        return np.full((len(t_target), values.shape[1] if values.ndim == 2 else 1), np.nan)
    return np.stack([np.interp(t_target, t_source, values[:, i]) for i in range(values.shape[1])], axis=1)


class RecordingReader:
    """Base class: one instance per session run. Subclasses set the class attributes and implement
    ``required_files``, ``extract`` and ``_segment_packets``."""

    format = "?"
    layout = "pair"                   # 'pair' or 'sbs'
    codec = "hevc"                    # FFmpeg input format of the packets: 'hevc' (Annex-B) or 'mjpeg'
    imu_unavailable: str | None = None
    # host_clock: frame times are host receive times (they jitter), so the frame rate is the mean over
    # the recording. frame_counter: each segment also has seq.npy, the camera's capture counter per row,
    # and gaps / lost frames are taken from it instead of the jittery times (qc.technical.frame_timing).
    host_clock = False
    frame_counter = False
    harmless_decode_errors: tuple[str, ...] = ()
    description = ""

    def __init__(self, listing, storage, cfg, *, work_dir: Path, keep_dir: Path, max_segments: int | None = None) -> None:
        """``work_dir``: temporary files (deleted after upload). ``keep_dir``: small metadata files of the
        recording (session.json, calibration.json, ...) that are copied into the results."""
        self.listing = listing
        self.storage = storage
        self.cfg = cfg
        self.bucket = cfg.get("source.bucket")
        self.chunk_bytes = int(cfg.get("source.read_chunk_mb") * (1 << 20))
        self.work_dir = work_dir
        self.keep_dir = keep_dir
        self.max_segments = max_segments

    # --- small metadata files kept with the results (downloaded first, even if the session FAILs) --
    metadata_files: tuple[str, ...] = ()

    def fetch_metadata(self) -> list[str]:
        fetched = []
        for name in self.metadata_files:
            obj = self.listing.files.get(name)
            if obj is not None:
                self.storage.download(self.bucket, obj, self.keep_dir / name)
                fetched.append(name)
        return fetched

    def kept_json(self, name: str) -> dict[str, Any] | None:
        path = self.keep_dir / name
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    # --- what the session must contain (section 1, before anything is read) ----------------------
    def required_files(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    # --- index every segment into stream_dir (resumable); returns the segment numbers in order ----
    def extract(self, stream_dir: Path) -> list[int]:
        raise NotImplementedError

    # --- packets ---------------------------------------------------------------------------------
    def _segment_packets(self, stream_dir: Path, segment: int, rows: Mapping[str, Sequence[tuple[int, int]]]
                         ) -> Iterator[tuple[str, int, bytes]]:
        """Yield (cam, row, raw packet) for the requested (row, ordinal) pairs of one segment."""
        raise NotImplementedError

    def stream_packets_from(self, stream_dir: Path) -> Callable[[PacketRequests, PacketSink], dict[str, int]]:
        def stream(requests: PacketRequests, sink: PacketSink) -> dict[str, int]:
            return self._stream(stream_dir, requests, sink)
        return stream

    def _stream(self, stream_dir: Path, requests: PacketRequests, sink: PacketSink) -> dict[str, int]:
        wanted: dict[int, dict[str, list[tuple[int, int]]]] = {}
        for cam, (table, start, stop) in requests.items():
            rows = table[start:stop]
            for row, (segment, ordinal) in enumerate(zip(rows["segment"].tolist(), rows["ordinal"].tolist()), start=start):
                wanted.setdefault(segment, {}).setdefault(cam, []).append((row, ordinal))
        delivered = {cam: 0 for cam in requests}
        expected = {cam: start for cam, (_t, start, _s) in requests.items()}
        for segment in sorted(wanted):
            try:
                for cam, row, data in self._segment_packets(stream_dir, segment, wanted[segment]):
                    table = requests[cam][0]
                    if row != expected[cam]:
                        raise IngestionError(f"segment {segment} {cam}: packet for row {row}, expected row {expected[cam]}")
                    sink(cam, row, data)
                    expected[cam] = row + 1
                    delivered[cam] += 1
            except IngestionError:
                raise
            except Exception as exc:  # noqa: BLE001 (network, corrupt data)
                raise IngestionError(f"Reading segment {segment} failed: {type(exc).__name__}: {exc}") from exc
        for cam, (_table, start, stop) in requests.items():
            if delivered[cam] != stop - start:
                raise IngestionError(f"{cam}: streamed {delivered[cam]} packets, expected {stop - start}")
        return delivered

    # --- side-by-side layout ------------------------------------------------------------------------
    def eye_crops(self, stream_dir: Path) -> dict[str, list[int]] | None:
        """x range [x0, x1) of each camera inside the picture (sbs layout only)."""
        return None

    def imu_nominal_hz(self, calibration_messages: Mapping[str, Any]) -> float | None:
        return None

    def fingerprint_extra(self) -> Any:
        """Format-specific settings the extract step depends on (None keeps MCAP fingerprints unchanged)."""
        return None

    # --- helpers -----------------------------------------------------------------------------------
    def open(self, obj):
        return self.storage.open_reader(self.bucket, obj, chunk_bytes=self.chunk_bytes)

    def segments(self):
        segs = list(self.listing.segments)
        return segs[:self.max_segments] if self.max_segments is not None else segs
