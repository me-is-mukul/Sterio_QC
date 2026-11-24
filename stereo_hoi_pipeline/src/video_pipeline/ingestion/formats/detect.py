"""Recognise the recording format of a folder from the names of the files directly inside it.

Formats (a folder is a session when its own files match one of them; nested folders are separate
sessions, e.g. robocap folders that hold one session folder each):

  mcap         ego_NNNN.mcap (+ session.json)                           both eyes + IMU in MCAP
  robocap      robocap_segmentN_<camera|imu>.(mp4|db)                   six MP4 cameras, SQLite IMU
  hydra        video.h265 + frames.idx + h265_frame_meta.csv + h265_imu.csv   side-by-side HEVC
  egocam_live  video_live_NNNN.mp4 + frame_map.csv + calibration.json   side-by-side HEVC MP4s
  egocam_egoc  capture.egoc + calibration.json                          side-by-side MJPEG container

Anything else (macOS index folders, empty folders) is not a session.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from video_pipeline.core.errors import IngestionError
from video_pipeline.ingestion.gcs import ObjectInfo, Segment, SessionListing, SessionRef

FORMATS = ("mcap", "robocap", "hydra", "egocam_live", "egocam_egoc")
DEFAULT_MCAP_PATTERN = r"^ego_(\d+)\.mcap$"
_ROBOCAP = re.compile(r"^robocap_segment(\d+)_(.+)$")
_LIVE = re.compile(r"^video_live_(\d+)\.mp4$")
_HYDRA_MARKERS = {"video.h265", "frames.idx", "h265_frame_meta.csv", "stereo_calibration.bin"}
_HYDRA_FILES = {"frames.idx", "h265_frame_meta.csv", "h265_imu.csv", "h265_mag.csv", "meta.json", "session.json",
                "stereo_calibration.bin"}
_EGOCAM_FILES = {"calibration.json", "metadata.json", "frame_map.csv", "encode_gaps.json", "card_verify.json",
                 "diagnostics.json", "diag_series.json", "take_activity.csv"}


def detect_format(names: Iterable[str], mcap_pattern: str = DEFAULT_MCAP_PATTERN) -> str | None:
    """Format of a folder from the names of its direct files (None: not a recording)."""
    names = set(names)
    mcap = re.compile(mcap_pattern)
    if any(mcap.fullmatch(n) for n in names):
        return "mcap"
    if any(_ROBOCAP.fullmatch(n) for n in names):
        return "robocap"
    if names & _HYDRA_MARKERS:
        return "hydra"
    if any(_LIVE.fullmatch(n) for n in names) or "frame_map.csv" in names:
        return "egocam_live"
    if "capture.egoc" in names or "calibration.json" in names:
        return "egocam_egoc"            # also EGO-CAM folders whose video is missing (they then FAIL)
    return None


def classify(session: SessionRef, objects: Iterable[ObjectInfo], mcap_pattern: str = DEFAULT_MCAP_PATTERN
             ) -> SessionListing:
    """Sort a session folder's objects into segments, session-level files and others."""
    direct: dict[str, ObjectInfo] = {}
    other: list[ObjectInfo] = []
    for obj in objects:
        rest = obj.name[len(session.prefix):]
        if "/" in rest or not rest:                     # nested folders are separate sessions
            other.append(obj)
        else:
            direct[rest] = obj
    fmt = detect_format(direct, mcap_pattern)
    segments: dict[int, dict[str, ObjectInfo]] = {}
    files: dict[str, ObjectInfo] = {}

    def keep(name: str, obj: ObjectInfo, wanted: bool) -> None:
        (files.__setitem__(name, obj) if wanted else other.append(obj))

    if fmt == "mcap":
        pattern = re.compile(mcap_pattern)
        for name, obj in direct.items():
            m = pattern.fullmatch(name)
            if m:
                segments.setdefault(int(m.group(1)), {})["mcap"] = obj
            else:
                keep(name, obj, name == "session.json")
    elif fmt == "robocap":
        for name, obj in direct.items():
            m = _ROBOCAP.fullmatch(name)
            if m and not name.endswith((".part", ".tmp")):
                segments.setdefault(int(m.group(1)), {})[m.group(2)] = obj
            else:
                other.append(obj)
    elif fmt == "hydra":
        for name, obj in direct.items():
            if name == "video.h265":
                segments.setdefault(0, {})["video"] = obj
            else:
                keep(name, obj, name in _HYDRA_FILES)
    elif fmt == "egocam_live":
        for name, obj in direct.items():
            m = _LIVE.fullmatch(name)
            if m:
                segments.setdefault(int(m.group(1)), {})["video"] = obj
            else:
                keep(name, obj, name in _EGOCAM_FILES)
    elif fmt == "egocam_egoc":
        for name, obj in direct.items():
            if name == "capture.egoc":
                segments.setdefault(0, {})["video"] = obj
            else:
                keep(name, obj, name in _EGOCAM_FILES)
    else:
        other.extend(direct.values())
    ordered = tuple(Segment(n, segments[n]) for n in sorted(segments))
    numbers = [s.number for s in ordered]
    if len(numbers) != len(set(numbers)):
        raise IngestionError(f"{session.url}: duplicate segment numbers {numbers}")
    return SessionListing(session=session, format=fmt, files=files, segments=ordered, other=tuple(other))
