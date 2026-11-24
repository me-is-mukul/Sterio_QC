"""Section 1 of the QA sheet: technical checks on a whole recording.

* ``decode_recording``  one streaming read of the recording feeds one FFmpeg process per video
  stream (stdin): a separate process per eye for two-stream formats, one process for a
  side-by-side stream that is cut into its two eyes. Each process does the full decode
  (integrity), blackdetect + freezedetect per eye (results in detect_<cam>.txt), and in the same
  pass writes the sampled frames for labelling (every `step` frames) and the full-resolution
  frames for the calibration check. The video itself is never written to disk.
* ``evaluate_technical``  pure function: every check of section 1 with its numbers.

The raw streams get the container clock n / fps (`-framerate`), so FFmpeg frame n is table row n.
"""

from __future__ import annotations

import contextvars
import logging
import math
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from video_pipeline.core.errors import ToolExecutionError
from video_pipeline.core.pipe_feeder import PipeFeeder

_LEVEL = re.compile(r"\[(error|fatal|panic)\]")

logger = logging.getLogger(__name__)

# (requests, sink) -> packets per camera; see ingestion/formats/base.py
PacketStreamer = Callable[[Mapping[str, tuple[np.ndarray, int, int]], Callable[[str, int, bytes], None]], dict]


@dataclass(frozen=True)
class DecodeSettings:
    fps: int
    sample_step: int              # frames between labelling samples
    sample_max_width: int
    jpeg_quality: int
    black_min_seconds: float
    black_pixel_threshold: float
    black_picture_threshold: float
    freeze_noise_db: float
    freeze_min_seconds: float
    timeout_seconds: float


@dataclass(frozen=True)
class EyeOutput:
    cam: str
    crop: Sequence[int] | None    # [x0, x1) inside the decoded picture (side-by-side), None = whole picture
    sample_dir: Path
    calib_dir: Path


def jpeg_qscale(quality: int) -> int:
    """1-100 JPEG quality -> FFmpeg mjpeg -q:v (2 best .. 31 worst)."""
    return max(2, min(31, round(2 + (100 - quality) * 29 / 99)))


def detection_file(cam: str) -> str:
    """Per-eye black/freeze results, written by FFmpeg's metadata filter (relative to the log dir)."""
    return f"detect_{cam}.txt"


def build_decode_command(ffmpeg: str, settings: DecodeSettings, *, eyes: Sequence[EyeOutput], calib_indices: Sequence[int],
                         progress_path: Path, codec: str = "hevc") -> list[str]:
    """FFmpeg command for one input stream. Run it with the log dir as working directory (the
    detection files are written there by name, which avoids escaping paths inside the filter graph)."""
    s = settings
    parts = []
    if len(eyes) > 1:
        parts.append("[0:v]split=" + str(len(eyes)) + "".join(f"[in{i}]" for i in range(len(eyes))))
    branches = 3 if calib_indices else 2
    for i, eye in enumerate(eyes):
        source = f"[in{i}]" if len(eyes) > 1 else "[0:v]"
        crop = f"crop=w={eye.crop[1] - eye.crop[0]}:h=ih:x={eye.crop[0]}:y=0," if eye.crop else ""
        parts.append(f"{source}{crop}split={branches}[d{i}][s{i}]{f'[c{i}]' if calib_indices else ''}")
        parts.append(f"[d{i}]blackdetect=d={s.black_min_seconds}:pix_th={s.black_pixel_threshold}:"
                     f"pic_th={s.black_picture_threshold},freezedetect=n={s.freeze_noise_db}dB:d={s.freeze_min_seconds},"
                     f"metadata=mode=print:file={detection_file(eye.cam)}[dout{i}]")
        parts.append(f"[s{i}]select='not(mod(n,{s.sample_step}))',scale=w='min({s.sample_max_width},iw)':h=-2[sout{i}]")
        if calib_indices:
            expr = "+".join(f"eq(n,{k})" for k in calib_indices)
            parts.append(f"[c{i}]select='{expr}'[cout{i}]")
    cmd = [ffmpeg, "-hide_banner", "-nostats", "-loglevel", "level+info", "-y",
           "-f", codec, "-framerate", str(s.fps), "-i", "pipe:0", "-filter_complex", ";".join(parts)]
    for i, eye in enumerate(eyes):
        cmd += ["-map", f"[dout{i}]", "-f", "null", "-",
                "-map", f"[sout{i}]", "-fps_mode", "passthrough", "-q:v", str(jpeg_qscale(s.jpeg_quality)),
                "-start_number", "0", str(eye.sample_dir / "%06d.jpg")]
        if calib_indices:
            cmd += ["-map", f"[cout{i}]", "-fps_mode", "passthrough", "-start_number", "0", str(eye.calib_dir / "%03d.png")]
    cmd += ["-progress", str(progress_path)]
    return cmd


def parse_detection_file(path: Path, total_seconds: float, black_min_seconds: float,
                         freeze_min_seconds: float) -> dict[str, list[tuple[float, float]]]:
    """Black and frozen intervals from a metadata=print file. A stretch still open at the end of the
    stream lasts until the end. Stretches shorter than the minimum are dropped (blackdetect marks
    every black start in the metadata, its d= only filters the log)."""
    black: list[tuple[float, float]] = []
    frozen: list[tuple[float, float]] = []
    black_start = freeze_start = None
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    for line in text.splitlines():
        key, _, value = line.strip().partition("=")
        try:
            number = float(value)
        except ValueError:
            continue
        if key == "lavfi.black_start":
            black_start = number
        elif key == "lavfi.black_end" and black_start is not None:
            black.append((black_start, number))
            black_start = None
        elif key == "lavfi.freezedetect.freeze_start":
            freeze_start = number
        elif key == "lavfi.freezedetect.freeze_end" and freeze_start is not None:
            frozen.append((freeze_start, number))
            freeze_start = None
    if black_start is not None:
        black.append((black_start, total_seconds))
    if freeze_start is not None:
        frozen.append((freeze_start, total_seconds))
    return {"black": [(a, b) for a, b in black if b - a >= black_min_seconds - 1e-6],
            "frozen": [(a, b) for a, b in frozen if b - a >= freeze_min_seconds - 1e-6]}


def decode_errors(stderr: str, harmless: Sequence[str] = ()) -> list[str]:
    """Error-level lines of the FFmpeg log, without messages known to be harmless for this format."""
    return [line.strip() for line in stderr.splitlines()
            if _LEVEL.search(line) and not any(h in line for h in harmless)]


def _last_progress_frame(path: Path) -> int | None:
    if not path.is_file():
        return None
    frames = re.findall(r"^frame=(\d+)", path.read_text(encoding="utf-8", errors="replace"), flags=re.MULTILINE)
    return int(frames[-1]) if frames else None


def _tail_progress_frame(path: Path, tail_bytes: int = 4096) -> int:
    """Latest frame= of an FFmpeg -progress file, reading only its end (the file grows all run)."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - tail_bytes))
            text = handle.read().decode("utf-8", "replace")
    except OSError:
        return 0
    frames = re.findall(r"^frame=(\d+)", text, flags=re.MULTILINE)
    return int(frames[-1]) if frames else 0


def _minutes(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    return f"{h} h {rem // 60:02d} min" if h else f"{rem // 60} min {rem % 60:02d} s"


def format_decode_progress(done: Mapping[str, int], total: int, elapsed: float, fps: float) -> str:
    """One progress line: frames per eye, speed vs real time, time left (the slower eye decides)."""
    slowest = min(done.values()) if done else 0
    per_eye = ", ".join(f"{cam} {n}/{total}" for cam, n in done.items())
    pct = 100.0 * slowest / total if total else 100.0
    if slowest <= 0 or elapsed <= 0:
        return f"Decoding: {per_eye} ({pct:.0f}%), starting ..."
    rate = slowest / elapsed                                 # frames per second, both eyes in parallel
    left = (total - slowest) / rate
    return (f"Decoding: {per_eye} ({pct:.0f}%) | speed {rate / fps:.2f}x real time | "
            f"elapsed {_minutes(elapsed)}, about {_minutes(left)} left")


class _ProgressReporter:
    """Logs decode progress every `interval` seconds from FFmpeg's -progress files, on its own thread."""

    def __init__(self, progress_files: Mapping[str, Path], total: int, fps: float, interval: float) -> None:
        self._files, self._total, self._fps, self._interval = dict(progress_files), total, fps, interval
        self._stop = threading.Event()
        self._started = time.perf_counter()
        context = contextvars.copy_context()                 # keep the step name on the log lines
        self._thread = threading.Thread(target=context.run, args=(self._run,), name="decode-progress", daemon=True)

    def __enter__(self) -> _ProgressReporter:
        if self._interval > 0:
            self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            done = {cam: _tail_progress_frame(path) for cam, path in self._files.items()}
            logger.info(format_decode_progress(done, self._total, time.perf_counter() - self._started, self._fps))


def decode_recording(ffmpeg: str, stream: PacketStreamer, tables: Mapping[str, np.ndarray], settings: DecodeSettings, *,
                     sample_dirs: Mapping[str, Path], calib_dirs: Mapping[str, Path], calib_indices: Sequence[int],
                     log_dir: Path, layout: str = "pair", codec: str = "hevc",
                     crops: Mapping[str, Sequence[int]] | None = None, harmless_errors: Sequence[str] = (),
                     progress_interval_seconds: float = 60.0) -> dict[str, dict[str, Any]]:
    """Decode the recording in ONE streaming read: the packets go straight from the source into FFmpeg
    (bounded in-memory queues; nothing video-sized on disk). ``layout`` 'pair': one process per eye;
    'sbs': one process for the shared stream (cam0's rows), cut into the eyes by ``crops``.
    Logs a progress line every `progress_interval_seconds` (0 = off)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    if layout == "sbs":
        if not crops:
            raise ToolExecutionError("side-by-side decode needs the eye crops")
        groups = {"sbs": [EyeOutput(cam, crops[cam], sample_dirs[cam], calib_dirs[cam]) for cam in tables]}
        requests = {"cam0": (tables["cam0"], 0, len(tables["cam0"]))}
        route = {"cam0": "sbs"}
    else:
        groups = {cam: [EyeOutput(cam, None, sample_dirs[cam], calib_dirs[cam])] for cam in tables}
        requests = {cam: (t, 0, len(t)) for cam, t in tables.items()}
        route = {cam: cam for cam in tables}
    total = max((len(t) for t in tables.values()), default=0)
    with _ProgressReporter({g: log_dir / f"decode_{g}.progress" for g in groups}, total, settings.fps,
                           progress_interval_seconds):
        return _decode(ffmpeg, stream, tables, settings, groups=groups, requests=requests, route=route,
                       calib_indices=calib_indices, log_dir=log_dir, codec=codec, harmless_errors=harmless_errors)


def _decode(ffmpeg: str, stream: PacketStreamer, tables: Mapping[str, np.ndarray], settings: DecodeSettings, *,
            groups: Mapping[str, Sequence[EyeOutput]], requests, route: Mapping[str, str], calib_indices: Sequence[int],
            log_dir: Path, codec: str, harmless_errors: Sequence[str]) -> dict[str, dict[str, Any]]:
    procs: dict[str, subprocess.Popen] = {}
    feeders: dict[str, PipeFeeder] = {}
    commands: dict[str, list[str]] = {}
    handles = {}
    started = time.perf_counter()
    try:
        for group, eyes in groups.items():
            for eye in eyes:
                for folder in (eye.sample_dir, eye.calib_dir):
                    if folder.exists():
                        shutil.rmtree(folder)
                    folder.mkdir(parents=True)
                (log_dir / detection_file(eye.cam)).unlink(missing_ok=True)
            progress = log_dir / f"decode_{group}.progress"
            progress.unlink(missing_ok=True)
            commands[group] = build_decode_command(ffmpeg, settings, eyes=eyes, calib_indices=calib_indices,
                                                   progress_path=progress.resolve(), codec=codec)
            handles[group] = (log_dir / f"decode_{group}.stderr.log").open("w", encoding="utf-8", errors="replace")
            try:
                procs[group] = subprocess.Popen(commands[group], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                                stderr=handles[group], cwd=str(log_dir))
            except OSError as exc:
                raise ToolExecutionError(f"Could not start FFmpeg: {exc}") from exc
            feeders[group] = PipeFeeder(procs[group])
        stream(requests, lambda cam, _row, data: feeders[route[cam]].write(data))
    except BaseException:
        for proc in procs.values():
            proc.kill()
        for feeder in feeders.values():
            feeder.close()
        for proc in procs.values():
            proc.wait()
        for handle in handles.values():
            handle.close()
        raise
    for feeder in feeders.values():
        feeder.close()
    results = {}
    try:
        for group, proc in procs.items():
            remaining = max(1.0, settings.timeout_seconds - (time.perf_counter() - started))
            try:
                returncode = proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                for p in procs.values():
                    p.kill()
                raise ToolExecutionError(f"FFmpeg decode of {group} timed out after {settings.timeout_seconds}s") from exc
            results[group] = returncode
    finally:
        for handle in handles.values():
            handle.close()
    out = {}
    for group, returncode in results.items():
        stderr = (log_dir / f"decode_{group}.stderr.log").read_text(encoding="utf-8", errors="replace")
        errors = decode_errors(stderr, harmless_errors)
        decoded = _last_progress_frame(log_dir / f"decode_{group}.progress")
        for eye in groups[group]:
            n = len(tables[eye.cam])
            detections = parse_detection_file(log_dir / detection_file(eye.cam), n / settings.fps,
                                              settings.black_min_seconds, settings.freeze_min_seconds)
            out[eye.cam] = {
                "cam": eye.cam,
                "process": group,
                "crop_x": list(eye.crop) if eye.crop else None,
                "command": commands[group],
                "returncode": returncode,
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "decoded_frames": decoded,
                "expected_frames": int(n),
                "decode_errors": errors[:50],
                "decode_error_count": len(errors),
                "ignored_harmless_errors": list(harmless_errors),
                "black_intervals": detections["black"],
                "frozen_intervals": detections["frozen"],
                "samples_written": len(list(eye.sample_dir.glob("*.jpg"))),
                "calibration_frames_written": len(list(eye.calib_dir.glob("*.png"))),
                "stderr_tail": stderr.strip().splitlines()[-8:] if returncode != 0 else [],
            }
    return out


# ---------------------------------------------------------------------------------------------
# Checks (pure)
# ---------------------------------------------------------------------------------------------

def merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted((min(a, b), max(a, b)) for a, b in intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _check(name: str, passed: bool, message: str, **details: Any) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "message": message, **details}


def check_required_files(listing: Mapping[str, Any]) -> list[dict[str, Any]]:
    numbers = [s["number"] for s in listing["segments"]]
    checks = [
        _check("session_json_present", listing["session_json"] is not None,
               "session.json present" if listing["session_json"] else "session.json missing"),
        _check("first_segment_present", 0 in numbers,
               "first video segment (ego_0000) present" if 0 in numbers else "first video segment ego_0000 missing"),
    ]
    missing = sorted(set(range(max(numbers) + 1)) - set(numbers)) if numbers else []
    checks.append(_check("segments_contiguous", bool(numbers) and not missing,
                         "segment numbers are contiguous" if numbers and not missing else
                         ("no segments" if not numbers else f"missing segment numbers {missing[:20]}"),
                         segments=len(numbers), missing_numbers=missing))
    return checks


def check_segment_topics(segment_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bad = {s["number"]: s["missing_topics"] for s in segment_summaries if s["missing_topics"]}
    wrong_format = {s["number"]: (s["cam0"]["wrong_format"], s["cam1"]["wrong_format"]) for s in segment_summaries
                    if s["cam0"]["wrong_format"] or s["cam1"]["wrong_format"]}
    ok = not bad and not wrong_format
    msg = "every segment has all its parts (both cameras, IMU where the format has one)" if ok else \
        f"segments with missing or unreadable parts: {bad}" + \
        (f"; frames in an unexpected codec: {wrong_format}" if wrong_format else "")
    return _check("core_topics_present", ok, msg, segments_missing_topics=bad, wrong_format_frames=wrong_format)


def frame_timing(table: np.ndarray, expected_fps: float, sequence: np.ndarray | None = None,
                 mean_rate: bool = False) -> dict[str, Any]:
    """Frame rate, gaps and drops of one camera.

    Times from the camera/sensor clock (default): rate = 1 / median interval, gaps = time differences.
    Host receive times (EGO-CAM) jitter by tens of ms although no frame is lost, so the rate is the mean
    over the recording (``mean_rate``); with ``sequence`` (the camera's capture counter) gaps and lost
    frames come from the counter (a jump of k = k - 1 lost frames, k / fps seconds), otherwise from the
    receive times."""
    t = table["t_ns"].astype(np.int64)
    if len(t) < 2:
        return {"frames": int(len(t)), "fps": None, "max_gap_s": None, "non_monotonic": 0, "dropped_frames_est": 0,
                "duration_s": round(len(t) / expected_fps, 3)}
    dt = np.diff(t)
    median = float(np.median(dt))
    nominal = 1e9 / expected_fps
    if sequence is not None and len(sequence) == len(t):
        steps = np.diff(np.asarray(sequence, dtype=np.int64))
        span = float(t[-1] - t[0])
        jumps = steps[steps > 1]
        return {
            "frames": int(len(t)),
            "clock": "host receive time: rate = mean over the recording, gaps from the camera frame counter",
            "fps": round((len(t) - 1) * 1e9 / span, 4) if span > 0 else None,
            "median_interval_ms": round(median / 1e6, 4),
            "host_max_interval_s": round(float(dt.max()) / 1e9, 4),
            "max_gap_s": round(float(steps.max()) / expected_fps, 4),
            "non_monotonic": int(np.sum(dt <= 0) + np.sum(steps <= 0)),
            "gaps_over_1_5_frames": int(len(jumps)),
            "dropped_frames_est": int(np.sum(jumps - 1)),
            "duration_s": round(span / 1e9 + 1 / expected_fps, 3),
        }
    gaps = dt[dt > 1.5 * nominal]
    span = float(t[-1] - t[0])
    if mean_rate:
        rate = (len(t) - 1) * 1e9 / span if span > 0 else None
        lost = max(0, int(round(span / nominal)) - (len(t) - 1))
        return {
            "frames": int(len(t)),
            "clock": "host receive time: rate = mean over the recording, gaps from the receive times",
            "fps": round(rate, 4) if rate else None,
            "median_interval_ms": round(median / 1e6, 4),
            "min_interval_ms": round(float(dt.min()) / 1e6, 4),
            "max_gap_s": round(float(dt.max()) / 1e9, 4),
            "non_monotonic": int(np.sum(dt <= 0)),
            "gaps_over_1_5_frames": int(len(gaps)),
            "dropped_frames_est": lost,
            "duration_s": round(span / 1e9 + 1 / expected_fps, 3),
        }
    return {
        "frames": int(len(t)),
        "fps": round(1e9 / median, 4) if median > 0 else None,
        "median_interval_ms": round(median / 1e6, 4),
        "min_interval_ms": round(float(dt.min()) / 1e6, 4),
        "max_gap_s": round(float(dt.max()) / 1e9, 4),
        "non_monotonic": int(np.sum(dt <= 0)),
        "gaps_over_1_5_frames": int(len(gaps)),
        "dropped_frames_est": int(sum(round(g / nominal) - 1 for g in gaps)),
        "duration_s": round(float(t[-1] - t[0]) / 1e9 + 1 / expected_fps, 3),
    }


def pair_eyes(cam0: np.ndarray, cam1: np.ndarray, *, tolerance_ns: int, max_edge_frames: int) -> dict[str, Any]:
    """Match the two eyes' frames by capture time.

    The eyes run at the same rate, so frame i of cam0 pairs with frame i + shift of cam1. The shift is
    the one (|shift| <= max_edge_frames) that matches the most frames within `tolerance_ns`. Inside the
    overlap every pair must match; frames outside it (an eye that started earlier or stopped later) are
    trimmed if there are at most `max_edge_frames` of them at each end. The paired range then starts at
    the first frame that is a keyframe in both eyes, so a stream fed from there decodes.

    Returns ok, the row ranges [start, stop) of each eye, and diagnostics.
    """
    n0, n1 = len(cam0), len(cam1)
    result: dict[str, Any] = {"ok": False, "cam0_frames": n0, "cam1_frames": n1}
    if n0 == 0 or n1 == 0:
        result["message"] = f"cam0 {n0} / cam1 {n1} frames"
        return result
    t0, t1 = cam0["t_ns"].astype(np.int64), cam1["t_ns"].astype(np.int64)
    best = None
    for shift in range(-max_edge_frames, max_edge_frames + 1):
        a0, a1 = max(0, -shift), max(0, shift)
        length = min(n0 - a0, n1 - a1)
        if length <= 0:
            continue
        diff = np.abs(t0[a0:a0 + length] - t1[a1:a1 + length])
        matched = int(np.sum(diff <= tolerance_ns))
        if best is None or matched > best[0]:
            best = (matched, shift, a0, a1, length, diff)
    matched, shift, a0, a1, length, diff = best
    bad = np.flatnonzero(diff > tolerance_ns)
    edges = {"cam0_start": a0, "cam1_start": a1, "cam0_end": n0 - (a0 + length), "cam1_end": n1 - (a1 + length)}
    result.update(shift=shift, overlap=length, unmatched_inside=int(len(bad)), unpaired_edge_frames=edges,
                  max_offset_ms=round(float(diff.max()) / 1e6, 4) if length else None)
    if len(bad):
        result["message"] = (f"{len(bad)} frame pairs differ by more than {tolerance_ns / 1e6:g} ms "
                             f"(first at pair {int(bad[0])}); max {result['max_offset_ms']} ms")
        return result
    if max(edges.values()) > max_edge_frames:
        result["message"] = f"too many unpaired frames at the ends: {edges}"
        return result
    keys = np.flatnonzero(cam0["key"][a0:a0 + length] & cam1["key"][a1:a1 + length])
    if not len(keys):
        result["message"] = "no frame is a keyframe in both eyes"
        return result
    skip = int(keys[0])
    result.update(ok=True, start_skipped_to_keyframe=skip,
                  cam0_rows=[a0 + skip, a0 + length], cam1_rows=[a1 + skip, a1 + length],
                  paired_frames=length - skip,
                  message=(f"{length - skip} frame pairs, max left/right offset {result['max_offset_ms']} ms; "
                           f"trimmed unpaired end frames {edges}, {skip} frames before the first common keyframe"))
    return result


def common_keyframes(cam0: np.ndarray, cam1: np.ndarray) -> np.ndarray:
    n = min(len(cam0), len(cam1))
    return np.flatnonzero(cam0["key"][:n] & cam1["key"][:n])


def evaluate_technical(*, listing: Mapping[str, Any], segment_summaries: Sequence[Mapping[str, Any]],
                       cam0: np.ndarray | None, cam1: np.ndarray | None, imu_t: np.ndarray | None,
                       imu_nominal_hz: float | None, decode: Mapping[str, Mapping[str, Any]] | None,
                       cfg: Mapping[str, Any], pairing: Mapping[str, Any] | None = None,
                       paired_cam0: np.ndarray | None = None,
                       required_checks: Sequence[Mapping[str, Any]] | None = None,
                       imu_unavailable: str | None = None,
                       frame_sequence: np.ndarray | None = None, host_times: bool = False) -> dict[str, Any]:
    """All section-1 checks. ``cfg`` is the 'technical' config section. ``cam0``/``cam1`` are the full
    frame tables (timing is checked on them); ``pairing`` is pair_eyes(); ``paired_cam0`` the cam0 rows
    of the paired range (length and IMU coverage refer to it). ``required_checks``: the format's file
    checks (default: the MCAP ones). ``imu_unavailable``: why the format has no readable IMU (its IMU
    checks are then reported as not checked). ``frame_sequence``: the camera's frame counter per row
    when the times are host receive times (see frame_timing). Missing inputs = not evaluated."""
    checks = [dict(c) for c in required_checks] if required_checks is not None else check_required_files(listing)
    if segment_summaries:
        checks.append(check_segment_topics(segment_summaries))
    fps = float(cfg["expected_fps"])
    info: dict[str, Any] = {}

    if cam0 is not None and cam1 is not None:
        timing = {cam: frame_timing(table, fps, frame_sequence, mean_rate=host_times)
                  for cam, table in (("cam0", cam0), ("cam1", cam1))}
        info["frame_timing"] = timing
        for cam, tm in timing.items():
            ok = (tm["frames"] >= 2 and tm["fps"] is not None and abs(tm["fps"] - fps) <= cfg["fps_tolerance"]
                  and tm["non_monotonic"] == 0 and tm["max_gap_s"] <= cfg["max_frame_gap_seconds"])
            checks.append(_check(f"frame_timing_{cam}", ok,
                                 f"{cam}: {tm['frames']} frames, {tm['fps']} fps, max gap {tm['max_gap_s']} s, "
                                 f"{tm['non_monotonic']} non-increasing timestamps", **tm))
        if pairing is not None:
            checks.append(_check("eyes_paired", pairing["ok"], pairing.get("message", ""),
                                 **{k: v for k, v in pairing.items() if k not in {"ok", "message"}}))
        params_ok = bool(np.all(cam0["params"][cam0["key"]])) and bool(np.all(cam1["params"][cam1["key"]]))
        checks.append(_check("keyframes_self_contained", params_ok,
                             "every keyframe carries VPS/SPS/PPS" if params_ok else
                             "some keyframes lack VPS/SPS/PPS (clips get the parameter sets prepended)",
                             informational=True))
        base = paired_cam0 if paired_cam0 is not None else cam0
        duration = frame_timing(base, fps)["duration_s"] if len(base) else 0.0
        info["duration_s"] = duration
        checks.append(_check("length", duration >= cfg["min_recording_seconds"],
                             f"recording lasts {duration} s (need >= {cfg['min_recording_seconds']} s for a 30 s piece)",
                             duration_s=duration))

        if imu_unavailable:
            checks.append(_check("imu_not_checked", True, imu_unavailable, informational=True))
        elif imu_t is not None:
            checks.extend(_imu_checks(imu_t, base, imu_nominal_hz, cfg))

    if decode is not None:
        total = (len(cam0) / fps) if cam0 is not None else 0.0
        defects_all: list[tuple[float, float]] = []
        for cam, result in decode.items():
            ok = (result["returncode"] == 0 and result["decode_error_count"] == 0
                  and result["decoded_frames"] == result["expected_frames"])
            checks.append(_check(f"decodes_end_to_end_{cam}", ok,
                                 f"{cam}: exit {result['returncode']}, {result['decoded_frames']}/"
                                 f"{result['expected_frames']} frames decoded, {result['decode_error_count']} decode errors",
                                 decode_errors=result["decode_errors"][:10], stderr_tail=result.get("stderr_tail", [])))
            defects = merge_intervals(list(map(tuple, result["black_intervals"])) + list(map(tuple, result["frozen_intervals"])))
            defects_all.extend(defects)
            black = sum(b - a for a, b in merge_intervals(list(map(tuple, result["black_intervals"]))))
            frozen = sum(b - a for a, b in merge_intervals(list(map(tuple, result["frozen_intervals"]))))
            share = (sum(b - a for a, b in defects) / total) if total > 0 else 0.0
            checks.append(_check(f"not_black_or_frozen_{cam}", share < cfg["max_defect_fraction"],
                                 f"{cam}: black {black:.1f} s, frozen {frozen:.1f} s "
                                 f"({share:.1%} of the recording; fail at {cfg['max_defect_fraction']:.0%})",
                                 black_seconds=round(black, 3), frozen_seconds=round(frozen, 3),
                                 defect_share=round(share, 4)))
        info["defect_intervals_s"] = merge_intervals(defects_all)

    graded = [c for c in checks if not c.get("informational")]
    failed = [c for c in graded if not c["passed"]]
    return {"passed": not failed, "checks": checks, "failed_checks": [c["check"] for c in failed], **info}


def _imu_checks(imu_t: np.ndarray, cam0: np.ndarray, nominal_hz: float | None,
                cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks = []
    if len(imu_t) < 2 or len(cam0) == 0:
        return [_check("imu_present", False, f"{len(imu_t)} IMU samples")]
    t = imu_t.astype(np.int64)
    first_frame, last_frame = int(cam0["t_ns"][0]), int(cam0["t_ns"][-1])
    tol = cfg["imu_coverage_tolerance_seconds"] * 1e9
    clock_offset = abs(int(t[0]) - first_frame) / 1e9
    checks.append(_check("imu_same_clock", clock_offset <= cfg["imu_same_clock_max_offset_seconds"],
                         f"first IMU sample is {clock_offset:.3f} s from the first frame", offset_s=round(clock_offset, 4)))
    start_late = (int(t[0]) - first_frame) / 1e9
    end_early = (last_frame - int(t[-1])) / 1e9
    covers = int(t[0]) <= first_frame + tol and int(t[-1]) >= last_frame - tol
    checks.append(_check("imu_covers_video", covers,
                         f"IMU starts {start_late:+.3f} s after and ends {end_early:+.3f} s before the video",
                         starts_after_video_s=round(start_late, 4), ends_before_video_s=round(end_early, 4)))
    dt = np.diff(t)
    max_gap = float(dt.max()) / 1e9
    non_mono = int(np.sum(dt <= 0))
    checks.append(_check("imu_continuous", non_mono == 0 and max_gap <= cfg["imu_max_gap_seconds"],
                         f"max IMU gap {max_gap * 1000:.2f} ms, {non_mono} non-increasing timestamps",
                         max_gap_ms=round(max_gap * 1000, 3), non_monotonic=non_mono))
    window = cfg["imu_rate_window_seconds"] * 1e9
    span = int(t[-1]) - int(t[0])
    rate_info: dict[str, Any] = {}
    if span >= window:
        edges = np.arange(int(t[0]), int(t[-1]) - window + 1, window, dtype=np.int64)
        counts = np.searchsorted(t, edges + int(window)) - np.searchsorted(t, edges)
        rates = counts / cfg["imu_rate_window_seconds"]
        nominal = float(nominal_hz) if nominal_hz else float(np.median(rates))
        deviation = np.abs(rates - nominal) / nominal
        ok = bool(np.all(deviation <= cfg["imu_rate_tolerance"]))
        rate_info = {"nominal_hz": round(nominal, 3), "min_window_hz": round(float(rates.min()), 2),
                     "max_window_hz": round(float(rates.max()), 2), "windows": int(len(rates))}
        checks.append(_check("imu_rate_steady", ok,
                             f"IMU rate per {cfg['imu_rate_window_seconds']} s window {rates.min():.1f}-{rates.max():.1f} Hz "
                             f"(nominal {nominal:.1f} Hz, tolerance {cfg['imu_rate_tolerance']:.0%})", **rate_info))
    else:
        overall = (len(t) - 1) / (span / 1e9) if span > 0 else 0.0
        nominal = float(nominal_hz) if nominal_hz else overall
        ok = nominal > 0 and abs(overall - nominal) / nominal <= cfg["imu_rate_tolerance"]
        checks.append(_check("imu_rate_steady", ok, f"IMU rate {overall:.1f} Hz over {span / 1e9:.1f} s (nominal {nominal:.1f})",
                             overall_hz=round(overall, 2)))
    return checks


def calibration_frame_indices(n_frames: int, step: int, pairs: int) -> list[int]:
    """`pairs` sample frames (multiples of `step`) spread evenly over the recording."""
    samples = math.ceil(n_frames / step) if n_frames > 0 else 0
    if samples == 0:
        return []
    count = min(pairs, samples)
    if count == 1:
        return [0]
    picks = sorted({round(j * (samples - 1) / (count - 1)) * step for j in range(count)})
    return [p for p in picks if p < n_frames]
