"""Write one kept piece as a clip: the video as MP4 (packets copied, no re-encode) + times + IMU.

Two-stream recordings give left.mp4 and right.mp4; side-by-side recordings (one picture holds both
eyes) give one stereo_sbs.mp4, with each eye's x range in clip.json.

The piece [start, stop) starts on a keyframe shared by both eyes. Its exact HEVC packets are streamed
from the recording (only the segments that hold them) straight into FFmpeg (`-c copy`), so the clip
is bit-identical to the recording and no raw video is stored. The eyes are written one after the
other; after each MP4 is verified with ffprobe (packet count, keyframe at the start) it is handed to
``finalize`` (the session uploads and deletes it), so at most one eye of one clip is on disk.

If the first packet lacks VPS/SPS/PPS (never seen so far), the parameter sets of the nearest earlier
packet that has them are prepended. MP4 timestamps are n / fps (the nominal rate); the measured
capture time of every frame is in frame_times.csv.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from video_pipeline.core.errors import SegmentationError, ToolExecutionError
from video_pipeline.core.io import write_json_atomic
from video_pipeline.core.pipe_feeder import PipeFeeder
from video_pipeline.core.tools import run_command
from video_pipeline.ingestion.mcap_reader import IMU_COLUMNS, parameter_set_prefix

# (requests, sink) -> packets per camera; see ingestion/formats/base.py
PacketStreamer = Callable[[Mapping[str, tuple[np.ndarray, int, int]], Callable[[str, int, bytes], None]], dict]


def _parameter_sets_row(table: np.ndarray, start: int) -> int | None:
    """Row whose parameter sets must be prepended (None: the first packet carries them)."""
    if table["params"][start]:
        return None
    earlier = np.flatnonzero(table["params"][:start])
    if not len(earlier):
        raise SegmentationError(f"no VPS/SPS/PPS before frame {start}; the clip could not be decoded")
    return int(earlier[-1])


def remux_streamed(ffmpeg: str, stream: PacketStreamer, cam: str, table: np.ndarray, start: int, stop: int, fps: int,
                   output: Path, timeout_seconds: float, codec: str = "hevc") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.stem + ".tmp.mp4")
    tmp.unlink(missing_ok=True)
    tag = ["-tag:v", "hvc1"] if codec == "hevc" else []
    cmd = [ffmpeg, "-hide_banner", "-nostats", "-loglevel", "error", "-y", "-f", codec, "-framerate", str(fps),
           "-i", "pipe:0", "-c", "copy", *tag, "-movflags", "+faststart", str(tmp)]
    prefix_row = _parameter_sets_row(table, start) if codec == "hevc" else None
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except OSError as exc:
        raise ToolExecutionError(f"Could not start FFmpeg: {exc}") from exc
    stderr_chunks: list[bytes] = []
    reader = threading.Thread(target=lambda: stderr_chunks.append(proc.stderr.read()), daemon=True)
    reader.start()
    feeder = PipeFeeder(proc)

    def sink(_cam: str, row: int, data: bytes) -> None:
        if row == prefix_row:                      # only its parameter sets are used
            feeder.write(parameter_set_prefix(data))
        elif row >= start:
            feeder.write(data)

    try:
        first = prefix_row if prefix_row is not None else start
        if prefix_row is not None and prefix_row + 1 < start:
            # stream the prefix packet alone, then the piece
            stream({cam: (table, prefix_row, prefix_row + 1)}, sink)
            first = start
        stream({cam: (table, first, stop)}, sink)
    except BaseException:
        proc.kill()
        feeder.close()
        proc.wait()
        reader.join()
        tmp.unlink(missing_ok=True)
        raise
    feeder.close()
    try:
        code = proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        proc.wait()
        tmp.unlink(missing_ok=True)
        raise ToolExecutionError(f"FFmpeg remux of {output.name} timed out") from exc
    reader.join()
    stderr = b"".join(stderr_chunks).decode("utf-8", "replace").strip()
    if code != 0:
        tmp.unlink(missing_ok=True)
        raise SegmentationError(f"FFmpeg remux of {output.name} failed (exit {code}): {stderr[-400:]}")
    tmp.replace(output)


def verify_clip(ffprobe: str, path: Path, expected_frames: int, timeout_seconds: float) -> dict[str, Any]:
    probe = run_command([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
                         "-show_entries", "stream=codec_name,width,height,nb_read_packets,r_frame_rate:format=duration",
                         "-of", "json", str(path)], timeout_seconds=timeout_seconds)
    first = run_command([ffprobe, "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1",
                         "-show_entries", "frame=key_frame,pict_type", "-of", "json", str(path)],
                        timeout_seconds=timeout_seconds)
    if probe.returncode != 0 or first.returncode != 0:
        raise SegmentationError(f"ffprobe failed on {path.name}: {probe.stderr.strip()[-300:]} {first.stderr.strip()[-300:]}")
    data = json.loads(probe.stdout)
    stream = (data.get("streams") or [{}])[0]
    frames = json.loads(first.stdout).get("frames") or [{}]
    packets = int(stream.get("nb_read_packets", -1))
    key = int(frames[0].get("key_frame", 0)) == 1
    if packets != expected_frames or not key:
        raise SegmentationError(f"{path.name}: {packets} packets (expected {expected_frames}), "
                                f"first frame keyframe: {key}")
    return {"codec": stream.get("codec_name"), "width": stream.get("width"), "height": stream.get("height"),
            "frames": packets, "first_frame_keyframe": key, "bytes": path.stat().st_size,
            "duration_s": float((data.get("format") or {}).get("duration", 0.0))}


def write_clip(*, clip_dir: Path, clip_id: str, ffmpeg: str, ffprobe: str, stream: PacketStreamer,
               tables: Mapping[str, np.ndarray], left_cam: str, right_cam: str,
               start: int, stop: int, fps: int, imu_t: np.ndarray, imu: np.ndarray, first_frame_ns: int,
               metadata: Mapping[str, Any], timeout_seconds: float,
               finalize: Callable[[Path], None] | None = None, layout: str = "pair", codec: str = "hevc",
               crops: Mapping[str, Sequence[int]] | None = None) -> dict[str, Any]:
    """``finalize(path)`` is called for each verified MP4 (e.g. upload it and delete the local copy).
    ``layout`` 'sbs': one stereo_sbs.mp4 from cam0's packets (both tables share them); ``crops`` gives
    each camera's x range inside that picture."""
    if clip_dir.exists():
        shutil.rmtree(clip_dir)
    clip_dir.mkdir(parents=True)
    n = stop - start
    verification = {}
    outputs = [("stereo_sbs", "cam0")] if layout == "sbs" else [("left", left_cam), ("right", right_cam)]
    for eye, cam in outputs:
        table = tables[cam]
        if not table["key"][start]:
            raise SegmentationError(f"{clip_id}: frame {start} is not a keyframe in {cam}")
        out = clip_dir / f"{eye}.mp4"
        remux_streamed(ffmpeg, stream, cam, table, start, stop, fps, out, timeout_seconds, codec=codec)
        verification[eye] = verify_clip(ffprobe, out, n, timeout_seconds)
        if finalize is not None:
            finalize(out)

    left, right = tables[left_cam], tables[right_cam]
    t0 = int(left["t_ns"][start])
    with (clip_dir / "frame_times.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["clip_frame", "recording_frame", "t_left_ns", "t_right_ns", "t_clip_s", "keyframe"])
        for k in range(start, stop):
            writer.writerow([k - start, k, int(left["t_ns"][k]), int(right["t_ns"][k]),
                             f"{(int(left['t_ns'][k]) - t0) / 1e9:.6f}", int(bool(left["key"][k] and right["key"][k]))])

    end_ns = int(left["t_ns"][stop]) if stop < len(left) else int(left["t_ns"][stop - 1]) + int(round(1e9 / fps))
    lo, hi = np.searchsorted(imu_t, t0, side="left"), np.searchsorted(imu_t, end_ns, side="left")
    with (clip_dir / "imu.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["t_ns", "t_clip_s", *IMU_COLUMNS])
        for t, row in zip(imu_t[lo:hi], imu[lo:hi]):
            writer.writerow([int(t), f"{(int(t) - t0) / 1e9:.6f}",
                             *["" if np.isnan(v) else f"{float(v):.7g}" for v in row]])

    info = {
        "clip_id": clip_id,
        "frames": n,
        "recording_frames": [start, stop],
        "recording_time_s": [round((t0 - first_frame_ns) / 1e9, 6), round((end_ns - first_frame_ns) / 1e9, 6)],
        "device_time_ns": [t0, end_ns],
        "duration_s": round((end_ns - t0) / 1e9, 6),
        "imu_samples": int(hi - lo),
        "eyes": {"left": left_cam, "right": right_cam},
        "files": ({"stereo_sbs": "stereo_sbs.mp4"} if layout == "sbs" else {"left": "left.mp4", "right": "right.mp4"})
        | {"frame_times": "frame_times.csv", "imu": "imu.csv"},
        "layout": "side_by_side" if layout == "sbs" else "two_streams",
        "verification": verification,
        **({"eye_crop_x": {"left": list(crops[left_cam]), "right": list(crops[right_cam]),
                           "note": "x range [start, end) of each eye inside stereo_sbs.mp4"}}
           if layout == "sbs" and crops else {}),
        **metadata,
    }
    write_json_atomic(clip_dir / "clip.json", info)
    return info
