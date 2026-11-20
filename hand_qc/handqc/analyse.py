"""Per-clip analysis (the expensive part): one decode pass, hand tracking, picture metrics.

Writes raw, threshold-free measurements to outputs/<name>/videos/<clip_id>/analysis/ :
  analysis.json          metadata, settings fingerprint, decode summary
  frames.csv.gz          one row per analysed frame per camera view: picture metrics
  detections.csv.gz      one row per detected hand: 21 landmarks (x, y px) + features
  blockiness.csv         FFmpeg blockdetect score on full-resolution frames
  decode_errors.txt      FFmpeg decoder error messages (empty when the decode is clean)
  preview_*.jpg          a few analysed frames, used for the annotated preview
All PASS/FAIL decisions are made later by score.py, so thresholds can be changed
without repeating this step.
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")          # quieten MediaPipe's C++ logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from . import __version__  # noqa: E402
from .config import analysis_settings  # noqa: E402
from .frames import FrameReader, geometry_for, stride_for  # noqa: E402
from .hands import LANDMARK_NAMES, HandTracker  # noqa: E402
from .manifest import Clip  # noqa: E402
from .probe import probe  # noqa: E402
from .quality import frame_quality  # noqa: E402
from .util import LOG, fingerprint, write_csv, write_json  # noqa: E402

ANALYSIS_VERSION = 2           # bump when the analysis code changes its output
PREVIEW_FRACTIONS = (0.2, 0.35, 0.5, 0.65, 0.8)

FRAME_FIELDS = ["frame", "time_s", "view", "hands_detected", "luma_mean", "luma_p02", "luma_p98", "sharpness",
                "lab_a", "lab_b", "lab_a_sd", "lab_b_sd", "neutral_a", "neutral_b", "neutral_share"]
FEATURE_FIELDS = ["handedness", "handedness_score", "mp_label", "palm_size", "pointing_down", "wrist_x_norm",
                  "wrist_y_norm", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "curl_index_deg", "curl_middle_deg",
                  "curl_ring_deg", "curl_pinky_deg", "curl_thumb_deg", "curl_mean_deg", "palm_length_m",
                  "pinch_index_ratio", "pinch_middle_ratio"]
LANDMARK_FIELDS = [f"{name}_{axis}" for name in LANDMARK_NAMES for axis in ("x", "y")]
DETECTION_FIELDS = ["frame", "time_s", "view", "view_width_px", "view_height_px", "hand_index"] + FEATURE_FIELDS \
    + LANDMARK_FIELDS


def analysis_fingerprint(clip: Clip, config: dict, model_sha256: str) -> str:
    return fingerprint({"analysis_version": ANALYSIS_VERSION, "settings": analysis_settings(config),
                        "model": model_sha256, "source": [clip.source_url, clip.size_bytes, clip.md5_base64],
                        "layout": clip.layout})


def analyse_clip(clip: Clip, video: Path, out_dir: Path, config: dict, model: Path, model_sha256: str) -> dict:
    """Analyse one clip; returns the analysis.json content."""
    started = time.monotonic()
    adir = out_dir / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    tools, a = config["tools"], config["analysis"]

    info = probe(tools["ffprobe"], video)
    if not info["fps"]:
        raise RuntimeError("frame rate unknown")
    geometry = geometry_for(clip.layout, info["display_width"], info["display_height"], int(a["max_view_width"]))
    stride = stride_for(info["fps"], a["stride_bands"])
    block_stride = max(1, int(round(info["fps"] / float(a["blockiness_fps"]))))
    expected_frames = info["nb_frames"] or int(round((info["duration_s"] or 0) * info["fps"]))
    expected_analysed = (expected_frames - 1) // stride + 1 if expected_frames else None
    preview_frames = {int(round(f * expected_frames / stride)) * stride for f in PREVIEW_FRACTIONS} \
        if expected_frames else set()
    LOG.info("[%s] %s %dx%d %.3f fps %.1fs, layout %s -> %d view(s) of %dx%d, every %d frame(s)",
             clip.clip_id, info["codec"], info["width"], info["height"], info["fps"], info["duration_s"] or 0,
             clip.layout, len(clip.views), geometry.out_view_w, geometry.out_view_h, stride)

    trackers = [HandTracker(model, a["mediapipe"]) for _ in clip.views]
    reader = FrameReader(tools["ffmpeg"], video, fps=info["fps"], geometry=geometry, stride=stride,
                         block_stride=block_stride, work_dir=adir)
    frame_rows: list[dict] = []
    detection_rows: list[dict] = []
    analysed = 0
    next_report = 0.1
    try:
        for n, t, frame in reader:
            ts_ms = int(round(t * 1000))
            for view_name, view_img, tracker in zip(clip.views, geometry.split(frame), trackers):
                q = frame_quality(view_img)
                hands = tracker.detect(view_img, ts_ms, geometry.view_w, geometry.view_h)
                frame_rows.append({"frame": n, "time_s": round(t, 4), "view": view_name,
                                   "hands_detected": len(hands), **q})
                for i, h in enumerate(hands):
                    row = {"frame": n, "time_s": round(t, 4), "view": view_name,
                           "view_width_px": round(geometry.view_w, 1), "view_height_px": round(geometry.view_h, 1),
                           "hand_index": i}
                    row.update({k: h[k] for k in FEATURE_FIELDS})
                    xy = h["xy"]
                    for j, name in enumerate(LANDMARK_NAMES):
                        row[f"{name}_x"] = round(float(xy[j, 0]), 1)
                        row[f"{name}_y"] = round(float(xy[j, 1]), 1)
                    detection_rows.append(row)
            if n in preview_frames:
                cv2.imwrite(str(adir / f"preview_{n:07d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
            analysed += 1
            if expected_analysed and analysed / expected_analysed >= next_report:
                LOG.info("[%s] %3.0f%% (%d/%d frames, %.0fs)", clip.clip_id, 100 * analysed / expected_analysed,
                         analysed, expected_analysed, time.monotonic() - started)
                next_report += 0.1
    finally:
        for tr in trackers:
            tr.close()

    blocks = reader.blockiness()
    write_csv(adir / "frames.csv.gz", FRAME_FIELDS, frame_rows, gz=True)
    write_csv(adir / "detections.csv.gz", DETECTION_FIELDS, detection_rows, gz=True)
    write_csv(adir / "blockiness.csv", ["frame", "time_s", "blockiness"],
              [{"frame": f, "time_s": round(f / info["fps"], 4), "blockiness": round(s, 4)} for f, s in blocks])
    (adir / "decode_errors.txt").write_text("\n".join(reader.errors) + ("\n" if reader.errors else ""),
                                            encoding="utf-8")
    (adir / "blockiness_raw.txt").unlink(missing_ok=True)

    import mediapipe  # noqa: PLC0415 (version only)
    meta = {
        "status": "ok",
        "clip_id": clip.clip_id,
        "analysis_version": ANALYSIS_VERSION,
        "fingerprint": analysis_fingerprint(clip, config, model_sha256),
        "settings": analysis_settings(config),
        "source": {"url": clip.source_url, "size_bytes": clip.size_bytes, "md5_base64": clip.md5_base64,
                   "file_name": clip.file_name},
        "probe": info,
        "layout": clip.layout,
        "views": clip.views,
        "geometry": {"columns": geometry.columns, "rows": geometry.rows, "view_width_px": geometry.view_w,
                     "view_height_px": geometry.view_h, "analysed_view_width_px": geometry.out_view_w,
                     "analysed_view_height_px": geometry.out_view_h},
        "stride": stride,
        "analysed_fps": info["fps"] / stride,
        "block_stride": block_stride,
        "frames_expected": expected_frames,
        "analysed_frames_expected": expected_analysed,
        "analysed_frames": analysed,
        "blockiness_frames": len(blocks),
        "decoder_exit_code": reader.returncode,
        "decode_error_lines": len(reader.errors),
        "detections": len(detection_rows),
        "elapsed_s": round(time.monotonic() - started, 1),
        "software": {"handqc": __version__, "mediapipe": mediapipe.__version__, "opencv": cv2.__version__,
                     "numpy": np.__version__, "python": platform.python_version(), "model_sha256": model_sha256},
    }
    if analysed == 0:
        meta["status"] = "failed"
        meta["error"] = "no frames could be decoded: " + "; ".join(reader.errors[:3])
    write_json(adir / "analysis.json", meta)
    LOG.info("[%s] analysed %d frames, %d hand detections, %d decoder messages in %.0fs", clip.clip_id, analysed,
             len(detection_rows), len(reader.errors), meta["elapsed_s"])
    return meta
