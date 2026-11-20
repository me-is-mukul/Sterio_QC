"""Scoring: turn one clip's raw analysis into metrics, PASS/FAIL tests and deliverable files.

Fast and deterministic; re-run after any threshold change.
Writes into outputs/<name>/videos/<clip_id>/ :
  summary.json        every metric, every test (value, threshold, verdict) and the overall verdict
  landmarks.csv       per analysed frame and detected hand: role, hand, grasp flag, 21 landmarks (x, y px)
  hand_timeline.csv   per analysed frame: left/right/either/both present, grasping, HOI, other hands
  preview.jpg         an analysed frame with the detected hands drawn
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .hands import CONNECTIONS, LANDMARK_NAMES
from .manifest import Clip
from .timeline import events, longest_false_run_s, pct, tracking_losses
from .util import read_csv, read_json, write_csv, write_json

PASS, FAIL, INFO, NA = "PASS", "FAIL", "INFO", "N/A"
LANDMARK_FIELDS = [f"{name}_{axis}" for name in LANDMARK_NAMES for axis in ("x", "y")]
LANDMARK_OUT_FIELDS = ["clip_id", "frame", "time_s", "view", "hand_index", "role", "hand", "handedness_score",
                       "grasp"] + LANDMARK_FIELDS
TIMELINE_FIELDS = ["frame", "time_s", "left", "right", "either", "both", "grasp_left", "grasp_right", "hoi",
                   "other_hands"]
COLOURS = {"Left": (255, 160, 0), "Right": (0, 200, 255), "other": (60, 60, 230)}   # BGR


def _f(value) -> float:
    return float(value) if value not in ("", None) else math.nan


# ----------------------------------------------------------------------------- per-frame decisions
def hand_size(h: dict) -> float:
    """Diagonal of the landmark box / view height (robust to foreshortening, unlike the palm length)."""
    return math.hypot(_f(h["bbox_x1"]) - _f(h["bbox_x0"]), _f(h["bbox_y1"]) - _f(h["bbox_y0"]))         / _f(h["view_height_px"])


def _overlap(a: dict, b: dict) -> float:
    """Intersection area / area of the smaller of the two landmark boxes."""
    ax0, ay0, ax1, ay1 = (_f(a[k]) for k in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"))
    bx0, by0, bx1, by1 = (_f(b[k]) for k in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"))
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    smaller = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return iw * ih / smaller if smaller > 0 else 0.0


def upright_share(detections: list[dict], config: dict) -> tuple[float | None, int]:
    """Share (%) of NEAR hands that point up the frame.

    Near hands = the largest quarter of the clip's detections (and at least tests.orientation.min_hand_size):
    the wearer's own hands are the closest. Other people's (smaller) hands are left out because they face
    the camera and point down even in correctly oriented footage.
    """
    floor = float(config["tests"]["orientation"]["min_hand_size"])
    sized = [(hand_size(d), _f(d["pointing_down"])) for d in detections]
    sized = [(s, p) for s, p in sized if s >= floor]
    if not sized:
        return None, 0
    # the largest quarter of the clip's hands: the wearer's own hands are the closest ones
    cutoff = max(floor, float(np.percentile([s for s, _ in sized], 75)))
    dirs = [p for s, p in sized if s >= cutoff]
    return round(100.0 * sum(1 for x in dirs if x < 0) / len(dirs), 2), len(dirs)


def is_inverted(detections: list[dict], config: dict, layout: str) -> bool:
    """True when most near hands point DOWN the frame: footage upside down (or not head-mounted).

    The wearer rules are then applied in the inverted direction; the orientation test fails separately.
    """
    if not config["wearer"]["use_direction_rule"][layout]:
        return False
    share, count = upright_share(detections, config)
    ori = config["tests"]["orientation"]
    return share is not None and count >= int(ori["min_hands"]) and share < float(ori["min_upright_pct"])


def classify_view_hands(hands: list[dict], config: dict, layout: str, inverted: bool = False) -> None:
    """Annotate the detections of one camera view in one frame with role / hand / grasp (in place)."""
    w, g = config["wearer"], config["grasp"]
    use_direction = bool(w["use_direction_rule"][layout])
    sign = -1.0 if inverted else 1.0
    # 1. duplicates: the same hand detected twice
    ordered = sorted(hands, key=lambda h: -_f(h["handedness_score"]))
    kept: list[dict] = []
    for h in ordered:
        if any(_overlap(h, k) > float(w["duplicate_overlap"]) for k in kept):
            h["role"], h["hand"] = "duplicate", ""
        else:
            kept.append(h)
    # 2. wearer vs other people's hands
    candidates = []
    for h in kept:
        size = hand_size(h)
        near_enough = size >= float(w["min_hand_size"])
        towards_camera = use_direction and sign * _f(h["pointing_down"]) > float(w["max_pointing_down"]) \
            and size < float(w["large_hand_size"])
        if near_enough and not towards_camera:
            candidates.append(h)
        else:
            h["role"], h["hand"] = "other", h["handedness"]
    if len(candidates) > 2:                         # a person has two hands: keep the two largest
        candidates.sort(key=lambda h: -hand_size(h))
        for h in candidates[2:]:
            h["role"], h["hand"] = "other", h["handedness"]
        candidates = candidates[:2]
    # 3. left / right
    for h in candidates:
        h["role"], h["hand"] = "wearer", h["handedness"]
    if len(candidates) == 2 and candidates[0]["hand"] == candidates[1]["hand"]:
        if use_direction:                           # upright egocentric view: the left hand is the left one
            a, b = sorted(candidates, key=lambda h: sign * _f(h["wrist_x_norm"]))
        else:                                       # rotated tiles: trust the more confident label
            a, b = sorted(candidates, key=lambda h: -_f(h["handedness_score"]))
            if a["hand"] == "Right":
                a, b = b, a
        a["hand"], b["hand"] = "Left", "Right"
    # 4. grasp shape (wearer hands only)
    for h in hands:
        if h["role"] == "wearer":
            pinch = min(_f(h["pinch_index_ratio"]), _f(h["pinch_middle_ratio"]))
            h["grasp"] = int(_f(h["curl_mean_deg"]) >= float(g["power_curl_deg"])
                             or (not math.isnan(pinch) and pinch <= float(g["pinch_ratio"])))
        else:
            h["grasp"] = ""


# ----------------------------------------------------------------------------- tests
def _rng(value, lo=None, hi=None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return NA
    if lo is None and hi is None:
        return INFO
    if lo is not None and value < lo:
        return FAIL
    if hi is not None and value > hi:
        return FAIL
    return PASS


def _threshold_text(lo=None, hi=None, unit="") -> str:
    u = f" {unit}" if unit else ""
    if lo is not None and hi is not None:
        return f"{lo}-{hi}{u}"
    if lo is not None:
        return f">= {lo}{u}"
    if hi is not None:
        return f"<= {hi}{u}"
    return "reported only"


def _test(tid, group, name, value, unit, status, threshold, detail=""):
    if isinstance(value, float):
        value = round(value, 3)
    return {"id": tid, "group": group, "name": name, "value": value, "unit": unit, "threshold": threshold,
            "status": status, "detail": detail}


def _simple(tid, group, name, value, unit, spec, detail=""):
    lo, hi = spec.get("min"), spec.get("max")
    return _test(tid, group, name, value, unit, _rng(value, lo, hi), _threshold_text(lo, hi, unit), detail)


# ----------------------------------------------------------------------------- main entry
def score_clip(clip: Clip, out_dir: Path, config: dict) -> dict:
    adir = out_dir / "analysis"
    meta = read_json(adir / "analysis.json")
    info = meta["probe"]
    fps, stride = float(info["fps"]), int(meta["stride"])
    dt = stride / fps
    views = meta["views"]
    tests_cfg = config["tests"]

    frame_rows = read_csv(adir / "frames.csv.gz")
    detections = read_csv(adir / "detections.csv.gz")
    blocks = [float(r["blockiness"]) for r in read_csv(adir / "blockiness.csv")]      # FFmpeg blockdetect (info)
    grid = read_json(adir / "grid_blockiness.json")                                     # graded measure
    decode_errors = [ln for ln in (adir / "decode_errors.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]

    frames = sorted({int(r["frame"]) for r in frame_rows})
    index_of = {f: i for i, f in enumerate(frames)}
    n = len(frames)
    if n == 0:
        raise RuntimeError("analysis contains no frames")

    # ---- per-frame hand decisions -------------------------------------------------------------
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for d in detections:
        groups[(int(d["frame"]), d["view"])].append(d)
    inverted = is_inverted(detections, config, meta["layout"])
    for hands in groups.values():
        classify_view_hands(hands, config, meta["layout"], inverted)

    present = {v: {"Left": [False] * n, "Right": [False] * n} for v in views}
    grasping = {v: {"Left": [False] * n, "Right": [False] * n} for v in views}
    others = {v: [0] * n for v in views}
    for (frame, view), hands in groups.items():
        i = index_of[frame]
        for h in hands:
            if h["role"] == "wearer":
                present[view][h["hand"]][i] = True
                grasping[view][h["hand"]][i] = grasping[view][h["hand"]][i] or bool(h["grasp"])
            elif h["role"] == "other":
                others[view][i] += 1
    upright_pct, upright_n = upright_share([d for d in detections if d["role"] != "duplicate"], config)

    def combine(per_view: dict) -> dict:
        return {side: [any(per_view[v][side][i] for v in views) for i in range(n)] for side in ("Left", "Right")}

    rig_present, rig_grasp = combine(present), combine(grasping)
    rig_others = [max(others[v][i] for v in views) for i in range(n)]
    left, right = rig_present["Left"], rig_present["Right"]
    either = [a or b for a, b in zip(left, right)]
    both = [a and b for a, b in zip(left, right)]

    g = config["grasp"]
    ev_left, mask_left = events(rig_grasp["Left"], dt, min_duration_s=float(g["min_duration_s"]),
                                max_gap_s=float(g["max_gap_s"]))
    ev_right, mask_right = events(rig_grasp["Right"], dt, min_duration_s=float(g["min_duration_s"]),
                                  max_gap_s=float(g["max_gap_s"]))
    hoi = [a or b for a, b in zip(mask_left, mask_right)]
    tr = config["tracking"]
    loss_kw = {"min_tracked_s": float(tr["min_tracked_s"]), "min_lost_s": float(tr["min_lost_s"])}
    losses = {"Left": tracking_losses(left, dt, **loss_kw), "Right": tracking_losses(right, dt, **loss_kw)}
    n_losses = len(losses["Left"]) + len(losses["Right"])
    analysed_minutes = n * dt / 60.0
    grasp_events = sorted([dict(e, hand="Left") for e in ev_left] + [dict(e, hand="Right") for e in ev_right],
                          key=lambda e: e["start_s"])
    grasp_total = round(sum(e["duration_s"] for e in grasp_events), 2)

    hands_metrics = {
        "either_hand_pct": pct(either), "both_hands_pct": pct(both), "left_hand_pct": pct(left),
        "right_hand_pct": pct(right), "hoi_pct": pct(hoi), "untracked_pct": round(100.0 - pct(either), 2),
        "tracking_losses": n_losses, "tracking_losses_left": len(losses["Left"]),
        "tracking_losses_right": len(losses["Right"]),
        "tracking_losses_per_min": round(n_losses / analysed_minutes, 2) if analysed_minutes else None,
        "grasp_count": len(grasp_events), "grasp_count_left": len(ev_left), "grasp_count_right": len(ev_right),
        "grasp_duration_total_s": grasp_total,
        "grasp_duration_mean_s": round(grasp_total / len(grasp_events), 2) if grasp_events else 0.0,
        "longest_hand_gap_s": longest_false_run_s(either, dt),
        "other_hands_pct": pct([c > 0 for c in rig_others]), "other_hands_max": max(rig_others) if n else 0,
        "analysed_frames": n, "analysed_fps": round(1.0 / dt, 3), "frame_interval_s": round(dt, 4),
    }
    per_view = {}
    for v in views:
        lv, rv = present[v]["Left"], present[v]["Right"]
        per_view[v] = {"either_hand_pct": pct([a or b for a, b in zip(lv, rv)]),
                       "both_hands_pct": pct([a and b for a, b in zip(lv, rv)]),
                       "left_hand_pct": pct(lv), "right_hand_pct": pct(rv),
                       "other_hands_pct": pct([c > 0 for c in others[v]])}

    # ---- technical metrics ----------------------------------------------------------------------
    fq = config["frame_quality"]
    luma = [_f(r["luma_mean"]) for r in frame_rows]
    sharp = [_f(r["sharpness"]) for r in frame_rows]
    lab_a, lab_b = [_f(r["lab_a"]) for r in frame_rows], [_f(r["lab_b"]) for r in frame_rows]
    neutral = [(_f(r["neutral_a"]), _f(r["neutral_b"])) for r in frame_rows if r["neutral_a"] not in ("", None)]
    dark = sum(1 for x in luma if x < fq["dark_luma"])
    bright = sum(1 for x in luma if x > fq["bright_luma"])
    scene_a, scene_b = float(np.mean(lab_a)), float(np.mean(lab_b))
    neutral_frame_pct = round(100.0 * len(neutral) / len(frame_rows), 2)
    if neutral and neutral_frame_pct >= float(tests_cfg["colour_cast"]["min_neutral_frame_pct"]):
        cast_a, cast_b = float(np.mean([x[0] for x in neutral])), float(np.mean([x[1] for x in neutral]))
        cast = round(math.hypot(cast_a, cast_b), 2)
    else:
        cast_a = cast_b = cast = None
    bitrate = info.get("video_bitrate_bps") or info.get("container_bitrate_bps")
    pixels = int(info["width"]) * int(info["height"])
    rotation = int(info.get("rotation_deg") or 0) % 360
    missing_frames = (meta["analysed_frames_expected"] - meta["analysed_frames"]) \
        if meta.get("analysed_frames_expected") else 0
    technical = {
        "duration_s": info.get("duration_s"), "fps": fps, "codec": info.get("codec"),
        "codec_profile": info.get("codec_profile"), "width": info["width"], "height": info["height"],
        "video_bitrate_mbps": round(bitrate / 1e6, 3) if bitrate else None,
        "required_bitrate_mbps": round(float(tests_cfg["bitrate_mbps"]["min_at_1080p"]) * pixels / (1920 * 1080), 2),
        "decode_error_messages": len(decode_errors), "decoder_exit_code": meta.get("decoder_exit_code"),
        "frames_missing": missing_frames,
        "luma_mean": round(float(np.mean(luma)), 2), "luma_median": round(float(np.median(luma)), 2),
        "dark_frame_pct": round(100.0 * dark / len(luma), 2), "bright_frame_pct": round(100.0 * bright / len(luma), 2),
        "bad_exposure_frame_pct": round(100.0 * (dark + bright) / len(luma), 2),
        "sharpness_median": round(float(np.median(sharp)), 1),
        "sharp_frame_pct": round(100.0 * sum(1 for s in sharp if s >= fq["sharp_laplacian_var"]) / len(sharp), 2),
        "blockiness": grid["blockiness"],
        "blockiness_grid_px": [grid["columns"]["period_px"], grid["rows"]["period_px"]],
        "blockdetect_mean": round(float(np.mean(blocks)), 3) if blocks else None,
        "colour_cast": cast, "colour_cast_a": round(cast_a, 2) if cast_a is not None else None,
        "colour_cast_b": round(cast_b, 2) if cast_b is not None else None,
        "neutral_frame_pct": neutral_frame_pct,
        "scene_colour": round(math.hypot(scene_a, scene_b), 2), "scene_colour_a": round(scene_a, 2),
        "scene_colour_b": round(scene_b, 2),
        "rotation_deg": int(info.get("rotation_deg") or 0), "upright_hand_pct": upright_pct,
        "hands_for_orientation": upright_n, "hands_enter_from_top": inverted,
    }

    # ---- tests ----------------------------------------------------------------------------------
    T = tests_cfg
    tests = []
    tests.append(_simple("duration", "Technical", "Duration", technical["duration_s"], "s", T["duration_s"]))
    fps_min = T["fps"].get("min")
    fps_status = _rng(fps, (fps_min - float(T["fps"].get("tolerance", 0))) if fps_min is not None else None)
    tests.append(_test("fps", "Technical", "Frame rate", round(fps, 3), "fps", fps_status,
                       _threshold_text(fps_min, None, "fps")))
    req = technical["required_bitrate_mbps"]
    tests.append(_test("bitrate", "Technical", "Bitrate", technical["video_bitrate_mbps"], "Mbit/s",
                       _rng(technical["video_bitrate_mbps"], req),
                       f">= {req} Mbit/s ({T['bitrate_mbps']['min_at_1080p']} at 1080p, scaled to "
                       f"{info['width']}x{info['height']})"))
    allowed = [c.lower() for c in T["codec"]["allowed"]]
    tests.append(_test("codec", "Technical", "Codec", info.get("codec"), "",
                       PASS if (info.get("codec") or "").lower() in allowed else FAIL, "one of " + ", ".join(allowed),
                       info.get("codec_profile") or ""))
    dec_count = len(decode_errors) + (1 if meta.get("decoder_exit_code") not in (0, None) else 0) \
        + (1 if missing_frames else 0)
    dec_detail = "; ".join(filter(None, [
        f"{len(decode_errors)} decoder message(s)" if decode_errors else "",
        f"decoder exit code {meta.get('decoder_exit_code')}" if meta.get("decoder_exit_code") not in (0, None) else "",
        f"{missing_frames} analysed frame(s) missing" if missing_frames else ""])) or "clean full decode"
    tests.append(_test("decode_errors", "Technical", "Decode errors", dec_count, "",
                       _rng(dec_count, None, T["decode_errors"]["max"]), _threshold_text(None, T["decode_errors"]["max"]),
                       dec_detail))
    lum = T["luminance"]
    lum_ok = lum["min_mean"] <= technical["luma_mean"] <= lum["max_mean"] and \
        technical["bad_exposure_frame_pct"] <= lum["max_bad_frame_pct"]
    tests.append(_test("luminance", "Technical", "Luminance", technical["luma_mean"], "/255", PASS if lum_ok else FAIL,
                       f"mean {lum['min_mean']}-{lum['max_mean']}, <= {lum['max_bad_frame_pct']}% frames too dark/bright",
                       f"{technical['dark_frame_pct']}% too dark, {technical['bright_frame_pct']}% too bright"))
    sh = T["sharpness"]
    tests.append(_test("sharpness", "Technical", "Sharpness", technical["sharp_frame_pct"], "% sharp frames",
                       _rng(technical["sharp_frame_pct"], sh["min_sharp_frame_pct"]),
                       f">= {sh['min_sharp_frame_pct']}% of frames with Laplacian variance >= {fq['sharp_laplacian_var']}",
                       f"median Laplacian variance {technical['sharpness_median']}"))
    tests.append(_simple("blockiness", "Technical", "Blockiness", technical["blockiness"], "", T["blockiness"],
                         f"visible block-edge steps on the {grid['columns']['period_px']}x{grid['rows']['period_px']} px "
                         f"grid, in percentage points above the rest of the picture ({grid['frames']} full-resolution "
                         f"frames; 0 = none); FFmpeg blockdetect {technical['blockdetect_mean']} (info)"))
    if cast is None:
        cast_detail = f"not measurable: only {neutral_frame_pct}% of frames contain enough grey/white surfaces"
    else:
        cast_detail = f"grey/white surfaces a* {technical['colour_cast_a']}, b* {technical['colour_cast_b']} " \
                      f"({neutral_frame_pct}% of frames measurable); overall scene colour " \
                      f"{technical['scene_colour']} (content, not tested)"
    tests.append(_simple("colour_cast", "Technical", "Colour cast", cast, "Lab", T["colour_cast"], cast_detail))
    ori = T["orientation"]
    rot_ok = rotation in [int(r) % 360 for r in ori["allowed_rotations"]]
    if not config["wearer"]["use_direction_rule"][meta["layout"]]:
        ori_status = PASS if rot_ok else FAIL
        ori_detail = f"rotation metadata {technical['rotation_deg']} deg; content check not applicable " \
                     "(multi-camera tiles are mounted at fixed rotations)"
    elif upright_pct is None or upright_n < int(ori["min_hands"]):
        ori_status = PASS if rot_ok else FAIL
        ori_detail = f"rotation metadata {technical['rotation_deg']} deg; too few near hands ({upright_n}) " \
                     "for the content check"
    else:
        ori_status = PASS if rot_ok and upright_pct >= ori["min_upright_pct"] else FAIL
        ori_detail = f"rotation metadata {technical['rotation_deg']} deg (applied); {upright_pct}% of " \
                     f"{upright_n} near hands point up the frame"
        if inverted:
            ori_detail += "; hands enter from the top of the frame - footage is upside down or the camera is " \
                          "not head-mounted (wearer rules applied in the inverted direction)"
    tests.append(_test("orientation", "Technical", "Orientation", upright_pct, "% upright hands", ori_status,
                       f"rotation in {ori['allowed_rotations']}, >= {ori['min_upright_pct']}% upright hands",
                       ori_detail))

    hm = hands_metrics
    tests.append(_simple("either_hand_pct", "Hands", "Either hand", hm["either_hand_pct"], "%", T["either_hand_pct"]))
    tests.append(_simple("both_hands_pct", "Hands", "Both hands", hm["both_hands_pct"], "%", T["both_hands_pct"]))
    tests.append(_simple("left_hand_pct", "Hands", "Left hand", hm["left_hand_pct"], "%", T["left_hand_pct"]))
    tests.append(_simple("right_hand_pct", "Hands", "Right hand", hm["right_hand_pct"], "%", T["right_hand_pct"]))
    tests.append(_simple("hoi_pct", "Hands", "HOI (grasping)", hm["hoi_pct"], "%", T["hoi_pct"]))
    tests.append(_simple("tracking_losses", "Hands", "Tracking losses", hm["tracking_losses_per_min"], "/min",
                         T["tracking_losses_per_min"],
                         f"{n_losses} losses (left {len(losses['Left'])}, right {len(losses['Right'])}) in "
                         f"{analysed_minutes:.2f} min"))
    tests.append(_simple("untracked_pct", "Hands", "Untracked", hm["untracked_pct"], "%", T["untracked_pct"]))
    tests.append(_simple("grasp_count", "Hands", "Grasp count", hm["grasp_count"], "", T["grasp_count"],
                         f"left {hm['grasp_count_left']}, right {hm['grasp_count_right']}"))
    tests.append(_simple("grasp_duration", "Hands", "Grasp duration", hm["grasp_duration_total_s"], "s",
                         T["grasp_duration_s"], f"mean {hm['grasp_duration_mean_s']} s per grasp"))
    tests.append(_simple("longest_hand_gap", "Hands", "Longest hand gap", hm["longest_hand_gap_s"], "s",
                         T["longest_hand_gap_s"]))
    tests.append(_simple("other_hands_pct", "Hands", "Other people's hands", hm["other_hands_pct"], "%",
                         T["other_hands_pct"], f"up to {hm['other_hands_max']} at once"))

    failed = [t for t in tests if t["status"] == FAIL]
    overall = FAIL if failed else PASS
    reasons = [f"{t['name']} {t['value']}{(' ' + t['unit']) if t['unit'] and not t['unit'].startswith('%') else t['unit']}"
               f" (needs {t['threshold']})" for t in failed]

    # ---- deliverable files ----------------------------------------------------------------------
    landmark_rows = []
    for d in sorted(detections, key=lambda r: (int(r["frame"]), r["view"], int(r["hand_index"]))):
        if d["role"] == "duplicate":
            continue
        row = {"clip_id": clip.clip_id, "frame": d["frame"], "time_s": d["time_s"], "view": d["view"],
               "hand_index": d["hand_index"], "role": d["role"], "hand": d["hand"],
               "handedness_score": d["handedness_score"], "grasp": d["grasp"]}
        row.update({k: d[k] for k in LANDMARK_FIELDS})
        landmark_rows.append(row)
    write_csv(out_dir / "landmarks.csv", LANDMARK_OUT_FIELDS, landmark_rows)
    write_csv(out_dir / "hand_timeline.csv", TIMELINE_FIELDS, [
        {"frame": f, "time_s": round(f / fps, 4), "left": int(left[i]), "right": int(right[i]),
         "either": int(either[i]), "both": int(both[i]), "grasp_left": int(mask_left[i]),
         "grasp_right": int(mask_right[i]), "hoi": int(hoi[i]), "other_hands": rig_others[i]}
        for i, f in enumerate(frames)])
    preview = _write_preview(out_dir, meta, groups)

    summary = {
        "clip": {"clip_no": clip.clip_no, "clip_id": clip.clip_id, "dataset": clip.dataset, "category": clip.category,
                 "task": clip.task, "camera": clip.camera, "layout": clip.layout, "views": views},
        "source": {"url": clip.source_url, "file_name": clip.file_name, "size_bytes": clip.size_bytes,
                   "resolution": clip.resolution, "original_reference": clip.original_reference},
        "overall": overall,
        "failed_tests": [t["id"] for t in failed],
        "reasons": reasons,
        "tests": tests,
        "technical": technical,
        "hands": hands_metrics,
        "per_view": per_view,
        "tracking_loss_events": losses,
        "grasp_events": grasp_events,
        "analysis": {k: meta.get(k) for k in ("fingerprint", "stride", "analysed_fps", "block_stride", "frames_expected",
                                               "analysed_frames", "blockiness_frames", "detections", "elapsed_s",
                                               "software", "settings", "geometry")},
        "probe": info,
        "files": {"landmarks": "landmarks.csv", "timeline": "hand_timeline.csv",
                  "preview": preview, "raw_analysis": "analysis/"},
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def _write_preview(out_dir: Path, meta: dict, groups: dict) -> str | None:
    """Draw the detected hands on the preview frame showing the most wearer hands."""
    adir = out_dir / "analysis"
    candidates = sorted(adir.glob("preview_*.jpg"))
    if not candidates:
        return None
    geo = meta["geometry"]
    views = meta["views"]

    def score(p: Path) -> int:
        f = int(p.stem.split("_")[1])
        return sum(1 for v in views for h in groups.get((f, v), []) if h["role"] == "wearer")

    best = max(candidates, key=lambda p: (score(p), -abs(candidates.index(p) - len(candidates) // 2)))
    frame_no = int(best.stem.split("_")[1])
    img = cv2.imread(str(best))
    sx = geo["analysed_view_width_px"] / geo["view_width_px"]
    sy = geo["analysed_view_height_px"] / geo["view_height_px"]
    thick = max(2, int(round(img.shape[1] / 900)))
    for vi, view in enumerate(views):
        ox = (vi % geo["columns"]) * geo["analysed_view_width_px"]
        oy = (vi // geo["columns"]) * geo["analysed_view_height_px"]
        for h in groups.get((frame_no, view), []):
            if h["role"] == "duplicate":
                continue
            pts = np.array([[_f(h[f"{nme}_x"]) * sx + ox, _f(h[f"{nme}_y"]) * sy + oy] for nme in LANDMARK_NAMES])
            pts = np.round(pts).astype(int)
            colour = COLOURS[h["hand"]] if h["role"] == "wearer" else COLOURS["other"]
            for a, b in CONNECTIONS:
                cv2.line(img, tuple(pts[a]), tuple(pts[b]), colour, thick, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, tuple(p), thick + 1, (255, 255, 255), -1, cv2.LINE_AA)
            label = (f"{h['hand']}{' grasp' if h.get('grasp') == 1 else ''}" if h["role"] == "wearer"
                     else "other person")
            org = (int(pts[:, 0].min()), max(20, int(pts[:, 1].min()) - 8))
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thick, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thick, colour, thick, cv2.LINE_AA)
    if img.shape[1] > 1920:
        img = cv2.resize(img, (1920, int(img.shape[0] * 1920 / img.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / "preview.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "preview.jpg"
