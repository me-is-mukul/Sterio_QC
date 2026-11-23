"""Section 2 of the QA sheet: stereo calibration check (decides rectified / unrectified; never fails).

Method, on held-out full-resolution left/right frame pairs (the factory calibration was not fitted
on them):
1. SIFT features in both raw (fisheye) images, ratio-test matches.
2. For each reading of the cam0->cam1 transform (its direction is not documented, see below), map
   the matched points through the Kannala-Brandt undistortion + rectification of
   cv2.fisheye.stereoRectify. This is the position each point has in the rectified frames (no
   image resampling, so no interpolation error).
3. Remove wrong matches with a RANSAC fundamental matrix (a free model: it does not force the
   epipolar lines to be horizontal, so a bad calibration still shows up as vertical offset).
4. Vertical offset |y_left - y_right| of the inliers: rectified if median <= 1.5 px,
   95th percentile <= 5 px and there are enough matches; otherwise unrectified.

Transform direction: FrameTransform(parent=cam0, child=cam1) is either the pose of cam1 in cam0
("pose": X_cam0 = R X_cam1 + t) or the point transform OpenCV expects ("points": X_cam1 = R X_cam0 + t).
Both are evaluated; the one with the smaller vertical offset is used and reported.

Eye order: from the undistorted (not rotated) matches, horizontal disparity x_cam0 - x_cam1. A scene
point appears further right in the LEFT eye, so mostly positive disparity means cam0 is the left eye.
"As labelled" (calibration.labelled_left: calibration) is the side the factory calibration puts each
camera on; eyes are "swapped" when the pixels disagree with it. On the first recording checked the
"pose" reading fitted (median 0.7 px vs 10.2 px) and both the geometry and the pixels put cam1 on the left.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        raise ValueError("zero quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def camera_model(intrinsics: Mapping[str, Any]) -> dict[str, Any]:
    if intrinsics.get("distortion_model") != "kannala_brandt":
        raise ValueError(f"unsupported distortion model {intrinsics.get('distortion_model')!r}")
    k = np.array(intrinsics["K"], dtype=np.float64).reshape(3, 3)
    d = np.array(intrinsics["D"], dtype=np.float64).reshape(4, 1)
    return {"K": k, "D": d, "size": (int(intrinsics["width"]), int(intrinsics["height"]))}


def transform_candidates(extrinsics: Mapping[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """OpenCV (R, T) with X_cam1 = R X_cam0 + T, for both readings of the message."""
    tr, rot = extrinsics["translation"], extrinsics["rotation"]
    t = np.array([tr.get("x", 0.0), tr.get("y", 0.0), tr.get("z", 0.0)], dtype=np.float64).reshape(3, 1)
    r = quaternion_to_matrix(rot.get("x", 0.0), rot.get("y", 0.0), rot.get("z", 0.0), rot.get("w", 1.0))
    return {"pose": (r.T, -r.T @ t), "points": (r, t)}


def match_pair(cv2, gray0: np.ndarray, gray1: np.ndarray, sift, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    k0, d0 = sift.detectAndCompute(gray0, None)
    k1, d1 = sift.detectAndCompute(gray1, None)
    if d0 is None or d1 is None or len(k0) < 2 or len(k1) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2))
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    good = []
    for pair in matcher.knnMatch(d0, d1, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
    p0 = np.array([k0[m.queryIdx].pt for m in good], dtype=np.float64).reshape(-1, 2)
    p1 = np.array([k1[m.trainIdx].pt for m in good], dtype=np.float64).reshape(-1, 2)
    return p0, p1


def _project(cv2, pts: np.ndarray, cam: Mapping[str, Any], r: np.ndarray | None, p: np.ndarray | None) -> np.ndarray:
    if len(pts) == 0:
        return pts.reshape(0, 2)
    out = cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2), cam["K"], cam["D"],
                                      R=r if r is not None else np.eye(3), P=p)
    return out.reshape(-1, 2)


def _inliers(cv2, a: np.ndarray, b: np.ndarray, threshold: float) -> np.ndarray:
    if len(a) < 8:
        return np.zeros(len(a), dtype=bool)
    _f, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, threshold, 0.999)
    return mask.ravel().astype(bool) if mask is not None else np.zeros(len(a), dtype=bool)


def evaluate_candidate(cv2, matches: Sequence[tuple[np.ndarray, np.ndarray]], cam0, cam1, r, t, cfg) -> dict[str, Any]:
    size = cam0["size"]
    r0, r1, p0, p1, q = cv2.fisheye.stereoRectify(cam0["K"], cam0["D"], cam1["K"], cam1["D"], size, r, t,
                                                  flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=size,
                                                  balance=0.0, fov_scale=1.0)
    dys: list[np.ndarray] = []
    pairs_used = 0
    for raw0, raw1 in matches:
        if len(raw0) < cfg["min_matches_per_pair"]:
            continue
        a = _project(cv2, raw0, cam0, r0, p0)
        b = _project(cv2, raw1, cam1, r1, p1)
        finite = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
        inside = finite & (a[:, 0] >= 0) & (a[:, 0] < size[0]) & (a[:, 1] >= 0) & (a[:, 1] < size[1]) \
            & (b[:, 0] >= 0) & (b[:, 0] < size[0]) & (b[:, 1] >= 0) & (b[:, 1] < size[1])
        a, b = a[inside], b[inside]
        keep = _inliers(cv2, a, b, cfg["ransac_threshold_px"])
        if keep.sum() < cfg["min_matches_per_pair"]:
            continue
        pairs_used += 1
        dys.append(np.abs(a[keep, 1] - b[keep, 1]))
    dy = np.concatenate(dys) if dys else np.zeros(0)
    result: dict[str, Any] = {"inlier_matches": int(len(dy)), "pairs_with_matches": pairs_used,
                              "median_vertical_px": None, "p95_vertical_px": None}
    if len(dy):
        result["median_vertical_px"] = round(float(np.median(dy)), 4)
        result["p95_vertical_px"] = round(float(np.percentile(dy, 95)), 4)
    result["rectification"] = {"R0": r0.tolist(), "R1": r1.tolist(), "P0": p0.tolist(), "P1": p1.tolist(),
                               "Q": q.tolist(), "image_size": list(size), "balance": 0.0, "fov_scale": 1.0,
                               "flags": "CALIB_ZERO_DISPARITY"}
    return result


def eye_order(cv2, matches: Sequence[tuple[np.ndarray, np.ndarray]], cam0, cam1, cfg) -> dict[str, Any]:
    focal = float((cam0["K"][0, 0] + cam1["K"][0, 0]) / 2)
    disparities = []
    for raw0, raw1 in matches:
        if len(raw0) < cfg["min_matches_per_pair"]:
            continue
        a = _project(cv2, raw0, cam0, None, None)       # normalised, undistorted, not rotated
        b = _project(cv2, raw1, cam1, None, None)
        finite = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
        a, b = a[finite], b[finite]
        keep = _inliers(cv2, a * focal, b * focal, cfg["ransac_threshold_px"])
        disparities.append((a[keep, 0] - b[keep, 0]) * focal)
    d = np.concatenate(disparities) if disparities else np.zeros(0)
    threshold = cfg["eye_order_min_disparity_px"]
    positive, negative = int(np.sum(d > threshold)), int(np.sum(d < -threshold))
    decided = positive + negative
    result: dict[str, Any] = {"points_with_disparity": decided, "cam0_left_votes": positive, "cam1_left_votes": negative,
                              "median_disparity_px": round(float(np.median(d)), 3) if len(d) else None,
                              "left": None, "determined": False}
    if decided >= cfg["eye_order_min_points"]:
        share = max(positive, negative) / decided
        result["agreement"] = round(share, 4)
        if share >= cfg["eye_order_min_agreement"]:
            result["left"] = "cam0" if positive > negative else "cam1"
            result["determined"] = True
    return result


def check_calibration(pairs: Sequence[tuple[Path, Path]], calibration: Mapping[str, Any],
                      cfg: Mapping[str, Any]) -> dict[str, Any]:
    """``calibration``: the first intrinsics/extrinsics messages (dicts) from the recording."""
    import cv2  # noqa: PLC0415

    labelled_left = cfg["labelled_left"]
    base: dict[str, Any] = {"calibration_source": "factory", "pairs_requested": len(pairs)}
    missing = [k for k in ("cam0_intrinsics", "cam1_intrinsics", "stereo_extrinsics") if k not in calibration]
    if missing:
        fallback = "cam0" if labelled_left == "calibration" else labelled_left
        return {**base, "result": "unrectified", "reason": f"calibration messages missing: {missing}",
                "eye_order": {"left": fallback, "determined": False, "eyes_swapped": False,
                              "labelled_left": fallback, "note": "no calibration: eye order unknown, cam0 assumed left"}}
    cam0 = camera_model(calibration["cam0_intrinsics"])
    cam1 = camera_model(calibration["cam1_intrinsics"])
    base["intrinsics"] = {"cam0": calibration["cam0_intrinsics"], "cam1": calibration["cam1_intrinsics"]}
    base["extrinsics_cam0_to_cam1"] = calibration["stereo_extrinsics"]

    sift = cv2.SIFT_create(nfeatures=4000)
    matches = []
    for path0, path1 in pairs:
        g0 = cv2.imread(str(path0), cv2.IMREAD_GRAYSCALE)
        g1 = cv2.imread(str(path1), cv2.IMREAD_GRAYSCALE)
        if g0 is None or g1 is None:
            continue
        if (g0.shape[1], g0.shape[0]) != cam0["size"] or (g1.shape[1], g1.shape[0]) != cam1["size"]:
            raise ValueError(f"calibration frame size {g0.shape[::-1]} differs from the intrinsics {cam0['size']}")
        matches.append(match_pair(cv2, g0, g1, sift, cfg["ratio_test"]))
    base["pairs_loaded"] = len(matches)
    base["raw_matches"] = int(sum(len(m[0]) for m in matches))

    candidates = {}
    for name, (r, t) in transform_candidates(calibration["stereo_extrinsics"]).items():
        candidates[name] = evaluate_candidate(cv2, matches, cam0, cam1, r, t, cfg)
    scored = [(c["median_vertical_px"], name) for name, c in candidates.items() if c["median_vertical_px"] is not None]
    best_name = min(scored)[1] if scored else None
    best = candidates.get(best_name) if best_name else None

    enough = best is not None and best["inlier_matches"] >= cfg["min_inlier_matches"] \
        and best["pairs_with_matches"] >= cfg["min_pairs_with_matches"]
    rectified = bool(enough and best["median_vertical_px"] <= cfg["max_median_vertical_px"]
                     and best["p95_vertical_px"] <= cfg["max_p95_vertical_px"])
    if best is None:
        reason = "no usable feature matches"
    elif not enough:
        reason = (f"too few matches: {best['inlier_matches']} inliers over {best['pairs_with_matches']} pairs "
                  f"(need {cfg['min_inlier_matches']} over {cfg['min_pairs_with_matches']})")
    elif not rectified:
        reason = (f"vertical offset median {best['median_vertical_px']} px / p95 {best['p95_vertical_px']} px "
                  f"(limits {cfg['max_median_vertical_px']} / {cfg['max_p95_vertical_px']})")
    else:
        reason = "within limits"

    if labelled_left == "calibration":
        # The side the calibration geometry puts cam1 on (cam1 centre in cam0 coordinates).
        reading = best_name or "pose"
        r, t = transform_candidates(calibration["stereo_extrinsics"])[reading]
        cam1_centre_x = float((-r.T @ t)[0, 0])
        labelled_left = "cam1" if cam1_centre_x < 0 else "cam0"
        base["labelled_left_from"] = {"transform_reading": reading, "cam1_centre_x_m": round(cam1_centre_x, 5)}
    order = eye_order(cv2, matches, cam0, cam1, cfg)
    left = order["left"] if order["determined"] else labelled_left
    order.update(left=left, eyes_swapped=left != labelled_left, labelled_left=labelled_left)
    return {
        **base,
        "result": "rectified" if rectified else "unrectified",
        "reason": reason,
        "transform_reading": best_name,
        "median_vertical_px": best["median_vertical_px"] if best else None,
        "p95_vertical_px": best["p95_vertical_px"] if best else None,
        "inlier_matches": best["inlier_matches"] if best else 0,
        "pairs_with_matches": best["pairs_with_matches"] if best else 0,
        "candidates": {k: {kk: vv for kk, vv in v.items() if kk != "rectification"} for k, v in candidates.items()},
        "rectification": best["rectification"] if (best and rectified) else None,
        "eye_order": order,
    }
