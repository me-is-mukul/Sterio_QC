"""MediaPipe Hand Landmarker wrapper and per-hand features.

For every detected hand we keep:
  * the 21 landmarks in image space (x, y in pixels of the original camera view);
  * MediaPipe's handedness label (corrected for non-mirrored video) and confidence;
  * shape features from MediaPipe's 3D hand model (metric "world" landmarks, which are
    independent of the viewpoint): finger curl angles and thumb-to-finger pinch ratios;
  * image features used to tell the camera wearer's hands from other people's:
    palm size relative to the view and the wrist->knuckle pointing direction.
"""

from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
               (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))))


def shape_features(world: np.ndarray) -> dict:
    """Finger flexion and pinch features from 21x3 world landmarks (metres)."""
    feats = {}
    curls = []
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        chain = [world[0], world[mcp], world[pip], world[dip], world[tip]]
        bones = [chain[i + 1] - chain[i] for i in range(4)]
        curl = sum(_angle_deg(bones[i], bones[i + 1]) for i in range(3))   # flexion at MCP + PIP + DIP
        feats[f"curl_{name}_deg"] = round(curl, 1)
        curls.append(curl)
    thumb = [world[i] for i in (1, 2, 3, 4)]
    tb = [thumb[i + 1] - thumb[i] for i in range(3)]
    feats["curl_thumb_deg"] = round(_angle_deg(tb[0], tb[1]) + _angle_deg(tb[1], tb[2]), 1)
    feats["curl_mean_deg"] = round(float(np.mean(curls)), 1)
    palm = float(np.linalg.norm(world[9] - world[0]))
    feats["palm_length_m"] = round(palm, 4)
    if palm > 1e-6:
        feats["pinch_index_ratio"] = round(float(np.linalg.norm(world[4] - world[8])) / palm, 3)
        feats["pinch_middle_ratio"] = round(float(np.linalg.norm(world[4] - world[12])) / palm, 3)
    else:
        feats["pinch_index_ratio"] = feats["pinch_middle_ratio"] = ""
    return feats


def image_features(xy_px: np.ndarray, view_w: float, view_h: float) -> dict:
    """Wearer-related image features from 21x2 pixel landmarks."""
    v = xy_px[9] - xy_px[0]
    length = float(np.linalg.norm(v))
    x0, y0 = xy_px.min(axis=0)
    x1, y1 = xy_px.max(axis=0)
    return {
        "palm_size": round(length / view_h, 4),                         # wrist -> middle knuckle / view height
        "pointing_down": round(float(v[1] / length), 3) if length > 1e-6 else 0.0,   # +1 = straight down
        "wrist_x_norm": round(float(xy_px[0, 0] / view_w), 4),
        "wrist_y_norm": round(float(xy_px[0, 1] / view_h), 4),
        "bbox_x0": round(float(x0), 1), "bbox_y0": round(float(y0), 1),
        "bbox_x1": round(float(x1), 1), "bbox_y1": round(float(y1), 1),
    }


class HandTracker:
    """One MediaPipe Hand Landmarker in VIDEO mode (temporal tracking) for one camera view."""

    def __init__(self, model: Path, settings: dict):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=int(settings["num_hands"]),
            min_hand_detection_confidence=float(settings["min_hand_detection_confidence"]),
            min_hand_presence_confidence=float(settings["min_hand_presence_confidence"]),
            min_tracking_confidence=float(settings["min_tracking_confidence"]),
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.swap = bool(settings["swap_handedness"])
        self.last_ts = -1

    def close(self) -> None:
        self.landmarker.close()

    def detect(self, rgb: np.ndarray, timestamp_ms: int, view_w: float, view_h: float) -> list[dict]:
        if timestamp_ms <= self.last_ts:
            timestamp_ms = self.last_ts + 1          # MediaPipe needs strictly increasing timestamps
        self.last_ts = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        hands = []
        for i, lms in enumerate(result.hand_landmarks):
            xy = np.array([[lm.x * view_w, lm.y * view_h] for lm in lms], dtype=np.float64)
            world = np.array([[w.x, w.y, w.z] for w in result.hand_world_landmarks[i]], dtype=np.float64)
            cat = result.handedness[i][0]
            raw_label = cat.category_name
            label = {"Left": "Right", "Right": "Left"}.get(raw_label, raw_label) if self.swap else raw_label
            hand = {"mp_label": raw_label, "handedness": label, "handedness_score": round(float(cat.score), 3)}
            hand.update(image_features(xy, view_w, view_h))
            hand.update(shape_features(world))
            hand["xy"] = xy
            hands.append(hand)
        return hands
