"""Decision rules for the hands detected in ONE camera view of ONE frame.

1. Duplicates   two detections whose boxes overlap more than `duplicate_overlap` of the smaller box
                are the same hand; the less confident one is dropped.
2. Wearer       a hand is someone else's when it is smaller than `min_hand_size` (far away), or when it
                points down the frame (`pointing_down` > `max_pointing_down`, i.e. reaching towards the
                camera) while smaller than `large_hand_size`. At most two wearer hands (the largest).
3. Left/Right   MediaPipe's handedness (correct for non-mirrored egocentric video); if both wearer
                hands get the same label, the one further left in the image is the left hand.
4. Grasp        mean finger curl >= `power_curl_deg` (power grasp) or thumb-tip to index/middle-tip
                distance / palm length <= `pinch_ratio` (precision grasp).
These are the rules calibrated by eye for the hand_qc module on the same kind of footage.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _overlap(a: list[float], b: list[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / smaller if smaller > 0 else 0.0


def classify_view(hands: list[dict[str, Any]], rules: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the hands of one view with `role` (wearer/other), `hand` (Left/Right) and `grasp` set.

    Duplicates are removed from the returned list.
    """
    kept: list[dict[str, Any]] = []
    for h in sorted(hands, key=lambda h: -h["score"]):
        if not any(_overlap(h["box"], k["box"]) > rules["duplicate_overlap"] for k in kept):
            kept.append(h)

    candidates = []
    for h in kept:
        towards_camera = h["pointing_down"] > rules["max_pointing_down"] and h["size"] < rules["large_hand_size"]
        if h["size"] >= rules["min_hand_size"] and not towards_camera:
            candidates.append(h)
        else:
            h["role"], h["hand"] = "other", h["label"]
    candidates.sort(key=lambda h: -h["size"])
    for h in candidates[2:]:
        h["role"], h["hand"] = "other", h["label"]
    wearer = candidates[:2]
    for h in wearer:
        h["role"], h["hand"] = "wearer", h["label"]
    if len(wearer) == 2 and wearer[0]["hand"] == wearer[1]["hand"]:
        a, b = sorted(wearer, key=lambda h: h["wrist_x"])
        a["hand"], b["hand"] = "Left", "Right"

    for h in kept:
        h["grasp"] = h["role"] == "wearer" and (h["curl_mean_deg"] >= rules["power_curl_deg"]
                                                or h["pinch_ratio"] <= rules["pinch_ratio"])
    return kept
