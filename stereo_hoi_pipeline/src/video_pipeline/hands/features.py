"""Per-hand features from MediaPipe Hand Landmarker output (pure functions, no MediaPipe import).

* Image features (normalised 0..1 coordinates of one camera view): hand size, pointing direction,
  bounding box. Used to tell the camera wearer's hands from other people's.
* Shape features (MediaPipe's metric 3D "world" landmarks, independent of the viewpoint):
  finger curl and thumb-to-finger pinch. Used to detect a grasp.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

LANDMARK_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}

Point2 = Sequence[float]
Point3 = Sequence[float]


def _sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _angle_deg(u: Sequence[float], v: Sequence[float]) -> float:
    nu, nv = _norm(u), _norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    c = sum(a * b for a, b in zip(u, v)) / (nu * nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def shape_features(world: Sequence[Point3]) -> dict[str, float]:
    """Finger flexion (sum of MCP+PIP+DIP flexion, degrees) and pinch ratios from 21x3 metric landmarks."""
    curls = []
    for mcp, pip, dip, tip in FINGERS.values():
        chain = [world[0], world[mcp], world[pip], world[dip], world[tip]]
        bones = [_sub(chain[i + 1], chain[i]) for i in range(4)]
        curls.append(sum(_angle_deg(bones[i], bones[i + 1]) for i in range(3)))
    palm = _norm(_sub(world[9], world[0]))
    # 99 = not measurable (degenerate palm); kept finite so the value stays valid JSON
    pinch = min(_norm(_sub(world[4], world[8])), _norm(_sub(world[4], world[12]))) / palm if palm > 1e-6 else 99.0
    return {"curl_mean_deg": round(sum(curls) / len(curls), 1), "pinch_ratio": round(pinch, 3)}


def image_features(xy: Sequence[Point2], aspect: float) -> dict[str, float]:
    """Image features from 21 normalised (x, y) points; aspect = view width / view height."""
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    # direction wrist -> middle knuckle in pixel-proportional units
    vx, vy = (xy[9][0] - xy[0][0]) * aspect, xy[9][1] - xy[0][1]
    length = math.hypot(vx, vy)
    return {
        "size": round(math.hypot((max(xs) - min(xs)) * aspect, max(ys) - min(ys)), 4),  # box diagonal / view height
        "pointing_down": round(vy / length, 3) if length > 1e-9 else 0.0,               # +1 = straight down
        "wrist_x": round(xy[0][0], 4),
        "box": [round(min(xs), 4), round(min(ys), 4), round(max(xs), 4), round(max(ys), 4)],
    }
