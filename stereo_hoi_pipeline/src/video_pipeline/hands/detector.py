"""Stereo-pair HOI classification with MediaPipe Hand Landmarker (no AI language model).

For each eye the hands are detected and classified (wearer / other, Left / Right, grasp) with
video_pipeline.hands.rules. The pair label is then:

    RELEVANT   a wearer hand is in a grasp pose in either eye (hand-object interaction)
    <setting>  wearer hands are visible but none is grasping -> detection.mediapipe.hands_without_grasp
               (default UNCERTAIN: preserved, because pressing or pushing is not a grasp)
    WASTE      no wearer hand in either eye

The result has the same fields as the vision-language classifier (classification, description,
per-camera observations, evidence, status) plus `hands` (21 landmarks per hand, normalised to the
eye image) and `hand_summary`, which the hand metrics and the dashboard use.
"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from video_pipeline.core.errors import InferenceError
from video_pipeline.hands.features import image_features, shape_features
from video_pipeline.hands.rules import classify_view

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
RULE_KEYS = ("min_hand_size", "max_pointing_down", "large_hand_size", "duplicate_overlap", "power_curl_deg",
             "pinch_ratio")
EYES = ("left_eye", "right_eye")


def ensure_model(path: Path, logger=None) -> str:
    """Make sure the hand model exists (downloading the pinned version if needed); return its SHA-256."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        if logger:
            logger.info("Downloading the MediaPipe hand model to %s ...", path)
        tmp = path.with_name(path.name + ".part")
        try:
            urllib.request.urlretrieve(MODEL_URL, tmp)  # noqa: S310 (fixed official URL)
        except OSError as exc:
            raise InferenceError(f"Could not download the MediaPipe hand model: {exc}") from exc
        os.replace(tmp, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MODEL_SHA256 and logger:
        logger.warning("Hand model checksum %s differs from the validated model; results may differ.", digest[:12])
    return digest


class MediaPipeStereoDetector:
    def __init__(self, model_path: Path, settings: Mapping[str, Any]) -> None:
        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        try:
            import cv2  # noqa: PLC0415 (optional dependency, loaded only for this detector)
            import mediapipe as mp  # noqa: PLC0415
            from mediapipe.tasks.python import BaseOptions, vision  # noqa: PLC0415
        except ImportError as exc:
            raise InferenceError("MediaPipe is not installed: pip install -r requirements.txt") from exc
        self._cv2, self._mp = cv2, mp
        self.mediapipe_version = mp.__version__
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,           # samples are seconds apart: no tracking
            num_hands=int(settings["num_hands"]),
            min_hand_detection_confidence=float(settings["min_hand_detection_confidence"]),
            min_hand_presence_confidence=float(settings["min_hand_presence_confidence"]),
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._rules = {k: float(settings[k]) for k in RULE_KEYS}
        self._without_grasp = settings["hands_without_grasp"]

    def close(self) -> None:
        self._landmarker.close()

    def _detect(self, jpeg: Path) -> list[dict[str, Any]]:
        cv2 = self._cv2
        bgr = cv2.imread(str(jpeg))
        if bgr is None:
            raise InferenceError(f"Cannot read image {jpeg.name}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        result = self._landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        hands = []
        for i, lms in enumerate(result.hand_landmarks):
            xy = [(lm.x, lm.y) for lm in lms]
            world = [(p.x, p.y, p.z) for p in result.hand_world_landmarks[i]]
            cat = result.handedness[i][0]
            hand = {"label": cat.category_name, "score": round(float(cat.score), 3),
                    "lm": [[round(x, 4), round(y, 4)] for x, y in xy]}
            hand.update(image_features(xy, w / h))
            hand.update(shape_features(world))
            hands.append(hand)
        return hands

    def classify_pair(self, left: Path, right: Path) -> dict[str, Any]:
        views = {"left_eye": classify_view(self._detect(left), self._rules),
                 "right_eye": classify_view(self._detect(right), self._rules)}
        present = {"Left": False, "Right": False}
        grasp = {"Left": False, "Right": False}
        others = 0
        cameras = {}
        for eye, hands in views.items():
            wearer = [h for h in hands if h["role"] == "wearer"]
            for h in wearer:
                present[h["hand"]] = True
                grasp[h["hand"]] = grasp[h["hand"]] or h["grasp"]
            n_other = sum(1 for h in hands if h["role"] == "other")
            others = max(others, n_other)
            parts = [f"{h['hand'].lower()} hand {'grasping' if h['grasp'] else 'visible'}" for h in
                     sorted(wearer, key=lambda h: h["hand"])]
            if n_other:
                parts.append(f"{n_other} other person's hand{'s' if n_other > 1 else ''}")
            cameras[eye] = {"hands_visible": bool(wearer), "interaction_detected": any(h["grasp"] for h in wearer),
                            "description": ", ".join(parts) if parts else "no hands"}

        if grasp["Left"] or grasp["Right"]:
            label = "RELEVANT"
            grasping = " and ".join(s.lower() for s in ("Left", "Right") if grasp[s])
            description = f"Wearer's {grasping} hand{'s' if grasping.count('and') else ''} grasping (hand-object interaction)"
        elif present["Left"] or present["Right"]:
            label = self._without_grasp
            description = "Wearer's hands visible but not grasping"
        else:
            label = "WASTE"
            description = "No hands of the camera wearer visible"
        evidence = [f"{eye.replace('_', ' ')}: {c['description']}" for eye, c in cameras.items()]
        return {
            "classification": label,
            "hoi_description": description,
            "evidence": evidence,
            "left_camera": cameras["left_eye"],
            "right_camera": cameras["right_eye"],
            "status": "ok",
            "validation_errors": [],
            "hand_summary": {"left": present["Left"], "right": present["Right"], "grasp_left": grasp["Left"],
                             "grasp_right": grasp["Right"], "other_hands": others},
            "hands": {eye: [{"role": h["role"], "hand": h["hand"], "grasp": h["grasp"], "score": h["score"],
                             "size": h["size"], "curl_mean_deg": h["curl_mean_deg"], "pinch_ratio": h["pinch_ratio"],
                             "lm": h["lm"]} for h in hands] for eye, hands in views.items()},
        }

    def timed_classify(self, left: Path, right: Path) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        result = self.classify_pair(left, right)
        return result, time.perf_counter() - started
