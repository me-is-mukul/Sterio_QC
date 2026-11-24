"""Model-free idle detection (provider: motion) and the per-labeller answer cache."""

import json

import cv2
import numpy as np
import pytest

from video_pipeline.analysis.cropping import decide_samples, flagged_runs
from video_pipeline.inference.ego_labeller import MotionIdleLabeller, Sample, label_samples

SETTINGS = {"image_width": 160, "pixel_threshold": 20, "max_changed_fraction": 0.02, "max_gyro_rad_s": 0.15}
INTERVAL = 2.0
N = 12                      # samples, 2 s apart
MOVING = {5, 6}             # samples where a "hand" moves across the picture


def _scene(rng_seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    base = cv2.resize(rng.integers(0, 255, (18, 32, 3), dtype=np.uint8), (640, 360), interpolation=cv2.INTER_CUBIC)
    return base


def _write_samples(folder, moving=MOVING):
    base = _scene()
    samples = []
    for k in range(N):
        img = base.copy()
        if k in moving:                                   # a large bright blob moved to a new place
            x = 60 + 120 * (k - min(moving))
            cv2.circle(img, (x, 200), 70, (255, 255, 255), -1)
        noise = np.random.default_rng(100 + k).normal(0, 2, img.shape)   # sensor noise / JPEG
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        path = folder / f"{k:06d}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        samples.append(Sample(k, k * 60, int(k * INTERVAL * 1e9), k * INTERVAL, path))
    return samples


def _imu(still_gyro: float, *, moving_head_between=None):
    t = np.arange(0, int((N * INTERVAL + 1) * 1e9), int(1e9 / 400), dtype=np.int64)
    imu = np.zeros((len(t), 10), dtype=np.float32)
    imu[:, 0] = still_gyro
    if moving_head_between:
        a, b = moving_head_between
        imu[(t >= a * 1e9) & (t <= b * 1e9), 1] = 1.0     # 1 rad/s head turn
    return t, imu


def _label_all(labeller, samples):
    return {s.index: labeller.label(s) for s in samples}


def test_still_picture_and_still_head_is_idle(tmp_path):
    samples = _write_samples(tmp_path, moving=set())
    labeller = MotionIdleLabeller("idle", 1, SETTINGS, INTERVAL)
    labeller.begin_session(samples, *_imu(0.01))
    out = _label_all(labeller, samples)
    assert all(r["status"] == "ok" and r["labels"]["activity"] == "idle" for r in out.values())
    assert out[0]["labels"]["changed_prev"] is None and out[0]["labels"]["changed_next"] < 0.02


def test_moving_hand_is_not_idle_and_neighbours_too(tmp_path):
    samples = _write_samples(tmp_path)
    labeller = MotionIdleLabeller("idle", 1, SETTINGS, INTERVAL)
    labeller.begin_session(samples, *_imu(0.01))
    activity = {k: r["labels"]["activity"] for k, r in _label_all(labeller, samples).items()}
    # samples 4..7 differ from a neighbour (the blob appears at 5, moves at 6, disappears at 7)
    assert [k for k, a in activity.items() if a == "moving"] == [4, 5, 6, 7]


def test_moving_head_is_not_idle(tmp_path):
    samples = _write_samples(tmp_path, moving=set())
    labeller = MotionIdleLabeller("idle", 1, SETTINGS, INTERVAL)
    labeller.begin_session(samples, *_imu(0.01, moving_head_between=(9.5, 14.5)))
    activity = {k: r["labels"]["activity"] for k, r in _label_all(labeller, samples).items()}
    assert [k for k, a in activity.items() if a == "moving"] == [5, 6, 7]


def test_no_imu_data_is_never_idle(tmp_path):
    samples = _write_samples(tmp_path, moving=set())
    labeller = MotionIdleLabeller("idle", 1, SETTINGS, INTERVAL)
    labeller.begin_session(samples, np.zeros(0, dtype=np.int64), np.zeros((0, 10), dtype=np.float32))
    out = _label_all(labeller, samples)
    assert all(r["labels"]["activity"] == "moving" and r["labels"]["gyro_median_rad_s"] is None for r in out.values())


def test_motion_and_hands_drive_cuts(tmp_path):
    samples = _write_samples(tmp_path)
    labeller = MotionIdleLabeller("idle", 1, SETTINGS, INTERVAL)
    labeller.begin_session(samples, *_imu(0.01))
    labels = {"idle": _label_all(labeller, samples),
              "hands": {s.index: {"status": "ok", "labels": {"hands": "one"}} for s in samples}}
    rows = [{"index": s.index, "frame": s.frame, "t_s": s.t_s} for s in samples]
    decisions = decide_samples(rows, labels, {"hands": ["hands"], "idle": ["activity"]}, min_models=1)
    cuts = flagged_runs(decisions, "idle", 6, "idle")
    # idle runs: 0..3 (6 s span) and 8..11 (6 s span); the moving stretch 4..7 is kept
    assert [(c.first_frame, c.last_frame) for c in cuts] == [(0, 180), (480, 660)]
    assert flagged_runs(decisions, "no_hands", 6, "no_hands") == []


def test_cache_is_reused_only_for_the_same_settings(tmp_path):
    samples = _write_samples(tmp_path, moving=set())
    imu = _imu(0.01)
    cache = tmp_path / "labels_idle.jsonl"
    first = MotionIdleLabeller("idle", 2, SETTINGS, INTERVAL)
    first.begin_session(samples, *imu)
    assert all(r["labels"]["activity"] == "idle" for r in label_samples(first, samples, cache, max_failure_ratio=0).values())
    lines = cache.read_text().splitlines()
    assert len(lines) == N and all(json.loads(x)["labeller"] == first.fingerprint() for x in lines)
    # same settings: nothing is labelled again
    again = MotionIdleLabeller("idle", 2, SETTINGS, INTERVAL)
    again.begin_session(samples, *imu)
    label_samples(again, samples, cache, max_failure_ratio=0)
    assert len(cache.read_text().splitlines()) == N
    # stricter head threshold: every sample is labelled again, and the new answers are used
    strict = MotionIdleLabeller("idle", 2, {**SETTINGS, "max_gyro_rad_s": 0.005}, INTERVAL)
    strict.begin_session(samples, *imu)
    out = label_samples(strict, samples, cache, max_failure_ratio=0)
    assert len(cache.read_text().splitlines()) == 2 * N
    assert all(r["labels"]["activity"] == "moving" for r in out.values())


def test_mediapipe_both_eyes(tmp_path):
    from video_pipeline.core.config import find_project_root
    from video_pipeline.core.ego_config import load_ego_configuration
    from video_pipeline.inference.ego_labeller import MediaPipeHandsLabeller

    root = find_project_root()
    settings = load_ego_configuration(root).get("mediapipe")
    assert settings["both_eyes"] is True
    samples = _write_samples(tmp_path, moving=set())
    labeller = MediaPipeHandsLabeller("hands", 1, root / settings["model_path"], settings)
    try:
        missing = labeller.label(samples[0])                      # no right-eye image given
        assert missing["status"] == "failed" and "right-eye" in missing["error"]
        both = Sample(0, 0, 0, 0.0, samples[0].image, samples[1].image)
        out = labeller.label(both)
        assert out["status"] == "ok" and out["labels"]["hands"] == "none"
        assert out["labels"]["hands_left_eye"] == 0 and out["labels"]["hands_right_eye"] == 0
        one_eye = MediaPipeHandsLabeller("hands", 1, root / settings["model_path"], {**settings, "both_eyes": False})
        try:
            assert one_eye.label(samples[0])["labels"]["hands_right_eye"] is None
            assert one_eye.fingerprint() != labeller.fingerprint()
        finally:
            one_eye.close()
    finally:
        labeller.close()


def test_label_before_begin_session_fails(tmp_path):
    samples = _write_samples(tmp_path, moving=set())
    with pytest.raises(Exception, match="failed or invalid"):
        label_samples(MotionIdleLabeller("idle", 1, SETTINGS, INTERVAL), samples, tmp_path / "c.jsonl",
                      max_failure_ratio=0)
