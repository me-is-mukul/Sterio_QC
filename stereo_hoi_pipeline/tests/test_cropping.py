"""Section 4 rules: idle / no-hands cuts, failed answers, keyframe snapping, 30 s minimum."""

from video_pipeline.analysis.cropping import (
    decide_samples,
    defect_frame_ranges,
    flagged_runs,
    kept_ranges,
    plan_pieces,
    snap_piece,
)

FPS = 30
STEP = 60  # one sample every 2 s


def _samples(n):
    return [{"index": k, "frame": k * STEP, "t_s": 2.0 * k} for k in range(n)]


def _labels(activity_a, activity_b, hands_a=None, hands_b=None, failed_b=()):
    n = len(activity_a)
    hands_a = hands_a or ["one"] * n
    hands_b = hands_b or ["one"] * n
    a = {k: {"status": "ok", "labels": {"activity": activity_a[k], "hands": hands_a[k]}} for k in range(n)}
    b = {k: ({"status": "failed"} if k in failed_b else
             {"status": "ok", "labels": {"activity": activity_b[k], "hands": hands_b[k]}}) for k in range(n)}
    return {"a": a, "b": b}


PROVIDES = {"a": ["hands", "activity"], "b": ["hands", "activity", "content"]}


def test_idle_needs_both_models():
    act_a = ["idle"] * 10
    act_b = ["idle"] * 5 + ["working"] + ["idle"] * 4
    d = decide_samples(_samples(10), _labels(act_a, act_b), PROVIDES, min_models=2)
    assert [x.idle for x in d] == [True] * 5 + [False] + [True] * 4


def test_run_of_exactly_10_seconds_is_cut_and_shorter_is_kept():
    # samples 0..5 idle: span t_last - t_first = 10 s -> cut; samples 7..11 idle: span 8 s -> kept
    act = ["idle"] * 6 + ["working"] + ["idle"] * 5 + ["working"] * 3
    d = decide_samples(_samples(len(act)), _labels(act, act), PROVIDES, min_models=2)
    cuts = flagged_runs(d, "idle", 10, "idle")
    assert len(cuts) == 1
    c = cuts[0]
    assert (c.first_frame, c.last_frame, c.samples) == (0, 5 * STEP, 6)
    assert c.t_end_s - c.t_start_s == 10.0


def test_9_99_seconds_is_kept():
    samples = [{"index": k, "frame": k * 60, "t_s": t} for k, t in enumerate([0.0, 2.0, 4.0, 6.0, 8.0, 9.99])]
    act = ["idle"] * 6
    d = decide_samples(samples, _labels(act, act), PROVIDES, min_models=2)
    assert flagged_runs(d, "idle", 10, "idle") == []


def test_failed_answer_breaks_an_idle_run():
    act = ["idle"] * 12
    d = decide_samples(_samples(12), _labels(act, act, failed_b={5}), PROVIDES, min_models=2)
    assert not d[5].idle
    cuts = flagged_runs(d, "idle", 10, "idle")
    # 0..4 spans 8 s (kept), 6..11 spans 10 s (cut)
    assert [(c.first_frame, c.last_frame) for c in cuts] == [(6 * STEP, 11 * STEP)]


def test_no_hands_needs_all_models_to_say_none():
    act = ["working"] * 8
    ha = ["none"] * 8
    hb = ["none"] * 7 + ["one"]
    d = decide_samples(_samples(8), _labels(act, act, ha, hb), PROVIDES, min_models=2)
    cuts = flagged_runs(d, "no_hands", 10, "no_hands")
    assert [(c.first_frame, c.last_frame) for c in cuts] == [(0, 6 * STEP)]


def test_missing_sample_index_breaks_a_run():
    # sample 6 is missing: 0..5 spans 10 s (cut), 7..12 spans 10 s (cut), never one run 0..12
    samples = [s for s in _samples(13) if s["index"] != 6]
    act = ["idle"] * 13
    d = decide_samples(samples, _labels(act, act), PROVIDES, min_models=2)
    cuts = flagged_runs(d, "idle", 10, "idle")
    assert [(c.first_frame, c.last_frame) for c in cuts] == [(0, 5 * STEP), (7 * STEP, 12 * STEP)]
    # and 7..11 alone (8 s) would be kept
    d = decide_samples([s for s in _samples(12) if s["index"] != 6], _labels(act[:12], act[:12]), PROVIDES,
                       min_models=2)
    assert [(c.first_frame, c.last_frame) for c in flagged_runs(d, "idle", 10, "idle")] == [(0, 5 * STEP)]


def test_min_models_enforced():
    import pytest

    with pytest.raises(ValueError):
        decide_samples(_samples(2), _labels(["idle"] * 2, ["idle"] * 2), {"a": ["hands", "activity"]}, min_models=2)


def test_kept_ranges_complement():
    assert kept_ranges([(10, 19), (15, 30), (50, 59)], 100) == [(0, 10), (31, 50), (60, 100)]
    assert kept_ranges([], 5) == [(0, 5)]
    assert kept_ranges([(0, 4)], 5) == []
    assert kept_ranges([(0, 0), (4, 4)], 5) == [(1, 4)]


def test_snap_moves_start_forward_and_end_back():
    keys = list(range(0, 3000, 30))           # a keyframe every 1 s
    assert snap_piece(31, 950, keys, 3000) == (60, 930)
    assert snap_piece(30, 960, keys, 3000) == (30, 960)   # already on keyframes
    assert snap_piece(2950, 3000, keys, 3000) == (2970, 3000)  # next keyframe inside, runs to the end
    assert snap_piece(2985, 3000, keys, 3000) is None     # no keyframe inside
    assert snap_piece(31, 59, keys, 3000) is None         # start snaps to 60, past the end
    assert snap_piece(31, 60, keys, 3000) is None         # start 60, end snaps back to 60: empty
    assert snap_piece(2910, 3000, keys, 3000) == (2910, 3000)  # runs to the end: last frame kept


def test_plan_pieces_drops_short_pieces_and_keeps_both_ends():
    n = 30 * 200                              # 200 s
    times = [i / FPS for i in range(n)]
    keys = list(range(0, n, 30))
    # cut 40 s .. 50 s, and 65 s .. 100 s
    cuts = [(40 * FPS, 50 * FPS), (65 * FPS, 100 * FPS)]
    pieces = plan_pieces(n_frames=n, frame_times_s=times, keyframes=keys, cuts=cuts, fps=FPS, min_piece_seconds=30)
    assert [(p.start, p.stop, p.kept) for p in pieces] == [
        (0, 40 * FPS, True),                  # 0-40 s: 40 s
        (51 * FPS, 65 * FPS, False),          # starts on the next keyframe after 50 s; 14 s -> dropped
        (101 * FPS, n, True),                 # to the end
    ]
    assert pieces[0].duration_s == 40.0
    assert pieces[1].drop_reason.startswith("shorter than")
    assert abs(pieces[1].start_snap_s - (1 - 1 / FPS)) < 1e-3     # rounded to 0.1 ms


def test_piece_of_29_9_seconds_is_dropped_and_30_kept():
    n = 30 * 100
    times = [i / FPS for i in range(n)]
    keys = list(range(0, n, 30))
    # keep [0, 30 s) exactly: cut from frame 900 onwards
    p30 = plan_pieces(n_frames=n, frame_times_s=times, keyframes=keys, cuts=[(900, n - 1)], fps=FPS,
                      min_piece_seconds=30)
    assert p30[0].kept and p30[0].duration_s == 30.0
    p299 = plan_pieces(n_frames=n, frame_times_s=times, keyframes=keys, cuts=[(897, n - 1)], fps=FPS,
                       min_piece_seconds=30)
    # [0, 897) snaps back to the keyframe at 870 -> 29 s
    assert not p299[0].kept


def test_pieces_survive_json_roundtrip_with_numpy_times(tmp_path):
    """Regression: numpy frame times once turned `kept` into the string "False" (truthy on resume)."""
    import json

    import numpy as np

    from video_pipeline.core.io import read_json, write_json_atomic

    n = 30 * 100
    times = np.arange(n) / FPS                              # numpy, as in the pipeline
    keys = np.arange(0, n, 30)
    pieces = plan_pieces(n_frames=n, frame_times_s=times, keyframes=keys.tolist(), cuts=[(20 * FPS, 30 * FPS)],
                         fps=FPS, min_piece_seconds=30)
    path = tmp_path / "p.json"
    write_json_atomic(path, [p.to_dict() for p in pieces])
    loaded = read_json(path)
    assert [p["kept"] for p in loaded] == [False, True]
    assert all(type(p["kept"]) is bool and type(p["duration_s"]) is float for p in loaded)
    write_json_atomic(path, {"b": np.bool_(False), "f": np.float64(1.5), "a": np.arange(3), "i": np.int64(7)})
    assert json.loads(path.read_text()) == {"b": False, "f": 1.5, "a": [0, 1, 2], "i": 7}


def test_defect_ranges():
    assert defect_frame_ranges([(1.0, 2.0)], 30, 1000) == [(30, 59)]
    assert defect_frame_ranges([(0.0, 100.0)], 30, 300) == [(0, 299)]
    assert defect_frame_ranges([(5.0, 5.01)], 30, 1000) == [(150, 150)]
