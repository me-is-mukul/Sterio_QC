"""Section 1 helpers: eye pairing, frame timing, detections, calibration frame choice."""

import numpy as np

from video_pipeline.ingestion.mcap_reader import FRAME_DTYPE
from video_pipeline.qc.technical import (
    build_decode_command,
    calibration_frame_indices,
    decode_errors,
    DecodeSettings,
    evaluate_technical,
    EyeOutput,
    frame_timing,
    merge_intervals,
    pair_eyes,
    parse_detection_file,
)
from pathlib import Path

NS = 1_000_000_000
DT = NS // 30


def table(n, start_ns=10 * NS, key_every=30, offset_ns=0, first_key=0, drop=()):
    rows = []
    for i in range(n):
        if i in drop:
            continue
        rows.append((start_ns + i * DT + offset_ns, (i - first_key) % key_every == 0 and i >= first_key, True, 0, 0, 1))
    return np.array(rows, dtype=FRAME_DTYPE)


def test_pair_equal():
    r = pair_eyes(table(100), table(100, offset_ns=8000), tolerance_ns=5_000_000, max_edge_frames=3)
    assert r["ok"] and r["cam0_rows"] == [0, 100] and r["cam1_rows"] == [0, 100]


def test_pair_extra_frame_at_end_is_trimmed():
    r = pair_eyes(table(57), table(58), tolerance_ns=5_000_000, max_edge_frames=3)
    assert r["ok"] and r["cam0_rows"] == [0, 57] and r["cam1_rows"] == [0, 57]
    assert r["unpaired_edge_frames"]["cam1_end"] == 1


def test_pair_extra_leading_frame_skips_to_common_keyframe():
    # cam1 started one frame earlier; keyframes happen at the same instants in both eyes
    # (cam1 rows 1, 31, ... = cam0 rows 0, 30, ...)
    cam1 = table(100, start_ns=10 * NS - DT, first_key=1)
    cam0 = table(99)
    r = pair_eyes(cam0, cam1, tolerance_ns=5_000_000, max_edge_frames=3)
    assert r["ok"] and r["shift"] == 1
    a0, b0 = r["cam0_rows"]
    a1, b1 = r["cam1_rows"]
    assert b0 - a0 == b1 - a1
    assert cam0["key"][a0] and cam1["key"][a1]
    assert abs(int(cam0["t_ns"][a0]) - int(cam1["t_ns"][a1])) < 5_000_000


def test_pair_lost_second_frame_trims_the_first_pair():
    # robocap 20260518_212645_session31: cam0 lost its 2nd frame, so its 1st frame pairs with nothing
    r = pair_eyes(table(100, drop={1}), table(100), tolerance_ns=5_000_000, max_edge_frames=3)
    assert r["ok"] and r["mismatched_edge_pairs_trimmed"] == {"start": 1, "end": 0}
    a0, b0 = r["cam0_rows"]
    a1, b1 = r["cam1_rows"]
    assert b0 - a0 == b1 - a1 > 0
    # the same at the end of the recording
    r = pair_eyes(table(100, drop={98}), table(100), tolerance_ns=5_000_000, max_edge_frames=3)
    assert r["ok"] and r["mismatched_edge_pairs_trimmed"]["end"] == 1


def test_pair_inner_gap_fails():
    r = pair_eyes(table(100), table(101, drop={50}), tolerance_ns=5_000_000, max_edge_frames=3)
    assert not r["ok"] and r["unmatched_inside"] > 0


def test_pair_too_many_edge_frames_fails():
    r = pair_eyes(table(100), table(110), tolerance_ns=5_000_000, max_edge_frames=3)
    assert not r["ok"]


def test_frame_timing_detects_gap_and_rate():
    tm = frame_timing(table(300, drop={100, 101, 102, 103}), 30)
    assert tm["frames"] == 296 and abs(tm["fps"] - 30) < 0.01
    assert tm["max_gap_s"] > 0.15 and tm["dropped_frames_est"] == 4
    assert frame_timing(table(1), 30)["duration_s"] == round(1 / 30, 3)


def test_parse_detection_file_open_intervals_and_minimum(tmp_path):
    path = tmp_path / "detect_cam0.txt"
    path.write_text("\n".join([
        "frame:45   pts:45   pts_time:1.5", "lavfi.black_start=1.5",
        "frame:210  pts:210  pts_time:7", "lavfi.black_end=7",
        "frame:300  pts:300  pts_time:10", "lavfi.black_start=10",          # 1 s only: too short
        "frame:330  pts:330  pts_time:11", "lavfi.black_end=11",
        "frame:450  pts:450  pts_time:15", "lavfi.freezedetect.freeze_start=10",
        "lavfi.freezedetect.freeze_duration=6", "lavfi.freezedetect.freeze_end=16",
        "frame:2850 pts:2850 pts_time:95", "lavfi.freezedetect.freeze_start=90",
        "frame:2940 pts:2940 pts_time:98", "lavfi.black_start=98",           # open at the end, 2 s
    ]))
    d = parse_detection_file(path, 100.0, black_min_seconds=5, freeze_min_seconds=5)
    assert d["black"] == [(1.5, 7.0)]
    assert d["frozen"] == [(10.0, 16.0), (90.0, 100.0)]
    assert parse_detection_file(tmp_path / "missing.txt", 100.0, 5, 5) == {"black": [], "frozen": []}


def test_decode_errors_and_harmless_messages():
    stderr = "\n".join([
        "[hevc @ 0x3] [error] Could not find ref with POC 3",
        "[mjpeg @ 0x5] [error] unable to decode APP fields: Invalid data found when processing input",
        "[swscaler @ 0x4] [warning] deprecated pixel format used",
    ])
    assert len(decode_errors(stderr)) == 2
    assert decode_errors(stderr, ("unable to decode APP fields",)) == ["[hevc @ 0x3] [error] Could not find ref with POC 3"]


def test_merge_intervals():
    assert merge_intervals([(5, 6), (0, 2), (1, 3)]) == [(0, 3), (5, 6)]


def test_calibration_frame_indices_are_sample_frames():
    idx = calibration_frame_indices(30 * 1200, 60, 24)
    assert len(idx) == 24 and idx[0] == 0 and all(i % 60 == 0 for i in idx) and idx[-1] < 30 * 1200
    assert calibration_frame_indices(100, 60, 24) == [0, 60]
    assert calibration_frame_indices(0, 60, 24) == []


def test_decode_command_shape():
    s = DecodeSettings(fps=30, sample_step=60, sample_max_width=1280, jpeg_quality=85, black_min_seconds=5,
                       black_pixel_threshold=0.1, black_picture_threshold=0.98, freeze_noise_db=-60,
                       freeze_min_seconds=5, timeout_seconds=10)
    cmd = build_decode_command("ffmpeg", s, eyes=[EyeOutput("cam0", None, Path("s"), Path("c"))], calib_indices=[0, 60],
                               progress_path=Path("p"))
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "split=3" in graph and "not(mod(n,60))" in graph and "eq(n,0)+eq(n,60)" in graph
    assert "metadata=mode=print:file=detect_cam0.txt" in graph and "crop" not in graph
    assert cmd[cmd.index("-framerate") + 1] == "30" and "pipe:0" in cmd and cmd[cmd.index("-f") + 1] == "hevc"
    # side by side: one input cut into both eyes, MJPEG input
    sbs = build_decode_command("ffmpeg", s, eyes=[EyeOutput("cam0", [160, 2080], Path("s0"), Path("c0")),
                                                  EyeOutput("cam1", [2080, 4000], Path("s1"), Path("c1"))],
                               calib_indices=[], progress_path=Path("p"), codec="mjpeg")
    graph = sbs[sbs.index("-filter_complex") + 1]
    assert graph.startswith("[0:v]split=2[in0][in1]")
    assert "crop=w=1920:h=ih:x=160:y=0" in graph and "crop=w=1920:h=ih:x=2080:y=0" in graph
    assert "file=detect_cam0.txt" in graph and "file=detect_cam1.txt" in graph and "cout" not in graph
    assert sbs[sbs.index("-f") + 1] == "mjpeg" and sbs.count("pipe:0") == 1


def _cfg():
    return {"expected_fps": 30, "fps_tolerance": 0.5, "max_frame_gap_seconds": 0.1, "max_eye_offset_ms": 5,
            "max_unpaired_edge_frames": 3, "imu_max_gap_seconds": 0.05, "imu_coverage_tolerance_seconds": 0.1,
            "imu_same_clock_max_offset_seconds": 5, "imu_rate_window_seconds": 10, "imu_rate_tolerance": 0.05,
            "max_defect_fraction": 0.5, "min_recording_seconds": 30}


def test_required_files_first_segment_missing():
    listing = {"session_json": {"name": "x"}, "segments": [{"number": 1}, {"number": 2}]}
    r = evaluate_technical(listing=listing, segment_summaries=[], cam0=None, cam1=None, imu_t=None,
                           imu_nominal_hz=None, decode=None, cfg=_cfg())
    assert not r["passed"] and "first_segment_present" in r["failed_checks"]


def test_imu_starting_late_fails():
    cam0, cam1 = table(30 * 60), table(30 * 60)
    imu_t = np.arange(12 * NS, 71 * NS, NS // 467, dtype=np.int64)   # starts 2 s after the video
    listing = {"session_json": {"name": "x"}, "segments": [{"number": 0}]}
    pairing = pair_eyes(cam0, cam1, tolerance_ns=5_000_000, max_edge_frames=3)
    r = evaluate_technical(listing=listing, segment_summaries=[], cam0=cam0, cam1=cam1, imu_t=imu_t,
                           imu_nominal_hz=467.0, decode=None, cfg=_cfg(), pairing=pairing, paired_cam0=cam0)
    assert "imu_covers_video" in r["failed_checks"]
    assert "length" not in r["failed_checks"] and "eyes_paired" not in r["failed_checks"]


def test_frame_timing_on_host_receive_times():
    # EGO-CAM: arrival times jitter (one 108 ms wait, then a burst) but no frame was lost
    rng = np.random.default_rng(1)
    n = 900
    t = (10 * NS + np.arange(n) * DT + rng.integers(-8_000_000, 8_000_000, n)).astype(np.int64)
    t[100] = t[99] + 108_000_000
    t = np.maximum.accumulate(t + np.arange(n))                         # still increasing
    tab = np.zeros(n, dtype=FRAME_DTYPE)
    tab["t_ns"] = t
    seq = np.arange(1, n + 1)
    host = frame_timing(tab, 30, seq)
    assert abs(host["fps"] - 30) < 0.5 and host["max_gap_s"] == round(1 / 30, 4) and host["dropped_frames_est"] == 0
    assert host["host_max_interval_s"] >= 0.1                           # visible, but not a gap
    assert frame_timing(tab, 30)["max_gap_s"] >= 0.1                    # the device-clock rule would fail it
    seq[500:] += 3                                                       # three frames lost
    lost = frame_timing(tab, 30, seq)
    assert lost["dropped_frames_est"] == 3 and lost["max_gap_s"] == round(4 / 30, 4)


def test_frame_timing_mean_rate_without_counter():
    # capture.egoc: receive times only; a camera that delivers 28 of every 30 frames shows a low mean rate
    t = (10 * NS + np.cumsum(np.where(np.arange(600) % 15 == 14, 2 * DT, DT))).astype(np.int64)
    tab = np.zeros(len(t), dtype=FRAME_DTYPE)
    tab["t_ns"] = t
    r = frame_timing(tab, 30, mean_rate=True)
    assert 27.5 < r["fps"] < 28.5 and r["dropped_frames_est"] == 40 and r["max_gap_s"] == round(2 * DT / NS, 4)
    assert frame_timing(tab, 30)["fps"] == round(NS / DT, 4)            # the median rule would not see it
