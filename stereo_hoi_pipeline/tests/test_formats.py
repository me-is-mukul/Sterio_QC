"""Every recording format through the real path: detect -> index -> decode -> clip.

Small synthetic sessions are built on disk in each format's layout (real HEVC / JPEG video made with
FFmpeg) and read through LocalStorage, which behaves like the bucket. Skipped without FFmpeg/libx265.
"""

from __future__ import annotations

import csv
import json
import shutil
import sqlite3
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from local_storage import LocalStorage
from test_streaming_video import _access_units, _has_x265

from video_pipeline.core.config import find_project_root
from video_pipeline.core.ego_config import load_ego_configuration
from video_pipeline.ingestion.formats import classify, make_reader
from video_pipeline.ingestion.gcs import SessionRef
from video_pipeline.ingestion.mcap_reader import assemble_session
from video_pipeline.qc.technical import DecodeSettings, decode_recording
from video_pipeline.segmentation.clip_writer import write_clip

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
pytestmark = pytest.mark.skipif(not (_has_x265() and FFPROBE), reason="needs ffmpeg with libx265 and ffprobe")

W, H = 192, 108                 # one eye
X265 = "keyint=30:min-keyint=30:bframes=0:scenecut=0:log-level=error"
START_US = 27_186_189_428       # device clock, microseconds


def run(args):
    subprocess.run([FFMPEG, "-v", "error", "-y", *args], check=True)


def segment_source(first: int, count: int, graph: str) -> list[str]:
    return ["-f", "lavfi", "-i", f"{graph},trim=start_frame={first}:end_frame={first + count},setpts=PTS-STARTPTS"]


SINGLE = f"testsrc2=size={W}x{H}:rate=30"
# side by side with a 16 px barcode strip on the left (EGO-CAM layout): width 16 + 2 * W
SBS_EGOCAM = (f"testsrc2=size={W}x{H}:rate=30,pad={W + 16}:{H}:16:0[a];testsrc=size={W}x{H}:rate=30[b];"
              "[a][b]hstack")
SBS_LEFT_BLACK = f"color=black:size={W}x{H}:rate=30[a];testsrc2=size={W}x{H}:rate=30[b];[a][b]hstack"


def frame_md5(args) -> list[str]:
    out = subprocess.run([FFMPEG, "-v", "error", *args, "-f", "framemd5", "-"], capture_output=True, text=True,
                         check=True).stdout
    return [line.rsplit(",", 1)[-1].strip() for line in out.splitlines() if line and not line.startswith("#")]


def config(tmp: Path, **overrides):
    return load_ego_configuration(find_project_root(), None, {"work.directory": str(tmp / "work"),
                                                              "upload.enabled": False, **overrides})


def settings(black_min=5.0) -> DecodeSettings:
    return DecodeSettings(fps=30, sample_step=60, sample_max_width=160, jpeg_quality=85, black_min_seconds=black_min,
                          black_pixel_threshold=0.1, black_picture_threshold=0.98, freeze_noise_db=-60,
                          freeze_min_seconds=black_min, timeout_seconds=120)


def index(storage: LocalStorage, relative: str, cfg, tmp: Path):
    session = SessionRef(bucket="b", raw_root="raw/", relative=relative)
    listing = classify(session, storage.list_objects("b", session.prefix))
    reader = make_reader(listing, storage, cfg, work_dir=tmp / "w", keep_dir=tmp / "keep")
    reader.fetch_metadata()
    checks = reader.required_files()
    assert all(c["passed"] for c in checks), checks
    stream_dir = tmp / "index"
    numbers = reader.extract(stream_dir)
    assemble_session(stream_dir, numbers)
    tables = {cam: np.load(stream_dir / f"{cam}_all.npy") for cam in ("cam0", "cam1")}
    return reader, stream_dir, tables, numbers


def decode(reader, stream_dir, tables, tmp, black_min=5.0):
    return decode_recording(FFMPEG, reader.stream_packets_from(stream_dir), tables, settings(black_min),
                            sample_dirs={c: tmp / "s" / c for c in tables}, calib_dirs={c: tmp / "c" / c for c in tables},
                            calib_indices=[0, 60], log_dir=tmp / "log", layout=reader.layout, codec=reader.codec,
                            crops=reader.eye_crops(stream_dir), harmless_errors=reader.harmless_decode_errors)


# ------------------------------------------------------------------------------------------------
# robocap: two MP4 cameras per segment, SQLite IMU, start time in the 'comment' tag
# ------------------------------------------------------------------------------------------------

def make_robocap(root: Path, frames_per_segment=(90, 90), cameras=("video_left_front", "video_right_front")) -> str:
    rel = "Op/03-11-2026/SDC606/20261103_054327_session1"
    folder = root / "raw" / rel
    folder.mkdir(parents=True)
    first = 0
    t_us = START_US
    for seg, count in enumerate(frames_per_segment, start=1):
        for k, cam in enumerate(cameras):
            # encoded straight into MP4: parameter sets only in hvcC (the reader must prepend them)
            run([*segment_source(first, count, SINGLE), "-c:v", "libx265", "-x265-params", X265,
                 "-tag:v", "hvc1", "-movflags", "use_metadata_tags", "-metadata", f"comment={t_us + 30 * k}",
                 "-metadata", "product=robocap", str(folder / f"robocap_segment{seg}_{cam}.mp4")])
        # a camera that is not used must not matter
        (folder / f"robocap_segment{seg}_video_left.mp4").write_bytes(b"not used")
        db = folder / f"robocap_segment{seg}_imu_left.db"
        con = sqlite3.connect(db)
        for table in ("gyro_data", "acc_data"):
            con.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY AUTOINCREMENT, imuid_ INTEGER, x INTEGER, "
                        "y INTEGER, z INTEGER, timestamp INTEGER)")
        start_ns = t_us * 1000 - 40_000_000
        end_ns = t_us * 1000 + int(count / 30 * 1e9)
        for t in range(start_ns, end_ns, 5_000_000):                 # 200 Hz
            con.execute("INSERT INTO gyro_data (imuid_, x, y, z, timestamp) VALUES (2, 655, 0, 0, ?)", (t,))
            con.execute("INSERT INTO acc_data (imuid_, x, y, z, timestamp) VALUES (2, 0, 0, 8192, ?)", (t + 1000,))
        con.commit()
        con.close()
        first += count
        t_us += int(round(count / 30 * 1e6))
    return rel


def test_robocap_index_decode_clip(tmp_path):
    storage = LocalStorage(tmp_path / "bucket")
    rel = make_robocap(tmp_path / "bucket")
    reader, stream_dir, tables, numbers = index(storage, rel, config(tmp_path), tmp_path)
    assert reader.format == "robocap" and reader.layout == "pair" and numbers == [1, 2]
    n = 180
    assert len(tables["cam0"]) == len(tables["cam1"]) == n
    # device times: start tag + presentation time; the right camera started 30 us later
    assert tables["cam0"]["t_ns"][0] == START_US * 1000
    assert tables["cam1"]["t_ns"][0] - tables["cam0"]["t_ns"][0] == 30_000
    assert abs(int(tables["cam0"]["t_ns"][90]) - (START_US * 1000 + 3_000_000_000)) < 1_000_000
    assert tables["cam0"]["key"][[0, 30, 90]].all() and tables["cam0"]["params"][tables["cam0"]["key"]].all()
    imu = np.load(stream_dir / "imu.npy")
    imu_t = np.load(stream_dir / "imu_t.npy")
    assert np.all(np.diff(imu_t) > 0)                                    # overlap between segments dropped
    assert np.allclose(imu[:, 0], 10 * np.pi / 180, atol=1e-6)           # 655 LSB / 65.5 = 10 deg/s
    assert np.allclose(imu[:, 5], 9.80665, atol=1e-4)                     # 8192 LSB = 1 g
    out = decode(reader, stream_dir, tables, tmp_path)
    for cam in ("cam0", "cam1"):
        assert out[cam]["returncode"] == 0 and out[cam]["decode_error_count"] == 0, out[cam]["decode_errors"]
        assert out[cam]["decoded_frames"] == n and out[cam]["samples_written"] == 3
        assert out[cam]["calibration_frames_written"] == 2
    finalized = []
    info = write_clip(clip_dir=tmp_path / "clip", clip_id="clip_000", ffmpeg=FFMPEG, ffprobe=FFPROBE,
                      stream=reader.stream_packets_from(stream_dir), tables=tables, left_cam="cam0", right_cam="cam1",
                      start=30, stop=150, fps=30, imu_t=imu_t, imu=imu, first_frame_ns=int(tables["cam0"]["t_ns"][0]),
                      metadata={}, timeout_seconds=120,
                      finalize=lambda p: finalized.append((p.name, frame_md5(["-i", str(p)]))))
    assert [f[0] for f in finalized] == ["left.mp4", "right.mp4"] and info["frames"] == 120
    folder = tmp_path / "bucket" / "raw" / rel
    source = frame_md5(["-i", str(folder / "robocap_segment1_video_left_front.mp4")]) + \
        frame_md5(["-i", str(folder / "robocap_segment2_video_left_front.mp4")])
    assert finalized[0][1] == source[30:150]                             # bit-identical across the segment border


def test_robocap_missing_parts_are_reported(tmp_path):
    storage = LocalStorage(tmp_path / "bucket")
    rel = make_robocap(tmp_path / "bucket", frames_per_segment=(60,))
    (tmp_path / "bucket" / "raw" / rel / "robocap_segment1_imu_left.db").unlink()
    session = SessionRef(bucket="b", raw_root="raw/", relative=rel)
    listing = classify(session, storage.list_objects("b", session.prefix))
    reader = make_reader(listing, storage, config(tmp_path), work_dir=tmp_path / "w", keep_dir=tmp_path / "k")
    reader.extract(tmp_path / "index")
    summary = json.loads((tmp_path / "index" / "seg_00001" / "segment.json").read_text())
    assert summary["missing_topics"] == ["imu_left.db"] and summary["cam0"]["frames"] == 60
    assert summary["device"]["product"] == "robocap"


# ------------------------------------------------------------------------------------------------
# Hydra: one side-by-side Annex-B stream + frames.idx + CSV sensor files
# ------------------------------------------------------------------------------------------------

def make_hydra(root: Path, n=180, graph=SBS_LEFT_BLACK) -> str:
    rel = "Op/30-10-2026/SDC087/S_20261030_111852"
    folder = root / "raw" / rel
    folder.mkdir(parents=True)
    raw = folder.parent / "tmp.h265"
    run(["-f", "lavfi", "-i", graph, "-frames:v", str(n), "-c:v", "libx265", "-x265-params",
         X265 + ":repeat-headers=1", "-f", "hevc", str(raw)])
    units = _access_units(raw.read_bytes())
    raw.unlink()
    assert len(units) == n
    (folder / "video.h265").write_bytes(b"".join(units))
    pts0 = 16_998_177_356
    with (folder / "frames.idx").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "host_ns", "pts_raw", "offset", "len"])
        offset = 0
        for i, u in enumerate(units):
            w.writerow([i, 16_990_905_460_352 + i * 33_333_333, 4_113_285_691 + i * 33_333, offset, len(u)])
            offset += len(u)
    with (folder / "h265_frame_meta.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_frame", "host_ns", "generation", "sample_index", "sensor_idx", "side", "frame_pts_us"])
        for i in range(n):
            if i == 5:
                continue                                                 # one frame without a meta row
            w.writerow([i, 0, 1, 0, 0, "left", pts0 + i * 33_333])
            w.writerow([i, 0, 1, 1, 1, "right", pts0 + i * 33_333 + 30])
    with (folder / "h265_imu.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_frame", "host_ns", "generation", "window_begin_pts_us", "window_end_pts_us", "sensor",
                    "sample_index", "x_raw", "y_raw", "z_raw", "temperature_raw", "pts_us", "x_scaled", "y_scaled",
                    "z_scaled", "units"])
        for k, t in enumerate(range(pts0 - 20_000, pts0 + n * 33_333 + 20_000, 2000)):      # 500 Hz
            w.writerow([0, 0, 1, 0, 0, "gyro", k, 0, 0, 0, 0, t, 5.0, 0.0, 0.0, "deg/s"])
            w.writerow([0, 0, 1, 0, 0, "accel", k, 0, 0, 0, 0, t + 100, 0.0, 0.0, -1.0, "g"])
        w.writerow([0, 0, 1, 0, 0, "accel", 0, 0, 0])                  # a cut-off last row is ignored
    (folder / "meta.json").write_text(json.dumps({"video": {"width": 2 * W, "height": H, "fps": 30}}))
    (folder / "session.json").write_text(json.dumps({"session": "S_20261030_111852"}))
    (folder / "stereo_calibration.bin").write_bytes(b"ZXCZ" + bytes(60))
    (folder / ".sensor-ready.json").write_text("{}")
    return rel


def test_hydra_index_and_per_eye_detection(tmp_path):
    storage = LocalStorage(tmp_path / "bucket")
    rel = make_hydra(tmp_path / "bucket")
    cfg = config(tmp_path, **{"formats.hydra.chunk_frames": 300})
    reader, stream_dir, tables, numbers = index(storage, rel, cfg, tmp_path)
    assert reader.format == "hydra" and reader.layout == "sbs" and numbers == [0]
    assert reader.eye_crops(stream_dir) == {"cam0": [0, W], "cam1": [W, 2 * W]}
    t0, t1 = tables["cam0"]["t_ns"], tables["cam1"]["t_ns"]
    assert t0[0] == 16_998_177_356_000 and (t1 - t0)[0] == 30_000
    assert t0[5] == t0[4] + 33_333_000                                    # missing meta row: from pts_raw
    assert np.array_equal(tables["cam0"]["ordinal"], tables["cam1"]["ordinal"])
    imu = np.load(stream_dir / "imu.npy")
    assert np.allclose(imu[:, 0], 5 * np.pi / 180) and np.allclose(imu[:, 5], -9.80665)
    out = decode(reader, stream_dir, tables, tmp_path, black_min=1.0)
    assert out["cam0"]["process"] == out["cam1"]["process"] == "sbs"    # one decoder for both eyes
    for cam in ("cam0", "cam1"):
        assert out[cam]["returncode"] == 0 and out[cam]["decode_error_count"] == 0
        assert out[cam]["decoded_frames"] == 180 and out[cam]["samples_written"] == 3
    # the left half is black: only cam0 reports it (detections are kept apart per eye)
    assert out["cam0"]["black_intervals"] == [(0.0, 6.0)] and out["cam1"]["black_intervals"] == []
    import cv2
    sample = cv2.imread(str(tmp_path / "s" / "cam1" / "000000.jpg"))
    assert sample.shape[1] == min(160, W)                                  # one eye, not the whole picture


def test_hydra_chunks_resume(tmp_path):
    storage = LocalStorage(tmp_path / "bucket")
    rel = make_hydra(tmp_path / "bucket", n=150, graph=SINGLE)
    cfg = config(tmp_path, **{"formats.hydra.chunk_frames": 300})
    reader, stream_dir, tables, numbers = index(storage, rel, cfg, tmp_path)
    assert numbers == [0] and len(tables["cam0"]) == 150
    storage.opened.clear()
    assert reader.extract(stream_dir) == [0]                              # already indexed: video not read again
    assert not any(name.endswith("video.h265") for name in storage.opened)
    assert not any(name.endswith(".csv") for name in storage.opened)       # parsed CSVs are cached


# ------------------------------------------------------------------------------------------------
# EGO-CAM live: side-by-side HEVC MP4 segments + frame_map.csv + calibration.json
# ------------------------------------------------------------------------------------------------

def make_egocam_live(root: Path, segments=(90, 90)) -> str:
    rel = "Op/10-11-2026/SDC195/ego_20261110_163352_16926E"
    folder = root / "raw" / rel
    folder.mkdir(parents=True)
    first = 0
    rows = []
    t = 1_794_308_632_357_496
    for seg, count in enumerate(segments):
        run([*segment_source(first, count, SBS_EGOCAM), "-c:v", "libx265", "-x265-params", X265, "-tag:v", "hvc1",
             str(folder / f"video_live_{seg:04d}.mp4")])
        for k in range(count):
            rows.append([seg, k, first + k + 1, t])
            t += 33_000 if k % 2 else 34_000                               # host receive jitter
        first += count
    with (folder / "frame_map.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment", "frame_in_segment", "capture_seq", "host_recv_us"])
        w.writerows(rows)
    (folder / "calibration.json").write_text(json.dumps({"image_size": [W, H],
                                                         "eye_crop_x": {"cam0": [16, 16 + W], "cam1": [16 + W, 16 + 2 * W]}}))
    (folder / "metadata.json").write_text(json.dumps({"camera": "GST MJPEG 4000x1200@30", "imu": {"embedded": True}}))
    (folder / "video_live_0002.mp4.part").write_bytes(b"incomplete")
    return rel


def test_egocam_live_index_decode_clip(tmp_path):
    storage = LocalStorage(tmp_path / "bucket")
    rel = make_egocam_live(tmp_path / "bucket")
    reader, stream_dir, tables, numbers = index(storage, rel, config(tmp_path), tmp_path)
    assert reader.format == "egocam_live" and reader.layout == "sbs" and reader.imu_unavailable
    assert numbers == [0, 1] and len(tables["cam0"]) == 180
    assert tables["cam0"]["t_ns"][1] - tables["cam0"]["t_ns"][0] == 34_000_000      # frame_map times
    assert reader.eye_crops(stream_dir) == {"cam0": [16, 16 + W], "cam1": [16 + W, 16 + 2 * W]}
    assert len(np.load(stream_dir / "imu_t.npy")) == 0
    out = decode(reader, stream_dir, tables, tmp_path)
    for cam in ("cam0", "cam1"):
        assert out[cam]["returncode"] == 0 and out[cam]["decode_error_count"] == 0
        assert out[cam]["decoded_frames"] == 180 and out[cam]["calibration_frames_written"] == 2
    import cv2
    calib = cv2.imread(str(tmp_path / "c" / "cam0" / "000.png"))
    assert calib.shape[:2] == (H, W)                                       # full-size crop of one eye
    finalized = []
    info = write_clip(clip_dir=tmp_path / "clip", clip_id="clip_000", ffmpeg=FFMPEG, ffprobe=FFPROBE,
                      stream=reader.stream_packets_from(stream_dir), tables=tables, left_cam="cam1", right_cam="cam0",
                      start=60, stop=150, fps=30, imu_t=np.zeros(0, dtype=np.int64), imu=np.zeros((0, 10)),
                      first_frame_ns=int(tables["cam1"]["t_ns"][0]), metadata={}, timeout_seconds=120,
                      finalize=lambda p: finalized.append((p.name, frame_md5(["-i", str(p)]))),
                      layout="sbs", codec="hevc", crops=reader.eye_crops(stream_dir))
    assert [f[0] for f in finalized] == ["stereo_sbs.mp4"] and info["files"]["stereo_sbs"] == "stereo_sbs.mp4"
    assert info["eye_crop_x"]["left"] == [16 + W, 16 + 2 * W]
    folder = tmp_path / "bucket" / "raw" / rel
    source = frame_md5(["-i", str(folder / "video_live_0000.mp4")]) + frame_md5(["-i", str(folder / "video_live_0001.mp4")])
    assert finalized[0][1] == source[60:150]


# ------------------------------------------------------------------------------------------------
# EGO-CAM capture.egoc: 'EGF2' records of side-by-side JPEGs
# ------------------------------------------------------------------------------------------------

def make_egoc(root: Path, n=150, cut_tail=True) -> tuple[str, list[bytes]]:
    rel = "Op/06-11-2026/SDC749/ego_20261106_143714_169238"
    folder = root / "raw" / rel
    folder.mkdir(parents=True)
    mjpeg = folder.parent / "f.mjpeg"
    run(["-f", "lavfi", "-i", SBS_EGOCAM, "-frames:v", str(n), "-c:v", "mjpeg", "-q:v", "5", "-f", "mjpeg", str(mjpeg)])
    data = mjpeg.read_bytes()
    mjpeg.unlink()
    jpegs, pos = [], 0
    while pos < len(data):
        end = data.index(b"\xff\xd9", pos) + 2
        while end < len(data) and data[end:end + 2] != b"\xff\xd8":
            end = data.index(b"\xff\xd9", end) + 2
        jpegs.append(data[pos:end])
        pos = end
    assert len(jpegs) == n
    records = bytearray()
    t = 1_793_940_831_293_444
    for i, j in enumerate(jpegs):
        records += struct.pack("<4sIQII", b"EGF2", i, t, len(j), 0) + j
        t += 33_333
    if cut_tail:
        records += struct.pack("<4sIQII", b"EGF2", n, t, 5000, 0) + b"\xff\xd8partial"   # recording cut off
    (folder / "capture.egoc").write_bytes(bytes(records))
    (folder / "calibration.json").write_text(json.dumps({"eye_crop_x": {"cam0": [16, 16 + W], "cam1": [16 + W, 16 + 2 * W]}}))
    return rel, jpegs


def test_egoc_index_decode_clip_and_resume(tmp_path):
    storage = LocalStorage(tmp_path / "bucket")
    rel, jpegs = make_egoc(tmp_path / "bucket")
    cfg = config(tmp_path, **{"formats.egocam.chunk_frames": 300})
    reader, stream_dir, tables, numbers = index(storage, rel, cfg, tmp_path)
    assert reader.format == "egocam_egoc" and reader.codec == "mjpeg" and numbers == [0]
    assert len(tables["cam0"]) == 150 and tables["cam0"]["key"].all()
    summary = json.loads((stream_dir / "seg_00000" / "segment.json").read_text())
    # a last frame cut off by the end of the recording is a note, not a failure
    assert summary["end_of_file"] and "cut off" in " ".join(summary["notes"]) and summary["missing_topics"] == []
    out = decode(reader, stream_dir, tables, tmp_path)
    for cam in ("cam0", "cam1"):
        assert out[cam]["returncode"] == 0 and out[cam]["decode_error_count"] == 0
        assert out[cam]["decoded_frames"] == 150
    packets = []
    reader.stream_packets_from(stream_dir)({"cam0": (tables["cam0"], 10, 13)}, lambda c, r, d: packets.append(d))
    assert packets == jpegs[10:13]                                         # the exact JPEG of every frame
    finalized = []
    write_clip(clip_dir=tmp_path / "clip", clip_id="clip_000", ffmpeg=FFMPEG, ffprobe=FFPROBE,
               stream=reader.stream_packets_from(stream_dir), tables=tables, left_cam="cam0", right_cam="cam1",
               start=0, stop=60, fps=30, imu_t=np.zeros(0, dtype=np.int64), imu=np.zeros((0, 10)),
               first_frame_ns=int(tables["cam0"]["t_ns"][0]), metadata={}, timeout_seconds=120,
               finalize=lambda p: finalized.append(p.name), layout="sbs", codec="mjpeg",
               crops=reader.eye_crops(stream_dir))
    assert finalized == ["stereo_sbs.mp4"]


def test_egoc_scan_in_chunks_resumes(tmp_path):
    storage = LocalStorage(tmp_path / "bucket")
    rel, _ = make_egoc(tmp_path / "bucket", n=700, cut_tail=False)
    cfg = config(tmp_path, **{"formats.egocam.chunk_frames": 300})
    session = SessionRef(bucket="b", raw_root="raw/", relative=rel)
    listing = classify(session, storage.list_objects("b", session.prefix))
    first = make_reader(listing, storage, cfg, work_dir=tmp_path / "w", keep_dir=tmp_path / "k", max_segments=2)
    assert first.extract(tmp_path / "index") == [0, 1]                    # interrupted after two chunks
    again = make_reader(listing, storage, cfg, work_dir=tmp_path / "w", keep_dir=tmp_path / "k")
    assert again.extract(tmp_path / "index") == [0, 1, 2]                 # continues with the third
    assemble_session(tmp_path / "index", [0, 1, 2])
    t = np.load(tmp_path / "index" / "cam0_all.npy")
    assert len(t) == 700 and np.all(np.diff(t["t_ns"]) == 33_333_000)


# ------------------------------------------------------------------------------------------------
# Whole pipeline (run_session, no upload) on a robocap and an EGO-CAM session
# ------------------------------------------------------------------------------------------------

class _Hands:
    """Stand-in for MediaPipe: the wearer's hands are always visible."""

    def __init__(self):
        from video_pipeline.inference.ego_labeller import FrameLabeller

        class Labeller(FrameLabeller):
            def describe(self):
                return {"provider": "test"}

            def label(self, sample):
                return {"status": "ok", "labels": {"hands": "one", "hands_left_eye": 1, "hands_right_eye": 1,
                                                   "other_hands": 0}}

        self.labeller = Labeller("hands", ("hands",), 1)


def _run_pipeline(tmp_path, rel, cfg, storage):
    from video_pipeline.core.logging_setup import LoggingController
    from video_pipeline.inference.ego_labeller import MotionIdleLabeller
    from video_pipeline.orchestration.ego_session import Services, run_session

    def factory():
        labellers = {"hands": _Hands().labeller,
                     "idle": MotionIdleLabeller("idle", 1, cfg.get("motion"), cfg.get("labelling.interval_seconds"))}
        return labellers, {}

    controller = LoggingController("WARNING")
    services = Services(config=cfg, project_root=find_project_root(), storage=storage, logging_controller=controller,
                        model_factory=factory)
    try:
        session = SessionRef(bucket="b", raw_root="raw/", relative=rel)
        row = run_session(session, services, force=True, upload=False)
    finally:
        services.close()
        controller.shutdown()
    out = tmp_path / "work" / rel / "out"
    return row, json.loads((out / "qa_report.json").read_text()), out


def test_pipeline_robocap_pass(tmp_path):
    storage = LocalStorage(tmp_path / "bucket")
    rel = make_robocap(tmp_path / "bucket", frames_per_segment=(600, 510))       # 37 s
    cfg = config(tmp_path, **{"source.bucket": "b", "source.raw_root": "raw/"})
    row, report, out = _run_pipeline(tmp_path, rel, cfg, storage)
    assert row["status"] == "PASS", report["fail_reasons"]
    assert report["session"]["format"] == "robocap"
    checks = {c["check"]: c for c in report["technical"]["checks"]}
    assert checks["imu_same_clock"]["passed"] and checks["imu_rate_steady"]["passed"]
    assert report["calibration"]["result"] == "unrectified"                       # robocap has no calibration file
    assert report["clips"] and set(report["clips"][0]["files"]) == {"left", "right", "frame_times", "imu"}
    clip = out / "clips" / "clip_000"
    assert (clip / "left.mp4").is_file() and len((clip / "imu.csv").read_text().splitlines()) > 100


def test_pipeline_egocam_live_pass_without_imu(tmp_path):
    storage = LocalStorage(tmp_path / "bucket")
    rel = make_egocam_live(tmp_path / "bucket", segments=(600, 510))
    cfg = config(tmp_path, **{"source.bucket": "b", "source.raw_root": "raw/"})
    row, report, out = _run_pipeline(tmp_path, rel, cfg, storage)
    assert row["status"] == "PASS", report["fail_reasons"]
    assert report["session"]["format"] == "egocam_live"
    assert report["imu"]["checked"] is False and "barcode" in report["imu"]["reason"]
    checks = {c["check"]: c for c in report["technical"]["checks"]}
    assert "imu_not_checked" in checks and "imu_continuous" not in checks
    assert set(report["clips"][0]["files"]) == {"stereo_sbs", "frame_times", "imu"}
    clip = json.loads((out / "clips" / "clip_000" / "clip.json").read_text())
    assert clip["layout"] == "side_by_side" and clip["recording_format"] == "egocam_live"
    assert json.loads((out / "recording" / "calibration.json").read_text())["eye_crop_x"]   # the recording's own
    assert (out / "recording" / "metadata.json").is_file() and (out / "calibration.json").is_file()
    labels = (out / "labels.csv").read_text()
    assert "moving" in labels                                                     # idle check ran on pictures only
