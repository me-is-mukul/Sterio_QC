"""Real HEVC through the streaming path: index -> decode both eyes -> clip, no video file on disk.

A short test video is encoded with libx265 (keyframe every 30 frames, no B-frames, parameter sets
on every keyframe, like the device), split into one packet per frame, and written into a 2-segment
MCAP with the device's schemas. Skipped if FFmpeg or libx265 is not available.
"""

import hashlib
import shutil
import subprocess
from contextlib import contextmanager

import numpy as np
import pytest

from test_mcap_reader import TOPICS, _schemas

from functools import partial

from video_pipeline.ingestion.mcap_reader import assemble_session, extract_segment, last_times_after, stream_packets
from video_pipeline.qc.technical import DecodeSettings, decode_recording
from video_pipeline.segmentation.clip_writer import write_clip

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
N = 180                      # 6 s at 30 fps


def _has_x265() -> bool:
    if not FFMPEG:
        return False
    out = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return "libx265" in out


pytestmark = pytest.mark.skipif(not (_has_x265() and FFPROBE), reason="needs ffmpeg with libx265 and ffprobe")


def _access_units(stream: bytes) -> list[bytes]:
    """Split an Annex-B HEVC stream into one packet per picture."""
    starts = []
    i = stream.find(b"\x00\x00\x01")
    while i >= 0:
        s = i - 1 if i > 0 and stream[i - 1] == 0 else i
        starts.append((s, i + 3))
        i = stream.find(b"\x00\x00\x01", i + 3)
    units, current, has_vcl = [], b"", False
    for k, (s, h) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(stream)
        nal = stream[s:end]
        t = (stream[h] >> 1) & 0x3F
        first_slice = t < 32 and (stream[h + 2] & 0x80)
        if has_vcl and (t in (32, 33, 34, 35, 39) or first_slice):
            units.append(current)
            current, has_vcl = b"", False
        current += nal
        has_vcl = has_vcl or t < 32
    if current:
        units.append(current)
    return units


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    from mcap_protobuf.writer import Writer

    tmp = tmp_path_factory.mktemp("rec")
    raw = tmp / "test.h265"
    subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=30", "-frames:v", str(N),
                    "-c:v", "libx265", "-x265-params",
                    "keyint=30:min-keyint=30:bframes=0:scenecut=0:repeat-headers=1:log-level=error",
                    "-f", "hevc", str(raw)], check=True)
    units = _access_units(raw.read_bytes())
    assert len(units) == N
    video_cls, imu_cls = _schemas()
    dt = 33_333_333
    paths = {}
    for seg, (a, b) in enumerate([(0, 90), (90, N)]):
        path = tmp / f"ego_{seg:04d}.mcap"
        with path.open("wb") as handle, Writer(handle) as writer:
            for i in range(a, b):
                t = 5_000_000_000 + i * dt
                for cam, topic in enumerate(TOPICS["cameras"]):
                    m = video_cls(frame_id=f"cam{cam}", data=units[i], format="h265")
                    m.timestamp.seconds, m.timestamp.nanos = divmod(t + cam * 8000, 1_000_000_000)
                    writer.write_message(topic, m, log_time=t, publish_time=t)
                imu = imu_cls()
                imu.first_sample_time.seconds, imu.first_sample_time.nanos = divmod(t, 1_000_000_000)
                imu.samples.add(t_offset_ns=0).angular_velocity.x = 0.1
                writer.write_message(TOPICS["imu_raw"], imu, log_time=t, publish_time=t)
        paths[seg] = path
    index = tmp / "index"
    last = {"cam0": None, "cam1": None, "imu": None}
    for seg, path in paths.items():
        with path.open("rb") as handle:
            s = extract_segment(handle, seg, path.name, index, TOPICS, video_format="h265", last_times=last)
        last = last_times_after(s, last)
    assemble_session(index, [0, 1])
    tables = {cam: np.load(index / f"{cam}_all.npy") for cam in ("cam0", "cam1")}
    opened = []

    @contextmanager
    def open_segment(number):
        opened.append(number)
        with paths[number].open("rb") as handle:
            yield handle

    return {"tmp": tmp, "raw": raw, "tables": tables, "open": open_segment, "opened": opened,
            "imu_t": np.load(index / "imu_t.npy"), "imu": np.load(index / "imu.npy")}


def test_decode_both_eyes_streamed(recording, tmp_path):
    settings = DecodeSettings(fps=30, sample_step=60, sample_max_width=160, jpeg_quality=85, black_min_seconds=5,
                              black_pixel_threshold=0.1, black_picture_threshold=0.98, freeze_noise_db=-60,
                              freeze_min_seconds=5, timeout_seconds=120)
    out = decode_recording(FFMPEG, partial(stream_packets, recording["open"], TOPICS["cameras"]), recording["tables"],
                           settings, sample_dirs={c: tmp_path / "s" / c for c in ("cam0", "cam1")},
                           calib_dirs={c: tmp_path / "c" / c for c in ("cam0", "cam1")},
                           calib_indices=[0, 120], log_dir=tmp_path / "log")
    for cam in ("cam0", "cam1"):
        r = out[cam]
        assert r["returncode"] == 0 and r["decode_error_count"] == 0
        assert r["decoded_frames"] == N and r["samples_written"] == 3 and r["calibration_frames_written"] == 2
    assert not list(tmp_path.rglob("*.h265"))


def _frame_hashes(path_args):
    out = subprocess.run([FFMPEG, "-v", "error", *path_args, "-f", "framemd5", "-"], capture_output=True, text=True,
                         check=True).stdout
    return [line.rsplit(",", 1)[-1].strip() for line in out.splitlines() if line and not line.startswith("#")]


def test_clip_is_identical_to_the_recording(recording, tmp_path):
    finalized = []
    recording["opened"].clear()
    info = write_clip(clip_dir=tmp_path / "clip_000", clip_id="clip_000", ffmpeg=FFMPEG, ffprobe=FFPROBE,
                      stream=partial(stream_packets, recording["open"], TOPICS["cameras"]), tables=recording["tables"],
                      left_cam="cam1", right_cam="cam0", start=30, stop=150, fps=30, imu_t=recording["imu_t"],
                      imu=recording["imu"], first_frame_ns=int(recording["tables"]["cam1"]["t_ns"][0]), metadata={},
                      timeout_seconds=120, finalize=lambda p: finalized.append((p.name, hashlib.sha256(p.read_bytes()).hexdigest(),
                                                                               _frame_hashes(["-i", str(p)]))))
    assert [f[0] for f in finalized] == ["left.mp4", "right.mp4"]
    assert info["frames"] == 120 and info["verification"]["left"]["first_frame_keyframe"]
    # decoded pictures are exactly frames 30..149 of the source
    source = _frame_hashes(["-f", "hevc", "-framerate", "30", "-i", str(recording["raw"])])
    assert finalized[0][2] == source[30:150]
    assert set(recording["opened"]) == {0, 1}       # only the segments holding the clip were read
    rows = (tmp_path / "clip_000" / "frame_times.csv").read_text().splitlines()
    assert len(rows) == 121 and rows[1].startswith("0,30,")
