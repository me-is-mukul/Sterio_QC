"""MCAP extraction on a synthetic recording written with the same schemas as the device."""

import numpy as np
import pytest

from contextlib import contextmanager

from video_pipeline.core.errors import IngestionError
from video_pipeline.ingestion.mcap_reader import (
    assemble_session,
    extract_segment,
    frame_flags,
    last_times_after,
    nal_units,
    parameter_set_prefix,
    stream_packets,
)

VPS = b"\x00\x00\x00\x01\x40\x01" + b"v" * 10
SPS = b"\x00\x00\x00\x01\x42\x01" + b"s" * 10
PPS = b"\x00\x00\x00\x01\x44\x01" + b"p" * 4
IDR = b"\x00\x00\x00\x01\x26\x01" + b"I" * 20          # type 19 (IDR_W_RADL)
TRAIL = b"\x00\x00\x00\x01\x02\x01" + b"P" * 8         # type 1

TOPICS = {"cameras": ["/ego/camera/0", "/ego/camera/1"], "imu_raw": "/ego/imu/0/raw",
          "intrinsics": ["/ego/camera/0/intrinsics", "/ego/camera/1/intrinsics"],
          "stereo_extrinsics": "/ego/camera/1/extrinsics", "imu_info": "/ego/imu/0/info",
          "imu_extrinsics": "/ego/imu/0/extrinsics"}


def test_nal_parsing():
    key = VPS + SPS + PPS + IDR
    assert [t for _, t in nal_units(key)] == [32, 33, 34, 19]
    assert frame_flags(key) == (True, True)
    assert frame_flags(TRAIL) == (False, False)
    assert frame_flags(IDR) == (True, False)
    assert parameter_set_prefix(key) == VPS + SPS + PPS


def _schemas():
    """Dynamic protobuf classes with the device's message layouts (foxglove + visio ImuRaw)."""
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, timestamp_pb2

    pool = descriptor_pool.DescriptorPool()
    ts_fd = descriptor_pb2.FileDescriptorProto()
    timestamp_pb2.DESCRIPTOR.CopyToProto(ts_fd)
    pool.Add(ts_fd)
    fd = descriptor_pb2.FileDescriptorProto(name="t.proto", package="foxglove", syntax="proto3",
                                            dependency=["google/protobuf/timestamp.proto"])
    F = descriptor_pb2.FieldDescriptorProto

    def msg(name, fields):
        m = fd.message_type.add(name=name)
        for number, (fname, ftype, label, tname) in enumerate(fields, start=1):
            f = m.field.add(name=fname, number=number, type=ftype, label=label)
            if tname:
                f.type_name = tname
        return m

    opt, rep = F.LABEL_OPTIONAL, F.LABEL_REPEATED
    msg("CompressedVideo", [("timestamp", F.TYPE_MESSAGE, opt, ".google.protobuf.Timestamp"),
                            ("frame_id", F.TYPE_STRING, opt, None), ("data", F.TYPE_BYTES, opt, None),
                            ("format", F.TYPE_STRING, opt, None)])
    msg("Vector3", [("x", F.TYPE_DOUBLE, opt, None), ("y", F.TYPE_DOUBLE, opt, None), ("z", F.TYPE_DOUBLE, opt, None)])
    sample = msg("ImuRawSample", [("t_offset_ns", F.TYPE_UINT64, opt, None),
                                  ("angular_velocity", F.TYPE_MESSAGE, opt, ".foxglove.Vector3"),
                                  ("linear_acceleration", F.TYPE_MESSAGE, opt, ".foxglove.Vector3"),
                                  ("magnetic_field", F.TYPE_MESSAGE, opt, ".foxglove.Vector3"),
                                  ("temperature_c", F.TYPE_FLOAT, opt, None)])
    sample.field[4].proto3_optional = True
    sample.oneof_decl.add(name="_temperature_c")
    sample.field[4].oneof_index = 0
    msg("ImuRaw", [("first_sample_time", F.TYPE_MESSAGE, opt, ".google.protobuf.Timestamp"),
                   ("samples", F.TYPE_MESSAGE, rep, ".foxglove.ImuRawSample")])
    pool.Add(fd)
    get = lambda n: message_factory.GetMessageClass(pool.FindMessageTypeByName(n))  # noqa: E731
    return get("foxglove.CompressedVideo"), get("foxglove.ImuRaw")


def _write_segment(path, start_frame, n_frames, *, skip_cam1=False, extra_overlap=0):
    pytest.importorskip("mcap_protobuf")
    from mcap_protobuf.writer import Writer

    video_cls, imu_cls = _schemas()
    dt = 33_333_333
    with path.open("wb") as handle, Writer(handle) as writer:
        for i in range(start_frame - extra_overlap, start_frame + n_frames):
            t = 10_000_000_000 + i * dt
            data = (VPS + SPS + PPS + IDR) if i % 30 == 0 else TRAIL
            for cam, topic in enumerate(TOPICS["cameras"]):
                if cam == 1 and skip_cam1:
                    continue
                m = video_cls(frame_id=f"cam{cam}", data=data + bytes([i % 256]), format="h265")
                m.timestamp.seconds, m.timestamp.nanos = divmod(t, 1_000_000_000)
                writer.write_message(topic, m, log_time=t, publish_time=t)
            imu = imu_cls()
            imu.first_sample_time.seconds, imu.first_sample_time.nanos = divmod(t, 1_000_000_000)
            for k in range(3):
                s = imu.samples.add(t_offset_ns=k * 11_000_000)
                s.angular_velocity.x = 0.1 * k
                if k == 0:
                    s.temperature_c = 40.0
            writer.write_message(TOPICS["imu_raw"], imu, log_time=t, publish_time=t)


def _opener(paths):
    @contextmanager
    def open_segment(number):
        with paths[number].open("rb") as handle:
            yield handle
    return open_segment


def _index(path, number, stream_dir, last):
    with path.open("rb") as handle:
        return extract_segment(handle, number, path.name, stream_dir, TOPICS, video_format="h265", last_times=last)


def test_extract_two_segments_with_overlap(tmp_path):
    seg0, seg1 = tmp_path / "ego_0000.mcap", tmp_path / "ego_0001.mcap"
    _write_segment(seg0, 0, 60)
    _write_segment(seg1, 60, 60, extra_overlap=2)        # repeats the last 2 frames of segment 0
    stream = tmp_path / "index"
    s0 = _index(seg0, 0, stream, {})
    last = last_times_after(s0, {"cam0": None, "cam1": None, "imu": None})
    s1 = _index(seg1, 1, stream, last)
    assert s0["cam0"]["frames"] == 60 and s0["cam0"]["keyframes"] == 2 and s0["missing_topics"] == []
    assert s1["cam0"]["duplicates_dropped"] == 2 and s1["cam0"]["frames"] == 60 and s1["cam0"]["messages"] == 62
    assert s1["imu"]["duplicates_dropped"] > 0
    assert not list(stream.rglob("*.h265")) and not list(stream.rglob("*.mcap"))      # no video on disk
    counts = assemble_session(stream, [0, 1])
    assert counts["cam0"] == 120 and counts["cam1"] == 120
    cam0 = np.load(stream / "cam0_all.npy")
    assert np.all(np.diff(cam0["t_ns"]) > 0)
    assert list(np.flatnonzero(cam0["key"])) == [0, 30, 60, 90]
    # segment 1 rows start at ordinal 2: its first two messages were the dropped duplicates
    assert cam0["ordinal"][60] == 2 and cam0["segment"][60] == 1

    got = []
    delivered = stream_packets(_opener({0: seg0, 1: seg1}), TOPICS["cameras"], {"cam0": (cam0, 59, 62)},
                               lambda cam, row, data: got.append((cam, row, data)))
    assert delivered == {"cam0": 3}
    assert [r for _, r, _ in got] == [59, 60, 61]
    assert b"".join(d for _, _, d in got) == TRAIL + bytes([59]) + (VPS + SPS + PPS + IDR) + bytes([60]) + TRAIL + bytes([61])

    bad = cam0.copy()
    bad["size"][60] += 1
    with pytest.raises(IngestionError, match="does not match the index"):
        stream_packets(_opener({0: seg0, 1: seg1}), TOPICS["cameras"], {"cam0": (bad, 59, 62)}, lambda *a: None)

    imu_t = np.load(stream / "imu_t.npy")
    imu = np.load(stream / "imu.npy")
    assert np.all(np.diff(imu_t) > 0) and imu.shape[1] == 10
    assert imu[0, 9] == 40.0 and np.isnan(imu[1, 9])


def test_missing_camera_topic_is_reported(tmp_path):
    seg = tmp_path / "ego_0000.mcap"
    _write_segment(seg, 0, 30, skip_cam1=True)
    s = _index(seg, 0, tmp_path / "index", {})
    assert s["missing_topics"] == ["/ego/camera/1"]
