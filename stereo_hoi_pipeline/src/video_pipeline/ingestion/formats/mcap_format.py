"""MCAP sessions (ego_NNNN.mcap + session.json): the original format, read by ingestion/mcap_reader."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from video_pipeline.ingestion.formats.base import PacketRequests, PacketSink, RecordingReader, check, contiguous_checks
from video_pipeline.ingestion.mcap_reader import extract_segment, last_times_after, load_segment_summary, stream_packets

logger = logging.getLogger("video_pipeline.ego")


class McapReader(RecordingReader):
    format = "mcap"
    layout = "pair"
    description = "MCAP (both eyes and IMU in ego_NNNN.mcap)"
    metadata_files = ("session.json",)

    def required_files(self) -> list[dict[str, Any]]:
        sjson = self.listing.session_json
        checks = [check("session_json_present", sjson is not None,
                        "session.json present" if sjson else "session.json missing")]
        return checks + contiguous_checks([s.number for s in self.segments()], 0, "ego_0000")

    def _opener(self):
        objects = {s.number: s.files["mcap"] for s in self.segments()}

        @contextmanager
        def open_segment(number: int):
            reader = self.open(objects[number])
            try:
                yield reader
            finally:
                reader.close()

        return open_segment

    def extract(self, stream_dir: Path) -> list[int]:
        topics = self.cfg.get("topics")
        last: dict[str, int | None] = {"cam0": None, "cam1": None, "imu": None}
        numbers = []
        segments = self.segments()
        opener = self._opener()
        for position, seg in enumerate(segments, start=1):
            numbers.append(seg.number)
            obj = seg.files["mcap"]
            name = obj.name.rsplit("/", 1)[-1]
            summary = load_segment_summary(stream_dir, seg.number)
            if summary is None or summary.get("file") != name:
                t0 = time.perf_counter()
                with opener(seg.number) as stream:
                    summary = extract_segment(stream, seg.number, name, stream_dir, topics,
                                              video_format=self.cfg.get("topics.video_format"), last_times=last)
                logger.info("Segment %d/%d (%s, %.0f MB streamed): %d+%d frames, %d IMU samples | %.1f s",
                            position, len(segments), name, obj.size / 1e6, summary["cam0"]["frames"],
                            summary["cam1"]["frames"], summary["imu"]["samples"], time.perf_counter() - t0)
            last = last_times_after(summary, last)
        return numbers

    def stream_packets_from(self, stream_dir: Path):
        opener = self._opener()
        cameras = self.cfg.get("topics.cameras")

        def stream(requests: PacketRequests, sink: PacketSink) -> dict[str, int]:
            return stream_packets(opener, cameras, requests, sink)

        return stream

    def imu_nominal_hz(self, calibration_messages):
        return (calibration_messages.get("imu_info") or {}).get("update_rate_hz")
