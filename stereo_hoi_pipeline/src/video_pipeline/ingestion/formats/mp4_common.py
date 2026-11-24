"""Shared by the MP4-based formats (robocap, EGO-CAM live): index a segment's MP4 from its moov box
and stream its samples again as Annex-B HEVC packets."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from video_pipeline.core.errors import IngestionError
from video_pipeline.ingestion.gcs import ObjectInfo
from video_pipeline.ingestion.mcap_reader import segment_dir
from video_pipeline.ingestion.mp4_index import Mp4Track, read_mp4_index, to_annexb

HEVC_SAMPLE_ENTRIES = {"hvc1", "hev1"}


def index_mp4(reader, obj: ObjectInfo) -> Mp4Track:
    with reader.open(obj) as stream:
        return read_mp4_index(stream, obj.size)


def mp4_info(track: Mp4Track, obj: ObjectInfo) -> dict[str, Any]:
    """What streaming needs later, stored in segment.json (offsets go into an .npy next to it)."""
    return {"object": obj.name, "generation": obj.generation, "codec": track.codec, "width": track.width,
            "height": track.height, "length_size": track.length_size, "parameter_sets": track.parameter_sets.hex(),
            "timescale": track.timescale, "samples": track.samples}


def stream_mp4_segment(reader, stream_dir: Path, segment: int, rows: Mapping[str, Sequence[tuple[int, int]]],
                       sources: Mapping[str, str]) -> Iterator[tuple[str, int, bytes]]:
    """Yield (cam, row, Annex-B packet) for the requested rows. ``sources``: cam -> stream key whose
    offsets (``<key>_offsets.npy``) and MP4 info (``segment.json`` 'mp4'[key]) describe the file.
    Cameras are interleaved by ordinal so that two decoders are fed at the same time."""
    seg_dir = segment_dir(stream_dir, segment)
    import json  # noqa: PLC0415

    summary = json.loads((seg_dir / "segment.json").read_text(encoding="utf-8"))
    handles: dict[str, Any] = {}
    state: dict[str, dict[str, Any]] = {}
    try:
        for cam in rows:
            key = sources[cam]
            info = summary["mp4"][key]
            obj = ObjectInfo(info["object"], 0, int(info["generation"]), None, None)
            if key not in handles:
                handles[key] = reader.open(obj)
            table = np.load(seg_dir / f"{cam}.npy")
            state[cam] = {"stream": handles[key], "offsets": np.load(seg_dir / f"{key}_offsets.npy"),
                          "sizes": table["size"], "keys": table["key"], "length_size": int(info["length_size"]),
                          "parameter_sets": bytes.fromhex(info["parameter_sets"]), "position": None}
        order = sorted(((ordinal, cam, row) for cam, items in rows.items() for row, ordinal in items),
                       key=lambda x: (x[0], x[1]))
        for ordinal, cam, row in order:
            s = state[cam]
            offset, size = int(s["offsets"][ordinal]), int(s["sizes"][ordinal])
            if s["position"] != offset:
                s["stream"].seek(offset)
            data = s["stream"].read(size)
            if len(data) != size:
                raise IngestionError(f"segment {segment} {cam}: sample {ordinal} is {len(data)} bytes, index says {size}")
            s["position"] = offset + size
            yield cam, row, to_annexb(data, s["length_size"], s["parameter_sets"] if s["keys"][ordinal] else b"")
    finally:
        for handle in handles.values():
            handle.close()
