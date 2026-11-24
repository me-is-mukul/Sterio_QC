"""Recording formats: detect a session folder's format and read it (see detect.py for the list)."""

from __future__ import annotations

from pathlib import Path

from video_pipeline.core.errors import IngestionError
from video_pipeline.ingestion.formats.base import RecordingReader
from video_pipeline.ingestion.formats.detect import FORMATS, classify, detect_format
from video_pipeline.ingestion.formats.egocam import EgocReader, EgoCamLiveReader
from video_pipeline.ingestion.formats.hydra import HydraReader
from video_pipeline.ingestion.formats.mcap_format import McapReader
from video_pipeline.ingestion.formats.robocap import RobocapReader

READERS: dict[str, type[RecordingReader]] = {
    "mcap": McapReader,
    "robocap": RobocapReader,
    "hydra": HydraReader,
    "egocam_live": EgoCamLiveReader,
    "egocam_egoc": EgocReader,
}
# formats judged one segment at a time (each segment is its own recording, <folder>/#seg<N>)
PER_SEGMENT_FORMATS = frozenset(name for name, reader in READERS.items() if reader.per_segment)


def make_reader(listing, storage, cfg, *, work_dir: Path, keep_dir: Path, max_segments: int | None = None
                ) -> RecordingReader:
    if listing.format not in READERS:
        raise IngestionError(f"{listing.session.url}: not a recording in a known format "
                             f"(files: {[o.name.rsplit('/', 1)[-1] for o in listing.other][:10]})")
    return READERS[listing.format](listing, storage, cfg, work_dir=work_dir, keep_dir=keep_dir,
                                   max_segments=max_segments)


__all__ = ["FORMATS", "PER_SEGMENT_FORMATS", "READERS", "RecordingReader", "classify", "detect_format", "make_reader"]
