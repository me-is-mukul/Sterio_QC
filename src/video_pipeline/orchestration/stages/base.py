"""Stage interface and helpers shared by stage adapters.

Stages are thin adapters: they read configuration from the RunContext, load their
inputs from workspace artifacts written by earlier stages, call domain modules, and
persist outputs. Domain modules never import from orchestration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from video_pipeline.core.errors import StageInputError
from video_pipeline.core.io import read_json
from video_pipeline.orchestration.context import RunContext


@dataclass(frozen=True)
class StageOutcome:
    outputs: tuple[Path, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)


class Stage(ABC):
    """A pipeline stage with a single responsibility.

    Contract:
    * ``run`` is idempotent: re-running it overwrites its own outputs only.
    * Inputs come from workspace files (never only from memory), so a stage can run
      after a resume. ``RunContext.state`` is an optional in-process shortcut.
    * Every file/directory it produces is listed in ``StageOutcome.outputs``.
    * ``config_keys`` lists the configuration keys its outputs depend on; when any of
      them changes, a resumed run re-executes the stage (and everything after it).
    * It never modifies the source video.
    """

    name: ClassVar[str]
    config_keys: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def run(self, ctx: RunContext) -> StageOutcome: ...


def require(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise StageInputError(f"Required input '{path.name}' is missing; it is produced by the '{produced_by}' stage.")
    return path


def load_video_info(ctx: RunContext) -> dict[str, Any]:
    return read_json(require(ctx.workspace.metadata_dir / "video_info.json", "metadata"))


def primary_video_stream(info: Mapping[str, Any]) -> dict[str, Any]:
    index = info.get("primary_video_stream_index")
    for stream in info.get("video_streams", []):
        if stream.get("index") == index:
            return stream
    raise StageInputError("video_info.json has no primary video stream.")


def video_start_time(info: Mapping[str, Any]) -> float:
    """Timestamp of the first video frame on the container timeline (usually 0)."""
    stream = primary_video_stream(info)
    for value in (stream.get("start_time_seconds"), info.get("format", {}).get("start_time_seconds")):
        if value is not None:
            return float(value)
    return 0.0


def video_duration(info: Mapping[str, Any]) -> float:
    duration = info.get("duration_seconds")
    if not duration:
        raise StageInputError("video_info.json has no duration.")
    return float(duration)
