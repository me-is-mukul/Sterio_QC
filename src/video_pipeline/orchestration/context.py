"""Run-scoped state shared between stages."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from video_pipeline.core.config import ResolvedConfig
from video_pipeline.core.io import utc_now
from video_pipeline.core.workspace import Workspace
from video_pipeline.ingestion.downloader import DownloadResult
from video_pipeline.reporting.manifest import RunManifest

ORIGIN_URL = "url"
ORIGIN_LOCAL = "local"


def generate_run_id(now: datetime | None = None) -> str:
    """Sortable, unique run ID, e.g. ``20261111T153045Z-1a2b3c4d``."""
    moment = now or utc_now()
    return f"{moment:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class SourceVideo:
    """The immutable input video and its identity."""

    path: Path
    origin: str  # ORIGIN_URL | ORIGIN_LOCAL
    url: str | None
    sha256: str
    size_bytes: int
    mtime_ns: int
    download: DownloadResult | None = None

    def to_dict(self, project_root: Path) -> dict[str, Any]:
        try:
            display_path = self.path.resolve().relative_to(project_root).as_posix()
        except ValueError:
            display_path = str(self.path.resolve())
        return {
            "filename": self.path.name,
            "path": display_path,
            "absolute_path": str(self.path.resolve()),
            "origin": self.origin,
            "url": self.url,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
        }


@dataclass
class RunContext:
    run_id: str
    config: ResolvedConfig
    project_root: Path
    workspace: Workspace
    manifest: RunManifest
    source: SourceVideo
    logger: logging.Logger
    resumed: bool = False
    # Process environment + .env. May contain secrets: never persist or log it.
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)
    # In-memory hand-off between stages within one invocation (never persisted).
    state: dict[str, Any] = field(default_factory=dict)

    def save_manifest(self) -> None:
        self.manifest.save(self.workspace.manifest_path)
