"""Output directory management: ``<output_root>/<video_name>/`` and its fixed layout."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from video_pipeline.core.errors import WorkspaceError

WORKSPACE_SUBDIRECTORIES: tuple[str, ...] = (
    "metadata",
    "frames",
    "imu",
    "inference",
    "segmentation",
    "processed",
    "reports",
)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def derive_video_name(video_path: Path) -> str:
    """Directory-safe video name derived from the file stem (e.g. GX010140.MP4 -> GX010140)."""
    name = _UNSAFE.sub("_", video_path.stem).strip("._")
    if not name:
        raise WorkspaceError(f"Cannot derive an output directory name from '{video_path.name}'.")
    return name


@dataclass(frozen=True)
class Workspace:
    root: Path
    video_name: str

    @classmethod
    def for_video(cls, output_root: Path, video_name: str) -> Workspace:
        return cls(root=(output_root / video_name).resolve(), video_name=video_name)

    def create(self) -> None:
        try:
            for name in WORKSPACE_SUBDIRECTORIES:
                (self.root / name).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise WorkspaceError(f"Cannot create output workspace '{self.root}': {exc}") from exc

    def ensure_separate_from(self, source_path: Path) -> None:
        """Generated artifacts must never live alongside (or contain) the source video."""
        source = source_path.resolve()
        if self.root in source.parents:
            raise WorkspaceError(f"Source video '{source}' is inside the output workspace '{self.root}'.")
        if self.root == source.parent.resolve():
            raise WorkspaceError("Output workspace must not be the source directory.")

    # Fixed sub-directories --------------------------------------------------
    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def imu_dir(self) -> Path:
        return self.root / "imu"

    @property
    def inference_dir(self) -> Path:
        return self.root / "inference"

    @property
    def segmentation_dir(self) -> Path:
        return self.root / "segmentation"

    @property
    def processed_dir(self) -> Path:
        return self.root / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    # Well-known files ---------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.reports_dir / "manifest.json"

    @property
    def log_path(self) -> Path:
        return self.reports_dir / "processing.log"

    @property
    def resolved_config_path(self) -> Path:
        return self.reports_dir / "resolved_config.yaml"

    @property
    def lock_path(self) -> Path:
        return self.root / ".run.lock"

    @property
    def history_dir(self) -> Path:
        return self.reports_dir / "history"

    def relative(self, path: Path) -> str:
        """Path relative to the workspace root, in POSIX form, for portable manifests."""
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path.resolve())

    def archive_previous_run(self, previous_run_id: str) -> Path:
        """Move the previous run's manifest, log and resolved config into reports/history/<run_id>/."""
        target = self.history_dir / previous_run_id
        if target.exists():
            raise WorkspaceError(f"History directory already exists: '{target}'.")
        target.mkdir(parents=True)
        for path in (self.manifest_path, self.log_path, self.resolved_config_path):
            if path.exists():
                shutil.move(str(path), str(target / path.name))
        return target
