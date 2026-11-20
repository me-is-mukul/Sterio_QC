"""The resolved clip list (outputs/<name>/clip_manifest.csv), built by sources.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .util import read_csv

LAYOUT_VIEWS = {"mono": ["main"], "sbs": ["left_eye", "right_eye"],
                "grid3x2": ["tile1", "tile2", "tile3", "tile4", "tile5", "tile6"]}
LAYOUT_GRID = {"mono": (1, 1), "sbs": (2, 1), "grid3x2": (3, 2)}      # (columns, rows)


@dataclass(frozen=True)
class Clip:
    clip_no: int
    clip_id: str
    dataset: str
    category: str
    task: str
    camera: str
    layout: str
    status: str
    source_url: str
    bucket: str
    object: str
    size_bytes: int | None
    md5_base64: str
    resolution: str
    original_reference: str
    notes: str

    @property
    def available(self) -> bool:
        return self.status == "OK"

    @property
    def views(self) -> list[str]:
        return LAYOUT_VIEWS[self.layout]

    def cache_path(self, cache_dir: Path) -> Path:
        return cache_dir.joinpath(self.bucket, *self.object.split("/"))

    @property
    def file_name(self) -> str:
        return self.object.rsplit("/", 1)[-1] if self.object else ""


def load_manifest(path: Path) -> list[Clip]:
    clips = []
    for r in read_csv(path):
        layout = r["layout"].strip()
        if layout not in LAYOUT_VIEWS:
            raise ValueError(f"{path}: clip {r['clip_id']} has unknown layout '{layout}'")
        clips.append(Clip(
            clip_no=int(r["clip_no"]), clip_id=r["clip_id"], dataset=r["dataset"], category=r["category"],
            task=r["task"], camera=r["camera"], layout=layout, status=r["status"], source_url=r["source_url"],
            bucket=r["bucket"], object=r["object"], size_bytes=int(r["size_bytes"]) if r["size_bytes"] else None,
            md5_base64=r["md5_base64"], resolution=r["resolution"], original_reference=r["original_reference"],
            notes=r["notes"]))
    ids = [c.clip_id for c in clips]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate clip_id values")
    return clips
