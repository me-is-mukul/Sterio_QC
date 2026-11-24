"""A local folder that behaves like the parts of GcsStorage the pipeline reads (for tests)."""

from __future__ import annotations

import shutil
from pathlib import Path

from video_pipeline.ingestion.gcs import ObjectInfo


class LocalStorage:
    """Objects are the files below `root`; the object name is the path relative to root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.opened: list[str] = []

    def list_objects(self, bucket: str, prefix: str) -> list[ObjectInfo]:
        out = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                name = path.relative_to(self.root).as_posix()
                if name.startswith(prefix):
                    out.append(ObjectInfo(name, path.stat().st_size, 1, None, None))
        return out

    def open_reader(self, bucket: str, obj: ObjectInfo, *, chunk_bytes: int):
        self.opened.append(obj.name)
        return (self.root / obj.name).open("rb")

    def download(self, bucket: str, obj: ObjectInfo, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / obj.name, target)
        return target

    def exists(self, bucket: str, name: str) -> bool:
        return (self.root / name).is_file()

    def read_text(self, bucket: str, name: str) -> str:
        return (self.root / name).read_text(encoding="utf-8")
