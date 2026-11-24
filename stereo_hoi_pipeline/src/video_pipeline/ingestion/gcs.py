"""Google Cloud Storage access for recordings: list, stream (segments), download (small files), upload.

A *session* is one folder ``<raw_root><path>/`` whose files form one recording in one of the
supported formats (see ingestion/formats). Results go to ``<processed_root><path>/`` (the same
relative path). Nothing is ever written below ``raw_root``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from video_pipeline.core.errors import IngestionError
from video_pipeline.core.hashing import sha256_json

logger = logging.getLogger(__name__)


def parse_gs_url(url: str) -> tuple[str, str]:
    """'gs://bucket/a/b/' -> ('bucket', 'a/b/'). The prefix keeps a trailing slash if given."""
    if not url.startswith("gs://"):
        raise IngestionError(f"Not a gs:// URL: '{url}'")
    bucket, _, prefix = url[5:].partition("/")
    if not bucket:
        raise IngestionError(f"No bucket in '{url}'")
    return bucket, prefix


@dataclass(frozen=True)
class ObjectInfo:
    name: str
    size: int
    generation: int
    crc32c: str | None
    md5: str | None

    def to_dict(self) -> dict:
        return {"name": self.name, "size": self.size, "generation": self.generation, "crc32c": self.crc32c,
                "md5": self.md5}


@dataclass(frozen=True)
class SessionRef:
    """One recording. ``relative`` is its path below raw_root, e.g. 'Op_DD/01-11-2026/SDC001/session_00002-1793528549'."""

    bucket: str
    raw_root: str
    relative: str

    @classmethod
    def from_url(cls, url: str, *, expected_bucket: str, raw_root: str) -> SessionRef:
        bucket, prefix = parse_gs_url(url.strip())
        if bucket != expected_bucket:
            raise IngestionError(f"'{url}' is not in bucket '{expected_bucket}' (source.bucket).")
        if not prefix.startswith(raw_root):
            raise IngestionError(f"'{url}' is not below gs://{bucket}/{raw_root} (source.raw_root).")
        relative = prefix[len(raw_root):].strip("/")
        parts = PurePosixPath(relative).parts
        if not parts or any(p in {".", ".."} for p in parts):
            raise IngestionError(f"'{url}' does not name a session folder below gs://{bucket}/{raw_root}.")
        return cls(bucket=bucket, raw_root=raw_root, relative="/".join(parts))

    @property
    def prefix(self) -> str:
        return f"{self.raw_root}{self.relative}/"

    @property
    def url(self) -> str:
        return f"gs://{self.bucket}/{self.prefix}"

    @property
    def name(self) -> str:
        return PurePosixPath(self.relative).name

    def describe(self) -> dict:
        parts = PurePosixPath(self.relative).parts
        return {
            "url": self.url,
            "relative_path": self.relative,
            "session": parts[-1],
            "device": parts[-2] if len(parts) >= 2 else None,
            "date": parts[-3] if len(parts) >= 3 else None,
        }


@dataclass(frozen=True)
class Segment:
    """One numbered piece of a recording: its file(s) by role, e.g. {'mcap': ...} or
    {'video_left_front.mp4': ..., 'video_right_front.mp4': ..., 'imu_left.db': ...}."""

    number: int
    files: Mapping[str, ObjectInfo]

    @property
    def size(self) -> int:
        return sum(o.size for o in self.files.values())

    @property
    def name(self) -> str:
        """A readable name for logs: the file name, or the common prefix of a multi-file segment."""
        names = sorted(o.name.rsplit("/", 1)[-1] for o in self.files.values())
        if len(names) == 1:
            return names[0]
        return f"segment {self.number} ({len(names)} files)"


@dataclass(frozen=True)
class SessionListing:
    session: SessionRef
    format: str | None                              # see ingestion/formats; None = not a known recording
    files: Mapping[str, ObjectInfo]                 # session-level files by name (session.json, frames.idx, ...)
    segments: tuple[Segment, ...]                   # sorted by number
    other: tuple[ObjectInfo, ...] = field(default=())   # everything else (not part of the recording)

    @property
    def session_json(self) -> ObjectInfo | None:
        return self.files.get("session.json")

    def objects(self) -> list[ObjectInfo]:
        return [o for s in self.segments for o in s.files.values()] + list(self.files.values())

    @property
    def total_bytes(self) -> int:
        return sum(o.size for o in self.objects())

    def identity(self) -> str:
        """Fingerprint of exactly which object generations make up this session."""
        return sha256_json(sorted((o.to_dict() for o in self.objects()), key=lambda o: o["name"]))

    def limited(self, max_segments: int | None) -> SessionListing:
        if max_segments is None:
            return self
        return SessionListing(self.session, self.format, self.files, self.segments[:max_segments], self.other)

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "files": {k: o.to_dict() for k, o in self.files.items()},
            "segments": [{"number": s.number, "files": {k: o.to_dict() for k, o in s.files.items()}}
                         for s in self.segments],
            "other_objects": [o.to_dict() for o in self.other],
            "total_bytes": self.total_bytes,
            "identity": self.identity(),
        }


class GcsStorage:
    def __init__(self, credentials, project: str) -> None:
        from google.cloud import storage  # noqa: PLC0415

        self._client = storage.Client(project=project, credentials=credentials)

    def list_objects(self, bucket: str, prefix: str) -> list[ObjectInfo]:
        try:
            blobs = self._client.list_blobs(bucket, prefix=prefix)
            return [ObjectInfo(b.name, int(b.size or 0), int(b.generation), b.crc32c, b.md5_hash) for b in blobs]
        except Exception as exc:  # noqa: BLE001 (google exceptions vary; reported with context)
            raise IngestionError(f"Cannot list gs://{bucket}/{prefix}: {exc}") from exc


    def exists(self, bucket: str, name: str) -> bool:
        try:
            return self._client.bucket(bucket).blob(name).exists()
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Cannot check gs://{bucket}/{name}: {exc}") from exc

    def open_reader(self, bucket: str, obj: ObjectInfo, *, chunk_bytes: int):
        """A seekable, read-only stream of exactly the listed generation (ranged reads, retried).
        Nothing is written to disk. Use as a context manager."""
        from google.cloud.storage.retry import DEFAULT_RETRY  # noqa: PLC0415

        blob = self._client.bucket(bucket).blob(obj.name, generation=obj.generation)
        try:
            return blob.open("rb", chunk_size=chunk_bytes, retry=DEFAULT_RETRY)
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Cannot open gs://{bucket}/{obj.name}: {exc}") from exc

    def download(self, bucket: str, obj: ObjectInfo, target: Path) -> Path:
        """Download exactly the listed generation; checksum-verified; atomic rename."""
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        blob = self._client.bucket(bucket).blob(obj.name, generation=obj.generation)
        try:
            blob.download_to_filename(str(partial), checksum="auto")
        except Exception as exc:  # noqa: BLE001
            partial.unlink(missing_ok=True)
            raise IngestionError(f"Download of gs://{bucket}/{obj.name} failed: {exc}") from exc
        size = partial.stat().st_size
        if size != obj.size:
            partial.unlink(missing_ok=True)
            raise IngestionError(f"gs://{bucket}/{obj.name}: {size} bytes downloaded, {obj.size} expected")
        partial.replace(target)
        return target

    def delete(self, bucket: str, name: str) -> None:
        from google.api_core import exceptions as gexc  # noqa: PLC0415

        try:
            self._client.bucket(bucket).blob(name).delete()
        except gexc.NotFound:
            pass
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Cannot delete gs://{bucket}/{name}: {exc} "
                                 "(replacing a result needs storage.objects.delete)") from exc

    def upload(self, local: Path, bucket: str, name: str, *, overwrite: bool, content_type: str | None = None) -> str:
        """Upload a file. Returns 'uploaded', 'identical' (the same bytes are already there, e.g. from an
        interrupted earlier attempt) or 'replaced'.

        Always tries create-only first: creating needs only storage.objects.create. Replacing an
        existing object needs storage.objects.delete as well, so it is done only when the content
        really differs and ``overwrite`` is True.
        """
        from google.api_core import exceptions as gexc  # noqa: PLC0415

        if not name or name.startswith("/"):
            raise IngestionError(f"Invalid object name '{name}'")
        blob = self._client.bucket(bucket).blob(name)
        try:
            blob.upload_from_filename(str(local), content_type=content_type, if_generation_match=0)
            return "uploaded"
        except gexc.PreconditionFailed as exc:
            existing = self._client.bucket(bucket).get_blob(name)
            if existing is not None and existing.size == local.stat().st_size and existing.crc32c == crc32c_base64(local):
                return "identical"
            if not overwrite:
                raise IngestionError(f"gs://{bucket}/{name} already exists with different content; "
                                     "upload.on_existing / --force decide whether to replace it.") from exc
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Upload to gs://{bucket}/{name} failed: {exc}{_permission_hint(exc)}") from exc
        try:
            blob.upload_from_filename(str(local), content_type=content_type)
            return "replaced"
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Replacing gs://{bucket}/{name} failed: {exc}{_permission_hint(exc)}") from exc


def _permission_hint(exc: Exception) -> str:
    if "403" not in str(exc):
        return ""
    return (" (permission denied: creating a result needs storage.objects.create; replacing or deleting an "
            "existing one (re-runs, --force) also needs storage.objects.delete, e.g. roles/storage.objectAdmin "
            "on processed/)")


def crc32c_base64(path: Path) -> str:
    """CRC32C of a file, base64 big-endian: the form Cloud Storage reports."""
    import base64  # noqa: PLC0415

    import google_crc32c  # noqa: PLC0415

    checksum = google_crc32c.Checksum()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode("ascii")


def guess_content_type(path: Path) -> str | None:
    return {
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".csv": "text/csv",
        ".mp4": "video/mp4",
        ".log": "text/plain",
        ".yaml": "text/yaml",
        ".txt": "text/plain",
    }.get(path.suffix.lower())
