"""Google Cloud Storage access for MCAP sessions: list, stream (segments), download (small files), upload.

A *session* is one folder ``<raw_root><path>/session_*/`` holding ``ego_NNNN.mcap`` segments and
``session.json``. Results go to ``<processed_root><path>/session_*/`` (the same relative path).
Nothing is ever written below ``raw_root``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
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
        if not parts or not parts[-1].startswith("session_") or any(p in {".", ".."} for p in parts):
            raise IngestionError(f"'{url}' does not name a session folder (expected .../session_<...>/).")
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
class SessionListing:
    session: SessionRef
    session_json: ObjectInfo | None
    segments: tuple[tuple[int, ObjectInfo], ...]   # (segment number, object), sorted by number
    other: tuple[ObjectInfo, ...]

    @property
    def total_bytes(self) -> int:
        return sum(o.size for _, o in self.segments) + (self.session_json.size if self.session_json else 0)

    def identity(self) -> str:
        """Fingerprint of exactly which object generations make up this session."""
        objects = [o.to_dict() for _, o in self.segments]
        if self.session_json:
            objects.append(self.session_json.to_dict())
        return sha256_json(sorted(objects, key=lambda o: o["name"]))

    def to_dict(self) -> dict:
        return {
            "session_json": self.session_json.to_dict() if self.session_json else None,
            "segments": [{"number": n, **o.to_dict()} for n, o in self.segments],
            "other_objects": [o.to_dict() for o in self.other],
            "total_bytes": self.total_bytes,
            "identity": self.identity(),
        }


def classify_session_objects(session: SessionRef, objects: Iterable[ObjectInfo], *, session_json: str,
                             segment_pattern: str) -> SessionListing:
    pattern = re.compile(segment_pattern)
    seg: list[tuple[int, ObjectInfo]] = []
    sjson = None
    other = []
    for obj in objects:
        rest = obj.name[len(session.prefix):]
        if "/" in rest:                      # nested objects are not part of the recording
            other.append(obj)
            continue
        match = pattern.fullmatch(rest)
        if match:
            seg.append((int(match.group(1)), obj))
        elif rest == session_json:
            sjson = obj
        else:
            other.append(obj)
    seg.sort(key=lambda item: item[0])          # numeric order, not string order
    numbers = [n for n, _ in seg]
    if len(numbers) != len(set(numbers)):
        raise IngestionError(f"{session.url}: duplicate segment numbers {numbers}")
    return SessionListing(session=session, session_json=sjson, segments=tuple(seg), other=tuple(other))


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

    def list_session(self, session: SessionRef, *, session_json: str, segment_pattern: str) -> SessionListing:
        objects = self.list_objects(session.bucket, session.prefix)
        return classify_session_objects(session, objects, session_json=session_json, segment_pattern=segment_pattern)

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
        """Upload a file. Returns 'uploaded', or 'identical' when (create-only mode) the same bytes are
        already there, e.g. from an earlier attempt that was interrupted."""
        from google.api_core import exceptions as gexc  # noqa: PLC0415

        if not name or name.startswith("/"):
            raise IngestionError(f"Invalid object name '{name}'")
        blob = self._client.bucket(bucket).blob(name)
        try:
            # if_generation_match=0: create only, never replace an existing object.
            blob.upload_from_filename(str(local), content_type=content_type,
                                      if_generation_match=None if overwrite else 0)
            return "uploaded"
        except gexc.PreconditionFailed as exc:
            existing = self._client.bucket(bucket).get_blob(name)
            if existing is not None and existing.size == local.stat().st_size and existing.crc32c == crc32c_base64(local):
                return "identical"
            raise IngestionError(f"gs://{bucket}/{name} already exists with different content; "
                                 "upload.on_existing / --force decide whether to replace it.") from exc
        except Exception as exc:  # noqa: BLE001
            hint = ""
            if "403" in str(exc):
                hint = " (permission denied: the service account needs storage.objects.create, plus delete to overwrite)"
            raise IngestionError(f"Upload to gs://{bucket}/{name} failed: {exc}{hint}") from exc


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
