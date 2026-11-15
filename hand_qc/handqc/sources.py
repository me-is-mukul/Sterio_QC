"""Resolve a client clip list (CSV) into a clip manifest with verified download URLs.

Input CSV columns (header row required; extra columns are ignored):
    Dataset       Mono | Stereo | Six-Cam   (camera layout: full frame | side by side | 3x2 grid)
    Category      free text
    Task          free text (may be empty)
    Camera        free text (may be empty)
    URL_or_File   https://storage.googleapis.com/<bucket>/<path>, gs://<bucket>/<path>,
                  a bare filename suffix, or empty
    Notes         free text (may be empty)

Each row is resolved against the public Google Cloud Storage bucket listings:
  * complete URL that exists                  -> used as is
  * URL whose file name was truncated         -> the single .mp4 inside the same folder
  * URL whose date folder is missing          -> the single session folder with that path
  * bare filename suffix (e.g. _GX010171.mp4) -> the single file in 'mono samples trimmed/electronics/'
                                                 of bucket dd_samples named <YYYY-MM-DD><suffix>
  * no usable URL                             -> SOURCE_MISSING
A row is only resolved when exactly one file matches; anything else is SOURCE_MISSING with the reason.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .util import LOG, read_csv, write_csv

GCS_PUBLIC = "https://storage.googleapis.com/"
GCS_LIST = "https://storage.googleapis.com/storage/v1/b/{bucket}/o?{query}"
ELECTRONICS = ("dd_samples", "mono samples trimmed/electronics/")
LAYOUTS = {"mono": "mono", "stereo": "sbs", "six-cam": "grid3x2", "sixcam": "grid3x2", "six cam": "grid3x2"}
REQUIRED_COLUMNS = ["Dataset", "Category", "Task", "Camera", "URL_or_File", "Notes"]
MANIFEST_FIELDS = ["clip_no", "clip_id", "dataset", "category", "task", "camera", "layout", "status", "source_url",
                   "bucket", "object", "size_bytes", "md5_base64", "resolution", "original_reference", "notes"]


class SourceListError(Exception):
    pass


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")


def _list_bucket(bucket: str) -> dict[str, dict]:
    objects: dict[str, dict] = {}
    token = None
    while True:
        params = {"fields": "items(name,size,md5Hash),nextPageToken", "maxResults": "1000"}
        if token:
            params["pageToken"] = token
        url = GCS_LIST.format(bucket=urllib.parse.quote(bucket), query=urllib.parse.urlencode(params))
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (public listing endpoint)
            page = json.load(resp)
        for item in page.get("items", []):
            objects[item["name"]] = {"size": int(item["size"]), "md5": item.get("md5Hash", "")}
        token = page.get("nextPageToken")
        if not token:
            return objects


class _Listings:
    """Bucket listings, fetched once per bucket on first use."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, dict] | None] = {}

    def get(self, bucket: str) -> dict[str, dict] | None:
        if bucket not in self._cache:
            try:
                LOG.info("Listing bucket '%s' ...", bucket)
                self._cache[bucket] = _list_bucket(bucket)
                LOG.info("  %d objects", len(self._cache[bucket]))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                LOG.warning("Could not list bucket '%s': %s", bucket, exc)
                self._cache[bucket] = None
        return self._cache[bucket]


def _split_reference(ref: str) -> tuple[str, str] | None:
    """(bucket, object) from an https or gs:// URL; None if it is not one."""
    if ref.startswith("gs://"):
        bucket, _, name = ref[5:].partition("/")
        return bucket, name
    if ref.startswith(GCS_PUBLIC):
        path = urllib.parse.unquote(ref[len(GCS_PUBLIC):].split("?", 1)[0])
        bucket, _, name = path.partition("/")
        return bucket, name
    return None


def _resolve(ref: str, listings: _Listings) -> tuple[str, str, str, str]:
    """(status, bucket, object, how)."""
    if not ref:
        return "SOURCE_MISSING", "", "", "no URL or filename in the source list"
    parts = _split_reference(ref)
    if parts:
        bucket, name = parts
        objects = listings.get(bucket)
        if objects is None:
            return "SOURCE_MISSING", "", "", f"bucket '{bucket}' could not be listed"
        if name in objects:
            return "OK", bucket, name, "URL used as given"
        folder = name.rsplit("/", 1)[0] + "/"
        in_folder = [n for n in objects if n.startswith(folder) and n.lower().endswith(".mp4")]
        if len(in_folder) == 1:
            return "OK", bucket, in_folder[0], "file name truncated in source; only .mp4 in the same folder"
        if not in_folder:
            matches = [n for n in objects if n.lower().endswith(".mp4") and ("/" + folder) in ("/" + n)]
            if len(matches) == 1:
                return "OK", bucket, matches[0], "folder path incomplete in source; unique session folder match"
            return "SOURCE_MISSING", "", "", f"URL not found and {len(matches)} folder matches"
        return "SOURCE_MISSING", "", "", f"URL not found and {len(in_folder)} .mp4 files in the folder"
    if ref.lower().endswith(".mp4") and "/" not in ref:
        bucket, prefix = ELECTRONICS
        objects = listings.get(bucket) or {}
        pattern = re.compile(re.escape(prefix) + r"\d{4}-\d{2}-\d{2}" + re.escape(ref))
        matches = [n for n in objects if pattern.fullmatch(n)]
        if len(matches) == 1:
            return "OK", bucket, matches[0], "filename suffix; unique <date><suffix> match in electronics folder"
        return "SOURCE_MISSING", "", "", f"filename suffix matched {len(matches)} files"
    return "SOURCE_MISSING", "", "", "reference is not a URL or filename"


def resolve_source_list(csv_path: Path) -> list[dict]:
    rows = read_csv(csv_path)
    if not rows:
        raise SourceListError(f"{csv_path} has no clip rows")
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise SourceListError(f"{csv_path} is missing column(s): {', '.join(missing)} "
                              f"(expected: {', '.join(REQUIRED_COLUMNS)})")
    listings = _Listings()
    out = []
    for number, row in enumerate(rows, start=1):
        dataset = (row.get("Dataset") or "").strip()
        layout = LAYOUTS.get(dataset.lower())
        if layout is None:
            raise SourceListError(f"{csv_path} row {number}: unknown Dataset '{dataset}' "
                                  "(use Mono, Stereo or Six-Cam)")
        ref = (row.get("URL_or_File") or "").strip()
        status, bucket, name, how = _resolve(ref, listings)
        meta = (listings.get(bucket) or {}).get(name, {}) if status == "OK" else {}
        task = (row.get("Task") or "").strip()
        # readable, unique id: number + dataset + task (or the session folder when the task is not given)
        label = slug(task) or slug(name.rsplit("/", 2)[-2] if name.count("/") >= 1 else "") or "clip"
        out.append({
            "clip_no": number,
            "clip_id": f"{number:02d}_{slug(dataset)}_{label}",
            "dataset": dataset,
            "category": (row.get("Category") or "").strip(),
            "task": task,
            "camera": (row.get("Camera") or "").strip(),
            "layout": layout,
            "status": status,
            "source_url": GCS_PUBLIC + bucket + "/" + urllib.parse.quote(name) if status == "OK" else "",
            "bucket": bucket,
            "object": name,
            "size_bytes": meta.get("size", ""),
            "md5_base64": meta.get("md5", ""),
            "resolution": how,
            "original_reference": ref,
            "notes": (row.get("Notes") or "").strip(),
        })
        LOG.info("  %-7s %-50s %s", "OK" if status == "OK" else "MISSING", out[-1]["clip_id"], how)
    return out


def write_manifest(path: Path, manifest: list[dict]) -> None:
    write_csv(path, MANIFEST_FIELDS, manifest)
