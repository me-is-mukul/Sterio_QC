"""Small, safe file-writing helpers shared by every stage."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def json_default(value: Any) -> Any:
    """JSON fallback: numpy scalars/arrays become their Python values (a numpy bool must stay a
    boolean, not the string "False", which would read as true); anything else becomes str."""
    item = getattr(value, "item", None)
    if callable(item) and type(value).__module__ == "numpy":
        try:
            return value.tolist() if getattr(value, "ndim", 0) else item()
        except (TypeError, ValueError):
            pass
    return str(value)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(moment: datetime) -> str:
    """ISO-8601 UTC timestamp with millisecond precision and a 'Z' suffix."""
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def utc_now_iso() -> str:
    return to_iso(utc_now())


def replace_with_retry(src: str | Path, dst: Path, attempts: int = 40, delay: float = 0.05) -> None:
    """os.replace that tolerates brief locks on Windows.

    On Windows the rename fails with PermissionError (WinError 5/32) while another process has the
    target open, e.g. the dashboard reading manifest.json during a live run, or an antivirus scan.
    Such locks last milliseconds, so retry for up to ~4 s before giving up.
    """
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(delay * (attempt + 1), 0.25))


def write_text_atomic(path: Path, text: str) -> None:
    """Write text via a temporary file + rename so readers never see partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        replace_with_retry(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, data: Any) -> None:
    write_text_atomic(path, json.dumps(data, indent=2, ensure_ascii=False, default=json_default) + "\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(path, buffer.getvalue())


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def write_jsonl_atomic(path: Path, rows: Iterable[Any]) -> None:
    write_text_atomic(path, "".join(json.dumps(r, ensure_ascii=False, default=json_default) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[Any]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class JsonlAppender:
    """Append-only JSONL writer that flushes every row (lets observers follow progress live)."""

    def __init__(self, path: Path, *, truncate: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w" if truncate else "a", encoding="utf-8", newline="\n")

    def write(self, row: Any) -> None:
        self._handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> JsonlAppender:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
