"""Small shared helpers: atomic file writes, JSON, CSV, logging."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Iterable

LOG = logging.getLogger("handqc")


def setup_logging(level: str = "INFO", log_file: Path | None = None, stream=None) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(stream)
    console.setLevel(level.upper())
    console.setFormatter(fmt)
    root.addHandler(console)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def write_json(path: Path, data) -> None:
    def w(tmp: Path) -> None:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _atomic(path, w)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict], gz: bool = False) -> None:
    def w(tmp: Path) -> None:
        if gz:
            fh = gzip.open(tmp, "wt", encoding="utf-8", newline="")
        else:
            fh = open(tmp, "w", encoding="utf-8", newline="")
        with fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    _atomic(path, w)


def read_csv(path: Path) -> list[dict]:
    if path.suffix == ".gz":
        fh = gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    else:
        fh = open(path, encoding="utf-8-sig", newline="")
    with fh:
        return list(csv.DictReader(fh))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
