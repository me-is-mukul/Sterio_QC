"""Download source videos into the cache with resume, size and MD5 verification."""

from __future__ import annotations

import base64
import hashlib
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .manifest import Clip
from .util import LOG

CHUNK = 1 << 20


class DownloadError(Exception):
    pass


def _md5_b64(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 (integrity check against the bucket's published MD5)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK * 8), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def _verify(clip: Clip, path: Path, verify_md5: bool) -> None:
    size = path.stat().st_size
    if clip.size_bytes is not None and size != clip.size_bytes:
        raise DownloadError(f"size mismatch: {size} bytes on disk, {clip.size_bytes} expected")
    if verify_md5 and clip.md5_base64:
        got = _md5_b64(path)
        if got != clip.md5_base64:
            raise DownloadError(f"MD5 mismatch: {got} on disk, {clip.md5_base64} expected")


def _stamp(clip: Clip) -> str:
    return f"{clip.source_url}|{clip.size_bytes}|{clip.md5_base64}"


def ensure_downloaded(clip: Clip, cache_dir: Path, *, retries: int, timeout: float, verify_md5: bool) -> Path:
    """Return the local path of the clip, downloading (and verifying) it if needed."""
    target = clip.cache_path(cache_dir)
    marker = target.with_name(target.name + ".verified")
    if target.is_file() and marker.is_file() and marker.read_text(encoding="utf-8").strip() == _stamp(clip):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if target.is_file():                        # present but not verified -> verify (and resume) it
        os.replace(target, partial)

    last_error = ""
    for attempt in range(1, retries + 2):
        try:
            have = partial.stat().st_size if partial.exists() else 0
            if clip.size_bytes is not None and have > clip.size_bytes:
                partial.unlink()
                have = 0
            if clip.size_bytes is None or have < clip.size_bytes:
                headers = {"Range": f"bytes={have}-"} if have else {}
                req = urllib.request.Request(clip.source_url, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (manifest URL)
                    if have and resp.status != 206:
                        have = 0                 # server ignored the range request: start over
                    started, done = time.monotonic(), 0
                    with partial.open("ab" if have else "wb") as fh:
                        while True:
                            chunk = resp.read(CHUNK)
                            if not chunk:
                                break
                            fh.write(chunk)
                            done += len(chunk)
                    rate = done / max(time.monotonic() - started, 1e-6) / 1e6
                    LOG.info("[%s] downloaded %.1f MB (%.1f MB/s)", clip.clip_id, done / 1e6, rate)
            _verify(clip, partial, verify_md5)
            os.replace(partial, target)
            marker.write_text(_stamp(clip), encoding="utf-8")
            return target
        except DownloadError as exc:
            last_error = str(exc)
            LOG.warning("[%s] verification failed (%s); downloading again", clip.clip_id, exc)
            partial.unlink(missing_ok=True)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(getattr(exc, "reason", exc))
            LOG.warning("[%s] download attempt %d failed: %s", clip.clip_id, attempt, last_error)
        if attempt <= retries:
            time.sleep(min(30, 2 ** attempt))
    raise DownloadError(f"could not download {clip.source_url}: {last_error}")
