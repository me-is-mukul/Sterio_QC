"""Download a video from an HTTP(S) URL.

Responsibility: network transfer only. No hashing, probing or workspace logic.

Guarantees:
* The original remote filename is preserved (Content-Disposition, else URL path).
* Data is streamed to ``<name>.part`` and atomically renamed on success, so a
  partially downloaded file is never mistaken for a complete one.
* An existing file is never overwritten unless ``confirm_overwrite`` returns True.
* Transient failures are retried with exponential backoff.
"""

from __future__ import annotations

import email.message
import http.client
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from video_pipeline.core.errors import DownloadError, FileConflictError
from video_pipeline.core.io import to_iso, utc_now

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = frozenset({"http", "https"})
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._\- ]+")


@dataclass(frozen=True)
class DownloadSettings:
    timeout_seconds: float
    max_retries: int
    retry_backoff_seconds: float
    chunk_size_bytes: int
    progress_log_percent_step: int
    user_agent: str


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    url: str
    final_url: str
    http_status: int
    content_type: str | None
    content_length: int | None
    etag: str | None
    last_modified: str | None
    bytes_written: int
    started_at: str
    finished_at: str
    elapsed_seconds: float
    attempts: int

    def to_dict(self) -> dict:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


class _RetryableError(Exception):
    pass


def validate_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES or not parsed.netloc:
        raise DownloadError(f"Unsupported URL '{url}'. Only http:// and https:// URLs are accepted.")


def sanitize_filename(name: str) -> str:
    """Strip directory components and unsafe characters; reject empty results."""
    base = Path(name.replace("\\", "/")).name.strip()
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", base).strip(" .")
    if not cleaned or cleaned in {".", ".."}:
        raise DownloadError(f"Cannot derive a safe filename from '{name}'.")
    return cleaned


def filename_from_headers(headers: Mapping[str, str] | email.message.Message) -> str | None:
    disposition = headers.get("Content-Disposition")
    if not disposition:
        return None
    message = email.message.Message()
    message["Content-Disposition"] = disposition
    return message.get_filename()  # handles RFC 2231 filename*=


def filename_from_url(url: str) -> str | None:
    path = urllib.parse.unquote(urllib.parse.urlparse(url).path)
    name = path.rstrip("/").rsplit("/", 1)[-1]
    return name or None


def resolve_remote_filename(url: str, settings: DownloadSettings) -> str:
    """Determine the remote filename before downloading (so conflicts are caught early).

    Tries a HEAD request (following redirects) for Content-Disposition, then the
    final URL path, then the original URL path.
    """
    validate_url(url)
    candidates: list[str | None] = []
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": settings.user_agent})
    try:
        with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:  # noqa: S310 (scheme validated)
            candidates.append(filename_from_headers(response.headers))
            candidates.append(filename_from_url(response.geturl()))
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
        logger.debug("HEAD request failed (%s); falling back to URL path for filename.", exc)
    candidates.append(filename_from_url(url))
    for candidate in candidates:
        if candidate:
            return sanitize_filename(candidate)
    raise DownloadError(f"Cannot determine a filename for '{url}'.")


def download(
    url: str,
    destination_dir: Path,
    settings: DownloadSettings,
    *,
    filename: str | None = None,
    confirm_overwrite: Callable[[Path], bool],
) -> DownloadResult:
    """Download ``url`` into ``destination_dir`` and return transfer metadata."""
    validate_url(url)
    name = sanitize_filename(filename) if filename else resolve_remote_filename(url, settings)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / name
    if destination.exists():
        if not confirm_overwrite(destination):
            raise FileConflictError(
                f"'{destination}' already exists and overwriting was not confirmed. "
                f"Use --input \"{destination}\" to process the existing file, or remove it first."
            )
        logger.warning("Existing file '%s' will be replaced after a successful download.", destination)

    part = destination.with_name(destination.name + ".part")
    started = utc_now()
    t0 = time.perf_counter()
    last_error: Exception | None = None

    for attempt in range(1, settings.max_retries + 2):
        try:
            status, headers, final_url, written = _attempt(url, part, settings)
        except _RetryableError as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            if attempt > settings.max_retries:
                break
            delay = settings.retry_backoff_seconds * (2 ** (attempt - 1))
            logger.warning("Download attempt %d failed: %s. Retrying in %.1fs.", attempt, exc, delay)
            time.sleep(delay)
            continue
        except BaseException:
            part.unlink(missing_ok=True)
            raise

        os.replace(part, destination)
        content_length = _int_or_none(headers.get("Content-Length"))
        result = DownloadResult(
            path=destination,
            url=url,
            final_url=final_url,
            http_status=status,
            content_type=headers.get("Content-Type"),
            content_length=content_length,
            etag=headers.get("ETag"),
            last_modified=headers.get("Last-Modified"),
            bytes_written=written,
            started_at=to_iso(started),
            finished_at=to_iso(utc_now()),
            elapsed_seconds=round(time.perf_counter() - t0, 3),
            attempts=attempt,
        )
        logger.info("Downloaded %s (%s bytes) in %.1fs.", destination.name, f"{written:,}", result.elapsed_seconds)
        return result

    raise DownloadError(f"Download failed after {settings.max_retries + 1} attempt(s): {last_error}") from last_error


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _attempt(url: str, part: Path, settings: DownloadSettings) -> tuple[int, email.message.Message, str, int]:
    request = urllib.request.Request(url, headers={"User-Agent": settings.user_agent})
    try:
        with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:  # noqa: S310 (scheme validated)
            headers = response.headers
            expected = _int_or_none(headers.get("Content-Length"))
            written = 0
            next_report = settings.progress_log_percent_step
            next_report_bytes = 256 * 1024 * 1024
            with part.open("wb") as handle:
                while chunk := response.read(settings.chunk_size_bytes):
                    handle.write(chunk)
                    written += len(chunk)
                    if expected:
                        percent = written * 100 // expected
                        if percent >= next_report:
                            logger.info("Download progress: %d%% (%s / %s bytes)", percent, f"{written:,}", f"{expected:,}")
                            next_report = (percent // settings.progress_log_percent_step + 1) * settings.progress_log_percent_step
                    elif written >= next_report_bytes:
                        logger.info("Download progress: %s bytes", f"{written:,}")
                        next_report_bytes += 256 * 1024 * 1024
            if expected is not None and written != expected:
                raise _RetryableError(f"incomplete transfer: received {written} of {expected} bytes")
            if written == 0:
                raise DownloadError(f"Server returned an empty body for '{url}'.")
            return response.status, headers, response.geturl(), written
    except urllib.error.HTTPError as exc:
        if exc.code in RETRYABLE_STATUS:
            raise _RetryableError(f"HTTP {exc.code} {exc.reason}") from exc
        raise DownloadError(f"HTTP {exc.code} {exc.reason} while downloading '{url}'.") from exc
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError) as exc:
        raise _RetryableError(str(exc)) from exc
