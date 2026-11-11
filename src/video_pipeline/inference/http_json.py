"""Small JSON-over-HTTP client with retries, shared by inference providers.

Retries connection errors, timeouts and HTTP 408/425/429/5xx with exponential
backoff; other HTTP errors fail immediately. Non-JSON bodies (e.g. a tunnel's
HTML error page) raise a clear InferenceError. Headers (which may carry an API
key) are never logged.
"""

from __future__ import annotations

import http.client
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any

from video_pipeline.core.errors import InferenceError

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class _Retryable(Exception):
    pass


class JsonHttpClient:
    def __init__(self, base_url: str, headers: Mapping[str, str], *, timeout_seconds: float, max_retries: int,
                 retry_backoff_seconds: float, label: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._headers = dict(headers)
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._backoff = retry_backoff_seconds
        self._label = label

    @property
    def host(self) -> str:
        return urllib.parse.urlparse(self.base_url).netloc

    def _once(self, method: str, url: str, payload: Mapping[str, Any] | None, timeout: float) -> Any:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, method=method, headers=self._headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (configured URL)
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            if exc.code in RETRYABLE_STATUS:
                raise _Retryable(f"HTTP {exc.code}: {detail}") from exc
            raise InferenceError(f"{self._label} returned HTTP {exc.code} for {url}: {detail}") from exc
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError) as exc:
            raise _Retryable(str(getattr(exc, "reason", exc))) from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise InferenceError(f"Non-JSON response from {url} (is the URL correct?): {body[:200]!r}") from exc

    def request(self, method: str, path: str, payload: Mapping[str, Any] | None = None, *,
                timeout: float | None = None, absolute_url: str | None = None, retries: int | None = None
                ) -> tuple[Any, int]:
        """Returns (parsed JSON, attempts)."""
        url = absolute_url or self.base_url + path
        max_retries = self._max_retries if retries is None else retries
        last: Exception | None = None
        for attempt in range(1, max_retries + 2):
            try:
                return self._once(method, url, payload, timeout or self._timeout), attempt
            except _Retryable as exc:
                last = exc
                if attempt > max_retries:
                    break
                delay = self._backoff * (2 ** (attempt - 1))
                logger.warning("%s %s failed (attempt %d): %s. Retrying in %.1fs.", self._label, path or url, attempt, exc, delay)
                time.sleep(delay)
        raise InferenceError(f"{self._label} {path or url} failed after {max_retries + 1} attempt(s): {last}") from last
