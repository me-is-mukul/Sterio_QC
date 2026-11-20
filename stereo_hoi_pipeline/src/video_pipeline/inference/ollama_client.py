"""Ollama HTTP implementation of InferenceProvider (``/api/chat``, ``/api/tags``, ``/api/version``).

Works with a local Ollama or one exposed through a tunnel such as ngrok. The
``ngrok-skip-browser-warning`` header is always sent (harmless elsewhere). An
optional API key is sent as a Bearer token; it is never logged or persisted.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

from video_pipeline import __version__
from video_pipeline.core.errors import InferenceError
from video_pipeline.inference.base import InferenceProvider, ModelInfo, ProviderRequest, ProviderResponse
from video_pipeline.inference.http_json import JsonHttpClient

logger = logging.getLogger(__name__)


class OllamaProvider(InferenceProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, *, timeout_seconds: float, max_retries: int,
                 retry_backoff_seconds: float, think: bool | None, keep_alive: str | None,
                 max_output_tokens: int | None = None, api_key: str | None = None) -> None:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"stereo-hoi-pipeline/{__version__}",
            "ngrok-skip-browser-warning": "true",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = JsonHttpClient(base_url, headers, timeout_seconds=timeout_seconds, max_retries=max_retries,
                                    retry_backoff_seconds=retry_backoff_seconds, label="Ollama")
        self._model = model
        self._timeout = timeout_seconds
        self._think = think
        self._keep_alive = keep_alive
        self._max_output_tokens = max_output_tokens
        self._authenticated = bool(api_key)

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "host": self._http.host, "model": self._model, "think": self._think,
                "authenticated": self._authenticated}

    def model_info(self) -> ModelInfo:
        try:
            version_payload, _ = self._http.request("GET", "/api/version", timeout=min(self._timeout, 30))
            tags_payload, _ = self._http.request("GET", "/api/tags", timeout=min(self._timeout, 30))
        except InferenceError as exc:
            raise InferenceError(
                f"Cannot reach Ollama at {self._http.host}. Check OLLAMA_BASE_URL in .env and that "
                f"Ollama/ngrok are running. ({exc})"
            ) from exc
        wanted = {self._model, f"{self._model}:latest"}
        for entry in tags_payload.get("models", []):
            if entry.get("name") in wanted or entry.get("model") in wanted:
                return ModelInfo(
                    provider=self.name,
                    model=self._model,
                    digest=entry.get("digest"),
                    runtime_version=version_payload.get("version"),
                    details={k: entry.get(k) for k in ("size", "modified_at", "details")},
                )
        available = sorted(e.get("name", "?") for e in tags_payload.get("models", []))
        raise InferenceError(f"Model '{self._model}' is not available on the Ollama server. Available: {available}. "
                             f"Run `ollama pull {self._model}`.")

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        options: dict[str, Any] = {"temperature": request.temperature}
        if self._max_output_tokens:
            options["num_predict"] = self._max_output_tokens
        payload: dict[str, Any] = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt,
                 "images": [base64.b64encode(img).decode("ascii") for img in request.images]},
            ],
            "options": options,
        }
        if request.response_schema is not None:
            payload["format"] = dict(request.response_schema)
        if self._think is not None:
            payload["think"] = self._think
        if self._keep_alive:
            payload["keep_alive"] = self._keep_alive

        started = time.perf_counter()
        raw, attempts = self._http.request("POST", "/api/chat", payload)
        latency = time.perf_counter() - started
        if isinstance(raw, dict) and raw.get("error"):
            raise InferenceError(f"Ollama error: {raw['error']}")
        message = raw.get("message") or {}
        text = message.get("content")
        if not isinstance(text, str):
            raise InferenceError("Ollama response has no message content.")
        thinking = message.get("thinking")
        if not text.strip() and isinstance(thinking, str) and thinking.strip():
            # Known Ollama/Qwen3-VL behaviour: with think=false plus a `format` schema the
            # structured answer can arrive in `message.thinking` with empty `content`.
            # Use it as the answer; it is still fully validated, and `raw` keeps both fields.
            logger.debug("Empty content; using message.thinking as the answer.")
            text = thinking
        return ProviderResponse(text=text, raw=raw, latency_seconds=latency, attempts=attempts)
