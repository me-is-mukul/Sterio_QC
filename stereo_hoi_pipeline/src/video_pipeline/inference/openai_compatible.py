"""OpenAI-compatible Chat Completions implementation of InferenceProvider.

Works with vLLM (`vllm serve ...`) and hosted OpenAI-compatible APIs that accept
image inputs. ``base_url`` is the API root including the version, e.g.
``http://gpu-box:8000/v1`` or ``https://api.deepinfra.com/v1/openai``.

* ``POST {base_url}/chat/completions`` with both images as base64 data URLs and
  ``response_format = {"type": "json_schema", ...}`` (vLLM structured output).
* ``think`` maps to ``chat_template_kwargs.enable_thinking`` (Qwen3 chat template,
  supported by vLLM). Set ``inference.think: null`` if a host rejects that field.
* ``GET {base_url}/models`` verifies the model is served. OpenAI APIs expose no
  weight digest, so the model identity used in cache keys is a hash of the served
  model id and its ``root`` (checkpoint path) when the server reports one.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

from video_pipeline import __version__
from video_pipeline.core.errors import InferenceError
from video_pipeline.core.hashing import sha256_json
from video_pipeline.inference.base import InferenceProvider, ModelInfo, ProviderRequest, ProviderResponse
from video_pipeline.inference.http_json import JsonHttpClient

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(InferenceProvider):
    name = "openai_compatible"

    def __init__(self, base_url: str, model: str, *, timeout_seconds: float, max_retries: int,
                 retry_backoff_seconds: float, think: bool | None, max_output_tokens: int | None,
                 api_key: str | None = None) -> None:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"stereo-hoi-pipeline/{__version__}",
            "ngrok-skip-browser-warning": "true",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = JsonHttpClient(base_url, headers, timeout_seconds=timeout_seconds, max_retries=max_retries,
                                    retry_backoff_seconds=retry_backoff_seconds, label="OpenAI-compatible API")
        self._model = model
        self._timeout = timeout_seconds
        self._think = think
        self._max_output_tokens = max_output_tokens
        self._authenticated = bool(api_key)

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "host": self._http.host, "model": self._model, "think": self._think,
                "authenticated": self._authenticated}

    def _runtime_version(self) -> str | None:
        # vLLM exposes GET /version at the server root (outside /v1). Optional.
        root = self._http.base_url.rsplit("/v1", 1)[0] if "/v1" in self._http.base_url else self._http.base_url
        try:
            payload, _ = self._http.request("GET", "", absolute_url=root + "/version", timeout=10, retries=0)
            return f"vllm {payload.get('version')}" if isinstance(payload, dict) and payload.get("version") else None
        except InferenceError:
            return None

    def model_info(self) -> ModelInfo:
        try:
            payload, _ = self._http.request("GET", "/models", timeout=min(self._timeout, 30))
        except InferenceError as exc:
            raise InferenceError(
                f"Cannot reach the OpenAI-compatible API at {self._http.host}. Check VLLM_BASE_URL (must include "
                f"the /v1 part) and VLLM_API_KEY in .env, and that the server is running. ({exc})"
            ) from exc
        models = payload.get("data", []) if isinstance(payload, dict) else []
        entry = next((m for m in models if m.get("id") == self._model), None)
        if entry is None:
            available = sorted(str(m.get("id")) for m in models)[:20]
            raise InferenceError(f"Model '{self._model}' is not served at {self._http.host}. Available: {available}. "
                                 "Set VLLM_MODEL to the exact model id.")
        identity = {"id": entry.get("id"), "root": entry.get("root")}
        return ModelInfo(
            provider=self.name,
            model=self._model,
            digest="openai-compat:" + sha256_json(identity)[:32],
            runtime_version=self._runtime_version(),
            details={k: entry.get(k) for k in ("root", "owned_by", "max_model_len") if entry.get(k) is not None},
        )

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        content: list[dict[str, Any]] = [{"type": "text", "text": request.user_prompt}]
        for image in request.images:
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")}})
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": request.temperature,
            "stream": False,
        }
        if self._max_output_tokens:
            payload["max_tokens"] = self._max_output_tokens
        if request.response_schema is not None:
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "hoi_classification",
                                                          "schema": dict(request.response_schema)}}
        if self._think is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self._think}

        started = time.perf_counter()
        raw, attempts = self._http.request("POST", "/chat/completions", payload)
        latency = time.perf_counter() - started
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if not choices:
            raise InferenceError(f"Response has no choices: {str(raw)[:200]}")
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if not text.strip() and isinstance(reasoning, str) and reasoning.strip():
            logger.debug("Empty content; using reasoning field as the answer.")
            text = reasoning
        if choices[0].get("finish_reason") == "length":
            logger.warning("Model output hit max_tokens (%s); the answer may be truncated.", self._max_output_tokens)
        return ProviderResponse(text=text, raw=raw, latency_seconds=latency, attempts=attempts)
