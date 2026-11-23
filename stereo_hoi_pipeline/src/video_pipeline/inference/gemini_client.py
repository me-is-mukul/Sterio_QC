"""Gemini on Vertex AI as an InferenceProvider (google-genai SDK).

Credentials come from core.gcp_auth (on the VM: its service account). The project needs the
Vertex AI API enabled and the account needs roles/aiplatform.user. Requests ask for JSON that
follows the prompt's schema (response_json_schema). Retries: 429 / 5xx / timeouts with backoff.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from video_pipeline.core.errors import InferenceError
from video_pipeline.inference.base import InferenceProvider, ModelInfo, ProviderRequest, ProviderResponse

logger = logging.getLogger(__name__)

_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


class GeminiProvider(InferenceProvider):
    name = "gemini"

    def __init__(self, *, project: str, location: str, model: str, credentials, timeout_seconds: float,
                 max_retries: int, retry_backoff_seconds: float, max_output_tokens: int | None,
                 thinking_level: str | None = None) -> None:
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        self._types = types
        self._model = model
        self._project = project
        self._location = location
        self._max_retries = max_retries
        self._backoff = retry_backoff_seconds
        self._max_output_tokens = max_output_tokens
        self._thinking = None
        if thinking_level:
            try:
                level = types.ThinkingLevel[thinking_level.upper()]
            except KeyError:
                raise InferenceError(f"Unknown Gemini thinking_level '{thinking_level}'. "
                                     f"Known: {[m.lower() for m in types.ThinkingLevel.__members__]}") from None
            self._thinking = types.ThinkingConfig(thinking_level=level)
        self._client = genai.Client(
            vertexai=True, project=project, location=location, credentials=credentials,
            http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self._model, "project": self._project, "location": self._location,
                "thinking_level": self._thinking.thinking_level.name.lower() if self._thinking else None}

    def model_info(self) -> ModelInfo:
        try:
            model = self._client.models.get(model=self._model)
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(
                f"Gemini model '{self._model}' is not available in {self._project}/{self._location}: {exc}. "
                "Check that the Vertex AI API is enabled and the account has roles/aiplatform.user."
            ) from exc
        version = getattr(model, "version", None)
        return ModelInfo(provider=self.name, model=self._model, digest=f"vertex:{model.name}@{version}",
                         runtime_version="vertex-ai", details={"name": model.name, "version": version})

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        types = self._types
        parts = [types.Part.from_bytes(data=image, mime_type="image/jpeg") for image in request.images]
        parts.append(types.Part.from_text(text=request.user_prompt))
        config = types.GenerateContentConfig(
            system_instruction=request.system_prompt,
            temperature=request.temperature,
            max_output_tokens=self._max_output_tokens,
            response_mime_type="application/json",
            response_json_schema=dict(request.response_schema) if request.response_schema else None,
            thinking_config=self._thinking,
        )
        last_error = ""
        started = time.perf_counter()
        for attempt in range(1, self._max_retries + 2):
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=[types.Content(role="user", parts=parts)], config=config)
                text = response.text
                if not isinstance(text, str) or not text.strip():
                    reason = None
                    if response.candidates:
                        reason = getattr(response.candidates[0], "finish_reason", None)
                    raise InferenceError(f"Gemini returned no text (finish reason: {reason})")
                raw = response.model_dump(mode="json", exclude_none=True)
                return ProviderResponse(text=text, raw=raw, latency_seconds=time.perf_counter() - started,
                                        attempts=attempt)
            except InferenceError:
                raise
            except Exception as exc:  # noqa: BLE001 (google.genai.errors.APIError and transport errors)
                code = getattr(exc, "code", None)
                last_error = f"{type(exc).__name__}: {exc}"
                retryable = code in _RETRYABLE_CODES or code is None
                if not retryable or attempt > self._max_retries:
                    raise InferenceError(f"Gemini request failed: {last_error}") from exc
                delay = self._backoff * (2 ** (attempt - 1))
                logger.debug("Gemini attempt %d failed (%s); retrying in %.1fs", attempt, last_error, delay)
                time.sleep(delay)
        raise InferenceError(f"Gemini request failed: {last_error}")
