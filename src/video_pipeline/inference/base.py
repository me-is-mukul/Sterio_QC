"""Inference provider interface.

The pipeline depends only on this interface. A provider knows how to talk to one
runtime (Ollama, vLLM, a cloud API, ...); it knows nothing about HOI, stereo or
the pipeline. HOI-specific prompting and response validation live in
``prompts.py`` and ``qwen_classifier.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProviderRequest:
    system_prompt: str
    user_prompt: str
    images: tuple[bytes, ...]                         # in prompt order (image 1, image 2, ...)
    response_schema: Mapping[str, Any] | None = None  # JSON schema for structured output, if supported
    temperature: float = 0.0


@dataclass(frozen=True)
class ProviderResponse:
    text: str                  # model's answer text
    raw: Mapping[str, Any]     # full provider response, preserved verbatim
    latency_seconds: float     # wall time of the successful request
    attempts: int


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    model: str
    digest: str | None             # identifies the exact weights; part of the cache key
    runtime_version: str | None
    details: Mapping[str, Any] = field(default_factory=dict)


class InferenceProvider(ABC):
    """A replaceable inference backend."""

    name: str

    @abstractmethod
    def model_info(self) -> ModelInfo:
        """Check the endpoint and model are available. Raises InferenceError if not."""

    @abstractmethod
    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Run one request (with the provider's own retry policy). Raises InferenceError on failure."""

    def describe(self) -> dict[str, Any]:
        """Non-secret description for manifests."""
        return {"provider": self.name}
