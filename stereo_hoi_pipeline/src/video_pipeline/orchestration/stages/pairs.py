"""Shared helpers for stages that extract and classify stereo pairs.

* ``PairExtractor`` extracts pairs for a time window (deterministic, byte-identical
  re-extraction, so images can be deleted right after use and regenerated later).
* ``create_pair_classifier`` returns the configured HOI detector (``detection.method``):
    - ``MediaPipePairClassifier`` (default): hand landmarks + grasp rules, runs locally;
    - ``PairClassifier``: a vision-language model through the inference provider.
  Both share one interface and a content-addressed cache; failures are never WASTE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_pipeline.core.config import ResolvedConfig
from video_pipeline.core.errors import InferenceError
from video_pipeline.core.hashing import sha256_file, sha256_json
from video_pipeline.core.io import read_json, utc_now_iso, write_json_atomic
from video_pipeline.core.tools import CommandResult, resolve_executable
from video_pipeline.inference.base import InferenceProvider
from video_pipeline.inference.ollama_client import OllamaProvider
from video_pipeline.inference.openai_compatible import OpenAICompatibleProvider
from video_pipeline.inference.prompts import get_prompt
from video_pipeline.inference.qwen_classifier import STATUS_OK, StereoHoiClassifier, failed_classification
from video_pipeline.orchestration.context import RunContext
from video_pipeline.orchestration.stages.base import (
    format_start_time,
    load_video_info,
    primary_video_stream,
    video_start_time,
)
from video_pipeline.stereo.frame_sampler import SampledFrame, sample_stereo_frames
from video_pipeline.stereo.stereo_splitter import StereoGeometry, compute_geometry

CACHE_FORMAT = 1

# Config keys that change what a classification means (used by stage fingerprints).
INFERENCE_CONFIG_KEYS = ("detection", "inference.provider", "inference.model", "inference.temperature",
                         "inference.think", "inference.max_output_tokens", "inference.prompt_version",
                         "inference.failure_policy")
FRAME_CONFIG_KEYS = ("stereo", "sampling.max_width", "sampling.jpeg_quality")


def create_provider(config: ResolvedConfig, environment) -> InferenceProvider:
    """Provider factory: the only place that knows concrete provider classes.

    API keys come from the environment / .env only and are never stored.
    """
    provider = config.get("inference.provider")
    common = dict(
        timeout_seconds=config.get("inference.timeout_seconds"),
        max_retries=config.get("inference.max_retries"),
        retry_backoff_seconds=config.get("inference.retry_backoff_seconds"),
        think=config.get("inference.think"),
        max_output_tokens=config.get("inference.max_output_tokens"),
    )
    if provider == "ollama":
        return OllamaProvider(
            config.get("inference.base_url"), config.get("inference.model"),
            keep_alive=config.get("inference.keep_alive"),
            api_key=(environment.get("OLLAMA_API_KEY") or "").strip() or None,
            **common,
        )
    if provider == "openai_compatible":
        return OpenAICompatibleProvider(
            config.get("inference.base_url"), config.get("inference.model"),
            api_key=(environment.get("VLLM_API_KEY") or "").strip() or None,
            **common,
        )
    raise InferenceError(f"Unknown inference provider '{provider}'.")


class PairExtractor:
    def __init__(self, ctx: RunContext) -> None:
        info = load_video_info(ctx)
        stream = primary_video_stream(info)
        self.geometry: StereoGeometry = compute_geometry(
            stream["width"], stream["height"], layout=ctx.config.get("stereo.layout"),
            split_ratio=ctx.config.get("stereo.split_ratio"))
        self._ctx = ctx
        self._ffmpeg = resolve_executable(ctx.config.get("tools.ffmpeg"))
        self._video_start = video_start_time(info)
        self._format_start = format_start_time(info)

    def extract(self, *, window_start: float, window_end: float, interval: float, left_dir: Path, right_dir: Path,
                prefix: str = "") -> tuple[list[SampledFrame], list[int], CommandResult]:
        cfg = self._ctx.config
        return sample_stereo_frames(
            self._ffmpeg, self._ctx.source.path, self.geometry, interval=interval, video_start=self._video_start,
            window_start=window_start, window_end=window_end, max_width=cfg.get("sampling.max_width"),
            jpeg_quality=cfg.get("sampling.jpeg_quality"), left_dir=left_dir, right_dir=right_dir,
            timeout_seconds=cfg.get("sampling.timeout_seconds"), name_prefix=prefix,
            format_start=self._format_start)


@dataclass
class PairResult:
    result: dict[str, Any]          # normalised classification (HoiClassification.to_dict())
    response_text: str | None
    provider_raw: dict[str, Any] | None
    latency_seconds: float | None
    attempts: int | None
    cached: bool
    cache_key: str
    error: str | None

    @property
    def label(self) -> str:
        return self.result["classification"]

    @property
    def reusable(self) -> bool:
        """True when the result is final (no need to keep the images for a retry)."""
        return self.result.get("status") == STATUS_OK


class PairClassifier:
    def __init__(self, ctx: RunContext) -> None:
        self._ctx = ctx
        self.provider = create_provider(ctx.config, ctx.environment)
        ctx.logger.info("Connecting to inference provider: %s", self.provider.describe())
        self.model = self.provider.model_info()
        ctx.logger.info("Model %s available (id %s, runtime %s).",
                        self.model.model, (self.model.digest or "?")[:28], self.model.runtime_version)
        self.prompt = get_prompt(ctx.config.get("inference.prompt_version"))
        temperature = ctx.config.get("inference.temperature")
        self._classifier = StereoHoiClassifier(self.provider, self.prompt, temperature=temperature)
        self.policy = ctx.config.get("inference.failure_policy")
        self.params = {"temperature": temperature, "think": ctx.config.get("inference.think"),
                       "max_output_tokens": ctx.config.get("inference.max_output_tokens")}
        self._cache_dir = ctx.workspace.inference_dir / "cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self.requests = 0

    def cache_key(self, left_sha: str, right_sha: str) -> str:
        return sha256_json({
            "format": CACHE_FORMAT, "left": left_sha, "right": right_sha, "provider": self.model.provider,
            "model": self.model.model, "digest": self.model.digest, "prompt_version": self.prompt.version,
            "params": self.params,
        })

    def cached(self, key: str) -> PairResult | None:
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        c = read_json(path)
        return PairResult(c["classification"], c["response_text"], c["provider_raw"], c["latency_seconds"],
                          c["attempts"], True, key, None)

    def classify(self, left: Path, right: Path, *, timestamp: float, key: str | None = None) -> PairResult:
        key = key or self.cache_key(sha256_file(left), sha256_file(right))
        hit = self.cached(key)
        if hit is not None:
            return hit
        self.requests += 1
        try:
            classification, response = self._classifier.classify(left.read_bytes(), right.read_bytes(),
                                                                 timestamp_seconds=timestamp)
        except InferenceError as exc:
            self._ctx.logger.error("Inference failed for t=%ss: %s", timestamp, exc)
            return PairResult(failed_classification(str(exc), self.policy).to_dict(), None, None, None, None,
                              False, key, str(exc))
        pair = PairResult(classification.to_dict(), response.text, dict(response.raw),
                          round(response.latency_seconds, 3), response.attempts, False, key, None)
        if pair.reusable:
            write_json_atomic(self._cache_dir / f"{key}.json", {
                "cache_format": CACHE_FORMAT, "classification": pair.result, "response_text": pair.response_text,
                "provider_raw": pair.provider_raw, "latency_seconds": pair.latency_seconds,
                "attempts": pair.attempts, "created_at": utc_now_iso(),
            })
        return pair

    def record_fields(self) -> dict[str, Any]:
        return {"detector": "vlm", "model": self.model.model, "model_digest": self.model.digest,
                "prompt_version": self.prompt.version}

    # common detector interface ----------------------------------------------------------------
    method = "vlm"

    def describe(self) -> str:
        return f"vision-language model {self.model.model} ({self.model.provider})"

    def metadata(self) -> dict[str, Any]:
        return {"detector": "vlm", "provider": self.provider.describe(), "model": self.model.model,
                "model_digest": self.model.digest, "runtime_version": self.model.runtime_version,
                "model_details": dict(self.model.details), "prompt_version": self.prompt.version,
                "parameters": self.params, "failure_policy": self.policy}

    def configuration_extra(self) -> dict[str, Any]:
        return {"detector": "vlm", "model_digest": self.model.digest, "prompt_version": self.prompt.version,
                "inference_runtime_version": self.model.runtime_version}

    def close(self) -> None:
        pass


class MediaPipePairClassifier:
    """HOI from hand landmarks (MediaPipe); same interface as PairClassifier."""

    method = "mediapipe"

    def __init__(self, ctx: RunContext) -> None:
        from video_pipeline.hands.detector import MODEL_SHA256, MediaPipeStereoDetector, ensure_model

        self._ctx = ctx
        settings = ctx.config.get("detection.mediapipe")
        model_path = Path(settings["model_path"])
        model_path = model_path if model_path.is_absolute() else ctx.project_root / model_path
        self.model_sha256 = ensure_model(model_path, ctx.logger)
        self.validated_model = self.model_sha256 == MODEL_SHA256
        self._detector = MediaPipeStereoDetector(model_path, settings)
        self.settings = {k: v for k, v in settings.items() if k != "model_path"}
        self.policy = ctx.config.get("inference.failure_policy")
        self._cache_dir = ctx.workspace.inference_dir / "cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self.requests = 0
        ctx.logger.info("HOI detector: MediaPipe Hand Landmarker %s (model %s); hands without a grasp -> %s",
                        self._detector.mediapipe_version, self.model_sha256[:12], settings["hands_without_grasp"])

    def describe(self) -> str:
        return f"MediaPipe Hand Landmarker {self._detector.mediapipe_version}"

    def cache_key(self, left_sha: str, right_sha: str) -> str:
        return sha256_json({"format": CACHE_FORMAT, "detector": "mediapipe", "left": left_sha, "right": right_sha,
                            "model": self.model_sha256, "mediapipe": self._detector.mediapipe_version,
                            "settings": self.settings})

    def cached(self, key: str) -> PairResult | None:
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        c = read_json(path)
        return PairResult(c["classification"], None, None, c["latency_seconds"], None, True, key, None)

    def classify(self, left: Path, right: Path, *, timestamp: float, key: str | None = None) -> PairResult:
        key = key or self.cache_key(sha256_file(left), sha256_file(right))
        hit = self.cached(key)
        if hit is not None:
            return hit
        self.requests += 1
        try:
            result, seconds = self._detector.timed_classify(left, right)
        except InferenceError as exc:
            self._ctx.logger.error("Hand detection failed for t=%ss: %s", timestamp, exc)
            return PairResult(failed_classification(str(exc), self.policy).to_dict(), None, None, None, None,
                              False, key, str(exc))
        pair = PairResult(result, None, None, round(seconds, 4), 1, False, key, None)
        write_json_atomic(self._cache_dir / f"{key}.json", {
            "cache_format": CACHE_FORMAT, "classification": result, "latency_seconds": pair.latency_seconds,
            "created_at": utc_now_iso()})
        return pair

    def record_fields(self) -> dict[str, Any]:
        return {"detector": "mediapipe", "model": "hand_landmarker", "model_digest": self.model_sha256}

    def metadata(self) -> dict[str, Any]:
        return {"detector": "mediapipe", "mediapipe_version": self._detector.mediapipe_version,
                "model": "MediaPipe Hand Landmarker (float16, v1)", "model_sha256": self.model_sha256,
                "validated_model": self.validated_model, "settings": self.settings, "failure_policy": self.policy,
                "rule": ("RELEVANT = a wearer hand grasping in either eye; WASTE = no wearer hand; wearer hands "
                         f"without a grasp -> {self.settings['hands_without_grasp']}")}

    def configuration_extra(self) -> dict[str, Any]:
        return {"detector": "mediapipe", "model_digest": self.model_sha256,
                "mediapipe_version": self._detector.mediapipe_version}

    def close(self) -> None:
        self._detector.close()


def create_pair_classifier(ctx: RunContext, method: str | None = None) -> PairClassifier | MediaPipePairClassifier:
    """The HOI detector selected by detection.method (CLI: --detector), or `method` if given."""
    if (method or ctx.config.get("detection.method")) == "mediapipe":
        return MediaPipePairClassifier(ctx)
    return PairClassifier(ctx)


def delete_pair(*paths: Path) -> None:
    for p in paths:
        p.unlink(missing_ok=True)
