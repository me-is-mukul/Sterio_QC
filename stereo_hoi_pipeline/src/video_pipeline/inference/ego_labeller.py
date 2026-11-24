"""Frame labelling for the QA sheet (sections 3 and 4): every model labels the same sampled frames.

Each answer is validated strictly against the prompt schema. A failed or invalid answer is stored
with status "failed"/"invalid" and is NEVER treated as idle / no hands (the cropping rules keep
such stretches). A model whose failure share is above `labelling.max_failure_ratio` stops the run
with an error instead, so an unavailable model can never make a recording pass by default.

Results are cached per model in ``labels_<model>.jsonl`` (one line per sample); a re-run only asks
for samples that have no successful answer yet.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from video_pipeline.core.errors import InferenceError
from video_pipeline.core.hashing import sha256_json
from video_pipeline.inference.base import InferenceProvider, ProviderRequest
from video_pipeline.inference.prompts import EGO_FRAME_SCHEMA, PromptTemplate, get_prompt
from video_pipeline.inference.qwen_classifier import extract_json_object

logger = logging.getLogger(__name__)

STATUS_OK = "ok"


@dataclass(frozen=True)
class Sample:
    index: int          # k: the k-th sample
    frame: int          # frame index in the recording
    t_ns: int           # capture time (device clock) of the left-eye frame
    t_s: float          # seconds since the first frame
    image: Path         # left-eye JPEG


def validate_answer(text: str, schema: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """Strict check of a JSON answer against a flat object schema (enums, booleans, strings)."""
    data = extract_json_object(text or "")
    if data is None:
        return None, ["answer is not a JSON object"]
    errors: list[str] = []
    out: dict[str, Any] = {}
    for key in schema["required"]:
        spec = schema["properties"][key]
        if key not in data:
            errors.append(f"missing '{key}'")
            continue
        value = data[key]
        if spec["type"] == "boolean":
            if not isinstance(value, bool):
                errors.append(f"'{key}' is not a boolean")
                continue
        elif spec["type"] == "string":
            if not isinstance(value, str):
                errors.append(f"'{key}' is not a string")
                continue
            if "enum" in spec:
                value = value.strip().lower()
                if value not in spec["enum"]:
                    errors.append(f"'{key}'={data[key]!r} not in {spec['enum']}")
                    continue
        out[key] = value
    return (out if not errors else None), errors


class FrameLabeller:
    """One model. ``provides`` is a subset of {'hands', 'activity', 'content'}."""

    def __init__(self, key: str, provides: Sequence[str], concurrency: int) -> None:
        self.key = key
        self.provides = tuple(provides)
        self.concurrency = concurrency

    def describe(self) -> dict[str, Any]:
        raise NotImplementedError

    def fingerprint(self) -> str:
        """Identifies the model and its settings; cached answers of another fingerprint are not reused."""
        return sha256_json(self.describe())[:16]

    def begin_session(self, samples: Sequence[Sample], imu_t: np.ndarray, imu: np.ndarray) -> None:
        """Called once per recording before labelling (for labellers that need more than one frame)."""

    def label(self, sample: Sample) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class VlmLabeller(FrameLabeller):
    def __init__(self, key: str, provides: Sequence[str], concurrency: int, provider: InferenceProvider,
                 prompt: PromptTemplate, temperature: float) -> None:
        super().__init__(key, provides, concurrency)
        self.provider = provider
        self.prompt = prompt
        self.temperature = temperature

    def describe(self) -> dict[str, Any]:
        return {**self.provider.describe(), "prompt_version": self.prompt.version, "provides": list(self.provides)}

    def ask(self, system: str, user: str, images: Sequence[bytes], schema: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            response = self.provider.generate(ProviderRequest(system, user, tuple(images), schema, self.temperature))
        except InferenceError as exc:
            return {"status": "failed", "error": str(exc)[:500], "latency_s": round(time.perf_counter() - started, 3)}
        labels, errors = validate_answer(response.text, schema)
        if labels is None:
            return {"status": "invalid", "error": "; ".join(errors), "answer": response.text[:500],
                    "latency_s": round(response.latency_seconds, 3), "attempts": response.attempts}
        return {"status": STATUS_OK, "labels": labels, "latency_s": round(response.latency_seconds, 3),
                "attempts": response.attempts}

    def label(self, sample: Sample) -> dict[str, Any]:
        image = sample.image.read_bytes()
        return self.ask(self.prompt.system, self.prompt.user_prompt(timestamp_seconds=sample.t_s), [image],
                        self.prompt.response_schema)


class MediaPipeHandsLabeller(FrameLabeller):
    """Counts the wearer's hands with MediaPipe (no activity, no content)."""

    def __init__(self, key: str, concurrency: int, model_path: Path, settings: Mapping[str, Any]) -> None:
        super().__init__(key, ("hands",), 1)       # the MediaPipe landmarker is not thread-safe
        from video_pipeline.hands.detector import MediaPipeStereoDetector, ensure_model  # noqa: PLC0415

        self._digest = ensure_model(model_path, logger)
        self._settings = {k: v for k, v in settings.items() if k != "model_path"}
        self._detector = MediaPipeStereoDetector(model_path, settings)
        self._lock = threading.Lock()

    def describe(self) -> dict[str, Any]:
        return {"provider": "mediapipe", "model": "MediaPipe Hand Landmarker", "model_sha256": self._digest,
                "mediapipe_version": self._detector.mediapipe_version, "settings": self._settings,
                "provides": list(self.provides)}

    def label(self, sample: Sample) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            with self._lock:
                hands = self._detector.view_hands(sample.image)
        except InferenceError as exc:
            return {"status": "failed", "error": str(exc)[:500]}
        wearer = [h for h in hands if h["role"] == "wearer"]
        count = {0: "none", 1: "one"}.get(len(wearer), "both")
        return {"status": STATUS_OK, "labels": {"hands": count, "other_hands": len(hands) - len(wearer)},
                "latency_s": round(time.perf_counter() - started, 3)}

    def close(self) -> None:
        self._detector.close()


class MotionIdleLabeller(FrameLabeller):
    """Model-free idle detection: a sample is idle when NOTHING moves around it.

    Both must hold:
      * the image barely changes between this sample and its neighbouring samples (previous and
        next, ``labelling.interval_seconds`` apart): the share of changed pixels (blurred, downscaled
        grey image, change > ``pixel_threshold`` grey levels) is at most ``max_changed_fraction``;
      * the head is still: the median angular speed of the gyroscope within +/- half the sampling
        interval is at most ``max_gyro_rad_s``.
    A working wearer moves their hands (the image changes) or their head, so this only flags
    stretches where the wearer is truly inactive; it errs on the side of keeping footage.
    """

    def __init__(self, key: str, concurrency: int, settings: Mapping[str, Any], interval_seconds: float) -> None:
        super().__init__(key, ("activity",), concurrency)
        self.settings = dict(settings)
        self.interval_seconds = float(interval_seconds)
        self._images: dict[int, Path] = {}
        self._imu_t = np.zeros(0, dtype=np.int64)
        self._gyro = np.zeros(0)

    def describe(self) -> dict[str, Any]:
        return {"provider": "motion", "model": "image change + gyroscope (no AI model)", "settings": self.settings,
                "interval_seconds": self.interval_seconds, "provides": list(self.provides)}

    def begin_session(self, samples: Sequence[Sample], imu_t: np.ndarray, imu: np.ndarray) -> None:
        self._images = {s.index: s.image for s in samples}
        self._imu_t = np.asarray(imu_t, dtype=np.int64)
        self._gyro = np.linalg.norm(np.asarray(imu)[:, 0:3].astype(np.float64), axis=1) if len(imu) else np.zeros(0)

    def _small(self, path: Path) -> np.ndarray:
        import cv2  # noqa: PLC0415

        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise InferenceError(f"Cannot read sampled frame {path}")
        width = int(self.settings["image_width"])
        small = cv2.resize(gray, (width, max(1, round(gray.shape[0] * width / gray.shape[1]))),
                           interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small, (5, 5), 0).astype(np.int16)

    def changed_fraction(self, a: Path, b: Path) -> float:
        return float(np.mean(np.abs(self._small(a) - self._small(b)) > self.settings["pixel_threshold"]))

    def label(self, sample: Sample) -> dict[str, Any]:
        started = time.perf_counter()
        if sample.index not in self._images:
            return {"status": "failed", "error": "begin_session() was not called for this recording"}
        changes = {}
        for name, other in (("prev", sample.index - 1), ("next", sample.index + 1)):
            if other in self._images:
                changes[name] = round(self.changed_fraction(sample.image, self._images[other]), 5)
        half = int(self.interval_seconds * 1e9 / 2)
        lo, hi = np.searchsorted(self._imu_t, sample.t_ns - half), np.searchsorted(self._imu_t, sample.t_ns + half)
        gyro = round(float(np.median(self._gyro[lo:hi])), 5) if hi - lo >= 10 else None
        image_still = bool(changes) and max(changes.values()) <= self.settings["max_changed_fraction"]
        head_still = gyro is not None and gyro <= self.settings["max_gyro_rad_s"]
        return {"status": STATUS_OK, "latency_s": round(time.perf_counter() - started, 3),
                "labels": {"activity": "idle" if image_still and head_still else "moving",
                           "changed_prev": changes.get("prev"), "changed_next": changes.get("next"),
                           "gyro_median_rad_s": gyro}}


# ---------------------------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------------------------

def build_provider(spec: Mapping[str, Any], *, inference: Mapping[str, Any], environment: Mapping[str, str],
                   gcp_project: str, credentials_factory: Callable[[], Any]) -> InferenceProvider:
    from video_pipeline.inference.gemini_client import GeminiProvider  # noqa: PLC0415
    from video_pipeline.inference.ollama_client import OllamaProvider  # noqa: PLC0415
    from video_pipeline.inference.openai_compatible import OpenAICompatibleProvider  # noqa: PLC0415

    provider = spec["provider"]
    common = dict(timeout_seconds=inference["timeout_seconds"], max_retries=inference["max_retries"],
                  retry_backoff_seconds=inference["retry_backoff_seconds"])
    if provider == "gemini":
        return GeminiProvider(project=gcp_project, location=spec["location"], model=spec["model"],
                              credentials=credentials_factory(), max_output_tokens=inference["max_output_tokens"],
                              thinking_level=spec.get("thinking_level"), **common)
    if provider == "env":
        provider = (environment.get("INFERENCE_PROVIDER") or "").strip()
        if provider not in {"ollama", "openai_compatible"}:
            raise InferenceError("provider 'env' needs INFERENCE_PROVIDER=ollama or openai_compatible in .env "
                                 f"(found {provider!r}).")
        prefix = "OLLAMA" if provider == "ollama" else "VLLM"
        spec = {**spec, "base_url": spec.get("base_url") or environment.get(f"{prefix}_BASE_URL", "").strip(),
                "model": spec.get("model") or environment.get(f"{prefix}_MODEL", "").strip()}
    if not spec.get("base_url") or not spec.get("model"):
        raise InferenceError(f"model '{spec}' needs base_url and model (in the config or .env).")
    if provider == "ollama":
        return OllamaProvider(spec["base_url"], spec["model"], think=inference["think"], keep_alive=spec.get("keep_alive"),
                              max_output_tokens=inference["max_output_tokens"],
                              api_key=(environment.get("OLLAMA_API_KEY") or "").strip() or None, **common)
    if provider == "openai_compatible":
        return OpenAICompatibleProvider(spec["base_url"], spec["model"], think=inference["think"],
                                        max_output_tokens=inference["max_output_tokens"],
                                        api_key=(environment.get("VLLM_API_KEY") or "").strip() or None, **common)
    raise InferenceError(f"Unknown provider '{provider}'.")


def build_labellers(models: Mapping[str, Mapping[str, Any]], *, inference: Mapping[str, Any],
                    mediapipe: Mapping[str, Any], environment: Mapping[str, str], gcp_project: str,
                    credentials_factory: Callable[[], Any], project_root: Path, motion: Mapping[str, Any],
                    interval_seconds: float) -> dict[str, FrameLabeller]:
    prompt = get_prompt(inference["frame_prompt"])
    if prompt.response_schema is not EGO_FRAME_SCHEMA:
        raise InferenceError(f"'{inference['frame_prompt']}' is not a frame-labelling prompt.")
    labellers: dict[str, FrameLabeller] = {}
    for key, spec in models.items():
        concurrency = int(spec.get("concurrency", 1))
        if spec["provider"] == "mediapipe":
            path = Path(mediapipe["model_path"])
            labellers[key] = MediaPipeHandsLabeller(key, concurrency, path if path.is_absolute() else project_root / path,
                                                    mediapipe)
            continue
        if spec["provider"] == "motion":
            labellers[key] = MotionIdleLabeller(key, concurrency, motion, interval_seconds)
            continue
        provider = build_provider(spec, inference=inference, environment=environment, gcp_project=gcp_project,
                                  credentials_factory=credentials_factory)
        labellers[key] = VlmLabeller(key, spec["provides"], concurrency, provider, prompt, inference["temperature"])
    return labellers


def preflight(labellers: Mapping[str, FrameLabeller], probe_image: bytes) -> dict[str, Any]:
    """Prove every model answers (a model listing can succeed while predictions are denied)."""
    info: dict[str, Any] = {}
    for key, labeller in labellers.items():
        if not isinstance(labeller, VlmLabeller):
            info[key] = labeller.describe()
            continue
        model_info = labeller.provider.model_info()
        result = labeller.ask(labeller.prompt.system, labeller.prompt.user_prompt(timestamp_seconds=0.0), [probe_image],
                              labeller.prompt.response_schema)
        if result["status"] == "failed":
            raise InferenceError(f"Model '{key}' ({labeller.provider.describe()}) cannot answer: {result['error']}")
        if result["status"] != STATUS_OK:
            logger.warning("Model '%s' preflight answer was invalid (%s); continuing.", key, result.get("error"))
        info[key] = {**labeller.describe(), "model_digest": model_info.digest, "runtime": model_info.runtime_version}
    return info


# ---------------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------------

def _read_cache(path: Path) -> dict[int, dict[str, Any]]:
    cached: dict[int, dict[str, Any]] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue                       # a line cut by a crash
                cached[int(row["sample"])] = row
    return cached


def label_samples(labeller: FrameLabeller, samples: Sequence[Sample], cache_path: Path, *,
                  max_failure_ratio: float, progress_every: int = 200) -> dict[int, dict[str, Any]]:
    cached = _read_cache(cache_path)
    fingerprint = labeller.fingerprint()
    results = {i: r for i, r in cached.items() if r.get("status") == STATUS_OK and r.get("labeller") == fingerprint}
    todo = [s for s in samples if s.index not in results]
    logger.info("Model '%s': %d samples cached, %d to label (concurrency %d).", labeller.key, len(results), len(todo),
                labeller.concurrency)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    done = 0
    with cache_path.open("a", encoding="utf-8", newline="\n") as handle:
        def run(sample: Sample) -> dict[str, Any]:
            try:
                record = labeller.label(sample)
            except Exception as exc:  # noqa: BLE001 (a crash in one sample must not lose the others)
                record = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:500]}
            return {"sample": sample.index, "frame": sample.frame, "t_s": round(sample.t_s, 3), "labeller": fingerprint,
                    **record}

        with ThreadPoolExecutor(max_workers=max(1, labeller.concurrency)) as pool:
            futures = [pool.submit(run, s) for s in todo]
            for future in as_completed(futures):
                row = future.result()
                with lock:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    results[row["sample"]] = row
                    done += 1
                    if done % progress_every == 0:
                        logger.info("Model '%s': %d/%d labelled", labeller.key, done, len(todo))
    failed = [i for i, r in results.items() if r.get("status") != STATUS_OK]
    ratio = len(failed) / len(samples) if samples else 0.0
    if ratio > max_failure_ratio:
        examples = sorted({results[i].get("error", "")[:160] for i in failed})[:3]
        raise InferenceError(f"Model '{labeller.key}': {len(failed)}/{len(samples)} samples failed or invalid "
                             f"({ratio:.1%} > {max_failure_ratio:.1%}). Re-run to retry them. Examples: {examples}")
    return results
