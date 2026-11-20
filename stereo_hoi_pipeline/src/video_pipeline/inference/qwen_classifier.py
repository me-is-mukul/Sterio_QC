"""Stereo HOI classification on top of any InferenceProvider.

Model output is treated as untrusted: it is parsed, validated against the schema
and normalised. Anything invalid becomes UNCERTAIN with the reasons recorded.
The raw text is always kept by the caller. Model-stated confidence is not used.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from video_pipeline.inference.base import InferenceProvider, ProviderRequest, ProviderResponse
from video_pipeline.inference.prompts import PromptTemplate

RELEVANT = "RELEVANT"
WASTE = "WASTE"
UNCERTAIN = "UNCERTAIN"
UNAVAILABLE = "UNAVAILABLE"   # inference failed and failure_policy == "unavailable"
MODEL_LABELS = (RELEVANT, WASTE, UNCERTAIN)

STATUS_OK = "ok"
STATUS_INVALID = "invalid_response"
STATUS_FAILED = "failed"

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class CameraObservation:
    hands_visible: bool | None
    interaction_detected: bool | None
    description: str


@dataclass(frozen=True)
class HoiClassification:
    classification: str
    hoi_description: str
    evidence: tuple[str, ...]
    left_camera: CameraObservation | None
    right_camera: CameraObservation | None
    status: str
    validation_errors: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = list(self.evidence)
        data["validation_errors"] = list(self.validation_errors)
        return data


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from model text, tolerating code fences or surrounding prose."""
    candidates = [text.strip()]
    candidates += [m.strip() for m in _FENCE.findall(text)]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _camera(value: Any, name: str, errors: list[str]) -> CameraObservation | None:
    if not isinstance(value, dict):
        errors.append(f"{name} missing or not an object")
        return None

    def as_bool(key: str) -> bool | None:
        v = value.get(key)
        if isinstance(v, bool):
            return v
        errors.append(f"{name}.{key} is not a boolean")
        return None

    description = value.get("description")
    if not isinstance(description, str):
        errors.append(f"{name}.description is not a string")
        description = ""
    return CameraObservation(as_bool("hands_visible"), as_bool("interaction_detected"), description.strip())


def normalize_response(text: str) -> HoiClassification:
    data = extract_json_object(text)
    if data is None:
        return HoiClassification(UNCERTAIN, "", (), None, None, STATUS_INVALID, ("response is not a JSON object",))

    errors: list[str] = []
    label = data.get("classification")
    label = label.strip().upper() if isinstance(label, str) else None
    if label not in MODEL_LABELS:
        errors.append(f"invalid classification {data.get('classification')!r}")
        label = None

    description = data.get("hoi_description")
    if not isinstance(description, str):
        errors.append("hoi_description is not a string")
        description = ""
    evidence_raw = data.get("evidence", [])
    if not isinstance(evidence_raw, list):
        errors.append("evidence is not a list")
        evidence_raw = []
    evidence = tuple(str(e).strip() for e in evidence_raw if str(e).strip())

    left = _camera(data.get("left_camera"), "left_camera", errors)
    right = _camera(data.get("right_camera"), "right_camera", errors)

    if label is None:
        # Without a valid label the result is unusable for segmentation: never guess.
        return HoiClassification(UNCERTAIN, description.strip(), evidence, left, right, STATUS_INVALID, tuple(errors))
    # Minor field problems don't invalidate a valid label; they are recorded.
    return HoiClassification(label, description.strip(), evidence, left, right, STATUS_OK, tuple(errors))


def failed_classification(error: str, failure_policy: str) -> HoiClassification:
    label = UNAVAILABLE if failure_policy == "unavailable" else UNCERTAIN
    return HoiClassification(label, "", (), None, None, STATUS_FAILED, (error,))


class StereoHoiClassifier:
    def __init__(self, provider: InferenceProvider, prompt: PromptTemplate, *, temperature: float) -> None:
        self._provider = provider
        self._prompt = prompt
        self._temperature = temperature

    @property
    def prompt_version(self) -> str:
        return self._prompt.version

    def classify(self, left_jpeg: bytes, right_jpeg: bytes, *, timestamp_seconds: float
                 ) -> tuple[HoiClassification, ProviderResponse]:
        request = ProviderRequest(
            system_prompt=self._prompt.system,
            user_prompt=self._prompt.user_prompt(timestamp_seconds=timestamp_seconds),
            images=(left_jpeg, right_jpeg),
            response_schema=self._prompt.response_schema,
            temperature=self._temperature,
        )
        response = self._provider.generate(request)
        return normalize_response(response.text), response
