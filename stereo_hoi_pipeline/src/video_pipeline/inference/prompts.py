"""Versioned prompts for paired stereo HOI classification.

A prompt version is immutable once used: change the text => add a new version.
The version is part of the inference cache key and of every stored result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from video_pipeline.core.errors import InferenceError

_CAMERA_SCHEMA = {
    "type": "object",
    "properties": {
        "hands_visible": {"type": "boolean"},
        "interaction_detected": {"type": "boolean"},
        "description": {"type": "string"},
    },
    "required": ["hands_visible", "interaction_detected", "description"],
}

HOI_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "left_camera": _CAMERA_SCHEMA,
        "right_camera": _CAMERA_SCHEMA,
        "classification": {"type": "string", "enum": ["RELEVANT", "WASTE", "UNCERTAIN"]},
        "hoi_description": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["left_camera", "right_camera", "classification", "hoi_description", "evidence"],
}


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    system: str
    user_template: str
    response_schema: dict[str, Any]

    def user_prompt(self, *, timestamp_seconds: float) -> str:
        return self.user_template.format(timestamp=f"{timestamp_seconds:.3f}")


_V1_SYSTEM = """You analyse egocentric (first-person, head-mounted) stereo video for Hand-Object Interaction (HOI).

You receive TWO images captured at the SAME instant:
- Image 1 is the LEFT camera.
- Image 2 is the RIGHT camera.
They are two viewpoints of ONE physical scene. Combine evidence from both views.
The same hand or interaction seen in both images is ONE interaction - never count it twice.

Classify the moment as exactly one of:
- RELEVANT: hands are interacting with objects in a task-related way: manipulating, holding,
  using tools, assembling, operating machinery, transferring objects. A stationary hand that
  is holding or touching an object still counts as RELEVANT. Lack of movement is NOT waste.
- WASTE: no relevant hand-object interaction: empty scene, hands absent or idle, camera pointed
  away from the activity, walking/transition footage, irrelevant surroundings.
- UNCERTAIN: you cannot decide confidently: hands obscured, severe motion blur, poor lighting,
  lens obstruction, or ambiguous evidence.

When in doubt between WASTE and anything else, prefer UNCERTAIN.
Respond ONLY with JSON matching the requested schema. Keep descriptions short and factual."""

_V1_USER = """Timestamp: {timestamp} s.
Image 1 = LEFT camera, Image 2 = RIGHT camera, same instant.
Describe what the hands do in each view, then give one combined classification.
Return JSON with keys: left_camera {{hands_visible, interaction_detected, description}},
right_camera {{hands_visible, interaction_detected, description}}, classification
(RELEVANT | WASTE | UNCERTAIN), hoi_description, evidence (list of short strings)."""

PROMPTS: dict[str, PromptTemplate] = {
    "hoi-stereo-v1": PromptTemplate("hoi-stereo-v1", _V1_SYSTEM, _V1_USER, HOI_RESPONSE_SCHEMA),
}


def get_prompt(version: str) -> PromptTemplate:
    try:
        return PROMPTS[version]
    except KeyError:
        raise InferenceError(f"Unknown prompt version '{version}'. Known: {sorted(PROMPTS)}") from None
