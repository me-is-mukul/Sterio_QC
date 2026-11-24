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

    def render(self, **values: object) -> str:
        return self.user_template.format(**values)


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

# --- Egocentric QA (QA sheet sections 3 and 4) ---------------------------------------------------

EGO_FRAME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hands": {"type": "string", "enum": ["both", "one", "none"]},
        "activity": {"type": "string", "enum": ["working", "idle"]},
        "phone_use": {"type": "boolean"},
        "minor_visible": {"type": "boolean"},
        "third_person_or_fixed_camera": {"type": "boolean"},
        "lens_covered_or_smudged": {"type": "boolean"},
        "desk_or_computer_work": {"type": "boolean"},
        "private_moment": {"type": "boolean"},
        "description": {"type": "string"},
    },
    "required": ["hands", "activity", "phone_use", "minor_visible", "third_person_or_fixed_camera",
                 "lens_covered_or_smudged", "desk_or_computer_work", "private_moment", "description"],
}

_EGO_FRAME_SYSTEM = """You review frames from egocentric video: a camera worn on the head of a worker (the "wearer").
The image is one frame from the wearer's LEFT eye camera. It may show strong fisheye distortion.

Answer each field about THIS frame only:
- hands: how many of the WEARER's own hands are visible: "both", "one" or "none". The wearer's hands
  enter from the bottom or sides of the frame, close to the camera. Other people's hands do not count.
- activity: "working" if the wearer is actively doing a manual task (handling tools, materials,
  parts or objects as part of work). "idle" otherwise: resting, waiting, walking without carrying
  work, looking around, talking, or using a phone. Phone use is ALWAYS "idle".
- phone_use: the wearer is using a mobile phone.
- minor_visible: a child or teenager (anyone who may be under 18) is visible.
- third_person_or_fixed_camera: the view is NOT from a head-worn camera (e.g. a camera standing on a
  table or tripod, or filming the worker from outside).
- lens_covered_or_smudged: the lens is covered, blocked, or so smudged/fogged that the scene is unclear.
- desk_or_computer_work: the wearer works at a desk, on a computer, laptop, tablet or paperwork.
- private_moment: a bathroom/toilet is visible, or the wearer is using a phone for personal matters.
- description: one short factual sentence.

Respond ONLY with JSON matching the requested schema."""

_EGO_FRAME_USER = """Frame at {timestamp} s of the recording (left eye). Return the JSON."""

# v2: the definitions of the reference QA (its cut reasons: Idle (no work in hand), Wearer's hands not
# in view, Phone use, Person under 18 in view, Private moment, Lens blocked or out of focus, Headset not
# worn). Matched to its decisions on robocap SDC482 (01-11-2026): walking with a duster whose handle
# enters the frame, the hand itself just below it, is kept as work with hands in view.
_EGO_FRAME_SYSTEM_V2 = """You review frames from egocentric video: a camera worn on the head of a worker (the "wearer").
The image is one frame from the wearer's LEFT eye camera. It may show strong fisheye distortion.
Footage is kept for training robots on manual work; your answers decide which seconds are cut.

Answer each field about THIS frame only:
- hands: how many of the WEARER's own hands are in view: "both", "one" or "none". The wearer's hands
  enter from the bottom or sides of the frame, close to the camera. Count a hand that is only partly
  in view (fingers or part of a hand at the edge, a gloved hand). Also answer at least "one" when a tool
  or object the wearer is clearly holding enters from the bottom or side edge (e.g. a broom or duster
  handle), even if the hand itself is just outside the frame. "none" = nothing of the wearer's hands and
  nothing they hold is in view. Other people's hands do not count.
- activity: "working" if the wearer has work in hand: handling, using, holding or carrying a tool,
  material or object as part of a task, including carrying a work tool while moving between spots.
  "idle" = no work in hand: empty-handed walking or standing, waiting, resting, eating, talking,
  looking around, or using a phone. Phone use is ALWAYS "idle".
- phone_use: the wearer is using or holding up a mobile phone (screen in view, typing, calling).
- minor_visible: a child or teenager (anyone who may be under 18) is visible.
- third_person_or_fixed_camera: the headset is NOT worn: the camera lies on a table or the floor,
  is held in the hand, or films the worker from outside.
- lens_covered_or_smudged: the lens is blocked (finger, cloth, cap), or the picture is so out of focus,
  fogged or smudged that the scene cannot be made out.
- desk_or_computer_work: the wearer works at a desk, on a computer, laptop, tablet or paperwork.
- private_moment: a bathroom/toilet is visible, or someone is undressing or changing clothes.
- description: one short factual sentence.

Respond ONLY with JSON matching the requested schema."""

EGO_RECORDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "real_work": {"type": "boolean"},
        "no_work_reason": {"type": "string",
                           "enum": ["none", "breaks", "desk_or_computer_work", "staged_or_pretend_work", "other"]},
        "explanation": {"type": "string"},
    },
    "required": ["real_work", "no_work_reason", "explanation"],
}

_EGO_RECORDING_SYSTEM = """You review a whole egocentric recording (head-worn camera) from {count} frames spread
evenly over it, in time order. Decide whether the recording shows REAL manual work by the wearer.

real_work = false when, over the recording as a whole, it is mainly:
- "breaks": resting, eating, chatting, waiting;
- "desk_or_computer_work": desk, computer, laptop, tablet or paperwork;
- "staged_or_pretend_work": the task is acted out, e.g. miming with no real material, moving the
  same object back and forth for show, or "work" that produces nothing;
- "other": anything else that is not real manual work.
Otherwise real_work = true and no_work_reason = "none". Short idle moments inside real work are normal.
Respond ONLY with JSON matching the requested schema."""

_EGO_RECORDING_USER = """{count} frames from a recording of {duration} s. Frame times (s): {times}.
Return the JSON."""

EGO_CLIP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "trivial_repeated_motion": {"type": "boolean"},
        "explanation": {"type": "string"},
    },
    "required": ["trivial_repeated_motion", "explanation"],
}

_EGO_CLIP_SYSTEM = """You review one clip of egocentric video (head-worn camera) from {count} frames spread evenly
over it, in time order.

trivial_repeated_motion = true only if the WHOLE clip is one trivial motion repeated over and over with
no task progress (e.g. tapping, wiping the same spot, opening and closing the same thing, moving one
object back and forth). Repetitive steps of real work that make progress (assembling several parts,
sewing a seam, cutting many pieces) are NOT trivial.
Respond ONLY with JSON matching the requested schema."""

_EGO_CLIP_USER = """{count} frames from a clip of {duration} s. Frame times (s): {times}. Return the JSON."""

PROMPTS: dict[str, PromptTemplate] = {
    "hoi-stereo-v1": PromptTemplate("hoi-stereo-v1", _V1_SYSTEM, _V1_USER, HOI_RESPONSE_SCHEMA),
    "ego-qa-frame-v1": PromptTemplate("ego-qa-frame-v1", _EGO_FRAME_SYSTEM, _EGO_FRAME_USER, EGO_FRAME_SCHEMA),
    "ego-qa-frame-v2": PromptTemplate("ego-qa-frame-v2", _EGO_FRAME_SYSTEM_V2, _EGO_FRAME_USER, EGO_FRAME_SCHEMA),
    "ego-qa-recording-v1": PromptTemplate("ego-qa-recording-v1", _EGO_RECORDING_SYSTEM, _EGO_RECORDING_USER,
                                          EGO_RECORDING_SCHEMA),
    "ego-qa-clip-v1": PromptTemplate("ego-qa-clip-v1", _EGO_CLIP_SYSTEM, _EGO_CLIP_USER, EGO_CLIP_SCHEMA),
}


def get_prompt(version: str) -> PromptTemplate:
    try:
        return PROMPTS[version]
    except KeyError:
        raise InferenceError(f"Unknown prompt version '{version}'. Known: {sorted(PROMPTS)}") from None
