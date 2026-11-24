"""Configuration of the egocentric QA flow (configs/ego_qa.yaml).

Separate from the MP4 pipeline's configuration (core/config.py) on purpose: the two flows
share helpers but not settings. Layers: configs/ego_qa.yaml -> --config file (strict: only
existing keys) -> CLI overrides.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from video_pipeline.core.config import (
    CONFIG_DIRECTORY_NAME,
    ResolvedConfig,
    _get_dotted,
    _set_dotted,
    deep_merge,
    load_yaml_mapping,
)
from video_pipeline.core.errors import ConfigError
from video_pipeline.core.gcp_auth import CREDENTIAL_MODES

EGO_CONFIG_FILE = "ego_qa.yaml"
PROVIDERS = frozenset({"gemini", "ollama", "openai_compatible", "env", "mediapipe", "motion"})
LOCAL_PROVIDERS = frozenset({"mediapipe", "motion"})     # run on this machine, no AI service
MODEL_ROLES = frozenset({"hands", "activity", "content"})
ON_EXISTING = frozenset({"skip", "overwrite", "new_run_folder"})

_N = (int, float)
_SCHEMA: dict[str, tuple[type, ...]] = {
    "gcp.project": (str,),
    "gcp.credentials": (str,),
    "source.bucket": (str,),
    "source.raw_root": (str,),
    "source.processed_root": (str,),
    "source.processed_bucket": (str, type(None)),
    "source.session_json": (str,),
    "source.segment_pattern": (str,),
    "source.read_chunk_mb": _N,
    "upload.enabled": (bool,),
    "upload.on_existing": (str,),
    "work.directory": (str,),
    "work.delete_local_files_after_upload": (bool,),
    "tools.ffmpeg": (str,),
    "tools.ffprobe": (str,),
    "topics.cameras": (list,),
    "topics.imu_raw": (str,),
    "topics.intrinsics": (list,),
    "topics.stereo_extrinsics": (str,),
    "topics.imu_info": (str,),
    "topics.imu_extrinsics": (str,),
    "topics.video_format": (str,),
    "technical.expected_fps": _N,
    "technical.fps_tolerance": _N,
    "technical.max_frame_gap_seconds": _N,
    "technical.max_eye_offset_ms": _N,
    "technical.max_unpaired_edge_frames": (int,),
    "technical.imu_max_gap_seconds": _N,
    "technical.imu_coverage_tolerance_seconds": _N,
    "technical.imu_same_clock_max_offset_seconds": _N,
    "technical.imu_rate_window_seconds": _N,
    "technical.imu_rate_tolerance": _N,
    "technical.black_min_seconds": _N,
    "technical.black_pixel_threshold": _N,
    "technical.black_picture_threshold": _N,
    "technical.freeze_noise_db": _N,
    "technical.freeze_min_seconds": _N,
    "technical.max_defect_fraction": _N,
    "technical.decode_timeout_seconds": _N,
    "technical.min_recording_seconds": _N,
    "calibration.pairs": (int,),
    "calibration.max_median_vertical_px": _N,
    "calibration.max_p95_vertical_px": _N,
    "calibration.min_inlier_matches": (int,),
    "calibration.min_pairs_with_matches": (int,),
    "calibration.min_matches_per_pair": (int,),
    "calibration.ratio_test": _N,
    "calibration.ransac_threshold_px": _N,
    "calibration.eye_order_min_agreement": _N,
    "calibration.eye_order_min_points": (int,),
    "calibration.eye_order_min_disparity_px": _N,
    "calibration.labelled_left": (str,),
    "labelling.interval_seconds": _N,
    "labelling.eye": (str,),
    "labelling.image_max_width": (int,),
    "labelling.jpeg_quality": (int,),
    "labelling.strip_image_width": (int,),
    "labelling.max_failure_ratio": _N,
    "models": (dict,),
    "inference.timeout_seconds": _N,
    "inference.max_retries": (int,),
    "inference.retry_backoff_seconds": _N,
    "inference.temperature": _N,
    "inference.max_output_tokens": (int,),
    "inference.think": (bool, type(None)),
    "inference.frame_prompt": (str,),
    "inference.recording_prompt": (str,),
    "inference.clip_prompt": (str,),
    "mediapipe": (dict,),
    "motion.image_width": (int,),
    "motion.pixel_threshold": _N,
    "motion.max_changed_fraction": _N,
    "motion.max_gyro_rad_s": _N,
    "content.enabled": (bool,),
    "content.minor_min_frames": (int,),
    "content.third_person_max_share": _N,
    "content.lens_max_share": _N,
    "content.recording_strip_frames": (int,),
    "content.private_min_frames": (int,),
    "content.clip_strip_frames": (int,),
    "cropping.idle_cut_seconds": _N,
    "cropping.no_hands_cut_seconds": _N,
    "cropping.min_models_per_decision": (int,),
    "cropping.cut_defects": (bool,),
    "cropping.keyframe_snap_max_seconds": _N,
    "cropping.min_piece_seconds": _N,
    "logging.console_level": (str,),
    "logging.file_level": (str,),
}

# Config keys each step's outputs depend on (a change re-runs that step and every later one).
STEP_CONFIG_KEYS: dict[str, tuple[str, ...]] = {
    "extract": ("topics",),
    "technical": ("technical", "labelling.interval_seconds", "labelling.image_max_width", "labelling.jpeg_quality",
                  "calibration.pairs"),
    "calibration": ("calibration",),
    "labelling": ("labelling", "models", "inference.frame_prompt", "inference.temperature", "mediapipe", "motion"),
    "content_recording": ("content", "inference.recording_prompt", "inference.temperature"),
    "cropping": ("cropping", "technical.expected_fps"),
    "content_clips": ("content", "inference.clip_prompt", "inference.temperature"),
    "clips": ("cropping",),
}


def _type_ok(value: Any, accepted: tuple[type, ...]) -> bool:
    if isinstance(value, bool) and bool not in accepted:
        return False
    return isinstance(value, accepted)


def validate_ego_config(data: Mapping[str, Any]) -> None:
    problems: list[str] = []
    for dotted, accepted in _SCHEMA.items():
        try:
            value = _get_dotted(data, dotted)
        except KeyError:
            problems.append(f"missing required key '{dotted}'")
            continue
        if not _type_ok(value, accepted):
            problems.append(f"'{dotted}' must be {' or '.join(t.__name__ for t in accepted)}, got {type(value).__name__}")
    if problems:
        raise ConfigError("Invalid ego QA configuration:\n  - " + "\n  - ".join(problems))

    get = lambda key: _get_dotted(data, key)  # noqa: E731

    def check(ok: bool, message: str) -> None:
        if not ok:
            problems.append(message)

    check(get("gcp.credentials") in CREDENTIAL_MODES, f"'gcp.credentials' must be one of {sorted(CREDENTIAL_MODES)}")
    check(get("upload.on_existing") in ON_EXISTING, f"'upload.on_existing' must be one of {sorted(ON_EXISTING)}")
    for key in ("source.raw_root", "source.processed_root"):
        value = get(key)
        check(value.endswith("/") and not value.startswith("/"), f"'{key}' must end with '/' and not start with '/'")
    raw, out = get("source.raw_root"), get("source.processed_root")
    if processed_bucket(data) == get("source.bucket"):
        check(not raw.startswith(out) and not out.startswith(raw),
              f"the destination gs://{get('source.bucket')}/{out} and the source gs://{get('source.bucket')}/{raw} "
              "must not contain each other")
    check(processed_bucket(data) != "", "'source.processed_bucket' must not be empty (null = same as source.bucket)")
    check(1 <= get("source.read_chunk_mb") <= 256, "'source.read_chunk_mb' must be 1..256")
    try:
        pattern = re.compile(get("source.segment_pattern"))
        check(pattern.groups == 1, "'source.segment_pattern' must have exactly one group (the segment number)")
    except re.error as exc:
        problems.append(f"'source.segment_pattern' is not a valid regular expression: {exc}")
    check(len(get("topics.cameras")) == 2, "'topics.cameras' must list exactly two topics (cam0, cam1)")
    check(len(get("topics.intrinsics")) == 2, "'topics.intrinsics' must list exactly two topics (cam0, cam1)")
    check(get("topics.video_format") == "h265", "'topics.video_format' currently supports only 'h265'")
    for key in ("expected_fps", "fps_tolerance", "max_frame_gap_seconds", "max_eye_offset_ms", "imu_max_gap_seconds",
                "imu_rate_window_seconds", "black_min_seconds", "freeze_min_seconds", "decode_timeout_seconds"):
        check(get(f"technical.{key}") > 0, f"'technical.{key}' must be > 0")
    check(get("technical.max_eye_offset_ms") * get("technical.expected_fps") < 500,
          "'technical.max_eye_offset_ms' must be under half a frame interval (else neighbouring frames would pair)")
    check(0 <= get("technical.max_unpaired_edge_frames") <= 30, "'technical.max_unpaired_edge_frames' must be 0..30")
    check(0 < get("technical.max_defect_fraction") <= 1, "'technical.max_defect_fraction' must be in (0, 1]")
    check(0 < get("technical.imu_rate_tolerance") < 1, "'technical.imu_rate_tolerance' must be in (0, 1)")
    check(0 <= get("technical.black_pixel_threshold") <= 1, "'technical.black_pixel_threshold' must be 0..1")
    check(0 <= get("technical.black_picture_threshold") <= 1, "'technical.black_picture_threshold' must be 0..1")
    check(get("technical.freeze_noise_db") < 0, "'technical.freeze_noise_db' must be negative (dB)")
    check(get("calibration.pairs") >= 2, "'calibration.pairs' must be >= 2")
    check(0 < get("calibration.ratio_test") < 1, "'calibration.ratio_test' must be in (0, 1)")
    check(0.5 < get("calibration.eye_order_min_agreement") <= 1, "'calibration.eye_order_min_agreement' must be in (0.5, 1]")
    check(get("calibration.labelled_left") in {"cam0", "cam1", "calibration"},
          "'calibration.labelled_left' must be calibration, cam0 or cam1")
    fps = get("technical.expected_fps")
    step = get("labelling.interval_seconds") * fps
    check(get("labelling.interval_seconds") > 0 and abs(step - round(step)) < 1e-9 and round(step) >= 1,
          "'labelling.interval_seconds' x 'technical.expected_fps' must be a whole number of frames")
    check(get("labelling.eye") == "left", "'labelling.eye' currently supports only 'left' (as in the QA sheet)")
    check(1 <= get("labelling.jpeg_quality") <= 100, "'labelling.jpeg_quality' must be 1..100")
    check(get("labelling.image_max_width") >= 64, "'labelling.image_max_width' must be >= 64")
    check(get("labelling.strip_image_width") >= 64, "'labelling.strip_image_width' must be >= 64")
    check(0 <= get("labelling.max_failure_ratio") < 1, "'labelling.max_failure_ratio' must be in [0, 1)")
    check(get("inference.timeout_seconds") > 0, "'inference.timeout_seconds' must be > 0")
    check(get("inference.max_retries") >= 0, "'inference.max_retries' must be >= 0")
    check(get("inference.max_output_tokens") >= 64, "'inference.max_output_tokens' must be >= 64")
    for key in ("minor_min_frames", "private_min_frames"):
        check(get(f"content.{key}") >= 1, f"'content.{key}' must be >= 1")
    for key in ("recording_strip_frames", "clip_strip_frames"):
        check(2 <= get(f"content.{key}") <= 32, f"'content.{key}' must be 2..32")
    for key in ("third_person_max_share", "lens_max_share"):
        check(0 < get(f"content.{key}") <= 1, f"'content.{key}' must be in (0, 1]")
    for key in ("idle_cut_seconds", "no_hands_cut_seconds", "min_piece_seconds", "keyframe_snap_max_seconds"):
        check(get(f"cropping.{key}") > 0, f"'cropping.{key}' must be > 0")
    check(get("cropping.min_models_per_decision") >= 1, "'cropping.min_models_per_decision' must be >= 1")
    check(16 <= get("motion.image_width") <= 1280, "'motion.image_width' must be 16..1280")
    check(0 < get("motion.pixel_threshold") < 255, "'motion.pixel_threshold' must be in (0, 255)")
    check(0 <= get("motion.max_changed_fraction") < 1, "'motion.max_changed_fraction' must be in [0, 1)")
    check(get("motion.max_gyro_rad_s") > 0, "'motion.max_gyro_rad_s' must be > 0")

    models = get("models")
    check(bool(models), "'models' must define at least one model")
    content_models = []
    providers_of: dict[str, int] = {"hands": 0, "activity": 0}
    for key, spec in (models or {}).items():
        if not isinstance(spec, Mapping):
            problems.append(f"'models.{key}' must be a mapping")
            continue
        provider = spec.get("provider")
        check(provider in PROVIDERS, f"'models.{key}.provider' must be one of {sorted(PROVIDERS)}")
        provides = spec.get("provides")
        if not isinstance(provides, list) or not provides or not set(provides) <= MODEL_ROLES:
            problems.append(f"'models.{key}.provides' must be a non-empty list of {sorted(MODEL_ROLES)}")
            continue
        concurrency = spec.get("concurrency", 1)
        check(isinstance(concurrency, int) and not isinstance(concurrency, bool) and 1 <= concurrency <= 64,
              f"'models.{key}.concurrency' must be 1..64")
        if provider == "mediapipe":
            check(set(provides) == {"hands"}, f"'models.{key}': the mediapipe provider can only provide 'hands'")
        if provider == "motion":
            check(set(provides) == {"activity"}, f"'models.{key}': the motion provider can only provide 'activity'")
        if provider == "gemini":
            check(isinstance(spec.get("model"), str) and bool(spec.get("model")), f"'models.{key}.model' is required for gemini")
            check(isinstance(spec.get("location"), str) and bool(spec.get("location")),
                  f"'models.{key}.location' is required for gemini")
        if "content" in provides:
            content_models.append(key)
            check(provider not in LOCAL_PROVIDERS, f"'models.{key}': the content model must be a vision-language model")
        for role in providers_of:
            if role in provides:
                providers_of[role] += 1
    if get("content.enabled"):
        check(len(content_models) == 1,
              f"the content screen is switched on (content.enabled / --content-screen on), so exactly one model must "
              f"provide 'content' (found {content_models}). See docs/ego_qa.md, 'Turning on the content screen'.")
    else:
        check(not content_models, f"content.enabled is false, so no model may provide 'content' (found {content_models})")
    minimum = get("cropping.min_models_per_decision")
    for role, count in providers_of.items():
        check(count >= minimum, f"{count} model(s) provide '{role}' but 'cropping.min_models_per_decision' is {minimum}")
    if problems:
        raise ConfigError("Invalid ego QA configuration:\n  - " + "\n  - ".join(problems))


def processed_bucket(data: Any) -> str:
    """Bucket the results are written to (source.processed_bucket; null = the source bucket)."""
    get = data.get if isinstance(data, ResolvedConfig) else (lambda key: _get_dotted(data, key))
    value = get("source.processed_bucket")
    return get("source.bucket") if value is None else value


def load_ego_configuration(project_root: Path, user_config_path: Path | None = None,
                           cli_overrides: Mapping[str, Any] | None = None) -> ResolvedConfig:
    base_path = project_root / CONFIG_DIRECTORY_NAME / EGO_CONFIG_FILE
    if not base_path.is_file():
        raise ConfigError(f"Configuration file is missing: '{base_path}'.")
    merged = load_yaml_mapping(base_path)
    layers = [str(base_path)]
    if user_config_path is not None:
        user_layer = load_yaml_mapping(user_config_path)
        # `models` is replaced as a whole, so a user file can drop or add a model.
        user_models = user_layer.pop("models", None)
        merged = deep_merge(merged, user_layer, strict=True)
        if user_models is not None:
            if not isinstance(user_models, Mapping):
                raise ConfigError("'models' in the user configuration must be a mapping.")
            merged["models"] = dict(user_models)
        layers.append(str(user_config_path.resolve()))
    overrides = dict(cli_overrides or {})
    for dotted, value in overrides.items():
        _set_dotted(merged, dotted, value)
    if overrides:
        layers.append("cli")
    validate_ego_config(merged)
    return ResolvedConfig(data=merged, layers=tuple(layers), cli_overrides=overrides)
