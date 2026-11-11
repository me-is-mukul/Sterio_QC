"""Layered YAML configuration loading and validation.

Precedence (highest first):
    1. CLI arguments
    2. Environment variables / project .env (INFERENCE_PROVIDER + that provider's variables)
    3. User-provided configuration file (--config)
    4. Default configuration (configs/default.yaml, inference.yaml, processing.yaml)

This module only loads, merges and validates configuration. It contains no
pipeline business logic.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from video_pipeline.core.errors import ConfigError
from video_pipeline.core.hashing import sha256_json

CONFIG_DIRECTORY_NAME = "configs"
DEFAULT_LAYER_FILES: tuple[str, ...] = ("default.yaml", "inference.yaml", "processing.yaml")

_MISSING = object()

# Sections excluded from the configuration fingerprint because they never change outputs.
FINGERPRINT_EXCLUDED_SECTIONS: frozenset[str] = frozenset({"logging", "web"})

# Environment variables -> dotted configuration keys. INFERENCE_PROVIDER selects the
# provider; then only that provider's variables are applied. Secrets (OLLAMA_API_KEY,
# VLLM_API_KEY) are deliberately absent: the provider factory reads them, nothing stores them.
ENV_PROVIDER_VARIABLE = "INFERENCE_PROVIDER"
ENV_OVERRIDES_BY_PROVIDER: dict[str, dict[str, str]] = {
    "ollama": {"OLLAMA_BASE_URL": "inference.base_url", "OLLAMA_MODEL": "inference.model"},
    "openai_compatible": {"VLLM_BASE_URL": "inference.base_url", "VLLM_MODEL": "inference.model"},
}
INFERENCE_PROVIDERS = frozenset(ENV_OVERRIDES_BY_PROVIDER)

_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

_NUMBER = (int, float)
_NONE = type(None)

# Dotted key -> accepted Python types. Every key listed here must exist after merging.
_SCHEMA: dict[str, tuple[type, ...]] = {
    "project.name": (str,),
    "project.output_directory": (str,),
    "project.preserve_intermediate_files": (bool,),
    "video.download_directory": (str,),
    "video.supported_formats": (list,),
    "video.validate_before_processing": (bool,),
    "download.timeout_seconds": _NUMBER,
    "download.max_retries": (int,),
    "download.retry_backoff_seconds": _NUMBER,
    "download.chunk_size_bytes": (int,),
    "download.progress_log_percent_step": (int,),
    "download.user_agent": (str,),
    "tools.ffmpeg": (str,),
    "tools.ffprobe": (str,),
    "tools.timeout_seconds": _NUMBER,
    "metadata.count_frames": (bool,),
    "hashing.chunk_size_bytes": (int,),
    "environment.probe_timeout_seconds": _NUMBER,
    "logging.console_level": (str,),
    "logging.file_level": (str,),
    "telemetry.parse_streams": (list,),
    "inference.provider": (str,),
    "inference.base_url": (str,),
    "inference.model": (str,),
    "inference.timeout_seconds": _NUMBER,
    "inference.max_retries": (int,),
    "inference.retry_backoff_seconds": _NUMBER,
    "inference.temperature": _NUMBER,
    "inference.think": (bool, _NONE),
    "inference.keep_alive": (str, _NONE),
    "inference.max_output_tokens": (int, _NONE),
    "inference.max_concurrency": (int,),
    "inference.prompt_version": (str,),
    "inference.failure_policy": (str,),
    "stereo.layout": (str,),
    "stereo.split_ratio": _NUMBER,
    "sampling.interval_seconds": _NUMBER,
    "sampling.image_format": (str,),
    "sampling.max_width": (int,),
    "sampling.jpeg_quality": (int,),
    "sampling.timeout_seconds": _NUMBER,
    "sampling.keep_frames": (bool,),
    "verification.enabled": (bool,),
    "verification.interval_seconds": _NUMBER,
    "verification.guard_seconds": _NUMBER,
    "qc.fail_run_on_failure": (bool,),
    "qc.min_alignment_psnr_db": _NUMBER,
    "qc.alignment_checkpoints_max": (int,),
    "qc.decode_check": (bool,),
    "qc.max_inference_problem_ratio": _NUMBER,
    "hoi.waste_threshold_seconds": (int,),
    "hoi.classifications": (list,),
    "hoi.uncertain_policy": (str,),
    "segmentation.preserve_stereo": (bool,),
    "segmentation.preserve_audio": (bool,),
    "segmentation.synchronize_imu": (bool,),
    "segmentation.video_encoder": (str,),
    "segmentation.video_encoder_args": (list,),
    "segmentation.pixel_format": (str,),
    "segmentation.audio_encoder": (str,),
    "segmentation.timeout_seconds": _NUMBER,
    "web.host": (str,),
    "web.port": (int,),
}


def find_project_root() -> Path:
    """Locate the project root (directory holding pyproject.toml and configs/default.yaml).

    Searches upward from this package first (src layout / editable install), then
    from the current working directory.
    """
    candidates = [*Path(__file__).resolve().parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / CONFIG_DIRECTORY_NAME / DEFAULT_LAYER_FILES[0]).is_file():
            return candidate
    raise ConfigError(
        "Could not locate the project root (a directory containing pyproject.toml and "
        f"{CONFIG_DIRECTORY_NAME}/{DEFAULT_LAYER_FILES[0]})."
    )


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read configuration file '{path}': {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in '{path}': {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"The top level of '{path}' must be a mapping.")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any], *, strict: bool, _prefix: str = "") -> dict[str, Any]:
    """Return a new dict with ``override`` merged into ``base``.

    With ``strict=True`` unknown keys are rejected so typos in user configuration
    fail loudly instead of being silently ignored.
    """
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        dotted = f"{_prefix}.{key}" if _prefix else str(key)
        if strict and key not in result:
            raise ConfigError(f"Unknown configuration key '{dotted}'.")
        existing = result.get(key, _MISSING)
        if isinstance(existing, Mapping):
            if not isinstance(value, Mapping):
                raise ConfigError(f"Configuration key '{dotted}' must be a mapping.")
            result[key] = deep_merge(existing, value, strict=strict, _prefix=dotted)
        else:
            if strict and isinstance(value, Mapping):
                raise ConfigError(f"Configuration key '{dotted}' must not be a mapping.")
            result[key] = copy.deepcopy(value)
    return result


def _get_dotted(data: Mapping[str, Any], dotted: str, default: Any = _MISSING) -> Any:
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            if default is _MISSING:
                raise KeyError(dotted)
            return default
        node = node[part]
    return node


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            raise ConfigError(f"Cannot apply override '{dotted}': '{part}' is not a configuration section.")
        node = child
    if parts[-1] not in node:
        raise ConfigError(f"Cannot apply override '{dotted}': unknown configuration key.")
    node[parts[-1]] = value


def _type_ok(value: Any, accepted: tuple[type, ...]) -> bool:
    # bool is a subclass of int; only accept it where bool is explicitly allowed.
    if isinstance(value, bool) and bool not in accepted:
        return False
    return isinstance(value, accepted)


def validate_config(data: Mapping[str, Any]) -> None:
    """Validate types and value ranges. Raises ConfigError describing every problem."""
    problems: list[str] = []
    for dotted, accepted in _SCHEMA.items():
        try:
            value = _get_dotted(data, dotted)
        except KeyError:
            problems.append(f"missing required key '{dotted}'")
            continue
        if not _type_ok(value, accepted):
            names = " or ".join(t.__name__ for t in accepted)
            problems.append(f"'{dotted}' must be of type {names}, got {type(value).__name__}")
    if problems:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))

    def check(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    get = lambda key: _get_dotted(data, key)  # noqa: E731
    check(get("hoi.waste_threshold_seconds") >= 1, "'hoi.waste_threshold_seconds' must be >= 1")
    check(get("hoi.uncertain_policy") == "preserve", "'hoi.uncertain_policy' currently supports only 'preserve'")
    check(get("download.max_retries") >= 0, "'download.max_retries' must be >= 0")
    check(get("download.timeout_seconds") > 0, "'download.timeout_seconds' must be > 0")
    check(get("download.chunk_size_bytes") > 0, "'download.chunk_size_bytes' must be > 0")
    check(1 <= get("download.progress_log_percent_step") <= 100, "'download.progress_log_percent_step' must be 1..100")
    check(get("hashing.chunk_size_bytes") > 0, "'hashing.chunk_size_bytes' must be > 0")
    check(get("tools.timeout_seconds") > 0, "'tools.timeout_seconds' must be > 0")
    check(get("environment.probe_timeout_seconds") > 0, "'environment.probe_timeout_seconds' must be > 0")
    check(bool(get("video.supported_formats")), "'video.supported_formats' must not be empty")
    check(
        all(isinstance(f, str) and f.strip() for f in get("video.supported_formats")),
        "'video.supported_formats' must contain non-empty strings",
    )
    check(get("logging.console_level").upper() in _LOG_LEVELS, f"'logging.console_level' must be one of {sorted(_LOG_LEVELS)}")
    check(get("logging.file_level").upper() in _LOG_LEVELS, f"'logging.file_level' must be one of {sorted(_LOG_LEVELS)}")
    check(get("stereo.layout") == "side_by_side", "'stereo.layout' currently supports only 'side_by_side'")
    check(0 < get("stereo.split_ratio") < 1, "'stereo.split_ratio' must be between 0 and 1 (exclusive)")
    check(get("sampling.interval_seconds") > 0, "'sampling.interval_seconds' must be > 0")
    check(get("sampling.max_width") > 0, "'sampling.max_width' must be > 0")
    check(1 <= get("sampling.jpeg_quality") <= 100, "'sampling.jpeg_quality' must be 1..100")
    check(get("inference.max_concurrency") == 1, "'inference.max_concurrency' currently supports only 1 (sequential)")
    check(get("inference.timeout_seconds") > 0, "'inference.timeout_seconds' must be > 0")
    check(get("inference.max_retries") >= 0, "'inference.max_retries' must be >= 0")
    check(get("inference.provider") in INFERENCE_PROVIDERS,
          f"'inference.provider' (or INFERENCE_PROVIDER) must be one of {sorted(INFERENCE_PROVIDERS)}")
    check(
        get("inference.base_url").lower().startswith(("http://", "https://")),
        "'inference.base_url' (or OLLAMA_BASE_URL / VLLM_BASE_URL) must start with http:// or https://",
    )
    max_tokens = get("inference.max_output_tokens")
    check(max_tokens is None or max_tokens >= 64, "'inference.max_output_tokens' must be null or >= 64")
    check(get("inference.failure_policy") in {"uncertain", "unavailable"},
          "'inference.failure_policy' must be 'uncertain' or 'unavailable' (failures are never WASTE)")
    check(set(get("hoi.classifications")) == {"RELEVANT", "WASTE", "UNCERTAIN"},
          "'hoi.classifications' must be exactly RELEVANT, WASTE, UNCERTAIN")
    check(get("sampling.image_format") == "jpg", "'sampling.image_format' currently supports only 'jpg'")
    check(get("sampling.timeout_seconds") > 0, "'sampling.timeout_seconds' must be > 0")
    check(get("segmentation.timeout_seconds") > 0, "'segmentation.timeout_seconds' must be > 0")
    check(get("segmentation.preserve_stereo") is True, "'segmentation.preserve_stereo' must be true (left/right are never cut independently)")
    check(all(isinstance(a, str) for a in get("segmentation.video_encoder_args")),
          "'segmentation.video_encoder_args' must be a list of strings")
    check(all(isinstance(s, str) for s in get("telemetry.parse_streams")), "'telemetry.parse_streams' must be a list of strings")
    check(1 <= get("web.port") <= 65535, "'web.port' must be 1..65535")
    check(0 < get("verification.interval_seconds") <= get("sampling.interval_seconds"),
          "'verification.interval_seconds' must be > 0 and <= 'sampling.interval_seconds'")
    check(get("verification.guard_seconds") >= 0, "'verification.guard_seconds' must be >= 0")
    check(get("qc.alignment_checkpoints_max") >= 1, "'qc.alignment_checkpoints_max' must be >= 1")
    check(0 <= get("qc.max_inference_problem_ratio") <= 1, "'qc.max_inference_problem_ratio' must be 0..1")
    if problems:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))


@dataclass(frozen=True)
class ResolvedConfig:
    """Fully merged, validated configuration for a single run. Treat as read-only."""

    data: Mapping[str, Any]
    layers: tuple[str, ...]
    cli_overrides: Mapping[str, Any] = field(default_factory=dict)
    # Environment variable name -> dotted key it set (values are in `data`).
    env_overrides: Mapping[str, str] = field(default_factory=dict)

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        try:
            return copy.deepcopy(_get_dotted(self.data, dotted))
        except KeyError:
            if default is _MISSING:
                raise ConfigError(f"Configuration key '{dotted}' is not defined.") from None
            return default

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.data))

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the canonical resolved configuration.

        Sections that cannot affect results (see FINGERPRINT_EXCLUDED_SECTIONS) are
        excluded, so e.g. changing the console log level does not flag a config change.
        """
        relevant = {k: v for k, v in self.data.items() if k not in FINGERPRINT_EXCLUDED_SECTIONS}
        return sha256_json(relevant)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)


def load_configuration(
    project_root: Path,
    user_config_path: Path | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
) -> ResolvedConfig:
    """Build the resolved configuration from defaults, a user file, environment and CLI overrides."""
    config_dir = project_root / CONFIG_DIRECTORY_NAME
    merged: dict[str, Any] = {}
    layers: list[str] = []
    section_owner: dict[str, str] = {}

    for filename in DEFAULT_LAYER_FILES:
        path = config_dir / filename
        if not path.is_file():
            raise ConfigError(f"Default configuration file is missing: '{path}'.")
        layer = load_yaml_mapping(path)
        for section in layer:
            if section in section_owner:
                raise ConfigError(
                    f"Section '{section}' is defined in both '{section_owner[section]}' and '{filename}'. "
                    "Each section must live in exactly one default file."
                )
            section_owner[section] = filename
        merged = deep_merge(merged, layer, strict=False)
        layers.append(str(path))

    if user_config_path is not None:
        user_layer = load_yaml_mapping(user_config_path)
        merged = deep_merge(merged, user_layer, strict=True)
        layers.append(str(user_config_path.resolve()))

    applied_env: dict[str, str] = {}
    env = environment or {}
    provider_value = env.get(ENV_PROVIDER_VARIABLE, "").strip()
    if provider_value:
        _set_dotted(merged, "inference.provider", provider_value)
        applied_env[ENV_PROVIDER_VARIABLE] = "inference.provider"
    provider = _get_dotted(merged, "inference.provider", None)
    for variable, dotted in ENV_OVERRIDES_BY_PROVIDER.get(provider, {}).items():
        value = env.get(variable, "").strip()
        if value:
            _set_dotted(merged, dotted, value)
            applied_env[variable] = dotted
    if applied_env:
        layers.append("env:" + ",".join(sorted(applied_env)))

    overrides = dict(cli_overrides or {})
    for dotted, value in overrides.items():
        _set_dotted(merged, dotted, value)
    if overrides:
        layers.append("cli")

    validate_config(merged)
    return ResolvedConfig(data=merged, layers=tuple(layers), cli_overrides=overrides, env_overrides=applied_env)
