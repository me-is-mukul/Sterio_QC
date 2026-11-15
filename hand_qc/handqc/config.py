"""Configuration loading: config/hand_qc.yaml, optionally merged with a user override file."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "hand_qc.yaml"


class ConfigError(Exception):
    pass


def _merge(base: dict, override: dict, where: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key not in out:
            raise ConfigError(f"Unknown setting '{where}{key}' (check the spelling against config/hand_qc.yaml)")
        if isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _merge(out[key], value, f"{where}{key}.")
        else:
            out[key] = value
    return out


def load_config(override_file: Path | None = None) -> dict:
    with DEFAULT_CONFIG.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if override_file:
        with Path(override_file).open(encoding="utf-8") as fh:
            override = yaml.safe_load(fh) or {}
        if not isinstance(override, dict):
            raise ConfigError(f"{override_file} must contain a mapping of settings")
        config = _merge(config, override)
    _validate(config)
    return config


def resolve_path(config: dict, key: str) -> Path:
    path = Path(config["paths"][key])
    return path if path.is_absolute() else (ROOT / path).resolve()


def _validate(c: dict) -> None:
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise ConfigError(msg)

    a = c["analysis"]
    need(int(a["workers"]) >= 1, "analysis.workers must be >= 1")
    bands = a["stride_bands"]
    need(isinstance(bands, list) and len(bands) > 0, "analysis.stride_bands must be a non-empty list")
    need(all(int(b["stride"]) >= 1 for b in bands), "analysis.stride_bands: stride must be >= 1")
    need([b["max_fps"] for b in bands] == sorted(b["max_fps"] for b in bands),
         "analysis.stride_bands must be sorted by max_fps")
    need(int(a["max_view_width"]) >= 320, "analysis.max_view_width must be >= 320")
    need(float(a["blockiness_fps"]) > 0, "analysis.blockiness_fps must be > 0")
    mp = a["mediapipe"]
    need(1 <= int(mp["num_hands"]) <= 10, "analysis.mediapipe.num_hands must be 1..10")
    for k in ("min_hand_detection_confidence", "min_hand_presence_confidence", "min_tracking_confidence"):
        need(0.0 <= float(mp[k]) <= 1.0, f"analysis.mediapipe.{k} must be within 0..1")
    g = c["grasp"]
    need(float(g["min_duration_s"]) >= 0 and float(g["max_gap_s"]) >= 0, "grasp durations must be >= 0")
    t = c["tracking"]
    need(float(t["min_tracked_s"]) >= 0 and float(t["min_lost_s"]) > 0, "tracking durations must be positive")
    fq = c["frame_quality"]
    need(0 <= fq["dark_luma"] < fq["bright_luma"] <= 255, "frame_quality: need 0 <= dark_luma < bright_luma <= 255")
    for layout in ("mono", "sbs", "grid3x2"):
        need(layout in c["wearer"]["use_direction_rule"], f"wearer.use_direction_rule needs an entry for '{layout}'")


def analysis_settings(config: dict) -> dict:
    """The settings that change analysis output (used for the analysis fingerprint)."""
    a = config["analysis"]
    return {"stride_bands": a["stride_bands"], "max_view_width": a["max_view_width"],
            "blockiness_fps": a["blockiness_fps"], "mediapipe": a["mediapipe"]}
