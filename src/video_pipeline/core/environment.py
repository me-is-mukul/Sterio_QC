"""Collect runtime and tool versions for reproducibility manifests."""

from __future__ import annotations

import json
import logging
import platform
import re
import urllib.error
import urllib.request

import yaml

from video_pipeline import __version__
from video_pipeline.core.config import ResolvedConfig
from video_pipeline.core.errors import ToolExecutionError, ToolNotFoundError
from video_pipeline.core.tools import resolve_executable, run_command, tool_version

logger = logging.getLogger(__name__)

_SEMVER = re.compile(r"\d+\.\d+\.\d+")


def _ollama_version(base_url: str, timeout_seconds: float) -> str:
    """Prefer the running server's version; fall back to the local CLI client version."""
    url = base_url.rstrip("/") + "/api/version"
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310 (configured local URL)
            payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, dict) and payload.get("version"):
                return f"{payload['version']} (server)"
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        logger.debug("Ollama server not reachable at %s", url)
    try:
        result = run_command([resolve_executable("ollama"), "--version"], timeout_seconds=timeout_seconds)
        match = _SEMVER.search(result.stdout + result.stderr)
        if match:
            return f"{match.group(0)} (client only; server unreachable)"
    except (ToolNotFoundError, ToolExecutionError):
        pass
    return "unavailable"


def collect_versions(config: ResolvedConfig) -> dict[str, str | None]:
    timeout = config.get("environment.probe_timeout_seconds")
    versions: dict[str, str | None] = {
        "pipeline": __version__,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "pyyaml": getattr(yaml, "__version__", None),
        "ffmpeg": tool_version(config.get("tools.ffmpeg"), timeout_seconds=timeout),
        "ffprobe": tool_version(config.get("tools.ffprobe"), timeout_seconds=timeout),
    }
    if config.get("inference.provider") == "ollama":
        versions["ollama"] = _ollama_version(config.get("inference.base_url"), timeout)
    return versions
