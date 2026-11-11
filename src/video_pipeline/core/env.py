"""Environment variables and the project ``.env`` file.

Values from the real process environment take precedence over ``.env``.
Only a small, documented set of variables is consumed (see ``config.ENV_OVERRIDES_BY_PROVIDER``
and the inference provider for secrets such as OLLAMA_API_KEY). Secrets are never
copied into configuration snapshots, manifests or logs.
"""

from __future__ import annotations

import os
from pathlib import Path


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines. Supports comments, blank lines, ``export`` and quotes."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            values[key] = value
    return values


def load_environment(project_root: Path) -> dict[str, str]:
    """``.env`` values overlaid with the process environment (process wins)."""
    merged = read_dotenv(project_root / ".env")
    merged.update(os.environ)
    return merged
