"""Run manifest: the reproducibility record of one processing run.

The manifest is rewritten atomically after every state change so an interrupted
run always leaves a consistent, resumable record. See docs/data_schema.md.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from video_pipeline.core.errors import ResumeError
from video_pipeline.core.io import read_json, utc_now_iso, write_json_atomic

MANIFEST_SCHEMA_VERSION = "1.0.0"

RUN_RUNNING = "running"
RUN_COMPLETED = "completed"
RUN_FAILED = "failed"
RUN_INTERRUPTED = "interrupted"
RUN_PARTIAL = "partial"  # stopped on purpose with --until

STAGE_RUNNING = "running"
STAGE_COMPLETED = "completed"
STAGE_FAILED = "failed"
STAGE_SKIPPED = "skipped"
STAGE_INTERRUPTED = "interrupted"


class RunManifest:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    # Construction -----------------------------------------------------------
    @classmethod
    def create(cls, run_id: str, video_name: str, pipeline_version: str) -> RunManifest:
        now = utc_now_iso()
        return cls(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "run_id": run_id,
                "video_name": video_name,
                "pipeline_version": pipeline_version,
                "status": RUN_RUNNING,
                "created_at": now,
                "updated_at": now,
                "finished_at": None,
                "attempts": [],
                "source": {},
                "configuration": {},
                "versions": {},
                "stages": {},
                "commands": [],
                "results": {
                    "original_duration": None,
                    "removed_duration": None,
                    "processed_duration": None,
                },
                "errors": [],
            }
        )

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        try:
            data = read_json(path)
        except (OSError, ValueError) as exc:
            raise ResumeError(f"Cannot read manifest '{path}': {exc}") from exc
        if not isinstance(data, dict) or "run_id" not in data:
            raise ResumeError(f"'{path}' is not a valid run manifest.")
        return cls(data)

    # Accessors ----------------------------------------------------------------
    @property
    def run_id(self) -> str:
        return self._data["run_id"]

    @property
    def status(self) -> str:
        return self._data["status"]

    @property
    def source(self) -> dict[str, Any]:
        return copy.deepcopy(self._data.get("source") or {})

    @property
    def configuration_fingerprint(self) -> str | None:
        return (self._data.get("configuration") or {}).get("fingerprint")

    def stage(self, name: str) -> dict[str, Any] | None:
        record = self._data["stages"].get(name)
        return copy.deepcopy(record) if record else None

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    # Mutation -----------------------------------------------------------------
    def _touch(self) -> None:
        self._data["updated_at"] = utc_now_iso()

    def begin_attempt(self, mode: str, argv: Sequence[str]) -> None:
        self._data["status"] = RUN_RUNNING
        self._data["finished_at"] = None
        self._data["attempts"].append(
            {
                "attempt": len(self._data["attempts"]) + 1,
                "mode": mode,
                "argv": list(argv),
                "started_at": utc_now_iso(),
                "finished_at": None,
                "status": RUN_RUNNING,
            }
        )
        self._touch()

    def set_source(self, source: Mapping[str, Any]) -> None:
        self._data["source"].update(copy.deepcopy(dict(source)))
        self._touch()

    def set_configuration(self, configuration: Mapping[str, Any]) -> None:
        previous = self._data.get("configuration") or {}
        new = copy.deepcopy(dict(configuration))
        history = list(previous.get("fingerprint_history", []))
        if previous.get("fingerprint") and previous.get("fingerprint") != new.get("fingerprint"):
            history.append({"fingerprint": previous["fingerprint"], "replaced_at": utc_now_iso()})
        new["fingerprint_history"] = history
        self._data["configuration"] = new
        self._touch()

    def set_configuration_extra(self, values: Mapping[str, Any]) -> None:
        """Add run-time facts to the configuration record (e.g. model digest)."""
        self._data.setdefault("configuration", {}).update(copy.deepcopy(dict(values)))
        self._touch()

    def set_planned_stages(self, names: Sequence[str]) -> None:
        self._data["planned_stages"] = list(names)
        self._touch()

    def set_versions(self, versions: Mapping[str, Any]) -> None:
        self._data["versions"] = dict(versions)
        self._touch()

    def set_result(self, key: str, value: Any) -> None:
        self._data["results"][key] = value
        self._touch()

    def record_command(self, stage: str, record: Mapping[str, Any]) -> None:
        self._data["commands"].append({"stage": stage, "recorded_at": utc_now_iso(), **dict(record)})
        self._touch()

    def mark_stage_running(self, name: str) -> None:
        previous = self._data["stages"].get(name) or {}
        self._data["stages"][name] = {
            "status": STAGE_RUNNING,
            "started_at": utc_now_iso(),
            "finished_at": None,
            "runs": previous.get("runs", 0) + 1,
            "outputs": [],
            "details": {},
            "error": None,
        }
        self._touch()

    def mark_stage_completed(self, name: str, outputs: Sequence[str], details: Mapping[str, Any],
                             config_fingerprint: str | None = None) -> None:
        record = self._data["stages"][name]
        record.update(
            status=STAGE_COMPLETED,
            finished_at=utc_now_iso(),
            outputs=list(outputs),
            details=copy.deepcopy(dict(details)),
            config_fingerprint=config_fingerprint,
        )
        self._touch()

    def mark_stage_skipped(self, name: str, reason: str) -> None:
        record = self._data["stages"].setdefault(name, {"status": STAGE_SKIPPED, "outputs": [], "details": {}})
        record.setdefault("skips", []).append({"at": utc_now_iso(), "reason": reason})
        self._touch()

    def mark_stage_failed(self, name: str, error: BaseException, *, interrupted: bool = False) -> None:
        record = self._data["stages"].setdefault(name, {})
        message = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
        record.update(status=STAGE_INTERRUPTED if interrupted else STAGE_FAILED, finished_at=utc_now_iso(), error=message)
        self._data["errors"].append({"stage": name, "at": utc_now_iso(), "error": message})
        self._touch()

    def finish(self, status: str) -> None:
        now = utc_now_iso()
        self._data["status"] = status
        self._data["finished_at"] = now
        if self._data["attempts"]:
            self._data["attempts"][-1].update(finished_at=now, status=status)
        self._touch()

    def save(self, path: Path) -> None:
        write_json_atomic(path, self._data)
