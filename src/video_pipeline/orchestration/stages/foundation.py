"""Phase 1 stages: ingestion, validation, metadata."""

from __future__ import annotations

from video_pipeline.core.io import utc_now_iso, write_json_atomic
from video_pipeline.core.tools import resolve_executable
from video_pipeline.ingestion.validator import validate_probe, validate_source_file
from video_pipeline.metadata.ffprobe import ProbeResult, probe_video, summarize_probe
from video_pipeline.orchestration.context import RunContext
from video_pipeline.orchestration.stages.base import Stage, StageOutcome

_PROBE_STATE_KEY = "ffprobe_result"


class IngestionStage(Stage):
    """Record source identity (hash, size, origin) and download metadata."""

    name = "ingestion"

    def run(self, ctx: RunContext) -> StageOutcome:
        source_record = {**ctx.source.to_dict(ctx.project_root), "run_id": ctx.run_id, "recorded_at": utc_now_iso()}
        source_path = ctx.workspace.metadata_dir / "source.json"
        write_json_atomic(source_path, source_record)
        ctx.manifest.set_source(ctx.source.to_dict(ctx.project_root))
        outputs = [source_path]

        if ctx.source.download is not None:
            download_path = ctx.workspace.metadata_dir / "download.json"
            write_json_atomic(download_path, ctx.source.download.to_dict())
            outputs.append(download_path)
        elif ctx.source.origin == "url":
            ctx.logger.warning("Source came from a URL but this invocation did not download it; no download metadata recorded.")

        ctx.logger.info("Source %s | %s bytes | sha256=%s", ctx.source.path.name, f"{ctx.source.size_bytes:,}", ctx.source.sha256)
        return StageOutcome(outputs=tuple(outputs), details={"sha256": ctx.source.sha256, "origin": ctx.source.origin})


def _probe(ctx: RunContext) -> ProbeResult:
    cached = ctx.state.get(_PROBE_STATE_KEY)
    if cached is not None:
        return cached
    executable = resolve_executable(ctx.config.get("tools.ffprobe"))
    result = probe_video(
        executable,
        ctx.source.path,
        count_frames=ctx.config.get("metadata.count_frames"),
        timeout_seconds=ctx.config.get("tools.timeout_seconds"),
    )
    ctx.state[_PROBE_STATE_KEY] = result
    return result


class InputValidationStage(Stage):
    """Filesystem and container-level validation of the source video."""

    name = "validation"
    config_keys = ("video.supported_formats", "video.validate_before_processing")

    def run(self, ctx: RunContext) -> StageOutcome:
        file_checks = validate_source_file(ctx.source.path, ctx.config.get("video.supported_formats"))
        warnings: list[str] = []
        probe_checked = False
        if ctx.config.get("video.validate_before_processing"):
            warnings = validate_probe(_probe(ctx).raw)
            probe_checked = True
        else:
            ctx.logger.warning("Container validation disabled by configuration (video.validate_before_processing=false).")
        for message in warnings:
            ctx.logger.warning("Validation: %s", message)

        report_path = ctx.workspace.metadata_dir / "validation.json"
        write_json_atomic(
            report_path,
            {"run_id": ctx.run_id, "validated_at": utc_now_iso(), "file_checks": file_checks,
             "container_checked": probe_checked, "warnings": warnings, "passed": True},
        )
        ctx.logger.info("Validation passed with %d warning(s).", len(warnings))
        return StageOutcome(outputs=(report_path,), details={"warnings": warnings})


class MetadataExtractionStage(Stage):
    """Persist complete FFprobe output plus a normalised video summary."""

    name = "metadata"
    config_keys = ("metadata.count_frames",)

    def run(self, ctx: RunContext) -> StageOutcome:
        result = _probe(ctx)
        ctx.manifest.record_command(self.name, result.command.to_record())

        raw_path = ctx.workspace.metadata_dir / "ffprobe.json"
        write_json_atomic(raw_path, result.raw)

        summary = summarize_probe(result.raw)
        summary["source_sha256"] = ctx.source.sha256
        summary["extracted_at"] = utc_now_iso()
        info_path = ctx.workspace.metadata_dir / "video_info.json"
        write_json_atomic(info_path, summary)

        duration = summary["duration_seconds"]
        ctx.manifest.set_source({"duration_seconds": duration})
        ctx.manifest.set_result("original_duration", duration)

        primary = next((v for v in summary["video_streams"] if v["index"] == summary["primary_video_stream_index"]), None)
        if primary:
            ctx.logger.info(
                "Video: %sx%s %s %s fps (avg %s) | duration %.3fs | audio streams: %d | GPMF: %s | chapters: %d",
                primary["width"], primary["height"], primary["codec_name"],
                _fmt(primary["fps_nominal"]), _fmt(primary["fps_average"]), duration or 0.0,
                len(summary["audio_streams"]), "yes" if summary["telemetry"]["gpmf_stream_present"] else "no",
                summary["chapters_count"],
            )
            if primary["possibly_variable_frame_rate"]:
                ctx.logger.warning("Nominal and average frame rates differ: video may be variable frame rate.")
        return StageOutcome(
            outputs=(raw_path, info_path),
            details={"duration_seconds": duration, "gpmf_stream_present": summary["telemetry"]["gpmf_stream_present"]},
        )


def _fmt(value: float | None) -> str:
    return "?" if value is None else f"{value:.3f}"
