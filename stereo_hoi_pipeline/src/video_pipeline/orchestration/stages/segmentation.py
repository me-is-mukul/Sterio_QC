"""Phase 6 stage: cut waste intervals out of the stereo video."""

from __future__ import annotations

from video_pipeline.analysis.temporal_filter import Interval
from video_pipeline.core.io import read_json, utc_now_iso, write_json_atomic
from video_pipeline.core.tools import resolve_executable
from video_pipeline.metadata.ffprobe import probe_video
from video_pipeline.orchestration.context import RunContext
from video_pipeline.orchestration.stages.base import (
    Stage,
    StageOutcome,
    load_video_info,
    require,
    video_duration,
    video_start_time,
)
from video_pipeline.segmentation.video_cutter import EncoderSettings, build_processing_timeline, cut_video


def _intervals(path, produced_by: str) -> list[Interval]:
    return [Interval(i["start"], i["end"]) for i in read_json(require(path, produced_by))["intervals"]]


class SegmentationStage(Stage):
    name = "segmentation"
    config_keys = ("segmentation",)

    def run(self, ctx: RunContext) -> StageOutcome:
        info = load_video_info(ctx)
        duration = video_duration(info)
        seg_dir = ctx.workspace.segmentation_dir
        retained = _intervals(seg_dir / "retained_intervals.json", "verification")
        removed = _intervals(seg_dir / "waste_intervals.json", "verification")
        include_audio = bool(info.get("audio_streams")) and ctx.config.get("segmentation.preserve_audio")
        settings = EncoderSettings(
            video_encoder=ctx.config.get("segmentation.video_encoder"),
            video_encoder_args=tuple(ctx.config.get("segmentation.video_encoder_args")),
            pixel_format=ctx.config.get("segmentation.pixel_format"),
            audio_encoder=ctx.config.get("segmentation.audio_encoder"),
        )
        output = ctx.workspace.processed_dir / f"{ctx.workspace.video_name}_processed.mp4"
        timeline_path = ctx.workspace.processed_dir / "processing_timeline.json"
        segments = build_processing_timeline(retained)
        nominal = round(sum(i.duration for i in retained), 6)

        measured = None
        command_record = None
        if retained:
            ctx.logger.info("Encoding %d retained segment(s) (%.1fs of %.1fs) with %s...",
                            len(retained), nominal, duration, settings.video_encoder)
            result = cut_video(resolve_executable(ctx.config.get("tools.ffmpeg")), ctx.source.path, retained, output,
                               start_offset=video_start_time(info), include_audio=include_audio, settings=settings,
                               timeout_seconds=ctx.config.get("segmentation.timeout_seconds"))
            command_record = result.to_record()
            ctx.manifest.record_command(self.name, command_record)
            probe = probe_video(resolve_executable(ctx.config.get("tools.ffprobe")), output, count_frames=False,
                                timeout_seconds=ctx.config.get("tools.timeout_seconds"))
            measured = float(probe.raw.get("format", {}).get("duration", 0.0)) or None
            ctx.logger.info("Processed video: %s (%.3fs, encoded in %.1fs)", output.name, measured or 0.0, result.elapsed_seconds)
        else:
            output.unlink(missing_ok=True)
            ctx.logger.warning("Every interval was classified as removable WASTE; no processed video was written.")

        write_json_atomic(timeline_path, {
            "source": ctx.source.path.name,
            "output": ctx.workspace.relative(output) if retained else None,
            "original_duration": duration,
            "processed_duration_nominal": nominal,
            "processed_duration_measured": measured,
            "removed_intervals": [i.to_dict() for i in removed],
            "retained_segments": segments,
            "audio_included": include_audio,
            "encoder": {"video": settings.video_encoder, "args": list(settings.video_encoder_args),
                        "pixel_format": settings.pixel_format, "audio": settings.audio_encoder if include_audio else None},
            "notes": ["Full side-by-side frames are cut; left/right are never cut independently.",
                      "Chapters and data streams are not carried over (their times refer to the original)."],
            "created_at": utc_now_iso(),
        })
        ctx.manifest.set_result("removed_duration", round(duration - nominal, 6))
        ctx.manifest.set_result("processed_duration", measured if measured is not None else nominal)
        outputs = (timeline_path, output) if retained else (timeline_path,)
        return StageOutcome(outputs=outputs, details={"segments": len(segments), "processed_duration": measured or nominal})
