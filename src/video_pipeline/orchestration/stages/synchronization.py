"""Phase 7 stage: apply the video's cuts to the IMU telemetry."""

from __future__ import annotations

from video_pipeline.core.io import read_csv, read_json, utc_now_iso, write_csv_atomic, write_json_atomic
from video_pipeline.orchestration.context import RunContext
from video_pipeline.orchestration.stages.base import Stage, StageOutcome, load_video_info, require, video_duration
from video_pipeline.synchronization.imu_aligner import align_samples


class ImuSyncStage(Stage):
    name = "imu_sync"
    config_keys = ("segmentation.synchronize_imu",)

    def run(self, ctx: RunContext) -> StageOutcome:
        imu_meta = read_json(require(ctx.workspace.imu_dir / "original" / "imu_metadata.json", "telemetry"))
        timeline = read_json(require(ctx.workspace.processed_dir / "processing_timeline.json", "segmentation"))
        out_dir = ctx.workspace.imu_dir / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / "synchronization_metadata.json"

        skip_reason = None
        if not ctx.config.get("segmentation.synchronize_imu"):
            skip_reason = "Disabled by configuration (segmentation.synchronize_imu=false)."
        elif not imu_meta.get("available"):
            skip_reason = f"No IMU telemetry available: {imu_meta.get('reason')}"
        if skip_reason:
            for stale in out_dir.glob("*.csv"):
                stale.unlink()
            write_json_atomic(meta_path, {"status": "skipped", "reason": skip_reason, "recorded_at": utc_now_iso()})
            ctx.logger.warning("IMU synchronization skipped: %s", skip_reason)
            return StageOutcome(outputs=(meta_path,), details={"status": "skipped"})

        duration = video_duration(load_video_info(ctx))
        segments = timeline["retained_segments"]
        outputs = [meta_path]
        stats = {}
        for code, filename in imu_meta.get("files", {}).items():
            rows, columns = read_csv(ctx.workspace.imu_dir / "original" / filename)
            kept, s = align_samples(rows, segments, duration=duration)
            path = out_dir / filename
            write_csv_atomic(path, kept, columns + ["segment_index", "time_processed_seconds"])
            outputs.append(path)
            times = [r["time_processed_seconds"] for r in kept]
            stats[code] = {**s.__dict__, "max_processed_time": max(times) if times else None,
                           "non_monotonic": any(b < a for a, b in zip(times, times[1:]))}
            ctx.logger.info("%s: kept %d / %d samples (%d removed, %d outside video range).",
                            code, s.kept_samples, s.input_samples, s.removed_samples, s.outside_video_samples)

        write_json_atomic(meta_path, {
            "status": "synchronized",
            "method": "timestamp-based: keep samples with time_video_seconds in a retained [start, end) interval; "
                      "time_processed_seconds = processed_start + (t - original_start)",
            "time_column_original": "time_video_seconds",
            "time_column_processed": "time_processed_seconds",
            "retained_segments": segments,
            "streams": stats,
            "recorded_at": utc_now_iso(),
        })
        return StageOutcome(outputs=tuple(outputs), details={"status": "synchronized", "streams": list(stats)})
