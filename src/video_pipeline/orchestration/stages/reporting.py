"""Phase 8 stage: before/after metrics and summary reports."""

from __future__ import annotations

from video_pipeline.analysis.hoi_analyzer import build_timeline
from video_pipeline.analysis.temporal_filter import Interval
from video_pipeline.core.io import read_json, read_jsonl, utc_now_iso, write_csv_atomic, write_json_atomic
from video_pipeline.metrics.hoi_metrics import compute_metrics
from video_pipeline.orchestration.context import RunContext
from video_pipeline.orchestration.stages.base import Stage, StageOutcome, load_video_info, require, video_duration


class MetricsStage(Stage):
    name = "metrics"
    config_keys = ("reporting",)

    def run(self, ctx: RunContext) -> StageOutcome:
        info = load_video_info(ctx)
        duration = video_duration(info)
        interval = ctx.config.get("sampling.interval_seconds")
        records = read_jsonl(require(ctx.workspace.inference_dir / "classifications.jsonl", "inference"))
        seg = ctx.workspace.segmentation_dir
        removed = [Interval(i["start"], i["end"]) for i in read_json(require(seg / "waste_intervals.json", "verification"))["intervals"]]
        retained = [Interval(i["start"], i["end"]) for i in read_json(require(seg / "retained_intervals.json", "verification"))["intervals"]]
        processing = read_json(require(ctx.workspace.processed_dir / "processing_timeline.json", "segmentation"))
        sync = read_json(require(ctx.workspace.imu_dir / "processed" / "synchronization_metadata.json", "imu_sync"))

        metrics = compute_metrics(
            build_timeline(records, interval=interval, duration=duration),
            original_duration=duration, removed=removed, retained=retained,
            processed_duration_measured=processing.get("processed_duration_measured"),
            inference_records=records, waste_threshold=ctx.config.get("hoi.waste_threshold_seconds"),
            sampling_interval=interval,
        )
        metrics["run_id"] = ctx.run_id
        metrics["created_at"] = utc_now_iso()
        reports = ctx.workspace.reports_dir
        outputs = []
        if ctx.config.get("reporting.generate_json"):
            path = reports / "metrics.json"
            write_json_atomic(path, metrics)
            outputs.append(path)
        if ctx.config.get("reporting.generate_csv"):
            path = reports / "metrics.csv"
            write_csv_atomic(path, metrics["comparison"], ["metric", "before", "after"])
            outputs.append(path)
        if ctx.config.get("reporting.generate_summary"):
            d, r = metrics["durations"], metrics["ratios_percent"]
            path = reports / "summary.json"
            write_json_atomic(path, {
                "run_id": ctx.run_id,
                "video": ctx.source.path.name,
                "source_sha256": ctx.source.sha256,
                "headline": (f"Removed {d['removed_duration']:.1f}s of {d['original_duration']:.1f}s "
                             f"({r['waste_reduction']}%) in {metrics['counts']['waste_intervals_removed']} interval(s); "
                             f"HOI coverage {r['hoi_coverage_before']}% -> {r['hoi_coverage_after']}%."),
                "waste_threshold_seconds": ctx.config.get("hoi.waste_threshold_seconds"),
                "model": ctx.config.get("inference.model"),
                "durations": d,
                "ratios_percent": r,
                "counts": metrics["counts"],
                "imu_synchronization": sync.get("status"),
                "verification": _verification_summary(seg / "verification.json"),
                "qc_status": "pending",
                "artifacts": {
                    "processed_video": processing.get("output"),
                    "processing_timeline": "processed/processing_timeline.json",
                    "verification": "segmentation/verification.json",
                    "metrics": "reports/metrics.json",
                    "qc_report": "reports/qc_report.json",
                    "manifest": "reports/manifest.json",
                },
                "created_at": utc_now_iso(),
            })
            outputs.append(path)
        ctx.logger.info("Removed %.1fs (%s%%); HOI coverage %s%% -> %s%%.", metrics["durations"]["removed_duration"],
                        metrics["ratios_percent"]["waste_reduction"], metrics["ratios_percent"]["hoi_coverage_before"],
                        metrics["ratios_percent"]["hoi_coverage_after"])
        return StageOutcome(outputs=tuple(outputs), details={"ratios_percent": metrics["ratios_percent"]})


def _verification_summary(path) -> dict | None:
    if not path.exists():
        return None
    v = read_json(path)
    outcomes = [c["outcome"] for c in v.get("candidates", [])]
    return {"enabled": v.get("enabled"), "interval_seconds": v.get("interval_seconds"),
            "candidate_duration": v.get("candidate_duration"), "removed_duration": v.get("removed_duration"),
            "candidates": len(outcomes), "removed": outcomes.count("removed"),
            "trimmed": outcomes.count("partially_removed"), "kept": outcomes.count("kept")}
