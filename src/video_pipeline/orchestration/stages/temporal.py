"""Phase 5 stage: HOI timeline + candidate waste intervals.

Candidates are continuous WASTE runs (1 sample per interval) lasting >= N seconds.
They are NOT cut directly: the verification stage re-checks each candidate at a
finer step and only removes the parts where no relevant HOI is found.
"""

from __future__ import annotations

from video_pipeline.analysis.hoi_analyzer import build_timeline, label_durations
from video_pipeline.analysis.temporal_filter import contiguous_runs
from video_pipeline.core.io import read_jsonl, utc_now_iso, write_json_atomic
from video_pipeline.orchestration.context import RunContext
from video_pipeline.orchestration.stages.base import Stage, StageOutcome, load_video_info, require, video_duration
from video_pipeline.segmentation.waste_detector import decide_waste


class TemporalStage(Stage):
    name = "temporal"
    config_keys = ("hoi.waste_threshold_seconds", "hoi.uncertain_policy", "sampling.interval_seconds")

    def run(self, ctx: RunContext) -> StageOutcome:
        duration = video_duration(load_video_info(ctx))
        interval = ctx.config.get("sampling.interval_seconds")
        threshold = ctx.config.get("hoi.waste_threshold_seconds")
        classifications = read_jsonl(require(ctx.workspace.inference_dir / "classifications.jsonl", "inference"))

        timeline = build_timeline(classifications, interval=interval, duration=duration)
        runs = contiguous_runs(timeline)
        decisions = decide_waste(runs, threshold)
        candidates = [d.run.interval for d in decisions if d.removed]

        seg = ctx.workspace.segmentation_dir
        timeline_path = seg / "classifications_timeline.json"
        write_json_atomic(timeline_path, {
            "interval_seconds": interval,
            "duration_seconds": duration,
            "samples": [{"sample_index": t.sample_index, **t.interval.to_dict(), "label": t.label} for t in timeline],
            "runs": [{"label": r.label, **r.interval.to_dict(), "samples": len(r.sample_indices)} for r in runs],
        })
        candidates_path = seg / "waste_candidates.json"
        write_json_atomic(candidates_path, {
            "threshold_seconds": threshold,
            "intervals": [i.to_dict() for i in candidates],
            "decisions": [{**d.to_dict(), "candidate": d.removed} for d in decisions],
        })
        candidate_total = round(sum(i.duration for i in candidates), 6)
        meta_path = seg / "segmentation_metadata.json"
        write_json_atomic(meta_path, {
            "waste_threshold_seconds": threshold,
            "sampling_interval_seconds": interval,
            "uncertain_policy": ctx.config.get("hoi.uncertain_policy"),
            "interval_convention": "half-open [start, end) seconds from video start",
            "rules": [
                "A candidate is a continuous WASTE run (1 sample per interval) with duration >= threshold.",
                "RELEVANT, UNCERTAIN, UNAVAILABLE and MISSING samples are preserved and break WASTE runs.",
                "Candidates are verified at a finer step before anything is removed (see verification.json).",
            ],
            "label_durations": label_durations(timeline),
            "waste_runs": len(decisions),
            "candidates": len(candidates),
            "candidate_duration": candidate_total,
            "created_at": utc_now_iso(),
        })
        ctx.logger.info("%d WASTE run(s); %d candidate section(s) >= %ss (%.1fs of %.1fs) go to verification.",
                        len(decisions), len(candidates), threshold, candidate_total, duration)
        return StageOutcome(outputs=(timeline_path, candidates_path, meta_path),
                            details={"candidates": len(candidates), "candidate_duration": candidate_total})
