"""Phase 5b stage: verify candidate waste sections at a finer step before cutting.

Writes the FINAL removal decision (waste_intervals.json / retained_intervals.json)
that the segmentation stage executes, plus verification.json explaining every decision.
"""

from __future__ import annotations

import shutil

from video_pipeline.analysis.temporal_filter import Interval, complement
from video_pipeline.core.io import JsonlAppender, read_json, utc_now_iso, write_json_atomic
from video_pipeline.orchestration.context import RunContext
from video_pipeline.orchestration.stages.base import Stage, StageOutcome, load_video_info, require, video_duration
from video_pipeline.orchestration.stages.pairs import (
    FRAME_CONFIG_KEYS,
    INFERENCE_CONFIG_KEYS,
    PairClassifier,
    PairExtractor,
    delete_pair,
)
from video_pipeline.segmentation.verification import fine_timestamps, verify_candidate


class VerificationStage(Stage):
    name = "verification"
    config_keys = ("verification", "hoi.waste_threshold_seconds") + INFERENCE_CONFIG_KEYS + FRAME_CONFIG_KEYS

    def run(self, ctx: RunContext) -> StageOutcome:
        duration = video_duration(load_video_info(ctx))
        seg = ctx.workspace.segmentation_dir
        candidates = [Interval(i["start"], i["end"])
                      for i in read_json(require(seg / "waste_candidates.json", "temporal"))["intervals"]]
        threshold = ctx.config.get("hoi.waste_threshold_seconds")
        enabled = ctx.config.get("verification.enabled")
        step = ctx.config.get("verification.interval_seconds")
        guard = ctx.config.get("verification.guard_seconds")
        keep_frames = ctx.config.get("sampling.keep_frames")

        samples_path = seg / "verification_samples.jsonl"
        verified = []
        requests = 0
        if enabled and candidates:
            pc = PairClassifier(ctx)
            extractor = PairExtractor(ctx)
            work = ctx.workspace.frames_dir / "verification"
            with JsonlAppender(samples_path, truncate=True) as out:
                for ci, cand in enumerate(candidates):
                    ctx.logger.info("Verifying candidate %d/%d [%.1fs, %.1fs) every %ss...",
                                    ci + 1, len(candidates), cand.start, cand.end, step)
                    frames, _missing, _cmd = extractor.extract(
                        window_start=cand.start, window_end=cand.end, interval=step,
                        left_dir=work / "left", right_dir=work / "right", prefix=f"c{ci:03d}_")
                    by_time = {round(f.target_timestamp_seconds, 6): f for f in frames}
                    labels = []
                    for t in fine_timestamps(cand, step):
                        frame = by_time.get(t)
                        if frame is None:
                            labels.append((t, "MISSING"))
                            out.write({"candidate": ci, "timestamp_seconds": t, "classification": "MISSING"})
                            continue
                        pair = pc.classify(frame.left_path, frame.right_path, timestamp=t)
                        labels.append((t, pair.label))
                        out.write({"candidate": ci, "timestamp_seconds": t, "frame_pts_seconds": frame.frame_pts_seconds,
                                   **pair.result, **pc.record_fields(), "latency_seconds": pair.latency_seconds,
                                   "cached": pair.cached, "cache_key": pair.cache_key, "error": pair.error})
                        if not keep_frames:
                            delete_pair(frame.left_path, frame.right_path)
                    result = verify_candidate(cand, labels, step=step, guard=guard, threshold=threshold)
                    verified.append(result)
                    non_waste = [s for s in result.samples if s.label != "WASTE"]
                    ctx.logger.info("  -> %s: %d/%d fine samples not WASTE%s; removing %.1fs.",
                                    result.outcome, len(non_waste), len(result.samples),
                                    f" (at {', '.join(f'{s.timestamp:g}s {s.label}' for s in non_waste[:4])})" if non_waste else "",
                                    sum(r.duration for r in result.removed))
            if not keep_frames:
                shutil.rmtree(work, ignore_errors=True)
            requests = pc.requests
            removed = [r for v in verified for r in v.removed]
        else:
            samples_path.write_text("", encoding="utf-8")
            removed = list(candidates)
            if candidates:
                ctx.logger.warning("Verification disabled: candidates are removed without a fine-grained re-check.")

        retained = complement(removed, 0.0, duration)
        write_json_atomic(seg / "waste_intervals.json", {
            "threshold_seconds": threshold,
            "verified": bool(enabled and candidates),
            "intervals": [i.to_dict() for i in removed],
        })
        write_json_atomic(seg / "retained_intervals.json", {"intervals": [i.to_dict() for i in retained]})
        removed_total = round(sum(i.duration for i in removed), 6)
        write_json_atomic(seg / "verification.json", {
            "enabled": enabled,
            "interval_seconds": step,
            "guard_seconds": guard,
            "threshold_seconds": threshold,
            "rule": ("A candidate is removed only where no fine sample shows non-WASTE content; every non-WASTE "
                     "fine sample is protected with the guard margin, and only remaining pieces >= threshold are removed."),
            "candidates": [{
                "candidate": v.candidate.to_dict(),
                "outcome": v.outcome,
                "fine_samples": len(v.samples),
                "non_waste_samples": [{"timestamp": s.timestamp, "label": s.label} for s in v.samples if s.label != "WASTE"],
                "protected": [p.to_dict() for p in v.protected],
                "removed": [r.to_dict() for r in v.removed],
                "kept_short_pieces": [k.to_dict() for k in v.kept_pieces],
            } for v in verified],
            "removed_duration": removed_total,
            "candidate_duration": round(sum(c.duration for c in candidates), 6),
            "inference_requests": requests,
            "created_at": utc_now_iso(),
        })
        ctx.logger.info("Verified removal: %d interval(s), %.1fs of %.1fs (candidates: %.1fs).",
                        len(removed), removed_total, duration, sum(c.duration for c in candidates))
        return StageOutcome(
            outputs=(seg / "waste_intervals.json", seg / "retained_intervals.json", seg / "verification.json", samples_path),
            details={"removed_intervals": len(removed), "removed_duration": removed_total, "requests": requests},
        )
