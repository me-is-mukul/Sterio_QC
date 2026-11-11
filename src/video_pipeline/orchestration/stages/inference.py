"""Phase 4 stage: classify every sampled stereo pair, deleting each pair once it is done."""

from __future__ import annotations

from statistics import mean

from video_pipeline.core.hashing import sha256_file
from video_pipeline.core.io import JsonlAppender, read_jsonl, utc_now_iso, write_json_atomic
from video_pipeline.orchestration.context import RunContext
from video_pipeline.orchestration.stages.base import Stage, StageOutcome, require
from video_pipeline.orchestration.stages.pairs import (
    INFERENCE_CONFIG_KEYS,
    PairClassifier,
    PairExtractor,
    delete_pair,
)


class InferenceStage(Stage):
    name = "inference"
    # base_url / timeouts / retries are deliberately excluded: a new tunnel URL must not invalidate results.
    config_keys = INFERENCE_CONFIG_KEYS

    def run(self, ctx: RunContext) -> StageOutcome:
        frames = read_jsonl(require(ctx.workspace.frames_dir / "frame_index.jsonl", "sampling"))
        keep_frames = ctx.config.get("sampling.keep_frames")
        interval = ctx.config.get("sampling.interval_seconds")
        pc = PairClassifier(ctx)
        extractor: PairExtractor | None = None

        out_dir = ctx.workspace.inference_dir
        classifications_path = out_dir / "classifications.jsonl"
        raw_path = out_dir / "raw_responses.jsonl"
        started_at = utc_now_iso()
        counts = {"requests": 0, "cached": 0, "failed": 0, "invalid": 0, "re_extracted": 0, "frames_deleted": 0}
        latencies: list[float] = []

        with JsonlAppender(classifications_path, truncate=True) as out, JsonlAppender(raw_path, truncate=True) as raw_out:
            for n, frame in enumerate(frames, start=1):
                ts = frame["target_timestamp_seconds"]
                left = ctx.workspace.root / frame["left"]
                right = ctx.workspace.root / frame["right"]
                key = pc.cache_key(frame["left_sha256"], frame["right_sha256"])
                pair = pc.cached(key)
                if pair is None:
                    if not (left.exists() and right.exists()):
                        # Images were deleted after an earlier attempt: re-extract this pair exactly.
                        extractor = extractor or PairExtractor(ctx)
                        extractor.extract(window_start=ts, window_end=ts + interval, interval=interval,
                                          left_dir=left.parent, right_dir=right.parent, prefix="_re_")
                        (left.parent / f"_re_{0:06d}.jpg").replace(left)
                        (right.parent / f"_re_{0:06d}.jpg").replace(right)
                        counts["re_extracted"] += 1
                        if sha256_file(left) != frame["left_sha256"] or sha256_file(right) != frame["right_sha256"]:
                            ctx.logger.warning("Re-extracted pair at t=%ss differs from the indexed hash.", ts)
                            key = None
                    pair = pc.classify(left, right, timestamp=ts, key=key)
                if pair.cached:
                    counts["cached"] += 1
                elif pair.error:
                    counts["failed"] += 1
                else:
                    counts["requests"] += 1
                    latencies.append(pair.latency_seconds)
                    if not pair.reusable:
                        counts["invalid"] += 1

                if not keep_frames and pair.reusable and (left.exists() or right.exists()):
                    delete_pair(left, right)
                    counts["frames_deleted"] += 1

                out.write({
                    "sample_index": frame["sample_index"],
                    "timestamp_seconds": ts,
                    "frame_pts_seconds": frame["frame_pts_seconds"],
                    "left": frame["left"],
                    "right": frame["right"],
                    **pair.result,
                    **pc.record_fields(),
                    "latency_seconds": pair.latency_seconds,
                    "attempts": pair.attempts,
                    "cached": pair.cached,
                    "cache_key": pair.cache_key,
                })
                raw_out.write({"sample_index": frame["sample_index"], "timestamp_seconds": ts, "cache_key": pair.cache_key,
                               "cached": pair.cached, "response_text": pair.response_text,
                               "provider_raw": pair.provider_raw, "error": pair.error})
                timing = "cached" if pair.cached else f"{pair.latency_seconds}s" if pair.latency_seconds is not None else "failed"
                desc = pair.result.get("hoi_description") or ""
                ctx.logger.info("[%d/%d] t=%6.1fs %-11s %s%s", n, len(frames), ts, pair.label, timing,
                                f" | {desc[:80]}" if desc else "")

        if not keep_frames:
            for directory in (ctx.workspace.frames_dir / "left", ctx.workspace.frames_dir / "right"):
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
        pending = counts["failed"] + counts["invalid"]
        meta_path = out_dir / "inference_metadata.json"
        write_json_atomic(meta_path, {
            "provider": pc.provider.describe(),
            "model": pc.model.model,
            "model_digest": pc.model.digest,
            "runtime_version": pc.model.runtime_version,
            "model_details": dict(pc.model.details),
            "prompt_version": pc.prompt.version,
            "parameters": pc.params,
            "failure_policy": pc.policy,
            "samples": len(frames),
            "counts": counts,
            "frames_kept": keep_frames,
            "mean_latency_seconds_uncached": round(mean(latencies), 3) if latencies else None,
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "note": "Model output is validated before use; model-stated confidence is not treated as a probability.",
        })
        ctx.manifest.set_configuration_extra({"model_digest": pc.model.digest, "prompt_version": pc.prompt.version,
                                              "inference_runtime_version": pc.model.runtime_version})
        if pending:
            ctx.logger.warning("%d pair(s) failed or returned invalid output; recorded as %s (never WASTE). "
                               "Their images are kept so --resume can retry them.", pending, pc.policy.upper())
        if not keep_frames:
            ctx.logger.info("Deleted %d classified stereo pair(s); only the frame index is kept.", counts["frames_deleted"])
        return StageOutcome(outputs=(classifications_path, raw_path, meta_path), details=counts)
