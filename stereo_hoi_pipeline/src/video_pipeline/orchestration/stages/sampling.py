"""Phase 3 stage: stereo splitting + timestamp-based frame sampling.

Images are written to frames/left and frames/right for the inference stage, which
deletes each pair once it has been classified (unless sampling.keep_frames). The
durable output is the frame index (timestamps + image hashes).
"""

from __future__ import annotations

from video_pipeline.core.hashing import sha256_file
from video_pipeline.core.io import utc_now_iso, write_json_atomic, write_jsonl_atomic
from video_pipeline.orchestration.context import RunContext
from video_pipeline.orchestration.stages.base import Stage, StageOutcome, load_video_info, video_duration
from video_pipeline.orchestration.stages.pairs import FRAME_CONFIG_KEYS, PairExtractor
from video_pipeline.stereo.frame_sampler import jpeg_qscale


class SamplingStage(Stage):
    name = "sampling"
    config_keys = FRAME_CONFIG_KEYS + ("sampling.interval_seconds", "sampling.image_format")

    def run(self, ctx: RunContext) -> StageOutcome:
        duration = video_duration(load_video_info(ctx))
        extractor = PairExtractor(ctx)
        geometry = extractor.geometry
        for warning in geometry.warnings:
            ctx.logger.warning("Stereo geometry: %s", warning)
        ctx.logger.info("Stereo split: left %dx%d @x=0, right %dx%d @x=%d", geometry.left.width, geometry.left.height,
                        geometry.right.width, geometry.right.height, geometry.right.x)

        interval = ctx.config.get("sampling.interval_seconds")
        frames_dir = ctx.workspace.frames_dir
        for directory in (frames_dir / "left", frames_dir / "right"):  # idempotent: start clean
            for old in directory.glob("*.jpg") if directory.is_dir() else ():
                old.unlink()
        ctx.logger.info("Sampling one stereo pair every %ss over %.1fs (single decode pass)...", interval, duration)
        frames, missing, command = extractor.extract(window_start=0.0, window_end=duration, interval=interval,
                                                     left_dir=frames_dir / "left", right_dir=frames_dir / "right")
        ctx.manifest.record_command(self.name, command.to_record())
        if missing:
            ctx.logger.warning("%d sample(s) had no frame in their interval: %s", len(missing), missing[:20])

        index_rows = [{
            "sample_index": f.sample_index,
            "target_timestamp_seconds": f.target_timestamp_seconds,
            "frame_pts_seconds": f.frame_pts_seconds,
            "container_pts_seconds": f.container_pts_seconds,
            "left": ctx.workspace.relative(f.left_path),
            "right": ctx.workspace.relative(f.right_path),
            "left_sha256": sha256_file(f.left_path),
            "right_sha256": sha256_file(f.right_path),
        } for f in frames]
        index_path = frames_dir / "frame_index.jsonl"
        write_jsonl_atomic(index_path, index_rows)

        meta_path = frames_dir / "sampling_metadata.json"
        write_json_atomic(meta_path, {
            "rule": "first decoded frame with pts >= video_start + k*interval (timestamp-based, VFR-safe)",
            "interval_seconds": interval,
            "duration_seconds": duration,
            "samples": len(frames),
            "missing_sample_indices": missing,
            "geometry": geometry.to_dict(),
            "max_width": ctx.config.get("sampling.max_width"),
            "jpeg_quality": ctx.config.get("sampling.jpeg_quality"),
            "ffmpeg_mjpeg_qscale": jpeg_qscale(ctx.config.get("sampling.jpeg_quality")),
            "keep_frames": ctx.config.get("sampling.keep_frames"),
            "re_extraction": "pairs are re-extracted byte-identically on demand from the source using the index timestamps",
            "created_at": utc_now_iso(),
        })
        ctx.logger.info("Sampled %d stereo pairs in %.1fs.", len(frames), command.elapsed_seconds)
        return StageOutcome(outputs=(index_path, meta_path), details={"samples": len(frames), "missing": len(missing)})
