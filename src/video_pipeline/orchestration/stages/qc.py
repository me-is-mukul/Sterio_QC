"""Quality-control stage: verify decisions, the processed video and telemetry.

Writes reports/qc_report.json. With qc.fail_run_on_failure, any failed check
fails the run (after the report is written) so a bad output is never mistaken
for a good one.
"""

from __future__ import annotations

from video_pipeline.analysis.hoi_analyzer import build_timeline
from video_pipeline.analysis.temporal_filter import Interval
from video_pipeline.core.errors import QualityCheckError
from video_pipeline.core.hashing import sha256_file
from video_pipeline.core.io import read_json, read_jsonl, utc_now_iso, write_json_atomic
from video_pipeline.core.tools import resolve_executable
from video_pipeline.orchestration.context import RunContext
from video_pipeline.orchestration.stages.base import (
    Stage,
    StageOutcome,
    load_video_info,
    primary_video_stream,
    require,
    video_duration,
)
from video_pipeline.qc import checks as qc
from video_pipeline.qc.video_checks import alignment_checkpoints, decode_errors, frame_psnr, probe_stream


def _intervals(data) -> list[Interval]:
    return [Interval(i["start"], i["end"]) for i in data["intervals"]]


class QualityCheckStage(Stage):
    name = "qc"
    config_keys = ("qc",)

    def run(self, ctx: RunContext) -> StageOutcome:
        cfg = ctx.config
        info = load_video_info(ctx)
        duration = video_duration(info)
        seg = ctx.workspace.segmentation_dir
        removed = _intervals(read_json(require(seg / "waste_intervals.json", "verification")))
        retained = _intervals(read_json(require(seg / "retained_intervals.json", "verification")))
        records = read_jsonl(require(ctx.workspace.inference_dir / "classifications.jsonl", "inference"))
        fine = [(r["timestamp_seconds"], r["timestamp_seconds"] + cfg.get("verification.interval_seconds"),
                 r["classification"]) for r in read_jsonl(require(seg / "verification_samples.jsonl", "verification"))]
        processing = read_json(require(ctx.workspace.processed_dir / "processing_timeline.json", "segmentation"))
        sync = read_json(require(ctx.workspace.imu_dir / "processed" / "synchronization_metadata.json", "imu_sync"))
        coarse = build_timeline(records, interval=cfg.get("sampling.interval_seconds"), duration=duration)
        timeout = cfg.get("tools.timeout_seconds")

        results: list[qc.CheckResult] = []
        add = results.append
        ctx.logger.info("Running quality checks...")

        # Source integrity: full re-hash.
        digest = sha256_file(ctx.source.path, cfg.get("hashing.chunk_size_bytes"))
        add(qc.CheckResult("source.unchanged", "Source video is unchanged (SHA-256)",
                           qc.PASS if digest == ctx.source.sha256 else qc.FAIL,
                           f"sha256 {digest[:16]}..." if digest == ctx.source.sha256 else "source hash changed during the run"))

        # Decisions.
        add(qc.check_partition(removed, retained, duration))
        add(qc.check_removed_only_waste(removed, coarse, fine))
        add(qc.check_min_duration(removed, cfg.get("hoi.waste_threshold_seconds")))
        add(qc.check_inference_health(records, cfg.get("qc.max_inference_problem_ratio")))
        add(qc.check_coverage(coarse))
        verification = read_json(seg / "verification.json") if (seg / "verification.json").exists() else {}
        cands = verification.get("candidates", [])
        add(qc.CheckResult(
            "decisions.verified", "Every removed section was re-checked at a finer step",
            qc.PASS if verification.get("enabled") or not removed else qc.WARN,
            f"{len(cands)} candidate(s) checked every {verification.get('interval_seconds')}s: "
            f"{sum(c['outcome'] == 'removed' for c in cands)} removed, "
            f"{sum(c['outcome'] == 'partially_removed' for c in cands)} trimmed, "
            f"{sum(c['outcome'] == 'kept' for c in cands)} kept" if verification.get("enabled")
            else "verification disabled" if removed else "nothing removed"))

        # Processed video.
        output_rel = processing.get("output")
        if not output_rel:
            for cid in ("output.layout", "output.duration", "output.decode", "output.alignment"):
                add(qc.CheckResult(cid, cid, qc.SKIPPED, "no processed video (everything removed)"))
        else:
            output = ctx.workspace.root / output_rel
            ffmpeg = resolve_executable(cfg.get("tools.ffmpeg"))
            ffprobe = resolve_executable(cfg.get("tools.ffprobe"))
            src = primary_video_stream(info)
            out = probe_stream(ffprobe, output, timeout)
            same = out.get("width") == src["width"] and out.get("height") == src["height"]
            add(qc.CheckResult("output.layout", "Processed video keeps the full stereo frame",
                               qc.PASS if same else qc.FAIL,
                               f"{out.get('width')}x{out.get('height')} {out.get('codec_name')} {out.get('pix_fmt')} "
                               f"(source {src['width']}x{src['height']})"))

            fps = src.get("fps_average") or src.get("fps_nominal") or 30.0
            nominal = processing["processed_duration_nominal"]
            delta = abs(out["duration"] - nominal)
            add(qc.CheckResult("output.duration", "Processed duration matches the retained intervals",
                               qc.PASS if delta <= 2 / fps else qc.FAIL,
                               f"measured {out['duration']:.3f}s vs expected {nominal:.3f}s (diff {delta * 1000:.0f} ms, "
                               f"tolerance {2000 / fps:.0f} ms)"))

            if cfg.get("qc.decode_check"):
                errors, took = decode_errors(ffmpeg, output, cfg.get("segmentation.timeout_seconds"))
                add(qc.CheckResult("output.decode", "Processed video decodes without errors",
                                   qc.FAIL if errors else qc.PASS,
                                   "; ".join(errors[:3]) if errors else f"full decode clean ({took:.1f}s)"))
            else:
                add(qc.CheckResult("output.decode", "Processed video decodes without errors", qc.SKIPPED, "disabled"))

            min_db = cfg.get("qc.min_alignment_psnr_db")
            frame = 1.0 / fps
            # Seeking lands on the first frame at/after the seek time. Frame timestamps of the
            # source and the re-encoded output sit on slightly different grids, so seek half a
            # frame early in both to compare the frames *nearest* the mapped times.
            half = frame / 2
            points = alignment_checkpoints(processing["retained_segments"], cfg.get("qc.alignment_checkpoints_max"))
            measured, off_by_one = [], 0
            for p, o in points:
                # Snap to the nominal frame grid so (t - half) is always half a frame from any frame time.
                p, o = round(p * fps) / fps, round(o * fps) / fps
                psnr_at = lambda shift: frame_psnr(ffmpeg, output, max(0.0, p - half), ctx.source.path,  # noqa: E731
                                                   max(0.0, o - half + shift * frame), timeout)
                db, offset = psnr_at(0), 0
                if db is None or db < min_db:  # distinguish a 1-frame slip from real misalignment
                    for cand in (-1, 1):
                        alt = psnr_at(cand)
                        if alt is not None and (db is None or alt > db):
                            db, offset = alt, cand
                if offset and db is not None and db >= min_db:
                    off_by_one += 1
                measured.append({"processed": round(p, 3), "original": round(o, 3),
                                 "psnr_db": None if db is None else round(db, 2), "frame_offset": offset})
            values = [m["psnr_db"] for m in measured if m["psnr_db"] is not None]
            unmeasured = sum(m["psnr_db"] is None for m in measured)
            bad = [m for m in measured if m["psnr_db"] is not None and m["psnr_db"] < min_db]
            status = qc.FAIL if bad or unmeasured else qc.WARN if off_by_one else qc.PASS
            add(qc.CheckResult("output.alignment", "Processed frames match the original at mapped times",
                               status,
                               (f"{len(measured)} checkpoints, worst PSNR {min(values):.1f} dB, median "
                                f"{sorted(values)[len(values) // 2]:.1f} dB (min {min_db} dB)" if values else "no measurements")
                               + (f"; {off_by_one} matched only at +/-1 frame" if off_by_one else "")
                               + (f"; {len(bad)} below threshold" if bad else "")
                               + (f"; {unmeasured} could not be measured" if unmeasured else ""),
                               {"checkpoints": measured}))

        add(qc.check_imu(sync, processing.get("processed_duration_measured")))

        overall = qc.overall_status(results)
        report_path = ctx.workspace.reports_dir / "qc_report.json"
        write_json_atomic(report_path, {
            "status": overall,
            "checks": [r.to_dict() for r in results],
            "counts": {s: sum(r.status == s for r in results) for s in (qc.PASS, qc.WARN, qc.FAIL, qc.SKIPPED)},
            "run_id": ctx.run_id,
            "created_at": utc_now_iso(),
        })
        ctx.manifest.set_result("qc_status", overall)
        summary_path = ctx.workspace.reports_dir / "summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            summary["qc_status"] = overall
            summary["qc_counts"] = {s: sum(r.status == s for r in results) for s in (qc.PASS, qc.WARN, qc.FAIL, qc.SKIPPED)}
            write_json_atomic(summary_path, summary)
        for r in results:
            log = ctx.logger.error if r.status == qc.FAIL else ctx.logger.warning if r.status == qc.WARN else ctx.logger.info
            log("QC %-7s %-26s %s", r.status.upper(), r.id, r.detail)
        if overall == qc.FAIL and cfg.get("qc.fail_run_on_failure"):
            failed = [r.id for r in results if r.status == qc.FAIL]
            raise QualityCheckError(f"Quality checks failed: {', '.join(failed)}. See reports/qc_report.json.")
        return StageOutcome(outputs=(report_path,), details={"status": overall})
