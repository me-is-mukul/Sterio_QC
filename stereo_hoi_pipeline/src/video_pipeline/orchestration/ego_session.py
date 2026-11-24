"""Run the QA sheet on ONE recording (a session folder in Cloud Storage, any supported format) and
upload the result. The recording format is detected from the folder's files (ingestion/formats);
the steps below only use the format-independent index the format's reader writes.

Steps (each records its config fingerprint in the manifest; a re-run skips completed steps whose
configuration did not change, and redoes everything after the first step that runs again):

    extract            index every segment from Cloud Storage (frame times, keyframes, IMU,
                       calibration); no video is written to disk
    technical          section 1: stream the video again, straight into FFmpeg (one process per eye,
                       or one for a side-by-side stream; full decode); the same pass writes the 2 s
                       label frames and the calibration frames
    calibration        section 2 (rectified / unrectified, eye order)
    labelling          every model labels every sampled left-eye frame (cached per sample)
    content_recording  section 3, recording level
    cropping           section 4
    content_clips      section 3, clip level
    clips              stream the kept pieces into MP4 (both eyes, stream copy), one eye at a time;
                       each verified MP4 is uploaded and deleted before the next is written
then qa_report.json, upload (qa_report.json last) and clean-up. A FAIL stops the steps that follow
it; the report and upload still happen.

Raw recordings are never stored on the local disk: only small tables, the sampled JPEG frames,
24 calibration frame pairs, and one clip MP4 at a time.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from video_pipeline import __version__
from video_pipeline.analysis.content_screen import (
    clip_findings,
    evenly_spaced,
    recording_frame_flags,
    recording_strip_finding,
)
from video_pipeline.analysis.cropping import decide_samples, defect_frame_ranges, flagged_runs, plan_pieces
from video_pipeline.core.config import ResolvedConfig
from video_pipeline.core.ego_config import STEP_CONFIG_KEYS, processed_bucket
from video_pipeline.core.errors import InferenceError, IngestionError, PipelineError
from video_pipeline.core.hashing import sha256_json
from video_pipeline.core.io import read_json, utc_now_iso, write_json_atomic, write_text_atomic
from video_pipeline.core.locking import RunLock
from video_pipeline.core.logging_setup import LoggingController, set_run_id, stage_scope
from video_pipeline.core.tools import resolve_executable
from video_pipeline.inference.ego_labeller import FrameLabeller, Sample, VlmLabeller, label_samples
from video_pipeline.inference.prompts import get_prompt
from video_pipeline.ingestion.formats import RecordingReader, classify, make_reader
from video_pipeline.ingestion.gcs import GcsStorage, SessionListing, SessionRef, guess_content_type
from video_pipeline.ingestion.mcap_reader import CAMERA_KEYS, assemble_session, load_segment_summary
from video_pipeline.orchestration.context import generate_run_id
from video_pipeline.qc.technical import (
    DecodeSettings,
    calibration_frame_indices,
    common_keyframes,
    decode_recording,
    evaluate_technical,
    pair_eyes,
)
from video_pipeline.reporting.manifest import RunManifest
from video_pipeline.segmentation.clip_writer import write_clip
from video_pipeline.stereo.calibration_check import check_calibration

logger = logging.getLogger("video_pipeline.ego")

STEPS = ("extract", "technical", "calibration", "labelling", "content_recording", "cropping", "content_clips",
         "clips")
REPORT_SCHEMA_VERSION = "1.0.0"

PASS, FAIL = "PASS", "FAIL"


@dataclass
class Services:
    config: ResolvedConfig
    project_root: Path
    storage: GcsStorage
    logging_controller: LoggingController
    # Builds and preflights the models: called on first use, so recordings that fail section 1
    # never need them. Returns (labellers, model info).
    model_factory: Callable[[], tuple[dict[str, FrameLabeller], dict[str, Any]]] | None = None
    _labellers: dict[str, FrameLabeller] | None = None
    _model_info: dict[str, Any] = field(default_factory=dict)
    _model_error: str | None = None

    def labellers(self) -> dict[str, FrameLabeller]:
        if self._labellers is None:
            if self._model_error:
                raise InferenceError(self._model_error)
            if self.model_factory is None:
                raise PipelineError("No models are configured.")
            try:
                with stage_scope("models"):
                    self._labellers, self._model_info = self.model_factory()
            except PipelineError as exc:
                self._model_error = f"Models are not usable: {exc}"
                raise InferenceError(self._model_error) from exc
            for key, info in self._model_info.items():
                logger.info("Model %s ready: %s", key, info)
        return self._labellers

    @property
    def model_info(self) -> dict[str, Any]:
        return dict(self._model_info)

    def close(self) -> None:
        for labeller in (self._labellers or {}).values():
            labeller.close()


@dataclass(frozen=True)
class SessionPaths:
    root: Path

    @property
    def work(self) -> Path: return self.root / "work"
    @property
    def stream(self) -> Path: return self.work / "index"
    @property
    def samples(self) -> Path: return self.work / "samples"
    @property
    def calib(self) -> Path: return self.work / "calibration_frames"
    @property
    def out(self) -> Path: return self.root / "out"
    @property
    def logs(self) -> Path: return self.out / "logs"
    @property
    def clips(self) -> Path: return self.out / "clips"
    @property
    def manifest(self) -> Path: return self.logs / "manifest.json"
    @property
    def lock(self) -> Path: return self.root / ".run.lock"


@dataclass
class State:
    session: SessionRef
    listing: SessionListing
    paths: SessionPaths
    manifest: RunManifest
    config: ResolvedConfig
    force: bool = False
    upload_prefix: str | None = None              # set when results are uploaded (object name prefix)
    uploaded: set[str] = field(default_factory=set)   # object names uploaded in this run
    verdict: str | None = None
    fail_reasons: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    reader: RecordingReader | None = None

    def fail(self, section: str, check: str, message: str) -> None:
        self.verdict = FAIL
        self.fail_reasons.append({"section": section, "check": check, "message": message})


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------

def target_prefix(cfg: ResolvedConfig, session: SessionRef, run_id: str) -> str:
    base = f"{cfg.get('source.processed_root')}{session.relative}/"
    if cfg.get("upload.on_existing") == "new_run_folder":
        return f"{base}runs/{run_id}/"
    return base


def run_session(session: SessionRef, services: Services, *, force: bool, upload: bool, until: str | None = None,
                max_segments: int | None = None, argv: tuple[str, ...] = ()) -> dict[str, Any]:
    cfg = services.config
    started = time.perf_counter()
    bucket = processed_bucket(cfg)
    existing_report = f"{cfg.get('source.processed_root')}{session.relative}/qa_report.json"
    on_existing = cfg.get("upload.on_existing")
    if upload and not force and on_existing == "skip" and services.storage.exists(bucket, existing_report):
        logger.info("%s: already processed (gs://%s/%s exists); skipped. Use --force to redo it.",
                    session.relative, bucket, existing_report)
        return {"session": session.url, "status": "SKIPPED", "reasons": "already processed", "elapsed_s": 0.0}

    work_root = Path(cfg.get("work.directory"))
    work_root = work_root if work_root.is_absolute() else services.project_root / work_root
    paths = SessionPaths((work_root / session.relative).resolve())
    if force and paths.root.exists():
        if paths.lock.exists():
            RunLock(paths.lock, "check").acquire()   # raises if another live run holds it
            paths.lock.unlink(missing_ok=True)
        shutil.rmtree(paths.root)
    paths.root.mkdir(parents=True, exist_ok=True)
    lock = RunLock(paths.lock, " ".join(argv) or "run_gcs_batch")
    lock.acquire()
    try:
        return _run_locked(session, services, paths, upload=upload, until=until, max_segments=max_segments,
                           argv=argv, started=started, force=force)
    finally:
        lock.release()


def _run_locked(session: SessionRef, services: Services, paths: SessionPaths, *, upload: bool, until: str | None,
                max_segments: int | None, argv: tuple[str, ...], started: float, force: bool) -> dict[str, Any]:
    cfg = services.config
    objects = services.storage.list_objects(session.bucket, session.prefix)
    if not objects:
        # Nothing at all: a wrong URL, not a recording with missing files. No report, nothing uploaded.
        raise IngestionError(f"No objects found at {session.url} (check the URL).")
    listing = classify(session, objects, cfg.get("source.segment_pattern"))
    if listing.format is None:
        raise IngestionError(f"{session.url} is not a recording in a known format (files: "
                             f"{sorted(o.name[len(session.prefix):] for o in objects)[:10]}).")
    listing = listing.limited(max_segments)

    manifest = _open_manifest(paths, session, listing, max_segments)
    set_run_id(manifest.run_id)
    services.logging_controller.attach_file(paths.logs / "processing.log", cfg.get("logging.file_level"))
    state = State(session, listing, paths, manifest, cfg, force=force)
    # the recording's own small metadata files (session.json, calibration.json, ...) go to out/recording/
    state.reader = make_reader(listing, services.storage, cfg, work_dir=paths.work, keep_dir=paths.out / "recording",
                               max_segments=max_segments)
    try:
        logger.info("Session %s | format %s | %d segments, %.2f GB | run %s", session.relative, listing.format,
                    len(listing.segments), listing.total_bytes / 1e9, manifest.run_id)
        write_json_atomic(paths.out / "source_listing.json", {**session.describe(), **listing.to_dict(),
                                                               "max_segments": max_segments})
        write_text_atomic(paths.logs / "resolved_config.yaml", cfg.to_yaml())
        manifest.set_configuration({"fingerprint": cfg.fingerprint, "layers": list(cfg.layers),
                                    "cli_overrides": dict(cfg.cli_overrides), "snapshot": cfg.to_dict()})
        manifest.set_versions(_versions(cfg))
        manifest.save(paths.manifest)
        if upload and until is None:
            state.upload_prefix = target_prefix(cfg, session, manifest.run_id)
            _prepare_target(state, services)
        _run_steps(state, services, until)
        if until is not None:
            manifest.finish("partial")
            manifest.save(paths.manifest)
            logger.info("Stopped after step '%s' as requested (no report, no upload).", until)
            return {"session": session.url, "status": "PARTIAL", "reasons": f"--until {until}",
                    "elapsed_s": round(time.perf_counter() - started, 1)}
        prefix = state.upload_prefix
        report = _write_report(state, services, prefix, max_segments)
        manifest.set_result("verdict", report["verdict"])
        if prefix is not None:
            # "completed" only once the upload succeeded: a failed upload is retried on the next run
            # (finished steps are skipped; the remaining files are uploaded again).
            manifest.set_result("upload", "in progress")
            manifest.save(paths.manifest)
            with stage_scope("upload"):
                _upload(state, services, prefix)
            manifest.set_result("upload", f"gs://{processed_bucket(cfg)}/{prefix}")
            manifest.finish("completed")
            manifest.save(paths.manifest)
            if cfg.get("work.delete_local_files_after_upload"):
                _cleanup(paths)
        else:
            manifest.set_result("upload", "disabled")
            manifest.finish("completed")
            manifest.save(paths.manifest)
            logger.info("Upload disabled; results are in %s", paths.out)
        logger.info("Session %s: %s%s (%.1f s)", session.relative, report["verdict"],
                    "" if report["verdict"] == PASS else " - " + "; ".join(r["message"] for r in report["fail_reasons"]),
                    time.perf_counter() - started)
        return {"session": session.url, "status": report["verdict"],
                "reasons": "; ".join(r["message"] for r in report["fail_reasons"]),
                "kept_seconds": report["cropping"].get("kept_seconds") if report.get("cropping") else 0,
                "clips": len(report.get("clips", [])),
                "calibration": (report.get("calibration") or {}).get("result"),
                "elapsed_s": round(time.perf_counter() - started, 1)}
    except BaseException as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        logger.error("Session %s %s: %s", session.relative, "interrupted" if interrupted else "failed", exc,
                     exc_info=not isinstance(exc, (PipelineError, KeyboardInterrupt)))
        manifest.finish("interrupted" if interrupted else "failed")
        try:
            manifest.save(paths.manifest)
        except OSError:
            pass
        raise
    finally:
        services.logging_controller.detach_file()


def _open_manifest(paths: SessionPaths, session: SessionRef, listing: SessionListing,
                   max_segments: int | None) -> RunManifest:
    identity = {"listing": listing.identity(), "max_segments": max_segments}
    if paths.manifest.is_file():
        previous = RunManifest.load(paths.manifest)
        if previous.source.get("identity") == identity and previous.status not in {"completed"}:
            previous.begin_attempt("resume", [])
            logger.info("Resuming run %s for %s.", previous.run_id, session.relative)
            return previous
        if previous.source.get("identity") != identity:
            logger.info("The session's objects changed since the last local run; starting from scratch.")
        for child in (paths.work, paths.out):
            shutil.rmtree(child, ignore_errors=True)
    manifest = RunManifest.create(generate_run_id(), session.relative, __version__)
    manifest.begin_attempt("new", [])
    manifest.set_source({**session.describe(), "identity": identity, "segments": len(listing.segments),
                         "bytes": listing.total_bytes})
    paths.logs.mkdir(parents=True, exist_ok=True)
    manifest.save(paths.manifest)
    return manifest


def _versions(cfg: ResolvedConfig) -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415
    import platform  # noqa: PLC0415

    from video_pipeline.core.tools import tool_version  # noqa: PLC0415

    versions: dict[str, Any] = {"pipeline": __version__, "python": platform.python_version(),
                                "platform": platform.platform(),
                                "ffmpeg": tool_version(cfg.get("tools.ffmpeg"), timeout_seconds=10),
                                "ffprobe": tool_version(cfg.get("tools.ffprobe"), timeout_seconds=10)}
    for package in ("mcap", "mcap-protobuf-support", "protobuf", "opencv-contrib-python", "numpy", "mediapipe",
                    "google-cloud-storage", "google-genai"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


# ---------------------------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------------------------

def _step_fingerprint(cfg: ResolvedConfig, step: str, extra: Any = None) -> str:
    return sha256_json({"keys": {k: cfg.get(k) for k in STEP_CONFIG_KEYS.get(step, ())}, "extra": extra})


def _step_extra(state: State, step: str) -> Any:
    """Format-specific settings a step depends on (None for MCAP, so its fingerprints are unchanged)."""
    return state.reader.fingerprint_extra() if step == "extract" and state.reader is not None else None


def _run_steps(state: State, services: Services, until: str | None) -> None:
    functions: dict[str, Callable[[State, Services], dict[str, Any]]] = {
        "extract": _step_extract, "technical": _step_technical, "calibration": _step_calibration,
        "labelling": _step_labelling, "content_recording": _step_content_recording, "cropping": _step_cropping,
        "content_clips": _step_content_clips, "clips": _step_clips,
    }
    upstream_ran = False
    for step in STEPS:
        if state.verdict == FAIL:
            break
        fingerprint = _step_fingerprint(state.config, step, _step_extra(state, step))
        record = state.manifest.stage(step)
        output = state.paths.out / f"_{step}.json"
        with stage_scope(step):
            if (not upstream_ran and record and record.get("status") == "completed"
                    and record.get("config_fingerprint") == fingerprint and output.is_file()):
                result = read_json(output)
                state.manifest.mark_stage_skipped(step, "completed earlier with the same configuration")
                logger.info("Step '%s' skipped (completed earlier).", step)
            else:
                logger.info("Step '%s' started.", step)
                state.manifest.mark_stage_running(step)
                state.manifest.save(state.paths.manifest)
                t0 = time.perf_counter()
                try:
                    result = functions[step](state, services)
                except BaseException as exc:
                    state.manifest.mark_stage_failed(step, exc, interrupted=isinstance(exc, KeyboardInterrupt))
                    state.manifest.save(state.paths.manifest)
                    raise
                write_json_atomic(output, result)
                state.manifest.mark_stage_completed(step, [state.paths.out.joinpath(output.name).name],
                                                    {"elapsed_seconds": round(time.perf_counter() - t0, 2)},
                                                    config_fingerprint=fingerprint)
                upstream_ran = True
                logger.info("Step '%s' completed in %.1f s.", step, time.perf_counter() - t0)
            state.manifest.save(state.paths.manifest)
        state.data[step] = result
        if state.upload_prefix is not None:
            # clip MP4s uploaded by an earlier attempt of this run are part of the result, not stale
            state.uploaded.update(n for n in result.get("uploaded_during_step", []) if n.startswith(state.upload_prefix))
        for reason in result.get("fail_reasons", []):
            state.fail(reason["section"], reason["check"], reason["message"])
        if until == step:
            return


# ---------------------------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------------------------

def _step_extract(state: State, services: Services) -> dict[str, Any]:
    reader, paths = state.reader, state.paths
    reader.fetch_metadata()
    required = reader.required_files()
    failed = [c for c in required if not c["passed"]]
    base = {"format": reader.format, "format_description": reader.description, "layout": reader.layout,
            "codec": reader.codec, "required_files": required}
    if failed:
        logger.warning("Required files missing: %s", "; ".join(c["message"] for c in failed))
        return {"extracted": False, **base,
                "fail_reasons": [{"section": "technical", "check": c["check"], "message": c["message"]} for c in failed]}
    paths.stream.mkdir(parents=True, exist_ok=True)
    numbers = reader.extract(paths.stream)
    counts = assemble_session(paths.stream, numbers)
    return {"extracted": True, **base, "segments": numbers, **counts}


def _load_tables(paths: SessionPaths) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    tables = {cam: np.load(paths.stream / f"{cam}.npy") for cam in CAMERA_KEYS}
    return tables, np.load(paths.stream / "imu_t.npy"), np.load(paths.stream / "imu.npy")


def _segment_summaries(state: State) -> list[dict[str, Any]]:
    return [load_segment_summary(state.paths.stream, n) for n in state.data["extract"]["segments"]]


def _session_calibration(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """First message of every calibration topic, and how often it changed during the recording."""
    first: dict[str, Any] = {}
    changes: dict[str, int] = {}
    for summary in summaries:
        for name, value in summary["calibration"].items():
            stripped = {k: v for k, v in value.items() if k != "timestamp"}
            if name not in first:
                first[name] = value
                changes[name] = 0
            elif stripped != {k: v for k, v in first[name].items() if k != "timestamp"}:
                changes[name] += 1
            changes[name] += summary["calibration_changes"].get(name, 0)
    return {"messages": first, "changes": changes}


def _sample_step(cfg: ResolvedConfig) -> int:
    return int(round(cfg.get("labelling.interval_seconds") * cfg.get("technical.expected_fps")))


def _step_technical(state: State, services: Services) -> dict[str, Any]:
    cfg, paths = state.config, state.paths
    tech = cfg.get("technical")
    summaries = _segment_summaries(state)
    full = {cam: np.load(paths.stream / f"{cam}_all.npy") for cam in CAMERA_KEYS}
    imu_t = np.load(paths.stream / "imu_t.npy")
    calibration = _session_calibration(summaries)
    reader = state.reader
    nominal = reader.imu_nominal_hz(calibration["messages"])
    crops = reader.eye_crops(paths.stream) if reader.layout == "sbs" else None
    pairing = pair_eyes(full["cam0"], full["cam1"], tolerance_ns=int(tech["max_eye_offset_ms"] * 1e6),
                        max_edge_frames=tech["max_unpaired_edge_frames"])
    for path in (paths.stream / "cam0.npy", paths.stream / "cam1.npy"):
        path.unlink(missing_ok=True)
    tables = {cam: np.zeros(0, dtype=full[cam].dtype) for cam in CAMERA_KEYS}
    if pairing["ok"]:
        for cam in CAMERA_KEYS:
            a, b = pairing[f"{cam}_rows"]
            tables[cam] = full[cam][a:b]
            np.save(paths.stream / f"{cam}.npy", tables[cam])
    n = len(tables["cam0"])
    step = _sample_step(cfg)
    decode = None
    calib_indices: list[int] = []
    if pairing["ok"] and n > 0:
        calib_indices = calibration_frame_indices(n, step, cfg.get("calibration.pairs"))
        settings = DecodeSettings(fps=int(tech["expected_fps"]), sample_step=step,
                                  sample_max_width=cfg.get("labelling.image_max_width"),
                                  jpeg_quality=cfg.get("labelling.jpeg_quality"),
                                  black_min_seconds=tech["black_min_seconds"],
                                  black_pixel_threshold=tech["black_pixel_threshold"],
                                  black_picture_threshold=tech["black_picture_threshold"],
                                  freeze_noise_db=tech["freeze_noise_db"], freeze_min_seconds=tech["freeze_min_seconds"],
                                  timeout_seconds=tech["decode_timeout_seconds"])
        ffmpeg = resolve_executable(cfg.get("tools.ffmpeg"))
        logger.info("Decoding both eyes (%d frames each = %.0f min of video%s), streamed from the bucket; "
                    "progress every minute ...", n, n / settings.fps / 60,
                    f", side by side, eyes at x {crops['cam0']} and {crops['cam1']}" if crops else "")
        decode = decode_recording(ffmpeg, reader.stream_packets_from(paths.stream), tables, settings,
                                  sample_dirs={cam: paths.samples / cam for cam in CAMERA_KEYS},
                                  calib_dirs={cam: paths.calib / cam for cam in CAMERA_KEYS},
                                  calib_indices=calib_indices, log_dir=paths.work / "ffmpeg", layout=reader.layout,
                                  codec=reader.codec, crops=crops, harmless_errors=reader.harmless_decode_errors)
        for cam, result in decode.items():
            logger.info("%s decoded in %.1f s: %s/%s frames, %d errors, %d samples", cam, result["elapsed_seconds"],
                        result["decoded_frames"], n, result["decode_error_count"], result["samples_written"])
    listing = state.listing.to_dict()
    sequence = None
    if reader.frame_counter:                   # host receive times: gaps from the camera's frame counter
        parts = [paths.stream / f"seg_{n:05d}" / "seq.npy" for n in state.data["extract"]["segments"]]
        if parts and all(p.is_file() for p in parts):
            sequence = np.concatenate([np.load(p) for p in parts])
    result = evaluate_technical(listing=listing, segment_summaries=summaries, cam0=full["cam0"], cam1=full["cam1"],
                                imu_t=imu_t, imu_nominal_hz=nominal, decode=decode, cfg=tech, pairing=pairing,
                                paired_cam0=tables["cam0"] if pairing["ok"] else None,
                                required_checks=state.data["extract"]["required_files"],
                                imu_unavailable=reader.imu_unavailable, frame_sequence=sequence,
                                host_times=reader.host_clock)
    expected_samples = -(-n // step) if n else 0
    if decode is not None:
        for cam, d in decode.items():
            if d["returncode"] == 0 and d["samples_written"] != expected_samples:
                raise PipelineError(f"{cam}: {d['samples_written']} sampled frames written, {expected_samples} expected")
            if d["returncode"] == 0 and d["calibration_frames_written"] != len(calib_indices):
                raise PipelineError(f"{cam}: {d['calibration_frames_written']} calibration frames written, "
                                    f"{len(calib_indices)} expected")
    result.update(frames=n, sample_step_frames=step, samples=expected_samples, calibration_frame_indices=calib_indices,
                  calibration_snapshot=calibration, decode=decode, imu_nominal_hz=nominal, format=reader.format,
                  layout=reader.layout, eye_crops=crops, imu_checked=reader.imu_unavailable is None,
                  imu_not_checked_reason=reader.imu_unavailable,
                  fail_reasons=[{"section": "technical", "check": c["check"], "message": c["message"]}
                                for c in result["checks"] if not c["passed"] and not c.get("informational")])
    for c in result["checks"]:
        logger.log(logging.INFO if c["passed"] else logging.WARNING, "%s %s: %s",
                   "ok  " if c["passed"] else "FAIL", c["check"], c["message"])
    return result


def _step_calibration(state: State, services: Services) -> dict[str, Any]:
    tech = state.data["technical"]
    calib_dir = state.paths.calib
    pairs = [(calib_dir / "cam0" / f"{j:03d}.png", calib_dir / "cam1" / f"{j:03d}.png")
             for j in range(len(tech["calibration_frame_indices"]))]
    result = check_calibration(pairs, tech["calibration_snapshot"]["messages"], state.config.get("calibration"))
    result["frame_indices"] = tech["calibration_frame_indices"]
    result["calibration_changes_during_recording"] = tech["calibration_snapshot"]["changes"]
    result["imu"] = {k: tech["calibration_snapshot"]["messages"].get(k) for k in ("imu_info", "imu_extrinsics")}
    order = result["eye_order"]
    logger.info("Calibration: %s (%s); median %s px, p95 %s px, %s inliers; left eye = %s (%s%s)",
                result["result"], result["reason"], result.get("median_vertical_px"), result.get("p95_vertical_px"),
                result.get("inlier_matches"), order["left"], "determined from the images" if order["determined"]
                else "not determinable, as labelled", ", swapped vs labels" if order["eyes_swapped"] else "")
    return result


def _eyes(state: State) -> tuple[str, str]:
    left = state.data["calibration"]["eye_order"]["left"]
    return left, ("cam1" if left == "cam0" else "cam0")


def _samples(state: State) -> list[Sample]:
    left, right = _eyes(state)
    tables, _, _ = _load_tables(state.paths)
    step = state.data["technical"]["sample_step_frames"]
    t = tables[left]["t_ns"]
    t0 = int(t[0])
    return [Sample(k, k * step, int(t[k * step]), (int(t[k * step]) - t0) / 1e9,
                   state.paths.samples / left / f"{k:06d}.jpg", state.paths.samples / right / f"{k:06d}.jpg")
            for k in range(state.data["technical"]["samples"])]


def _frame_stats(samples: list[Sample], imu_t: np.ndarray, imu: np.ndarray) -> dict[int, dict[str, float | None]]:
    """Cheap model-free cross-checks per sample: brightness, sharpness, head motion (gyro)."""
    import cv2  # noqa: PLC0415

    stats = {}
    gyro = np.linalg.norm(imu[:, 0:3].astype(np.float64), axis=1) if len(imu) else np.zeros(0)
    for s in samples:
        gray = cv2.imread(str(s.image), cv2.IMREAD_GRAYSCALE)
        luma = sharp = None
        if gray is not None:
            luma = round(float(gray.mean()), 2)
            sharp = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)
        lo, hi = np.searchsorted(imu_t, s.t_ns - 1_000_000_000), np.searchsorted(imu_t, s.t_ns + 1_000_000_000)
        gyro_std = round(float(gyro[lo:hi].std()), 5) if hi - lo > 1 else None
        stats[s.index] = {"luma_mean": luma, "sharpness": sharp, "gyro_std_rad_s": gyro_std}
    return stats


def _labellers(services: Services) -> dict[str, FrameLabeller]:
    return services.labellers()


def _step_labelling(state: State, services: Services) -> dict[str, Any]:
    cfg = state.config
    samples = _samples(state)
    missing = [str(p) for s in samples for p in (s.image, s.other_image) if p is not None and not p.is_file()]
    if missing:
        raise PipelineError(f"{len(missing)} sampled frames are missing (e.g. {missing[:3]}); re-run with --force.")
    cache_dir = state.paths.out / "labels"
    per_model: dict[str, Any] = {}
    _tables, imu_t, imu = _load_tables(state.paths)
    for key, labeller in _labellers(services).items():
        t0 = time.perf_counter()
        labeller.begin_session(samples, imu_t, imu, imu_available=state.reader.imu_unavailable is None)
        results = label_samples(labeller, samples, cache_dir / f"labels_{key}.jsonl",
                                max_failure_ratio=cfg.get("labelling.max_failure_ratio"))
        ok = sum(1 for r in results.values() if r.get("status") == "ok")
        per_model[key] = {"samples": len(samples), "ok": ok, "failed_or_invalid": len(samples) - ok,
                          "elapsed_s": round(time.perf_counter() - t0, 1), "provides": list(labeller.provides)}
        logger.info("Model '%s': %d/%d samples labelled successfully.", key, ok, len(samples))
    stats = _frame_stats(samples, imu_t, imu)
    write_json_atomic(state.paths.out / "labels" / "frame_stats.json", {str(k): v for k, v in stats.items()})
    return {"samples": len(samples), "interval_s": cfg.get("labelling.interval_seconds"), "eye": _eyes(state)[0],
            "models": per_model, "model_info": services.model_info}


def _load_labels(state: State, services: Services) -> dict[str, dict[int, dict[str, Any]]]:
    out = {}
    for key in services.config.get("models"):
        path = state.paths.out / "labels" / f"labels_{key}.jsonl"
        rows: dict[int, dict[str, Any]] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("status") == "ok" or int(row["sample"]) not in rows:
                        rows[int(row["sample"])] = row
        out[key] = rows
    return out


CONTENT_DISABLED = "content screen (section 3) disabled: content.enabled is false"


def _content_model(cfg: ResolvedConfig) -> str | None:
    """The content model, or None when the content screen is disabled."""
    if not cfg.get("content.enabled"):
        return None
    return next(k for k, spec in cfg.get("models").items() if "content" in spec["provides"])


def _strip_images(state: State, samples: list[Sample]) -> list[bytes]:
    import cv2  # noqa: PLC0415

    width = state.config.get("labelling.strip_image_width")
    images = []
    for s in samples:
        img = cv2.imread(str(s.image))
        if img is None:
            raise PipelineError(f"Cannot read sampled frame {s.image}")
        if img.shape[1] > width:
            img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, state.config.get("labelling.jpeg_quality")])
        if not ok:
            raise PipelineError(f"Cannot encode {s.image}")
        images.append(buf.tobytes())
    return images


def _ask_strip(state: State, services: Services, prompt_name: str, samples: list[Sample], duration_s: float,
               cache_key: str) -> dict[str, Any]:
    cache_path = state.paths.out / "labels" / "strips.json"
    cache = read_json(cache_path) if cache_path.is_file() else {}
    frames = [s.index for s in samples]
    entry = cache.get(cache_key)
    if entry and entry.get("frames") == frames and entry.get("prompt") == prompt_name and entry.get("status") == "ok":
        return entry
    labeller = _labellers(services)[_content_model(state.config)]
    if not isinstance(labeller, VlmLabeller):
        raise PipelineError("The content model must be a vision-language model.")
    prompt = get_prompt(prompt_name)
    values = {"count": len(samples), "duration": f"{duration_s:.0f}", "times": ", ".join(f"{s.t_s:.0f}" for s in samples)}
    answer = labeller.ask(prompt.system.format(**values), prompt.render(**values), _strip_images(state, samples),
                          prompt.response_schema)
    entry = {"frames": frames, "prompt": prompt_name, "times_s": [round(s.t_s, 3) for s in samples], **answer}
    cache[cache_key] = entry
    write_json_atomic(cache_path, cache)
    if answer["status"] != "ok":
        raise InferenceError(f"Content model could not judge {cache_key}: {answer.get('error')}")
    return entry


def _step_content_recording(state: State, services: Services) -> dict[str, Any]:
    cfg = state.config
    content_cfg = cfg.get("content")
    model = _content_model(cfg)
    if model is None:
        logger.info("Content screen skipped (content.enabled: false).")
        return {"checked": False, "reason": CONTENT_DISABLED, "findings": []}
    samples = _samples(state)
    times = {s.index: round(s.t_s, 3) for s in samples}
    labels = _load_labels(state, services)
    flags = recording_frame_flags(labels[model], times, content_cfg)
    duration = state.data["technical"].get("duration_s") or 0.0
    strip = _ask_strip(state, services, cfg.get("inference.recording_prompt"),
                       evenly_spaced(samples, content_cfg["recording_strip_frames"]), duration, "recording")
    findings = list(flags["findings"])
    no_work = recording_strip_finding(strip)
    if no_work:
        findings.append(no_work)
    for f in findings:
        logger.warning("Content: %s", f["message"])
    return {"checked": True, "content_model": model, **flags, "findings": findings, "recording_strip": strip,
            "fail_reasons": [{"section": "content", "check": f["finding"], "message": f["message"]} for f in findings]}


def _step_cropping(state: State, services: Services) -> dict[str, Any]:
    cfg = state.config
    crop = cfg.get("cropping")
    fps = cfg.get("technical.expected_fps")
    left, _ = _eyes(state)
    tables, _, _ = _load_tables(state.paths)
    n = len(tables[left])
    t = tables[left]["t_ns"].astype(np.int64)
    times_s = (t - t[0]) / 1e9
    samples = _samples(state)
    labels = _load_labels(state, services)
    provides = {k: spec["provides"] for k, spec in cfg.get("models").items()}
    sample_rows = [{"index": s.index, "frame": s.frame, "t_s": s.t_s} for s in samples]
    decisions = decide_samples(sample_rows, labels, provides, min_models=crop["min_models_per_decision"])
    cuts = flagged_runs(decisions, "idle", crop["idle_cut_seconds"], "idle") + \
        flagged_runs(decisions, "no_hands", crop["no_hands_cut_seconds"], "no_hands")
    ranges = [(c.first_frame, c.last_frame) for c in cuts]
    defects = []
    if crop["cut_defects"]:
        defects = defect_frame_ranges(state.data["technical"].get("defect_intervals_s", []), fps, n)
        ranges += defects
    keyframes = common_keyframes(tables["cam0"], tables["cam1"]).tolist()
    pieces = plan_pieces(n_frames=n, frame_times_s=times_s, keyframes=keyframes, cuts=ranges, fps=fps,
                         min_piece_seconds=crop["min_piece_seconds"])
    kept = [p for p in pieces if p.kept]
    late = [p.to_dict() for p in kept if p.start_snap_s > crop["keyframe_snap_max_seconds"] + 1e-6]
    if late:
        logger.warning("%d piece(s) start more than %s s after their raw start (keyframes missing).", len(late),
                       crop["keyframe_snap_max_seconds"])
    write_json_atomic(state.paths.out / "labels" / "decisions.json", [d.__dict__ for d in decisions])
    logger.info("Cropping: %d idle cut(s), %d no-hands cut(s), %d defect cut(s); %d piece(s), %d kept (%.1f s)",
                sum(c.reason == "idle" for c in cuts), sum(c.reason == "no_hands" for c in cuts), len(defects),
                len(pieces), len(kept), sum(p.duration_s for p in kept))
    return {"cuts": [c.to_dict() for c in cuts], "defect_cuts_frames": defects, "pieces": [p.to_dict() for p in pieces],
            "kept_pieces": len(kept), "kept_seconds": round(sum(p.duration_s for p in kept), 3),
            "pieces_starting_late": late, "recording_seconds": round(float(times_s[-1]) + 1 / fps, 3) if n else 0.0,
            "fail_reasons": [] if kept else [{"section": "cropping", "check": "nothing_left",
                                              "message": f"no piece of {crop['min_piece_seconds']} s or more is left"}]}


def _step_content_clips(state: State, services: Services) -> dict[str, Any]:
    cfg = state.config
    content_cfg = cfg.get("content")
    samples = _samples(state)
    times = {s.index: round(s.t_s, 3) for s in samples}
    by_frame = {s.frame: s for s in samples}
    labels = _load_labels(state, services)
    model = _content_model(cfg)
    step = state.data["technical"]["sample_step_frames"]
    results = []
    for number, piece in enumerate(p for p in state.data["cropping"]["pieces"] if p["kept"]):
        first = -(-piece["start"] // step) * step
        inside = [by_frame[f] for f in range(first, piece["stop"], step) if f in by_frame]
        if model is None:                      # content screen disabled: every piece is kept
            results.append({"start": piece["start"], "stop": piece["stop"], "duration_s": piece["duration_s"],
                            "samples": len(inside), "checked": False, "strip": None, "findings": [], "kept": True})
            continue
        strip = None
        if len(inside) >= 2:
            strip = _ask_strip(state, services, cfg.get("inference.clip_prompt"),
                               evenly_spaced(inside, content_cfg["clip_strip_frames"]), piece["duration_s"],
                               f"piece_{piece['start']}_{piece['stop']}")
        findings = clip_findings([s.index for s in inside], labels[model], strip, times, content_cfg)
        for f in findings:
            logger.warning("Piece %d (%.0f-%.0f s): %s", number, times_at(state, piece["start"]),
                           times_at(state, piece["stop"] - 1), f["message"])
        results.append({"start": piece["start"], "stop": piece["stop"], "duration_s": piece["duration_s"],
                        "samples": len(inside), "checked": True, "strip": strip, "findings": findings,
                        "kept": not findings})
    kept = [r for r in results if r["kept"]]
    return {"pieces": results, "kept_pieces": len(kept), "kept_seconds": round(sum(r["duration_s"] for r in kept), 3),
            "fail_reasons": [] if kept else [{"section": "content", "check": "nothing_left_after_clip_screen",
                                              "message": "every remaining piece was dropped by the content screen"}]}


def times_at(state: State, frame: int) -> float:
    left, _ = _eyes(state)
    t = np.load(state.paths.stream / f"{left}.npy", mmap_mode="r")["t_ns"]
    return (int(t[frame]) - int(t[0])) / 1e9


def _step_clips(state: State, services: Services) -> dict[str, Any]:
    cfg = state.config
    left, right = _eyes(state)
    tables, imu_t, imu = _load_tables(state.paths)
    calibration = state.data["calibration"]
    ffmpeg = resolve_executable(cfg.get("tools.ffmpeg"))
    ffprobe = resolve_executable(cfg.get("tools.ffprobe"))
    meta = {"session": state.session.describe(), "fps_nominal": cfg.get("technical.expected_fps"),
            "calibration_result": calibration["result"], "eyes_swapped_vs_labels": calibration["eye_order"]["eyes_swapped"],
            "imu_time_offset_to_cam0_s": ((calibration.get("imu") or {}).get("imu_info") or {}).get("time_offset_to_cam0_s"),
            "time_reference": "device monotonic clock (ns); t_clip_s = seconds since the clip's first left frame"}
    if state.paths.clips.exists():
        shutil.rmtree(state.paths.clips)
    finalize: Callable[[Path], None] | None = None
    if state.upload_prefix is not None:
        bucket = processed_bucket(cfg)

        def upload_and_delete(path: Path) -> None:
            """Upload a verified clip MP4 right away and delete the local copy (one MP4 on disk at a time)."""
            name = state.upload_prefix + path.relative_to(state.paths.out).as_posix()
            services.storage.upload(path, bucket, name, overwrite=True, content_type=guess_content_type(path))
            state.uploaded.add(name)
            path.unlink()
            logger.info("Uploaded and deleted %s", path.relative_to(state.paths.out).as_posix())

        finalize = upload_and_delete
    clips = []
    first_ns = int(tables[left]["t_ns"][0])
    reader = state.reader
    stream = reader.stream_packets_from(state.paths.stream)
    crops = state.data["technical"].get("eye_crops")
    meta["recording_format"] = reader.format
    if reader.imu_unavailable:
        meta["imu_note"] = reader.imu_unavailable
    for number, piece in enumerate(p for p in state.data["content_clips"]["pieces"] if p["kept"]):
        clip_id = f"clip_{number:03d}"
        info = write_clip(clip_dir=state.paths.clips / clip_id, clip_id=clip_id, ffmpeg=ffmpeg, ffprobe=ffprobe,
                          stream=stream, tables=tables,
                          left_cam=left, right_cam=right, start=piece["start"], stop=piece["stop"],
                          fps=int(cfg.get("technical.expected_fps")), imu_t=imu_t, imu=imu, first_frame_ns=first_ns,
                          metadata=meta, timeout_seconds=cfg.get("technical.decode_timeout_seconds"),
                          finalize=finalize, layout=reader.layout, codec=reader.codec, crops=crops)
        clips.append(info)
        logger.info("%s: %d frames, %.1f s, %d IMU samples", clip_id, info["frames"], info["duration_s"],
                    info["imu_samples"])
    return {"clips": clips, "uploaded_during_step": sorted(state.uploaded)}


# ---------------------------------------------------------------------------------------------
# Report, upload, clean-up
# ---------------------------------------------------------------------------------------------

def _write_labels_csv(state: State, services: Services) -> None:
    if "labelling" not in state.data:
        return
    samples = _samples(state)
    labels = _load_labels(state, services)
    stats_path = state.paths.out / "labels" / "frame_stats.json"
    stats = read_json(stats_path) if stats_path.is_file() else {}
    decisions_path = state.paths.out / "labels" / "decisions.json"
    decisions = {d["index"]: d for d in read_json(decisions_path)} if decisions_path.is_file() else {}
    content_model = _content_model(state.config)
    content_fields = ["phone_use", "minor_visible", "third_person_or_fixed_camera", "lens_covered_or_smudged",
                      "desk_or_computer_work", "private_moment", "description"] if content_model else []
    specs = state.config.get("models")
    models = list(specs)
    # Extra per-model columns of the local (non-AI) labellers.
    extra = {"mediapipe": ["hands_left_eye", "hands_right_eye", "other_hands"],
             "motion": ["changed_prev", "changed_next", "gyro_median_rad_s"]}
    header = ["sample", "frame", "t_s", "t_left_ns"]
    for m in models:
        header += [f"{m}_status", f"{m}_hands", f"{m}_activity"] + [f"{m}_{f}" for f in extra.get(specs[m]["provider"], [])]
    header += [f"{content_model}_{f}" for f in content_fields] + ["idle", "no_hands", "luma_mean", "sharpness",
                                                                  "gyro_std_rad_s"]
    path = state.paths.out / "labels.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for s in samples:
            row: list[Any] = [s.index, s.frame, f"{s.t_s:.3f}", s.t_ns]
            for m in models:
                rec = labels.get(m, {}).get(s.index, {})
                lab = rec.get("labels", {}) if rec.get("status") == "ok" else {}
                row += [rec.get("status", "missing"), lab.get("hands", ""), lab.get("activity", "")]
                row += [lab.get(f, "") for f in extra.get(specs[m]["provider"], [])]
            if content_model:
                rec = labels.get(content_model, {}).get(s.index, {})
                lab = rec.get("labels", {}) if rec.get("status") == "ok" else {}
                row += [lab.get(f, "") for f in content_fields]
            d = decisions.get(s.index)
            row += [d["idle"] if d else "", d["no_hands"] if d else ""]
            st = stats.get(str(s.index), {})
            row += [st.get("luma_mean", ""), st.get("sharpness", ""), st.get("gyro_std_rad_s", "")]
            writer.writerow(row)


def _write_report(state: State, services: Services, upload_prefix: str | None, max_segments: int | None) -> dict[str, Any]:
    verdict = state.verdict or PASS
    d = state.data
    tech = d.get("technical")
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "verdict": verdict,
        "fail_reasons": state.fail_reasons,
        "session": {**state.session.describe(), "format": state.listing.format,
                    "format_description": state.reader.description if state.reader else None,
                    "segments": len(state.listing.segments),
                    "bytes": state.listing.total_bytes, "identity": state.listing.identity()},
        "partial_test_run": max_segments is not None,
        "run_id": state.manifest.run_id,
        "generated_at": utc_now_iso(),
        "pipeline_version": __version__,
        "config_fingerprint": state.config.fingerprint,
        "uploaded_to": f"gs://{processed_bucket(state.config)}/{upload_prefix}" if upload_prefix else None,
        "steps_run": [s for s in STEPS if s in d],
        "technical": None,
        "calibration": None,
        "labelling": d.get("labelling"),
        "content": None,
        "cropping": None,
        "clips": [],
    }
    if "extract" in d and not d["extract"].get("extracted"):
        report["technical"] = {"passed": False, "checks": d["extract"]["required_files"]}
    if state.reader is not None and state.reader.imu_unavailable:
        report["imu"] = {"checked": False, "reason": state.reader.imu_unavailable}
    if tech:
        report["technical"] = {k: v for k, v in tech.items() if k not in {"decode", "calibration_snapshot", "fail_reasons"}}
        report["technical"]["decode"] = {cam: {k: v for k, v in r.items() if k != "command"}
                                         for cam, r in (tech.get("decode") or {}).items()}
    if "calibration" in d:
        report["calibration"] = {k: d["calibration"][k] for k in ("result", "reason", "transform_reading",
                                                                    "median_vertical_px", "p95_vertical_px",
                                                                    "inlier_matches", "pairs_with_matches", "eye_order")}
        write_json_atomic(state.paths.out / "calibration.json", d["calibration"])
    if not state.config.get("content.enabled"):
        report["content"] = {"checked": False, "reason": CONTENT_DISABLED}
    elif "content_recording" in d:
        report["content"] = {"recording": {k: v for k, v in d["content_recording"].items() if k != "fail_reasons"}}
    if "content_clips" in d and state.config.get("content.enabled"):
        report["content"] = report["content"] or {}
        report["content"]["clips"] = d["content_clips"]["pieces"]
    if "cropping" in d:
        crop = d["cropping"]
        final = d.get("content_clips")
        report["cropping"] = {"cuts": crop["cuts"], "defect_cuts_frames": crop["defect_cuts_frames"],
                              "pieces": crop["pieces"], "recording_seconds": crop["recording_seconds"],
                              "kept_pieces": final["kept_pieces"] if final else crop["kept_pieces"],
                              "kept_seconds": final["kept_seconds"] if final else crop["kept_seconds"],
                              "pieces_starting_late": crop["pieces_starting_late"]}
    if "clips" in d:
        report["clips"] = [{k: c[k] for k in ("clip_id", "frames", "duration_s", "recording_time_s", "eyes", "files")}
                           | {"folder": f"clips/{c['clip_id']}/"} for c in d["clips"]["clips"]]
    _write_labels_csv(state, services)
    write_json_atomic(state.paths.out / "qa_report.json", report)
    return report


def _check_prefix(state: State, prefix: str) -> None:
    cfg = state.config
    root = cfg.get("source.processed_root")
    into_source = processed_bucket(cfg) == cfg.get("source.bucket") and prefix.startswith(cfg.get("source.raw_root"))
    if not prefix.startswith(root) or into_source:
        raise PipelineError(f"Refusing to write outside gs://{processed_bucket(cfg)}/{root}: {prefix}")


def _prepare_target(state: State, services: Services) -> None:
    """Before anything is uploaded: if a complete result is already there (only possible with --force or
    upload.on_existing: overwrite; 'skip' skipped the session), delete its qa_report.json so the folder
    reads as incomplete until the new result is complete."""
    prefix = state.upload_prefix
    _check_prefix(state, prefix)
    bucket = processed_bucket(state.config)
    if services.storage.exists(bucket, prefix + "qa_report.json"):
        logger.info("Replacing the existing result in gs://%s/%s (its qa_report.json is removed first)", bucket, prefix)
        services.storage.delete(bucket, prefix + "qa_report.json")


def _upload(state: State, services: Services, prefix: str) -> None:
    """Upload out/ to the prefix, qa_report.json last.

    The folder holds no complete result at this point (see _prepare_target), so files are written with
    overwrite. Objects of an earlier result or an interrupted attempt that the new result does not have
    are deleted just before qa_report.json is uploaded. Only this session's own result folder is
    touched (never runs/ below it, never raw/).
    """
    _check_prefix(state, prefix)
    bucket = processed_bucket(state.config)
    out = state.paths.out
    files = sorted(p for p in out.rglob("*") if p.is_file() and not p.name.endswith((".tmp", ".part", ".tmp.mp4")))
    report = out / "qa_report.json"
    logs = [p for p in files if p.parent == state.paths.logs]
    ordered = [p for p in files if p != report and p not in logs] + logs + [report]
    names = {p: prefix + p.relative_to(out).as_posix() for p in ordered}
    keep = set(names.values()) | state.uploaded
    existing = [o.name for o in services.storage.list_objects(bucket, prefix) if not o.name.startswith(prefix + "runs/")]
    stale = sorted(set(existing) - keep - {prefix + "qa_report.json"})
    total = sum(p.stat().st_size for p in ordered)
    logger.info("Uploading %d files (%.2f GB) to gs://%s/%s (qa_report.json last)", len(ordered), total / 1e9,
                bucket, prefix)
    for path in ordered:
        if path == report and stale:
            for name in stale:
                services.storage.delete(bucket, name)
            logger.info("Deleted %d object(s) of an earlier result that the new one does not have.", len(stale))
        services.storage.upload(path, bucket, names[path], overwrite=True, content_type=guess_content_type(path))
        state.uploaded.add(names[path])
    logger.info("Upload complete: gs://%s/%sqa_report.json", bucket, prefix)


def _cleanup(paths: SessionPaths) -> None:
    for folder in (paths.work, paths.clips):
        shutil.rmtree(folder, ignore_errors=True)
    logger.info("Deleted the local working files (index, sampled frames, clips); reports kept in %s", paths.out)
