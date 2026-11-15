"""Command-line runner: download -> analyse -> score -> reports."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import shutil
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

import yaml

from . import __version__
from .config import ConfigError, load_config, resolve_path
from .download import DownloadError, ensure_downloaded
from .manifest import Clip, load_manifest
from .util import LOG, read_json, setup_logging, sha256_file, write_json

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


# ----------------------------------------------------------------------------- worker (separate process)
def _init_worker(level: str, native_log: str) -> None:
    """Worker start-up: MediaPipe's C++ runtime prints INFO lines straight to the process
    stderr; send those to a log file and keep the console for our own progress messages."""
    import os
    console = os.fdopen(os.dup(2), "w", buffering=1, encoding="utf-8", errors="replace")
    with open(native_log, "ab") as fh:
        os.dup2(fh.fileno(), 2)
    setup_logging(level, stream=console)


def _analyse_worker(clip: Clip, video: str, out_dir: str, config: dict, model: str, model_sha: str,
                    do_analysis: bool) -> dict:
    """Hand/picture analysis (if needed) followed by the full-resolution blockiness measurement (if needed)."""
    from .analyse import analyse_clip          # imported here: MediaPipe loads only in workers
    try:
        if do_analysis:
            meta = analyse_clip(clip, Path(video), Path(out_dir), config, Path(model), model_sha)
            if meta.get("status") != "ok":
                return meta
        else:
            meta = read_json(Path(out_dir) / "analysis" / "analysis.json")
        if _needs_blocking(clip, Path(out_dir), config):
            _run_blocking(clip, Path(video), Path(out_dir), config, meta)
        return meta
    except Exception as exc:  # noqa: BLE001 (reported per clip, the run continues)
        return {"status": "failed", "clip_id": clip.clip_id, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}


def _blocking_fingerprint(clip: Clip, config: dict) -> str:
    from .blocking import BLOCKING_VERSION, PERIODS
    from .util import fingerprint
    return fingerprint({"version": BLOCKING_VERSION, "periods": PERIODS,
                        "blockiness_fps": config["analysis"]["blockiness_fps"],
                        "source": [clip.source_url, clip.size_bytes, clip.md5_base64]})


def _needs_blocking(clip: Clip, out_dir: Path, config: dict) -> bool:
    path = out_dir / "analysis" / "grid_blockiness.json"
    if not path.is_file():
        return True
    data = read_json(path)
    return data.get("status") != "ok" or data.get("fingerprint") != _blocking_fingerprint(clip, config)


def _run_blocking(clip: Clip, video: Path, out_dir: Path, config: dict, meta: dict) -> None:
    from .blocking import measure_blockiness
    probe = meta["probe"]
    LOG.info("[%s] measuring grid blockiness at full resolution", clip.clip_id)
    result = measure_blockiness(config["tools"]["ffmpeg"], video, width=int(probe["display_width"]),
                                height=int(probe["display_height"]), frame_stride=int(meta["block_stride"]))
    result["fingerprint"] = _blocking_fingerprint(clip, config)
    write_json(out_dir / "analysis" / "grid_blockiness.json", result)
    if result["status"] != "ok":
        raise RuntimeError(f"blockiness measurement failed: {result.get('error')}")
    LOG.info("[%s] grid blockiness %.3f (%d frames, %.0fs)", clip.clip_id, result["blockiness"],
             result["frames"], result["elapsed_s"])


# ----------------------------------------------------------------------------- helpers
def _check_tools(config: dict) -> None:
    for key in ("ffmpeg", "ffprobe"):
        exe = config["tools"][key]
        if not (shutil.which(exe) or Path(exe).is_file()):
            raise SystemExit(f"'{exe}' not found. Install FFmpeg >= 6.0 and put it on PATH, or set tools.{key} "
                             "in the config.")
    import subprocess
    filters = subprocess.run([config["tools"]["ffmpeg"], "-hide_banner", "-filters"], capture_output=True,
                             text=True, errors="replace").stdout
    if " blockdetect " not in filters:
        raise SystemExit("This FFmpeg build has no 'blockdetect' filter. Install FFmpeg >= 6.0.")


def _ensure_model(model: Path) -> str:
    if not model.is_file():
        import urllib.request
        LOG.info("Downloading the MediaPipe hand model ...")
        model.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, model)  # noqa: S310 (fixed official URL)
    digest = sha256_file(model)
    if digest != MODEL_SHA256:
        LOG.warning("Hand model checksum differs from the validated model (%s); results may differ.", digest[:12])
    return digest


def _select(clips: list[Clip], only: str | None) -> list[Clip]:
    if not only:
        return clips
    wanted = {x.strip() for x in only.split(",") if x.strip()}
    chosen = [c for c in clips if c.clip_id in wanted or str(c.clip_no) in wanted or f"{c.clip_no:02d}" in wanted]
    if not chosen:
        raise SystemExit(f"--only matched no clips: {only}")
    return chosen


def _needs_analysis(clip: Clip, out_dir: Path, config: dict, model_sha: str) -> bool:
    from .analyse import analysis_fingerprint    # light import (no MediaPipe initialisation)
    meta_file = out_dir / "analysis" / "analysis.json"
    if not meta_file.is_file():
        return True
    meta = read_json(meta_file)
    return meta.get("status") != "ok" or meta.get("fingerprint") != analysis_fingerprint(clip, config, model_sha)


# ----------------------------------------------------------------------------- datasets
def _dataset_dir(results_root: Path, name: str) -> Path:
    from .sources import slug
    clean = slug(name).replace("-", "_")
    if not clean:
        raise SystemExit(f"Invalid dataset name: {name!r}")
    return results_root / clean


def _prepare_dataset(csv_path: Path, ds_dir: Path, refresh: bool) -> Path:
    """Copy the clip list into the dataset folder and (re)build its manifest when the list changed."""
    from .sources import SourceListError, resolve_source_list, write_manifest
    ds_dir.mkdir(parents=True, exist_ok=True)
    source_copy, manifest = ds_dir / "source_list.csv", ds_dir / "clip_manifest.csv"
    new_bytes = csv_path.read_bytes()
    unchanged = source_copy.is_file() and source_copy.read_bytes() == new_bytes and manifest.is_file()
    if unchanged and not refresh:
        LOG.info("Clip list unchanged; using %s", manifest)
        return manifest
    LOG.info("Resolving download sources for %s ...", csv_path.name)
    try:
        resolved = resolve_source_list(csv_path)
    except SourceListError as exc:
        raise SystemExit(f"Clip list error: {exc}") from exc
    write_manifest(manifest, resolved)
    source_copy.write_bytes(new_bytes)
    ok = sum(1 for r in resolved if r["status"] == "OK")
    LOG.info("%d of %d clips resolved, %d SOURCE_MISSING -> %s", ok, len(resolved), len(resolved) - ok, manifest)
    return manifest


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_hand_qc.py",
        description="Hand & technical QC checklist for the clips listed in a CSV file. Each clip list gets its own "
                    "results folder: results/<name>/ (name = CSV file name unless --name is given).")
    parser.add_argument("--csv", type=Path, required=True,
                        help="clip list CSV (columns: Dataset, Category, Task, Camera, URL_or_File, Notes)")
    parser.add_argument("--name", help="results folder name (default: the CSV file name without .csv)")
    parser.add_argument("--config", type=Path, help="YAML file with settings to override (merged on top of "
                                                     "config/hand_qc.yaml)")
    parser.add_argument("--only", help="comma-separated clip numbers or ids, e.g. 1,58,64")
    parser.add_argument("--workers", type=int, help="clips analysed in parallel (default: analysis.workers)")
    parser.add_argument("--score-only", action="store_true",
                        help="do not download or analyse; re-score existing analyses and rebuild the reports")
    parser.add_argument("--reanalyse", action="store_true", help="analyse again even if a valid analysis exists")
    parser.add_argument("--refresh-sources", action="store_true",
                        help="resolve the download URLs again even if the clip list is unchanged")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (ConfigError, OSError, yaml.YAMLError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if args.workers:
        config["analysis"]["workers"] = args.workers
    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        print(f"Clip list not found: {csv_path}", file=sys.stderr)
        return 2

    dataset = args.name or csv_path.stem
    results = _dataset_dir(resolve_path(config, "results_dir"), dataset)
    videos_dir = results / "videos"
    setup_logging(args.log_level, results / "logs" / f"run_{dt.datetime.now():%Y%m%d_%H%M%S}.log")
    started = time.monotonic()
    LOG.info("Hand QC %s | clip list: %s | results: %s", __version__, csv_path, results)

    manifest_path = _prepare_dataset(csv_path, results, args.refresh_sources)
    all_clips = load_manifest(manifest_path)
    clips = _select(all_clips, args.only)
    available = [c for c in clips if c.available]
    LOG.info("%d clip(s) selected: %d available, %d source missing", len(clips), len(available),
             len(clips) - len(available))

    model = resolve_path(config, "model")
    model_sha = _ensure_model(model)
    errors: dict[str, str] = {}

    if not args.score_only:
        _check_tools(config)
        todo: dict[str, bool] = {}          # clip_id -> hand analysis needed (else: only blockiness)
        for c in available:
            out_dir = videos_dir / c.clip_id
            if args.reanalyse or _needs_analysis(c, out_dir, config, model_sha):
                todo[c.clip_id] = True
            elif _needs_blocking(c, out_dir, config):
                todo[c.clip_id] = False
        LOG.info("%d clip(s) to analyse, %d already analysed", len(todo), len(available) - len(todo))
        if todo:
            errors.update(_download_and_analyse([c for c in available if c.clip_id in todo], todo, config,
                                                videos_dir, model, model_sha, args.log_level))

    # Scoring is cheap: every analysed clip is re-scored with the CURRENT settings, so the checklist
    # never mixes thresholds (--only limits the download/analysis step only).
    from .report import build_reports
    from .score import score_clip
    summaries: dict[str, dict] = {}
    for c in (x for x in all_clips if x.available):
        out_dir = videos_dir / c.clip_id
        meta_file = out_dir / "analysis" / "analysis.json"
        if c.clip_id in errors:
            continue
        if not meta_file.is_file() or read_json(meta_file).get("status") != "ok":
            errors[c.clip_id] = "not analysed yet" if not meta_file.is_file() else \
                read_json(meta_file).get("error", "analysis failed")
            continue
        if _needs_blocking(c, out_dir, config):
            errors[c.clip_id] = "blockiness not measured yet (run without --score-only)"
            continue
        try:
            summaries[c.clip_id] = score_clip(c, out_dir, config)
        except Exception as exc:  # noqa: BLE001
            LOG.error("[%s] scoring failed: %s", c.clip_id, exc)
            LOG.debug(traceback.format_exc())
            errors[c.clip_id] = f"scoring failed: {exc}"

    outputs = build_reports(all_clips, summaries, errors, config, results, dataset=results.name,
                            source_csv=csv_path.name)
    with (results / "config_used.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)
    write_json(results / "run_info.json", {
        "finished_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": results.name, "clip_list": str(csv_path),
        "handqc_version": __version__, "model_sha256": model_sha, "clips_scored": len(summaries),
        "clips_with_errors": errors, "elapsed_s": round(time.monotonic() - started, 1)})

    n_pass = sum(1 for s in summaries.values() if s["overall"] == "PASS")
    LOG.info("Done in %.0fs: %d scored (%d PASS, %d FAIL), %d error(s), %d source missing", time.monotonic() - started,
             len(summaries), n_pass, len(summaries) - n_pass, len(errors), sum(1 for c in all_clips if not c.available))
    for path in outputs:
        LOG.info("  -> %s", path)
    return 1 if errors else 0


def _download_and_analyse(todo: list[Clip], full: dict[str, bool], config: dict, videos_dir: Path, model: Path,
                          model_sha: str, level: str) -> dict[str, str]:
    cache = resolve_path(config, "cache_dir")
    d = config["download"]
    errors: dict[str, str] = {}
    workers = int(config["analysis"]["workers"])
    with ThreadPoolExecutor(max_workers=2) as downloads, \
            ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                initargs=(level, str(videos_dir.parent / "logs" / "mediapipe_native.log"))) as pool:
        pending: dict[Future, tuple[str, Clip]] = {}
        for c in todo:
            fut = downloads.submit(ensure_downloaded, c, cache, retries=int(d["retries"]),
                                   timeout=float(d["timeout_seconds"]), verify_md5=bool(d["verify_md5"]))
            pending[fut] = ("download", c)
        done_count = 0
        while pending:
            finished, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in finished:
                kind, c = pending.pop(fut)
                if kind == "download":
                    try:
                        video = fut.result()
                    except DownloadError as exc:
                        LOG.error("[%s] %s", c.clip_id, exc)
                        errors[c.clip_id] = f"download failed: {exc}"
                        continue
                    out_dir = videos_dir / c.clip_id
                    pending[pool.submit(_analyse_worker, c, str(video), str(out_dir), config, str(model),
                                        model_sha, full[c.clip_id])] = ("analyse", c)
                else:
                    meta = fut.result()
                    done_count += 1
                    if meta.get("status") != "ok":
                        LOG.error("[%s] analysis failed: %s", c.clip_id, meta.get("error"))
                        LOG.debug(meta.get("traceback", ""))
                        errors[c.clip_id] = f"analysis failed: {meta.get('error')}"
                    else:
                        LOG.info("[%s] analysis complete (%d/%d)", c.clip_id, done_count, len(todo))
    return errors


if __name__ == "__main__":
    sys.exit(main())
