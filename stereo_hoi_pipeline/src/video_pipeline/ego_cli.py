"""Command line for the egocentric QA batch (recordings in Cloud Storage, every supported format:
MCAP, robocap, Hydra, EGO-CAM live and capture.egoc; the format of each session folder is detected).

    python scripts/run_gcs_batch.py --source gs://op-datoric/raw/Op_DD/31-10-2026/ --list-sessions
    python scripts/run_gcs_batch.py --source gs://op-datoric/raw/Op_DD/31-10-2026/ \
        --dest gs://op-datoric/processed/Op_DD/31-10-2026/ --start 1 --end 10
    python scripts/run_gcs_batch.py --sessions-file sessions_10.txt
    python scripts/run_gcs_batch.py --session gs://op-datoric/raw/Op_DD/<date>/<device>/<session>/

Exit codes: 0 every session PASS/FAIL/SKIPPED, 1 at least one session ended in ERROR,
2 usage/configuration error, 130 interrupted.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from video_pipeline import __version__
from video_pipeline.core.config import find_project_root
from video_pipeline.core.ego_config import load_ego_configuration, processed_bucket
from video_pipeline.core.env import load_environment
from video_pipeline.core.errors import ConfigError, PipelineError
from video_pipeline.core.gcp_auth import get_credentials
from video_pipeline.core.io import utc_now, write_text_atomic
from video_pipeline.core.logging_setup import LoggingController
from video_pipeline.ingestion.formats import PER_SEGMENT_FORMATS, classify, detect_format
from video_pipeline.ingestion.gcs import GcsStorage, SessionRef, parse_gs_url

EXIT_OK, EXIT_ERRORS, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130
SUMMARY_FIELDS = ["session", "status", "reasons", "source_seconds", "kept_seconds", "clips", "calibration",
                  "elapsed_s"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_gcs_batch",
                                description="Egocentric video QA for recordings in Google Cloud Storage "
                                            "(MCAP, robocap, Hydra, EGO-CAM; the format is detected per session).")
    p.add_argument("--source", metavar="GS_URL",
                   help="Folder to read recordings from (every recording folder below it, in the numbered order "
                        "--list-sessions shows). Default: source.bucket + source.raw_root of the config.")
    p.add_argument("--dest", metavar="GS_URL",
                   help="Folder to write results to; each result goes to <dest>/<path of the session below the "
                        "source folder>/. Default: source.processed_bucket + source.processed_root of the config.")
    p.add_argument("--start", type=int, metavar="N",
                   help="First session to process: its number in the --list-sessions list (1 = first). Default 1.")
    p.add_argument("--end", type=int, metavar="N",
                   help="Last session to process (inclusive), as numbered by --list-sessions. Default: the last.")
    p.add_argument("--session", action="append", metavar="GS_URL",
                   help="Instead of the whole source folder: this session folder (repeatable).")
    p.add_argument("--sessions-file", type=Path, metavar="FILE",
                   help="Instead of the whole source folder: one session URL per line (# comments allowed).")
    p.add_argument("--list-sessions", nargs="?", const="", metavar="GS_URL",
                   help="Print the numbered list of sessions in the source folder (or in GS_URL) with their "
                        "size, mark the ones --start/--end select, then exit.")
    p.add_argument("--content-screen", choices=["on", "off"],
                   help="Switch for section 3 (minors, private moments, ...): overrides content.enabled. "
                        "'on' needs a content model in the config (not configured yet; see docs/ego_qa.md).")
    p.add_argument("--config", type=Path, help="YAML file overriding configs/ego_qa.yaml (only existing keys).")
    p.add_argument("--force", action="store_true",
                   help="Process again even if processed/ already has a qa_report.json; discard local partial work.")
    p.add_argument("--no-upload", action="store_true", help="Keep results locally; write nothing to the bucket.")
    p.add_argument("--max-segments", type=int, metavar="N",
                   help="TEST ONLY: use just the first N segments of each session (the report says so).")
    p.add_argument("--until", choices=["extract", "technical", "calibration", "labelling", "content_recording",
                                       "cropping", "content_clips", "clips"],
                   help="Stop after this step (no report, no upload). For testing.")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def read_sessions_file(path: Path) -> list[str]:
    urls = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            urls.append(line)
    return urls


def folder_prefix(url: str, what: str) -> tuple[str, str]:
    """gs://bucket/a/b -> ('bucket', 'a/b/'). A folder is required (not the bucket root)."""
    bucket, prefix = parse_gs_url(url.strip())
    prefix = prefix.strip("/")
    if not prefix or any(part in {".", ".."} for part in prefix.split("/")):
        raise PipelineError(f"{what} '{url}' must name a folder inside the bucket, e.g. gs://{bucket or '<bucket>'}/raw/")
    return bucket, prefix + "/"


def source_root(prefix: str) -> str:
    """The folder sessions are numbered and named below (before the folder is listed). A folder named
    session_* given as the source is itself the only session, so its parent folder is the root; for
    other formats this is found out after listing (see session_root)."""
    parts = prefix.rstrip("/").split("/")
    if parts[-1].startswith("session_"):
        if len(parts) == 1:
            raise PipelineError("A session folder directly in the bucket root cannot be a source folder.")
        return "/".join(parts[:-1]) + "/"
    return prefix


def session_root(bucket: str, prefix: str, found: Sequence[dict]) -> str:
    """Root below which the found sessions are named: the source folder itself, or its parent when the
    source folder is itself a recording (it must then not be in the bucket root)."""
    if any(s.get("folder", s["url"]) == f"gs://{bucket}/{prefix}" for s in found):
        parts = prefix.rstrip("/").split("/")
        if len(parts) == 1:
            raise PipelineError("A recording folder directly in the bucket root cannot be a source folder.")
        return "/".join(parts[:-1]) + "/"
    return prefix


def discover_sessions(storage: GcsStorage, bucket: str, prefix: str, segment_pattern: str) -> tuple[list[dict], int]:
    """Recordings below a prefix, sorted by path: [{'url', 'folder', 'format', 'bytes', 'files', 'segments',
    'segment'}], plus the number of folders skipped because their files are not a recording in a known format.

    A folder is a recording when the files directly inside it match a format (ingestion/formats);
    sub-folders are looked at separately (robocap keeps one session per sub-folder). In the formats whose
    segments are recordings of their own (robocap) every segment is one entry, '<folder>/#seg<N>'."""
    folders: dict[str, list] = defaultdict(list)
    for obj in storage.list_objects(bucket, prefix):
        if obj.name.endswith("/"):
            continue
        folders[obj.name.rsplit("/", 1)[0] + "/" if "/" in obj.name else ""].append(obj)
    found, skipped = [], 0
    for folder in sorted(folders):
        names = [o.name[len(folder):] for o in folders[folder]]
        fmt = detect_format(names, segment_pattern)
        if fmt is None or not folder:
            skipped += 1
            continue
        listing = classify(SessionRef(bucket=bucket, raw_root="", relative=folder.rstrip("/")), folders[folder],
                           segment_pattern)
        url = f"gs://{bucket}/{folder}"
        if fmt in PER_SEGMENT_FORMATS and listing.segments:
            for seg in listing.segments:
                found.append({"url": f"{url}#seg{seg.number}", "folder": url, "format": fmt, "bytes": seg.size,
                              "files": len(seg.files), "segments": 1, "segment": seg.number})
            continue
        found.append({"url": url, "folder": url, "format": fmt, "bytes": listing.total_bytes,
                      "files": len(listing.objects()), "segments": len(listing.segments), "segment": None})
    return found, skipped


def expand_segments(storage: GcsStorage, sessions: Sequence[SessionRef], segment_pattern: str) -> list[SessionRef]:
    """A folder of a per-segment format (robocap) given as one session becomes one session per segment."""
    out: list[SessionRef] = []
    for ref in sessions:
        if ref.segment is not None:
            out.append(ref)
            continue
        try:
            listing = classify(ref, storage.list_objects(ref.bucket, ref.prefix), segment_pattern)
        except PipelineError:
            out.append(ref)                 # the run reports what is wrong with it
            continue
        if listing.format in PER_SEGMENT_FORMATS and listing.segments:
            out.extend(ref.with_segment(seg.number) for seg in listing.segments)
        else:
            out.append(ref)
    return out


def select_range(items: Sequence, start: int | None, end: int | None) -> list:
    """Items start..end (1-based, inclusive)."""
    if not items:
        raise PipelineError("No sessions found.")
    first = 1 if start is None else start
    last = len(items) if end is None else end
    if not 1 <= first <= last <= len(items):
        raise PipelineError(f"--start {first} / --end {last}: there are {len(items)} sessions (numbered 1-{len(items)}); "
                            "start must be <= end.")
    return list(items[first - 1:last])


def format_session_list(found: Sequence[dict], skipped: int, start: int | None, end: int | None) -> str:
    first = 1 if start is None else start
    last = len(found) if end is None else end
    marked = start is not None or end is not None
    lines = [f"{'>' if marked and first <= n <= last else ' '}{n:4d}  {s['format']:<12} {s['bytes'] / 1e9:8.2f} GB"
             f"  {('one segment' if s.get('segment') is not None else str(s['segments']) + ' segments'):>12}  {s['url']}"
             for n, s in enumerate(found, start=1)]
    by_format: dict[str, int] = defaultdict(int)
    for s in found:
        by_format[s["format"]] += 1
    lines.append(f"{len(found)} recordings, {sum(s['bytes'] for s in found) / 1e9:.1f} GB"
                 + (f" ({', '.join(f'{k}: {v}' for k, v in sorted(by_format.items()))})" if found else "")
                 + (f"; {skipped} folder(s) without a recording in a known format not listed" if skipped else ""))
    if found and marked:
        chosen = found[first - 1:last]
        lines.append(f"'>' = selected by --start {first} --end {last}: {len(chosen)} sessions, "
                     f"{sum(s['bytes'] for s in chosen) / 1e9:.1f} GB")
    return "\n".join(lines)


def _summary_csv(rows: Sequence[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=SUMMARY_FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    if args.max_segments is not None and args.max_segments < 1:
        print("--max-segments must be >= 1", file=sys.stderr)
        return EXIT_USAGE
    for name in ("start", "end"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            print(f"--{name} must be >= 1 (sessions are numbered from 1)", file=sys.stderr)
            return EXIT_USAGE
    if args.session and args.sessions_file:
        print("Use either --session or --sessions-file, not both.", file=sys.stderr)
        return EXIT_USAGE
    controller = LoggingController(args.log_level or "INFO")
    try:
        try:
            project_root = find_project_root()
            overrides: dict = {"upload.enabled": False} if args.no_upload else {}
            if args.log_level:
                overrides["logging.console_level"] = args.log_level
            if args.source:
                bucket, prefix = folder_prefix(args.source, "--source")
                overrides.update({"source.bucket": bucket, "source.raw_root": source_root(prefix)})
            if args.dest:
                bucket, prefix = folder_prefix(args.dest, "--dest")
                overrides.update({"source.processed_bucket": bucket, "source.processed_root": prefix})
            if args.content_screen:
                overrides["content.enabled"] = args.content_screen == "on"
            config = load_ego_configuration(project_root, args.config, overrides)
        except (ConfigError, PipelineError) as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return EXIT_USAGE
        controller.set_console_level(config.get("logging.console_level"))
        environment = load_environment(project_root)
        try:
            credentials, credentials_desc = get_credentials(config.get("gcp.credentials"))
            storage = GcsStorage(credentials, config.get("gcp.project"))
        except PipelineError as exc:
            print(f"Google Cloud credentials: {exc}", file=sys.stderr)
            return EXIT_USAGE

        pattern = config.get("source.segment_pattern")
        if args.list_sessions is not None:
            try:
                if args.list_sessions:
                    bucket, prefix = folder_prefix(args.list_sessions, "--list-sessions")
                elif args.source:
                    bucket, prefix = folder_prefix(args.source, "--source")
                else:
                    bucket, prefix = config.get("source.bucket"), config.get("source.raw_root")
                found, skipped = discover_sessions(storage, bucket, prefix, pattern)
            except (PipelineError, ConfigError) as exc:
                print(f"Listing: {exc}", file=sys.stderr)
                return EXIT_USAGE
            print(format_session_list(found, skipped, args.start, args.end))
            return EXIT_OK

        try:
            if args.session or args.sessions_file:
                urls = args.session or read_sessions_file(args.sessions_file)
                sessions = expand_segments(storage, [SessionRef.from_url(u, expected_bucket=config.get("source.bucket"),
                                                                         raw_root=config.get("source.raw_root"))
                                                     for u in urls], pattern)
            elif args.source:
                bucket, prefix = folder_prefix(args.source, "--source")
                found, _ = discover_sessions(storage, bucket, prefix, pattern)
                if not found:
                    raise PipelineError(f"No recordings in a known format found below {args.source}.")
                root = session_root(bucket, prefix, found)
                if root != config.get("source.raw_root"):
                    # the source folder is itself a recording: name it relative to its parent
                    overrides["source.raw_root"] = root
                    config = load_ego_configuration(project_root, args.config, overrides)
                sessions = [SessionRef.from_url(s["url"], expected_bucket=config.get("source.bucket"),
                                                raw_root=config.get("source.raw_root")) for s in found]
            else:
                print("Say which recordings to process: --source FOLDER (optionally with --start/--end), "
                      "--session URL or --sessions-file FILE.", file=sys.stderr)
                return EXIT_USAGE
            sessions = select_range(sessions, args.start, args.end)
        except (PipelineError, ConfigError) as exc:
            print(f"Session list: {exc}", file=sys.stderr)
            return EXIT_USAGE
        if len({s.url for s in sessions}) != len(sessions):
            print("The session list contains duplicates.", file=sys.stderr)
            return EXIT_USAGE
        return _run_batch(args, raw_argv, config, project_root, environment, credentials, credentials_desc, storage,
                          sessions, controller)
    finally:
        controller.shutdown()


def _run_batch(args, raw_argv, config, project_root, environment, credentials, credentials_desc, storage, sessions,
               controller) -> int:
    import logging  # noqa: PLC0415

    from video_pipeline.inference.ego_labeller import build_labellers, preflight  # noqa: PLC0415
    from video_pipeline.orchestration.ego_session import Services, run_session  # noqa: PLC0415

    log = logging.getLogger("video_pipeline.ego")
    upload = bool(config.get("upload.enabled"))
    batch_id = utc_now().strftime("%Y%m%dT%H%M%SZ")
    log.info("Batch %s: %d session(s), upload %s, credentials: %s", batch_id, len(sessions),
             "on" if upload else "OFF", credentials_desc)
    log.info("Source: gs://%s/%s | results: %s | content screen (section 3): %s",
             config.get("source.bucket"), config.get("source.raw_root"),
             f"gs://{processed_bucket(config)}/{config.get('source.processed_root')}" if upload else "local only",
             "ON" if config.get("content.enabled") else "off")

    def model_factory():
        labellers = build_labellers(config.get("models"), inference=config.get("inference"),
                                    mediapipe=config.get("mediapipe"), environment=environment,
                                    gcp_project=config.get("gcp.project"),
                                    credentials_factory=lambda: credentials, project_root=project_root,
                                    motion=config.get("motion"),
                                    interval_seconds=config.get("labelling.interval_seconds"))
        try:
            return labellers, preflight(labellers, _probe_image())
        except BaseException:
            for labeller in labellers.values():
                labeller.close()
            raise

    services = Services(config=config, project_root=project_root, storage=storage, logging_controller=controller,
                        model_factory=model_factory)
    rows = []
    interrupted = False
    try:
        if args.until not in ("extract", "technical", "calibration"):
            # Start the models now: a missing library or model file stops the batch in seconds,
            # not after the first recording has been streamed and decoded.
            try:
                services.labellers()
            except Exception as exc:  # noqa: BLE001
                log.error("The models could not start, so nothing was processed: %s: %s", type(exc).__name__, exc)
                if "libEGL" in str(exc) or "libGLES" in str(exc):
                    log.error("MediaPipe needs the system graphics libraries. On the VM run once: "
                              "sudo apt-get install -y libegl1 libgles2   (see docs/ego_qa.md, 'VM setup')")
                return EXIT_ERRORS
        for number, session in enumerate(sessions, start=1):
            log.info("===== Session %d/%d: %s", number, len(sessions), session.url)
            try:
                row = run_session(session, services, force=args.force, upload=upload, until=args.until,
                                  max_segments=args.max_segments, argv=tuple(raw_argv))
            except PipelineError as exc:
                row = {"session": session.url, "status": "ERROR", "reasons": str(exc)[:500]}
            except KeyboardInterrupt:
                rows.append({"session": session.url, "status": "INTERRUPTED"})
                interrupted = True
                break
            except Exception as exc:  # noqa: BLE001 (one broken session must not stop the batch)
                log.exception("Unexpected error in %s", session.url)
                row = {"session": session.url, "status": "ERROR", "reasons": f"{type(exc).__name__}: {exc}"[:500]}
            rows.append(row)
    finally:
        services.close()
        _write_batch_summary(config, project_root, storage, batch_id, sessions, rows, upload and args.until is None, log)
    if interrupted:
        return EXIT_INTERRUPTED
    return EXIT_ERRORS if any(r["status"] == "ERROR" for r in rows) else EXIT_OK


def _write_batch_summary(config, project_root, storage, batch_id, sessions, rows, upload, log) -> None:
    work = Path(config.get("work.directory"))
    work = work if work.is_absolute() else project_root / work
    folder = work / "_batches" / batch_id
    write_text_atomic(folder / "sessions.txt", "".join(s.url + "\n" for s in sessions))
    write_text_atomic(folder / "summary.csv", _summary_csv(rows))
    counts = defaultdict(int)
    for r in rows:
        counts[r["status"]] += 1
    log.info("Batch %s finished: %s. Summary: %s", batch_id, dict(counts), folder / "summary.csv")
    for r in rows:
        log.info("  %-11s %s %s", r["status"], r["session"], ("- " + r["reasons"]) if r.get("reasons") else "")
    if upload:
        prefix = f"{config.get('source.processed_root')}_batches/{batch_id}/"
        try:
            for name in ("sessions.txt", "summary.csv"):
                storage.upload(folder / name, processed_bucket(config), prefix + name, overwrite=False,
                               content_type="text/plain" if name.endswith(".txt") else "text/csv")
            log.info("Batch summary uploaded to gs://%s/%s", processed_bucket(config), prefix)
        except PipelineError as exc:
            log.error("Could not upload the batch summary: %s", exc)


def _probe_image() -> bytes:
    """A small grey JPEG used to prove each model answers before the batch starts."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    ok, buf = cv2.imencode(".jpg", np.full((256, 448, 3), 128, dtype=np.uint8))
    if not ok:
        raise PipelineError("cannot encode the probe image")
    return buf.tobytes()


if __name__ == "__main__":
    raise SystemExit(main())
