"""Command line for the egocentric QA batch (MCAP sessions in Cloud Storage).

    python scripts/run_gcs_batch.py --list-sessions gs://op-datoric/raw/Op_DD/
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
from pathlib import Path, PurePosixPath

from video_pipeline import __version__
from video_pipeline.core.config import find_project_root
from video_pipeline.core.ego_config import load_ego_configuration
from video_pipeline.core.env import load_environment
from video_pipeline.core.errors import ConfigError, PipelineError
from video_pipeline.core.gcp_auth import get_credentials
from video_pipeline.core.io import utc_now, write_text_atomic
from video_pipeline.core.logging_setup import LoggingController
from video_pipeline.ingestion.gcs import GcsStorage, SessionRef, parse_gs_url

EXIT_OK, EXIT_ERRORS, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130
SUMMARY_FIELDS = ["session", "status", "reasons", "kept_seconds", "clips", "calibration", "elapsed_s"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_gcs_batch",
                                description="Egocentric video QA for MCAP recordings in Google Cloud Storage.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--session", action="append", metavar="GS_URL",
                     help="A session folder gs://<bucket>/<raw_root>.../session_*/ (repeatable).")
    src.add_argument("--sessions-file", type=Path, metavar="FILE",
                     help="Text file with one session URL per line (# comments allowed).")
    src.add_argument("--list-sessions", metavar="GS_URL",
                     help="List every session below this prefix with its size and segment count, then exit.")
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


def list_sessions(storage: GcsStorage, url: str) -> str:
    bucket, prefix = parse_gs_url(url)
    sizes: dict[str, int] = defaultdict(int)
    files: dict[str, int] = defaultdict(int)
    for obj in storage.list_objects(bucket, prefix):
        parts = PurePosixPath(obj.name).parts
        idx = next((i for i, part in enumerate(parts[:-1]) if part.startswith("session_")), None)
        if idx is None:
            continue
        folder = "/".join(parts[: idx + 1]) + "/"
        sizes[folder] += obj.size
        files[folder] += 1
    lines = [f"{sizes[f] / 1e9:8.2f} GB  {files[f]:4d} files  gs://{bucket}/{f}" for f in sorted(sizes)]
    lines.append(f"{len(sizes)} sessions, {sum(sizes.values()) / 1e9:.1f} GB")
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
    controller = LoggingController(args.log_level or "INFO")
    try:
        try:
            project_root = find_project_root()
            overrides = {"upload.enabled": False} if args.no_upload else {}
            if args.log_level:
                overrides["logging.console_level"] = args.log_level
            config = load_ego_configuration(project_root, args.config, overrides)
        except ConfigError as exc:
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

        if args.list_sessions:
            print(list_sessions(storage, args.list_sessions))
            return EXIT_OK

        urls = args.session or read_sessions_file(args.sessions_file)
        try:
            sessions = [SessionRef.from_url(u, expected_bucket=config.get("source.bucket"),
                                            raw_root=config.get("source.raw_root")) for u in urls]
        except PipelineError as exc:
            print(f"Session list: {exc}", file=sys.stderr)
            return EXIT_USAGE
        if not sessions:
            print("No sessions given.", file=sys.stderr)
            return EXIT_USAGE
        if len({s.relative for s in sessions}) != len(sessions):
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

    def model_factory():
        labellers = build_labellers(config.get("models"), inference=config.get("inference"),
                                    mediapipe=config.get("mediapipe"), environment=environment,
                                    gcp_project=config.get("gcp.project"),
                                    credentials_factory=lambda: credentials, project_root=project_root)
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
                storage.upload(folder / name, config.get("source.bucket"), prefix + name, overwrite=False,
                               content_type="text/plain" if name.endswith(".txt") else "text/csv")
            log.info("Batch summary uploaded to gs://%s/%s", config.get("source.bucket"), prefix)
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
