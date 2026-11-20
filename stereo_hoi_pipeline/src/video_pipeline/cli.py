"""Command-line interface: argument parsing, validation and wiring of the pipeline.

Exit codes: 0 success, 1 pipeline error, 2 usage error, 130 interrupted.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from video_pipeline import __version__
from video_pipeline.core.config import find_project_root, load_configuration
from video_pipeline.core.env import load_environment
from video_pipeline.core.errors import ConfigError, PipelineError
from video_pipeline.core.logging_setup import LoggingController
from video_pipeline.orchestration.pipeline import Pipeline, PipelineRequest
from video_pipeline.orchestration.stages import STAGE_NAMES, build_stages

EXIT_OK = 0
EXIT_PIPELINE_ERROR = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number of seconds, got '{value}'") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got '{value}'") from None
    if number <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="Stereo video preprocessing and HOI waste-removal pipeline.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="HTTP(S) URL of a video to download into videos/ and process.")
    source.add_argument("--input", type=Path,
                        help="Video to process: a path, or a file name inside videos/.")
    parser.add_argument("--config", type=Path, help="YAML file overriding the default configuration.")
    parser.add_argument(
        "--waste-threshold", type=_positive_int, metavar="N",
        help="Minimum continuous WASTE duration in seconds before footage is removed (overrides config).",
    )
    parser.add_argument("--detector", choices=["mediapipe", "vlm"],
                        help="How stereo pairs are classified: mediapipe (default; hand landmarks, runs locally) "
                             "or vlm (vision-language model configured in .env).")
    parser.add_argument("--verify-with", choices=["same", "mediapipe", "vlm"],
                        help="Detector that re-checks sections before they are cut (default: same as --detector). "
                             "'vlm' sends only those sections to the vision-language model.")
    parser.add_argument("--sample-fps", type=_positive_float, metavar="FPS",
                        help="Stereo pairs analysed per second of video (default 1). Candidate waste is re-checked "
                             "at twice this rate.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the previous run for this video; completed, unaffected stages are skipped.")
    parser.add_argument("--until", choices=STAGE_NAMES, metavar="STAGE",
                        help=f"Stop after this stage. One of: {', '.join(STAGE_NAMES)}.")
    parser.add_argument("--output-dir", type=Path, help="Root directory for outputs (overrides project.output_directory).")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level (overrides logging.console_level).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.config is not None and not args.config.is_file():
        parser.error(f"--config: file not found: '{args.config}'")
    if args.url is not None and not args.url.lower().startswith(("http://", "https://")):
        parser.error("--url must start with http:// or https://")
    if args.output_dir is not None and args.output_dir.exists() and not args.output_dir.is_dir():
        parser.error(f"--output-dir exists and is not a directory: '{args.output_dir}'")


def _cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.waste_threshold is not None:
        overrides["hoi.waste_threshold_seconds"] = args.waste_threshold
    if args.detector is not None:
        overrides["detection.method"] = args.detector
    if args.verify_with is not None:
        overrides["verification.detector"] = args.verify_with
    if args.sample_fps is not None:
        interval = round(1.0 / args.sample_fps, 6)
        overrides["sampling.interval_seconds"] = interval
        overrides["verification.interval_seconds"] = round(interval / 2, 6)
    if args.output_dir is not None:
        # Resolved against the current directory, like any other CLI path.
        overrides["project.output_directory"] = str(args.output_dir.resolve())
    if args.log_level is not None:
        overrides["logging.console_level"] = args.log_level
    return overrides


def _resolve_input(path: Path, project_root: Path, videos_dir: str) -> Path | None:
    """An existing path as given, else the same name inside the videos folder."""
    if path.is_file():
        return path
    folder = Path(videos_dir)
    folder = folder if folder.is_absolute() else project_root / folder
    candidate = folder / path
    return candidate if candidate.is_file() else None


def confirm_overwrite_interactively(path: Path) -> bool:
    """Ask before overwriting. Non-interactive sessions never overwrite."""
    if sys.stdin is None or not sys.stdin.isatty():
        return False
    try:
        answer = input(f"'{path}' already exists. Overwrite it with the downloaded file? [y/N]: ")
    except EOFError:
        # Some terminals (e.g. Git Bash) report a TTY even when stdin is closed.
        print(file=sys.stderr)
        return False
    return answer.strip().lower() in {"y", "yes"}


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    _validate_args(parser, args)

    logging_controller = LoggingController(args.log_level or "INFO")
    try:
        try:
            project_root = find_project_root()
            environment = load_environment(project_root)
            config = load_configuration(project_root, args.config, _cli_overrides(args), environment)
        except ConfigError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return EXIT_USAGE
        logging_controller.set_console_level(config.get("logging.console_level"))
        if args.input is not None:
            resolved = _resolve_input(args.input, project_root, config.get("video.download_directory"))
            if resolved is None:
                print(f"--input: file not found: '{args.input}' (looked in the current folder and in "
                      f"{config.get('video.download_directory')}/)", file=sys.stderr)
                return EXIT_USAGE
            args.input = resolved

        pipeline = Pipeline(
            config=config,
            project_root=project_root,
            stages=build_stages(),
            logging_controller=logging_controller,
            confirm_overwrite=confirm_overwrite_interactively,
            environment=environment,
        )
        request = PipelineRequest(url=args.url, input_path=args.input, resume=args.resume,
                                  argv=tuple(raw_argv), until=args.until)
        try:
            summary = pipeline.run(request)
        except PipelineError as exc:
            print(f"\nPipeline failed: {exc}", file=sys.stderr)
            return EXIT_PIPELINE_ERROR
        except KeyboardInterrupt:
            print("\nInterrupted. Re-run with --resume to continue (finished inference is cached).", file=sys.stderr)
            return EXIT_INTERRUPTED

        print()
        print(f"Run ID     : {summary.run_id}{' (resumed)' if summary.resumed else ''}")
        print(f"Status     : {summary.status}")
        print(f"Source     : {summary.source_path}")
        print(f"Workspace  : {summary.workspace_root}")
        print(f"Manifest   : {summary.manifest_path}")
        print("Stages     : " + ", ".join(f"{name}={status}" for name, status in summary.stage_statuses.items()))
        print(f"Elapsed    : {summary.elapsed_seconds:.1f}s")
        return EXIT_OK
    finally:
        logging_controller.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
