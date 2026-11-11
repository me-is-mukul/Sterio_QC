"""Pipeline runner.

Flow for one invocation:
    1. Acquire the source (download from URL into the project root, or use a local file).
    2. Identify it (SHA-256, size, mtime).
    3. Create/open the workspace ``<output_root>/<video_name>/``.
    4. Create a new manifest (archiving any previous run) or reopen it for --resume.
    5. Attach the per-run log file, persist the resolved configuration and tool versions.
    6. Run each stage in order. On --resume a stage is skipped only if it completed,
       its outputs exist, its config fingerprint is unchanged and no earlier stage re-ran.
    7. Verify the source video was not modified, then finalise the manifest.

The runner depends only on the abstract ``Stage`` interface; concrete stages are injected.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from video_pipeline import __version__
from video_pipeline.core.config import ResolvedConfig
from video_pipeline.core.environment import collect_versions
from video_pipeline.core.errors import (
    InputValidationError,
    PipelineError,
    ResumeError,
    SourceModifiedError,
    WorkspaceError,
)
from video_pipeline.core.hashing import sha256_file, sha256_json
from video_pipeline.core.io import write_text_atomic
from video_pipeline.core.locking import RunLock
from video_pipeline.core.logging_setup import LoggingController, set_run_id, stage_scope
from video_pipeline.core.workspace import Workspace, derive_video_name
from video_pipeline.ingestion.downloader import (
    DownloadResult,
    DownloadSettings,
    download,
    resolve_remote_filename,
)
from video_pipeline.orchestration.context import (
    ORIGIN_LOCAL,
    ORIGIN_URL,
    RunContext,
    SourceVideo,
    generate_run_id,
)
from video_pipeline.orchestration.stages import Stage
from video_pipeline.reporting.manifest import (
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_INTERRUPTED,
    RUN_PARTIAL,
    STAGE_COMPLETED,
    RunManifest,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineRequest:
    url: str | None
    input_path: Path | None
    resume: bool
    argv: tuple[str, ...] = ()
    until: str | None = None  # stop after this stage (inclusive)

    def __post_init__(self) -> None:
        if (self.url is None) == (self.input_path is None):
            raise ValueError("Exactly one of url or input_path must be provided.")


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    status: str
    resumed: bool
    source_path: Path
    workspace_root: Path
    manifest_path: Path
    stage_statuses: dict[str, str]
    elapsed_seconds: float


class Pipeline:
    def __init__(
        self,
        *,
        config: ResolvedConfig,
        project_root: Path,
        stages: Sequence[Stage],
        logging_controller: LoggingController,
        confirm_overwrite: Callable[[Path], bool],
        environment: Mapping[str, str] | None = None,
    ) -> None:
        names = [s.name for s in stages]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate stage names: {names}")
        self._config = config
        self._project_root = project_root
        self._stages = list(stages)
        self._logging = logging_controller
        self._confirm_overwrite = confirm_overwrite
        self._environment: Mapping[str, str] = dict(environment or {})

    # Public API -------------------------------------------------------------
    def run(self, request: PipelineRequest) -> RunSummary:
        t0 = time.perf_counter()
        stages = self._select_stages(request.until)
        with stage_scope("ingestion"):
            path, download_result, origin = self._acquire_source(request)
            source = self._identify_source(path, origin, request.url, download_result)

        workspace = Workspace.for_video(self._resolve_path(self._config.get("project.output_directory")), derive_video_name(path))
        workspace.ensure_separate_from(path)
        workspace.create()

        lock = RunLock(workspace.lock_path, " ".join(request.argv))
        lock.acquire()  # one run per video at a time
        try:
            return self._run_in_workspace(request, stages, source, workspace, t0)
        finally:
            lock.release()

    def _run_in_workspace(self, request: PipelineRequest, stages: Sequence[Stage], source: SourceVideo,
                          workspace: Workspace, t0: float) -> RunSummary:
        manifest, resumed = self._open_manifest(workspace, source, request)
        set_run_id(manifest.run_id)
        self._logging.attach_file(workspace.log_path, self._config.get("logging.file_level"))
        logger.info("Run %s %s | workspace: %s", manifest.run_id, "resumed" if resumed else "started", workspace.root)

        ctx = RunContext(
            run_id=manifest.run_id,
            config=self._config,
            project_root=self._project_root,
            workspace=workspace,
            manifest=manifest,
            source=source,
            logger=logging.getLogger("video_pipeline.stage"),
            resumed=resumed,
            environment=self._environment,
        )
        current_stage: str | None = None
        try:
            self._record_run_setup(ctx, stages)
            upstream_ran = False
            for stage in stages:
                current_stage = stage.name
                upstream_ran = self._run_stage(ctx, stage, upstream_ran) or upstream_ran
            current_stage = None
            self._verify_source_unchanged(source)
        except KeyboardInterrupt as exc:
            self._finalise_failure(ctx, current_stage, exc, interrupted=True)
            raise
        except Exception as exc:
            self._finalise_failure(ctx, current_stage, exc, interrupted=False)
            raise

        status = RUN_PARTIAL if len(stages) < len(self._stages) else RUN_COMPLETED
        manifest.finish(status)
        ctx.save_manifest()
        elapsed = time.perf_counter() - t0
        if status == RUN_PARTIAL:
            logger.info("Run %s stopped after stage '%s' as requested (%.1fs).", manifest.run_id, stages[-1].name, elapsed)
        else:
            logger.info("Run %s completed in %.1fs.", manifest.run_id, elapsed)
        return RunSummary(
            run_id=manifest.run_id,
            status=status,
            resumed=resumed,
            source_path=source.path,
            workspace_root=workspace.root,
            manifest_path=workspace.manifest_path,
            stage_statuses={s.name: (manifest.stage(s.name) or {}).get("status", "not run") for s in self._stages},
            elapsed_seconds=elapsed,
        )

    def _select_stages(self, until: str | None) -> list[Stage]:
        if until is None:
            return list(self._stages)
        names = [s.name for s in self._stages]
        if until not in names:
            raise PipelineError(f"Unknown stage '{until}'. Stages: {', '.join(names)}")
        return self._stages[: names.index(until) + 1]

    # Source acquisition -------------------------------------------------------
    def _resolve_path(self, configured: str) -> Path:
        path = Path(configured)
        return (path if path.is_absolute() else self._project_root / path).resolve()

    def _download_settings(self) -> DownloadSettings:
        get = self._config.get
        return DownloadSettings(
            timeout_seconds=get("download.timeout_seconds"),
            max_retries=get("download.max_retries"),
            retry_backoff_seconds=get("download.retry_backoff_seconds"),
            chunk_size_bytes=get("download.chunk_size_bytes"),
            progress_log_percent_step=get("download.progress_log_percent_step"),
            user_agent=f"{get('download.user_agent')}/{__version__}",
        )

    def _acquire_source(self, request: PipelineRequest) -> tuple[Path, DownloadResult | None, str]:
        if request.input_path is not None:
            path = request.input_path.resolve()
            if not path.is_file():
                raise InputValidationError(f"Input video not found: '{request.input_path}'.")
            logger.info("Using local input video: %s", path)
            return path, None, ORIGIN_LOCAL

        assert request.url is not None
        settings = self._download_settings()
        download_dir = self._resolve_path(self._config.get("video.download_directory"))
        filename = resolve_remote_filename(request.url, settings)
        target = download_dir / filename
        if request.resume and target.is_file():
            logger.info("Resume: reusing previously downloaded file '%s'; download skipped.", target)
            return target, None, ORIGIN_URL
        logger.info("Downloading %s -> %s", request.url, target)
        result = download(request.url, download_dir, settings, filename=filename, confirm_overwrite=self._confirm_overwrite)
        return result.path, result, ORIGIN_URL

    def _identify_source(self, path: Path, origin: str, url: str | None, download_result: DownloadResult | None) -> SourceVideo:
        logger.info("Computing SHA-256 of %s ...", path.name)
        stat = path.stat()
        digest = sha256_file(path, self._config.get("hashing.chunk_size_bytes"))
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise SourceModifiedError(f"Source video '{path}' changed while it was being hashed.")
        return SourceVideo(
            path=path, origin=origin, url=url, sha256=digest,
            size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, download=download_result,
        )

    # Manifest / run management ----------------------------------------------
    def _open_manifest(self, workspace: Workspace, source: SourceVideo, request: PipelineRequest) -> tuple[RunManifest, bool]:
        existing_path = workspace.manifest_path
        if request.resume:
            if not existing_path.is_file():
                raise ResumeError(f"--resume was given but no previous run exists at '{existing_path}'.")
            manifest = RunManifest.load(existing_path)
            recorded = manifest.source.get("sha256")
            if recorded and recorded != source.sha256:
                raise ResumeError(
                    f"Cannot resume run {manifest.run_id}: the source video's SHA-256 differs from the recorded one."
                )
            manifest.begin_attempt("resume", request.argv)
            return manifest, True

        if existing_path.is_file():
            previous = RunManifest.load(existing_path)
            recorded = previous.source.get("sha256")
            if recorded and recorded != source.sha256:
                raise WorkspaceError(
                    f"Output workspace '{workspace.root}' belongs to a different video with the same name "
                    f"(sha256 {recorded[:12]}... vs {source.sha256[:12]}...). Use --output-dir to choose another location."
                )
            archived = workspace.archive_previous_run(previous.run_id)
            logger.info("Archived previous run %s to %s", previous.run_id, archived)

        manifest = RunManifest.create(generate_run_id(), workspace.video_name, __version__)
        manifest.begin_attempt("new", request.argv)
        return manifest, False

    def _layer_display(self, layer: str) -> str:
        if layer == "cli" or layer.startswith("env:"):
            return layer
        try:
            return Path(layer).resolve().relative_to(self._project_root).as_posix()
        except ValueError:
            return layer

    def _record_run_setup(self, ctx: RunContext, stages: Sequence[Stage]) -> None:
        manifest, config, workspace = ctx.manifest, ctx.config, ctx.workspace
        previous_fingerprint = manifest.configuration_fingerprint
        if ctx.resumed and previous_fingerprint and previous_fingerprint != config.fingerprint:
            logger.info(
                "Configuration changed since the previous attempt (fingerprint %s -> %s); "
                "only stages affected by the change (and later stages) will re-run.",
                previous_fingerprint[:12], config.fingerprint[:12],
            )
        write_text_atomic(workspace.resolved_config_path, config.to_yaml())
        manifest.set_planned_stages([s.name for s in stages])
        manifest.set_configuration(
            {
                "fingerprint": config.fingerprint,
                "resolved_config_path": workspace.relative(workspace.resolved_config_path),
                "layers": [self._layer_display(layer) for layer in config.layers],
                "cli_overrides": dict(config.cli_overrides),
                "env_overrides": dict(config.env_overrides),
                "sampling_interval": config.get("sampling.interval_seconds"),
                "waste_threshold_seconds": config.get("hoi.waste_threshold_seconds"),
                "uncertain_policy": config.get("hoi.uncertain_policy"),
                "inference_provider": config.get("inference.provider"),
                "model": config.get("inference.model"),
                "prompt_version": config.get("inference.prompt_version"),
                "snapshot": config.to_dict(),
            }
        )
        versions = collect_versions(config)
        manifest.set_versions(versions)
        for tool in ("ffmpeg", "ffprobe"):
            if versions.get(tool) is None:
                logger.warning("Could not determine the %s version.", tool)
        logger.info("Versions: %s", ", ".join(f"{k}={v}" for k, v in versions.items() if k != "platform"))
        ctx.save_manifest()

    # Stage execution ----------------------------------------------------------
    def _stage_fingerprint(self, stage: Stage) -> str:
        return sha256_json({key: self._config.get(key) for key in stage.config_keys})

    def _skip_reason(self, ctx: RunContext, stage: Stage, upstream_ran: bool) -> str | None:
        """Why the stage may be skipped, or None if it must run."""
        if not ctx.resumed or upstream_ran:
            return None
        record = ctx.manifest.stage(stage.name)
        if not record or record.get("status") != STAGE_COMPLETED:
            return None
        if record.get("config_fingerprint") != self._stage_fingerprint(stage):
            logger.info("Stage '%s': relevant configuration changed; re-running.", stage.name)
            return None
        missing = [rel for rel in record.get("outputs", []) if not (ctx.workspace.root / rel).exists()]
        if missing:
            logger.warning("Stage '%s' was completed but outputs are missing (%s); re-running.", stage.name, missing)
            return None
        return "completed in a previous attempt with the same configuration"

    def _run_stage(self, ctx: RunContext, stage: Stage, upstream_ran: bool) -> bool:
        """Run (or skip) one stage. Returns True if it ran."""
        with stage_scope(stage.name):
            reason = self._skip_reason(ctx, stage, upstream_ran)
            if reason:
                ctx.manifest.mark_stage_skipped(stage.name, reason)
                ctx.save_manifest()
                logger.info("Stage '%s' skipped (%s).", stage.name, reason)
                return False
            logger.info("Stage '%s' started.", stage.name)
            ctx.manifest.mark_stage_running(stage.name)
            ctx.save_manifest()
            started = time.perf_counter()
            outcome = stage.run(ctx)
            elapsed = time.perf_counter() - started
            ctx.manifest.mark_stage_completed(
                stage.name,
                [ctx.workspace.relative(p) for p in outcome.outputs],
                {**dict(outcome.details), "elapsed_seconds": round(elapsed, 3)},
                config_fingerprint=self._stage_fingerprint(stage),
            )
            ctx.save_manifest()
            logger.info("Stage '%s' completed in %.2fs.", stage.name, elapsed)
            return True

    def _verify_source_unchanged(self, source: SourceVideo) -> None:
        stat = source.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (source.size_bytes, source.mtime_ns):
            raise SourceModifiedError(f"The source video '{source.path}' was modified during processing.")
        logger.debug("Source video unchanged (size and mtime verified).")

    def _finalise_failure(self, ctx: RunContext, stage_name: str | None, exc: BaseException, *, interrupted: bool) -> None:
        with stage_scope(stage_name or "pipeline"):
            if interrupted:
                logger.error("Run interrupted by user.")
            else:
                logger.error("Run failed: %s", exc, exc_info=not isinstance(exc, PipelineError))
            if stage_name is not None:
                ctx.manifest.mark_stage_failed(stage_name, exc, interrupted=interrupted)
            ctx.manifest.finish(RUN_INTERRUPTED if interrupted else RUN_FAILED)
            try:
                ctx.save_manifest()
            except OSError:
                logger.exception("Could not save the manifest after failure.")
