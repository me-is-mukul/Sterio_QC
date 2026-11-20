# Module Interfaces

Public interfaces only. Domain modules take plain values and return plain values.
They never read configuration or import orchestration.

## core

### `core.config`
```python
find_project_root() -> Path
load_configuration(project_root: Path, user_config_path: Path | None, cli_overrides: Mapping[str, Any] | None) -> ResolvedConfig
validate_config(data: Mapping) -> None                       # raises ConfigError listing every problem

class ResolvedConfig:                                         # frozen
    data: Mapping; layers: tuple[str, ...]; cli_overrides: Mapping
    get(dotted_key: str, default=...) -> Any                  # deep copy; ConfigError if missing and no default
    to_dict() -> dict;  to_yaml() -> str
    fingerprint: str                                          # sha256 of canonical JSON, excluding `logging`
```
CLI override keys are dotted paths and must already exist (`hoi.waste_threshold_seconds`).

### `core.workspace`
```python
derive_video_name(video_path: Path) -> str                    # stem, sanitised
class Workspace:                                              # frozen
    for_video(output_root, video_name) -> Workspace
    create() -> None
    ensure_separate_from(source_path) -> None
    metadata_dir | frames_dir | imu_dir | inference_dir | segmentation_dir | processed_dir | reports_dir
    manifest_path | log_path | resolved_config_path | history_dir
    relative(path) -> str                                     # POSIX path relative to workspace root
    archive_previous_run(previous_run_id) -> Path
```

### `core.tools`
```python
resolve_executable(name_or_path: str) -> str                  # ToolNotFoundError
run_command(args: Sequence[str], *, timeout_seconds: float) -> CommandResult   # argument list, never shell
tool_version(name_or_path, *, timeout_seconds) -> str | None
```

### `core.logging_setup`
```python
LoggingController(console_level)
    .attach_file(path, level) / .set_console_level(level) / .shutdown()
stage_scope(stage_name)                                       # context manager tagging log records
set_run_id(run_id)
```
Modules log via `logging.getLogger(__name__)` (children of `video_pipeline`).

### `core.hashing`, `core.io`
`sha256_file(path, chunk_size)`, `sha256_json(obj)`, `write_json_atomic(path, data)`,
`write_text_atomic(path, text)`, `read_json(path)`, `utc_now_iso()`.

## ingestion

### `ingestion.downloader`
```python
@dataclass DownloadSettings(timeout_seconds, max_retries, retry_backoff_seconds, chunk_size_bytes, progress_log_percent_step, user_agent)
resolve_remote_filename(url, settings) -> str
download(url, destination_dir, settings, *, filename=None, confirm_overwrite: Callable[[Path], bool]) -> DownloadResult
sanitize_filename(name) -> str
```
Raises `DownloadError`, or `FileConflictError` when overwrite is not confirmed.

### `ingestion.validator`
```python
validate_source_file(path, supported_formats) -> dict         # file checks; InputValidationError
validate_probe(raw_probe: Mapping) -> list[str]               # warnings; InputValidationError on fatal issues
```

## metadata

### `metadata.ffprobe`
```python
probe_video(ffprobe_executable, video_path, *, count_frames, timeout_seconds) -> ProbeResult(raw, command)
summarize_probe(raw: Mapping) -> dict                         # schema in data_schema.md
```

## reporting

### `reporting.manifest`
```python
RunManifest.create(run_id, video_name, pipeline_version) / RunManifest.load(path)
  .begin_attempt(mode, argv) .set_source(d) .set_configuration(d) .set_versions(d) .set_planned_stages(names)
  .set_result(key, value) .record_command(stage, record)
  .mark_stage_running(name) .mark_stage_completed(name, outputs, details)
  .mark_stage_skipped(name, reason) .mark_stage_failed(name, exc, interrupted=False)
  .finish(status) .save(path)
```

## orchestration

```python
class Stage(ABC):
    name: ClassVar[str]
    config_keys: ClassVar[tuple[str, ...]]              # keys whose change forces a re-run on --resume
    def run(self, ctx: RunContext) -> StageOutcome      # StageOutcome(outputs: tuple[Path,...], details: Mapping)

build_stages() -> list[Stage];  STAGE_NAMES
Pipeline(config=, project_root=, stages=, logging_controller=, confirm_overwrite=, environment=)
    .run(PipelineRequest(url, input_path, resume, argv, until)) -> RunSummary
```
`RunContext.environment` holds the process environment plus `.env` (may contain
secrets). Only the provider factory reads from it.
`RunContext.state` is an in-memory dict for passing data between stages within one
invocation (e.g. the FFprobe result). Anything a later stage needs **after a resume**
must be persisted to the workspace, never kept only in `state`.

### `core.env`
```python
read_dotenv(path) -> dict[str, str]
load_environment(project_root) -> dict[str, str]          # .env overlaid by the process environment
```
`load_configuration(..., environment=)` applies `config.ENV_OVERRIDES`
(`OLLAMA_BASE_URL → inference.base_url`, `OLLAMA_MODEL → inference.model`).

## Phase 2: telemetry

```python
# metadata.gpmf_extractor
extract_gpmf(ffmpeg, ffprobe, video_path, stream_index, raw_path, *, timeout_seconds) -> GpmfExtraction
    # GpmfExtraction(raw_path, stream_index, packets: tuple[GpmfPacket(index, pts_seconds, duration_seconds, offset, size)], commands)
# metadata.imu_parser
parse_klv(buffer, start=0, end=None) -> list[Klv]
decode_rows(klv) -> list[tuple]
parse_imu(raw: bytes, packets, fourccs, *, video_start_seconds) -> dict[str, SensorSeries]
series_metadata(series) -> dict
```

## Phase 3: stereo

```python
# stereo.stereo_splitter
compute_geometry(frame_width, frame_height, *, layout, split_ratio) -> StereoGeometry(left: Crop, right: Crop, warnings)
# stereo.frame_sampler
sample_stereo_frames(ffmpeg, video, geometry, *, interval, video_start, window_start, window_end, max_width,
                     jpeg_quality, left_dir, right_dir, timeout_seconds, name_prefix="")
    -> (list[SampledFrame], missing_indices, CommandResult)     # windowed; byte-identical to a full pass
jpeg_qscale(quality) -> int
```

## Verification and QC

```python
# segmentation.verification
fine_timestamps(candidate: Interval, step) -> list[float]
verify_candidate(candidate, labels: [(t, label)], *, step, guard, threshold) -> VerifiedCandidate
    # .protected, .removed, .kept_pieces, .outcome in {removed, partially_removed, kept}
# qc.checks (pure)
check_partition / check_removed_only_waste / check_min_duration / check_inference_health / check_coverage / check_imu
    -> CheckResult(id, title, status in {pass, warn, fail, skipped}, detail, data);  overall_status(results)
# qc.video_checks (FFmpeg)
probe_stream(ffprobe, path, timeout);  decode_errors(ffmpeg, path, timeout);  frame_psnr(ffmpeg, a, ta, b, tb, timeout)
alignment_checkpoints(segments, max_points) -> [(processed_t, original_t)]
# orchestration.stages.pairs (stage helpers)
PairExtractor(ctx).extract(window_start=, window_end=, interval=, left_dir=, right_dir=, prefix=)
PairClassifier(ctx).cache_key(left_sha, right_sha) / .cached(key) / .classify(left, right, timestamp=, key=) -> PairResult
create_provider(config, environment) -> InferenceProvider
```

## Phase 4: inference

```python
# inference.base
class InferenceProvider(ABC): model_info() -> ModelInfo; generate(ProviderRequest) -> ProviderResponse; describe() -> dict
# inference.ollama_client
OllamaProvider(base_url, model, *, timeout_seconds, max_retries, retry_backoff_seconds, think, keep_alive, api_key=None)
# inference.prompts
get_prompt(version) -> PromptTemplate(version, system, user_template, response_schema); .user_prompt(timestamp_seconds=)
# inference.qwen_classifier
StereoHoiClassifier(provider, prompt, *, temperature).classify(left_jpeg, right_jpeg, *, timestamp_seconds)
    -> (HoiClassification, ProviderResponse)
normalize_response(text) -> HoiClassification
failed_classification(error, failure_policy) -> HoiClassification
```

## Phase 5: temporal

```python
# analysis.temporal_filter
Interval(start, end).duration / .overlap(other);  TimedLabel(sample_index, interval, label);  Run(label, interval, sample_indices)
contiguous_runs(timed_labels) -> list[Run];  merge_intervals(...);  complement(removed, start, end) -> list[Interval]
# analysis.hoi_analyzer
build_timeline(classifications, *, interval, duration) -> list[TimedLabel]   # missing samples -> "MISSING"
label_durations(timeline, within=None) -> dict[label, seconds]
# segmentation.waste_detector
decide_waste(runs, threshold_seconds) -> list[WasteDecision(run, removed, reason)]
split_intervals(decisions, duration) -> (removed, retained)
```

## Phases 6–8

```python
# segmentation.video_cutter
build_processing_timeline(retained) -> list[segment dict]
cut_video(ffmpeg, video, retained, output, *, start_offset, include_audio, settings: EncoderSettings, timeout_seconds) -> CommandResult
# synchronization.imu_aligner
align_samples(rows, timeline_segments, *, duration, time_key="time_video_seconds") -> (kept_rows, AlignmentStats)
# metrics.hoi_metrics
compute_metrics(timeline, *, original_duration, removed, retained, processed_duration_measured,
                inference_records, waste_threshold, sampling_interval) -> dict
```

## Web

```python
# web.server
Dashboard(output_root).runs() / .overview(workspace) / .samples(workspace)
main(argv) -> int     # scripts/serve_ui.py
```
