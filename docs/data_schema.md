# Data Schema

All JSON is UTF-8 and written atomically (temporary file + rename). Timestamps are
ISO-8601 UTC with milliseconds and a `Z` suffix. Paths inside a workspace are
POSIX-style and relative to `output/<video_name>/`.

## Workspace layout

```text
output/<video_name>/
├── metadata/        source.json, download.json, validation.json, ffprobe.json (authoritative), video_info.json
├── imu/
│   ├── original/    imu_metadata.json [, telemetry.bin, gpmf_packets.json, parser_output.txt, accelerometer.csv, gyroscope.csv]
│   └── processed/   synchronization_metadata.json [, accelerometer.csv, gyroscope.csv]
├── frames/          frame_index.jsonl, sampling_metadata.json  (images only while being classified)
├── inference/       classifications.jsonl, raw_responses.jsonl, inference_metadata.json, cache/<key>.json
├── segmentation/    classifications_timeline.json, waste_candidates.json, verification.json, verification_samples.jsonl,
│                    waste_intervals.json, retained_intervals.json, segmentation_metadata.json
├── processed/       <video_name>_processed.mp4, processing_timeline.json
└── reports/         manifest.json, processing.log, resolved_config.yaml, metrics.json, metrics.csv, summary.json,
                     qc_report.json, history/<run_id>/
```
`NNNNNN` = sample index `k` (target time `k × interval`). Times are seconds from
the video start unless a field says `container`.

## Phases 2–8 artifacts

### `frames/frame_index.jsonl` (one line per pair)
`sample_index, target_timestamp_seconds, frame_pts_seconds (video-relative actual PTS),
container_pts_seconds, left, right (workspace paths), left_sha256, right_sha256`.

### `frames/sampling_metadata.json`
`rule, interval_seconds, duration_seconds, samples, missing_sample_indices,
geometry{layout, frame_width, frame_height, split_ratio, left{x,y,width,height}, right{…}, warnings},
max_width, jpeg_quality, ffmpeg_mjpeg_qscale`.

### `inference/classifications.jsonl` (one line per pair)
```jsonc
{ "sample_index": 12, "timestamp_seconds": 12.0, "frame_pts_seconds": 12.033, "left": "...", "right": "...",
  "classification": "RELEVANT | WASTE | UNCERTAIN | UNAVAILABLE",
  "hoi_description": "...", "evidence": ["..."],
  "left_camera":  { "hands_visible": true, "interaction_detected": true, "description": "..." },
  "right_camera": { ... },
  "status": "ok | invalid_response | failed", "validation_errors": [],
  "model": "qwen3-vl:4b", "model_digest": "...", "prompt_version": "hoi-stereo-v1",
  "latency_seconds": 2.41, "attempts": 1, "cached": false, "cache_key": "..." }
```
### `inference/raw_responses.jsonl`
`sample_index, timestamp_seconds, cache_key, cached, response_text, provider_raw (full Ollama response), error`.

### `segmentation/*`
* `classifications_timeline.json`: `samples[{sample_index,start,end,duration,label}]`, `runs[{label,start,end,duration,samples}]`.
* `waste_candidates.json` (temporal): `threshold_seconds`, `intervals[]` (candidates), `decisions[]` (every WASTE run: `candidate`, `reason`).
* `verification.json`: `enabled, interval_seconds, guard_seconds, threshold_seconds, rule`,
  `candidates[{candidate, outcome: removed|partially_removed|kept, fine_samples, non_waste_samples[{timestamp,label}], protected[], removed[], kept_short_pieces[]}]`,
  `removed_duration, candidate_duration, inference_requests`.
* `verification_samples.jsonl`: one line per fine sample (`candidate, timestamp_seconds, classification, …` same fields as classifications).
* `waste_intervals.json` (**final**, from verification): `threshold_seconds, verified, intervals[]`.
* `retained_intervals.json`: `intervals[]`.
* `segmentation_metadata.json`: parameters, rules, per-label durations, candidate totals.

### `reports/qc_report.json`
`status: pass|warn|fail`, `counts{pass,warn,fail,skipped}`, `checks[{id, title, status, detail, data}]`
(`output.alignment.data.checkpoints[{processed, original, psnr_db, frame_offset}]`), `run_id, created_at`.
`summary.json` gains `qc_status`, `qc_counts`, `verification{candidates, removed, trimmed, kept, …}`.

### Frames
`frames/left|right/` hold images only temporarily (deleted after classification unless
`sampling.keep_frames`). `frame_index.jsonl` is the permanent record.

All intervals: `{start, end, duration}`, half-open `[start, end)`.

### `processed/processing_timeline.json`
```jsonc
{ "source": "STEREO_01.mp4", "output": "processed/STEREO_01_processed.mp4",
  "original_duration": 149.733, "processed_duration_nominal": 111.733, "processed_duration_measured": 111.733,
  "removed_intervals": [{ "start": 20, "end": 32, "duration": 12 }],
  "retained_segments": [{ "segment_index": 0, "original_start": 0, "original_end": 20,
                          "processed_start": 0, "processed_end": 20, "duration": 20 }],
  "audio_included": false, "encoder": { "video": "libx264", "args": [...], ... } }
```

### `imu/*`
See [imu_synchronization.md](imu_synchronization.md) for CSV columns.
`imu_metadata.json` has `available` (bool) and `reason`; when available, also
`streams{ACCL|GYRO: samples, units, scale, axis_order, approx_sampling_rate_hz, …}` and `files`.
`synchronization_metadata.json` has `status` (`synchronized | skipped`), `reason`/`method`, `streams{kept,removed,…}`.

### `reports/metrics.json`
`parameters`, `durations{original, processed, processed_measured, removed, relevant, waste_total, waste_removed,
waste_kept, uncertain, unavailable, missing}`, `ratios_percent{hoi_coverage_before, hoi_coverage_after,
waste_reduction, retention}`, `counts{frames_analyzed, samples_by_label, hoi_intervals, waste_intervals_removed,
inference_failures, invalid_responses, cached_results}`, `inference_latency_seconds{mean, median,
mean_uncached_this_run}`, `comparison[{metric, before, after}]` (= `metrics.csv`), `notes`.

### `reports/summary.json`
`run_id, video, source_sha256, headline, waste_threshold_seconds, model, durations, ratios_percent, counts,
imu_synchronization, artifacts{…}`.

## `metadata/source.json`
| Field | Type | Notes |
|---|---|---|
| `filename` | str | Original filename |
| `path` | str | Relative to project root when inside it, else absolute |
| `absolute_path` | str | |
| `origin` | `"url"` \| `"local"` | |
| `url` | str \| null | |
| `sha256` | str | Hex digest of the full file |
| `size_bytes` | int | |
| `mtime_ns` | int | Used to detect modification during the run |
| `run_id`, `recorded_at` | str | |

## `metadata/download.json`
`path, url, final_url, http_status, content_type, content_length, etag,
last_modified, bytes_written, started_at, finished_at, elapsed_seconds, attempts`.

## `metadata/validation.json`
`run_id, validated_at, file_checks{exists,is_file,size_bytes,extension},
container_checked, warnings[], passed`.

## `metadata/video_info.json` (schema 1.0.0)
```jsonc
{
  "schema_version": "1.0.0",
  "duration_seconds": 300.3,              // format duration, else primary video stream
  "primary_video_stream_index": 0,        // first non-cover-art video stream
  "format": { "filename", "format_name", "format_long_name", "duration_seconds",
              "start_time_seconds", "size_bytes", "bit_rate", "nb_streams", "tags" },
  "video_streams": [{
      "index", "codec_name", "codec_long_name", "profile", "codec_tag_string",
      "width", "height", "pix_fmt", "field_order",
      "r_frame_rate", "avg_frame_rate",          // raw rationals as reported
      "fps_nominal", "fps_average",              // parsed floats (null for 0/0)
      "possibly_variable_frame_rate",            // heuristic: |nominal - average| > 0.01
      "time_base", "start_pts", "start_time_seconds", "duration_seconds",
      "bit_rate", "nb_frames", "nb_read_frames", // nb_read_frames only with count_frames
      "rotation_degrees", "is_attached_picture" }],
  "audio_streams": [{ "index", "codec_name", "sample_rate", "channels", "channel_layout",
                      "time_base", "start_time_seconds", "duration_seconds", "bit_rate" }],
  "other_streams": [{ "index", "codec_type", "codec_name", "codec_tag_string", "handler_name",
                      "time_base", "start_time_seconds", "duration_seconds", "nb_frames", "is_gpmf" }],
  "chapters_count": 0,
  "telemetry": { "gpmf_stream_present": true, "gpmf_stream_indices": [3] },
  "source_sha256": "...",
  "extracted_at": "..."
}
```
A GPMF stream is detected by `codec_tag_string == "gpmd"` or a `handler_name`
containing “GoPro MET”. Phase 2 uses `gpmf_stream_indices`.

`possibly_variable_frame_rate` is a hint only. Phase 3 must sample by presentation
timestamps either way (plan §7.3).

## `reports/manifest.json` (schema 1.0.0)
```jsonc
{
  "schema_version": "1.0.0",
  "run_id": "20261111T102521Z-38c5e74e",
  "video_name": "GX010140",
  "pipeline_version": "0.1.0",
  "status": "running | completed | partial | failed | interrupted",
  "created_at", "updated_at", "finished_at",
  "planned_stages": ["ingestion", "validation", "metadata"],
  "attempts": [{ "attempt": 1, "mode": "new | resume", "argv": [...],
                 "started_at", "finished_at", "status" }],
  "source": { ...source.json fields..., "duration_seconds": 300.3 },
  "configuration": {
    "fingerprint", "fingerprint_history": [{ "fingerprint", "replaced_at" }],
    "resolved_config_path": "reports/resolved_config.yaml",
    "layers": ["configs/default.yaml", "configs/inference.yaml", "configs/processing.yaml", "<user file>", "cli"],
    "cli_overrides": { "hoi.waste_threshold_seconds": 5 },
    "sampling_interval", "waste_threshold_seconds", "uncertain_policy",
    "inference_provider", "model", "prompt_version",
    "snapshot": { ...full resolved configuration... }
  },
  "versions": { "pipeline", "python", "python_implementation", "platform", "pyyaml",
                "ffmpeg", "ffprobe", "ollama" },
  "stages": {
    "<name>": { "status": "running | completed | failed | interrupted",
                "started_at", "finished_at", "runs", "outputs": ["metadata/ffprobe.json"],
                "config_fingerprint": "sha256 of the stage's config_keys",
                "details": { ..., "elapsed_seconds" }, "error": null,
                "skips": [{ "at", "reason" }] }
  },
  "commands": [{ "stage", "recorded_at", "args": [...], "returncode", "elapsed_seconds" }],
  "results": { "original_duration", "removed_duration", "processed_duration" },
  "errors": [{ "stage", "at", "error" }]
}
```
`configuration` also gets `env_overrides` (variable names only) and, after
inference, `model_digest` and `inference_runtime_version`. `results.removed_duration`
and `processed_duration` are filled by the segmentation stage. Breaking changes bump
`schema_version`.

## `reports/processing.log`
```text
2026-11-11 15:56:19,009 | INFO     | run=<run_id> | stage=<stage> | <logger> | <message>
```
One log per run. A new run archives the previous log. A resume appends to it.
