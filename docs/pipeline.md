# Pipeline

## Running

```powershell
python scripts/run_pipeline.py --url "VIDEO_URL" --waste-threshold 5
python scripts/run_pipeline.py --input STEREO_01.mp4 --waste-threshold 5
python scripts/run_pipeline.py --input STEREO_01.mp4 --waste-threshold 8 --resume
python scripts/run_pipeline.py --input STEREO_01.mp4 --until sampling      # no Ollama needed
```

| Argument | Meaning |
|---|---|
| `--url URL` / `--input PATH` | Source. Exactly one is required. URL downloads go to the project root. |
| `--config PATH` | YAML overriding defaults (e.g. `configs/nvenc.yaml`). Unknown keys are rejected. |
| `--waste-threshold N` | Integer ≥ 1. Minimum continuous WASTE seconds to remove. |
| `--resume` | Reopen this video's run. Re-run only what's needed (see below). |
| `--until STAGE` | Stop after STAGE. The run is marked `partial`. |
| `--output-dir PATH` | Output root (default `output/`). |
| `--log-level LEVEL` | Console verbosity. |

Exit codes: `0` ok · `1` pipeline error · `2` usage/config error · `130` interrupted.

The Ollama endpoint comes from `.env` (`OLLAMA_BASE_URL`, `OLLAMA_MODEL`, optional
`OLLAMA_API_KEY`). See [inference.md](inference.md).

## Stages

| Stage | Phase | Reads | Writes |
|---|---|---|---|
| `ingestion` | 1 | source | `metadata/source.json`, `metadata/download.json` |
| `validation` | 1 | source | `metadata/validation.json` |
| `metadata` | 1 | source | `metadata/ffprobe.json`, `metadata/video_info.json` |
| `telemetry` | 2 | source, video_info | `imu/original/{telemetry.bin, gpmf_packets.json, parser_output.txt, accelerometer.csv, gyroscope.csv, imu_metadata.json}` |
| `sampling` | 3 | source, video_info | `frames/frame_index.jsonl`, `frames/sampling_metadata.json` (+ temporary images) |
| `inference` | 4 | frame_index, images | `inference/{classifications.jsonl, raw_responses.jsonl, inference_metadata.json, cache/}`; deletes each pair once done |
| `temporal` | 5 | classifications, video_info | `segmentation/{classifications_timeline.json, waste_candidates.json, segmentation_metadata.json}` |
| `verification` | 5 | candidates, source | `segmentation/{waste_intervals.json, retained_intervals.json, verification.json, verification_samples.jsonl}` |
| `segmentation` | 6 | source, retained/waste intervals | `processed/<name>_processed.mp4`, `processed/processing_timeline.json` |
| `imu_sync` | 7 | imu/original, processing_timeline | `imu/processed/{accelerometer.csv, gyroscope.csv, synchronization_metadata.json}` |
| `metrics` | 8 | everything above | `reports/{metrics.json, metrics.csv, summary.json}` |
| `qc` | – | everything above | `reports/qc_report.json` (fails the run on any failed check) |

### Phase 1: ingestion, validation, metadata
* **Source acquisition:**
  * The URL filename is resolved before downloading.
  * The download is streamed to `.part`, then renamed.
  * Transient errors are retried.
  * An existing file is never overwritten without confirmation.
* **Validation:** file checks (exists, non-empty, extension) plus container checks
  (≥ 1 real video stream, positive size and duration). There are warnings for no
  audio, no GPMF, or multiple video streams.
* **Metadata:** the full FFprobe JSON (format, streams, chapters, programs) is
  stored unchanged, plus a typed summary.

### Phase 2: telemetry
If `video_info.json` lists a GPMF stream, it is copied byte-for-byte into
`telemetry.bin`. The copy is checked against the packet table, which is saved with
per-packet timing and byte offsets. ACCL/GYRO are then parsed into CSVs. With no
GPMF stream, `imu_metadata.json` records `available: false` with a reason, and the
IMU stages skip. Details: [imu_synchronization.md](imu_synchronization.md).

### Phase 3: stereo sampling
* The left/right crops come from `stereo.split_ratio` and are validated against
  the real frame width (even widths; a warning if unequal or the frame is too narrow).
* One FFmpeg decode pass selects, for each `k`, the first frame with
  `pts ≥ start + k·interval`. Selection uses real timestamps, so it is safe for
  variable-frame-rate video.
* Each selected frame is cropped into left and right, so both images of a pair
  share one timestamp.
* Images are scaled to ≤ `max_width` and saved as JPEG.
* The target time, actual PTS and image SHA-256 are recorded per pair.
* Intervals with no frame are reported as missing, never invented.

### Phase 4: inference
Each pair is one request: image 1 = left, image 2 = right, versioned prompt,
JSON-schema structured output. Responses are validated:
* An invalid response → UNCERTAIN (`status: invalid_response`).
* A failed request (after retries) → UNCERTAIN or UNAVAILABLE per
  `inference.failure_policy`. **Never WASTE.**

Results are written live (one JSONL line per pair) and cached by content. See
[inference.md](inference.md).

### Frames are not stored
Sampling writes each pair to `frames/left|right/NNNNNN.jpg` only so the model can
see it. Inference deletes a pair as soon as it has a valid (cached) result. Failed
or invalid pairs keep their images until a retry succeeds. The durable record is
`frame_index.jsonl` (timestamps + SHA-256 of both images). Extraction is
deterministic, so a missing pair is re-extracted byte-identically when needed
(checked against the stored hashes). Set `sampling.keep_frames: true` to keep all
images.

### Phase 5: temporal candidates
* Sample `k` stands for `[k·I, (k+1)·I)`, clipped to the duration.
* Consecutive, touching samples with the same label form runs.
* A WASTE run with duration ≥ N (from timestamps) becomes a **candidate**. Every
  WASTE run gets a recorded decision.
* RELEVANT, UNCERTAIN, UNAVAILABLE and MISSING are preserved and break WASTE runs.
* Intervals are half-open `[start, end)`.

Example: with N = 5, WASTE at samples 2–6 gives the candidate `[2, 7)` (5 s).

### Phase 5b: verification (nothing with HOI is cut)
One sample per second can miss short interactions, so every candidate is re-sampled
every `verification.interval_seconds` (0.5 s) and classified (cached by content):
* every fine sample that is **not WASTE** (RELEVANT, UNCERTAIN, failed, or missing)
  is protected together with `verification.guard_seconds` (0.5 s) on each side;
* the unprotected pieces that still last **≥ N** are removed; shorter pieces are kept.

The outcome per candidate is `removed`, `partially_removed` or `kept`.
`verification.json` records every protected time and the reason for each decision.
The result is the final `waste_intervals.json` / `retained_intervals.json`.
Verification costs roughly `2 × candidate seconds` extra model calls; integer-second
samples come from the cache.

### Phase 6: segmentation
* Retained intervals are cut from the **full side-by-side frame** with
  `trim`/`setpts` (audio via `atrim` when present), then concatenated and re-encoded.
* `processing_timeline.json` maps each retained segment from original to processed
  time and records nominal vs measured output duration.
* Chapters/data streams are not copied, because their times refer to the original.

Verified on `STEREO_01.mp4`: frames at mapped times match the original
(PSNR > 50 dB).

### Phase 7: IMU synchronization
An IMU sample at original video time `t` is kept iff `t` is in a retained
interval. It gets `time_processed_seconds = processed_start + (t − original_start)`,
and the original time columns are kept. If there's no telemetry, it is skipped
with a recorded reason.

### Quality checks (`qc`, last stage)
Written to `reports/qc_report.json` and shown on the dashboard. Status is
`pass | warn | fail | skipped` per check. With `qc.fail_run_on_failure` (default),
any `fail` fails the run.

| Check | Fails when |
|---|---|
| `source.unchanged` | the source's SHA-256 (full re-hash) differs from ingestion |
| `decisions.partition` | removed + retained intervals don't tile `[0, duration)` exactly |
| `decisions.no_hoi_removed` | any removed time overlaps a non-WASTE sample (1 s grid or fine) |
| `decisions.min_duration` | a removed interval is shorter than N |
| `decisions.verified` | (warn) removal happened without fine verification |
| `inference.health` | failed + invalid results exceed `qc.max_inference_problem_ratio` (warn below it) |
| `inference.coverage` | (warn) seconds without any classification |
| `output.layout` | processed resolution differs from the source (stereo frame not intact) |
| `output.duration` | measured duration differs from the retained total by > 2 frames |
| `output.decode` | a full decode of the processed video reports errors |
| `output.alignment` | at up to `qc.alignment_checkpoints_max` points (start/middle/end of each segment), the processed frame vs the original frame at the mapped time scores below `qc.min_alignment_psnr_db` (warn if it only matches at ±1 frame) |
| `imu.sync` | synchronized sample counts don't add up, times are non-monotonic, or samples lie beyond the processed end |

On `STEREO_01.mp4`, all checkpoints matched at offset 0 with 41–47 dB.

### Phase 8: metrics
* Before/after durations per label come from intervals (after = intersection with
  retained intervals).
* Also reported: HOI coverage, waste reduction, retention, HOI interval counts,
  removed intervals, and latency.
* Outputs: `metrics.json`, `metrics.csv` (comparison table) and `summary.json`
  (headline).

## Resume and re-run rules

`--resume` requires the same source SHA-256. A stage is **skipped** only if **all**
of these hold:
1. it completed in a previous attempt,
2. all of its recorded outputs still exist,
3. the configuration keys it declares (`Stage.config_keys`) are unchanged
   (per-stage fingerprint), and
4. no earlier stage re-ran in this invocation.

So `--waste-threshold 8 --resume` re-runs `temporal → metrics` only, and
`--config configs/nvenc.yaml --resume` re-runs `segmentation → metrics`.
Changing `OLLAMA_BASE_URL` (e.g. a new ngrok URL) does not invalidate inference.

Without `--resume`, all stages run again. The previous manifest/log/config are
archived to `reports/history/<run_id>/`. Inference is still served from the cache
when images, model digest, prompt version and parameters match.

An interrupted inference stage resumes from the cache. Completed pairs are not
re-sent.
