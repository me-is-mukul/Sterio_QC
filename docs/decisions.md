# Decision Log

Newest last. Each entry: decision, reason, consequence.

### D1 — `src/` layout with a `core/` infrastructure package (Phase 1)
Shared concerns (config, logging, workspace, hashing, subprocess, I/O, errors)
live in `core/`, not in a domain package. This keeps the separation of concerns
(plan §17). Domain packages match the plan's tree exactly.

### D2 — Stages are adapters; domain modules are config-agnostic
`orchestration/stages.py` reads configuration and calls domain functions with
plain arguments. Domain modules never import orchestration. This makes them
reusable and testable in isolation, and lets the runner depend only on `Stage`.

### D3 — Three default config files, one owner per section
`default.yaml` (project/video/download/tools/metadata/hashing/environment/logging/
reporting), `inference.yaml` (inference), `processing.yaml` (stereo/sampling/hoi/
segmentation). Defining a section twice is an error, so every key has a single
source of truth. User `--config` files may override any section.

### D4 — Strict user overrides
Unknown keys in `--config` raise `ConfigError`. A typo such as
`waste_treshold_seconds` would otherwise be silently ignored, which conflicts
with the reproducibility goals.

### D5 — Only `preserve` for `hoi.uncertain_policy`, only `side_by_side` layout
These are the only behaviours the plan defines. Rejecting other values now
prevents configurations that later code would silently misinterpret.

### D6 — Standard-library HTTP for downloads (no `requests`)
`urllib` covers streaming, redirects and timeouts. That keeps the runtime
dependencies to PyYAML only (plan: avoid unnecessary dependencies).

### D7 — Filename resolved before downloading
A HEAD request (falling back to the URL path) determines the filename first. Name
conflicts are then caught before transferring gigabytes, and `--resume` can reuse
an already-downloaded file.

### D8 — Overwrite requires interactive confirmation
Non-interactive sessions (or a closed stdin) never overwrite. The error tells the
user to pass the existing file via `--input`. There is no `--force` flag yet,
because the plan asks for *explicit user confirmation*.

### D9 — Run ID format `YYYYMMDDTHHMMSSZ-<8 hex>`
Human-readable, sorts chronologically, and unique enough for local runs.

### D10 — One workspace per video name; earlier runs archived
Output is `output/<video_name>/` as the plan specifies. A new run moves the
previous manifest, log and resolved config to `reports/history/<run_id>/`. This
gives each run a separate log (plan §16) at the plan's path. A same-named
workspace for a different file (hash mismatch) is refused rather than mixed.

### D11 — Resume = same run ID, new attempt
`--resume` keeps the run ID, appends to `attempts`, and skips stages only when
they are `completed` and their outputs still exist. A configuration change is
allowed but logged and recorded in `fingerprint_history`.

### D12 — Early log records are buffered
The workspace (and so the log file) is known only after the source filename is
known. A `MemoryHandler` buffers earlier records and flushes them into the log,
tagged with the final run ID.

### D13 — Logging excluded from the config fingerprint
Changing the console verbosity must not count as a configuration change.

### D14 — FFprobe runs once per invocation
Validation and metadata extraction share the probe result in memory
(`RunContext.state`). After a resume, metadata re-probes if needed, because
in-memory state is never relied upon across invocations.

### D15 — `-count_frames` off by default
Exact frame counting decodes the whole file. That is slow for long 4K stereo
footage and isn't needed, because sampling is timestamp-based (plan §7.3).

### D16 — Source immutability check by size + mtime
Rehashing multi-GB files at the end of every run is costly. Size + `mtime_ns` is
checked before/after hashing and at the end of the run. The full SHA-256 is
recomputed at the start of every invocation, including resumes.

### D17 — Later-phase modules exist as docstring-only placeholders *(superseded by D18)*
They created the plan's directory structure in Phase 1 without committing to
interfaces that hadn't been designed yet.

### D18 — Phases 2–8 implemented together, against a real sample video
`STEREO_01.mp4` became available, so the remaining phases were built in plan
order and checked end to end. The order still holds: each phase reads only the
artifacts of earlier phases.

### D19 — Ollama endpoint from `.env`, secrets kept out of config
`OLLAMA_BASE_URL` / `OLLAMA_MODEL` form an environment layer between `--config` and
the CLI, because the ngrok URL changes between sessions and shouldn't require
editing YAML. `OLLAMA_API_KEY` is read only by the provider factory and is never
stored. No `python-dotenv` dependency: the format is trivial.

### D20 — Stage-level config fingerprints + downstream invalidation
Each stage declares the config keys it depends on. On `--resume`, a stage re-runs
if those change or if any earlier stage re-ran. That makes "try another N" cheap
(temporal → metrics only) and correct. `inference.base_url`, timeouts and retries
are excluded from the inference fingerprint, so a new tunnel URL does not
invalidate results.

### D21 — Sampling in one decode pass with `select` on timestamps
Per-timestamp seeks would decode from the nearest keyframe 150+ times. A single
pass with `select='isnan(prev_selected_t)+gt(bucket(t), bucket(prev_selected_t))'`
picks the first frame at or after each grid time. It is deterministic and
VFR-safe, and took 23 s for 150 s of 3840×1200 HEVC. Actual PTS values come from
`showinfo`. Left/right are crops of the same decoded frame.

### D22 — Sample k covers `[k·I, (k+1)·I)`
A sample stands for the interval that starts at its timestamp. This matches the
plan's worked example (WASTE at 2–6 s, N = 5 → `[2, 7)`) and gives exact
interval arithmetic for metrics. It is documented as approximate (plan §9.4).

### D23 — Structured output + strict validation; only valid results are cached
Ollama's `format` JSON schema constrains the model. The classifier still
validates everything, and an invalid label becomes UNCERTAIN, never WASTE.
Invalid/failed results are not cached, so they are retried next time.

### D24 — Re-encode the full frame with trim/concat
Frame-accurate cutting needs re-encoding. Cutting the full side-by-side frame
keeps the stereo views in lockstep. `libx264` is the portable default;
`configs/nvenc.yaml` switches to `h264_nvenc` (27 s vs 137 s for the sample on
the RTX 4050). Chapters aren't copied because their times would be wrong.

### D25 — IMU channels exported as c0..cN with ORIN recorded
GoPro axis order varies by model. Naming the columns x/y/z would be an
assumption, so the declared order is recorded instead.

### D26 — No pandas/numpy for IMU
Filtering timestamped rows and remapping times needs only `csv` + `bisect`.
Runtime dependencies stay at PyYAML only.

### D27 — Read-only stdlib dashboard
A single HTML file plus `http.server` with Range support. It has no build step or
dependencies and cannot affect runs. It reads the same artifacts the reports do,
so it can't disagree with them.

### D28 — Frames are deleted right after classification
Requested: frames are an intermediate, not an output. Each pair is deleted once it
has a valid cached result. Failed/invalid pairs keep their images until a retry
succeeds. Windowed extraction was verified to be byte-identical to the full pass,
so the index hashes stay valid and any pair can be regenerated on demand
(`PairExtractor`). `sampling.keep_frames: true` restores the old behaviour.

### D29 — Remove a section only where no relevant HOI was found (verification stage)
Requested: a section may only be removed when relevant HOI is not happening in it.
At 1 sample/s a short interaction between samples can be missed, so every
candidate is re-sampled at 0.5 s. Each non-WASTE fine sample (including
UNCERTAIN/failed/missing) is protected with a ±0.5 s guard, and only remaining
pieces ≥ N are removed. Temporal analysis now yields *candidates*; verification
writes the final decision.

### D30 — QC is a gate, not a report
Requested: proper quality checks. A `qc` stage verifies the source, decisions,
output layout/duration/decodability, frame alignment and IMU consistency. Any
failure fails the run (`qc.fail_run_on_failure`). Metrics run *before* QC so the
reports exist for diagnosis even when QC fails.

### D31 — PSNR alignment measured on the nominal frame grid
The source's frame timestamps sit slightly below the 1/30 s grid, while the
re-encoded output's sit on it. Seeking both to the same time therefore compared
adjacent frames (a false one-frame offset), and FFmpeg's `psnr` filter also skipped
frames when the two inputs' timebases differed. Checkpoints are snapped to frame
centres and seeked half a frame early, and both frames get identical timebase/pts
before `psnr`. Result on the sample: 12/12 checkpoints at offset 0, 41–47 dB.

### D32 — Ollama answer may arrive in `message.thinking`
With `think: false` plus a JSON-schema `format`, Qwen3-VL on Ollama 0.32 put the
answer in `thinking` and left `content` empty. The client falls back to `thinking`
when `content` is empty. The answer is still fully validated, and both fields are
kept in the raw response.

### D33 — Docker deployment: Ollama and pipeline in separate containers
Ollama's official image (GPU) plus one image for pipeline + dashboard. Ollama is
not published to the host. The dashboard mounts `output/` read-only. Debian FFmpeg
has no NVENC, so containers encode with libx264.

### D34 — Qwen3.6-35B-A3B via an OpenAI-compatible provider (vLLM / hosted)
Requested: switch to Qwen3.6 35B MoE. The smallest Ollama build (`qwen3.6:35b`) is
23 GB, which can't fit in the laptop's 6 GB VRAM + 15 GB RAM, so the user chose vLLM /
a hosted API. Added `OpenAICompatibleProvider` (Chat Completions with two image parts,
`response_format` json_schema, `chat_template_kwargs.enable_thinking`) next to Ollama.
Shared HTTP/retry code moved to `inference/http_json.py`. The default provider is now
`openai_compatible` with model `Qwen/Qwen3.6-35B-A3B`. `INFERENCE_PROVIDER` selects the
provider, and only that provider's env vars apply, so the two sets can't silently mix.
OpenAI APIs have no weight digest, so the model identity is a hash of the served id +
checkpoint root. `inference.max_output_tokens` bounds generation for both providers.
The compose file gains a `vllm` service (`vllm/vllm-openai:v0.32.0`,
`--limit-mm-per-prompt '{"image": 2}'`). Ollama moves behind a profile.

### D35 — Back to Ollama (+ ngrok) as the default; workspace cleanup
Requested: keep it simple: Ollama on the user's machine, exposed through ngrok. The
default provider and model are `ollama` / `qwen3-vl:4b` again, so the existing `.env`
works unchanged. The OpenAI-compatible provider stays as an optional, tested
alternative (the plan requires a replaceable provider). The vLLM service was removed
from `docker-compose.yml`. Dead code (`waste_detector.split_intervals`) was removed.
Added `docs/qc_playbook.md` for reviewers.

### D36 — One run per video (workspace lock) and an exclusive dashboard port
Incident: two runs on `STEREO_01` started 90 s apart (an automatic watcher and a manual
command). Both appended to `inference/classifications.jsonl`; the interleaved writes
produced an invalid line and both runs failed at `temporal`. The content-addressed cache
was unaffected (atomic per-file writes), so no model work was lost. Fix:
`core/locking.py::RunLock`, an `O_EXCL` lock file `output/<video>/.run.lock` holding the
owner PID, taken before the manifest is touched. Live owner → clear error; dead owner →
stale lock replaced. The Windows liveness check uses `OpenProcess`/`GetExitCodeProcess`
(never `os.kill`, which terminates processes on Windows). Separately, `http.server` sets
`SO_REUSEADDR`, which on Windows let two dashboards bind port 8000 at once; the
dashboard now binds with `SO_EXCLUSIVEADDRUSE` and reports a clear error instead.
