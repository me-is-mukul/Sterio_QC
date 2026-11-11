# Reproducibility

## What identifies a run

| Item | Where |
|---|---|
| Run ID `YYYYMMDDTHHMMSSZ-<8 hex>` (UTC, sortable, unique) | `manifest.run_id`, every log line |
| Source SHA-256, size, mtime | `manifest.source`, `metadata/source.json` |
| Download provenance (URL, final URL, headers, bytes) | `metadata/download.json` |
| Full resolved configuration | `reports/resolved_config.yaml`, `manifest.configuration.snapshot` |
| Configuration fingerprint (SHA-256) | `manifest.configuration.fingerprint` |
| Config layers that were applied, CLI overrides | `manifest.configuration.layers`, `.cli_overrides` |
| Python / FFmpeg / FFprobe / Ollama / pipeline versions | `manifest.versions` |
| External commands + exit codes + timing | `manifest.commands` |
| Status and timing of every stage and attempt | `manifest.stages`, `manifest.attempts` |

## Configuration precedence

1. CLI arguments (`--waste-threshold`, `--output-dir`, `--log-level`)
2. Environment / `.env`: `INFERENCE_PROVIDER`, then only the selected provider's
   variables (`VLLM_BASE_URL`/`VLLM_MODEL` or `OLLAMA_BASE_URL`/`OLLAMA_MODEL`). The
   process environment wins over `.env`. The manifest records the variable *names*.
3. User file (`--config`), merged strictly: an unknown key is an error, so typos can't be silently ignored
4. Defaults: `configs/default.yaml` → `inference.yaml` → `processing.yaml`. Each
   top-level section is owned by exactly one file. Duplicates are an error.

The merged result is validated (types and ranges) before anything runs. Secrets
(`VLLM_API_KEY`, `OLLAMA_API_KEY`) never enter the configuration. For OpenAI-compatible
endpoints, which expose no weight digest, the recorded model identity is a hash of the
served model id + checkpoint root. Pin the exact checkpoint on the server side if
bit-for-bit model identity matters.

### Fingerprints
* **Run:** `sha256(canonical_json(config without "logging" and "web"))`.
* **Stage:** `sha256` of the values of the stage's `config_keys`, stored per stage.
  It drives `--resume` (see [pipeline.md](pipeline.md#resume-and-re-run-rules)).
* **Inference cache key:** left/right image hashes, provider, model, model digest,
  prompt version, temperature, think, and cache format (plan §15).

## Data integrity guarantees (Phase 1)

* The source video is opened read-only. Its size and mtime are checked at the end
  of every run, and before/after hashing. A change aborts with `SourceModifiedError`.
* Downloads land in `<name>.part` and are renamed only once complete. An existing
  file is never overwritten without interactive confirmation.
* The source must not be inside the output workspace.
* All JSON/YAML artifacts are written atomically.
* The raw FFprobe output is stored unmodified next to the derived summary.
* Earlier runs are archived, never deleted (`reports/history/<run_id>/`).

## Resuming

See [pipeline.md](pipeline.md#resume-semantics). Resume is keyed by video name.
It requires the same source hash and skips only stages whose outputs still exist.

## Known limitations

* Re-running without `--resume` overwrites the previous run's `metadata/*` files
  (they are regenerated deterministically from the same source). Only manifest,
  log and resolved config are archived.
* Ollama's version is taken from the server's `/api/version`, or from the CLI client
  when the server is down (flagged as such). The model digest from `/api/tags` is
  recorded in `inference_metadata.json`, in every classification, and in the manifest.
* LLM sampling is not bit-reproducible even at low temperature. Reproducibility of
  labels comes from the cache plus the stored raw responses, not from re-querying.
* Re-encoding with NVENC vs libx264 gives different bytes but the same timeline.
  The encoder and its arguments are recorded in `processing_timeline.json`.
