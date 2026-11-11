# Stereo HOI Preprocessing Pipeline

Removes footage without Hand-Object Interaction (HOI) from egocentric side-by-side
stereo video. Stereo frame pairs are classified by a vision-language model
(Qwen3-VL via Ollama). Long stretches of waste are re-verified at a finer step and
cut frame-accurately, and the GoPro IMU telemetry is cut identically. Quality checks
gate the result, and a small web dashboard shows everything.

* **How to use it (start here):** [user_guide.md](user_guide.md)
* **Review and approve the output (QC):** [docs/qc_playbook.md](docs/qc_playbook.md)
* **Install / deploy:** [docs/guide.md](docs/guide.md)
* Specification: [technicalplan.md](technicalplan.md)

## Quick start (Windows)

```powershell
python -m venv venv; venv\Scripts\activate; pip install -r requirements.txt
copy .env.example .env
```
On the machine with the GPU: `ollama pull qwen3-vl:4b`, start Ollama, and (if the
pipeline runs elsewhere) `ngrok http 11434 --host-header="localhost:11434"`. Then in `.env`:
```ini
OLLAMA_BASE_URL=https://<name>.ngrok-free.dev     # or http://localhost:11434
OLLAMA_MODEL=qwen3-vl:4b
```
```powershell
python scripts/run_pipeline.py --input STEREO_01.mp4 --waste-threshold 5 --config configs/nvenc.yaml
python scripts/serve_ui.py      # http://localhost:8000 (updates live during the run)
```

## What it guarantees

* **Nothing with relevant HOI is cut.** Candidates (≥ N s of WASTE at 1 sample/s) are
  re-checked every 0.5 s. Any moment that isn't clearly waste is protected (±0.5 s),
  and only remaining pieces ≥ N s are removed. Uncertain or failed answers are never cut.
* **Frames are not stored.** Each stereo pair is deleted right after classification.
  It can be re-extracted byte-identically from the video if needed.
* **Quality-checked output.** QC covers the source hash, the decisions, the output
  layout, duration and decode, and frame alignment vs the original. Any failure
  fails the run.
* **Reproducible.** There's a run manifest, resolved config, model digest, versioned
  prompt, raw model responses, an original↔processed time map, and a log per run.

## Documentation

| Doc | Contents |
|---|---|
| [user_guide.md](user_guide.md) | **Day-to-day use: start, process, dashboard, results, settings, shut down** |
| [guide.md](docs/guide.md) | Install, share, deploy (Docker), troubleshooting |
| [qc_playbook.md](docs/qc_playbook.md) | Step-by-step review checklist, decisions, QC record |
| [pipeline.md](docs/pipeline.md) | Every stage, CLI, resume rules, QC checks |
| [architecture.md](docs/architecture.md) | Layers, dependency rules |
| [module_interfaces.md](docs/module_interfaces.md) | Public interfaces |
| [data_schema.md](docs/data_schema.md) | Every artifact's format |
| [inference.md](docs/inference.md) | Providers (Ollama, optional OpenAI-compatible), prompt, cache |
| [imu_synchronization.md](docs/imu_synchronization.md) | GPMF parsing and alignment |
| [reproducibility.md](docs/reproducibility.md) | Run identity, config precedence |
| [dashboard.md](docs/dashboard.md) | Web UI |
| [decisions.md](docs/decisions.md) | Decision log |
