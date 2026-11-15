# Stereo QC: egocentric video quality tools

Two independent tools for egocentric (first-person) video:

* **Hand & technical QC** ([`hand_qc/`](hand_qc/)): turns a CSV list of clips into a PASS/FAIL
  checklist per clip. It covers hand presence, hand-object interaction, tracking, 21 hand landmarks
  and technical checks such as bitrate, sharpness and blockiness. It runs locally with MediaPipe and
  FFmpeg.
* **Stereo HOI pipeline** ([`src/`](src/), [`scripts/`](scripts/)): removes stretches without
  hand-object interaction from side-by-side stereo video, using a vision-language model. It includes
  quality checks and a web dashboard.

## Setup (once)

Requires Python 3.11–3.13 and FFmpeg ≥ 6.0 (`ffmpeg` and `ffprobe` on PATH).

```bash
python -m venv venv
# Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate
pip install -r requirements.txt -r hand_qc/requirements.txt
```

## Hand & technical QC

```bash
cd hand_qc
python run_hand_qc.py --csv <file path> --workers 4
```

* Results go to `hand_qc/results/<csv file name>/`: `qc_checklist.xlsx` (the checklist) and
  `qc_report.html` (interactive report).
* The same command resumes an interrupted run. Add `--score-only` to re-score in seconds after
  editing the thresholds in [`hand_qc/config/hand_qc.yaml`](hand_qc/config/hand_qc.yaml).

### CSV format

One header row, then one row per clip:

```csv
Dataset,Category,Task,Camera,URL_or_File,Notes
Mono,<category>,<task>,,https://storage.googleapis.com/<bucket>/<path>.mp4,
Stereo,<category>,<task>,<camera>,gs://<bucket>/<path>.mp4,
Six-Cam,<category>,,<camera>,gs://<bucket>/<path>.mp4,<notes>
```

| Column | Required | Content |
|---|---|---|
| `Dataset` | yes | `Mono` (single view), `Stereo` (left/right side by side) or `Six-Cam` (3×2 grid). Sets how frames are split into camera views. |
| `URL_or_File` | yes | Public Google Cloud Storage link (`https://storage.googleapis.com/...` or `gs://...`). Rows with an empty or unresolvable link are reported as `SOURCE MISSING`. |
| `Category`, `Task`, `Camera`, `Notes` | column required, value optional | Free text, copied into the checklist |

## Stereo HOI pipeline

1. Copy `.env.example` to `.env` and set the model endpoint (Ollama or an OpenAI-compatible server).
2. Check the endpoint: `python scripts/check_endpoint.py --input <file path>`
3. Run, then watch progress at http://localhost:8000:
   ```bash
   python scripts/run_pipeline.py --input <file path> --waste-threshold 5
   python scripts/serve_ui.py
   ```

## Documentation

| Hand & technical QC | Stereo HOI pipeline |
|---|---|
| [Usage and PASS rules](hand_qc/README.md) | [User guide](user_guide.md) |
| [Methodology and limitations](hand_qc/docs/METHODOLOGY.md) | [Install, deploy, troubleshooting](docs/guide.md) |
| [Output file formats](hand_qc/docs/DATA_FORMAT.md) | [Review playbook](docs/qc_playbook.md) |
| | [Pipeline stages and CLI](docs/pipeline.md) · [Inference](docs/inference.md) · [Dashboard](docs/dashboard.md) |
| | [Data schema](docs/data_schema.md) · [IMU sync](docs/imu_synchronization.md) · [Architecture](docs/architecture.md) |
| | [Interfaces](docs/module_interfaces.md) · [Reproducibility](docs/reproducibility.md) · [Decisions](docs/decisions.md) |

Not in the repository (see `.gitignore`): `.env`, virtual environments, videos, and all generated
output (`output/`, `hand_qc_cache/`, `hand_qc/results/`).
