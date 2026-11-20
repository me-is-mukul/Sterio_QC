# Stereo QC: egocentric video quality tools

Two independent modules for egocentric (first-person) video. Each one keeps its input videos in
`videos/` and its results in `outputs/`.

| Module | What it does |
|---|---|
| [`stereo_hoi_pipeline/`](stereo_hoi_pipeline/) | Cuts the stretches without hand-object interaction out of a side-by-side stereo video, using MediaPipe hand landmarks (a vision-language model is optional). It produces the processed video, hand metrics, quality checks and a live web dashboard. |
| [`hand_qc/`](hand_qc/) | Turns a CSV list of clips into a PASS/FAIL checklist per clip. It covers hand presence, hand-object interaction, tracking, 21 hand landmarks and technical checks (bitrate, sharpness, blockiness, …). |

```text
stereo_hoi_pipeline/                     hand_qc/
├── videos/          input videos        ├── clip_lists/      input clip lists (CSV)
├── outputs/<video>/ results             ├── videos/          downloaded videos (automatic)
├── configs/  scripts/  src/  docs/      ├── outputs/<list>/  results
└── README.md                            ├── config/  handqc/  docs/
                                         └── README.md
```

## Setup (once)

Requires Python 3.11–3.13 and FFmpeg ≥ 6.0 (`ffmpeg` and `ffprobe` on PATH).

```bash
python -m venv venv
# Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate
pip install -r stereo_hoi_pipeline/requirements.txt -r hand_qc/requirements.txt
```

## Stereo HOI pipeline

Put the video in `stereo_hoi_pipeline/videos/`, then:

```bash
cd stereo_hoi_pipeline
python scripts/run_pipeline.py --input <video file> --waste-threshold 5
python scripts/serve_ui.py                      # dashboard: http://localhost:8000
```

Results are written to `stereo_hoi_pipeline/outputs/<video name>/`. Useful options:
`--sample-fps <frames per second>`, `--verify-with vlm` and `--resume`. See
[stereo_hoi_pipeline/README.md](stereo_hoi_pipeline/README.md).

## Hand & technical QC

```bash
cd hand_qc
python run_hand_qc.py --csv <file path> --workers 4
```

Results are written to `hand_qc/outputs/<csv file name>/`: `qc_checklist.xlsx` and
`qc_report.html`.

### Clip list (CSV) format

One header row, then one row per clip (template:
[`hand_qc/clip_lists/example_clip_list.csv`](hand_qc/clip_lists/example_clip_list.csv)):

```csv
Dataset,Category,Task,Camera,URL_or_File,Notes
Mono,<category>,<task>,,https://storage.googleapis.com/<bucket>/<path>.mp4,
Stereo,<category>,<task>,<camera>,gs://<bucket>/<path>.mp4,
Six-Cam,<category>,,<camera>,gs://<bucket>/<path>.mp4,<notes>
```

| Column | Required | Content |
|---|---|---|
| `Dataset` | yes | `Mono` (single view), `Stereo` (left/right side by side) or `Six-Cam` (3×2 grid) |
| `URL_or_File` | yes | Public Google Cloud Storage link (`https://storage.googleapis.com/...` or `gs://...`). Rows without a usable link are reported as `SOURCE MISSING`. |
| `Category`, `Task`, `Camera`, `Notes` | column required, value optional | Free text, copied into the checklist |

## Documentation

| Stereo HOI pipeline | Hand & technical QC |
|---|---|
| [Overview and usage](stereo_hoi_pipeline/README.md) | [Usage and PASS rules](hand_qc/README.md) |
| [Setup, model option, Docker, troubleshooting](stereo_hoi_pipeline/docs/guide.md) | [Methodology and limitations](hand_qc/docs/METHODOLOGY.md) |
| [Review playbook](stereo_hoi_pipeline/docs/qc_playbook.md) | [Output file formats](hand_qc/docs/DATA_FORMAT.md) |
| [Stages and CLI](stereo_hoi_pipeline/docs/pipeline.md) · [Detectors](stereo_hoi_pipeline/docs/inference.md) · [Dashboard](stereo_hoi_pipeline/docs/dashboard.md) | |
| [Data schema](stereo_hoi_pipeline/docs/data_schema.md) · [IMU sync](stereo_hoi_pipeline/docs/imu_synchronization.md) · [Architecture](stereo_hoi_pipeline/docs/architecture.md) | |
| [Interfaces](stereo_hoi_pipeline/docs/module_interfaces.md) · [Reproducibility](stereo_hoi_pipeline/docs/reproducibility.md) · [Decisions](stereo_hoi_pipeline/docs/decisions.md) | |

Not in the repository (see `.gitignore`): `.env`, virtual environments, videos, clip lists with
private links, and all generated output.
