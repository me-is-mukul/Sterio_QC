# Stereo HOI pipeline

Removes the stretches **without hand-object interaction (HOI)** from side-by-side stereo
egocentric video. Everything with HOI is kept, and the result comes with metrics, quality checks
and a live web dashboard.

```text
videos/<video file> ─► sample 1 stereo pair/s ─► detect the wearer's hands (MediaPipe)
                    ─► ≥ N s without hands = candidate ─► re-check at 2 pairs/s ─► cut
                    ─► quality checks ─► outputs/<video name>/ + dashboard
```

* **Detector:** MediaPipe hand landmarks + grasp rules by default. It runs locally (≈ 70 ms per pair),
  with no GPU or model server needed. A vision-language model (Qwen-VL) is optional.
* **Nothing uncertain is cut:** hands visible but not grasping, failed detections and anything near
  a relevant moment are always kept.
* **Frames are not stored.** The dashboard renders any analysed frame on demand, with the
  detected hand landmarks drawn on it.

## Folders

```text
stereo_hoi_pipeline/
├── videos/                 put input videos here
├── outputs/<video name>/   everything one run produces (see below)
├── configs/                settings (default.yaml, inference.yaml, processing.yaml, nvenc.yaml)
├── models/                 MediaPipe hand model
├── scripts/                run_pipeline.py, serve_ui.py, check_endpoint.py
├── src/video_pipeline/     the code
└── docs/                   documentation
```

## Run

Setup once (from the repository root): `pip install -r stereo_hoi_pipeline/requirements.txt`,
plus FFmpeg ≥ 6 on PATH. Then, from `stereo_hoi_pipeline/`:

```bash
python scripts/run_pipeline.py --input <video file> --waste-threshold 5     # process videos/<video file>
python scripts/serve_ui.py                                                # dashboard: http://localhost:8000
```

| Option | Meaning |
|---|---|
| `--input <file>` | Video in `videos/` (or any path). `--url <link>` downloads into `videos/` first. |
| `--waste-threshold N` | Shortest stretch without HOI that is cut, in seconds (default 5). |
| `--sample-fps FPS` | Stereo pairs analysed per second of video (default 1). |
| `--detector mediapipe\|vlm` | How pairs are classified (default `mediapipe`). |
| `--verify-with vlm` | Let the vision-language model re-check only the sections about to be cut (safety net for hands MediaPipe cannot see). |
| `--config configs/nvenc.yaml` | Encode on an NVIDIA GPU (≈ 5× faster). |
| `--resume` | Continue an interrupted run, or re-cut with a new threshold, reusing finished work. |

`python scripts/run_pipeline.py --help` lists every option. The model options need `.env`
(copy `.env.example`); see [docs/guide.md](docs/guide.md#part-2-optional-vision-language-model).

## Results: `outputs/<video name>/`

| File | Content |
|---|---|
| `processed/<video name>_processed.mp4` | the video without the waste stretches |
| `reports/summary.json` | headline numbers and verdicts |
| `reports/hand_metrics.json` | the wearer's hand metrics, original vs processed (hands in frame, both hands, grasping, longest gap, tracking losses, other people's hands) |
| `reports/metrics.csv` | before/after table (opens in Excel) |
| `reports/qc_report.json` | quality checks |
| `inference/classifications.jsonl` | per sampled pair: label, description, 21 landmarks of every hand |
| `processed/processing_timeline.json` | which original time ranges are in the processed video |

## Known limitation

MediaPipe misses hands it cannot see well: strong glare, gloves, clay or paint, heavy occlusion.
Those moments count as "no hands" and can be cut even though work is happening. On the sample
video, 22.5 s of hammering in direct sunlight was missed. For such footage use
`--verify-with vlm`, and review cuts with the [QC playbook](docs/qc_playbook.md).

## Egocentric QA on recordings in Cloud Storage

A second entry point, `scripts/run_gcs_batch.py`, applies the QA pass/fail sheet to the recordings
in `gs://op-datoric/raw/`, in every format found there (MCAP, robocap, Hydra, EGO-CAM live, EGO-CAM
`capture.egoc`, detected per session folder), and writes `qa_report.json`, `calibration.json`,
`labels.csv` and the kept clips (both eyes, no re-encode) to `gs://op-datoric/processed/`.
Settings: `configs/ego_qa.yaml`. Details: [docs/ego_qa.md](docs/ego_qa.md).
It does not change the MP4 pipeline above.

## Documentation

| Document | Contents |
|---|---|
| [docs/ego_qa.md](docs/ego_qa.md) | Egocentric QA batch for recordings in Cloud Storage (all formats) |
| [docs/guide.md](docs/guide.md) | Setup, everyday use, optional model, sharing, Docker, troubleshooting |
| [docs/qc_playbook.md](docs/qc_playbook.md) | How to review and approve a processed video |
| [docs/pipeline.md](docs/pipeline.md) | Every stage, CLI options, resume rules, QC checks |
| [docs/inference.md](docs/inference.md) | Detectors: MediaPipe rules, vision-language model setup |
| [docs/dashboard.md](docs/dashboard.md) | Web dashboard and its API |
| [docs/data_schema.md](docs/data_schema.md) | Format of every output file |
| [docs/imu_synchronization.md](docs/imu_synchronization.md) | GoPro IMU parsing and alignment |
| [docs/architecture.md](docs/architecture.md) · [module_interfaces.md](docs/module_interfaces.md) | Code structure |
| [docs/reproducibility.md](docs/reproducibility.md) · [decisions.md](docs/decisions.md) | Run identity, design decisions |
