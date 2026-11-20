# Run & Deploy Guide

* [Part 1: Run locally](#part-1-run-locally)
* [Part 2: Optional vision-language model](#part-2-optional-vision-language-model)
* [Part 3: Show it to someone](#part-3-show-it-to-someone)
* [Part 4: Deploy with Docker](#part-4-deploy-with-docker)
* [Troubleshooting](#troubleshooting)

---

## Part 1: Run locally

Everything runs on one computer. The default HOI detector (MediaPipe hand landmarks) needs no
GPU and no model server.

### One-time setup

| Need | How | Check |
|---|---|---|
| Python 3.11–3.13 | python.org installer | `python --version` |
| FFmpeg + FFprobe ≥ 6 on PATH | Windows: `winget install Gyan.FFmpeg` · Ubuntu: `apt install ffmpeg` · macOS: `brew install ffmpeg` | `ffmpeg -version` |

From the repository root:
```powershell
python -m venv venv
venv\Scripts\activate                          # macOS/Linux: source venv/bin/activate
pip install -r stereo_hoi_pipeline/requirements.txt
cd stereo_hoi_pipeline
```
The MediaPipe hand model is in `models/` (it is downloaded automatically if missing).

### Every session

1. **Put the video** in `stereo_hoi_pipeline/videos/` (side-by-side stereo).
2. **Run the pipeline** (from `stereo_hoi_pipeline/`):
   ```powershell
   python scripts/run_pipeline.py --input <video file> --waste-threshold 5
   ```
   * `--input` takes a file name from `videos/` or any path.
   * `--sample-fps 2` analyses 2 stereo pairs per second (default 1).
   * On an NVIDIA GPU add `--config configs/nvenc.yaml` for about 5× faster encoding.
   * Interrupted? Run the same command with `--resume`. Finished work is reused.
3. **Open the dashboard** in a second terminal (venv active, in `stereo_hoi_pipeline/`):
   ```powershell
   python scripts/serve_ui.py
   ```
   Then open http://localhost:8000. It updates live while the pipeline runs.
4. **Review:** follow the [QC playbook](qc_playbook.md).

Results are in `outputs/<video name>/`: the processed video in `processed/`, the reports in
`reports/` (`summary.json`, `metrics.csv`, `hand_metrics.json`, `qc_report.json`).

### What a run does

```text
ingestion → validation → metadata → telemetry → sampling → inference → temporal
          → verification → segmentation → imu_sync → metrics → qc
```
* **sampling + inference:** one stereo pair per sampling interval. MediaPipe finds the hands
  in each eye, and the pair is classified RELEVANT (a wearer hand is grasping), WASTE (no wearer
  hand) or UNCERTAIN (hands visible but not grasping; kept). The images are **deleted
  immediately** after classification.
* **temporal:** continuous WASTE of at least N seconds becomes a *candidate* for removal.
* **verification:** each candidate is re-checked at twice the sampling rate. Anything that is not
  clearly waste is protected (±0.5 s). A section is cut **only where no relevant HOI was
  found**, and only if the remaining piece still lasts ≥ N seconds.
* **segmentation:** a frame-accurate cut of the full stereo frame.
* **metrics:** before/after durations and the wearer's hand metrics.
* **qc:** checks the decisions, the output video and the telemetry. It fails the run if anything is
  wrong.

### Useful variations

```powershell
# Different threshold without re-detecting (re-cuts and re-checks only)
python scripts/run_pipeline.py --input <video file> --waste-threshold 8 --resume

# Denser analysis
python scripts/run_pipeline.py --input <video file> --sample-fps 3

# Let the vision-language model re-check every section before it is cut (see Part 2)
python scripts/run_pipeline.py --input <video file> --verify-with vlm

# Download from a URL into videos/ first
python scripts/run_pipeline.py --url "https://example.com/<video file>" --waste-threshold 5
```

---

## Part 2: Optional vision-language model

MediaPipe cannot see every hand. Strong glare, gloves, clay and heavy occlusion can hide hands
that are working, and those moments then look like "no hands". A vision-language model (Qwen-VL)
can be used in two ways:

| Option | Model requests | Use when |
|---|---|---|
| `--verify-with vlm` | only for sections about to be cut (a few dozen per video) | MediaPipe for speed, the model as a safety net |
| `--detector vlm` | every sample (≈ 5 s each on a 6 GB GPU) | footage where hand landmarks don't work at all |

Setup:
1. Copy `.env.example` to `.env` (in `stereo_hoi_pipeline/`) and set the endpoint:
   * local Ollama: install from ollama.com, `ollama pull qwen3-vl:4b`, `OLLAMA_BASE_URL=http://localhost:11434`;
   * Ollama on another machine through ngrok:
     `ngrok http 11434 --host-header="localhost:11434" --url=https://<your-name>.ngrok-free.dev`;
   * an OpenAI-compatible server (vLLM or hosted): `INFERENCE_PROVIDER=openai_compatible`,
     `VLLM_BASE_URL`, `VLLM_MODEL`, `VLLM_API_KEY`.
2. Check it: `python scripts/check_endpoint.py --input <video file>`. It must end with
   `Ready: run the pipeline.`

Details: [inference.md](inference.md).

---

## Part 3: Show it to someone

**Same network (office Wi-Fi):**
```powershell
python scripts/serve_ui.py --host 0.0.0.0
```
Others open `http://<your-IP>:8000` (find the IP with `ipconfig`). Allow Python through the
firewall on **Private** networks only.

**Over the internet (temporary link):** `ngrok http 8000`, then send the link. The dashboard is
read-only but has **no login**, so add ngrok authentication if your plan supports it, and stop the
tunnel after the demo. Never expose a model server (Ollama 11434, vLLM 8000) publicly without an API key.

---

## Part 4: Deploy with Docker

The Dockerfile and compose file are in `stereo_hoi_pipeline/`.
*They are syntax-checked but have not been built and run end to end. Do a test run first.*

```bash
cd stereo_hoi_pipeline
cp /path/to/<video file> videos/
docker compose up -d --build dashboard                              # dashboard on :8000
docker compose run --rm pipeline --input <video file> --waste-threshold 5
```
Results go to `./outputs/<video name>/` on the host.

| Service | Purpose | Exposed |
|---|---|---|
| `dashboard` | read-only UI over `./outputs` (and `./videos` for frame rendering) | port 8000 |
| `pipeline` (profile `run`) | one-off runs | – |
| `ollama` (profile `vlm`) | optional model server with GPU: `docker compose --profile vlm up -d ollama`, then `docker compose exec ollama ollama pull qwen3-vl:4b` | internal only |

Notes:
* The container uses Debian's FFmpeg with **libx264** (no NVENC).
* **Public access:** put the dashboard behind a reverse proxy with HTTPS and authentication.
* **Updates:** `git pull && docker compose up -d --build dashboard`.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `--input: file not found` | Put the video in `videos/`, or give the full path in quotes. |
| `MediaPipe is not installed` | `pip install -r requirements.txt` in the active venv. |
| Active work was cut | MediaPipe missed the hands (glare, gloves, clay). Re-run with `--verify-with vlm` (Part 2), or with `--waste-threshold` higher. |
| Too little is cut | Hands are visible but idle (UNCERTAIN, kept). Set `detection.mediapipe.hands_without_grasp: WASTE` in a `--config` file. |
| Dashboard frame says "Source video not found" | The source video was moved; put it back in `videos/`. |
| `Cannot reach Ollama at …` / `Cannot reach the OpenAI-compatible API at …` | Only with the model options: endpoint down or wrong URL/key. Run `scripts/check_endpoint.py`. |
| `Quality checks failed: …` | Open `reports/qc_report.json` (or the dashboard's QC card). Treat the video as not approved. |
| `h264_nvenc` errors | No NVIDIA GPU/driver. Drop `--config configs/nvenc.yaml`. |
| Want the frames for debugging | Set `sampling.keep_frames: true` in a `--config` file. |
| Dashboard empty | It reads `outputs/`. Run the pipeline first, or point `--output-dir` at the right folder. |
