# Run & Deploy Guide

* [Part 1: Run locally (Windows laptop)](#part-1--run-locally-windows-laptop)
* [Part 2: Show it to someone (share the dashboard)](#part-2--show-it-to-someone)
* [Part 3: Deploy on a GPU server (Docker)](#part-3--deploy-on-a-gpu-server-docker)
* [Troubleshooting](#troubleshooting)

---

## Part 1: Run locally (Windows laptop)

The model runs in **Ollama** on the machine with the GPU, which is reachable
directly or through an **ngrok** tunnel. The pipeline and dashboard run on any
Windows machine.

### One-time setup

| Need | How | Check |
|---|---|---|
| Python 3.11+ | python.org installer | `python --version` |
| FFmpeg + FFprobe on PATH | e.g. gyan.dev build, add `bin` to PATH | `ffmpeg -version` |
| Ollama (GPU machine) | ollama.com installer | http://localhost:11434 → "Ollama is running" |
| The model | `ollama pull qwen3-vl:4b` | `ollama list` |
| ngrok (only if the pipeline runs on another machine) | ngrok.com, `ngrok config add-authtoken <token>` | `ngrok version` |

In the project folder:
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```
In `.env`:
```ini
OLLAMA_BASE_URL=https://<your-name>.ngrok-free.dev   # tunnel; or http://localhost:11434 on the same machine
OLLAMA_MODEL=qwen3-vl:4b
```
(Optional alternative backend, vLLM or a hosted OpenAI-compatible API: see
[inference.md](inference.md).)

### Every session

1. **Start the model endpoint** on the GPU machine:
   * start Ollama from the Start menu (tray icon);
   * if you use the tunnel, run this in its own terminal and keep it open:
     `ngrok http 11434 --host-header="localhost:11434" --url=https://<your-name>.ngrok-free.dev`
   * check: `curl.exe -H "ngrok-skip-browser-warning: true" <OLLAMA_BASE_URL>/api/tags`
     lists the model.
2. **Open a terminal** in the project folder and run `venv\Scripts\activate`.
3. **Run the pipeline:**
   ```powershell
   python scripts/run_pipeline.py --input STEREO_01.mp4 --waste-threshold 5 --config configs/nvenc.yaml
   ```
   * `--config configs/nvenc.yaml` encodes on the NVIDIA GPU (about 5× faster). Leave it
     out on machines without NVENC.
   * The run prints one line per stereo pair. With qwen3-vl:4b on an RTX 4050 that's
     about 5 s per pair.
   * The first lines show the provider, host and model it connected to. A wrong URL,
     key or model id stops the run immediately with a clear message.
   * Interrupted? Run the same command with `--resume`. Finished pairs are cached
     (per model, so switching model starts fresh).
4. **Open the dashboard** in a second terminal (venv active):
   ```powershell
   python scripts/serve_ui.py
   ```
   Then open http://localhost:8000. It updates live while the pipeline runs.
5. **Review:** follow the [QC playbook](qc_playbook.md).
6. **Shut down:** Ctrl+C in the pipeline, dashboard and ngrok terminals, then right-click
   the Ollama tray icon → Quit.

### What a run does

```text
ingestion → validation → metadata → telemetry → sampling → inference → temporal
          → verification → segmentation → imu_sync → metrics → qc
```
* **sampling + inference:** one stereo pair per second. Each pair is classified
  RELEVANT / WASTE / UNCERTAIN, and the images are **deleted immediately** afterwards.
* **temporal:** continuous WASTE of at least N seconds becomes a *candidate* for removal.
* **verification:** each candidate is re-checked every 0.5 s. Anything that is not
  clearly waste is protected (±0.5 s). A section is cut **only where no relevant HOI
  was found**, and only if the remaining piece still lasts ≥ N seconds.
* **segmentation:** a frame-accurate cut of the full stereo frame.
* **qc:** checks the decisions, the output video and the telemetry. It fails the
  run if anything is wrong. See `output/<video>/reports/qc_report.json`.

### Useful variations

```powershell
# Try a different threshold without asking the model again (re-cuts and re-checks only)
python scripts/run_pipeline.py --input STEREO_01.mp4 --waste-threshold 8 --resume --config configs/nvenc.yaml

# Download from a URL into the project folder first
python scripts/run_pipeline.py --url "https://example.com/GX010140.MP4" --waste-threshold 5

# Stop after a given stage (e.g. no model needed)
python scripts/run_pipeline.py --input STEREO_01.mp4 --until sampling
```

---

## Part 2: Show it to someone

**Same network (office Wi-Fi):**
```powershell
python scripts/serve_ui.py --host 0.0.0.0
```
Others open `http://<your-laptop-IP>:8000` (find the IP with `ipconfig`). Windows may
ask to allow Python through the firewall: allow it on **Private** networks only.

**Over the internet (temporary link):** tunnel the *dashboard* (not Ollama):
```powershell
ngrok http 8000
```
Send the `https://….ngrok-free.dev` link. The dashboard is read-only but has **no
login**, so add ngrok authentication if your plan supports it (e.g.
`--basic-auth "user:password"`), and stop the tunnel after the demo.

Never expose a model server (Ollama 11434, vLLM 8000) publicly without an API key.
Anyone with the URL could use your GPU.

---

## Part 3: Deploy on a GPU server (Docker)

For a Linux machine with an NVIDIA GPU that runs everything continuously.
*The Dockerfile and compose file are syntax-checked but have not yet been built or
run end to end. Do a test run before relying on them.*

### Server prerequisites
* Docker Engine + Compose plugin
* NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
  (check: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`)
* ≥ 6 GB GPU memory for `qwen3-vl:4b`

### Install and start
```bash
git clone <your-repo-url> stereo-QC && cd stereo-QC
mkdir -p videos output
cp /path/to/STEREO_01.mp4 videos/

docker compose up -d --build ollama dashboard          # Ollama (GPU) + dashboard on :8000
docker compose exec ollama ollama pull qwen3-vl:4b     # once; stored in a named volume
```

### Run the pipeline
```bash
docker compose run --rm pipeline --input /app/videos/STEREO_01.mp4 --waste-threshold 5
docker compose run --rm pipeline --input /app/videos/STEREO_01.mp4 --waste-threshold 8 --resume
```
Results go to `./output/<video>/` on the host, and the dashboard shows them at
`http://<server>:8000`.

What the compose file sets up:

| Service | Purpose | Exposed |
|---|---|---|
| `ollama` | model server with GPU; models persist in volume `ollama-models` | internal only |
| `dashboard` | read-only UI over `./output` (mounted read-only) | port 8000 |
| `pipeline` (profile `run`) | one-off runs; talks to `http://ollama:11434` | – |

Notes:
* The container uses Debian's FFmpeg with **libx264** (no NVENC). Expect a few
  minutes of encoding per 2–3 minutes of 4K stereo video.
* Put videos in `./videos` and use `--input /app/videos/<file>`. A `--url` download
  would land inside the throwaway container.
* **Public access:** put the dashboard behind a reverse proxy with HTTPS and
  authentication (e.g. Caddy or nginx with basic auth). Don't publish port 8000
  directly to the internet.
* **Updates:** `git pull && docker compose up -d --build dashboard`.

### Without Docker (Linux)
Install Python 3.11+, FFmpeg and Ollama on the host, then follow Part 1 using
`source venv/bin/activate`. To keep the dashboard running, use a systemd service:
```ini
# /etc/systemd/system/stereo-hoi-dashboard.service
[Unit]
Description=Stereo HOI dashboard
After=network.target

[Service]
WorkingDirectory=/opt/stereo-QC
ExecStart=/opt/stereo-QC/venv/bin/python scripts/serve_ui.py --host 127.0.0.1 --port 8000
Restart=on-failure
User=stereo

[Install]
WantedBy=multi-user.target
```
`sudo systemctl enable --now stereo-hoi-dashboard`, then reverse-proxy
`127.0.0.1:8000` with authentication.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Cannot reach the OpenAI-compatible API at …` | Wrong `VLLM_BASE_URL` (it must end with `/v1`), wrong `VLLM_API_KEY` (HTTP 401), or the server is down. Try `curl.exe -H "Authorization: Bearer <key>" <VLLM_BASE_URL>/models`. |
| `Model 'Qwen/…' is not served at …` | `VLLM_MODEL` must exactly match an id from `<VLLM_BASE_URL>/models`. |
| HTTP 400 mentioning images / `limit_mm_per_prompt` | Start vLLM with `--limit-mm-per-prompt '{"image": 2}'`. |
| HTTP 400 mentioning `chat_template_kwargs` | The host doesn't support it. Set `inference.think: null` in a `--config` file. |
| `Cannot reach Ollama at …` | Ollama not running, or wrong `OLLAMA_BASE_URL`. Open `<URL>/api/version` in a browser. |
| `HTTP 502 … failed to establish a connection to the upstream web service at localhost:11434` | ngrok is up but Ollama isn't. Start Ollama. |
| `connection forcibly closed` mid-run | The tunnel dropped. Use `http://localhost:11434` when Ollama is on the same machine. Re-run with `--resume`. |
| Everything UNCERTAIN, empty descriptions | Model answers not parsed. Check `inference/raw_responses.jsonl` (the client already handles answers in `message.thinking`). |
| `Model 'qwen3-vl:4b' is not available` | `ollama pull qwen3-vl:4b` on the machine running Ollama. |
| `Quality checks failed: …` | Open `reports/qc_report.json` (or the dashboard's QC card). The processed video is kept for inspection, but treat it as not approved. |
| `h264_nvenc` errors | No NVIDIA GPU/driver. Drop `--config configs/nvenc.yaml`. |
| Want the frames for debugging | Set `sampling.keep_frames: true` in a `--config` file. |
| Dashboard empty | It reads `output/`. Run the pipeline first, or point `--output-dir` at the right folder. |
