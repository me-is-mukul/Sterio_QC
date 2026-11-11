# User Guide

How to use the Stereo HOI system from start to finish: start it, process a video,
watch it in the dashboard, collect the results and shut down.

| Want to… | Read |
|---|---|
| **Use the system** | this guide |
| Review and approve a result | [qc_playbook.md](docs/qc_playbook.md) |
| Install from scratch / deploy on a server | [guide.md](docs/guide.md) |

---

## 1. What it does

You give it a **stereo video** (left and right camera side by side, e.g. `STEREO_01.mp4`).
It gives back the **same video with the idle parts removed**, keeping every moment where
hands are working with objects, plus a report and a quality check.

```text
 your video ──► AI looks at 1 frame pair per second ──► finds long idle stretches (≥ N s)
            ──► double-checks them every 0.5 s ──► cuts only what is truly idle
            ──► quality checks ──► processed video + reports + dashboard
```

* **N** (the *waste threshold*) is the only setting you normally choose: the shortest
  idle stretch worth cutting. **5 seconds** is the default.
* Anything the AI is unsure about is **kept**. Frames are not stored, and the original
  video is never modified.

---

## 2. Before you start (checklist)

- [ ] The project is set up (`venv` exists). If not, see [guide.md](docs/guide.md).
- [ ] The video is in the project folder (e.g. `STEREO_01.mp4`).
- [ ] `.env` contains the model address:
  ```ini
  OLLAMA_BASE_URL=https://ecologist-eloquence-reliable.ngrok-free.dev   # or http://localhost:11434
  OLLAMA_MODEL=qwen3-vl:4b
  ```

---

## 3. Start up (3 terminals)

**Terminal 1: the AI model** (on the computer with the GPU)
1. Start **Ollama** from the Start menu (tray icon appears).
2. Only if you use the tunnel, run this and keep the window open:
   ```powershell
   ngrok http 11434 --host-header="localhost:11434" --url=https://ecologist-eloquence-reliable.ngrok-free.dev
   ```
   It should show `Session Status  online`.

**Terminal 2: the dashboard** (in the project folder)
```powershell
venv\Scripts\activate
python scripts/serve_ui.py
```
It prints `Dashboard: http://localhost:8000`. Open that in your browser and leave the
terminal running.

**Terminal 3: processing** (in the project folder)
```powershell
venv\Scripts\activate
```
You're ready.

---

## 4. Process a video

In terminal 3:
```powershell
python scripts/run_pipeline.py --input STEREO_01.mp4 --waste-threshold 5 --config configs/nvenc.yaml
```

What you'll see:
1. A few setup lines: the video is checked and measured.
2. `Sampled 150 stereo pairs` → the frames are taken.
3. `Connecting to inference provider … Model qwen3-vl:4b available` → the AI is reachable.
4. One line per second of video, e.g.
   `[37/150] t=  36.0s RELEVANT    4.8s | Hands manipulating tools near engine`.
   This is the slow part, about 5 s per line.
5. `Verifying candidate 1/2 …` → idle stretches are double-checked.
6. `Encoding … with h264_nvenc` → the new video is written.
7. `QC PASS …` lines → the quality checks.
8. A summary block ending with `Status : completed`.

**Rough timing** for a 2½-minute video: 15–20 minutes, mostly AI time.
Watch it live in the dashboard while it runs.

> If the run stops (network drop, laptop sleep, Ctrl+C), run **the same command with
> `--resume`** added. Finished work is reused.
>
> Run **one command per video at a time**. A second run on the same video is refused
> with `Another run is already processing this video`.

---

## 5. The dashboard

Open http://localhost:8000. Pick the video at the top left. While a run is active the page
refreshes on its own every 3 seconds.

| Panel | What it tells you |
|---|---|
| **Top cards** | Original vs processed length, how much was removed, HOI share before → after, progress (`frames analysed`), AI speed |
| **HOI timeline** | One block per second. 🟩 relevant · 🟥 idle · 🟨 unsure · ⬜ no answer. **Hatched** = cut out. **Dashed box** = an idle stretch that the 0.5 s double-check saved or shortened. The labels underneath mark the chapters of the source video. |
| **Stereo pair** | Click any block (or use ← →) to read what the AI saw at that second. **▶ play in processed** jumps the video player to that moment. |
| **Processed video** | The result. Play it, and seek around the cut points. |
| **Quality checks** | The automatic verdict (PASS / WARN / FAIL) and each check in plain words |
| **Waste verification** | Every idle stretch considered for cutting: removed, partially removed or kept, and why |
| **Before / after** | The numbers comparison (same as `reports/metrics.csv`) |
| **Pipeline** | Which step is running or done, and how long each took |

**To show someone else:**
* On the same Wi-Fi: start the dashboard with `python scripts/serve_ui.py --host 0.0.0.0`
  and share `http://<your-IP>:8000` (`ipconfig` shows your IP).
* Over the internet: `ngrok http 8000` (the dashboard has no login, so stop it after the demo).

---

## 6. Your results

Everything is in `output\<video name>\`. What to hand over:

| File | What it is |
|---|---|
| `processed\STEREO_01_processed.mp4` | **The cleaned video** |
| `reports\summary.json` | One-line headline + key numbers |
| `reports\qc_report.json` | Quality check results |
| `reports\metrics.csv` | Before/after table (opens in Excel) |
| `processed\processing_timeline.json` | Which original time ranges are in the new video, and where |

The rest (`inference\`, `segmentation\`, `metadata\`, `reports\manifest.json`,
`reports\processing.log`) is the evidence trail: what the AI said each second, why each
cut was made, settings and versions. Keep it with the video.

---

## 7. Common tasks

| Task | Command (terminal 3) |
|---|---|
| Process another video | `python scripts/run_pipeline.py --input OTHER.mp4 --waste-threshold 5 --config configs/nvenc.yaml` |
| Video is online | `python scripts/run_pipeline.py --url "https://…/video.mp4" --waste-threshold 5` (downloads into the project folder first) |
| Cut **less** (keep more) | re-run with a **larger N**, e.g. `--waste-threshold 10 --resume` |
| Cut **more** | re-run with a **smaller N**, e.g. `--waste-threshold 3 --resume` |
| Continue after a stop / retry failed AI answers | add `--resume` to the same command |
| Start completely fresh | run without `--resume` (the previous run is archived in `reports\history\`; AI answers are still reused) |
| No NVIDIA GPU on this computer | leave out `--config configs/nvenc.yaml` |
| Less console output | add `--log-level WARNING` |
| See all options | `python scripts/run_pipeline.py --help` |

With `--resume`, changing N only redoes the cutting and checks (a minute or two). The AI
answers are reused.

---

## 8. Settings you may want to change

Don't edit the files in `configs\`. Create your own file, e.g. `my_settings.yaml`, with
only the values you want to change, and pass it with `--config my_settings.yaml`:

```yaml
verification:
  interval_seconds: 0.25     # double-check idle stretches more finely (slower, safer)
  guard_seconds: 1.0         # keep 1 s extra around anything relevant
sampling:
  keep_frames: true          # keep the frame images (to look at them in the dashboard)
qc:
  min_alignment_psnr_db: 35  # stricter frame-match check
```
(You can only pass one `--config` file. To also use GPU encoding, copy the
`segmentation:` block from `configs\nvenc.yaml` into your file.)

| Setting | Default | Meaning |
|---|---|---|
| `hoi.waste_threshold_seconds` | 5 | Same as `--waste-threshold` |
| `sampling.interval_seconds` | 1 | How often the AI looks (seconds) |
| `verification.interval_seconds` | 0.5 | Double-check spacing inside idle stretches |
| `verification.guard_seconds` | 0.5 | Safety margin kept around anything relevant |
| `sampling.keep_frames` | false | Keep frame images instead of deleting them |
| `inference.failure_policy` | uncertain | What a failed AI answer becomes (never "idle") |
| `qc.fail_run_on_failure` | true | Mark the run failed if any quality check fails |

The full list is in [pipeline.md](docs/pipeline.md).

---

## 9. Review and approve

Before a processed video is used, someone should review it with the
**[QC playbook](docs/qc_playbook.md)**. It takes 10–15 minutes and has a checklist and a
record to fill in.

---

## 10. Shut down

1. Terminal 3: wait for `Status : completed` (or press Ctrl+C; you can `--resume` later).
2. Terminal 2 (dashboard): Ctrl+C.
3. Terminal 1 (ngrok): Ctrl+C.
4. Ollama: right-click the tray icon → **Quit Ollama**.

---

## 11. Quick fixes

| Message | Fix |
|---|---|
| `Cannot reach Ollama at …` | Ollama or ngrok isn't running. Do step 3, then re-run with `--resume`. |
| `HTTP 502 … localhost:11434` | ngrok is running but Ollama isn't. Start Ollama. |
| `connection forcibly closed` | The tunnel dropped. Restart ngrok, then `--resume`. If Ollama is on this computer, use `OLLAMA_BASE_URL=http://localhost:11434`. |
| `Model 'qwen3-vl:4b' is not available` | On the GPU computer: `ollama pull qwen3-vl:4b` |
| `Quality checks failed: …` | Don't use the video. Open the Quality checks panel and follow the [QC playbook](docs/qc_playbook.md#3-problems-and-what-to-do). |
| `h264_nvenc` error | No NVIDIA GPU. Leave out `--config configs/nvenc.yaml`. |
| `Another run is already processing this video` | Only one run per video at a time. Wait for the other one (watch the dashboard), or stop it with Ctrl+C in its terminal. |
| `Cannot start the dashboard … Is another dashboard already running?` | One is already running. Just open http://localhost:8000. |
| Dashboard shows "No runs yet" | No video has been processed yet (section 4). |
| `--input: file not found` | Check the file name, or give the full path in quotes. |

More: [guide.md → Troubleshooting](docs/guide.md#troubleshooting).
