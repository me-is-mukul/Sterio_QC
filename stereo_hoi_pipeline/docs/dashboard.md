# Dashboard

A deliberately minimal, read-only web page over the pipeline's output files.
It uses only the standard library and one static HTML file. It never starts,
stops or changes a run.

```powershell
python scripts/serve_ui.py                    # http://localhost:8000
python scripts/serve_ui.py --host 0.0.0.0     # others on your network: http://<your-ip>:8000
python scripts/serve_ui.py --port 8080 --output-dir outputs
```
Defaults come from `web.host` / `web.port` in `configs/default.yaml`.

## What it shows

* **Run picker:** every `outputs/<video>/` with a manifest. `?run=<name>` deep-links, and
  `&t=<seconds>` opens that moment.
* **KPIs:** original vs processed duration, waste removed, HOI coverage
  before → after, frames analysed, detector and time per pair.
* **Hand KPIs** (MediaPipe detector), original → processed: hands in frame, both hands,
  hand-object interaction (grasping), longest hand gap, tracking losses per minute, other
  people's hands.
* **HOI timeline:** one cell per sampled second, coloured by label. Removed intervals
  are hatched, and chapter markers come from the source file. Click a cell to inspect it.
  Below it, **Left hand / Right hand** lanes show per sample whether the wearer's hand was
  grasping, visible or not detected.
* **Stereo pair:** the analysed frame (both eyes), rendered on demand from the source video,
  so no frames are stored. The 21 landmarks of every detected hand are drawn on top: blue =
  wearer left, yellow = wearer right, red = other person; thick lines = grasping. Below are
  the label, description, per-eye notes and validation problems. Use ← → to step through.
  "Play in processed" seeks the processed video to that moment (if retained). If the
  source video has been moved away, a note is shown instead.
* **Quality checks:** the QC verdict and every check with its detail.
* **Waste verification:** each candidate section with its outcome (removed /
  partially removed / kept) and the times that protected it. On the timeline, dashed
  outlines mark candidates that verification trimmed or kept.
* **Processed video** player (HTTP range requests, so seeking works).
* **Before / after** table (= `reports/metrics.csv`), **hand metrics** table
  (= `reports/hand_metrics.json`, original vs processed) and **pipeline** stage status.
* **Live progress:** while a run is `running`, the page refreshes every 3 s, and
  the timeline fills in as inference writes `classifications.jsonl`.

## API

| Endpoint | Returns |
|---|---|
| `GET /api/runs` | `[{name, run_id, status, updated_at, headline}]` |
| `GET /api/runs/<name>` | overview: run, source, config, stages, video, chapters, sampling, inference, timeline, waste, processing, imu, metrics, summary, progress |
| `GET /api/runs/<name>/samples` | per pair: time, image paths, label, description, evidence, per-camera notes, latency, hand summary and landmarks |
| `GET /api/runs/<name>/frame?t=<s>` | the source frame at `t` seconds (the analysed frame) as JPEG, max 1920 px wide |
| `GET /files/<name>/<path>` | workspace files (jpg/mp4/json/jsonl/csv/log/yaml/txt) with Range support |

Safety: names are validated, and file paths must resolve inside the workspace.
Only the listed extensions are served, and `.env` is never exposed. The source
video itself is never served, only single frames through `/frame`. The inference endpoint host is not included in API responses.
