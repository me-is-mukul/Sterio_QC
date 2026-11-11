# Dashboard

A deliberately minimal, read-only web page over the pipeline's output files.
It uses only the standard library and one static HTML file. It never starts,
stops or changes a run.

```powershell
python scripts/serve_ui.py                    # http://localhost:8000
python scripts/serve_ui.py --host 0.0.0.0     # others on your network: http://<your-ip>:8000
python scripts/serve_ui.py --port 8080 --output-dir output
```
Defaults come from `web.host` / `web.port` in `configs/default.yaml`.

## What it shows

* **Run picker:** every `output/<video>/` with a manifest. `?run=<name>` deep-links.
* **KPIs:** original vs processed duration, waste removed, HOI coverage
  before → after, frames analysed, mean inference latency.
* **HOI timeline:** one cell per sampled second, coloured by label. Removed intervals
  are hatched, and chapter markers come from the source file. Click a cell to inspect it.
* **Stereo pair:** left and right images, label, model description, evidence and
  validation problems. Use ← → to step through. "Play in processed" seeks the
  processed video to that moment (if retained).
  Frames are deleted after classification, so the viewer shows the model's
  description, and you see the moment itself via "play in processed". With
  `sampling.keep_frames: true` the images are shown too.
* **Quality checks:** the QC verdict and every check with its detail.
* **Waste verification:** each candidate section with its outcome (removed /
  partially removed / kept) and the times that protected it. On the timeline, dashed
  outlines mark candidates that verification trimmed or kept.
* **Processed video** player (HTTP range requests, so seeking works).
* **Before / after** table (= `reports/metrics.csv`) and **pipeline** stage status.
* **Live progress:** while a run is `running`, the page refreshes every 3 s, and
  the timeline fills in as inference writes `classifications.jsonl`.

## API

| Endpoint | Returns |
|---|---|
| `GET /api/runs` | `[{name, run_id, status, updated_at, headline}]` |
| `GET /api/runs/<name>` | overview: run, source, config, stages, video, chapters, sampling, inference, timeline, waste, processing, imu, metrics, summary, progress |
| `GET /api/runs/<name>/samples` | per pair: time, image paths, label, description, evidence, per-camera notes, latency |
| `GET /files/<name>/<path>` | workspace files (jpg/mp4/json/jsonl/csv/log/yaml/txt) with Range support |

Safety: names are validated, and file paths must resolve inside the workspace.
Only the listed extensions are served, and the source video and `.env` are never
exposed. The inference endpoint host is not included in API responses.
