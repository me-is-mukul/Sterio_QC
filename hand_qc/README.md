# Egocentric Video QC: Hand & Technical Checklist

A reproducible quality check for egocentric video clips listed in a CSV file. Every clip gets:

* **Technical checks**: duration, frame rate, bitrate, codec, decode errors, luminance, sharpness,
  blockiness, colour cast, orientation.
* **Hand checks**: either / both / left / right hand presence, HOI (grasping), tracking losses,
  untracked time, grasp count and duration, longest hand gap, and other people's hands.
* **21 hand landmarks** (x, y pixel coordinates) for every detected hand in every analysed frame.
* A **PASS / FAIL** verdict per test and per clip, with the reason for every failure.

Everything runs locally with open-source tools (Google MediaPipe Hand Landmarker and FFmpeg). There
are no cloud services and no AI language models, and the same inputs and settings always give the
same results.

Each clip list (CSV) gets its own results folder, `results/<name>/`, where `<name>` is the CSV file
name.

> **Quick start** (from the repository root, after `pip install -r hand_qc/requirements.txt`):
> `cd hand_qc` then `python run_hand_qc.py --csv ../egocentric_sample_clips.csv --workers 4`.
> `results/` is not stored in the git repository; it is created by the first run.

---

## Results at a glance

### `egocentric_sample_clips` (68 clips in the list)

| | Clips |
|---|---|
| Analysed | 50 |
| **PASS** (every graded test) | **30 (60%)** |
| FAIL | 20 |
| SOURCE MISSING | 18: the list has no usable URL (13 mono rows "truncated after GX0", 5 stereo rows) |

Reasons for failing: HOI below 55% (12 clips); hands in frame below 65% / untracked above 35%
(9); orientation (5: #16 and #26 have −180° rotation metadata that makes the video play upside
down; in #2, #32 and #46 the hands enter from the top of the frame); longest hand gap above 20 s
(5); tracking losses (1).

### `stereo_sixcam_sample_clips` (11 clips in the list)

| | Clips |
|---|---|
| Analysed | 11 (9 stereo, 2 six-cam) |
| **PASS** | **7 (64%)** |
| FAIL | 4 |

Failing: #4 (OpenRobot) is just under the limits (hands in 63% of frames, HOI 46%); #6 (OpenRobot,
pottery) has hands in only 15% of frames because the clay-covered hands are rarely detected, even
though they are in view; #8 (GI Labs) has hands in 42% of frames; #10 (BitRobot, mopping) has hands
in 52% of frames. Six of these videos are also in `egocentric_sample_clips` and give identical
results in both lists.

---

## Open the results

For each clip list, in `results/<name>/`:

| File | What it is |
|---|---|
| **`qc_checklist.xlsx`** | The checklist: one row per clip, every metric, PASS/FAIL colour-coded. Sheets: *Checklist*, *Summary* (pass rate per test), *Thresholds*, *Definitions*. |
| `qc_checklist.csv` | The same checklist as CSV |
| **`qc_report.html`** | Interactive report: filter and search clips, click a clip for every test with its threshold and an annotated frame. Opens in any browser, offline. `qc_report.html#<clip_id>` opens one clip. |
| `videos/<clip>/landmarks.csv` | 21 landmarks (x, y px) for every detected hand, with role (wearer / other), Left/Right and grasp flag |
| `videos/<clip>/hand_timeline.csv` | Per analysed frame: hands present, grasping, HOI |
| `videos/<clip>/summary.json` | Every metric and test for the clip (machine-readable) |
| `source_list.csv` / `clip_manifest.csv` | The clip list as received, and with the resolved download URLs |

File formats: [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md). How each number is computed, and its
limits: [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

---

## Folder contents

```text
hand_qc/
├── README.md                  this file
├── run_hand_qc.py             the one command to run
├── requirements.txt           Python packages (pinned)
├── config/hand_qc.yaml        ALL settings and PASS/FAIL thresholds
├── models/hand_landmarker.task       MediaPipe hand model (Apache-2.0)
├── handqc/                    the Python package (sources, analysis, scoring, reports)
├── docs/                      methodology and data formats
└── results/
    ├── egocentric_sample_clips/       one folder per clip list (see above)
    └── stereo_sixcam_sample_clips/
```

---

## Clip list (CSV) format

One row per clip, with a header row:

| Column | Content |
|---|---|
| `Dataset` | `Mono`, `Stereo` (left/right side by side) or `Six-Cam` (3×2 grid). This decides how the frame is split into camera views. |
| `Category`, `Task`, `Camera` | Free text, copied to the checklist (`Task` may be empty) |
| `URL_or_File` | `https://storage.googleapis.com/<bucket>/<path>`, `gs://<bucket>/<path>`, or empty |
| `Notes` | Free text |

Links to public Google Cloud Storage buckets are checked against the bucket listing. Truncated file
names or missing date folders are repaired only when exactly one file matches. Rows that can't be
resolved are listed as `SOURCE MISSING` and are not analysed.

---

## Run it yourself

### 1. Install (once)

* **Python 3.11–3.13**
* **FFmpeg 6.0 or newer** (`ffmpeg` and `ffprobe` on PATH). On Windows:
  `winget install Gyan.FFmpeg`. On Ubuntu: `sudo apt install ffmpeg`. On macOS: `brew install ffmpeg`.

```bash
cd hand_qc
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Run

```bash
python run_hand_qc.py --csv path/to/my_clips.csv
```

This downloads the clips into `../hand_qc_cache/` (outside this folder; each file is verified
against its size and MD5, and shared between clip lists), analyses them, and writes
`results/my_clips/`. On an 8-core laptop with `--workers 4`, about 2 hours of video took about
1 hour. Finished work is kept: if the run stops, run the same command again and it resumes. Every
run re-scores all analysed clips of that list with the current settings, so a checklist never mixes
thresholds.

| Command | Use |
|---|---|
| `python run_hand_qc.py --csv LIST.csv` | full run (only new or changed clips are analysed again) |
| `... --name client_batch_3` | use this results folder name instead of the CSV file name |
| `... --score-only` | re-score after changing thresholds (seconds, no video processing) |
| `... --only 1,5,9` | download/analyse only these clips (row numbers or clip ids); the checklist is always rebuilt for every clip |
| `... --workers 4` | analyse 4 clips in parallel (default 2) |
| `... --reanalyse` | force the video analysis to run again |
| `... --refresh-sources` | resolve the download links again (normally done only when the CSV changes) |
| `... --config my.yaml` | use your own settings file (see below) |

### 3. Change thresholds

All thresholds are in [`config/hand_qc.yaml`](config/hand_qc.yaml) under `tests:`. Set
`min`/`max` to `null` to report a metric without grading it. For example, to require both hands in
40% of frames and allow 8 tracking losses per minute, create `my.yaml`:

```yaml
tests:
  both_hands_pct: {min: 40}
  tracking_losses_per_min: {max: 8}
```

and run `python run_hand_qc.py --csv LIST.csv --config my.yaml --score-only`. The checklist is rebuilt in
seconds. Settings under `analysis:` change the video processing itself and trigger a full
re-analysis.

---

## Default PASS rules

Policy: **hands in frame and hand-object interaction decide the verdict**; the technical checks only
reject clearly unusable footage. Tuned so that about 60% of the sample clips pass.

| Test | Rule |
|---|---|
| **either_hand_pct** | **≥ 65%** |
| **untracked_pct** | **≤ 35%** |
| **longest_hand_gap** | **≤ 20 s** |
| **HOI_pct** | **≥ 55%** |
| tracking_losses | ≤ 15 per minute |
| Duration | ≥ 30 s |
| Frame rate | ≥ 24 fps (23.976 accepted) |
| Bitrate | ≥ 1 Mbit/s at 1920×1080, scaled by pixel count (e.g. 4 Mbit/s for 3840×2160) |
| Codec | H.264 or HEVC |
| Decode errors | 0 (full decode of every frame) |
| Luminance | mean 30–230 (of 255), and ≤ 25% of frames too dark or too bright |
| Sharpness | ≥ 60% of frames sharp (Laplacian variance ≥ 60) |
| Blockiness | visible block edges ≤ 8.5 percentage points above the rest of the picture (0 = none) |
| Colour cast | grey/white surfaces within 12 Lab units of neutral |
| Orientation | rotation metadata a multiple of 90°, and ≥ 35% of the nearest hands pointing up the frame (upright footage: 60–100%, upside-down: 0–12%) |
| both / left / right_hand_pct, grasp count / duration, other hands | reported only |

Hand metrics count **only the camera wearer's hands**; for stereo and six-cam clips, all camera views
are combined. Other people's hands are reported separately as `other_hands_pct`.
