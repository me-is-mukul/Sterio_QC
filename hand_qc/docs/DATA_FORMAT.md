# Output files

Each clip list gets its own folder, `outputs/<name>/`, where `<name>` is the CSV file name (or the
`--name` value).

```text
outputs/<name>/
├── source_list.csv          the clip list as received
├── qc_checklist.xlsx        the checklist: one row per clip, every metric, PASS/FAIL per test (+ Summary,
│                            Thresholds and Definitions sheets)
├── qc_checklist.csv         the same checklist as plain CSV
├── qc_report.html           interactive report (open in any browser, works offline)
├── clip_manifest.csv        the clip list with resolved download URLs, sizes and MD5s
├── config_used.yaml         the exact settings used
├── run_info.json            time, versions, clips with errors
├── logs/                    run logs
└── videos/<clip_id>/
    ├── summary.json         all metrics and tests for the clip
    ├── landmarks.csv        21 hand landmarks per detected hand per analysed frame
    ├── hand_timeline.csv    per analysed frame: which hands are present / grasping
    ├── preview.jpg          an analysed frame with the detected hands drawn
    └── analysis/            raw measurements (used for re-scoring; see below)
```

## landmarks.csv

One row per detected hand per analysed frame (all views). A hand that MediaPipe detected twice in
the same frame is listed once.

| Column | Meaning |
|---|---|
| `clip_id` | clip identifier (`<no>_<dataset>_<task>`) |
| `frame` | source frame number (0-based, after rotation metadata is applied) |
| `time_s` | `frame / fps`, seconds from the start of the clip |
| `view` | `main` (mono), `left_eye` / `right_eye` (stereo), `tile1`…`tile6` (six-cam, rows left→right, top row first) |
| `hand_index` | detection index within the view and frame |
| `role` | `wearer` (the camera wearer's hand) or `other` (another person's hand) |
| `hand` | `Left` / `Right` |
| `handedness_score` | MediaPipe handedness confidence (0–1) |
| `grasp` | 1 = grasp pose, 0 = no grasp (wearer hands only; empty for `other`) |
| `wrist_x`, `wrist_y`, `thumb_cmc_x`, … `pinky_tip_y` | the 21 landmarks: x, y in **pixels of the original camera view**, origin top-left, y down |

Landmark order: `wrist`, `thumb_cmc`, `thumb_mcp`, `thumb_ip`, `thumb_tip`, `index_mcp`, `index_pip`,
`index_dip`, `index_tip`, `middle_mcp`, `middle_pip`, `middle_dip`, `middle_tip`, `ring_mcp`,
`ring_pip`, `ring_dip`, `ring_tip`, `pinky_mcp`, `pinky_pip`, `pinky_dip`, `pinky_tip`.
These are MediaPipe landmarks 0–20.

For stereo and six-cam clips, coordinates are relative to the eye or tile named in `view`. To get
full-frame coordinates, add the view's offset: the right eye starts at x = half the frame width, and
a tile starts at (column × tile width, row × tile height).

## hand_timeline.csv

One row per analysed frame (views combined). Values are 0/1 except `other_hands`.

| Column | Meaning |
|---|---|
| `frame`, `time_s` | as above |
| `left`, `right`, `either`, `both` | wearer hand presence |
| `grasp_left`, `grasp_right` | inside a grasp event of that hand |
| `hoi` | inside a grasp event of either hand (counts towards HOI_pct) |
| `other_hands` | number of other people's hands visible |

## summary.json

* `overall`: `PASS` / `FAIL`; `reasons`: the failed tests in words.
* `tests[]`: `id`, `group` (Technical / Hands), `name`, `value`, `unit`, `threshold`,
  `status` (PASS / FAIL / INFO / N/A), `detail`.
* `technical`, `hands`: every metric. `per_view`: presence per eye or tile.
  `tracking_loss_events`, `grasp_events`: the individual events with times.
* `analysis`, `probe`: analysis settings, software versions, decode summary, ffprobe metadata.

## analysis/ (raw)

| File | Content |
|---|---|
| `analysis.json` | probe, settings fingerprint, frame counts, software versions |
| `detections.csv.gz` | every detection with landmarks and the features used by the rules (palm size, pointing direction, finger curl angles, pinch ratios, bounding box) |
| `frames.csv.gz` | per analysed frame and view: luma mean/percentiles, sharpness, Lab colour statistics, neutral-pixel colour |
| `grid_blockiness.json` | the graded blockiness measurement: value, per-direction grid period/offset and step rates, frames used |
| `blockiness.csv` | FFmpeg blockdetect value per full-resolution frame (information only) |
| `decode_errors.txt` | FFmpeg decoder messages (empty = clean decode) |
| `preview_*.jpg` | analysed frames used for `preview.jpg` |
