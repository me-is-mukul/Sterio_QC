# Methodology

How every number in the QC checklist is produced, which tools are used, and the known limits.
All thresholds referred to here live in [`config/hand_qc.yaml`](../config/hand_qc.yaml).

---

## 1. Processing overview

```text
clip list (CSV) ──► source resolution ──► download + integrity check (size, MD5)
      │
      ▼
ffprobe metadata ──► ONE FFmpeg decode of every frame ──┬─► every Nth frame, scaled ──► MediaPipe hands
                                                        │                          └─► picture metrics
                                                        ├─► 2 full-resolution frames/s ──► FFmpeg blockdetect (info)
                                                        └─► decoder error messages
second decode: 2 full-resolution grey frames/s ──► visible block edges (graded blockiness)
      ▼
raw analysis (results/videos/<clip>/analysis/)  ──►  scoring with the thresholds  ──►  checklist + reports
```

The expensive analysis is stored raw and threshold-free. Scoring is a separate, deterministic
step, so changing a threshold and running `python run_hand_qc.py --score-only` updates every result
in seconds without decoding any video again.

## 2. Inputs

Each clip list is a CSV passed with `--csv` (format: see README). At the start of a run it is copied
to `results/<name>/source_list.csv` and resolved into `results/<name>/clip_manifest.csv` by matching
each row against the public bucket listings. `https://storage.googleapis.com/…` and `gs://…` links
are both accepted. Resolution runs again only when the CSV changes, or with `--refresh-sources`.
`Dataset` decides the camera layout: `Mono` = full frame, `Stereo` = side by side, `Six-Cam` = 3×2
grid.

| Source entry | Resolution |
|---|---|
| Complete URL | used as given |
| URL with a truncated file name (`…/sampl.mp4`) | the only `.mp4` in that session folder |
| URL with a missing date folder | the only session folder with that path |
| Bare filename (`_GX010171.mp4`) | the only file named `<YYYY-MM-DD>_GX010171.mp4` in `mono samples trimmed/electronics/` of bucket `dd_samples` |
| No URL / truncated beyond recovery | `SOURCE MISSING` (listed in the checklist, not analysed) |

A row is resolved only when exactly one file matches. Each download is checked against the bucket's
size, and against its MD5 when the bucket publishes one.

## 3. Frame sampling and camera views

* **Rotation**: FFmpeg applies the container's rotation metadata, so frames are analysed the way a
  player shows them.
* **Frame stride** (`analysis.stride_bands`): every 4th frame at ~30 fps, every 8th at ~60 fps,
  every 20th at ~120 fps. That gives 6–7.5 analysed frames per second, which is enough to measure gaps
  of 0.5 s and grasps of 0.3 s. Each analysed frame stands for one frame interval (`stride / fps`
  seconds) in all durations and percentages.
* **Views**: *mono* = the whole frame. *Stereo* (side by side) = left and right eye, each analysed
  on its own. *Six-cam* (3 × 2 grid) = six tiles, numbered 1–3 on the top row and 4–6 on the bottom
  row, left to right. Each view is downscaled to at most 1280 px wide for analysis. Landmark
  coordinates are always reported in pixels of the **original** view.
* **Combining views** (stereo / six-cam): a hand is present in a frame when it is present in *any*
  view. The same applies to grasping. Per-view percentages are also reported (`summary.json`, HTML
  report).

## 4. Hand tracking

**Model**: Google MediaPipe Hand Landmarker (`models/hand_landmarker.task`, float16 v1, SHA-256 in
`handqc/cli.py`) in VIDEO mode, which tracks hands from frame to frame. It runs as one tracker per
camera view and looks for up to 4 hands, so other people's hands are also seen. Confidence limits are
in `analysis.mediapipe`.

**21 landmarks**: 0 wrist; 1–4 thumb CMC, MCP, IP, tip; 5–8 index MCP, PIP, DIP, tip; 9–12 middle;
13–16 ring; 17–20 pinky. Each is reported as `x, y` in pixels of the original camera view, with the
origin at the top-left and y pointing down.

**Left / right**: this is MediaPipe's handedness label. Older MediaPipe documentation says labels
assume mirrored (selfie) input. We checked this on these clips: with MediaPipe 1.2.3 the labels are
correct for non-mirrored egocentric video. For example, the wearer's watch-wearing left hand is
labelled *Left*, and the hand holding the comb or razor is labelled *Right*. So the labels are used as
they are (`swap_handedness: false`).

### 4.1 Whose hands: the camera wearer's or someone else's?

Hand metrics count **only the wearer's hands**. These rules apply per view and per frame:

1. **Duplicates**: when two detections share more than `duplicate_overlap` of the smaller box, the
   one with the lower confidence is dropped.
2. **Someone else's hand** when either of these holds:
   * its **hand size** is below `min_hand_size`, so it is far away. Hand size is the diagonal of the
     box around the 21 landmarks divided by the view height. It is used rather than palm length
     because a hand seen edge-on or pointing at the lens keeps its box size, while its
     wrist-to-knuckle length collapses;
   * it points down the frame (`pointing_down` > `max_pointing_down`) and is smaller than
     `large_hand_size`. Hands reaching towards the camera belong to people facing the wearer, and
     only the wearer's own hands come closer than `large_hand_size`.
3. At most two wearer hands per view: if there are more candidates, the two largest (by hand size) are kept.
4. If both wearer hands get the same label, the one further left in the image becomes *Left*.
5. **Upside-down footage**: if fewer than `min_upright_pct` of the near hands point up the frame,
   the clip is treated as inverted. Rules 2 and 4 are then mirrored, and the orientation test fails.

Six-cam tiles are mounted at fixed rotations, so the direction rules (2, second case; 4; 5) are not
used there. Only the size and duplicate rules apply.

Calibration: in the barbering clips, the customer's hands measured 0.054–0.086 and the wearer's
outstretched hands 0.07–0.17. Foreshortened wearer hands on the six-cam rig measured 0.17–0.28
(their palm length read about 0.01). That gives `min_hand_size` 0.08 and `large_hand_size` 0.175.
The orientation test uses only *near hands*: the largest quarter of the clip's hand detections, and
at least 0.12. The wearer's own hands are the closest, while other people's hands face the camera
and point down even in upright footage.

Other people's hands are reported as `other_hands_pct`, for information.

### 4.2 Grasp and hand–object interaction (HOI)

The grasp check uses MediaPipe's 3D hand model ("world landmarks", in metres), so it does not depend
on the viewing angle. For each wearer hand in each frame:

* **Finger curl**: for each of the index, middle, ring and pinky fingers, the flexion angles at the
  MCP, PIP and DIP joints are summed (0° = straight). `curl_mean_deg` is the mean over the 4 fingers.
* **Pinch**: the distance from the thumb tip to the index or middle fingertip, divided by the palm
  length.
* **Grasp pose** when `curl_mean_deg ≥ power_curl_deg` (power grasp) **or**
  `pinch ≤ pinch_ratio` (precision grasp).

A **grasp event** is a grasp pose held for at least `min_duration_s`; dropouts up to `max_gap_s` are
bridged. Events are counted per hand.
**HOI_pct** is the share of analysed frames that fall inside a grasp event of either hand.

The thresholds were calibrated by checking sampled detections from barbering and factory clips by
eye (`power_curl_deg 120`, `pinch_ratio 0.40`). Hands holding tools, stickers, boxes or combs are
detected reliably. Open hands are correctly excluded, and so is flat-hand contact such as pressing,
pushing or steadying (see Limitations).

## 5. Metric definitions

| Metric | Definition |
|---|---|
| `either_hand_pct` | % of analysed frames with at least one wearer hand |
| `both_hands_pct` | % of analysed frames with both wearer hands |
| `left_hand_pct`, `right_hand_pct` | % of analysed frames with the wearer's left / right hand |
| `untracked_pct` | % of analysed frames with no wearer hand (= 100 − `either_hand_pct`) |
| `HOI_pct` | % of analysed frames inside a grasp event (section 4.2) |
| `tracking_losses` | times a hand that had been tracked for ≥ `min_tracked_s` then disappears for ≥ `min_lost_s`, counted per hand (left + right); also per minute |
| `grasp_count` | number of grasp events (left + right) |
| `grasp_duration` | total and mean seconds of grasp events |
| `longest_hand_gap` | longest continuous time with no wearer hand, in seconds |
| `other_hands_pct` | % of analysed frames in which another person's hand is visible |
| Duration, FPS, codec | ffprobe: stream duration, average frame rate, codec name and profile |
| Bitrate | video stream bitrate. The required value is `min_at_1080p` scaled by pixel count relative to 1920×1080. |
| Decode errors | every error message from the full FFmpeg decode, plus one error if the decode was incomplete (missing frames or a non-zero exit code) |
| Luminance | mean BT.709 luma (0–255) of the analysed frames. Frames whose mean is below `dark_luma` or above `bright_luma` count as badly exposed. |
| Sharpness | variance of the Laplacian of the luma at analysis resolution. The test is the % of frames at or above `sharp_laplacian_var`. |
| Blockiness | visible block edges on full-resolution luma (`blockiness_fps` frames per second). A pixel boundary is *flat* when the luma differences on both sides of it are ≤ 1 grey level, and a flat boundary is a *visible step* when its own difference is ≥ 3 grey levels. The step rate per column (and per row) is accumulated over the clip. The strongest periodic grid (8, 16, 24, 32, 48 or 64 px, any offset, chosen on medians) is then compared with the rest of the picture. Value = step rate on the grid − step rate everywhere, in percentage points, averaged over columns and rows (0 = no visible block edges). FFmpeg `blockdetect` is also reported, for information only. |
| Colour cast | mean CIELAB a\*/b\* of the near-neutral pixels (chroma < 15, mid lightness), as a distance from neutral |
| Orientation | rotation metadata must be a multiple of 90° (FFmpeg applies it, as players do), and ≥ `min_upright_pct` of the near hands (largest quarter of the clip's hands) must point up the frame, i.e. wrist below knuckles. Not applied to six-cam tiles. |

### Calibration notes

* **Blockiness**: two measures were tried first and rejected. FFmpeg's `blockdetect` scored the
  visibly blocked 3840×2880 electronics clips (16–64 px blocks) as clean (2.8, the same as clean
  footage). A plain edge-strength ratio could not tell visible blocks from the faint, invisible
  64 px HEVC tile boundaries in smooth high-bitrate scenes. Counting visible steps in flat surroundings
  separates the two. On the 50 clips the values form two groups. Every GoPro, stereo and six-cam clip
  scores 0.49–2.54, with no block edges visible at 2× magnification. The low-bitrate electronics
  clips score 3.45–8.12, with block edges visible even when scaled down (checked at 3.45, 3.67, 5.3
  and 7.9). Re-encodes of a reference clip: x264 CRF 23 → 0.6; CRF 42 / 51 → 3.9 / 5.9; MPEG-2 with
  clear blocking → 8–11. The limit of 3.0 sits in the gap between the groups.
* **Colour cast**: the whole-frame mean colour gets confused by content. For example, blue ESD mats
  read 25–28 Lab units with no cast at all. The neutral-pixel measure reads 0.6–6 on the sample
  footage, where the highest values come from coloured shadows in green/blue factory scenes. Visible
  artificial casts (±12–15% channel gain) read 9–10 in scenes with grey or white surfaces. The limit
  of 8 avoids false alarms on the natural footage.

* **Orientation**: across the 50 clips the share of near hands pointing up falls into two groups.
  Upright footage scores 59.6–100% (the lowest is a bench task where the arms lie sideways). Inverted
  footage scores 0.1–11.6%, and every one was checked by eye: two clips whose −180° rotation metadata
  turns correctly recorded video upside down, and three where the hands enter from the top of the
  frame. The limit of 35% sits in the gap. Using the largest quarter of hands matters: with all near
  hands, an upright clip with co-workers reaching across the table read 39%.

## 6. Overall verdict

A clip **PASSES** when every graded test passes. Tests with `min`/`max` set to `null` are reported
only. `SOURCE MISSING`: the client list has no usable URL. `ERROR`: download, decode or analysis
failed (the reason is in the checklist).

## 7. Reproducibility

* Pinned software (`requirements.txt`), with the hand model's checksum verified at start-up.
* Each clip's `analysis/analysis.json` records the probe, the analysis settings fingerprint, the
  software versions and the decode summary. Analysis is redone automatically only when the settings,
  model or source file change.
* `results/config_used.yaml` and `results/run_info.json` record the exact settings and versions of
  the last run.
* There are no network services or AI language models involved. Everything runs locally from the
  scripts in this folder.

## 8. Limitations

* **Wearer vs other hands** uses geometric rules. It can misjudge an outstretched wearer hand
  pointing down (e.g. reaching over a customer's head), or a large hand of a person standing very
  close.
* **HOI_pct is grasp-based.** It cannot see objects, so flat-hand contact is not counted. Treat it
  as a conservative lower bound.
* **MediaPipe recall** drops for strongly occluded hands, gloved hands, hands covered in clay or
  paint, motion blur and fisheye edges. Those frames count as "no hand", which lowers the presence
  metrics. A low `either_hand_pct` therefore means "no trackable hand", which is not always the
  same as "no hand in view" (e.g. the pottery clip in `stereo_sixcam_sample_clips`).
* **Sharpness** depends on scene content (low-texture scenes score lower). The limit applies at
  analysis resolution.
* **Colour cast** cannot be measured when frames contain almost no grey or white surfaces. The test
  then reports N/A.
