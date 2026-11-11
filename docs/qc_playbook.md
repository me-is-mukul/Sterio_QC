# QC Playbook: Reviewing a Processed Video

**For:** the person who checks and approves the pipeline's output.
**You need:** the dashboard link (e.g. `http://localhost:8000`) and a video player
(VLC or similar) with access to the original video file.
**Time:** about 10–15 minutes per 2–3 minute video.

---

## 1. What the system did (in one minute)

The input is a stereo video (left and right camera side by side). The pipeline:

1. looked at **one stereo frame pair per second** and asked an AI model:
   *are hands doing something with an object?* → **Relevant** / **Waste** / **Uncertain**;
2. took every stretch of **Waste lasting at least N seconds** (N is shown on the
   dashboard) as a *candidate* to cut;
3. **re-checked each candidate every 0.5 s**. Anything not clearly waste is protected
   (±0.5 s), and a part is cut only if it still lasts at least N seconds;
4. cut those parts out of the video (both cameras together) and ran automatic
   **quality checks**.

Rules the system follows on its own:
* **Uncertain** or failed answers are **never cut**.
* Nothing within 0.5 s of a moment that showed relevant activity is cut.

**Your job** is to confirm that what was cut really contains no relevant activity,
and that the result is a usable video.

---

## 2. Review checklist

Work top to bottom. Tick each item in the QC record (section 4).

### Step 1: Status (top of the page)
- [ ] The run picker shows the right video, and the status pill says **completed**.
  - `running`: the page refreshes on its own; wait until it finishes.
  - `failed`: stop and see section 3 (*Run failed*).
- [ ] **Quality checks** panel verdict:
  - **PASS**: continue.
  - **WARN**: continue, but read every yellow line (section 3).
  - **FAIL**: do **not** approve. See section 3.

### Step 2: Headline numbers (top cards)
- [ ] **Original → processed duration** and **Waste removed (%)** look plausible for
  this footage. For a working session, removing well over half the video is
  unusual. Check the timeline carefully if so.
- [ ] **Frames analysed** = all samples (e.g. `150 / 150`).

### Step 3: Timeline (the coloured bar)
Colours: **green** = relevant, **red** = waste, **yellow** = uncertain, **grey** = no result.
**Hatched** = cut from the video. **Dashed outline** = a candidate that the 0.5 s re-check
trimmed or saved.

- [ ] Every hatched block sits on red. A hatched block over green or yellow is a defect.
- [ ] Click a few **green** cells around each hatched block and read the description.
  Does it describe hands working with objects?
- [ ] Click a few **red** cells inside hatched blocks. Does the description say no
  hands, idle, looking away, walking, etc.?
- [ ] Look at the **yellow** cells: these were kept on purpose. If there are many,
  note it (the model was often unsure, e.g. blur or darkness).

### Step 4: Waste verification table
For each candidate it shows the **outcome** (removed / partially removed / kept) and
where relevant activity protected it.
- [ ] **For every removed or partially removed row:** open the **original video** in
  VLC and watch the removed time range (e.g. `2:00.0–2:20.0`) at normal speed.
  - **No** hands working with objects during that range → OK.
  - Relevant activity **inside** the removed range → defect. Note the times
    (section 3, *Wrong cut*).
- [ ] Watch about 2 s on each side of every cut boundary. Activity that starts
  right at a boundary is the most likely miss.
- [ ] For **kept** candidates: the "Protected at" times should show real
  activity. Spot-check one.

### Step 5: Processed video (player on the dashboard)
- [ ] Play through each cut point. To jump to one, select a sample next to a hatched
  block and press **▶ play at … in processed**.
  - Left and right halves are both present and in sync.
  - No frozen frames, glitches or black frames at the joins.
  - No action is chopped mid-way.
- [ ] Skim the rest of the video at 2× speed.

### Step 6: Decide
| Situation | Decision |
|---|---|
| All checks pass, no defects found | **Approve** |
| Too little was removed (obvious idle footage left), no defects | **Approve with note**, or ask for a re-run with a **smaller N** |
| Something relevant was cut | **Reject**. Report the times, and ask for a re-run with a **larger N** |
| QC = FAIL, or the video is visibly broken | **Reject**. Send to the operator (section 3) |
| Many uncertain / failed samples | **Hold**. Ask for a re-run when the model server is stable |

---

## 3. Problems and what to do

| You see | Meaning | Action |
|---|---|---|
| Status **failed**, *"Cannot reach Ollama"* | The model server or ngrok tunnel was down | Operator restarts Ollama/ngrok and re-runs with `--resume` (nothing is lost) |
| QC **FAIL** `decisions.no_hoi_removed` | A cut overlaps a relevant/uncertain sample (software defect) | **Reject**, send the run ID to the developer |
| QC **FAIL** `output.alignment` / `output.decode` / `output.duration` / `output.layout` | The processed video doesn't match the original or is damaged | **Reject**, operator re-runs. If it repeats, send to the developer |
| QC **FAIL** `source.unchanged` | The original file changed during processing | **Reject**. Find out who modified the file, then re-run |
| QC **WARN** `inference.health` | Some answers failed or were unreadable (kept, never cut) | OK if only a few. Many → operator re-runs with `--resume` to retry |
| QC **WARN** `inference.coverage` | Some seconds had no result (kept) | As above |
| QC **SKIPPED** `imu.sync` | The video has no IMU telemetry (normal for re-encoded files) | No action |
| **Wrong cut** found while watching | The model labelled relevant activity as waste | **Reject**. Record the time range, re-run with larger N (e.g. 8 or 10) |
| Obvious waste left in | Waste shorter than N, or the model was unsure | Usually fine. For a tighter cut, re-run with smaller N |

---

## 4. QC record (copy per video)

```text
Video file:            ______________________   (Source line in the Pipeline panel)
Run ID:                ______________________   (Pipeline panel)
Source sha256 (first 16): __________________
Waste threshold N:     ____ s
Model / prompt:        ______________________
Reviewer / date:       ______________________

Pipeline status:       completed / failed
QC verdict:            PASS / WARN / FAIL       Warnings: __________________
Original → processed:  ______ → ______  (removed ____ %)

Step 3 timeline OK:                 yes / no
Step 4 removed ranges checked:      __ of __   defects: ____________________
Step 5 processed video OK:          yes / no   notes:   ____________________

Decision:  APPROVE / APPROVE WITH NOTE / REJECT / HOLD
Requested action (if any): re-run with N = __ / retry inference / developer
Notes:
```

---

## 5. For the operator: re-running

From the project folder, with the virtual environment active (`venv\Scripts\activate`):

```powershell
# Retry failed/unreadable answers and continue an interrupted run
python scripts/run_pipeline.py --input STEREO_01.mp4 --waste-threshold 5 --resume --config configs/nvenc.yaml

# Different threshold (re-cuts and re-checks; earlier model answers are reused)
python scripts/run_pipeline.py --input STEREO_01.mp4 --waste-threshold 8 --resume --config configs/nvenc.yaml
```
Each new run (without `--resume`) archives the previous one under
`output/<video>/reports/history/<run_id>/`.

## 6. Where the evidence is

All under `output/<video>/`:

| File | What |
|---|---|
| `processed/<video>_processed.mp4` | The approved/rejected deliverable |
| `processed/processing_timeline.json` | Exactly which original ranges were kept, and where they are in the new video |
| `segmentation/verification.json` | Every candidate, why it was cut/trimmed/kept |
| `inference/classifications.jsonl` | The model's label and description for every second |
| `reports/qc_report.json` | All automatic checks with details |
| `reports/summary.json` | Headline numbers |
| `reports/manifest.json` | Run ID, settings, model, software versions |
| `reports/processing.log` | Full log of the run |

## 7. Glossary

* **Stereo pair:** the left and right camera images taken at the same instant.
* **Relevant (HOI):** hands interacting with objects (holding, using tools, assembling…).
  A hand simply holding something still counts.
* **Waste:** no relevant hand-object activity (empty scene, idle, looking away, walking).
* **Uncertain:** the model couldn't tell (blur, dark, hands hidden). Always kept.
* **N (waste threshold):** the minimum length of continuous waste before a cut is considered.
* **Candidate:** a waste stretch ≥ N s at 1-second resolution, before the 0.5 s re-check.
* **PSNR:** a similarity score between a processed frame and the original frame;
  above 30 dB means the same picture.
