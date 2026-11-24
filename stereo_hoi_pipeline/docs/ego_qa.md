# Egocentric QA on MCAP recordings (Cloud Storage → `processed/`)

`scripts/run_gcs_batch.py` applies the "Egocentric video QA: pass / fail criteria" sheet to
recordings in `gs://op-datoric/raw/` and writes one result folder per recording to
`gs://op-datoric/processed/`. Settings: [`configs/ego_qa.yaml`](../configs/ego_qa.yaml).
It is separate from the MP4 HOI pipeline (`scripts/run_pipeline.py`), which is unchanged.

## Run it

From `stereo_hoi_pipeline/`, after `pip install -r requirements.txt` (FFmpeg/FFprobe on PATH):

```bash
# 1. see the numbered list of recordings in a source folder (and which ones --start/--end pick)
python scripts/run_gcs_batch.py --source gs://op-datoric/raw/Op_DD/31-10-2026/ --list-sessions --start 1 --end 10

# 2. process recordings 1 to 10 of that folder and write the results to the destination folder
python scripts/run_gcs_batch.py --source gs://op-datoric/raw/Op_DD/31-10-2026/ \
    --dest gs://op-datoric/processed/Op_DD/31-10-2026/ --start 1 --end 10

# or: an explicit list (one URL per line; '#' starts a comment)
python scripts/run_gcs_batch.py --sessions-file sessions_10.txt
```

| Option | Meaning |
|---|---|
| `--source URL` | Folder to read recordings from: every `session_*` folder with MCAP segments below it, numbered 1, 2, … in path order (as `--list-sessions` prints them; folder names sort as text, so `01-11-2026` comes before `31-10-2026`). A single session folder works too. Default: `source.bucket` + `source.raw_root` (`gs://op-datoric/raw/`). Nothing is ever written below it. |
| `--dest URL` | Folder to write results to. Each recording's result goes to `<dest>/<path of the recording below the source folder>/`, e.g. source `…/raw/Op_DD/31-10-2026/`, dest `…/processed/Op_DD/31-10-2026/` → `…/processed/Op_DD/31-10-2026/SDC509/session_00005-…/`. May be in another bucket. Must not be inside the source folder (or contain it). Default: `gs://op-datoric/processed/`. |
| `--start N` / `--end N` | Process only recordings N…M of the numbered list (1 = first, `--end` included). Either may be left out (from the first / to the last). Also applies to a `--session`/`--sessions-file` list. |
| `--list-sessions [URL]` | Print the numbered list for `--source` (or URL) with sizes, mark what `--start`/`--end` select, then exit. Processes nothing. |
| `--session URL` (repeatable) / `--sessions-file FILE` | Instead of the whole source folder: these recordings (they must be below the source folder). |
| `--content-screen on\|off` | Switch for section 3 (minors, private moments, …). Default off. `on` needs a content model, see "Turning on the content screen". |
| `--no-upload` | Keep everything in `outputs/ego_qa/…`; write nothing to the bucket. The clip MP4s then stay on the local disk. |
| `--force` | Redo a session even if `processed/…/qa_report.json` exists, and replace that result: its old `qa_report.json` is deleted first, the new files overwrite the old ones, objects the new result does not have are deleted, the new `qa_report.json` comes last. Needs `storage.objects.delete`. Also discards local partial work. |
| `--config FILE` | Override `configs/ego_qa.yaml` (only existing keys; `models:` is replaced as a whole). |
| `--until STEP` | Stop after a step (no report, no upload). For testing. |
| `--max-segments N` | TEST ONLY: first N segments of each session (the report says `partial_test_run: true`). |

Exit code 0: every session ended PASS, FAIL or SKIPPED. 1: at least one ERROR (a technical problem
of the pipeline or a model, not a QA verdict; re-run to continue). 130: interrupted.
A URL with no objects at all is an ERROR (nothing is written); a session folder that exists but
lacks `session.json` or `ego_0000.mcap`, or has a gap in its segment numbers, is a FAIL.

A run that stops (crash, Ctrl+C, lost SSH) continues where it stopped when started again with the
same command: finished steps are skipped, labelled frames are cached, extracted segments are kept.
Use `tmux` on the VM.

## What happens to one recording

| Step | Sheet | What it does |
|---|---|---|
| `extract` | 1 | Lists the session; checks `session.json`, `ego_0000.mcap` and gap-free segment numbers. Then streams each segment from the bucket (exact listed object version, MCAP chunk CRCs checked) and indexes it: per frame its capture time, keyframe flag, size and position; IMU; calibration. Channels are found by topic name. No video is written. |
| `technical` | 1 | Pairs the eyes by capture time (≤ 3 unpaired frames trimmed at each end; any unpaired frame inside = FAIL). Frame timing (≈ 30 fps, no gap > 0.1 s), IMU (same clock, covers the video, no gap > 50 ms, rate steady within 5 % per 10 s), full decode of both eyes (no errors, every frame), black/frozen < 50 %, length ≥ 30 s. The decode streams the segments again straight into one FFmpeg per eye; the same pass writes the 2 s label frames and the calibration frames. |
| `calibration` | 2 | Rectified if median vertical offset ≤ 1.5 px and p95 ≤ 5 px on ≥ 200 inlier matches over ≥ 10 held-out frame pairs; else unrectified. Eye order decided from the images. Never fails a recording. |
| `labelling` | 4 | One left-eye frame every 2 s. Default: MediaPipe counts the wearer's hands (both/one/none); the motion check marks it idle or moving (see Models). |
| `content_recording` | 3 | Off by default (`content.enabled: false`; the report says `"content": {"checked": false}`). With a content model: minor in ≥ 2 frames, lens covered/smudged or third-person in > 50 % of frames, or "no real work" for a strip of 16 frames over the recording → FAIL. |
| `cropping` | 4 | Cut ≥ 10 s where all models say idle, and ≥ 10 s where all say no wearer hand. Pieces start/end on keyframes shared by both eyes; pieces < 30 s dropped; nothing ≥ 30 s left → FAIL. |
| `content_clips` | 3 | Off by default (every piece is kept). With a content model: private moment in ≥ 2 frames, or "one trivial motion repeated" for a 12-frame strip → drop the piece. |
| `clips` | — | Writes each kept piece: `left.mp4`, `right.mp4` (packets streamed from the bucket and copied, no re-encode, verified with ffprobe), `frame_times.csv`, `imu.csv`, `clip.json`. Each verified MP4 is uploaded and deleted before the next one is written. |

Then `qa_report.json` is written and the rest is uploaded, `qa_report.json` last (it marks the
result as complete). Local working files are deleted after a successful upload.

## Disk use on the VM: no raw video is stored

The `.mcap` recordings are never saved on the VM: every pass reads them from the bucket as a
stream (ranged reads of the exact listed object version) and passes the video packets through
memory straight to FFmpeg. Each recording is read up to three times: index, decode, and the parts
that become clips. Reads inside the same region are free.

What is written locally (`work.directory`, per recording), and when it goes away:

| What | Size | Removed |
|---|---|---|
| Frame and IMU index tables | ≈ 0.1 GB per hour of recording | after a successful upload |
| Sampled frames (JPEG, 1280 px, one per 2 s per eye) | see measured numbers below | after a successful upload |
| 24 full-size calibration frame pairs (PNG) | ≈ 0.15 GB | after a successful upload |
| One clip MP4, one eye at a time | up to the length of the longest kept piece | uploaded and deleted at once |
| Reports, labels, logs | a few MB | kept |

The largest single file is one eye of the longest clip: at most about half the size of the
recording's segments for that stretch (a 2 h 14 min recording with nothing cut: ≈ 9 GB).
With `--no-upload` the clips stay on disk.

## Result layout

```text
gs://op-datoric/processed/Op_DD/<date>/<device>/<session>/
  qa_report.json        verdict PASS|FAIL, fail_reasons, every check with its numbers   (uploaded last)
  calibration.json      rectified|unrectified, vertical offsets, eye order, intrinsics/extrinsics used
  labels.csv            one row per 2 s sample: hands, idle/moving (with its numbers), idle/no_hands decision, brightness, sharpness, head motion
  session.json          copy of the recording's session.json
  source_listing.json   the exact objects (name, generation, checksum) that were processed
  clips/clip_NNN/       left.mp4  right.mp4  frame_times.csv  imu.csv  clip.json
  labels/               raw per-model answers (jsonl), strip answers, per-sample decisions
  logs/                 manifest.json  processing.log  resolved_config.yaml
  _<step>.json          intermediate result of each step
gs://op-datoric/processed/_batches/<batch id>/  sessions.txt  summary.csv
```

* A FAILed recording gets the report, labels (if it got that far) and logs, no clips.
* `upload.on_existing`: `skip` (default: a session with a `qa_report.json` is not redone without
  `--force`), `overwrite`, or `new_run_folder` (`…/<session>/runs/<run id>/`). A complete result
  (one with `qa_report.json`) is only replaced with `--force` or `overwrite`: its `qa_report.json` is
  deleted before anything new is uploaded, so the folder reads as incomplete until the new result is
  done. Objects of an earlier result or an interrupted attempt that the new result does not have are
  deleted just before the new `qa_report.json` is uploaded. Nothing outside the session's own result
  folder is ever written or deleted.
* Clip MP4s use the nominal 30 fps clock; the measured capture time of every frame is in
  `frame_times.csv` (device monotonic clock, ns). IMU rows are on the same clock.

## Models (`models:` in the config)

**Default: no AI service at all.** Everything runs on the VM:

* `hands` (`provider: mediapipe`): MediaPipe Hand Landmarker (`models/hand_landmarker.task`, in
  the repo) counts the wearer's hands on each sampled frame pair, in **both eyes**
  (`mediapipe.both_eyes`), with confidences 0.3. "No hands" = it finds none of the wearer's hands
  in either eye. Known limit, measured on a night recording: MediaPipe misses many hands in dark
  scenes, at the fisheye edge and under motion blur. Both eyes + 0.3 raised the frames with a
  found hand from 22 % to 42 % (kept footage 96 s → 159 s of 250 s) with no false hands on a static
  recording, but some real work in dark footage is still cut as "no hands". Raise
  `cropping.no_hands_cut_seconds` to cut less.
* `idle` (`provider: motion`): a sample is idle when nothing moves. The picture barely changes
  compared with the previous and the next sample (share of changed pixels ≤
  `motion.max_changed_fraction`), **and** the head is still (median gyroscope speed within ±1 s ≤
  `motion.max_gyro_rad_s`). Moving hands or a moving head mean "not idle", so it only cuts
  stretches where the wearer is truly inactive, and it keeps footage when unsure. Its numbers per
  sample are in `labels.csv` (`idle_changed_prev`, `idle_changed_next`, `idle_gyro_median_rad_s`).
* Section 3 (minors, private moments, lens, third-person, "no real work") is **not checked**:
  `content.enabled: false` (see "Turning on the content screen" below).

A cut needs **all** models that provide `hands`/`activity` to agree (`cropping.min_models_per_decision`:
1 by default; the sheet asks for two models). A failed answer never counts as idle or no-hands.
Answers are cached per sample in `labels/labels_<model>.jsonl` and reused only while the model and
its settings are unchanged.

### Turning on the content screen (section 3) later

Section 3 (minor in view, private moment, lens covered, third-person camera, no real work, one
trivial motion repeated) is **switched off and not configured** for now: the current recordings
contain none of it. It needs a vision-language model (it cannot be judged by MediaPipe or the
motion check). To turn it on:

1. Pick the model and give the VM access to it:
   * **Gemini on Vertex AI** (simplest on the VM): a project admin grants the VM's service account
     `roles/aiplatform.user` (Console: IAM → Grant access → `tech-user@datadensity.iam.gserviceaccount.com`
     → role "Vertex AI User"). Check on the VM:
     `python -c "from google import genai; print(genai.Client(vertexai=True, project='datadensity', location='asia-south1').models.generate_content(model='gemini-3.5-flash', contents='say ok').text)"`
     must print an answer, not `403 PERMISSION_DENIED`.
   * **Or an own model server** (Qwen on Ollama / vLLM): put its address in `.env`
     (`INFERENCE_PROVIDER=ollama` + `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, or `openai_compatible` + `VLLM_*`)
     and use `provider: env` below.
2. In `configs/ego_qa.yaml`, uncomment the `content:` block under `models:` and fill it in. It
   provides only `content`, so hands and idle stay with MediaPipe and the motion check.
3. Switch it on: `content.enabled: true` in `configs/ego_qa.yaml`, or `--content-screen on` per run.
4. Test on one recording before a batch:
   `python scripts/run_gcs_batch.py --source <one session folder> --content-screen on --no-upload --max-segments 2`
   The log must say `content screen (section 3): ON` and `Model content ready`; `labels.csv` then has
   the content columns (`content_minor_visible`, `content_private_moment`, …).

If the switch is `on` without a content model, the run stops before doing anything with
"the content screen is switched on …, so exactly one model must provide 'content'". Recordings
processed while it was off say `"content": {"checked": false}` in `qa_report.json`; to screen them
too, run them again with `--force` (needs delete permission on the destination).

**Optional, the sheet's full setup with vision-language models:**
`--config configs/ego_qa_with_vlm.yaml` (two models: the endpoint in `.env` and Gemini on Vertex
AI; turns section 3 on; both models must agree before a cut). Providers: `gemini`, `ollama`,
`openai_compatible`, `env`. Each such model must answer one real request before the first
recording; one that fails more than 2 % of the frames stops that recording with ERROR.

## Requirements on Google Cloud

* Read access to `raw/` and create access to `processed/` (+ delete for `overwrite`).
* Only with `ego_qa_with_vlm.yaml`: `roles/aiplatform.user` for the account that runs the batch
  (on the VM: its service account). Without it Gemini answers `403 aiplatform.endpoints.predict denied`.
* Credentials: on the VM nothing to set up (`gcp.credentials: auto` uses the VM's service account).
  Elsewhere `auto` falls back to `gcloud auth print-access-token` (honours impersonation).

## Measured on the recordings (op-datoric, Nov 2026)

See talking_to_mukul.md, section 1, for the data format. Findings the code relies on:

* Channel IDs differ between files: topics are matched by name.
* Every keyframe carries VPS/SPS/PPS; keyframes every 1.0 s at the same instants in both eyes;
  no B-frame reordering (`has_b_frames=0`), so file order = decode order.
* One eye can record a frame more at the end (seen: cam1 +1 frame), hence the edge trimming.
* The cam0→cam1 `FrameTransform` is the pose of cam1 in cam0 ("pose" reading: median vertical
  offset 0.71 px vs 10.2 px for the other reading). cam1 sits 6.2 cm to the left of cam0 and the
  images agree: **cam1 is the left eye**. `calibration.labelled_left: calibration` uses this
  geometry as "as labelled"; "eyes swapped" means the images disagree with it.

## Tests

`python -m pytest tests` (needs `pip install pytest`): cropping rules, eye pairing, section-1 checks,
MCAP indexing and packet streaming on a synthetic recording with the device's schemas, the full
streaming decode + clip path on real HEVC video (needs FFmpeg with libx265; clip frames are checked
to be identical to the source), session paths, configuration validation, content rules and answer
validation, the motion idle check and the answer cache.
