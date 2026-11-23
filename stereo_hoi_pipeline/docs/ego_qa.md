# Egocentric QA on MCAP recordings (Cloud Storage → `processed/`)

`scripts/run_gcs_batch.py` applies the "Egocentric video QA: pass / fail criteria" sheet to
recordings in `gs://op-datoric/raw/` and writes one result folder per recording to
`gs://op-datoric/processed/`. Settings: [`configs/ego_qa.yaml`](../configs/ego_qa.yaml).
It is separate from the MP4 HOI pipeline (`scripts/run_pipeline.py`), which is unchanged.

## Run it

From `stereo_hoi_pipeline/`, after `pip install -r requirements.txt` (FFmpeg/FFprobe on PATH):

```bash
# 1. pick sessions: every session with its size
python scripts/run_gcs_batch.py --list-sessions gs://op-datoric/raw/Op_DD/

# 2. first check on one small session: sections 1 and 2 only, no models, nothing uploaded, no clips
python scripts/run_gcs_batch.py --until calibration --session gs://op-datoric/raw/Op_DD/<date>/<device>/<session>/

# 3. the batch (one URL per line in the file; '#' starts a comment)
python scripts/run_gcs_batch.py --sessions-file sessions_10.txt
```

| Option | Meaning |
|---|---|
| `--session URL` (repeatable) / `--sessions-file FILE` | Sessions to process, in that order, one at a time. |
| `--list-sessions URL` | List sessions below a prefix with size and file count, then exit. |
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
| `labelling` | 3, 4 | One left-eye frame every 2 s; every model labels every frame (hands both/one/none, working/idle, content flags). |
| `content_recording` | 3 | Minor in ≥ 2 frames, lens covered/smudged or third-person in > 50 % of frames, or "no real work" for a strip of 16 frames over the recording → FAIL. |
| `cropping` | 4 | Cut ≥ 10 s where all models say idle, and ≥ 10 s where all say no wearer hand. Pieces start/end on keyframes shared by both eyes; pieces < 30 s dropped; nothing ≥ 30 s left → FAIL. |
| `content_clips` | 3 | Per piece: private moment in ≥ 2 frames, or "one trivial motion repeated" for a 12-frame strip → drop the piece. |
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
  labels.csv            one row per 2 s sample: each model's labels, content flags, idle/no_hands, brightness, sharpness, head motion
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

Two models label every frame. A cut needs **all** models that provide `hands`/`activity` to agree;
a failed or invalid answer never counts as idle or no-hands. One model (`content`) also judges the
content flags and the frame strips. Default:

* `model_a`: `provider: env`: the endpoint in `.env` (`INFERENCE_PROVIDER` + `OLLAMA_*` or `VLLM_*`).
  This stands in for the sheet's "fine-tuned vision model" until that model exists.
* `model_b`: Gemini (`gemini-3.5-flash`, Vertex AI, `asia-south1`).

Providers: `gemini`, `ollama`, `openai_compatible`, `env`, `mediapipe` (hands only). Before the first
recording that needs them, every model is started and must answer one real request (a model
listing can succeed while predictions are denied). A model that fails more than 2 % of the frames
stops that recording with ERROR. Re-running retries only the failed frames.

Using MediaPipe for hands and only one model for activity needs an explicit
`cropping.min_models_per_decision: 1` in a `--config` file (the sheet asks for two models).

## Requirements on Google Cloud

* Read access to `raw/` and create access to `processed/` (+ delete for `overwrite`).
* Vertex AI: `roles/aiplatform.user` for the account that runs the batch (on the VM: its service
  account). Without it Gemini answers `403 aiplatform.endpoints.predict denied`.
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
validation.
