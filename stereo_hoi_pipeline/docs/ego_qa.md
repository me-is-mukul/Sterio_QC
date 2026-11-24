# Egocentric QA on recordings in Cloud Storage (→ `processed/`)

`scripts/run_gcs_batch.py` applies the "Egocentric video QA: pass / fail criteria" sheet to
recordings in `gs://op-datoric/raw/` and writes one result folder per recording to
`gs://op-datoric/processed/`. Settings: [`configs/ego_qa.yaml`](../configs/ego_qa.yaml).
It reads every recording format in the bucket (MCAP, robocap, Hydra, EGO-CAM live, EGO-CAM
`capture.egoc`) and recognises the format of each session folder by itself; see
[Recording formats](#recording-formats).
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
| `--source URL` | Folder to read recordings from: every folder below it whose files are a recording in a known format, numbered 1, 2, … in path order (as `--list-sessions` prints them, with each one's format; folder names sort as text, so `01-11-2026` comes before `31-10-2026`). A single recording folder works too. Default: `source.bucket` + `source.raw_root` (`gs://op-datoric/raw/`). Nothing is ever written below it. |
| `--dest URL` | Folder to write results to. Each recording's result goes to `<dest>/<path of the recording below the source folder>/`, e.g. source `…/raw/Op_DD/31-10-2026/`, dest `…/processed/Op_DD/31-10-2026/` → `…/processed/Op_DD/31-10-2026/SDC509/session_00005-…/`. May be in another bucket. Must not be inside the source folder (or contain it). Default: `gs://op-datoric/processed/`. |
| `--start N` / `--end N` | Process only recordings N…M of the numbered list (1 = first, `--end` included). Either may be left out (from the first / to the last). Also applies to a `--session`/`--sessions-file` list. |
| `--list-sessions [URL]` | Print the numbered list for `--source` (or URL) with sizes, mark what `--start`/`--end` select, then exit. Processes nothing. |
| `--session URL` (repeatable) / `--sessions-file FILE` | Instead of the whole source folder: these recordings (they must be below the source folder). |
| `--content-screen on\|off` | Switch for section 3 (minors, private moments, …). Default off. `on` needs a content model, see "Turning on the content screen". |
| `--no-upload` | Keep everything in `outputs/ego_qa/…`; write nothing to the bucket. The clip MP4s then stay on the local disk. |
| `--force` | Redo a session even if `processed/…/qa_report.json` exists, and replace that result: its old `qa_report.json` is deleted first, the new files overwrite the old ones, objects the new result does not have are deleted, the new `qa_report.json` comes last. Needs `storage.objects.delete`. Also discards local partial work. |
| `--config FILE` | Override `configs/ego_qa.yaml` (only existing keys; `models:` is replaced as a whole). |
| `--until STEP` | Stop after a step (no report, no upload). For testing. |
| `--max-segments N` | TEST ONLY: first N segments of each session (the report says `partial_test_run: true`). For the one-file formats (Hydra `video.h265`, `capture.egoc`) a segment is a chunk of `formats.*.chunk_frames` frames. |

Exit code 0: every session ended PASS, FAIL or SKIPPED. 1: at least one ERROR (a technical problem
of the pipeline or a model, not a QA verdict; re-run to continue). 130: interrupted.
A URL with no objects at all, or a folder that is not a recording in a known format, is an ERROR
(nothing is written); a recording that lacks a required file (per format, see Recording formats)
or has a gap in its segment numbers is a FAIL.

A run that stops (crash, Ctrl+C, lost SSH) continues where it stopped when started again with the
same command: finished steps are skipped, labelled frames are cached, extracted segments are kept.
Use `tmux` on the VM.

## VM setup (once per VM)

MediaPipe (the hand detector) needs the system graphics libraries, also on a VM without a screen
or GPU. Without them the batch stops at the start with `libEGL.so.1: cannot open shared object file`:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg tmux libegl1 libgles2
```

The batch starts the models before the first recording, so a missing library shows up in seconds.

## What happens to one recording

| Step | Sheet | What it does |
|---|---|---|
| `extract` | 1 | Lists the session and detects its format; checks the format's required files and gap-free segment numbers. Then indexes every segment from the bucket (exact listed object version): per frame its capture time, keyframe flag, size and position; IMU; calibration. No video is written. |
| `technical` | 1 | Pairs the eyes by capture time (≤ 3 unpaired frames trimmed at each end; any unpaired frame inside = FAIL). Frame timing: the camera runs at 30 ± 0.5 fps and at most 5 % of its frames are lost (`technical.max_missing_frame_share`; the reference QA accepts up to ~4 % and rejects from ~7 %); lost frames are not a FAIL below that, the kept footage is split at every gap > 0.25 s and the missing time is reported as "no video". IMU (same clock, covers the video, no gap > 50 ms, rate steady within 5 % per 10 s; not checked for EGO-CAM), full decode of both eyes (no decoder errors, every frame; FFmpeg messages about our own outputs, e.g. `[null @ …] non monotonically increasing dts to muxer`, are kept in the report as `output_errors` and do not fail it), black/frozen < 50 % per eye, length ≥ 30 s. The decode streams the video again straight into FFmpeg (one process per eye, or one for a side-by-side picture that is cut into the two eyes); the same pass writes the 2 s label frames and the calibration frames. Progress is logged every minute. |
| `calibration` | 2 | Rectified if median vertical offset ≤ 1.5 px and p95 ≤ 5 px on ≥ 200 inlier matches over ≥ 10 held-out frame pairs; else unrectified. Eye order decided from the images. Never fails a recording. |
| `labelling` | 4 | One left-eye frame every 2 s. Default: MediaPipe counts the wearer's hands (both/one/none); the motion check marks it idle or moving (see Models). |
| `content_recording` | 3 | Off by default (`content.enabled: false`; the report says `"content": {"checked": false}`). With a content model: "no real work" for a strip of 16 frames over the recording → FAIL. Minor, lens, headset not worn, phone and private moments are cut as stretches in `cropping` (`content.cut_as_stretches`, as the reference QA does); a flag removed from that list gets the old whole-recording rule (minor in ≥ 2 frames, lens / third-person in > 50 % → FAIL). |
| `cropping` | 4 | Every sample gets its reasons to cut: idle, no wearer hand (all models agree), and with the content screen phone use, minor, private moment, lens blocked, headset not worn. A run of samples that each have a reason, spanning ≥ 10 s, is cut (`cropping.cut_seconds`; reasons may mix inside one run, as in the reference). Pieces are split at gaps of lost frames, start/end on keyframes shared by both eyes; pieces < 30 s dropped; nothing ≥ 30 s left → FAIL. |
| `content_clips` | 3 | Off by default (every piece is kept). With a content model: "one trivial motion repeated" for a 12-frame strip → drop the piece. |
| `clips` | — | Writes each kept piece: `left.mp4` + `right.mp4` (two-stream formats) or one `stereo_sbs.mp4` (side-by-side formats; each eye's x range is in `clip.json`), packets streamed from the bucket and copied, no re-encode, verified with ffprobe; plus `frame_times.csv`, `imu.csv`, `clip.json`. Each verified MP4 is uploaded and deleted before the next one is written. |

Then `qa_report.json` is written and the rest is uploaded, `qa_report.json` last (it marks the
result as complete). Local working files are deleted after a successful upload.

## Recording formats

The format of a folder is decided from the names of the files directly inside it
(`ingestion/formats/detect.py`); sub-folders are separate sessions. `--list-sessions` shows each
recording's format; folders that match no format (macOS index folders, empty folders) are counted
as "not a recording" and skipped. On 01-11-2026 the bucket holds 453 recordings: 205 MCAP, 237
robocap, 6 EGO-CAM live, 5 `capture.egoc`.

**robocap is judged one segment at a time.** Every robocap segment has its own MP4 files and start
times, so each segment is a recording of its own: it is listed, numbered, passed or failed and
stored separately, as `<session folder>/#seg<N>` (results in `<session>/seg<N>/`). A broken segment
(e.g. its MP4 has no index because the recorder was stopped) fails alone; the other segments of the
session are still judged. `--list-sessions` therefore shows one line per robocap segment, and
`--session <robocap folder>/` runs all its segments (`--session <folder>/#seg2` only that one).
Joining the segments into one stream also made the decoder report one error at a segment
boundary (measured on SDC482 session13: 0 errors when each segment is decoded alone).

| Format | Files in the session folder | Eyes | Frame times | IMU |
|---|---|---|---|---|
| `mcap` | `ego_NNNN.mcap` (from 0), `session.json` | two HEVC streams | device clock | in the MCAP, same clock |
| `robocap` | `robocap_segmentN_<file>` (from 1, 10 or 30 min each depending on the device): six MP4 cameras, `imu_left.db`, `imu_right.db`, `mag_middle.db` | `video_left_front.mp4` + `video_right_front.mp4` (`formats.robocap.cameras`) | each MP4's `comment` tag = its first frame's device time (µs) + MP4 times | `imu_left.db` (SQLite, raw units, ns, same clock) |
| `hydra` | `video.h265`, `frames.idx`, `h265_frame_meta.csv`, `h265_imu.csv`, `meta.json`, `session.json`, `stereo_calibration.bin` | one side-by-side HEVC picture, 3840x1200: left half / right half | sensor time per eye (`frame_pts_us`) | `h265_imu.csv` (deg/s, g; 500 Hz, same clock) |
| `egocam_live` | `video_live_NNNN.mp4` (30 s each, from 0), `frame_map.csv`, `calibration.json`, `metadata.json` | one side-by-side HEVC picture, 4000x1200: x 160-2080 and 2080-4000 (`calibration.json` `eye_crop_x`) | host receive time (`frame_map.csv`) + the camera's capture counter | inside each frame's barcode: **not checked** |
| `egocam_egoc` | `capture.egoc`, `calibration.json`, `metadata.json` | same picture, as JPEGs (MJPEG) | host receive time per frame | inside the barcode: **not checked** |

Required files (missing = FAIL): MCAP `session.json` + `ego_0000.mcap`; robocap: the segment's files;
Hydra `video.h265`, `frames.idx`, `h265_frame_meta.csv`; EGO-CAM live `video_live_0000.mp4` +
`frame_map.csv`; egoc `capture.egoc`. Every format: segment numbers without gaps; in every segment
both cameras (and the IMU where the format has one) readable.

Facts the readers rely on, checked on real files on 24-11-2026:

* **robocap**: one video track per MP4, index (`moov`) after the video data, no B-frames, keyframe
  every ~30 frames; parameter sets only in the MP4 header, so they are put in front of every
  keyframe when streaming. Segments follow each other with one frame interval. The left/right front
  cameras start ~30 µs apart. IMU about 203 Hz. **IMU units are not documented**: accel 8192 per g
  (the ±4 g range; 1 g measured at rest), gyro 65.5 per deg/s (the ±500 deg/s range; fitted from the
  recordings as 58-71 per deg/s). If the device team gives other ranges, change
  `formats.robocap.gyro_lsb_per_dps` / `accel_lsb_per_g`. There is no calibration file: the
  calibration check reports `unrectified` and decides the eye order from the raw pixels.
* **Hydra**: `frames.idx` gives every frame's byte range in `video.h265`; a keyframe with
  parameter sets every 30 frames; the eyes are ~30 µs apart. `stereo_calibration.bin` is in an
  unknown binary format and is not used (calibration as for robocap). The 100 GB files are indexed
  in chunks of `formats.hydra.chunk_frames` frames, so an interrupted index resumes.
* **EGO-CAM** (both): `calibration.json` holds OpenCV fisheye (Kannala-Brandt) K/D per camera and the
  stereo R/T; it is used for the calibration check. It names cam0 the *right* camera; the image
  check agrees (left eye = cam1). Frame times are when the computer received the frame, which jitter
  by tens of ms (a 108 ms wait with no frame lost was seen), so the frame rate is the mean over the
  recording; for EGO-CAM live gaps and lost frames come from the camera's capture counter
  (`capture_seq`), for `capture.egoc` (whose counter only numbers written frames) from the receive
  times. `capture.egoc` records are `EGF2`, frame number, time (µs), length, a checksum of unknown
  kind (not verified), then one JPEG. The camera's JPEGs carry an "AVI1" header block that FFmpeg
  reports as an error on every frame although the picture decodes; exactly that message is ignored.
  A last frame cut off because the recording stopped while writing is a note, not a FAIL.
* **EGO-CAM IMU**: the 300 Hz IMU is only inside a barcode in every frame (the strip at x < 160).
  Decoding it needs the device team's `tools/imu_from_video.py`; until then the report says
  `"imu": {"checked": false, ...}`, the IMU checks are skipped, and the idle check uses the
  picture alone.

Adding a format: write a reader in `ingestion/formats/` (index each segment into the common tables
with `write_segment`, stream packets with `_segment_packets`), add its file-name rule to
`detect.py`, and register it in `formats/__init__.py`.

## Disk use on the VM: no raw video is stored

The recordings' video files are never saved on the VM: every pass reads them from the bucket as
a stream (ranged reads of the exact listed object version) and passes the video packets through
memory straight to FFmpeg. Each recording is read up to three times: index, decode, and the parts
that become clips (MP4 formats index from the file's small `moov` box only, so their index pass
reads about 100 kB per file). Small sensor files (robocap's SQLite IMU, a few MB per segment) are
downloaded, read and deleted; large CSV files (Hydra's `h265_imu.csv`, up to 4 GB) are parsed
while streaming.

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
  recording/            copies of the recording's small metadata files (session.json, meta.json,
                        calibration.json, metadata.json, ... depending on the format)
  source_listing.json   the format and the exact objects (name, generation, checksum) that were processed
  clips/clip_NNN/       left.mp4 right.mp4 (or stereo_sbs.mp4)  frame_times.csv  imu.csv  clip.json
  labels/               raw per-model answers (jsonl), strip answers, per-sample decisions
  logs/                 manifest.json  processing.log  resolved_config.yaml
  _<step>.json          intermediate result of each step
gs://op-datoric/processed/_batches/<batch id>/  sessions.txt  summary.csv  sheet.csv  detail.csv
```

`summary.csv` has one row per recording (per segment for robocap): `session, status, reasons,
source_seconds` (length of the paired video), `kept_seconds, clips, calibration, elapsed_s`.

`sheet.csv` and `detail.csv` hold the same results in the reference QA's format
(`reporting/reference_sheet.py`):

* `sheet.csv`, one row per recording: `Upload date folder, Headset, Recording` (`raw/…/<session>#seg<N>`),
  `Segment, Rig` (RoboCap, OpenRobot = hydra, Eye 2 = capture.egoc, Eye 2 (live), MCAP), `Status`
  (Accepted / Accepted (partly cut) / Rejected / Error), `Source hours, Accepted hours, Reason`
  ("Cut: Idle (no work in hand) 3.4 min, …", or why it was rejected), `Rejection group, Hours cut,
  Hours with no video frames`. Source = accepted + cut + no video.
* `detail.csv`, one row per removed stretch: start, end, length (s), main reason and the share of the
  stretch's 2 s samples per reason ("Piece under 30 s between cuts" = a kept piece too short to keep).

Recordings skipped because they were processed before are in the sheets too (their `qa_report.json`
is read from the bucket), so running the same batch command again rebuilds both sheets in seconds.
Reports written before the sheets existed have no per-sample shares: each of their stretches counts
under one reason.

* A FAILed recording gets the report, labels (if it got that far) and logs, no clips.
* `upload.on_existing`: `skip` (default: a session with a `qa_report.json` is not redone without
  `--force`), `overwrite`, or `new_run_folder` (`…/<session>/runs/<run id>/`). A complete result
  (one with `qa_report.json`) is only replaced with `--force` or `overwrite`: its `qa_report.json` is
  deleted before anything new is uploaded, so the folder reads as incomplete until the new result is
  done. Objects of an earlier result or an interrupted attempt that the new result does not have are
  deleted just before the new `qa_report.json` is uploaded. Nothing outside the session's own result
  folder is ever written or deleted.
* Clip MP4s use the nominal 30 fps clock; the measured time of every frame is in
  `frame_times.csv` (ns; the clock depends on the format, see Recording formats). IMU rows are on
  the same clock. EGO-CAM clips have an empty `imu.csv` (the IMU is not decoded yet).

## Models (`models:` in the config)

**Default: no AI service at all.** Everything runs on the VM:

* `hands` (`provider: mediapipe`): MediaPipe Hand Landmarker (`models/hand_landmarker.task`, in
  the repo) counts the wearer's hands on each sampled frame pair, in **both eyes**
  (`mediapipe.both_eyes`), with confidences 0.3. "No hands" = it finds none of the wearer's hands
  in either eye. Known limit, measured on a night recording: MediaPipe misses many hands in dark
  scenes, at the fisheye edge and under motion blur. Both eyes + 0.3 raised the frames with a
  found hand from 22 % to 42 % (kept footage 96 s → 159 s of 250 s) with no false hands on a static
  recording, but some real work in dark footage is still cut as "no hands". Measured against the reference QA
  (8 SDC482 recordings, 2159 samples): MediaPipe finds no hand on 57 % of the samples the reference
  keeps and 70 % of those it cuts, so only 62 % of the samples are judged the same. No setting of
  the run length (10-60 s) or smoothing did better; the reference's cuts are mostly "Idle (no work in
  hand)" and "Phone use", which need a vision-language model (`configs/ego_qa_gemini.yaml`).
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

**Optional, closest to the reference QA (billed):** `--config configs/ego_qa_gemini.yaml`. Gemini
alone labels every 2 s frame with the reference's definitions (prompt `ego-qa-frame-v2`: hands in
view counts a hand only partly in view or a held tool entering from the edge; idle = no work in hand;
phone use, minor, private moment, lens blocked, headset not worn) and the content screen is on. One
Gemini call per 2 s of video (900 per 30 min) plus a few strips; answers are cached, so a re-run does
not pay again.

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
validation, the motion idle check and the answer cache. `tests/test_formats.py` builds a small
session in each of the other formats (robocap, Hydra, EGO-CAM live, `capture.egoc`; real HEVC/JPEG
video) and runs it through detection, indexing, decoding (per-eye black detection on a
side-by-side picture) and clips that must be bit-identical to the source, plus the whole pipeline
on a robocap and an EGO-CAM session.
