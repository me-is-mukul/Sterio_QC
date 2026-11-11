# IMU Extraction and Synchronization

## Extraction (`metadata/gpmf_extractor.py`)

* The GPMF stream index comes from `video_info.json → telemetry.gpmf_stream_indices`
  (`codec_tag_string == "gpmd"` or handler “GoPro MET”).
* `ffprobe -show_packets` gives each packet's `pts_time`, `duration_time` and `size`.
  This is saved as `imu/original/gpmf_packets.json` with byte offsets.
* `ffmpeg -map 0:<idx> -c copy -f data` writes `imu/original/telemetry.bin`. Its size
  must equal the sum of packet sizes. **The raw file is authoritative and never deleted.**

## Parsing (`metadata/imu_parser.py`)

* Nested KLV: FourCC, type, struct size, repeat (BE), payload padded to 4 bytes.
  Type `\0` = nested (`DEVC → STRM → …`).
* Sticky properties per `STRM`: `SCAL` (scalar or per-channel), `SIUN`/`UNIT`,
  `ORIN`, `STNM`, `STMP` (µs).
* Values are divided by `SCAL`. Raw integers are kept as `raw_c*`.
* Channels are exported as `c0, c1, c2` in stored order. The camera's declared
  order (`ORIN`, e.g. `ZXY`) is recorded instead of assuming axis names.
* Timing: sample `i` of `n` in a packet → `pts + duration·i/n` (container time).
  `time_video_seconds = container time − video start time`. The packet's `STMP` is
  kept as the original sensor clock.
* Truncated/unknown data is skipped with a warning, never fabricated.

CSV columns: `sample_index, packet_index, sample_in_packet, packet_pts_seconds,
packet_stmp_us, time_container_seconds, time_video_seconds, c0..cN, raw_c0..raw_cN`.

## Synchronization (`synchronization/imu_aligner.py`)

Given `processed/processing_timeline.json`:
* keep a sample iff `time_video_seconds` ∈ some retained `[original_start, original_end)`;
* `time_processed_seconds = processed_start + (t − original_start)`, plus `segment_index`;
* samples outside `[0, duration)` are dropped and counted separately.

Rows are never removed by row number. Original time columns are preserved.
`imu/processed/synchronization_metadata.json` records the method, segments and
per-stream kept/removed counts.

## Status on the sample video

`STEREO_01.mp4` was re-encoded (encoder `Lavf61`) and contains no GPMF stream. Its
only data stream is a 5-entry chapter text track. `imu_metadata.json` records
`available: false` and synchronization is `skipped`. The parser and aligner were
checked with synthetic GPMF payloads only. Run an original GoPro `GX*.MP4` to
exercise them on real data.
