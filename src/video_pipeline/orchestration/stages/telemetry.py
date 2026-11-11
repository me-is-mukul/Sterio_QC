"""Phase 2 stage: GPMF extraction + IMU parsing."""

from __future__ import annotations

from video_pipeline.core.io import utc_now_iso, write_csv_atomic, write_json_atomic, write_text_atomic
from video_pipeline.core.tools import resolve_executable
from video_pipeline.metadata.gpmf_extractor import extract_gpmf
from video_pipeline.metadata.imu_parser import PARSER_VERSION, parse_imu, parse_klv, series_metadata
from video_pipeline.orchestration.context import RunContext
from video_pipeline.orchestration.stages.base import Stage, StageOutcome, load_video_info, video_start_time

CSV_NAMES = {"ACCL": "accelerometer.csv", "GYRO": "gyroscope.csv"}
_BASE_COLUMNS = ["sample_index", "packet_index", "sample_in_packet", "packet_pts_seconds", "packet_stmp_us",
                 "time_container_seconds", "time_video_seconds"]


def _klv_tree(items, depth: int = 0) -> list[str]:
    lines = []
    for item in items:
        lines.append(f"{'  ' * depth}{item.key} type={item.type!r} size={item.struct_size} repeat={item.repeat}")
        lines.extend(_klv_tree(item.children, depth + 1))
    return lines


class TelemetryStage(Stage):
    name = "telemetry"
    config_keys = ("telemetry.parse_streams",)

    def run(self, ctx: RunContext) -> StageOutcome:
        info = load_video_info(ctx)
        original_dir = ctx.workspace.imu_dir / "original"
        original_dir.mkdir(parents=True, exist_ok=True)
        meta_path = original_dir / "imu_metadata.json"
        indices = info.get("telemetry", {}).get("gpmf_stream_indices", [])

        if not indices:
            for stale in original_dir.glob("*"):
                if stale.is_file():
                    stale.unlink()
            write_json_atomic(meta_path, {
                "available": False,
                "reason": "No GoPro GPMF telemetry stream in the source video (re-encoded files usually lose it).",
                "parser_version": PARSER_VERSION,
                "recorded_at": utc_now_iso(),
            })
            ctx.logger.warning("No GPMF telemetry stream: IMU extraction skipped and recorded as unavailable.")
            return StageOutcome(outputs=(meta_path,), details={"available": False})

        stream_index = indices[0]
        timeout = ctx.config.get("tools.timeout_seconds")
        extraction = extract_gpmf(
            resolve_executable(ctx.config.get("tools.ffmpeg")), resolve_executable(ctx.config.get("tools.ffprobe")),
            ctx.source.path, stream_index, original_dir / "telemetry.bin", timeout_seconds=timeout,
        )
        for command in extraction.commands:
            ctx.manifest.record_command(self.name, command.to_record())
        packets_path = original_dir / "gpmf_packets.json"
        write_json_atomic(packets_path, [p.to_dict() for p in extraction.packets])

        raw = extraction.raw_path.read_bytes()
        fourccs = ctx.config.get("telemetry.parse_streams")
        series = parse_imu(raw, extraction.packets, fourccs, video_start_seconds=video_start_time(info))

        dump = [f"GPMF stream {stream_index}: {len(extraction.packets)} packets, {len(raw)} bytes",
                "KLV structure of the first packet:"]
        first = extraction.packets[0]
        dump += _klv_tree(parse_klv(raw[first.offset:first.offset + first.size]))
        outputs = [meta_path, extraction.raw_path, packets_path]
        files = {}
        for code, s in series.items():
            dump.append(f"{code}: {len(s.rows)} samples, units={s.units}, axis_order={s.axis_order}, warnings={len(s.warnings)}")
            if not s.rows:
                ctx.logger.warning("Telemetry stream %s not found in GPMF data.", code)
                continue
            path = original_dir / CSV_NAMES.get(code, f"{code.lower()}.csv")
            columns = _BASE_COLUMNS + [f"c{i}" for i in range(s.channels or 0)] + [f"raw_c{i}" for i in range(s.channels or 0)]
            write_csv_atomic(path, s.rows, columns)
            files[code] = path.name
            outputs.append(path)
            ctx.logger.info("%s: %d samples (~%s Hz)", code, len(s.rows), series_metadata(s)["approx_sampling_rate_hz"])
        parser_output = original_dir / "parser_output.txt"
        write_text_atomic(parser_output, "\n".join(dump) + "\n")
        outputs.append(parser_output)

        write_json_atomic(meta_path, {
            "available": bool(files),
            "reason": None if files else "GPMF present but none of the requested streams were found.",
            "source_stream_index": stream_index,
            "parser_version": PARSER_VERSION,
            "video_start_seconds": video_start_time(info),
            "time_reference": "time_video_seconds = container time - video start; sample i of n in a packet at pts + duration*i/n",
            "channel_note": "c0..cN follow the camera's stored order (see axis_order / ORIN); values scaled by SCAL.",
            "streams": {code: series_metadata(s) for code, s in series.items()},
            "files": files,
            "recorded_at": utc_now_iso(),
        })
        return StageOutcome(outputs=tuple(outputs), details={"available": bool(files), "streams": list(files)})
