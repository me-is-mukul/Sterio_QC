"""Read-only dashboard server.

Serves one static page plus a small JSON API over the artifacts in the output
directory. It never runs or modifies the pipeline. Standard library only.

    GET /                          dashboard page
    GET /api/runs                  list of workspaces with status
    GET /api/runs/<name>           run overview (manifest, metrics, timeline, ...)
    GET /api/runs/<name>/samples   per-sample frames + classifications
    GET /files/<name>/<path>       artifact files (images, video with Range support)
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from video_pipeline.core.config import find_project_root, load_configuration
from video_pipeline.core.env import load_environment
from video_pipeline.core.errors import ConfigError

STATIC_DIR = Path(__file__).parent / "static"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_SERVABLE = {".jpg", ".jpeg", ".png", ".mp4", ".json", ".jsonl", ".csv", ".log", ".yaml", ".txt"}
_CHUNK = 1024 * 1024


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_jsonl(path: Path) -> list[Any]:
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        break  # a line being written right now
    except OSError:
        pass
    return rows


class _ExclusiveServer(ThreadingHTTPServer):
    """Refuse to share the port. On Windows, SO_REUSEADDR (the http.server default) lets a
    second server silently bind the same port, so two dashboards would answer at random."""

    allow_reuse_address = sys.platform != "win32"
    daemon_threads = True

    def server_bind(self) -> None:
        if sys.platform == "win32":
            import socket

            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class Dashboard:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root.resolve()

    def workspace(self, name: str) -> Path | None:
        if not _SAFE_NAME.match(name):
            return None
        path = (self.output_root / name).resolve()
        return path if path.parent == self.output_root and (path / "reports" / "manifest.json").is_file() else None

    def runs(self) -> list[dict[str, Any]]:
        items = []
        if self.output_root.is_dir():
            for path in self.output_root.iterdir():
                manifest = _read_json(path / "reports" / "manifest.json") if path.is_dir() else None
                if not manifest:
                    continue
                summary = _read_json(path / "reports" / "summary.json") or {}
                items.append({"name": path.name, "run_id": manifest.get("run_id"), "status": manifest.get("status"),
                              "updated_at": manifest.get("updated_at"), "headline": summary.get("headline")})
        return sorted(items, key=lambda r: r.get("updated_at") or "", reverse=True)

    def overview(self, ws: Path) -> dict[str, Any]:
        manifest = _read_json(ws / "reports" / "manifest.json") or {}
        info = _read_json(ws / "metadata" / "video_info.json") or {}
        probe = _read_json(ws / "metadata" / "ffprobe.json") or {}
        config = manifest.get("configuration", {})
        primary = next((v for v in info.get("video_streams", []) if v.get("index") == info.get("primary_video_stream_index")), {})
        stages = [{"name": n, **{k: (manifest.get("stages", {}).get(n) or {}).get(k) for k in ("status", "started_at", "finished_at", "error")},
                   "elapsed": ((manifest.get("stages", {}).get(n) or {}).get("details") or {}).get("elapsed_seconds")}
                  for n in manifest.get("planned_stages", [])]
        inference_meta = _read_json(ws / "inference" / "inference_metadata.json") or {}
        inference_meta.pop("provider", None)  # contains the endpoint host
        processing = _read_json(ws / "processed" / "processing_timeline.json")
        return {
            "name": ws.name,
            "run": {k: manifest.get(k) for k in ("run_id", "status", "created_at", "updated_at", "finished_at", "pipeline_version")},
            "source": {k: (manifest.get("source") or {}).get(k) for k in ("filename", "sha256", "size_bytes", "duration_seconds")},
            "config": {k: config.get(k) for k in ("waste_threshold_seconds", "sampling_interval", "model", "prompt_version",
                                                  "uncertain_policy", "model_digest")},
            "versions": {k: v for k, v in (manifest.get("versions") or {}).items() if k != "platform"},
            "stages": stages,
            "video": {"width": primary.get("width"), "height": primary.get("height"), "codec": primary.get("codec_name"),
                      "fps": primary.get("fps_average"), "duration": info.get("duration_seconds"),
                      "audio_streams": len(info.get("audio_streams", [])),
                      "gpmf": (info.get("telemetry") or {}).get("gpmf_stream_present")},
            "chapters": [{"start": float(c.get("start_time", 0)), "end": float(c.get("end_time", 0)),
                          "title": (c.get("tags") or {}).get("title", "")} for c in probe.get("chapters", [])],
            "sampling": _read_json(ws / "frames" / "sampling_metadata.json"),
            "inference": inference_meta or None,
            "timeline": _read_json(ws / "segmentation" / "classifications_timeline.json"),
            "waste": _read_json(ws / "segmentation" / "waste_intervals.json"),
            "processing": processing,
            "imu": _read_json(ws / "imu" / "original" / "imu_metadata.json"),
            "imu_sync": _read_json(ws / "imu" / "processed" / "synchronization_metadata.json"),
            "verification": _read_json(ws / "segmentation" / "verification.json"),
            "qc": _read_json(ws / "reports" / "qc_report.json"),
            "metrics": _read_json(ws / "reports" / "metrics.json"),
            "summary": _read_json(ws / "reports" / "summary.json"),
            "progress": {"frames": len(_read_jsonl(ws / "frames" / "frame_index.jsonl")),
                         "classified": len(_read_jsonl(ws / "inference" / "classifications.jsonl"))},
        }

    def samples(self, ws: Path) -> list[dict[str, Any]]:
        frames = {f["sample_index"]: f for f in _read_jsonl(ws / "frames" / "frame_index.jsonl")}
        results = {c["sample_index"]: c for c in _read_jsonl(ws / "inference" / "classifications.jsonl")}
        out = []
        for k in sorted(frames):
            f, c = frames[k], results.get(k, {})
            out.append({
                "sample_index": k, "t": f["target_timestamp_seconds"], "pts": f["frame_pts_seconds"],
                "left": f["left"], "right": f["right"],
                "has_images": (ws / f["left"]).is_file() and (ws / f["right"]).is_file(),
                "label": c.get("classification"), "status": c.get("status"),
                "description": c.get("hoi_description"), "evidence": c.get("evidence"),
                "left_camera": c.get("left_camera"), "right_camera": c.get("right_camera"),
                "latency": c.get("latency_seconds"), "cached": c.get("cached"),
                "errors": c.get("validation_errors"),
            })
        return out


def make_handler(dashboard: Dashboard) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "StereoHOIDashboard/1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet
            pass

        def _json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _not_found(self) -> None:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_GET(self) -> None:  # noqa: N802
            path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
            parts = [p for p in path.split("/") if p]
            try:
                if not parts:
                    return self._file(STATIC_DIR / "index.html")
                if parts == ["api", "runs"]:
                    return self._json(dashboard.runs())
                if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
                    ws = dashboard.workspace(parts[2])
                    if ws is None:
                        return self._not_found()
                    if len(parts) == 3:
                        return self._json(dashboard.overview(ws))
                    if parts[3:] == ["samples"]:
                        return self._json(dashboard.samples(ws))
                if len(parts) >= 3 and parts[0] == "files":
                    ws = dashboard.workspace(parts[1])
                    if ws is None:
                        return self._not_found()
                    target = (ws / Path(*parts[2:])).resolve()
                    if ws not in target.parents or target.suffix.lower() not in _SERVABLE or not target.is_file():
                        return self._not_found()
                    return self._file(target)
                return self._not_found()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _file(self, path: Path) -> None:
            size = path.stat().st_size
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            start, end = 0, size - 1
            range_header = self.headers.get("Range")
            status = HTTPStatus.OK
            if range_header:
                m = re.match(r"bytes=(\d*)-(\d*)$", range_header.strip())
                if not m or (not m.group(1) and not m.group(2)):
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                if m.group(1):
                    start = int(m.group(1))
                    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
                else:  # suffix range: last N bytes
                    start = max(0, size - int(m.group(2)))
                if start > end or start >= size:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-cache")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = handle.read(min(_CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="serve_ui", description="Read-only dashboard for pipeline outputs.")
    parser.add_argument("--host", help="Bind address (default from config web.host). Use 0.0.0.0 to share on your network.")
    parser.add_argument("--port", type=int, help="Port (default from config web.port).")
    parser.add_argument("--output-dir", type=Path, help="Output root to browse (default: project.output_directory).")
    args = parser.parse_args(argv)
    try:
        root = find_project_root()
        config = load_configuration(root, environment=load_environment(root))
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if args.output_dir is not None:
        output = args.output_dir.resolve()
    else:
        configured = Path(config.get("project.output_directory"))
        output = configured if configured.is_absolute() else root / configured
    host = args.host or config.get("web.host")
    port = args.port or config.get("web.port")
    try:
        server = _ExclusiveServer((host, port), make_handler(Dashboard(output)))
    except OSError as exc:
        print(f"Cannot start the dashboard on {host}:{port}: {exc}. Is another dashboard already running? "
              f"Use that one, or pick another port with --port.", file=sys.stderr)
        return 1
    shown = "localhost" if host in {"127.0.0.1", "0.0.0.0"} else host
    print(f"Dashboard: http://{shown}:{port}   (browsing {output})   Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
