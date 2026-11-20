"""Pre-flight check of the vision-language model endpoint (uses .env + configs).

Needed only for --detector vlm / --verify-with vlm; the default MediaPipe detector runs locally.

Runs, in order:
  1. reachability      (GET /health for OpenAI-compatible servers, /api/version for Ollama)
  2. auth + model      (the provider's model check: API key accepted, model listed)
  3. real request      (one stereo pair from the video: 2 images + JSON-schema answer)

Usage (from the stereo_hoi_pipeline folder):
    python scripts/check_endpoint.py --input <video file>            # a path, or a file name in videos/
    python scripts/check_endpoint.py --input <video file> --at 42
Exit code 0 = ready to run the pipeline.
"""

import argparse
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from video_pipeline.core.config import find_project_root, load_configuration  # noqa: E402
from video_pipeline.core.env import load_environment  # noqa: E402
from video_pipeline.core.errors import PipelineError  # noqa: E402
from video_pipeline.core.tools import resolve_executable, run_command  # noqa: E402
from video_pipeline.inference.prompts import get_prompt  # noqa: E402
from video_pipeline.inference.qwen_classifier import StereoHoiClassifier  # noqa: E402
from video_pipeline.metadata.ffprobe import probe_video, summarize_probe  # noqa: E402
from video_pipeline.orchestration.stages.pairs import create_provider  # noqa: E402
from video_pipeline.stereo.stereo_splitter import compute_geometry  # noqa: E402


def step(n: int, title: str) -> None:
    print(f"\n[{n}] {title}")


def reachability(config) -> bool:
    base = config.get("inference.base_url").rstrip("/")
    if config.get("inference.provider") == "ollama":
        url = base + "/api/version"
    else:
        url = (base.rsplit("/v1", 1)[0] if base.endswith("/v1") else base) + "/health"
    started = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers={"ngrok-skip-browser-warning": "true"})
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 (configured URL)
            print(f"    OK  {url} -> HTTP {resp.status} in {time.perf_counter() - started:.2f}s")
            return True
    except urllib.error.HTTPError as exc:
        print(f"    FAIL {url} -> HTTP {exc.code}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"    FAIL {url} -> {getattr(exc, 'reason', exc)}")
        print("         No answer at all: the server is down, or a firewall blocks the port "
              "(e.g. a GCP VPC firewall rule for tcp:8000 is missing).")
    return False


def extract_pair(config, video: Path, at: float, out_dir: Path) -> tuple[bytes, bytes]:
    ffmpeg = resolve_executable(config.get("tools.ffmpeg"))
    ffprobe = resolve_executable(config.get("tools.ffprobe"))
    info = summarize_probe(probe_video(ffprobe, video, count_frames=False, timeout_seconds=60).raw)
    stream = next(v for v in info["video_streams"] if v["index"] == info["primary_video_stream_index"])
    geometry = compute_geometry(stream["width"], stream["height"], layout=config.get("stereo.layout"),
                                split_ratio=config.get("stereo.split_ratio"))
    images = []
    for name, crop in (("left", geometry.left), ("right", geometry.right)):
        path = out_dir / f"{name}.jpg"
        result = run_command([ffmpeg, "-v", "error", "-y", "-ss", f"{at}", "-i", str(video), "-frames:v", "1",
                              "-vf", f"{crop.ffmpeg_filter()},scale=w='min({config.get('sampling.max_width')},iw)':h=-2",
                              "-q:v", "3", str(path)], timeout_seconds=60)
        if result.returncode != 0:
            raise PipelineError(f"Could not extract the {name} frame: {result.stderr.strip()}")
        images.append(path.read_bytes())
    return images[0], images[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that the configured model endpoint is ready.")
    parser.add_argument("--input", type=Path, required=True,
                        help="Video to take a test pair from (a path, or a file name in videos/).")
    parser.add_argument("--at", type=float, default=10.0, help="Timestamp (s) of the test pair.")
    args = parser.parse_args()

    root = find_project_root()
    env = load_environment(root)
    config = load_configuration(root, environment=env)
    print(f"Provider: {config.get('inference.provider')}  |  model: {config.get('inference.model')}  |  "
          f"endpoint: {config.get('inference.base_url')}")

    step(1, "Reachability")
    if not reachability(config):
        return 1

    step(2, "Authentication and model")
    provider = create_provider(config, env)
    try:
        info = provider.model_info()
    except PipelineError as exc:
        print(f"    FAIL {exc}")
        return 1
    extra = ", ".join(f"{k}={v}" for k, v in dict(info.details).items())
    print(f"    OK  model '{info.model}' served (runtime {info.runtime_version or 'n/a'}{'; ' + extra if extra else ''})")

    step(3, f"Real stereo-pair request ({args.input.name} @ {args.at:g}s)")
    video = args.input if args.input.is_absolute() else Path.cwd() / args.input
    if not video.is_file():
        video = root / config.get("video.download_directory") / args.input
    if not video.is_file():
        print(f"    FAIL video not found: {video}")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        left, right = extract_pair(config, video, args.at, Path(tmp))
    classifier = StereoHoiClassifier(provider, get_prompt(config.get("inference.prompt_version")),
                                     temperature=config.get("inference.temperature"))
    try:
        result, response = classifier.classify(left, right, timestamp_seconds=args.at)
    except PipelineError as exc:
        print(f"    FAIL {exc}")
        if "limit" in str(exc).lower() and "image" in str(exc).lower():
            print("         The server must accept 2 images per request: vLLM --limit-mm-per-prompt '{\"image\": 2}'.")
        return 1
    print(f"    {'OK ' if result.status == 'ok' else 'WARN'} {result.classification} in {response.latency_seconds:.2f}s "
          f"(status {result.status})")
    print(f"        description: {result.hoi_description}")
    if result.left_camera and result.right_camera:
        print(f"        left : {result.left_camera.description}")
        print(f"        right: {result.right_camera.description}")
    if result.validation_errors:
        print(f"        issues: {'; '.join(result.validation_errors)}")
        print(f"        raw answer: {response.text[:300]!r}")
    if result.status != "ok":
        return 1
    print("\nReady: run the pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
