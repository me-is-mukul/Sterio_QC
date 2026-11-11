"""Single entry point for the pipeline.

Usage (from the project root):
    python scripts/run_pipeline.py --url "VIDEO_URL" --waste-threshold 5
    python scripts/run_pipeline.py --input "GX010140.MP4" --waste-threshold 5

Works without installing the package; ``pip install -e .`` also provides the
equivalent ``stereo-hoi`` command.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from video_pipeline.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
