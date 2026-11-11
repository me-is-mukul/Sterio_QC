"""Start the read-only dashboard.

Usage (from the project root):
    python scripts/serve_ui.py                  # http://localhost:8000
    python scripts/serve_ui.py --host 0.0.0.0   # reachable from other machines on your network
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from video_pipeline.web.server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
