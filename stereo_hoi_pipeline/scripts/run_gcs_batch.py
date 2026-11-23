"""Egocentric video QA for MCAP recordings in Google Cloud Storage (QA sheet sections 1-4).

Usage (from stereo_hoi_pipeline/):
    python scripts/run_gcs_batch.py --list-sessions gs://op-datoric/raw/Op_DD/
    python scripts/run_gcs_batch.py --sessions-file sessions_10.txt
    python scripts/run_gcs_batch.py --session gs://op-datoric/raw/Op_DD/<date>/<device>/<session>/ --no-upload

Settings: configs/ego_qa.yaml. See docs/ego_qa.md.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from video_pipeline.ego_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
