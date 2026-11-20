"""Hand & technical QC checklist - entry point.

    python run_hand_qc.py --csv ../my_clips.csv                # download, analyse, score -> outputs/my_clips/
    python run_hand_qc.py --csv ../my_clips.csv --score-only   # re-score after changing thresholds (seconds)
    python run_hand_qc.py --help                               # all options

See README.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from handqc.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
