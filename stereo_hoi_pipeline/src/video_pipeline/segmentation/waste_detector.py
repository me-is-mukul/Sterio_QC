"""Select WASTE runs that last at least N seconds as removal candidates.

Only runs labelled exactly WASTE qualify. RELEVANT, UNCERTAIN, UNAVAILABLE and
MISSING samples are preserved and break WASTE runs. Every WASTE run gets a
recorded decision. Candidates are then re-checked by ``segmentation.verification``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from video_pipeline.analysis.temporal_filter import Run

WASTE = "WASTE"


@dataclass(frozen=True)
class WasteDecision:
    run: Run
    removed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.run.interval.to_dict(),
            "sample_indices": [self.run.sample_indices[0], self.run.sample_indices[-1]],
            "samples": len(self.run.sample_indices),
            "removed": self.removed,
            "reason": self.reason,
        }


def decide_waste(runs: Sequence[Run], threshold_seconds: float, tolerance: float = 1e-6) -> list[WasteDecision]:
    decisions = []
    for run in runs:
        if run.label != WASTE:
            continue
        duration = run.interval.duration
        if duration + tolerance >= threshold_seconds:
            decisions.append(WasteDecision(run, True, f"continuous WASTE {duration:.3f}s >= threshold {threshold_seconds}s"))
        else:
            decisions.append(WasteDecision(run, False, f"continuous WASTE {duration:.3f}s < threshold {threshold_seconds}s"))
    return decisions
