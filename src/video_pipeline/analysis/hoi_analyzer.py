"""Build the chronological HOI timeline from per-sample classifications.

Each sample ``k`` (taken at ``k * interval``) represents ``[k*interval, (k+1)*interval)``,
clipped to the video duration. This is an approximation: one frame per interval does
not prove the whole interval has that label (plan §9.4). Samples with no frame or no
classification are labelled MISSING, which is never treated as WASTE.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from video_pipeline.analysis.temporal_filter import Interval, TimedLabel

MISSING = "MISSING"


def expected_sample_count(duration: float, interval: float) -> int:
    return max(0, math.ceil(duration / interval - 1e-6))


def build_timeline(classifications: Sequence[Mapping[str, Any]], *, interval: float, duration: float) -> list[TimedLabel]:
    by_index = {int(c["sample_index"]): str(c["classification"]) for c in classifications}
    timeline: list[TimedLabel] = []
    for k in range(expected_sample_count(duration, interval)):
        start = k * interval
        end = min((k + 1) * interval, duration)
        timeline.append(TimedLabel(k, Interval(start, end), by_index.get(k, MISSING)))
    return timeline


def label_durations(timeline: Sequence[TimedLabel], within: Sequence[Interval] | None = None) -> dict[str, float]:
    """Total duration per label, optionally only the parts inside ``within`` intervals."""
    totals: dict[str, float] = {}
    for item in timeline:
        if within is None:
            d = item.interval.duration
        else:
            d = sum(item.interval.overlap(w) for w in within)
        totals[item.label] = totals.get(item.label, 0.0) + d
    return {k: round(v, 6) for k, v in totals.items()}
