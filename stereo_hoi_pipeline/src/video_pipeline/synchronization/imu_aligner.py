"""Apply the video's retained intervals to IMU samples, by timestamp.

A sample at original video time ``t`` is kept iff ``t`` lies in a retained
interval ``[original_start, original_end)``; its processed time is
``processed_start + (t - original_start)``. Original timestamps are preserved.
Samples outside the video's time range are dropped and counted.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AlignmentStats:
    input_samples: int
    kept_samples: int
    removed_samples: int
    outside_video_samples: int


def align_samples(rows: Sequence[Mapping[str, Any]], timeline: Sequence[Mapping[str, Any]], *,
                  duration: float, time_key: str = "time_video_seconds") -> tuple[list[dict[str, Any]], AlignmentStats]:
    starts = [seg["original_start"] for seg in timeline]
    kept: list[dict[str, Any]] = []
    removed = outside = 0
    for row in rows:
        t = float(row[time_key])
        if t < 0 or t >= duration:
            outside += 1
            continue
        i = bisect.bisect_right(starts, t) - 1
        if i >= 0 and t < timeline[i]["original_end"]:
            seg = timeline[i]
            kept.append({**row,
                         "segment_index": seg["segment_index"],
                         "time_processed_seconds": round(seg["processed_start"] + (t - seg["original_start"]), 6)})
        else:
            removed += 1
    return kept, AlignmentStats(len(rows), len(kept), removed, outside)
