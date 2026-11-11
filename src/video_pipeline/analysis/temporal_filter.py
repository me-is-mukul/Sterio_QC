"""Time intervals and continuous runs of labels.

All intervals are half-open ``[start, end)`` in seconds on the original video
timeline (relative to the video start). Durations always come from timestamps,
never from row counts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def overlap(self, other: Interval) -> float:
        return max(0.0, min(self.end, other.end) - max(self.start, other.start))

    def to_dict(self) -> dict[str, float]:
        return {"start": round(self.start, 6), "end": round(self.end, 6), "duration": round(self.duration, 6)}


@dataclass(frozen=True)
class TimedLabel:
    sample_index: int
    interval: Interval
    label: str


@dataclass(frozen=True)
class Run:
    label: str
    interval: Interval
    sample_indices: tuple[int, ...]


def contiguous_runs(labels: Sequence[TimedLabel], tolerance: float = 1e-6) -> list[Run]:
    """Merge consecutive, touching intervals that share a label. Gaps always split runs."""
    runs: list[Run] = []
    for item in sorted(labels, key=lambda t: t.interval.start):
        if runs:
            last = runs[-1]
            if last.label == item.label and abs(last.interval.end - item.interval.start) <= tolerance:
                runs[-1] = Run(last.label, Interval(last.interval.start, item.interval.end),
                               last.sample_indices + (item.sample_index,))
                continue
        runs.append(Run(item.label, item.interval, (item.sample_index,)))
    return runs


def merge_intervals(intervals: Iterable[Interval], tolerance: float = 1e-6) -> list[Interval]:
    merged: list[Interval] = []
    for iv in sorted(intervals, key=lambda i: i.start):
        if merged and iv.start <= merged[-1].end + tolerance:
            merged[-1] = Interval(merged[-1].start, max(merged[-1].end, iv.end))
        else:
            merged.append(iv)
    return merged


def complement(removed: Sequence[Interval], start: float, end: float) -> list[Interval]:
    """Parts of [start, end) not covered by ``removed``."""
    kept: list[Interval] = []
    cursor = start
    for iv in merge_intervals(removed):
        if iv.start > cursor:
            kept.append(Interval(cursor, min(iv.start, end)))
        cursor = max(cursor, iv.end)
    if cursor < end:
        kept.append(Interval(cursor, end))
    return [k for k in kept if k.duration > 0]
