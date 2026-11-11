"""Fine-grained verification of candidate waste sections.

A candidate ``[s, e)`` is re-sampled every ``step`` seconds. Each fine sample covers
``[t, min(t + step, e))``. Every fine sample that is not WASTE (RELEVANT, UNCERTAIN,
UNAVAILABLE, or missing) is *protected* together with ``guard`` seconds on both
sides. Only the unprotected pieces that still last >= the threshold are removed.

Result: nothing is removed within ``guard`` seconds of any moment where relevant
HOI was seen (or could not be ruled out) at the fine step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from video_pipeline.analysis.temporal_filter import Interval, complement

WASTE = "WASTE"


@dataclass(frozen=True)
class FineSample:
    timestamp: float
    interval: Interval
    label: str        # classification, or "MISSING" when no frame/result exists


@dataclass(frozen=True)
class VerifiedCandidate:
    candidate: Interval
    samples: tuple[FineSample, ...]
    protected: tuple[Interval, ...]
    removed: tuple[Interval, ...]
    kept_pieces: tuple[Interval, ...]   # unprotected but shorter than the threshold

    @property
    def outcome(self) -> str:
        if not self.removed:
            return "kept"
        if len(self.removed) == 1 and abs(self.removed[0].duration - self.candidate.duration) < 1e-6:
            return "removed"
        return "partially_removed"


def fine_timestamps(candidate: Interval, step: float) -> list[float]:
    out, k = [], 0
    while True:
        t = round(candidate.start + k * step, 6)
        if t >= candidate.end - 1e-9:
            return out
        out.append(t)
        k += 1


def verify_candidate(candidate: Interval, labels: Sequence[tuple[float, str]], *, step: float, guard: float,
                     threshold: float, tolerance: float = 1e-6) -> VerifiedCandidate:
    samples = tuple(FineSample(t, Interval(t, min(t + step, candidate.end)), label) for t, label in labels)
    protected = [Interval(max(candidate.start, s.interval.start - guard), min(candidate.end, s.interval.end + guard))
                 for s in samples if s.label != WASTE]
    pieces = complement(protected, candidate.start, candidate.end)
    removed = tuple(p for p in pieces if p.duration + tolerance >= threshold)
    kept = tuple(p for p in pieces if p.duration + tolerance < threshold)
    return VerifiedCandidate(candidate, samples, tuple(protected), removed, kept)
