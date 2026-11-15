"""Timeline helpers for boolean per-frame series sampled every `dt` seconds."""

from __future__ import annotations

from typing import Sequence


def runs(series: Sequence[bool]) -> list[tuple[bool, int, int]]:
    """Run-length encoding: (value, start_index, length)."""
    out: list[tuple[bool, int, int]] = []
    for i, v in enumerate(series):
        v = bool(v)
        if out and out[-1][0] == v:
            value, start, length = out[-1]
            out[-1] = (value, start, length + 1)
        else:
            out.append((v, i, 1))
    return out


def pct(series: Sequence[bool]) -> float:
    return round(100.0 * sum(1 for v in series if v) / len(series), 2) if series else 0.0


def bridge_gaps(series: Sequence[bool], max_gap_frames: int) -> list[bool]:
    """Fill False runs of <= max_gap_frames that lie between two True runs."""
    out = [bool(v) for v in series]
    if max_gap_frames <= 0:
        return out
    r = runs(out)
    for k, (value, start, length) in enumerate(r):
        if not value and 0 < k < len(r) - 1 and length <= max_gap_frames:
            for i in range(start, start + length):
                out[i] = True
    return out


def events(series: Sequence[bool], dt: float, *, min_duration_s: float, max_gap_s: float) -> tuple[list[dict], list[bool]]:
    """Grasp-style events: bridge short gaps, keep True runs lasting >= min_duration_s.

    Returns (events, mask) where mask marks the frames inside a qualifying event.
    Durations count each analysed frame as dt seconds.
    """
    bridged = bridge_gaps(series, int(max_gap_s / dt + 1e-9))
    mask = [False] * len(bridged)
    found = []
    for value, start, length in runs(bridged):
        duration = length * dt
        if value and duration + 1e-9 >= min_duration_s:
            found.append({"start_s": round(start * dt, 3), "duration_s": round(duration, 3)})
            for i in range(start, start + length):
                mask[i] = True
    return found, mask


def tracking_losses(present: Sequence[bool], dt: float, *, min_tracked_s: float, min_lost_s: float) -> list[dict]:
    """A loss = the hand was tracked for >= min_tracked_s and then missing for >= min_lost_s."""
    r = runs(present)
    losses = []
    for k in range(len(r) - 1):
        (v1, s1, l1), (v2, s2, l2) = r[k], r[k + 1]
        if v1 and not v2 and l1 * dt + 1e-9 >= min_tracked_s and l2 * dt + 1e-9 >= min_lost_s:
            losses.append({"lost_at_s": round(s2 * dt, 3), "missing_for_s": round(l2 * dt, 3)})
    return losses


def longest_false_run_s(series: Sequence[bool], dt: float) -> float:
    longest = max((length for value, _, length in runs(series) if not value), default=0)
    return round(longest * dt, 3)
