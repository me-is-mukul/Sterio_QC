"""Section 4 of the QA sheet: time cropping inside a passing recording (pure functions).

Everything is in frame indices of the recording (frame i of the left and right eye form a pair).

1. A sample is *idle* when every model that provides `activity` answered successfully and said idle;
   *no_hands* when every model that provides `hands` answered successfully and said none. A model
   that failed on a sample makes that sample NOT flagged, so it is kept.
2. A *cut* is a maximal run of consecutive flagged samples whose span t_last - t_first is at least
   the threshold (10 s). It removes frames [frame_first, frame_last] inclusive. Shorter runs stay.
3. Idle cuts, no-hands cuts and (optionally) black/frozen stretches are merged; the kept pieces are
   the rest of the recording.
4. Each piece is shrunk inward to keyframes shared by both eyes: its start moves forward to the next
   common keyframe, its end back to the last common keyframe before the next cut (the piece ends just
   before that keyframe). A piece that runs to the end of the recording keeps its last frame.
5. Pieces shorter than 30 s are dropped.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SampleDecision:
    index: int
    frame: int
    t_s: float
    idle: bool
    no_hands: bool
    activity_votes: dict[str, str | None]
    hands_votes: dict[str, str | None]


@dataclass(frozen=True)
class Cut:
    reason: str
    first_frame: int
    last_frame: int          # inclusive
    t_start_s: float
    t_end_s: float
    samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Piece:
    start: int               # first frame (a common keyframe)
    stop: int                # one past the last frame
    raw_start: int           # before snapping
    raw_stop: int
    duration_s: float
    start_snap_s: float
    end_snap_s: float
    kept: bool
    drop_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_samples(samples: Sequence[Mapping[str, Any]], labels: Mapping[str, Mapping[int, Mapping[str, Any]]],
                   provides: Mapping[str, Sequence[str]], *, min_models: int) -> list[SampleDecision]:
    """``samples``: dicts with index, frame, t_s. ``labels[model][sample_index]``: cached records."""
    activity_models = [m for m, roles in provides.items() if "activity" in roles]
    hands_models = [m for m, roles in provides.items() if "hands" in roles]
    if len(activity_models) < min_models or len(hands_models) < min_models:
        raise ValueError(f"need >= {min_models} models for activity ({activity_models}) and hands ({hands_models})")
    decisions = []
    for s in samples:
        def vote(model: str, field: str) -> str | None:
            record = labels.get(model, {}).get(s["index"])
            if not record or record.get("status") != "ok":
                return None
            return record["labels"].get(field)

        activity = {m: vote(m, "activity") for m in activity_models}
        hands = {m: vote(m, "hands") for m in hands_models}
        decisions.append(SampleDecision(
            index=s["index"], frame=s["frame"], t_s=s["t_s"],
            idle=all(v == "idle" for v in activity.values()),
            no_hands=all(v == "none" for v in hands.values()),
            activity_votes=activity, hands_votes=hands))
    return decisions


def flagged_runs(decisions: Sequence[SampleDecision], flag: str, min_seconds: float, reason: str) -> list[Cut]:
    """Maximal runs of consecutive samples (by sample index) with `flag` set, spanning >= min_seconds."""
    cuts: list[Cut] = []
    run: list[SampleDecision] = []

    def close() -> None:
        if run and run[-1].t_s - run[0].t_s >= min_seconds - 1e-9:
            cuts.append(Cut(reason, run[0].frame, run[-1].frame, run[0].t_s, run[-1].t_s, len(run)))

    for d in sorted(decisions, key=lambda d: d.index):
        if getattr(d, flag) and (not run or d.index == run[-1].index + 1):
            run.append(d)
            continue
        close()
        run = [d] if getattr(d, flag) else []
    close()
    return cuts


def merge_frame_ranges(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge inclusive [a, b] frame ranges (touching ranges merge too)."""
    merged: list[list[int]] = []
    for a, b in sorted(ranges):
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def kept_ranges(removed: Sequence[tuple[int, int]], n_frames: int) -> list[tuple[int, int]]:
    """Complement of inclusive removed ranges, as half-open [start, stop) ranges."""
    kept = []
    cursor = 0
    for a, b in merge_frame_ranges([(max(0, a), min(n_frames - 1, b)) for a, b in removed if b >= 0 and a < n_frames]):
        if a > cursor:
            kept.append((cursor, a))
        cursor = max(cursor, b + 1)
    if cursor < n_frames:
        kept.append((cursor, n_frames))
    return kept


def snap_piece(start: int, stop: int, keyframes: Sequence[int], n_frames: int) -> tuple[int, int] | None:
    """Shrink [start, stop) inward to common keyframes. ``keyframes`` sorted ascending."""
    i = bisect.bisect_left(keyframes, start)
    if i >= len(keyframes):
        return None
    new_start = keyframes[i]
    if stop >= n_frames:
        new_stop = n_frames
    else:
        j = bisect.bisect_right(keyframes, stop) - 1
        if j < 0:
            return None
        new_stop = keyframes[j]
    return (new_start, new_stop) if new_stop > new_start else None


def plan_pieces(*, n_frames: int, frame_times_s: Sequence[float], keyframes: Sequence[int],
                cuts: Sequence[tuple[int, int]], fps: float, min_piece_seconds: float) -> list[Piece]:
    """``frame_times_s``: seconds since the first frame, per frame (left eye)."""
    pieces = []
    for raw_start, raw_stop in kept_ranges(cuts, n_frames):
        snapped = snap_piece(raw_start, raw_stop, keyframes, n_frames)
        if snapped is None:
            pieces.append(Piece(raw_start, raw_start, raw_start, raw_stop, 0.0, 0.0, 0.0, False,
                                "no common keyframe inside the piece"))
            continue
        start, stop = int(snapped[0]), int(snapped[1])
        # float()/bool(): frame times may be numpy values; results are stored as JSON and must stay
        # real booleans/numbers (a "False" string would read as true on resume).
        duration = float(frame_times_s[stop - 1] - frame_times_s[start]) + 1.0 / fps
        start_snap = float(frame_times_s[start] - frame_times_s[raw_start])
        end_snap = float(frame_times_s[raw_stop - 1] - frame_times_s[stop - 1]) if stop < raw_stop else 0.0
        kept = bool(duration >= min_piece_seconds - 1e-9)
        pieces.append(Piece(start, stop, int(raw_start), int(raw_stop), round(duration, 4), round(start_snap, 4),
                            round(end_snap, 4), kept, None if kept else f"shorter than {min_piece_seconds} s"))
    return pieces


def defect_frame_ranges(intervals_s: Sequence[Sequence[float]], fps: float, n_frames: int) -> list[tuple[int, int]]:
    """Black/frozen intervals (stream seconds, frame n at n/fps) -> inclusive frame ranges."""
    ranges = []
    for a, b in intervals_s:
        first = max(0, int(a * fps + 1e-6))
        last = min(n_frames - 1, int(b * fps - 1e-6))
        if last >= first:
            ranges.append((first, last))
    return ranges
