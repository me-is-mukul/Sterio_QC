"""Section 4 of the QA sheet: time cropping inside a passing recording (pure functions).

Everything is in frame indices of the recording (frame i of the left and right eye form a pair).

1. A sample is *idle* when every model that provides `activity` answered successfully and said idle;
   *no_hands* when every model that provides `hands` answered successfully and said none. A model
   that failed on a sample makes that sample NOT flagged, so it is kept. With the content screen on,
   the content model's flags listed in content.cut_as_stretches (phone use, minor, private moment,
   lens blocked, headset not worn) are reasons too.
2. A *cut* is a maximal run of consecutive samples flagged for ANY reason whose span t_last - t_first
   is at least the threshold (10 s). It removes frames [frame_first, frame_last] inclusive. Shorter
   runs stay. (As the reference QA does: 8 s lens blocked followed by 8 s without hands is one cut.)
3. Cuts and (optionally) black/frozen stretches are merged; the kept pieces are the rest of the
   recording, split where frames were lost (a gap; no piece runs across missing video).
4. Each piece is shrunk inward to keyframes shared by both eyes: its start moves forward to the next
   common keyframe, its end back to the last common keyframe before the next cut (the piece ends just
   before that keyframe). A piece that runs to the end of the recording keeps its last frame.
5. Pieces shorter than 30 s are dropped.
6. ``removed_stretches``: every removed stretch with the share of its samples per reason (for the
   reference-format sheets); a sample in a dropped short piece counts as ``short_piece``.
"""

from __future__ import annotations

import bisect
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

# Why a sample is cut, most important first: a sample flagged for several reasons counts under the first.
REASONS = ("minor_visible", "private_moment", "phone_use", "third_person_or_fixed_camera",
           "lens_covered_or_smudged", "idle", "no_hands")
# Reasons of removed footage that are not per-sample flags.
SHORT_PIECE = "short_piece"            # a piece under min_piece_seconds between cuts
DEFECT = "black_or_frozen"             # black / frozen stretch found in section 1
CONTENT_FLAGS = tuple(r for r in REASONS if r not in {"idle", "no_hands"})


@dataclass(frozen=True)
class SampleDecision:
    index: int
    frame: int
    t_s: float
    idle: bool
    no_hands: bool
    activity_votes: dict[str, str | None]
    hands_votes: dict[str, str | None]
    content: dict[str, bool] = field(default_factory=dict)   # content-model flags that cut (see REASONS)

    @property
    def reasons(self) -> list[str]:
        flags = {**self.content, "idle": self.idle, "no_hands": self.no_hands}
        return [r for r in REASONS if flags.get(r)]


@dataclass(frozen=True)
class Cut:
    reason: str              # the reason of most of its samples
    first_frame: int
    last_frame: int          # inclusive
    t_start_s: float
    t_end_s: float
    samples: int
    reasons: dict[str, int] = field(default_factory=dict)    # samples per (first) reason

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
                   provides: Mapping[str, Sequence[str]], *, min_models: int, content_model: str | None = None,
                   content_fields: Sequence[str] = ()) -> list[SampleDecision]:
    """``samples``: dicts with index, frame, t_s. ``labels[model][sample_index]``: cached records.
    ``content_fields``: flags of ``content_model`` that make a sample cut (a failed answer flags nothing)."""
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
        content = {f: vote(content_model, f) is True for f in content_fields} if content_model else {}
        decisions.append(SampleDecision(
            index=s["index"], frame=s["frame"], t_s=s["t_s"],
            idle=all(v == "idle" for v in activity.values()),
            no_hands=all(v == "none" for v in hands.values()),
            activity_votes=activity, hands_votes=hands, content=content))
    return decisions


def _make_cut(run: Sequence[SampleDecision]) -> Cut:
    counts = Counter(d.reasons[0] for d in run)
    main = max(counts, key=lambda r: (counts[r], -REASONS.index(r)))      # ties: the more important reason
    return Cut(main, run[0].frame, run[-1].frame, run[0].t_s, run[-1].t_s, len(run), dict(counts))


def reason_runs(decisions: Sequence[SampleDecision], min_seconds: float) -> list[Cut]:
    """Maximal runs of consecutive samples (by sample index) flagged for any reason, spanning
    >= min_seconds. The cut's reason is the reason of most of its samples."""
    cuts: list[Cut] = []
    run: list[SampleDecision] = []

    def close() -> None:
        if run and run[-1].t_s - run[0].t_s >= min_seconds - 1e-9:
            cuts.append(_make_cut(run))

    for d in sorted(decisions, key=lambda d: d.index):
        if d.reasons and (not run or d.index == run[-1].index + 1):
            run.append(d)
            continue
        close()
        run = [d] if d.reasons else []
    close()
    return cuts


def flagged_runs(decisions: Sequence[SampleDecision], flag: str, min_seconds: float, reason: str) -> list[Cut]:
    """Maximal runs of consecutive samples (by sample index) with one `flag` set, spanning >= min_seconds."""
    cuts: list[Cut] = []
    run: list[SampleDecision] = []

    def close() -> None:
        if run and run[-1].t_s - run[0].t_s >= min_seconds - 1e-9:
            cuts.append(Cut(reason, run[0].frame, run[-1].frame, run[0].t_s, run[-1].t_s, len(run),
                            {reason: len(run)}))

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


def split_ranges(ranges: Sequence[tuple[int, int]], splits: Sequence[int]) -> list[tuple[int, int]]:
    """Split half-open [start, stop) ranges at each split row strictly inside them (a split row starts
    a new range: the row after a gap of lost frames)."""
    out = []
    cuts = sorted(set(int(s) for s in splits))
    for start, stop in ranges:
        inner = [s for s in cuts if start < s < stop]
        bounds = [start, *inner, stop]
        out.extend((a, b) for a, b in zip(bounds, bounds[1:]))
    return out


def plan_pieces(*, n_frames: int, frame_times_s: Sequence[float], keyframes: Sequence[int],
                cuts: Sequence[tuple[int, int]], fps: float, min_piece_seconds: float,
                splits: Sequence[int] = ()) -> list[Piece]:
    """``frame_times_s``: seconds since the first frame, per frame (left eye). ``splits``: rows that
    must start a new piece (after a gap of lost frames)."""
    pieces = []
    for raw_start, raw_stop in split_ranges(kept_ranges(cuts, n_frames), splits):
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


def removed_stretches(*, n_frames: int, frame_times_s: Sequence[float], fps: float,
                      kept: Sequence[tuple[int, int]], samples: Sequence[tuple[int, Sequence[str]]],
                      cuts: Sequence[Mapping[str, Any]], defects: Sequence[Sequence[int]] = (),
                      dropped: Sequence[tuple[int, int, str]] = ()) -> list[dict[str, Any]]:
    """Every stretch of the recording that is not in a kept piece, with the share of its samples per reason.

    ``kept``: final kept pieces [start, stop). ``samples``: (frame, reasons) per labelling sample.
    ``cuts``: Cut dicts (first_frame/last_frame inclusive). ``defects``: inclusive black/frozen frame
    ranges. ``dropped``: (start, stop, reason) of pieces the content screen dropped. A sample in a cut
    counts under its own first reason (else the cut's); otherwise under the dropped piece's reason, the
    defect, or ``short_piece`` (a piece under the minimum length between cuts). A stretch without a
    sample (a sliver from keyframe snapping) gets the reason of its middle frame."""
    def reason_at(frame: int, own: Sequence[str] = ()) -> str:
        for c in cuts:
            if c["first_frame"] <= frame <= c["last_frame"]:
                return own[0] if own else c["reason"]
        for a, b, why in dropped:
            if a <= frame < b:
                return why
        if any(a <= frame <= b for a, b in defects):
            return DEFECT
        return SHORT_PIECE

    removed = []
    cursor = 0
    for start, stop in sorted((int(a), int(b)) for a, b in kept):
        if start > cursor:
            removed.append((cursor, start))
        cursor = max(cursor, stop)
    if cursor < n_frames:
        removed.append((cursor, n_frames))
    out = []
    for a, b in removed:
        counts = Counter(reason_at(f, r) for f, r in samples if a <= f < b)
        if not counts:
            counts[reason_at((a + b - 1) // 2)] = 1
        total = sum(counts.values())
        order = sorted(counts, key=lambda r: (-counts[r], REASONS.index(r) if r in REASONS else len(REASONS)))
        end = float(frame_times_s[b - 1]) + 1.0 / fps
        out.append({"first_frame": a, "stop_frame": b, "start_s": round(float(frame_times_s[a]), 3),
                    "end_s": round(end, 3), "length_s": round(end - float(frame_times_s[a]), 3),
                    "samples": total if sum(1 for f, _ in samples if a <= f < b) else 0,
                    "main_reason": order[0], "reasons": {r: round(counts[r] / total, 4) for r in order}})
    return out


def defect_frame_ranges(intervals_s: Sequence[Sequence[float]], fps: float, n_frames: int) -> list[tuple[int, int]]:
    """Black/frozen intervals (stream seconds, frame n at n/fps) -> inclusive frame ranges."""
    ranges = []
    for a, b in intervals_s:
        first = max(0, int(a * fps + 1e-6))
        last = min(n_frames - 1, int(b * fps - 1e-6))
        if last >= first:
            ranges.append((first, last))
    return ranges
