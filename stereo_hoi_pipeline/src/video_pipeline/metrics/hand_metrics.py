"""Hand metrics of the camera wearer, from the MediaPipe classifications (one record per sample).

Every sample stands for one sampling interval. "Before" uses all samples of the original video;
"after" uses the samples whose time lies inside a retained interval, i.e. what is left in the
processed video. Both eyes are combined: a hand counts when it is seen in either eye.

    either_hand_pct      % of samples with at least one wearer hand
    both_hands_pct       % of samples with both wearer hands
    left/right_hand_pct  % of samples with the wearer's left / right hand
    untracked_pct        % of samples with no wearer hand (100 - either_hand_pct)
    hoi_grasp_pct        % of samples where a wearer hand is grasping (hand-object interaction)
    grasp_count          continuous grasp runs, counted per hand
    grasp_seconds        total time with a grasping hand (per hand, summed)
    tracking_losses      times a hand seen in one sample is missing in the next, per hand
    longest_hand_gap_s   longest stretch with no wearer hand
    other_hands_pct      % of samples showing another person's hand
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from video_pipeline.analysis.temporal_filter import Interval

DEFINITIONS = {
    "either_hand_pct": "% of samples with at least one of the wearer's hands (either eye)",
    "both_hands_pct": "% of samples with both of the wearer's hands",
    "left_hand_pct": "% of samples with the wearer's left hand",
    "right_hand_pct": "% of samples with the wearer's right hand",
    "untracked_pct": "% of samples with no wearer hand",
    "hoi_grasp_pct": "% of samples where a wearer hand is grasping (hand-object interaction)",
    "grasp_count": "continuous grasp runs, counted per hand",
    "grasp_seconds": "total grasp time, per hand, summed",
    "tracking_losses": "times a hand seen in one sample is missing in the next (per hand)",
    "tracking_losses_per_min": "tracking losses per minute of analysed video",
    "longest_hand_gap_s": "longest stretch with no wearer hand, in seconds",
    "other_hands_pct": "% of samples showing another person's hand",
    "left_eye_either_pct": "either_hand_pct using the left eye only",
    "right_eye_either_pct": "either_hand_pct using the right eye only",
}


def _pct(n: int, total: int) -> float | None:
    return round(100.0 * n / total, 2) if total else None


def _runs(flags: Sequence[bool]) -> list[int]:
    """Lengths of the True runs."""
    out, run = [], 0
    for f in flags:
        if f:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return out


def _eye_has_wearer(record: Mapping[str, Any], eye: str) -> bool:
    return any(h.get("role") == "wearer" for h in (record.get("hands") or {}).get(eye, []))


def summarise(records: Sequence[Mapping[str, Any]], interval: float) -> dict[str, Any]:
    n = len(records)
    s = [r["hand_summary"] for r in records]
    left = [x["left"] for x in s]
    right = [x["right"] for x in s]
    either = [a or b for a, b in zip(left, right)]
    both = [a and b for a, b in zip(left, right)]
    g_left = [x["grasp_left"] for x in s]
    g_right = [x["grasp_right"] for x in s]
    grasp = [a or b for a, b in zip(g_left, g_right)]
    losses = sum(1 for seq in (left, right) for a, b in zip(seq, seq[1:]) if a and not b)
    minutes = n * interval / 60.0
    grasp_runs = _runs(g_left) + _runs(g_right)
    return {
        "samples": n,
        "analysed_seconds": round(n * interval, 3),
        "either_hand_pct": _pct(sum(either), n),
        "both_hands_pct": _pct(sum(both), n),
        "left_hand_pct": _pct(sum(left), n),
        "right_hand_pct": _pct(sum(right), n),
        "untracked_pct": _pct(n - sum(either), n),
        "hoi_grasp_pct": _pct(sum(grasp), n),
        "grasp_count": len(grasp_runs),
        "grasp_seconds": round(sum(grasp_runs) * interval, 3),
        "tracking_losses": losses,
        "tracking_losses_per_min": round(losses / minutes, 2) if minutes else None,
        "longest_hand_gap_s": round(max(_runs([not e for e in either]), default=0) * interval, 3),
        "other_hands_pct": _pct(sum(1 for x in s if x["other_hands"]), n),
        "left_eye_either_pct": _pct(sum(_eye_has_wearer(r, "left_eye") for r in records), n),
        "right_eye_either_pct": _pct(sum(_eye_has_wearer(r, "right_eye") for r in records), n),
    }


def compute_hand_metrics(records: Sequence[Mapping[str, Any]], *, interval: float,
                         retained: Sequence[Interval]) -> dict[str, Any] | None:
    """None when the records carry no hand data (e.g. the vision-language detector was used)."""
    rows = sorted((r for r in records if r.get("hand_summary")), key=lambda r: r["timestamp_seconds"])
    if not rows:
        return None
    kept = [r for r in rows if any(w.start <= r["timestamp_seconds"] < w.end for w in retained)]
    before, after = summarise(rows, interval), summarise(kept, interval) if kept else None
    return {
        "sampling_interval_seconds": interval,
        "before": before,
        "after": after,
        "definitions": DEFINITIONS,
        "notes": [
            "Each sample stands for one sampling interval; both eyes are combined.",
            "'After' uses the samples inside the retained intervals (the processed video); a gap across a cut "
            "is measured as if the retained pieces were joined.",
            "Hands that MediaPipe cannot detect (e.g. gloved, covered in clay, heavily occluded) count as absent.",
        ],
    }


# rows added to the before/after comparison table (metrics.csv)
COMPARISON_ROWS = (("either_hand_pct", "hands_in_frame_percent"), ("both_hands_pct", "both_hands_percent"),
                   ("hoi_grasp_pct", "hand_object_interaction_percent"),
                   ("longest_hand_gap_s", "longest_hand_gap_seconds"),
                   ("tracking_losses_per_min", "tracking_losses_per_minute"),
                   ("other_hands_pct", "other_peoples_hands_percent"))


def comparison_rows(hand: Mapping[str, Any]) -> list[dict[str, Any]]:
    after = hand.get("after") or {}
    return [{"metric": name, "before": hand["before"].get(key), "after": after.get(key)}
            for key, name in COMPARISON_ROWS]
