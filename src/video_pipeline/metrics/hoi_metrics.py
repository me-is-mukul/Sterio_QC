"""Before/after HOI metrics, computed from time intervals (not row counts).

"Before" is the original timeline. "After" is the same per-sample intervals
intersected with the retained intervals, i.e. measured on the processed video
through the original->processed mapping. A higher RELEVANT share after removal
reflects less waste footage, not better HOI detection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import mean, median
from typing import Any

from video_pipeline.analysis.hoi_analyzer import label_durations
from video_pipeline.analysis.temporal_filter import Interval, TimedLabel, contiguous_runs

LABELS = ("RELEVANT", "WASTE", "UNCERTAIN", "UNAVAILABLE", "MISSING")


def _pct(part: float, whole: float) -> float | None:
    return round(100.0 * part / whole, 3) if whole > 0 else None


def _count_runs_within(timeline: Sequence[TimedLabel], label: str, within: Sequence[Interval]) -> int:
    """Number of continuous `label` runs on the processed timeline."""
    clipped: list[TimedLabel] = []
    cursor = 0.0
    for w in within:
        for item in timeline:
            o = item.interval.overlap(w)
            if o > 0:
                start = cursor + max(item.interval.start, w.start) - w.start
                clipped.append(TimedLabel(item.sample_index, Interval(start, start + o), item.label))
        cursor += w.duration
    return sum(1 for r in contiguous_runs(clipped) if r.label == label)


def compute_metrics(timeline: Sequence[TimedLabel], *, original_duration: float, removed: Sequence[Interval],
                    retained: Sequence[Interval], processed_duration_measured: float | None,
                    inference_records: Sequence[Mapping[str, Any]], waste_threshold: int,
                    sampling_interval: float) -> dict[str, Any]:
    before = label_durations(timeline)
    after = label_durations(timeline, within=retained)
    removed_duration = round(sum(i.duration for i in removed), 6)
    processed_duration = round(sum(i.duration for i in retained), 6)

    latencies = [r["latency_seconds"] for r in inference_records if r.get("latency_seconds") is not None and not r.get("cached")]
    all_latencies = [r["latency_seconds"] for r in inference_records if r.get("latency_seconds") is not None]
    counts_before = {lab: sum(1 for t in timeline if t.label == lab) for lab in LABELS}
    runs_before = contiguous_runs(timeline)

    rows = []
    for label in LABELS:
        rows.append({"metric": f"{label.lower()}_duration_seconds", "before": before.get(label, 0.0), "after": after.get(label, 0.0)})
    rows += [
        {"metric": "total_duration_seconds", "before": round(original_duration, 6), "after": processed_duration},
        {"metric": "hoi_coverage_percent", "before": _pct(before.get("RELEVANT", 0.0), original_duration),
         "after": _pct(after.get("RELEVANT", 0.0), processed_duration)},
        {"metric": "waste_share_percent", "before": _pct(before.get("WASTE", 0.0), original_duration),
         "after": _pct(after.get("WASTE", 0.0), processed_duration)},
        {"metric": "hoi_intervals", "before": sum(1 for r in runs_before if r.label == "RELEVANT"),
         "after": _count_runs_within(timeline, "RELEVANT", retained)},
        {"metric": "hoi_positive_samples", "before": counts_before["RELEVANT"],
         "after": sum(1 for t in timeline if t.label == "RELEVANT" and any(t.interval.overlap(w) > 0 for w in retained))},
    ]

    return {
        "parameters": {"waste_threshold_seconds": waste_threshold, "sampling_interval_seconds": sampling_interval},
        "durations": {
            "original_duration": round(original_duration, 6),
            "processed_duration": processed_duration,
            "processed_duration_measured": processed_duration_measured,
            "retained_duration": processed_duration,
            "removed_duration": removed_duration,
            "relevant_duration": before.get("RELEVANT", 0.0),
            "waste_duration_total": before.get("WASTE", 0.0),
            "waste_duration_removed": removed_duration,
            "waste_duration_kept": after.get("WASTE", 0.0),
            "uncertain_duration": before.get("UNCERTAIN", 0.0),
            "unavailable_duration": before.get("UNAVAILABLE", 0.0),
            "missing_duration": before.get("MISSING", 0.0),
        },
        "ratios_percent": {
            "hoi_coverage_before": _pct(before.get("RELEVANT", 0.0), original_duration),
            "hoi_coverage_after": _pct(after.get("RELEVANT", 0.0), processed_duration),
            "waste_reduction": _pct(removed_duration, original_duration),
            "retention": _pct(processed_duration, original_duration),
        },
        "counts": {
            "frames_analyzed": len(inference_records),
            "samples_expected": len(timeline),
            "samples_by_label": counts_before,
            "hoi_intervals": sum(1 for r in runs_before if r.label == "RELEVANT"),
            "waste_intervals_removed": len(removed),
            "inference_failures": sum(1 for r in inference_records if r.get("status") == "failed"),
            "invalid_responses": sum(1 for r in inference_records if r.get("status") == "invalid_response"),
            "cached_results": sum(1 for r in inference_records if r.get("cached")),
        },
        "inference_latency_seconds": {
            "mean": round(mean(all_latencies), 3) if all_latencies else None,
            "median": round(median(all_latencies), 3) if all_latencies else None,
            "mean_uncached_this_run": round(mean(latencies), 3) if latencies else None,
        },
        "comparison": rows,
        "notes": [
            "Durations are derived from 1 sample per interval; each sample stands for its whole interval (approximate boundaries).",
            "A higher RELEVANT share after processing reflects removed waste, not improved HOI detection.",
        ],
    }
