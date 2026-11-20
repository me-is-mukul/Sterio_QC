"""Pure (no I/O) quality checks on decisions and data. Each returns a CheckResult."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from video_pipeline.analysis.temporal_filter import Interval, TimedLabel

PASS, WARN, FAIL, SKIPPED = "pass", "warn", "fail", "skipped"
_EPS = 1e-6


@dataclass(frozen=True)
class CheckResult:
    id: str
    title: str
    status: str
    detail: str
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def overall_status(results: Sequence[CheckResult]) -> str:
    statuses = {r.status for r in results}
    return FAIL if FAIL in statuses else WARN if WARN in statuses else PASS


def check_partition(removed: Sequence[Interval], retained: Sequence[Interval], duration: float) -> CheckResult:
    """Removed + retained must tile [0, duration) without gaps or overlaps."""
    pieces = sorted([*removed, *retained], key=lambda i: i.start)
    problems = []
    cursor = 0.0
    for p in pieces:
        if abs(p.start - cursor) > _EPS:
            problems.append(f"{'gap' if p.start > cursor else 'overlap'} at {cursor:.3f}s")
        cursor = max(cursor, p.end)
    if abs(cursor - duration) > _EPS:
        problems.append(f"coverage ends at {cursor:.3f}s, video ends at {duration:.3f}s")
    return CheckResult("decisions.partition", "Removed + retained intervals cover the video exactly",
                       FAIL if problems else PASS, "; ".join(problems) or "no gaps or overlaps")


def check_removed_only_waste(removed: Sequence[Interval], coarse: Sequence[TimedLabel],
                             fine: Sequence[tuple[float, float, str]]) -> CheckResult:
    """No removed time may overlap a sample (coarse or fine) that is not WASTE."""
    violations = []
    for s in coarse:
        if s.label != "WASTE" and any(s.interval.overlap(r) > _EPS for r in removed):
            violations.append(f"{s.label} @ {s.interval.start:g}s (1 s grid)")
    for start, end, label in fine:
        if label != "WASTE" and any(Interval(start, end).overlap(r) > _EPS for r in removed):
            violations.append(f"{label} @ {start:g}s (fine)")
    return CheckResult("decisions.no_hoi_removed", "No removed section contains relevant/uncertain content",
                       FAIL if violations else PASS,
                       "; ".join(violations[:10]) or f"all {len(removed)} removed interval(s) contain only WASTE samples",
                       {"violations": len(violations)})


def check_min_duration(removed: Sequence[Interval], threshold: float) -> CheckResult:
    short = [r for r in removed if r.duration + _EPS < threshold]
    return CheckResult("decisions.min_duration", f"Every removed interval lasts >= N ({threshold}s)",
                       FAIL if short else PASS,
                       ", ".join(f"[{r.start:g}, {r.end:g})" for r in short) or "ok")


def check_inference_health(records: Sequence[Mapping[str, Any]], max_ratio: float) -> CheckResult:
    if not records:
        return CheckResult("inference.health", "Inference results are valid", FAIL, "no inference results")
    failed = sum(1 for r in records if r.get("status") == "failed")
    invalid = sum(1 for r in records if r.get("status") == "invalid_response")
    ratio = (failed + invalid) / len(records)
    status = PASS if ratio == 0 else WARN if ratio <= max_ratio else FAIL
    return CheckResult("inference.health", "Inference results are valid", status,
                       f"{failed} failed, {invalid} invalid of {len(records)} ({ratio:.1%}); "
                       "these were preserved, never removed. Re-run with --resume to retry them.",
                       {"failed": failed, "invalid": invalid, "ratio": round(ratio, 4)})


def check_coverage(coarse: Sequence[TimedLabel]) -> CheckResult:
    missing = [t.interval.start for t in coarse if t.label == "MISSING"]
    return CheckResult("inference.coverage", "Every sampled second has a classification",
                       WARN if missing else PASS,
                       f"{len(missing)} second(s) without a result (preserved): {missing[:10]}" if missing else
                       f"{len(coarse)} of {len(coarse)} seconds classified")


def check_imu(sync: Mapping[str, Any] | None, processed_duration: float | None) -> CheckResult:
    title = "IMU telemetry matches the processed timeline"
    if not sync or sync.get("status") != "synchronized":
        return CheckResult("imu.sync", title, SKIPPED, (sync or {}).get("reason") or "no IMU telemetry")
    problems = []
    for code, s in (sync.get("streams") or {}).items():
        if s["kept_samples"] + s["removed_samples"] + s["outside_video_samples"] != s["input_samples"]:
            problems.append(f"{code}: sample counts do not add up")
        if s.get("max_processed_time") is not None and processed_duration is not None \
                and s["max_processed_time"] > processed_duration + 0.05:
            problems.append(f"{code}: samples beyond processed video end")
        if s.get("non_monotonic"):
            problems.append(f"{code}: processed timestamps not monotonic")
    return CheckResult("imu.sync", title, FAIL if problems else PASS, "; ".join(problems) or "consistent")
