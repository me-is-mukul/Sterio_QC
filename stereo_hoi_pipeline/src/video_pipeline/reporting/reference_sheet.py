"""Batch results in the reference QA's format (one row per recording, one row per removed stretch).

* ``sheet.csv``   one row per recording: status, source / accepted / cut / no-video hours, reason,
                  rejection group (columns as in the reference sheet).
* ``detail.csv``  one row per removed stretch: start, end, length, main reason and the share of its
                  2 s samples per reason (columns as in the reference detail list).

Built from each recording's qa_report.json (pure functions; the batch writes the files).
Reason wording follows the reference; reasons the reference does not have keep our own words.
"""

from __future__ import annotations

import csv
import io
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from video_pipeline.analysis.cropping import SHORT_PIECE, removed_stretches

SHEET_FIELDS = ["Upload date folder", "Headset", "Recording", "Segment", "Rig", "Status", "Source hours",
                "Accepted hours", "Reason", "Rejection group", "Hours cut",
                "Hours with no video frames (dropped frames, missing tail)"]
DETAIL_FIELDS = ["Recording", "Segment", "Cut start (s)", "Cut end (s)", "Cut length (s)", "Main reason",
                 "Reasons in this stretch (share of samples)"]

REASON_LABELS = {
    "idle": "Idle (no work in hand)",
    "no_hands": "Wearer's hands not in view",
    "phone_use": "Phone use",
    "minor_visible": "Person under 18 in view",
    "private_moment": "Private moment (bathroom, changing)",
    "lens_covered_or_smudged": "Lens blocked or out of focus",
    "third_person_or_fixed_camera": "Headset not worn",
    SHORT_PIECE: "Piece under 30 s between cuts",
    "black_or_frozen": "Black or frozen video",
    "trivial_repeated_motion": "Not real work (staged, home or no work)",
    "content": "Dropped by the content screen",
}
RIGS = {"robocap": "RoboCap", "hydra": "OpenRobot", "egocam_egoc": "Eye 2", "egocam_live": "Eye 2 (live)",
        "mcap": "MCAP"}
NOTHING_LEFT = "No stretch of 30 s or more left after cuts"
_GROUPS = {
    "length": "Shorter than 30 s",
    "eyes_paired": "Left and right videos do not pair",
    "nothing_left": NOTHING_LEFT,
    "nothing_left_after_clip_screen": NOTHING_LEFT,
    "minor_in_view": "Person under 18 in view",
    "third_person_or_fixed_camera": "Headset not worn",
    "lens_covered_or_smudged": "Lens blocked or out of focus",
    "no_work": "Not real work (staged, home or no work)",
}
_MISSING_FILES = {"session_json_present", "first_segment_present", "segments_contiguous", "segment_present",
                  "core_topics_present"}
_HEADSET = re.compile(r"^SDC\d+$", re.IGNORECASE)


def label(reason: str) -> str:
    return REASON_LABELS.get(reason, reason.replace("_", " "))


def rejection_group(check: str) -> str:
    if check in _GROUPS:
        return _GROUPS[check]
    if check.startswith("decodes_end_to_end"):
        return "Video file does not decode"
    if check.startswith("frame_timing"):
        return "Frame rate too low or unsteady"
    if check in _MISSING_FILES or check.startswith("file_present"):
        return "Missing data files"
    if check.startswith("imu"):
        return "IMU data missing or broken"
    if check.startswith("not_black_or_frozen"):
        return "Video black or frozen"
    return check


def recording_name(url: str) -> tuple[str, str]:
    """gs://bucket/raw/a/b/session/#seg3 -> ('raw/a/b/session#seg3', '3')."""
    path = re.sub(r"^gs://[^/]+/", "", url.strip())
    path, _, seg = path.partition("#")
    path = path.rstrip("/")
    number = seg[3:] if seg.startswith("seg") else ""
    return (f"{path}#{seg}" if seg else path), number


def date_and_headset(recording: str) -> tuple[str, str]:
    parts = recording.split("#", 1)[0].split("/")
    for i, part in enumerate(parts):
        if _HEADSET.match(part):
            return (parts[i - 1] if i else ""), part
    return (parts[-2] if len(parts) >= 2 else ""), ""


def _hours(seconds: float | None) -> str:
    return f"{max(0.0, float(seconds or 0.0)) / 3600:.3f}"


def stretches_of(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The report's removed stretches; reports written before they existed get them rebuilt from the
    pieces and cuts (no per-sample shares: each stretch counts under one reason)."""
    crop = report.get("cropping") or {}
    if "stretches" in crop:
        return list(crop["stretches"])
    if not crop.get("pieces"):
        return []
    fps = 30.0
    n = int(round(float(crop.get("recording_seconds") or 0) * fps))
    if n <= 0:
        return []
    kept = [(p["start"], p["stop"]) for p in crop["pieces"] if p.get("kept")]
    clips = ((report.get("content") or {}).get("clips")) or []
    if clips:
        kept = [(c["start"], c["stop"]) for c in clips if c.get("kept")]
    return removed_stretches(n_frames=n, frame_times_s=[i / fps for i in range(n)], fps=fps, kept=kept, samples=[],
                             cuts=crop.get("cuts", []), defects=crop.get("defect_cuts_frames", []))


def reason_seconds(stretches: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Seconds removed per reason (a stretch's length split by the share of its samples)."""
    out: dict[str, float] = defaultdict(float)
    for s in stretches:
        for reason, share in s["reasons"].items():
            out[reason] += float(s["length_s"]) * float(share)
    return dict(out)


def _ranked(seconds: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(seconds.items(), key=lambda kv: (-kv[1], kv[0]))


def sheet_row(row: Mapping[str, Any]) -> dict[str, str]:
    """One sheet row from a batch row ({'session', 'status', 'reasons', 'report'})."""
    report = row.get("report") or None
    recording, segment = recording_name(str(row.get("session", "")))
    date, headset = date_and_headset(recording)
    session = (report or {}).get("session") or {}
    out = {"Upload date folder": date, "Headset": headset, "Recording": recording, "Segment": segment,
           "Rig": RIGS.get(session.get("format") or "", session.get("format") or ""), "Status": "",
           "Source hours": "", "Accepted hours": "", "Reason": "", "Rejection group": "", "Hours cut": "",
           "Hours with no video frames (dropped frames, missing tail)": ""}
    if report is None:
        status = str(row.get("status", ""))
        out.update({"Status": "Error" if status == "ERROR" else status.title(),
                    "Reason": str(row.get("reasons") or ""),
                    "Rejection group": "Processing error (run again)" if status == "ERROR" else ""})
        return out
    tech = report.get("technical") or {}
    crop = report.get("cropping") or {}
    source = float(tech.get("duration_s") or 0.0)
    stretches = stretches_of(report)
    out["Source hours"] = _hours(source)
    if report.get("verdict") != "PASS":
        fails = report.get("fail_reasons") or []
        group = rejection_group(fails[0]["check"]) if fails else "Rejected"
        if group == "Shorter than 30 s":
            reason = group
        elif group == NOTHING_LEFT:
            seconds = reason_seconds(stretches)
            total = sum(seconds.values()) or 1.0
            reason = f"{NOTHING_LEFT}: " + ", ".join(f"{label(r)} {100 * s / total:.0f}%"
                                                     for r, s in _ranked(seconds)[:3])
        else:
            reason = "; ".join(f["message"] for f in fails)
        out.update({"Status": "Rejected", "Accepted hours": _hours(0), "Reason": reason, "Rejection group": group})
        return out
    kept = float(crop.get("kept_seconds") or 0.0)
    no_video = float(crop.get("no_video_seconds") or tech.get("no_video_seconds") or 0.0)
    seconds = reason_seconds(stretches)
    out.update({
        "Status": "Accepted (partly cut)" if stretches else "Accepted",
        "Accepted hours": _hours(kept),
        "Reason": ("Cut: " + ", ".join(f"{label(r)} {s / 60:.1f} min" for r, s in _ranked(seconds))) if seconds else "",
        "Hours cut": _hours(source - kept - no_video),
        "Hours with no video frames (dropped frames, missing tail)": _hours(no_video),
    })
    return out


def detail_rows(row: Mapping[str, Any]) -> list[dict[str, str]]:
    report = row.get("report") or None
    if report is None:
        return []
    recording, segment = recording_name(str(row.get("session", "")))
    out = []
    for s in stretches_of(report):
        shares = sorted(s["reasons"].items(), key=lambda kv: -kv[1])
        out.append({"Recording": recording, "Segment": segment, "Cut start (s)": f"{s['start_s']:.1f}",
                    "Cut end (s)": f"{s['end_s']:.1f}", "Cut length (s)": f"{s['length_s']:.1f}",
                    "Main reason": label(s["main_reason"]),
                    "Reasons in this stretch (share of samples)": ", ".join(f"{label(r)} {100 * v:.0f}%"
                                                                           for r, v in shares)})
    return out


def to_csv(fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def sheet_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    return to_csv(SHEET_FIELDS, [sheet_row(r) for r in rows if r.get("status") not in {"PARTIAL", "INTERRUPTED"}])


def detail_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    return to_csv(DETAIL_FIELDS, [d for r in rows for d in detail_rows(r)])
