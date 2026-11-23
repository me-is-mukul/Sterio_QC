"""Section 3 of the QA sheet: content screen (the content model judges sampled frames).

Recording level (drop the recording):
  minor in view            >= content.minor_min_frames samples flagged minor_visible
  no work                  the recording strip (frames spread over the whole recording) is judged
                           not real work (breaks / desk or computer work / staged or pretend work)
  third-person / fixed     > content.third_person_max_share of the samples flagged
  lens covered / smudged   > content.lens_max_share of the samples flagged ("most of the recording")
Clip level (drop the clip):
  private moment           >= content.private_min_frames samples inside the clip flagged
  one trivial motion       the clip strip is judged trivial_repeated_motion
Shares and counts use the content model's successful answers only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def evenly_spaced(items: Sequence[Any], count: int) -> list[Any]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    return [items[round(j * (len(items) - 1) / (count - 1))] for j in range(count)]


def _ok_labels(records: Mapping[int, Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {i: r["labels"] for i, r in records.items() if r.get("status") == "ok"}


def recording_frame_flags(content_records: Mapping[int, Mapping[str, Any]], sample_times: Mapping[int, float],
                          cfg: Mapping[str, Any]) -> dict[str, Any]:
    labels = _ok_labels(content_records)
    n = len(labels)

    def flagged(field: str) -> list[int]:
        return sorted(i for i, lab in labels.items() if lab.get(field) is True)

    minors = flagged("minor_visible")
    third = flagged("third_person_or_fixed_camera")
    lens = flagged("lens_covered_or_smudged")
    desk = flagged("desk_or_computer_work")
    working = sum(1 for lab in labels.values() if lab.get("activity") == "working")
    share = (lambda k: round(len(k) / n, 4) if n else 0.0)
    findings = []
    if len(minors) >= cfg["minor_min_frames"]:
        findings.append({"finding": "minor_in_view", "action": "drop_recording",
                         "message": f"a minor was seen in {len(minors)} sampled frames",
                         "times_s": [sample_times[i] for i in minors][:50]})
    if n and share(third) > cfg["third_person_max_share"]:
        findings.append({"finding": "third_person_or_fixed_camera", "action": "drop_recording",
                         "message": f"{share(third):.0%} of the samples are not from a head-worn camera"})
    if n and share(lens) > cfg["lens_max_share"]:
        findings.append({"finding": "lens_covered_or_smudged", "action": "drop_recording",
                         "message": f"lens covered or smudged in {share(lens):.0%} of the samples"})
    return {
        "samples_judged": n,
        "minor_frames": len(minors), "minor_times_s": [sample_times[i] for i in minors][:200],
        "third_person_share": share(third), "lens_share": share(lens),
        "desk_or_computer_share": share(desk), "working_share": round(working / n, 4) if n else 0.0,
        "findings": findings,
    }


def recording_strip_finding(answer: Mapping[str, Any]) -> dict[str, Any] | None:
    labels = answer.get("labels") or {}
    if labels.get("real_work") is False:
        return {"finding": "no_work", "action": "drop_recording",
                "message": f"not real work ({labels.get('no_work_reason')}): {labels.get('explanation', '')}"[:400]}
    return None


def clip_findings(piece_samples: Sequence[int], content_records: Mapping[int, Mapping[str, Any]],
                  strip_answer: Mapping[str, Any] | None, sample_times: Mapping[int, float],
                  cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    labels = _ok_labels(content_records)
    private = sorted(i for i in piece_samples if labels.get(i, {}).get("private_moment") is True)
    findings = []
    if len(private) >= cfg["private_min_frames"]:
        findings.append({"finding": "private_moment", "action": "drop_clip",
                         "message": f"private moment (bathroom / personal phone use) in {len(private)} sampled frames",
                         "times_s": [sample_times[i] for i in private][:50]})
    if strip_answer is not None and (strip_answer.get("labels") or {}).get("trivial_repeated_motion") is True:
        findings.append({"finding": "trivial_repeated_motion", "action": "drop_clip",
                         "message": f"one trivial motion repeated for the whole clip: "
                                    f"{strip_answer['labels'].get('explanation', '')}"[:400]})
    return findings
