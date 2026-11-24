"""Rules matched to the reference QA (detail.csv / reference_sheet.csv of 01-11-2026 and later):
mixed-reason cuts, content flags cut as stretches, gap splits, removed stretches and the two sheets."""

from video_pipeline.analysis.content_screen import clip_findings, recording_frame_flags
from video_pipeline.analysis.cropping import (
    decide_samples,
    plan_pieces,
    reason_runs,
    removed_stretches,
    split_ranges,
)
from video_pipeline.reporting.reference_sheet import (
    date_and_headset,
    detail_rows,
    recording_name,
    rejection_group,
    sheet_row,
)

FPS = 30
STEP = 60


def _samples(n):
    return [{"index": k, "frame": k * STEP, "t_s": 2.0 * k} for k in range(n)]


def _labels(hands, activity, content=None):
    n = len(hands)
    lab = {"m": {k: {"status": "ok", "labels": {"hands": hands[k], "activity": activity[k]}} for k in range(n)}}
    if content is not None:
        lab["c"] = {k: {"status": "ok", "labels": content[k]} for k in range(n)}
    return lab


PROVIDES = {"m": ["hands", "activity"], "c": ["content"]}


def test_mixed_reasons_form_one_cut():
    # reference: "Lens blocked 44%, Wearer's hands not in view 44%": neither alone spans 10 s
    hands = ["one"] * 4 + ["none"] * 4 + ["one"] * 4
    act = ["working"] * 12
    content = [{"lens_covered_or_smudged": k < 4} for k in range(12)]
    d = decide_samples(_samples(12), _labels(hands, act, content), PROVIDES, min_models=1, content_model="c",
                       content_fields=["lens_covered_or_smudged"])
    cuts = reason_runs(d, 10)
    assert len(cuts) == 1
    c = cuts[0]
    assert (c.first_frame, c.last_frame) == (0, 7 * STEP) and c.reasons == {"lens_covered_or_smudged": 4, "no_hands": 4}
    assert c.reason == "lens_covered_or_smudged"           # tie: the more important reason


def test_content_flag_needs_the_content_screen_and_a_successful_answer():
    hands, act = ["one"] * 8, ["working"] * 8
    content = [{"phone_use": True}] * 8
    labels = _labels(hands, act, content)
    labels["c"][3] = {"status": "failed"}
    d = decide_samples(_samples(8), labels, PROVIDES, min_models=1, content_model="c", content_fields=["phone_use"])
    assert [x.reasons for x in d][2:5] == [["phone_use"], [], ["phone_use"]]
    off = decide_samples(_samples(8), labels, PROVIDES, min_models=1)          # content screen off
    assert all(not x.reasons for x in off)


def test_pieces_split_at_gaps():
    assert split_ranges([(0, 100), (150, 300)], [50, 150, 200, 400]) == [(0, 50), (50, 100), (150, 200), (200, 300)]
    n = 30 * 100
    times = [i / FPS for i in range(n)]
    pieces = plan_pieces(n_frames=n, frame_times_s=times, keyframes=list(range(n)), cuts=[], fps=FPS,
                         min_piece_seconds=30, splits=[40 * FPS])
    assert [(p.start, p.stop, p.kept) for p in pieces] == [(0, 40 * FPS, True), (40 * FPS, n, True)]


def test_removed_stretches_shares():
    n = 30 * 100
    times = [i / FPS for i in range(n)]
    cuts = [{"first_frame": 10 * FPS, "last_frame": 30 * FPS, "reason": "idle"}]
    samples = [(k * STEP, ["idle"] if 5 <= k <= 15 else ([] if k < 5 or k > 20 else ["no_hands"])) for k in range(50)]
    samples[7] = (7 * STEP, ["phone_use", "idle"])
    # kept: 0-10 s is too short (dropped), 31-100 s kept
    stretches = removed_stretches(n_frames=n, frame_times_s=times, fps=FPS, kept=[(31 * FPS, n)], samples=samples,
                                  cuts=cuts)
    assert len(stretches) == 1
    s = stretches[0]
    assert s["start_s"] == 0.0 and s["end_s"] == 31.0 and s["samples"] == 16
    # samples 0..4 (short piece), 5..15 in the cut (10 idle + 1 phone)
    assert s["reasons"] == {"idle": round(10 / 16, 4), "short_piece": round(5 / 16, 4), "phone_use": round(1 / 16, 4)}
    assert s["main_reason"] == "idle"
    # nothing removed
    assert removed_stretches(n_frames=n, frame_times_s=times, fps=FPS, kept=[(0, n)], samples=samples, cuts=[]) == []


CONTENT_CFG = {"minor_min_frames": 2, "third_person_max_share": 0.5, "lens_max_share": 0.5, "private_min_frames": 2}


def test_flags_cut_as_stretches_do_not_drop_the_recording():
    recs = {i: {"status": "ok", "labels": {"minor_visible": True, "private_moment": True,
                                           "lens_covered_or_smudged": True}} for i in range(4)}
    times = {i: 2.0 * i for i in range(4)}
    assert {f["finding"] for f in recording_frame_flags(recs, times, CONTENT_CFG)["findings"]} == \
        {"minor_in_view", "lens_covered_or_smudged"}
    cfg = {**CONTENT_CFG, "cut_as_stretches": ["minor_visible", "private_moment", "lens_covered_or_smudged"]}
    assert recording_frame_flags(recs, times, cfg)["findings"] == []
    assert clip_findings([0, 1, 2, 3], recs, None, times, cfg) == []
    assert [f["finding"] for f in clip_findings([0, 1, 2, 3], recs, None, times, CONTENT_CFG)] == ["private_moment"]


def test_recording_names():
    url = "gs://op-datoric/raw/Op_DD/01-11-2026/SDC482/20260518_154730_session1/#seg1"
    assert recording_name(url) == ("raw/Op_DD/01-11-2026/SDC482/20260518_154730_session1#seg1", "1")
    assert recording_name("gs://b/raw/Op_DD/01-11-2026/SDC339/ego_20261101_105850_169F92/") == \
        ("raw/Op_DD/01-11-2026/SDC339/ego_20261101_105850_169F92", "")
    assert date_and_headset("delivery-01/SDC612/20261103_055541_session1#seg10") == ("delivery-01", "SDC612")
    assert rejection_group("decodes_end_to_end_cam1") == "Video file does not decode"
    assert rejection_group("frame_timing_cam0") == "Frame rate too low or unsteady"


def _report(verdict="PASS", fails=(), kept=1062.0, source=1299.1, stretches=None, no_video=0.0):
    return {"verdict": verdict, "fail_reasons": list(fails), "session": {"format": "robocap"},
            "technical": {"duration_s": source},
            "cropping": {"kept_seconds": kept, "no_video_seconds": no_video, "stretches": stretches or []}}


ROW = {"session": "gs://op-datoric/raw/Op_DD/01-11-2026/SDC482/20260518_154802_session2/#seg1", "status": "PASS"}


def test_sheet_row_partly_cut_matches_the_reference_layout():
    stretches = [
        {"start_s": 0.0, "end_s": 153.0, "length_s": 153.0, "main_reason": "idle",
         "reasons": {"idle": 0.56, "phone_use": 0.40, "short_piece": 0.04}},
        {"start_s": 560.0, "end_s": 595.0, "length_s": 35.0, "main_reason": "private_moment",
         "reasons": {"private_moment": 0.94, "short_piece": 0.06}},
    ]
    row = sheet_row({**ROW, "report": _report(stretches=stretches, kept=1111.1)})
    assert row["Upload date folder"] == "01-11-2026" and row["Headset"] == "SDC482" and row["Segment"] == "1"
    assert row["Rig"] == "RoboCap" and row["Status"] == "Accepted (partly cut)"
    assert row["Source hours"] == "0.361" and row["Accepted hours"] == "0.309" and row["Hours cut"] == "0.052"
    assert row["Reason"] == ("Cut: Idle (no work in hand) 1.4 min, Phone use 1.0 min, "
                             "Private moment (bathroom, changing) 0.5 min, Piece under 30 s between cuts 0.1 min")
    details = detail_rows({**ROW, "report": _report(stretches=stretches)})
    assert details[0]["Cut start (s)"] == "0.0" and details[0]["Cut length (s)"] == "153.0"
    assert details[0]["Main reason"] == "Idle (no work in hand)"
    assert details[0]["Reasons in this stretch (share of samples)"] == \
        "Idle (no work in hand) 56%, Phone use 40%, Piece under 30 s between cuts 4%"


def test_sheet_row_rejections():
    short = sheet_row({**ROW, "report": _report("FAIL", [{"check": "length", "message": "recording lasts 4.0 s"}],
                                                source=4.0)})
    assert (short["Status"], short["Reason"], short["Rejection group"], short["Hours cut"]) == \
        ("Rejected", "Shorter than 30 s", "Shorter than 30 s", "")
    nothing = sheet_row({**ROW, "report": _report("FAIL", [{"check": "nothing_left", "message": "x"}], stretches=[
        {"start_s": 0.0, "end_s": 229.0, "length_s": 229.0, "main_reason": "idle",
         "reasons": {"idle": 0.57, "private_moment": 0.22, "short_piece": 0.11, "no_hands": 0.10}}])})
    assert nothing["Reason"] == ("No stretch of 30 s or more left after cuts: Idle (no work in hand) 57%, "
                                 "Private moment (bathroom, changing) 22%, Piece under 30 s between cuts 11%")
    error = sheet_row({**ROW, "status": "ERROR", "reasons": "boom"})
    assert error["Status"] == "Error" and error["Reason"] == "boom"
    whole = sheet_row({**ROW, "report": _report(kept=58.0, source=58.0)})
    assert whole["Status"] == "Accepted" and whole["Hours cut"] == "0.000" and whole["Reason"] == ""


def test_optional_configs_are_valid():
    from pathlib import Path

    from video_pipeline.core.config import find_project_root
    from video_pipeline.core.ego_config import load_ego_configuration
    from video_pipeline.inference.prompts import get_prompt

    root = find_project_root()
    for name in ("ego_qa_gemini.yaml", "ego_qa_with_vlm.yaml"):
        cfg = load_ego_configuration(root, Path("configs") / name, {})
        assert cfg.get("content.enabled") and get_prompt(cfg.get("inference.frame_prompt"))
