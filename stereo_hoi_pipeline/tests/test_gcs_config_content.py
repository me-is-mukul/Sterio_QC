"""Session paths, configuration validation, content-screen rules, answer validation."""

from pathlib import Path

import pytest
import yaml

from video_pipeline.analysis.content_screen import clip_findings, evenly_spaced, recording_frame_flags, \
    recording_strip_finding
from video_pipeline.core.config import find_project_root
from video_pipeline.core.ego_config import load_ego_configuration
from video_pipeline.core.errors import ConfigError, IngestionError
from video_pipeline.inference.ego_labeller import validate_answer
from video_pipeline.inference.prompts import EGO_FRAME_SCHEMA
from video_pipeline.ingestion.gcs import ObjectInfo, SessionRef, classify_session_objects, parse_gs_url

RAW = "gs://op-datoric/raw/Op_DD/01-11-2026/SDC001/session_00002-1793528549/"


def test_session_ref():
    s = SessionRef.from_url(RAW, expected_bucket="op-datoric", raw_root="raw/")
    assert s.relative == "Op_DD/01-11-2026/SDC001/session_00002-1793528549"
    assert s.prefix == "raw/Op_DD/01-11-2026/SDC001/session_00002-1793528549/"
    assert s.describe()["device"] == "SDC001" and s.describe()["date"] == "01-11-2026"
    assert SessionRef.from_url(RAW.rstrip("/"), expected_bucket="op-datoric", raw_root="raw/").relative == s.relative


@pytest.mark.parametrize("url", [
    "gs://other/raw/Op_DD/x/session_1/",          # wrong bucket
    "gs://op-datoric/processed/Op_DD/session_1/",  # not under raw/
    "gs://op-datoric/raw/Op_DD/01-11-2026/",       # not a session folder
    "gs://op-datoric/raw/../session_1/",           # path tricks
    "https://storage.googleapis.com/op-datoric/raw/x/session_1/",
])
def test_session_ref_rejects(url):
    with pytest.raises(IngestionError):
        SessionRef.from_url(url, expected_bucket="op-datoric", raw_root="raw/")


def test_parse_gs_url():
    assert parse_gs_url("gs://b/a/c/") == ("b", "a/c/")
    with pytest.raises(IngestionError):
        parse_gs_url("gs:///x")


def _obj(name, size=1):
    return ObjectInfo(name, size, 1, "crc", "md5")


def test_classify_segments_numeric_order_and_extras():
    s = SessionRef.from_url(RAW, expected_bucket="op-datoric", raw_root="raw/")
    p = s.prefix
    listing = classify_session_objects(s, [_obj(p + "ego_0010.mcap"), _obj(p + "ego_0002.mcap"), _obj(p + "ego_10000.mcap"),
                                           _obj(p + "session.json"), _obj(p + "notes.txt"), _obj(p + "sub/ego_0001.mcap")],
                                       session_json="session.json", segment_pattern=r"^ego_(\d+)\.mcap$")
    assert [n for n, _ in listing.segments] == [2, 10, 10000]
    assert listing.session_json.name.endswith("session.json")
    assert {o.name.rsplit("/", 1)[-1] for o in listing.other} == {"notes.txt", "ego_0001.mcap"}
    assert listing.identity() == classify_session_objects(s, [_obj(p + "session.json"), _obj(p + "ego_10000.mcap"),
                                                              _obj(p + "ego_0002.mcap"), _obj(p + "ego_0010.mcap")],
                                                          session_json="session.json",
                                                          segment_pattern=r"^ego_(\d+)\.mcap$").identity()


def test_default_config_is_valid():
    cfg = load_ego_configuration(find_project_root())
    assert cfg.get("labelling.interval_seconds") == 2 and cfg.get("cropping.idle_cut_seconds") == 10
    assert cfg.get("cropping.min_piece_seconds") == 30
    assert cfg.get("calibration.max_median_vertical_px") == 1.5 and cfg.get("calibration.max_p95_vertical_px") == 5.0


def _user_config(tmp_path, data) -> Path:
    path = tmp_path / "user.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_user_config_replaces_models_and_checks_minimum(tmp_path):
    only_one = {"models": {"g": {"provider": "gemini", "model": "m", "location": "l",
                                 "provides": ["hands", "activity", "content"]}}}
    with pytest.raises(ConfigError, match="min_models_per_decision"):
        load_ego_configuration(find_project_root(), _user_config(tmp_path, only_one))
    ok = {**only_one, "cropping": {"min_models_per_decision": 1}}
    cfg = load_ego_configuration(find_project_root(), _user_config(tmp_path, ok))
    assert list(cfg.get("models")) == ["g"]


def test_user_config_rejects_unknown_keys_and_two_content_models(tmp_path):
    with pytest.raises(ConfigError):
        load_ego_configuration(find_project_root(), _user_config(tmp_path, {"cropping": {"idle_seconds": 5}}))
    two = {"models": {k: {"provider": "gemini", "model": "m", "location": "l", "provides": ["hands", "activity", "content"]}
                      for k in ("a", "b")}}
    with pytest.raises(ConfigError, match="exactly one model"):
        load_ego_configuration(find_project_root(), _user_config(tmp_path, two))


CONTENT_CFG = {"minor_min_frames": 2, "third_person_max_share": 0.5, "lens_max_share": 0.5, "private_min_frames": 2}


def _rec(**flags):
    base = {"hands": "one", "activity": "working", "phone_use": False, "minor_visible": False,
            "third_person_or_fixed_camera": False, "lens_covered_or_smudged": False, "desk_or_computer_work": False,
            "private_moment": False, "description": ""}
    return {"status": "ok", "labels": {**base, **flags}}


def test_recording_flags():
    times = {i: 2.0 * i for i in range(10)}
    one_minor = {i: _rec(minor_visible=(i == 3)) for i in range(10)}
    assert recording_frame_flags(one_minor, times, CONTENT_CFG)["findings"] == []
    two_minors = {i: _rec(minor_visible=i in (3, 7)) for i in range(10)}
    f = recording_frame_flags(two_minors, times, CONTENT_CFG)["findings"]
    assert [x["finding"] for x in f] == ["minor_in_view"] and f[0]["times_s"] == [6.0, 14.0]
    lens_half = {i: _rec(lens_covered_or_smudged=i < 5) for i in range(10)}
    assert recording_frame_flags(lens_half, times, CONTENT_CFG)["findings"] == []          # 50 % is not "most"
    lens_most = {i: _rec(lens_covered_or_smudged=i < 6) for i in range(10)}
    assert [x["finding"] for x in recording_frame_flags(lens_most, times, CONTENT_CFG)["findings"]] == \
        ["lens_covered_or_smudged"]
    failed = {i: {"status": "failed"} for i in range(10)}
    assert recording_frame_flags(failed, times, CONTENT_CFG)["samples_judged"] == 0


def test_strip_and_clip_findings():
    assert recording_strip_finding({"labels": {"real_work": True, "no_work_reason": "none"}}) is None
    assert recording_strip_finding({"labels": {"real_work": False, "no_work_reason": "breaks",
                                               "explanation": "x"}})["finding"] == "no_work"
    times = {i: 2.0 * i for i in range(10)}
    recs = {i: _rec(private_moment=i in (2, 3)) for i in range(10)}
    assert [f["finding"] for f in clip_findings([0, 1, 2, 3], recs, None, times, CONTENT_CFG)] == ["private_moment"]
    assert clip_findings([0, 1, 2], recs, None, times, CONTENT_CFG) == []
    strip = {"labels": {"trivial_repeated_motion": True, "explanation": "tapping"}}
    assert [f["finding"] for f in clip_findings([0], recs, strip, times, CONTENT_CFG)] == ["trivial_repeated_motion"]


def test_evenly_spaced():
    assert evenly_spaced(list(range(100)), 5) == [0, 25, 50, 74, 99]
    assert evenly_spaced([1, 2], 5) == [1, 2]


def test_validate_answer():
    good = ('{"hands": "Both", "activity": "idle", "phone_use": true, "minor_visible": false, '
            '"third_person_or_fixed_camera": false, "lens_covered_or_smudged": false, '
            '"desk_or_computer_work": false, "private_moment": false, "description": "x"}')
    labels, errors = validate_answer("```json\n" + good + "\n```", EGO_FRAME_SCHEMA)
    assert errors == [] and labels["hands"] == "both" and labels["phone_use"] is True
    labels, errors = validate_answer(good.replace('"idle"', '"resting"'), EGO_FRAME_SCHEMA)
    assert labels is None and "activity" in errors[0]
    labels, errors = validate_answer(good.replace("true", '"yes"'), EGO_FRAME_SCHEMA)
    assert labels is None
    assert validate_answer("no json here", EGO_FRAME_SCHEMA)[0] is None
