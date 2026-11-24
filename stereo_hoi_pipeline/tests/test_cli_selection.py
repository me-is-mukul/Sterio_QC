"""Command line: --source / --dest / --start / --end / --content-screen."""

import pytest

from video_pipeline.core.config import find_project_root
from video_pipeline.core.ego_config import load_ego_configuration, processed_bucket
from video_pipeline.core.errors import ConfigError, PipelineError
from video_pipeline.ego_cli import discover_sessions, folder_prefix, format_session_list, select_range, source_root
from video_pipeline.ingestion.gcs import ObjectInfo, SessionRef

PATTERN = r"^ego_(\d+)\.mcap$"


class FakeStorage:
    def __init__(self, names):
        self.names = names

    def list_objects(self, bucket, prefix):
        return [ObjectInfo(n, 100, 1, None, None) for n in self.names if n.startswith(prefix)]


OBJECTS = [
    "raw/d2/devB/session_00001-2/ego_0000.mcap", "raw/d2/devB/session_00001-2/session.json",
    "raw/d1/devA/session_00002-9/ego_0000.mcap", "raw/d1/devA/session_00002-9/ego_0001.mcap",
    "raw/d1/devA/session_00001-5/ego_0000.mcap",
    "raw/d1/devA/session_00003-1/video.h265",             # another recording format: not listed
    "raw/d1/devA/session_00004-1/sub/ego_0000.mcap",       # nested: not a segment of the session
    "raw/d1/other.txt",
]


def test_folder_prefix_and_source_root():
    assert folder_prefix("gs://b/raw/Op_DD", "--source") == ("b", "raw/Op_DD/")
    assert folder_prefix("gs://b/raw/Op_DD/", "--source") == ("b", "raw/Op_DD/")
    for bad in ("gs://b/", "gs://b", "gs://b/raw/../x/", "/local/path"):
        with pytest.raises(PipelineError):
            folder_prefix(bad, "--source")
    assert source_root("raw/d1/") == "raw/d1/"
    assert source_root("raw/d1/devA/session_00002-9/") == "raw/d1/devA/"     # one session as the source
    with pytest.raises(PipelineError):
        source_root("session_1/")


def test_discover_sessions_numbering_and_formats():
    found, skipped = discover_sessions(FakeStorage(OBJECTS), "b", "raw/", PATTERN)
    assert [s["url"] for s in found] == ["gs://b/raw/d1/devA/session_00001-5/", "gs://b/raw/d1/devA/session_00002-9/",
                                         "gs://b/raw/d2/devB/session_00001-2/"]
    assert [s["segments"] for s in found] == [1, 2, 1] and skipped == 2
    one, _ = discover_sessions(FakeStorage(OBJECTS), "b", "raw/d1/devA/session_00002-9/", PATTERN)
    assert [s["url"] for s in one] == ["gs://b/raw/d1/devA/session_00002-9/"]
    text = format_session_list(found, skipped, 2, 3)
    assert ">   2" in text and ">   3" in text and "    1" in text and "selected by --start 2 --end 3: 2 sessions" in text


def test_select_range():
    items = list("abcde")
    assert select_range(items, None, None) == items
    assert select_range(items, 2, 4) == ["b", "c", "d"]
    assert select_range(items, 5, None) == ["e"] and select_range(items, None, 1) == ["a"]
    for start, end in ((0, 2), (3, 2), (1, 6), (6, None)):
        with pytest.raises(PipelineError):
            select_range(items, start, end)
    with pytest.raises(PipelineError):
        select_range([], None, None)


def test_source_and_dest_overrides():
    root = find_project_root()
    cfg = load_ego_configuration(root, None, {"source.bucket": "in", "source.raw_root": "raw/Op_DD/31-10-2026/",
                                              "source.processed_bucket": "out", "source.processed_root": "results/"})
    assert processed_bucket(cfg) == "out"
    s = SessionRef.from_url("gs://in/raw/Op_DD/31-10-2026/SDC509/session_00005-1/", expected_bucket="in",
                            raw_root=cfg.get("source.raw_root"))
    assert s.relative == "SDC509/session_00005-1"          # result: gs://out/results/SDC509/session_00005-1/
    assert processed_bucket(load_ego_configuration(root)) == "op-datoric"
    # a different bucket may use any folder names; the same bucket must not nest source and destination
    load_ego_configuration(root, None, {"source.processed_bucket": "other", "source.processed_root": "raw/x/"})
    for raw, out in (("raw/", "raw/out/"), ("data/", "data/"), ("a/b/", "a/")):
        with pytest.raises(ConfigError, match="must not contain each other"):
            load_ego_configuration(root, None, {"source.raw_root": raw, "source.processed_root": out})


def test_content_screen_switch():
    root = find_project_root()
    assert load_ego_configuration(root, None, {"content.enabled": False}).get("content.enabled") is False
    with pytest.raises(ConfigError, match="content screen is switched on"):
        load_ego_configuration(root, None, {"content.enabled": True})
