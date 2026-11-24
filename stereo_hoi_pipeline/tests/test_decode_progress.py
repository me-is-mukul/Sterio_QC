"""Progress lines while both eyes decode (long recordings take an hour or more on a small VM)."""

import logging
import time

from video_pipeline.qc.technical import _ProgressReporter, _tail_progress_frame, format_decode_progress


def test_progress_line_numbers():
    line = format_decode_progress({"cam0": 90_000, "cam1": 108_000}, 270_000, elapsed=2000.0, fps=30)
    # the slower eye decides: 90 000 frames in 2000 s = 45 frames/s = 1.5x real time; 180 000 left -> 4000 s
    assert "cam0 90000/270000, cam1 108000/270000 (33%)" in line
    assert "speed 1.50x real time" in line
    assert "elapsed 33 min 20 s, about 1 h 06 min left" in line


def test_progress_line_before_first_frame():
    assert "starting" in format_decode_progress({"cam0": 0, "cam1": 0}, 1000, elapsed=5.0, fps=30)


def test_tail_reads_last_frame_of_a_long_file(tmp_path):
    path = tmp_path / "decode_cam0.progress"
    path.write_text("".join(f"frame={i}\nfps=45.0\nspeed=1.5x\nprogress=continue\n" for i in range(0, 200_000, 50)))
    assert _tail_progress_frame(path) == 199_950
    assert _tail_progress_frame(tmp_path / "missing.progress") == 0


def test_reporter_logs_with_the_step_name(tmp_path, caplog):
    from video_pipeline.core import logging_setup

    path = tmp_path / "decode_cam0.progress"
    path.write_text("frame=300\nprogress=continue\n")
    token = logging_setup._STAGE.set("technical")
    try:
        with caplog.at_level(logging.INFO, logger="video_pipeline.qc.technical"):
            caplog.handler.addFilter(logging_setup._StageFilter())
            with _ProgressReporter({"cam0": path}, 600, 30, interval=0.05):
                time.sleep(0.3)
    finally:
        logging_setup._STAGE.reset(token)
    lines = [r for r in caplog.records if r.getMessage().startswith("Decoding:")]
    assert lines and all(r.stage == "technical" for r in lines)
    assert "cam0 300/600 (50%)" in lines[0].getMessage()
    count = len(lines)
    time.sleep(0.2)                                     # stopped on exit: no more lines
    assert len([r for r in caplog.records if r.getMessage().startswith("Decoding:")]) == count
