"""Deliverable reports: qc_checklist.csv, qc_checklist.xlsx and qc_report.html."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from . import __version__
from .manifest import Clip
from .util import write_csv

TEST_ORDER = ["duration", "fps", "bitrate", "codec", "decode_errors", "luminance", "sharpness", "blockiness",
              "colour_cast", "orientation", "either_hand_pct", "both_hands_pct", "left_hand_pct", "right_hand_pct",
              "hoi_pct", "tracking_losses", "untracked_pct", "grasp_count", "grasp_duration", "longest_hand_gap",
              "other_hands_pct"]

# (column header, getter, test id whose verdict colours the cell or None)
def _columns():
    t = lambda s, k: s["technical"].get(k)   # noqa: E731
    h = lambda s, k: s["hands"].get(k)       # noqa: E731
    return [
        ("Duration (s)", lambda s: t(s, "duration_s"), "duration"),
        ("FPS", lambda s: t(s, "fps"), "fps"),
        ("Resolution", lambda s: f"{t(s, 'width')}x{t(s, 'height')}", None),
        ("Bitrate (Mbit/s)", lambda s: t(s, "video_bitrate_mbps"), "bitrate"),
        ("Bitrate required (Mbit/s)", lambda s: t(s, "required_bitrate_mbps"), None),
        ("Codec", lambda s: t(s, "codec"), "codec"),
        ("Decode errors", lambda s: next(x["value"] for x in s["tests"] if x["id"] == "decode_errors"), "decode_errors"),
        ("Luminance mean (0-255)", lambda s: t(s, "luma_mean"), "luminance"),
        ("Too dark/bright frames (%)", lambda s: t(s, "bad_exposure_frame_pct"), "luminance"),
        ("Sharp frames (%)", lambda s: t(s, "sharp_frame_pct"), "sharpness"),
        ("Sharpness median", lambda s: t(s, "sharpness_median"), "sharpness"),
        ("Blockiness", lambda s: t(s, "blockiness"), "blockiness"),
        ("Colour cast (Lab)", lambda s: t(s, "colour_cast"), "colour_cast"),
        ("Scene colour (Lab, info)", lambda s: t(s, "scene_colour"), None),
        ("Rotation (deg)", lambda s: t(s, "rotation_deg"), "orientation"),
        ("Upright hands (%)", lambda s: t(s, "upright_hand_pct"), "orientation"),
        ("either_hand_pct", lambda s: h(s, "either_hand_pct"), "either_hand_pct"),
        ("both_hands_pct", lambda s: h(s, "both_hands_pct"), "both_hands_pct"),
        ("left_hand_pct", lambda s: h(s, "left_hand_pct"), "left_hand_pct"),
        ("right_hand_pct", lambda s: h(s, "right_hand_pct"), "right_hand_pct"),
        ("HOI_pct", lambda s: h(s, "hoi_pct"), "hoi_pct"),
        ("tracking_losses", lambda s: h(s, "tracking_losses"), "tracking_losses"),
        ("tracking_losses_per_min", lambda s: h(s, "tracking_losses_per_min"), "tracking_losses"),
        ("untracked_pct", lambda s: h(s, "untracked_pct"), "untracked_pct"),
        ("grasp_count", lambda s: h(s, "grasp_count"), "grasp_count"),
        ("grasp_duration_total (s)", lambda s: h(s, "grasp_duration_total_s"), "grasp_duration"),
        ("grasp_duration_mean (s)", lambda s: h(s, "grasp_duration_mean_s"), "grasp_duration"),
        ("longest_hand_gap (s)", lambda s: h(s, "longest_hand_gap_s"), "longest_hand_gap"),
        ("other_hands_pct", lambda s: h(s, "other_hands_pct"), "other_hands_pct"),
    ]


ID_COLUMNS = ["Clip no", "Clip ID", "Dataset", "Category", "Task", "Camera", "Layout", "QC result",
              "Tests passed", "Failed tests"]
TAIL_COLUMNS = ["Landmarks file", "Source URL", "Original reference", "Notes"]


def _verdict_map(summary: dict) -> dict[str, str]:
    return {t["id"]: t["status"] for t in summary["tests"]}


def _rows(clips: list[Clip], summaries: dict, errors: dict) -> list[dict]:
    cols = _columns()
    rows = []
    for c in clips:
        row = {"Clip no": c.clip_no, "Clip ID": c.clip_id, "Dataset": c.dataset, "Category": c.category,
               "Task": c.task, "Camera": c.camera, "Layout": c.layout}
        s = summaries.get(c.clip_id)
        if not c.available:
            row.update({"QC result": "SOURCE MISSING", "Tests passed": "",
                        "Failed tests": "No usable source URL in the clip list"})
        elif s is None:
            row.update({"QC result": "ERROR", "Tests passed": "", "Failed tests": errors.get(c.clip_id, "not processed")})
        else:
            graded = [t for t in s["tests"] if t["status"] in ("PASS", "FAIL")]
            row.update({"QC result": s["overall"],
                        "Tests passed": f"{sum(1 for t in graded if t['status'] == 'PASS')}/{len(graded)}",
                        "Failed tests": "; ".join(s["reasons"])})
        for header, getter, _ in cols:
            row[header] = getter(s) if s else ""
        row.update({"Landmarks file": f"videos/{c.clip_id}/landmarks.csv" if s else "",
                    "Source URL": c.source_url, "Original reference": c.original_reference, "Notes": c.notes})
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- Excel
FILL = {"PASS": PatternFill("solid", fgColor="D9F2E1"), "FAIL": PatternFill("solid", fgColor="F8D7D7"),
        "SOURCE MISSING": PatternFill("solid", fgColor="E7E7E7"), "ERROR": PatternFill("solid", fgColor="FCE8C8"),
        "INFO": None, "N/A": None}
FONT_RESULT = {"PASS": Font(bold=True, color="146C34"), "FAIL": Font(bold=True, color="A61B1B"),
               "SOURCE MISSING": Font(bold=True, color="555555"), "ERROR": Font(bold=True, color="9A5B00")}
HEADER_FILL = {"id": PatternFill("solid", fgColor="1F2A44"), "tech": PatternFill("solid", fgColor="2F5D8A"),
               "hands": PatternFill("solid", fgColor="3C7D5A"), "tail": PatternFill("solid", fgColor="5A5A5A")}
THIN = Side(style="thin", color="D0D0D0")


def _write_xlsx(path: Path, clips, summaries, errors, rows, config, generated: str, context: dict) -> None:
    wb = Workbook()
    cols = _columns()
    tech_headers = {h for h, _, _ in cols[:16]}
    headers = ID_COLUMNS + [h for h, _, _ in cols] + TAIL_COLUMNS

    # --- Checklist ---
    ws = wb.active
    ws.title = "Checklist"
    ws.append(headers)
    for ci, hname in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=ci)
        group = "id" if hname in ID_COLUMNS else "tech" if hname in tech_headers else "tail" \
            if hname in TAIL_COLUMNS else "hands"
        cell.fill = HEADER_FILL[group]
        cell.font = Font(bold=True, color="FFFFFF")
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    by_id = {c.clip_id: c for c in clips}
    for ri, row in enumerate(rows, start=2):
        ws.append([row[hname] for hname in headers])
        s = summaries.get(row["Clip ID"])
        verdicts = _verdict_map(s) if s else {}
        result = row["QC result"]
        rc = ws.cell(row=ri, column=headers.index("QC result") + 1)
        if result in FILL and FILL[result]:
            rc.fill = FILL[result]
        if result in FONT_RESULT:
            rc.font = FONT_RESULT[result]
        for header, _, tid in cols:
            if tid and verdicts.get(tid) in ("PASS", "FAIL"):
                ws.cell(row=ri, column=headers.index(header) + 1).fill = FILL[verdicts[tid]]
        if not by_id[row["Clip ID"]].available:
            for ci in range(1, len(headers) + 1):
                if ws.cell(row=ri, column=ci).value in ("", None):
                    ws.cell(row=ri, column=ci).fill = FILL["SOURCE MISSING"]
        url_cell = ws.cell(row=ri, column=headers.index("Source URL") + 1)
        if row["Source URL"]:
            url_cell.hyperlink = row["Source URL"]
            url_cell.font = Font(color="1F5FBF", underline="single")
        lm_cell = ws.cell(row=ri, column=headers.index("Landmarks file") + 1)
        if row["Landmarks file"]:
            lm_cell.hyperlink = row["Landmarks file"]
            lm_cell.font = Font(color="1F5FBF", underline="single")
    widths = {"Clip no": 7, "Clip ID": 34, "Dataset": 9, "Category": 26, "Task": 26, "Camera": 15, "Layout": 9,
              "QC result": 16, "Tests passed": 9, "Failed tests": 60, "Landmarks file": 40, "Source URL": 60,
              "Original reference": 40, "Notes": 40, "Resolution": 11}
    for ci, hname in enumerate(headers, start=1):
        ws.column_dimensions[get_column_letter(ci)].width = widths.get(hname, 12)
    reason_col = get_column_letter(headers.index("Failed tests") + 1)
    for row_cells in ws.iter_rows(min_row=2):
        for cell in row_cells:
            cell.border = Border(top=THIN, bottom=THIN, left=THIN, right=THIN)
            cell.alignment = Alignment(vertical="top", wrap_text=cell.column_letter == reason_col)
    ws.row_dimensions[1].height = 45
    ws.freeze_panes = ws.cell(row=2, column=headers.index("QC result") + 2)
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(rows) + 1}"

    # --- Summary ---
    sm = wb.create_sheet("Summary")
    sm.append(["Hand & Technical QC - summary"])
    sm["A1"].font = Font(bold=True, size=14)
    sm.append([f"Dataset: {context['dataset']} (clip list {context['source_csv']}) | Generated {generated} | "
               f"handqc {__version__}"])
    sm.append([])
    counts = {}
    for row in rows:
        counts[row["QC result"]] = counts.get(row["QC result"], 0) + 1
    sm.append(["QC result", "Clips"])
    for cell in sm[sm.max_row]:
        cell.font = Font(bold=True)
    for key in ("PASS", "FAIL", "ERROR", "SOURCE MISSING"):
        sm.append([key, counts.get(key, 0)])
        cell = sm.cell(row=sm.max_row, column=1)
        if FILL.get(key):
            cell.fill = FILL[key]
    sm.append(["Total", len(rows)])
    sm.append([])
    sm.append(["Test", "Group", "Threshold", "PASS", "FAIL", "Reported only / N/A", "Pass rate (graded)"])
    header_row = sm.max_row
    for ci in range(1, 8):
        sm.cell(row=header_row, column=ci).font = Font(bold=True, color="FFFFFF")
        sm.cell(row=header_row, column=ci).fill = HEADER_FILL["id"]
    for tid in TEST_ORDER:
        tests = [t for s in summaries.values() for t in s["tests"] if t["id"] == tid]
        if not tests:
            continue
        p = sum(1 for t in tests if t["status"] == "PASS")
        f = sum(1 for t in tests if t["status"] == "FAIL")
        sm.append([tests[0]["name"], tests[0]["group"], tests[0]["threshold"] if tid != "bitrate" else
                   f"{config['tests']['bitrate_mbps']['min_at_1080p']} Mbit/s at 1080p, scaled by resolution",
                   p, f, len(tests) - p - f, f"{100 * p / (p + f):.0f}%" if p + f else "-"])
    for ci, w in enumerate([28, 11, 60, 8, 8, 18, 18], start=1):
        sm.column_dimensions[get_column_letter(ci)].width = w

    # --- Thresholds ---
    th = wb.create_sheet("Thresholds")
    th.append(["Setting", "Value"])
    for cell in th[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = HEADER_FILL["id"]

    def flatten(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from flatten(f"{prefix}.{k}" if prefix else k, v)
        else:
            yield prefix, json.dumps(obj) if isinstance(obj, (list, dict)) else obj
    for section in ("tests", "frame_quality", "wearer", "grasp", "tracking", "analysis"):
        for k, v in flatten(section, config[section]):
            th.append([k, v])
    th.column_dimensions["A"].width = 50
    th.column_dimensions["B"].width = 60

    # --- Definitions ---
    de = wb.create_sheet("Definitions")
    de.append(["Metric", "Definition"])
    for cell in de[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = HEADER_FILL["id"]
    for name, text in DEFINITIONS:
        de.append([name, text])
        de.cell(row=de.max_row, column=2).alignment = Alignment(wrap_text=True, vertical="top")
        de.cell(row=de.max_row, column=1).alignment = Alignment(vertical="top")
    de.column_dimensions["A"].width = 28
    de.column_dimensions["B"].width = 120

    tmp = path.with_name(path.name + ".tmp")
    wb.save(tmp)
    tmp.replace(path)


DEFINITIONS = [
    ("Analysed frames", "Every 4th frame of ~30 fps video, every 8th of ~60 fps, every 20th of ~120 fps "
                        "(6-7.5 analysed frames per second). All percentages are shares of analysed frames; durations "
                        "count each analysed frame as its frame interval."),
    ("Camera views", "Mono: the full frame. Stereo (side by side): left and right eye analysed separately and "
                     "combined. Six-cam (3x2 grid): six tiles analysed separately and combined. Combined = a hand "
                     "counts as present when it is seen in any view."),
    ("Wearer's hands", "Hand metrics count only the camera wearer's hands. A detected hand is someone else's when its "
                       "size (diagonal of the box around its 21 landmarks / view height) is below wearer.min_hand_size "
                       "(far away), or when it points down the frame (reaching towards the camera) while smaller than "
                       "wearer.large_hand_size (only the wearer's own hands come that close). At most one left and one "
                       "right wearer hand per view; a hand detected twice is counted once. In footage where the near "
                       "hands enter from the top (upside down) the direction rule is mirrored. Other people's hands "
                       "are reported as other_hands_pct."),
    ("Left / Right", "MediaPipe handedness (verified on this footage to be correct for non-mirrored video). If both "
                     "wearer hands get the same label, the one further left in the image is the left hand."),
    ("21 landmarks", "MediaPipe Hand Landmarker points 0-20 (wrist, thumb CMC/MCP/IP/tip, index/middle/ring/pinky "
                     "MCP/PIP/DIP/tip) as x, y pixel coordinates of the original camera view (origin top-left; for "
                     "stereo/six-cam, of that eye/tile)."),
    ("either_hand_pct", "Share of analysed frames with at least one wearer hand."),
    ("both_hands_pct", "Share of analysed frames with both wearer hands."),
    ("left_hand_pct / right_hand_pct", "Share of analysed frames with the wearer's left / right hand."),
    ("untracked_pct", "Share of analysed frames with no wearer hand (= 100 - either_hand_pct)."),
    ("Grasp (frame)", "A wearer hand is in a grasp pose when its 4 fingers are curled (mean flexion of the MCP, PIP "
                      "and DIP joints at/above the configured angle: power grasp) OR its thumb tip is close to the index "
                      "or middle finger tip (distance / palm length at/below the configured ratio: precision grasp). "
                      "Computed on MediaPipe's 3D hand model, so independent of the viewing angle."),
    ("grasp_count / grasp_duration", "A grasp event is a grasp pose held for at least the minimum duration (short "
                                     "dropouts bridged), counted per hand. grasp_duration = total and mean seconds."),
    ("HOI_pct", "Share of analysed frames inside a grasp event of either hand (hand-object interaction estimated "
                "from hand shape; it cannot see the object itself)."),
    ("tracking_losses", "Times a hand that had been tracked for at least min_tracked_s disappears for at least "
                        "min_lost_s, counted per hand (left + right). Also given per minute."),
    ("longest_hand_gap", "Longest continuous time with no wearer hand, in seconds."),
    ("other_hands_pct", "Share of analysed frames in which a hand that is not the wearer's is visible."),
    ("Duration / FPS / Bitrate / Codec", "From the container (ffprobe). Bitrate = video stream bitrate; required "
                                         "bitrate = configured value at 1920x1080 scaled by pixel count."),
    ("Decode errors", "Every error message from a full FFmpeg decode of every frame, plus an incomplete decode "
                      "(missing frames, non-zero exit)."),
    ("Luminance", "Mean BT.709 luma (0-255) over analysed frames; frames whose mean is below the dark or above the "
                  "bright limit count as badly exposed."),
    ("Sharpness", "Variance of the Laplacian of the luma at analysis resolution (views scaled to <= 1280 px wide); "
                  "a frame is sharp at/above the configured value."),
    ("Blockiness", "Visible compression-block edges on full-resolution frames (2 per second): among pixel boundaries "
                   "with flat surroundings (neighbouring differences <= 1 grey level), the share with a visible step "
                   "(>= 3 grey levels) on the strongest 8-64 px block grid minus the share everywhere else, in "
                   "percentage points (0 = no visible block edges). On these clips clean footage scored 0.5-2.5 and "
                   "visibly blocky footage 3.4 and up. FFmpeg's blockdetect is reported for information only (it missed "
                   "the large blocks of heavily compressed 4K footage)."),
    ("Colour cast", "White-balance error: the mean CIELAB a*/b* of the near-neutral pixels (grey/white surfaces, "
                    "chroma < 15, mid lightness), which take on the colour of a cast, as a distance from neutral "
                    "(Lab units). Strongly coloured content such as blue ESD mats is excluded, so it is not "
                    "mistaken for a cast; the whole-frame mean is reported separately as 'scene colour' and not "
                    "tested. Not measurable when fewer than the configured share of frames contain enough "
                    "neutral surfaces."),
    ("Orientation", "Rotation metadata must be a multiple of 90 deg (FFmpeg applies it, as players do), and in "
                    "upright egocentric footage the wearer's hands point up the frame (wrist below knuckles). Only "
                    "the nearest hands are used (the largest quarter of the clip's hands; other people's hands face "
                    "the camera). On these clips upright footage scored 60-100% and upside-down footage 0-12%. "
                    "Not applied to six-cam tiles, which are mounted at fixed rotations."),
    ("HOI_pct (limitation)", "Counts grasping hand shapes only. Flat-hand contact (pressing a sticker, pushing, "
                             "supporting) is not counted, so HOI_pct is a conservative lower bound of hand-object "
                             "interaction."),
]


# ----------------------------------------------------------------------------- HTML
def _write_html(path: Path, clips, summaries, errors, rows, config, generated: str, context: dict) -> None:
    data = []
    for row in rows:
        s = summaries.get(row["Clip ID"])
        item = {"no": row["Clip no"], "id": row["Clip ID"], "dataset": row["Dataset"], "category": row["Category"],
                "task": row["Task"], "camera": row["Camera"], "layout": row["Layout"], "result": row["QC result"],
                "passed": row["Tests passed"], "reason": row["Failed tests"], "url": row["Source URL"],
                "ref": row["Original reference"]}
        if s:
            item.update({"tests": s["tests"], "technical": s["technical"], "hands": s["hands"],
                         "per_view": s["per_view"], "preview": f"videos/{row['Clip ID']}/{s['files']['preview']}"
                         if s["files"].get("preview") else None, "views": s["clip"]["views"]})
        data.append(item)
    payload = json.dumps({"generated": generated, "version": __version__, "clips": data, **context,
                          "definitions": DEFINITIONS, "test_order": TEST_ORDER}, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")
    text = HTML_TEMPLATE.replace("/*__DATA__*/null", payload)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def build_reports(clips: list[Clip], summaries: dict, errors: dict, config: dict, results: Path, *,
                  dataset: str, source_csv: str) -> list[Path]:
    generated = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    context = {"dataset": dataset, "source_csv": source_csv}
    rows = _rows(clips, summaries, errors)
    headers = ID_COLUMNS + [h for h, _, _ in _columns()] + TAIL_COLUMNS
    csv_path, xlsx_path, html_path = (results / "qc_checklist.csv", results / "qc_checklist.xlsx",
                                      results / "qc_report.html")
    write_csv(csv_path, headers, rows)
    _write_xlsx(xlsx_path, clips, summaries, errors, rows, config, generated, context)
    _write_html(html_path, clips, summaries, errors, rows, config, generated, context)
    return [xlsx_path, csv_path, html_path]


HTML_TEMPLATE = Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")
