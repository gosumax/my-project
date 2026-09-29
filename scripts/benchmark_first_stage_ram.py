"""First-stage Smoke14 RAM path; no OCR, journal mutation, or crop files.

The historical run is read only after timing, solely as a parity oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import av
import cv2
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.chat_rows import (Alignment, align_upward, exclude_first_visible_span,
                                   match_rows, padded_row_span, physical_spans,
                                   row_distance)
from parser_core.layout import load_fixed_layout
from parser_core.ram_first_stage import FirstStage


ROOT = Path(__file__).resolve().parents[1]


def historical_gray(rgb: np.ndarray) -> np.ndarray:
    """Match the original PIL RGB PNG -> OpenCV IMREAD_GRAYSCALE pixels in RAM."""
    ok, packed = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise ValueError("RAM chat conversion failed")
    gray = cv2.imdecode(packed, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("RAM chat grayscale decode failed")
    return gray


def resume_artifact_groups(db: sqlite3.Connection, baseline: Path, frames: int):
    """Prove which historical row decisions came from an empty resume checkpoint.

    A group is excluded only when the checkpoint had prior row history but no
    row decisions for that session and every row at its first changed chat ROI
    matches the previous chat crop at the same position and shift zero.
    """
    summary_path = baseline / "row_tracking_summary.json"
    if not summary_path.exists():
        return {}
    checkpoint = json.loads(summary_path.read_text("utf-8")).get("resumed_after_frame", -1)
    if checkpoint < 0:
        return {}
    groups = {}
    sessions = [row[0] for row in db.execute("SELECT table_session_id FROM table_sessions")]
    for sid in sessions:
        prior_count = db.execute("""SELECT COUNT(*) FROM row_decisions d
            JOIN observations o ON o.observation_id=d.observation_id
            WHERE o.table_session_id=? AND o.frame_id<?""", (sid, checkpoint)).fetchone()[0]
        checkpoint_count = db.execute("""SELECT COUNT(*) FROM row_decisions d
            JOIN observations o ON o.observation_id=d.observation_id
            WHERE o.table_session_id=? AND o.frame_id=?""", (sid, checkpoint)).fetchone()[0]
        if not prior_count or checkpoint_count:
            continue
        first = db.execute("""SELECT frame_id,crop_path FROM observations
            WHERE table_session_id=? AND roi_type='chat' AND frame_id>?
            AND json_extract(quality_json,'$.capture_disposition')='SAVED_PIXEL_CHANGE'
            ORDER BY frame_id LIMIT 1""", (sid, checkpoint)).fetchone()
        if first is None or first[0] >= frames:
            continue
        fid, current_path = first
        before = db.execute("""SELECT frame_id,crop_path FROM observations
            WHERE table_session_id=? AND roi_type='chat' AND frame_id<?
            ORDER BY frame_id DESC LIMIT 1""", (sid, fid)).fetchone()
        if before is None:
            continue
        old_gray = cv2.imread(str(baseline / before[1]), cv2.IMREAD_GRAYSCALE)
        new_gray = cv2.imread(str(baseline / current_path), cv2.IMREAD_GRAYSCALE)
        if old_gray is None or new_gray is None:
            raise FileNotFoundError(f"Missing resume evidence for {sid} at {fid}")
        old_spans, _ = exclude_first_visible_span(physical_spans(old_gray))
        new_spans, suppressed = exclude_first_visible_span(physical_spans(new_gray))
        matches = match_rows(old_gray, old_spans, new_gray, new_spans, 0)
        if len(matches) != len(old_spans) or len(matches) != len(new_spans):
            continue
        distances = [row_distance(old_gray, old_spans[old_index],
                                  new_gray, new_spans[new_index])
                     for new_index, old_index in matches.items()]
        groups[(fid, sid)] = {
            "frame_id": fid, "table_session_id": sid,
            "checkpoint_frame_id": checkpoint, "previous_frame_id": before[0],
            "matched_rows": len(matches), "max_row_distance": max(distances, default=0.0),
            "matched_source_span_indices": [index + len(suppressed)
                                            for index in sorted(matches)],
            "previous_chat_gray_sha256": hashlib.sha256(old_gray.tobytes()).hexdigest(),
            "current_chat_gray_sha256": hashlib.sha256(new_gray.tobytes()).hexdigest(),
        }
    return groups


def expected_rows(baseline: Path, frames: int, compare_start: int = 0):
    with sqlite3.connect(baseline / "journal.sqlite3") as db:
        artifact_groups = resume_artifact_groups(db, baseline, frames)
        rows = db.execute("""SELECT o.frame_id,o.table_session_id,o.quality_json,
                d.decision,o.crop_path FROM row_decisions d JOIN observations o
                ON o.observation_id=d.observation_id
                WHERE o.frame_id>=? AND o.frame_id<? AND d.decision IN
                ('INITIAL_VISIBLE','NEW_CANDIDATE','UNRESOLVED_IDENTITY')
                ORDER BY o.frame_id,o.table_session_id,o.bbox_json""",
                (compare_start, frames)).fetchall()
        changes = db.execute("""SELECT frame_id,table_session_id,roi_type,
                quality_json FROM observations
                WHERE frame_id<? AND roi_type IN ('chat','table')
                ORDER BY frame_id,table_session_id,roi_type""", (frames,)).fetchall()
    expected = {}
    excluded_rows = []
    raw_rows = len(rows)
    for fid, sid, quality, decision, path in rows:
        index = json.loads(quality)["source_span_index"]
        group = artifact_groups.get((fid, sid))
        if (decision == "UNRESOLVED_IDENTITY" and group is not None
                and index in group["matched_source_span_indices"]):
            excluded_rows.append((fid, sid, index))
            continue
        gray = cv2.imread(str(baseline / path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(baseline / path)
        expected[(fid, sid, index)] = (decision, hashlib.sha256(gray.tobytes()).hexdigest())
    changed = set()
    raw_changes = 0
    false_old_changes = []
    missed_old_changes = []
    previous = {}
    for fid, sid, roi, quality in changes:
        info = json.loads(quality)
        digest = info["native_pixel_sha256"]
        key = (sid, roi)
        prior = previous.get(key)
        actual_change = prior is None or digest != prior[0]
        reported_change = info["capture_disposition"] == "SAVED_PIXEL_CHANGE"
        if fid >= compare_start:
            raw_changes += reported_change
            if actual_change:
                changed.add((fid, sid, roi))
            elif reported_change:
                false_old_changes.append({"frame_id": fid, "table_session_id": sid,
                    "roi_type": roi, "previous_frame_id": fid-1,
                    "previous_pixel_sha256": prior[0], "current_pixel_sha256": digest})
            if actual_change and not reported_change:
                missed_old_changes.append((fid, sid, roi))
        previous[key] = (digest, fid)
    evidence = {"raw_old_rows": raw_rows, "excluded_resume_rows": excluded_rows,
                "resume_groups": [g for g in artifact_groups.values()
                                  if g["frame_id"] >= compare_start],
                "raw_old_changes": raw_changes,
                "false_old_changes": false_old_changes,
                "missed_old_changes": missed_old_changes}
    return expected, changed, evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--compare-start-frame", type=int, default=0,
                   help="Process from frame zero for state, compare only this frame onward")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.frames < 1:
        p.error("--frames must be positive")
    if not 0 <= args.compare_start_frame < args.frames:
        p.error("--compare-start-frame must lie within the processed frames")
    layout = load_fixed_layout(ROOT / "configs/nine_table_layout.json",
                               ROOT / "configs/layout_847x404.json",
                               ROOT / "configs/chat_log_bottom_rois.json")
    sessions = json.loads((args.baseline / "summary.json").read_text("utf-8"))["table_sessions"]
    process = psutil.Process()
    io_before = process.io_counters()
    cpu_before = time.process_time()
    started = time.perf_counter()
    peak_rss = process.memory_info().rss
    stage = FirstStage(args.video, sessions, args.frames, root=ROOT)
    for _row in stage.rows():
        if stage.seen_frames % 500 == 0:
            peak_rss = max(peak_rss, process.memory_info().rss)
    seen_frames = stage.seen_frames
    detected = {key: value for key, value in stage.detected.items()
                if key[0] >= args.compare_start_frame}
    changed = {key for key in stage.changed if key[0] >= args.compare_start_frame}
    metrics = stage.metrics
    wall = time.perf_counter() - started
    cpu = time.process_time() - cpu_before
    peak_rss = max(peak_rss, process.memory_info().rss)
    io_after = process.io_counters()
    expected, old_changed, oracle = expected_rows(
        args.baseline, seen_frames, args.compare_start_frame)
    keys_old, keys_new = set(expected), set(detected)
    wrong = [key for key in keys_old & keys_new if expected[key] != detected[key]]
    result = {
        "mode": "RAM_FIRST_STAGE_ONLY", "frames": seen_frames,
        "comparison_start_frame": args.compare_start_frame,
        "wall_seconds": wall, "frames_per_second": seen_frames / wall,
        "process_cpu_seconds": cpu,
        "peak_rss_mb": peak_rss / 1048576,
        "process_write_bytes": io_after.write_bytes - io_before.write_bytes,
        "png_files_written": 0, "png_disk_bytes": 0,
        "metrics": dict(metrics),
        "parity": {"old_rows_raw": oracle["raw_old_rows"],
                   "old_rows": len(expected), "new_rows": len(detected),
                   "missing_rows": len(keys_old-keys_new), "extra_rows": len(keys_new-keys_old),
                   "wrong_decision_or_hash": len(wrong),
                   "old_changed_rois_raw": oracle["raw_old_changes"],
                   "old_changed_rois": len(old_changed), "new_changed_rois": len(changed),
                   "change_missing": len(old_changed-changed),
                   "change_extra": len(changed-old_changed),
                   "examples": {"missing": list(sorted(keys_old-keys_new))[:5],
                                "extra": list(sorted(keys_new-keys_old))[:5],
                                "wrong": [(key, expected[key], detected[key]) for key in wrong[:5]]}},
        "oracle_evidence": oracle,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
