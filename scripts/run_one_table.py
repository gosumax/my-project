"""Run the bounded, non-certifying one-table pilot from video or saved evidence."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.chat_rows import ROW_TRACKER_VERSION
from parser_core.message_assembly import ASSEMBLER_VERSION
from parser_core.normalize_chat import NORMALIZER_VERSION
from scripts.ocr_chat_pilot import OCR_POLICY, ROW_OCR_RETRY_POLICY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--table-roi", help="x1,y1,x2,y2 in source pixels")
    parser.add_argument("--chat-roi", help="x1,y1,x2,y2 in source pixels")
    parser.add_argument("--layout-id", default="configured_fixed_roi_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--include-table", action="store_true")
    parser.add_argument("--skip-capture", action="store_true",
                        help="Replay derived stages from the existing raw journal")
    parser.add_argument("--tiny-config", type=Path,
                        default=ROOT / "configs" / "ocr_tiny_local.json")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.skip_capture:
        if not (output / "journal.sqlite3").is_file():
            parser.error("--skip-capture requires an existing journal.sqlite3")
    elif not all((args.video, args.table_roi, args.chat_roi, args.max_frames)):
        parser.error("Video, both ROIs and a positive --max-frames are required")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.start_frame < 0:
        parser.error("--start-frame must be nonnegative")
    output.mkdir(parents=True, exist_ok=True)
    stages = []
    started = datetime.now(timezone.utc).isoformat()
    commands = []
    if not args.skip_capture:
        capture = [sys.executable, str(ROOT / "scripts" / "capture_video.py"),
                   str(args.video), "--table-roi", args.table_roi,
                   "--chat-roi", args.chat_roi, "--layout-id", args.layout_id,
                   "--output", str(output), "--start-frame", str(args.start_frame),
                   "--max-frames", str(args.max_frames)]
        if not args.include_table:
            capture.append("--chat-only")
        commands.append(("capture", capture))
    for stage, script, extra in (
        ("track_rows", "track_chat_rows.py", []),
        ("ocr_rows", "ocr_chat_pilot.py", ["--roi-type", "CHAT_ROW",
                                           "--tiny-config", str(args.tiny_config)]),
        ("assemble_messages", "assemble_messages.py", []),
        ("normalize_messages", "normalize_messages.py", []),
        ("reduce_hands", "reduce_hands.py", []),
        ("export_diagnostics", "export_diagnostics.py", []),
    ):
        commands.append((stage, [sys.executable, str(ROOT / "scripts" / script),
                                 str(output), *extra]))
    status = "FAILED"
    try:
        for stage, command in commands:
            tick = time.perf_counter()
            subprocess.run(command, check=True, cwd=ROOT)
            stages.append({"stage": stage, "wall_seconds": round(time.perf_counter() - tick, 3)})
        status = "PARTIAL_PIPELINE"
    finally:
        capture_state = None
        source_path = str(args.video.resolve()) if args.video else None
        frame_range = None
        database = output / "journal.sqlite3"
        if database.is_file():
            with sqlite3.connect(database) as db:
                source_row = db.execute("SELECT path,state FROM sources LIMIT 1").fetchone()
                frame_row = db.execute("SELECT MIN(frame_id),MAX(frame_id) FROM frames").fetchone()
                if source_row:
                    source_path, capture_state = source_row
                if frame_row[0] is not None:
                    frame_range = list(frame_row)
        report = {"status": status, "started_utc": started,
                  "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "source": source_path, "source_state": capture_state,
                  "output": str(output), "captured_frame_range": frame_range,
                  "requested_start_frame": None if args.skip_capture else args.start_frame,
                  "requested_max_frames": args.max_frames,
                  "versions": {"row_tracker": ROW_TRACKER_VERSION,
                               "ocr_policy": OCR_POLICY,
                               "row_ocr_retry": ROW_OCR_RETRY_POLICY,
                               "tiny_config": str(args.tiny_config.resolve()),
                               "message_assembler": ASSEMBLER_VERSION,
                               "normalizer": NORMALIZER_VERSION},
                  "stages": stages,
                  "performance_status": "PILOT_TIMING_NOT_FULL_BENCHMARK",
                  "hh_status": "NO_VERIFIED_EXPORT"}
        (output / "pilot_run.json").write_text(json.dumps(report, ensure_ascii=False,
                                                      indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
