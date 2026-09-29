"""Run or replay the bounded nine-slot pilot without HH certification."""

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
    parser.add_argument("--layout", type=Path,
                        default=ROOT / "configs" / "nine_table_layout.json")
    parser.add_argument("--profile", type=Path,
                        default=ROOT / "configs" / "layout_847x404.json")
    parser.add_argument("--calibration", type=Path,
                        default=ROOT / "configs" / "chat_log_bottom_rois.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--skip-capture", action="store_true")
    parser.add_argument("--tiny-config", type=Path,
                        default=ROOT / "configs" / "ocr_tiny_local.json")
    args = parser.parse_args()
    if args.skip_capture:
        if not (args.output / "journal.sqlite3").is_file():
            parser.error("--skip-capture requires an existing journal.sqlite3")
    elif args.video is None or args.max_frames is None:
        parser.error("--video and --max-frames are required for capture")
    if args.start_frame < 0 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("--start-frame must be nonnegative and --max-frames positive")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    stages = []
    commands = []
    if not args.skip_capture:
        commands.append(("capture", [sys.executable,
            str(ROOT / "scripts" / "capture_nine_tables.py"), str(args.video),
            "--layout", str(args.layout), "--profile", str(args.profile),
            "--calibration", str(args.calibration),
            "--output", str(output), "--start-frame", str(args.start_frame),
            "--max-frames", str(args.max_frames)]))
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
            stages.append({"stage": stage,
                           "wall_seconds": round(time.perf_counter() - tick, 3)})
        status = "PARTIAL_PIPELINE"
    finally:
        source_state = None
        frame_range = None
        counts = {}
        current_version_counts = {}
        database = output / "journal.sqlite3"
        if database.is_file():
            with sqlite3.connect(database) as db:
                row = db.execute("SELECT state FROM sources LIMIT 1").fetchone()
                source_state = row[0] if row else None
                frame_range = db.execute("SELECT MIN(frame_id),MAX(frame_id) FROM frames").fetchone()
                for table in ("table_sessions", "frames", "observations",
                              "recognitions", "messages", "event_revisions",
                              "hand_revisions", "exports"):
                    counts[table] = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                current_version_counts["messages"] = db.execute(
                    "SELECT COUNT(*) FROM messages WHERE assembler_version=?",
                    (ASSEMBLER_VERSION,)).fetchone()[0]
                current_version_counts["events"] = db.execute("""SELECT COUNT(*)
                    FROM event_revisions e JOIN messages m ON m.message_id=e.message_id
                    WHERE m.assembler_version=? AND e.revision=(
                        SELECT MAX(e2.revision) FROM event_revisions e2
                        WHERE e2.event_id=e.event_id)""", (ASSEMBLER_VERSION,)).fetchone()[0]
        if status == "PARTIAL_PIPELINE":
            hand_report = json.loads((output / "partial_hands.json").read_text(encoding="utf-8"))
            current_version_counts["partial_hand_windows"] = len(hand_report["hand_windows"])
        report = {"status": status, "started_utc": started,
                  "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "source_state": source_state, "captured_frame_range": frame_range,
                  "counts": counts, "current_version_counts": current_version_counts,
                  "stages": stages,
                  "versions": {"row_tracker": ROW_TRACKER_VERSION,
                               "ocr_policy": OCR_POLICY,
                               "row_ocr_retry": ROW_OCR_RETRY_POLICY,
                               "tiny_config": str(args.tiny_config.resolve()),
                               "message_assembler": ASSEMBLER_VERSION,
                               "normalizer": NORMALIZER_VERSION},
                  "layout_status": "FIXED_EXPERIMENT_ONLY",
                  "table_identity_status": "UNRESOLVED",
                  "hh_status": "NO_VERIFIED_EXPORT"}
        (output / "pilot_run.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
