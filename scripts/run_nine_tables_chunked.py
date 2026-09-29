"""Run nine-slot pilot in capture/track/OCR waves before final assembly.

This is a conservative pipeline bridge: it keeps the existing journal and module
contracts, but starts row tracking and OCR after bounded capture chunks instead
of waiting for the whole video capture to finish.
"""

from __future__ import annotations

import argparse
import json
import os
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


def run_json_stage(name: str, command: list[str], cwd: Path) -> dict:
    tick = time.perf_counter()
    completed = subprocess.run(command, check=True, cwd=cwd, text=True,
                               encoding="utf-8", capture_output=True,
                               env={**os.environ, "PYTHONUTF8": "1"})
    wall = time.perf_counter() - tick
    parsed = None
    stdout = completed.stdout.strip()
    if stdout:
        try:
            if "{" not in stdout:
                raise ValueError("Stage output is plain text")
            parsed = json.loads(stdout[stdout.find("{"):])
        except Exception:
            parsed = {"stdout_tail": stdout[-4000:]}
    return {"stage": name, "wall_seconds": round(wall, 3),
            "details": parsed, "stderr_tail": completed.stderr[-4000:]}


def db_snapshot(database: Path) -> dict:
    if not database.is_file():
        return {}
    with sqlite3.connect(database) as db:
        tables = ("table_sessions", "frames", "observations", "recognitions",
                  "physical_rows", "row_decisions", "row_suppressions",
                  "messages", "event_revisions", "hand_revisions", "exports")
        result = {table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                  for table in tables}
        result["frame_range"] = db.execute(
            "SELECT MIN(frame_id),MAX(frame_id) FROM frames").fetchone()
        row = db.execute("SELECT state FROM sources LIMIT 1").fetchone()
        result["source_state"] = row[0] if row else None
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--layout", type=Path,
                        default=ROOT / "configs" / "nine_table_layout.json")
    parser.add_argument("--profile", type=Path,
                        default=ROOT / "configs" / "layout_847x404.json")
    parser.add_argument("--calibration", type=Path,
                        default=ROOT / "configs" / "chat_log_bottom_rois.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--chunk-frames", type=int, default=1000)
    parser.add_argument("--db-batch-frames", type=int, default=25)
    parser.add_argument("--ocr-batch-size", type=int, default=8)
    parser.add_argument("--tiny-config", type=Path,
                        default=ROOT / "configs" / "ocr_tiny_local.json")
    args = parser.parse_args()
    if (args.start_frame < 0 or args.max_frames < 1 or args.chunk_frames < 1 or
            args.ocr_batch_size < 1):
        parser.error("frame arguments must be positive")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    stages = []
    status = "FAILED"
    database = output / "journal.sqlite3"
    try:
        remaining = args.max_frames
        chunk_start = args.start_frame
        chunk_index = 0
        while remaining > 0:
            chunk_index += 1
            chunk_count = min(args.chunk_frames, remaining)
            prefix = f"chunk_{chunk_index:03d}"
            stages.append(run_json_stage(prefix + "_capture", [
                sys.executable, str(ROOT / "scripts" / "capture_nine_tables.py"),
                str(args.video), "--layout", str(args.layout),
                "--profile", str(args.profile),
                "--calibration", str(args.calibration),
                "--output", str(output),
                "--start-frame", str(chunk_start),
                "--max-frames", str(chunk_count),
                "--db-batch-frames", str(args.db_batch_frames),
            ], ROOT))
            stages[-1]["snapshot"] = db_snapshot(database)
            stages.append(run_json_stage(prefix + "_track_rows", [
                sys.executable, str(ROOT / "scripts" / "track_chat_rows.py"),
                str(output),
            ], ROOT))
            stages[-1]["snapshot"] = db_snapshot(database)
            stages.append(run_json_stage(prefix + "_ocr_rows", [
                sys.executable, str(ROOT / "scripts" / "ocr_chat_pilot.py"),
                str(output), "--roi-type", "CHAT_ROW",
                "--tiny-config", str(args.tiny_config),
                "--batch-size", str(args.ocr_batch_size),
            ], ROOT))
            stages[-1]["snapshot"] = db_snapshot(database)
            chunk_start += chunk_count
            remaining -= chunk_count
        for stage, script in (
            ("assemble_messages", "assemble_messages.py"),
            ("normalize_messages", "normalize_messages.py"),
            ("reduce_hands", "reduce_hands.py"),
            ("export_diagnostics", "export_diagnostics.py"),
        ):
            stages.append(run_json_stage(stage, [
                sys.executable, str(ROOT / "scripts" / script), str(output),
            ], ROOT))
            stages[-1]["snapshot"] = db_snapshot(database)
        status = "PARTIAL_PIPELINE"
    finally:
        snapshot = db_snapshot(database)
        current_version_counts = {}
        if database.is_file():
            with sqlite3.connect(database) as db:
                current_version_counts["messages"] = db.execute(
                    "SELECT COUNT(*) FROM messages WHERE assembler_version=?",
                    (ASSEMBLER_VERSION,)).fetchone()[0]
                current_version_counts["events"] = db.execute("""SELECT COUNT(*)
                    FROM event_revisions e JOIN messages m ON m.message_id=e.message_id
                    WHERE m.assembler_version=? AND e.revision=(
                        SELECT MAX(e2.revision) FROM event_revisions e2
                        WHERE e2.event_id=e.event_id)""", (ASSEMBLER_VERSION,)).fetchone()[0]
        if status == "PARTIAL_PIPELINE" and (output / "partial_hands.json").is_file():
            hand_report = json.loads((output / "partial_hands.json").read_text(encoding="utf-8"))
            current_version_counts["partial_hand_windows"] = len(hand_report["hand_windows"])
        report = {"status": status, "started_utc": started,
                  "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "counts": snapshot,
                  "current_version_counts": current_version_counts,
                  "stages": stages,
                  "versions": {"row_tracker": ROW_TRACKER_VERSION,
                               "ocr_policy": OCR_POLICY,
                               "row_ocr_retry": ROW_OCR_RETRY_POLICY,
                               "tiny_config": str(args.tiny_config.resolve()),
                               "message_assembler": ASSEMBLER_VERSION,
                               "normalizer": NORMALIZER_VERSION},
                  "runner": {"mode": "chunked_capture_track_ocr",
                             "chunk_frames": args.chunk_frames,
                             "db_batch_frames": args.db_batch_frames,
                             "ocr_batch_size": args.ocr_batch_size},
                  "layout_status": "FIXED_EXPERIMENT_ONLY",
                  "table_identity_status": "UNRESOLVED",
                  "hh_status": "NO_VERIFIED_EXPORT"}
        (output / "pilot_run.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
