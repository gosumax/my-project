"""Stream changed seat-stack crops from table observations into a bounded OCR queue.

This is raw visual evidence. It does not assign hands, players or START_STACK.
Queue overflow and OCR failures are explicit in the output summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import sqlite3
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.start_stack import classify_stack_text
from parser_core.tiny_chat_client import TinyChatClient


def material_roi_change(previous: np.ndarray, current: np.ndarray) -> bool:
    """Ignore small overlay animation; keep major text/state transitions.

    This is a candidate filter, not evidence that an action occurred. The raw
    parent table observation remains available for every source frame.
    """
    if previous.shape != current.shape:
        return True
    old = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    new = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(old, new)
    return float(np.mean(diff)) >= 10 and float(np.mean(diff > 20)) >= .12


def table_observations(db: sqlite3.Connection, after: int, limit: int,
                       start_frame: int = 0, end_frame: int | None = None,
                       table_session: str | None = None):
    return db.execute("""SELECT o.rowid,o.frame_id,o.table_session_id,o.crop_path,
        o.crop_sha256,f.source_pts,f.time_base_num,f.time_base_den,s.geometry_json
        FROM observations o JOIN frames f ON f.source_id=o.source_id
          AND f.frame_id=o.frame_id
        JOIN table_sessions s ON s.table_session_id=o.table_session_id
        WHERE o.roi_type='table' AND o.rowid>? AND o.frame_id>=?
          AND (? IS NULL OR o.frame_id<=?)
          AND (? IS NULL OR o.table_session_id=?)
        ORDER BY o.rowid LIMIT ?""",
        (after, start_frame, end_frame, end_frame,
         table_session, table_session, limit)).fetchall()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--watch-until", type=Path)
    parser.add_argument("--tiny-config", type=Path, default=ROOT / "configs/ocr_tiny_local.json")
    parser.add_argument("--queue-size", type=int, default=32)
    parser.add_argument("--max-table-observations", type=int)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--table-session")
    parser.add_argument("--output", type=Path,
                        help="Write evidence outside a read-only source run")
    args = parser.parse_args()
    if (args.queue_size < 1 or args.start_frame < 0 or
            (args.end_frame is not None and args.end_frame < args.start_frame) or
            (args.max_table_observations is not None and args.max_table_observations < 1)):
        parser.error("Queue size and observation limit must be positive")
    run = args.run.resolve(strict=True)
    output_dir = args.output.resolve() if args.output else run
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = ROOT / "configs/layout_847x404.json"
    profile_bytes = profile_path.read_bytes()
    profile_sha256 = hashlib.sha256(profile_bytes).hexdigest()
    profile = json.loads(profile_bytes)
    config = json.loads(args.tiny_config.read_text("utf-8"))
    db = sqlite3.connect(run / "journal.sqlite3", timeout=30)
    db.execute("PRAGMA busy_timeout=30000")
    jobs: queue.Queue[dict | None] = queue.Queue(maxsize=args.queue_size)
    output = output_dir / "start_stack_observations.jsonl"
    summary_path = output_dir / "start_stack_observer_summary.json"
    crop_dir = output_dir / "start_stack_crops"
    counters = Counter()
    dropped: set[tuple[str, str, str]] = set()
    previous: dict[tuple[str, str, str], np.ndarray] = {}
    errors = []
    lock = threading.Lock()
    started = time.perf_counter()

    def worker():
        client = TinyChatClient(Path(config["python"]), Path(config["model_dir"]),
            config["model_name"], config["weights_sha256"], config["device"],
            output_dir / "start_stack_tiny.stderr.log")
        recognizer_failed = False
        try:
            with output.open("w", encoding="utf-8") as stream:
                while True:
                    job = jobs.get()
                    try:
                        if job is None:
                            return
                        record = {key: value for key, value in job.items() if key != "pixels"}
                        try:
                            crop_dir.mkdir(exist_ok=True)
                            ok, packed = cv2.imencode(".png", job["pixels"])
                            if not ok:
                                raise ValueError("Stack crop encode failed")
                            crop_path = crop_dir / (f"{job['table_id']}_{job['seat']}_{job['roi_type']}_"
                                f"{job['frame_index']}_{job['crop_sha256'][:12]}.png")
                            crop_path.write_bytes(packed.tobytes())
                            record["crop_path"] = str(crop_path.relative_to(output_dir))
                            record["crop_file_sha256"] = hashlib.sha256(packed.tobytes()).hexdigest()
                            item = {"item_id": f"{job['table_context']}:{job['seat']}:{job['roi_type']}:{job['frame_index']}",
                                    "path": str(crop_path),
                                    "source_sha256": record["crop_file_sha256"]}
                            if recognizer_failed:
                                raise RuntimeError("Recognizer disabled after earlier error")
                            response = client.read_batch([item], 1)
                            result = response["results"][0]
                            raw = result.get("raw_text")
                            if job["roi_type"] == "stack":
                                kind, amount = classify_stack_text(raw)
                            else:
                                kind = "NICKNAME" if raw and result.get("score", 0) >= .90 else "UNREADABLE"
                                amount = None
                            record.update(raw_recognition=raw, confidence=result.get("score"),
                                recognition_status=result.get("status"), visual_kind=kind,
                                visible_stack=str(amount) if amount is not None else None)
                        except Exception as exc:
                            recognizer_failed = True
                            record.update(raw_recognition=None, confidence=None,
                                recognition_status="ERROR", visual_kind="UNREADABLE",
                                visible_stack=None, error=repr(exc))
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        with lock:
                            counters["processed"] += 1
                            counters[record["visual_kind"]] += 1
                            if record["recognition_status"] == "ERROR":
                                counters["ocr_errors"] += 1
                    finally:
                        jobs.task_done()
        except BaseException as exc:
            errors.append(f"STACK_WORKER_FAILED: {exc!r}")
        finally:
            try:
                client.close()
            except Exception as exc:
                errors.append(f"STACK_WORKER_CLOSE_FAILED: {exc!r}")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    after = 0
    try:
        while True:
            batch = table_observations(db, after, 100, args.start_frame,
                                       args.end_frame, args.table_session)
            if not batch:
                if args.watch_until is None or args.watch_until.exists():
                    break
                time.sleep(.2)
                continue
            for rowid, frame, session, relative, table_sha, pts, num, den, geometry_raw in batch:
                after = rowid
                counters["table_observations"] += 1
                geometry = json.loads(geometry_raw)
                if geometry.get("profile_sha256") != profile_sha256:
                    raise ValueError("Seat ROI profile differs from capture profile")
                slot_id = int(geometry["slot_id"])
                table_id = f"T{slot_id:02d}"
                path = (run / relative).resolve(strict=True)
                if not path.is_relative_to(run):
                    raise ValueError("Table crop outside run")
                packed_table = np.fromfile(path, dtype=np.uint8)
                if hashlib.sha256(packed_table.tobytes()).hexdigest() != table_sha:
                    raise ValueError("Saved table crop SHA256 mismatch")
                image = cv2.imdecode(packed_table, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Cannot read {path}")
                width, height = profile["frame_dimensions_wh"]
                if image.shape[:2] != (height, width):
                    raise ValueError("Table crop dimensions differ from seat ROI profile")
                # Captured table PNG already uses local 847x404 coordinates.
                for seat, boxes in profile["slots"][f"slot_{slot_id:02d}"]["seats"].items():
                    for roi_type in ("nick", "stack"):
                        box = boxes[roi_type]
                        if box is None:
                            continue
                        x1, y1, x2, y2 = box
                        pixels = image[y1:y2, x1:x2]
                        key = (session, seat, roi_type)
                        if key in previous and not material_roi_change(previous[key], pixels):
                            continue
                        previous[key] = np.ascontiguousarray(pixels)
                        job = {"table_id": table_id, "table_context": session,
                               "seat": seat, "roi_type": roi_type,
                               "frame_index": frame, "source_pts": pts,
                               "timestamp": pts * num / den if pts is not None and num and den else None,
                               "table_crop_sha256": table_sha,
                               "crop_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                               "pixels": previous[key]}
                        try:
                            jobs.put_nowait(job)
                            counters["submitted"] += 1
                        except queue.Full:
                            dropped.add(key)
                            counters["dropped_candidates"] += 1
                        counters["max_queue"] = max(counters["max_queue"], jobs.qsize())
                if args.max_table_observations and counters["table_observations"] >= args.max_table_observations:
                    break
            if args.max_table_observations and counters["table_observations"] >= args.max_table_observations:
                break
    except Exception as exc:
        errors.append(repr(exc))
        raise
    finally:
        while jobs.unfinished_tasks and thread.is_alive():
            time.sleep(.05)
        if thread.is_alive():
            jobs.put(None)
            thread.join(timeout=10)
            if thread.is_alive():
                errors.append("STACK_WORKER_DID_NOT_EXIT")
        else:
            while True:
                try:
                    jobs.get_nowait()
                except queue.Empty:
                    break
                jobs.task_done()
        db.close()
        summary_path.write_text(json.dumps({"status": "PARTIAL_RAW_EVIDENCE" if dropped or errors or counters["ocr_errors"] else "RAW_EVIDENCE_ONLY",
            "metrics": dict(counters), "dropped_seats": sorted([list(key) for key in dropped]),
            "errors": errors, "elapsed_seconds": time.perf_counter() - started,
            "recognizer": config["model_name"], "model_sha256": config["weights_sha256"]},
            ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    main()
