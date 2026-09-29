"""Recognize physical chat rows once with PP-OCRv6 tiny, without fallback."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.chat_rows import ROW_TRACKER_VERSION
from parser_core.journal import Journal
from parser_core.tiny_chat_client import TinyChatClient


ROOT = Path(__file__).resolve().parents[1]
OCR_POLICY = "PP_OCRV6_TINY_PRIMARY_SINGLE_PASS_V1"
ROW_OCR_RETRY_POLICY = "NONE_SINGLE_PASS"
ENGINE = "PP-OCRv6_tiny_rec"


def existing_reading(journal: Journal, observation_id: str, version: str,
                     preprocessing: dict, engine: str = ENGINE):
    rows = journal.db.execute("""SELECT recognition_id,revision,raw_text,status,
        preprocessing_json
        FROM recognitions WHERE observation_id=? AND engine=?
          AND model_version=?
          AND status IN ('RAW_UNREVIEWED','UNREADABLE','ERROR')
        ORDER BY revision DESC""", (observation_id, engine, version)).fetchall()
    for row in rows:
        stored = json.loads(row[4])
        stored.pop("raw_score", None)
        if stored == preprocessing:
            return row[:4]
    return None


def rows_for_ocr(journal: Journal) -> list[tuple]:
    return journal.db.execute("""SELECT o.observation_id,o.crop_path,o.crop_sha256,
        p.physical_row_id,o.frame_id,p.table_session_id
        FROM physical_rows p JOIN row_observations ro ON ro.physical_row_id=p.physical_row_id
        JOIN observations o ON o.observation_id=ro.observation_id
        WHERE p.chat_epoch LIKE ? AND ro.ordinal=(SELECT MIN(ro2.ordinal)
          FROM row_observations ro2 WHERE ro2.physical_row_id=p.physical_row_id)
        ORDER BY o.frame_id,o.observation_id""",
        (ROW_TRACKER_VERSION + ":%",)).fetchall()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--max-observations", type=int)
    parser.add_argument("--roi-type", choices=("CHAT_ROW",), default="CHAT_ROW")
    parser.add_argument("--tiny-config", type=Path,
                        default=ROOT / "configs" / "ocr_tiny_local.json")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--watch-until", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    config = json.loads(args.tiny_config.read_text("utf-8"))
    batch_size = args.batch_size or int(config["batch_size"])
    if batch_size < 1 or (args.max_observations is not None and args.max_observations < 1):
        parser.error("batch size and observation count must be positive")
    if config.get("retry_policy") != "NONE_SINGLE_PASS":
        parser.error("Production tiny config must disable retries")

    journal = Journal(run / "journal.sqlite3")
    client = TinyChatClient(Path(config["python"]), Path(config["model_dir"]),
                            config["model_name"], config["weights_sha256"],
                            config["device"], run / "tiny_chat_worker.stderr.log")
    metrics = {"tiny_inference_wall_seconds": 0.0, "tiny_jobs": 0,
               "tiny_batches": 0, "tiny_cache_hits": 0,
               "db_write_wall_seconds": 0.0, "max_batch_size": 0,
               "configured_batch_size": batch_size, "retry_jobs": 0,
               "fallback_jobs": 0}
    revisions = []
    seen: set[str] = set()
    processed = skipped = 0
    started = time.perf_counter()
    cpu_started = time.process_time()
    last_progress = 0.0

    try:
        while True:
            rows = [row for row in rows_for_ocr(journal) if row[0] not in seen]
            if args.max_observations is not None:
                rows = rows[:max(0, args.max_observations - len(seen))]
            for offset in range(0, len(rows), batch_size):
                batch = rows[offset:offset + batch_size]
                pending = []
                for observation_id, relative, digest, physical_row_id, frame_id, session_id in batch:
                    seen.add(observation_id)
                    crop = (run / relative).resolve(strict=True)
                    if not crop.is_relative_to(run):
                        raise ValueError(f"Crop outside run: {crop}")
                    prep = {"policy": OCR_POLICY, "input": "physical_row_crop_original_png",
                            "source_crop_sha256": digest, "model_name": config["model_name"],
                            "weights_sha256": config["weights_sha256"],
                            "device": config["device"], "batch_size": batch_size,
                            "retry_policy": "NONE_SINGLE_PASS"}
                    cached = existing_reading(journal, observation_id,
                                              config["model_version"], prep)
                    if cached:
                        metrics["tiny_cache_hits"] += 1
                        skipped += 1
                    else:
                        pending.append({"item_id": observation_id, "path": str(crop),
                                        "source_sha256": digest, "preprocessing": prep,
                                        "physical_row_id": physical_row_id,
                                        "frame_id": frame_id,
                                        "table_session_id": session_id})
                if not pending:
                    continue
                batch_tick = time.perf_counter()
                response = client.read_batch([{key: item[key] for key in
                    ("item_id", "path", "source_sha256")} for item in pending], batch_size)
                metrics["tiny_inference_wall_seconds"] += response["inference_seconds"]
                metrics["tiny_batches"] += 1
                metrics["tiny_jobs"] += len(pending)
                metrics["max_batch_size"] = max(metrics["max_batch_size"], len(pending))
                by_id = {result["item_id"]: result for result in response["results"]}
                db_tick = time.perf_counter()
                journal.begin_batch()
                try:
                    for item in pending:
                        result = by_id.get(item["item_id"], {"status": "ERROR",
                            "raw_text": "", "score": None,
                            "error": "WORKER_RESULT_MISSING"})
                        raw_text = result.get("raw_text") or None
                        status = ("RAW_UNREVIEWED" if result["status"] == "OK" and raw_text
                                  else "UNREADABLE" if result["status"] == "EMPTY"
                                  else "ERROR")
                        prep = {**item["preprocessing"], "raw_score": result.get("score")}
                        rec_id, revision = journal.add_recognition(
                            item["item_id"], raw_text, ENGINE, config["model_version"],
                            prep, status, None if status == "RAW_UNREVIEWED" else
                            result.get("error", "EMPTY_SINGLE_PASS_NO_RETRY"))
                        revisions.append({"observation_id": item["item_id"],
                            "physical_row_id": item["physical_row_id"],
                            "recognition_id": rec_id, "revision": revision,
                            "status": status, "raw_text": raw_text,
                            "score": result.get("score")})
                        processed += 1
                    journal.end_batch()
                except Exception:
                    journal.end_batch(commit=False)
                    raise
                metrics["db_write_wall_seconds"] += time.perf_counter() - db_tick
                metrics.setdefault("batch_wall_seconds", 0.0)
                metrics["batch_wall_seconds"] += time.perf_counter() - batch_tick

            now = time.perf_counter()
            if now - last_progress >= 5:
                last_progress = now
                progress = {"visited": len(seen), "processed": processed,
                            "skipped": skipped, "pending_tiny": 0,
                            "performance": metrics, "wall_seconds": now - started}
                (run / "ocr_progress.json").write_text(
                    json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
            limit_reached = args.max_observations is not None and len(seen) >= args.max_observations
            if limit_reached or args.watch_until is None or args.watch_until.exists():
                if not [row for row in rows_for_ocr(journal) if row[0] not in seen]:
                    break
            time.sleep(0.25)
    finally:
        startup_seconds = client.ready.get("startup_seconds")
        client.close()
        totals = journal.counts()
        journal.close()

    wall = time.perf_counter() - started
    report = {"engine": ENGINE, "model_version": config["model_version"],
              "policy": OCR_POLICY, "retry_policy": "NONE_SINGLE_PASS",
              "processed": processed, "skipped": skipped,
              "wall_seconds": wall,
              "process_cpu_seconds": time.process_time() - cpu_started,
              "model_startup_seconds": startup_seconds,
              "images_per_inference_second": (metrics["tiny_jobs"] /
                  metrics["tiny_inference_wall_seconds"]
                  if metrics["tiny_inference_wall_seconds"] else None),
              "performance": metrics, "journal_counts": totals,
              "revisions": [] if args.summary_only else revisions,
              "interpretation": "RAW_SINGLE_PASS_OCR_NOT_GROUND_TRUTH"}
    (run / "ocr_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
