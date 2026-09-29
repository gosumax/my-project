"""Prepare complete Smoke14 Tiny OCR data and old/new speed metrics for Excel."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def flatten(prefix: str, value, rows: list[dict], run_label: str,
            module: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, child, rows,
                    run_label, module)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        rows.append({"run": run_label, "module": module,
                     "metric": prefix, "value": value})


def stage_metrics(report: dict, label: str) -> list[dict]:
    rows: list[dict] = []
    for stage in report.get("stages", []):
        module = stage["stage"]
        if isinstance(stage.get("wall_seconds"), (int, float)):
            rows.append({"run": label, "module": module,
                         "metric": "stage.wall_seconds",
                         "value": stage["wall_seconds"]})
        flatten("performance", stage.get("details", {}).get("performance", {}),
                rows, label, module)
    return rows


def resource_metrics(path: Path, label: str) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    rows = []
    for column in records[0] if records else []:
        if column == "elapsed_seconds":
            continue
        values = []
        for record in records:
            try:
                values.append(float(record[column]))
            except (TypeError, ValueError):
                pass
        if not values:
            continue
        ordered = sorted(values)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]
        for suffix, value in (("mean", statistics.fmean(values)),
                              ("p95", p95), ("max", max(values))):
            rows.append({"run": label, "module": "resources",
                         "metric": f"{column}.{suffix}", "value": value})
    return rows


def stage(report: dict, name: str) -> dict:
    return next((item for item in report.get("stages", [])
                 if item.get("stage") == name), {})


def build_comparison(new_report: dict, old_report: dict,
                     new_ocr: dict) -> list[dict]:
    new_capture = stage(new_report, "capture")
    old_capture = stage(old_report, "capture")
    new_track = stage(new_report, "track_rows")
    old_track = stage(old_report, "track_rows")
    old_ocr = stage(old_report, "ocr_rows")
    old_perf = old_ocr.get("details", {}).get("performance", {})
    new_perf = new_ocr.get("performance", {})
    old_frames = old_report.get("counts", {}).get("frames", 0)
    new_frames = new_report.get("counts", {}).get("frames", 0)
    old_total = old_report.get("wall_seconds", 0)
    new_total = new_report.get("wall_seconds", 0)
    old_ocr_jobs = old_perf.get("tesseract_primary_jobs", 0)
    old_vl_jobs = old_perf.get("vl_jobs", 0)
    new_jobs = new_perf.get("tiny_jobs", 0)
    values = [
        ("Total pipeline", "wall_seconds", old_total, new_total, "Critical-path elapsed time"),
        ("Total pipeline", "frames_per_second",
         old_frames / old_total if old_total else None,
         new_frames / new_total if new_total else None, "Different videos; throughput comparison"),
        ("Capture", "stage_wall_seconds", old_capture.get("wall_seconds"),
         new_capture.get("wall_seconds"), "Streaming stage duration"),
        ("Capture", "frames_per_second",
         old_frames / old_capture.get("wall_seconds", 1),
         new_frames / new_capture.get("wall_seconds", 1), "All decoded frames"),
        ("Row tracking", "stage_wall_seconds", old_track.get("wall_seconds"),
         new_track.get("wall_seconds"), "Includes streaming idle wait"),
        ("OCR", "stage_wall_seconds", old_ocr.get("wall_seconds"),
         stage(new_report, "ocr_rows").get("wall_seconds"), "Includes streaming wait"),
        ("OCR model", "jobs", old_ocr_jobs + old_vl_jobs, new_jobs,
         "Old count includes Tesseract primary plus VL"),
        ("Tesseract old", "seconds_per_primary_job",
         old_perf.get("tesseract_primary_wall_seconds_sum", 0) / old_ocr_jobs
         if old_ocr_jobs else None, None, "Five-worker sum divided by jobs"),
        ("VL old", "seconds_per_job",
         old_perf.get("vl_wall_seconds", 0) / old_vl_jobs if old_vl_jobs else None,
         None, "Sequential VL bottleneck"),
        ("Tiny new", "seconds_per_crop",
         None, new_perf.get("tiny_inference_wall_seconds", 0) / new_jobs
         if new_jobs else None, "Model inference only, batch=8"),
        ("Tiny new", "crops_per_inference_second", None,
         new_jobs / new_perf.get("tiny_inference_wall_seconds", 1)
         if new_jobs else None, "Model inference only, batch=8"),
        ("OCR policy", "retry_jobs", old_perf.get("tesseract_retry_jobs"),
         new_perf.get("retry_jobs"), "New architecture forbids retries"),
        ("OCR policy", "fallback_jobs", old_vl_jobs,
         new_perf.get("fallback_jobs"), "New architecture forbids fallback"),
    ]
    output = []
    for module, metric, old_value, new_value, note in values:
        speedup = (old_value / new_value if isinstance(old_value, (int, float)) and
                   isinstance(new_value, (int, float)) and new_value else None)
        output.append({"module": module, "metric": metric,
                       "old_smoke13": old_value, "new_smoke14": new_value,
                       "old_over_new": speedup, "note": note})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    new_run = args.new_run.resolve(strict=True)
    old_run = args.old_run.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    new_report = read_json(new_run / "pilot_run.json")
    old_report = read_json(old_run / "pilot_run.json")
    new_ocr = read_json(new_run / "ocr_summary.json")

    metrics = stage_metrics(old_report, "Smoke13 old")
    metrics += stage_metrics(new_report, "Smoke14 Tiny")
    metrics += resource_metrics(old_run / "resource_samples.csv", "Smoke13 old")
    metrics += resource_metrics(new_run / "resource_samples.csv", "Smoke14 Tiny")
    comparison = build_comparison(new_report, old_report, new_ocr)

    database = new_run / "journal.sqlite3"
    with sqlite3.connect(database) as db:
        rows = db.execute("""SELECT p.physical_row_id,p.table_session_id,o.frame_id,
            o.observation_id,o.crop_path,o.crop_sha256,r.recognition_id,r.revision,
            r.raw_text,r.status,r.reason,r.preprocessing_json
            FROM physical_rows p
            JOIN row_observations ro ON ro.physical_row_id=p.physical_row_id
            JOIN observations o ON o.observation_id=ro.observation_id
            JOIN recognitions r ON r.observation_id=o.observation_id
            WHERE ro.ordinal=(SELECT MIN(ro2.ordinal) FROM row_observations ro2
                WHERE ro2.physical_row_id=p.physical_row_id)
              AND r.engine='PP-OCRv6_tiny_rec'
              AND r.revision=(SELECT MAX(r2.revision) FROM recognitions r2
                WHERE r2.observation_id=r.observation_id AND r2.engine=r.engine)
            ORDER BY o.frame_id,p.table_session_id,p.physical_row_id""").fetchall()
    rows_path = output / "recognitions.jsonl"
    with rows_path.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            prep = json.loads(row[11])
            record = {"index": index, "physical_row_id": row[0],
                      "table_session_id": row[1], "frame_id": row[2],
                      "observation_id": row[3],
                      "crop_path": str((new_run / row[4]).resolve()),
                      "crop_sha256": row[5], "recognition_id": row[6],
                      "revision": row[7], "raw_text": row[8] or "",
                      "status": row[9], "reason": row[10] or "",
                      "score": prep.get("raw_score"),
                      "batch_size": prep.get("batch_size"),
                      "retry_policy": prep.get("retry_policy")}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    payload = {"schema": "smoke14.tiny.excel-data.v1",
               "new_run": str(new_run), "old_run": str(old_run),
               "new_report": new_report, "old_report": old_report,
               "new_ocr": new_ocr, "comparison": comparison,
               "module_metrics": metrics, "recognition_count": len(rows)}
    (output / "workbook_data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "recognitions": len(rows),
                      "metrics": len(metrics)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
