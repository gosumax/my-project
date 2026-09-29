"""Prepare chronological raw Tiny OCR rows and measured metrics for Excel."""

import argparse
import csv
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    report = json.loads((run / "pilot_run.json").read_text("utf-8"))
    sessions = json.loads((run / "summary.json").read_text("utf-8"))["table_sessions"]
    table_for_session = {session: int(table) for table, session in sessions.items()}
    query = """SELECT p.physical_row_id,p.table_session_id,o.frame_id,
        f.source_pts,f.time_base_num,f.time_base_den,o.bbox_json,o.crop_path,
        o.crop_sha256,r.raw_text,r.raw_score,r.status,r.reason,r.engine,r.model_version,
        r.preprocessing_json,
        o.quality_json
        FROM physical_rows p JOIN row_observations ro ON ro.physical_row_id=p.physical_row_id
        JOIN observations o ON o.observation_id=ro.observation_id
        JOIN frames f ON f.source_id=o.source_id AND f.frame_id=o.frame_id
        JOIN recognitions r ON r.observation_id=o.observation_id
        WHERE ro.ordinal=(SELECT MIN(ro2.ordinal) FROM row_observations ro2
            WHERE ro2.physical_row_id=p.physical_row_id)
          AND r.engine='PP-OCRv6_tiny_rec'
          AND r.revision=(SELECT MAX(r2.revision) FROM recognitions r2
            WHERE r2.observation_id=r.observation_id AND r2.engine=r.engine)
        ORDER BY o.frame_id,p.table_session_id,json_extract(o.bbox_json,'$[1]'),p.physical_row_id"""
    with sqlite3.connect(run / "journal.sqlite3") as db:
        source = db.execute(query).fetchall()
    by_table = defaultdict(list)
    for values in source:
        row_id, session, frame_id, pts, num, den, bbox_json, relative, digest, text, score, status, reason, engine, version, prep_json, quality_json = values
        table = table_for_session[session]
        bbox = json.loads(bbox_json)
        quality = json.loads(quality_json)
        prep = json.loads(prep_json)
        by_table[table].append({
            "video_seconds": pts * num / den if pts is not None and den else None,
            "frame_id": frame_id, "row_y": bbox[1],
            "source_span_index": quality.get("source_span_index"),
            "raw_text": text or "", "score": float(score) if score is not None else prep.get("raw_score"),
            "status": status, "reason": reason or "", "crop_path": str((run / relative).resolve()),
            "crop_sha256": digest, "physical_row_id": row_id,
            "engine": engine, "model_version": version})
    for table, rows in by_table.items():
        for sequence, row in enumerate(rows, 1):
            row["sequence"] = sequence
    resources = list(csv.DictReader((run / "resource_samples.csv").open(encoding="utf-8")))
    resource_summary = {}
    for key in ("system_cpu_percent", "pipeline_cpu_percent", "pipeline_rss_mb",
                "gpu_percent", "gpu_memory_mb"):
        values = [float(item[key]) for item in resources if item[key]]
        resource_summary[key] = {"mean": statistics.fmean(values), "peak": max(values)} if values else None
    benchmark = json.loads(args.benchmark.read_text("utf-8"))
    payload = {"source_run": str(run), "report": report,
               "resources": resource_summary, "benchmark": benchmark,
               "tables": {str(i): by_table[i] for i in range(1, 10)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
    print(json.dumps({"rows": len(source), "tables": {i: len(by_table[i]) for i in range(1, 10)},
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
