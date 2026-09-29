"""Collect measured stage timings and raw chat evidence for the Smoke13 workbook."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime
from collections import defaultdict
from pathlib import Path


def load(path):
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run.resolve(strict=True)
    report = load(run / "pilot_run.json")
    metrics = []
    modules = []
    for stage in report.get("stages", []):
        name = stage["stage"]
        details = stage.get("details") or {}
        modules.append([name, stage["wall_seconds"], stage.get("exit_code", 0),
                        "Concurrent" if name in ("capture", "track_rows", "ocr_rows") else "Sequential"])
        if isinstance(details, dict):
            for metric, value in details.get("performance", {}).items():
                if name == "track_rows" and metric == "db_commit_wall_seconds":
                    metric = "stream_flush_commit_wall_seconds_partial"
                metrics.append([name, metric, value,
                                "seconds" if "seconds" in metric else "count"])
            for metric, value in details.get("decisions", {}).items():
                metrics.append([name, metric, value, "count"])
            for metric, value in details.get("crop_storage", {}).items():
                if isinstance(value, (int, float)):
                    metrics.append([name, metric, value, "count"])
    samples = []
    chart_buckets = defaultdict(list)
    with (run / "resource_samples.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            sample = [float(row[key]) if row[key] else None for key in row]
            samples.append(sample)
            chart_buckets[int(sample[0] // 30)].append(sample)
    chart_samples = [[bucket * 30,
                      sum(row[1] for row in entries) / len(entries),
                      sum(row[6] for row in entries if row[6] is not None) /
                      max(1, sum(row[6] is not None for row in entries))]
                     for bucket, entries in sorted(chart_buckets.items())]
    # The stopped old full run contains capture evidence only; its later stages
    # were never executed. Keep those missing measurements explicit.
    old = root / "runs/smoke13_full_reuse_vl15_20260929_v1"
    old_counts = {}
    if (old / "journal.sqlite3").exists():
        with sqlite3.connect(old / "journal.sqlite3") as db:
            old_counts = {table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                          for table in ("frames", "observations", "physical_rows", "recognitions")}
    b1 = load(root / "runs/perf_capture_batch1_20260929/summary.json")
    b25 = load(root / "runs/perf_capture_batch25_20260929/summary.json")
    comparisons = [
        ["SQLite commit, same 5 frames", b1.get("performance", {}).get("db_commit_wall_seconds"),
         b25.get("performance", {}).get("db_commit_wall_seconds"),
         "1-frame vs 25-frame transactions. Commit component only.",
         str(root / "runs/perf_capture_batch1_20260929/summary.json"),
         str(root / "runs/perf_capture_batch25_20260929/summary.json")],
        ["Tesseract, same 18 chat crops", 13.4528, 8.2632,
         "1 vs 2 workers. Texts identical. Recorded command wall times from this session.",
         "Paired benchmark in current session", "Paired benchmark in current session"],
    ]
    paired_path = Path("C:/ParserData/runs/smoke13_streaming_ssd_20f_compare_20260929_v1")
    paired = load(paired_path / "pilot_run.json")
    legacy_path = root / "runs/dev_chunked_smoke13_20f_20260929"
    legacy = load(legacy_path / "pilot_run.json")
    if paired.get("status") == "PARTIAL_PIPELINE" and legacy:
        legacy_wall = (datetime.fromisoformat(legacy["finished_utc"]) -
                       datetime.fromisoformat(legacy["started_utc"])).total_seconds()
        comparisons.append(["Полный путь, те же 20 кадров и VL", legacy_wall, paired["wall_seconds"],
                            "Все этапы. Изменены архитектура и диск; запуск модели включён.",
                            str(legacy_path / "pilot_run.json"), str(paired_path / "pilot_run.json")])
        for stage in paired["stages"]:
            name = stage["stage"]
            old_stages = [item for item in legacy["stages"]
                          if item["stage"] == name or item["stage"].endswith("_" + name)]
            if not old_stages:
                continue
            old_wall = sum(item["wall_seconds"] for item in old_stages)
            new_wall = stage["wall_seconds"]
            note = "Те же 20 кадров; старый путь повторно обрабатывал предыдущие блоки."
            if name == "track_rows":
                new_wall -= stage["details"]["performance"].get("idle_wall_seconds", 0)
                note += " У нового трекера вычтено ожидание захвата."
            comparisons.append([name, old_wall, new_wall, note,
                                str(legacy_path / "pilot_run.json"), str(paired_path / "pilot_run.json")])
    snapshots = []
    for directory in ("smoke13_full_chunked_newarch_20260929_v2",
                      "smoke13_full_streaming_newarch_20260929_v1"):
        path = root / "runs" / directory
        snapshot = load(path / "stopped_snapshot.json") or load(path / "pilot_run.json")
        snapshots.append({"run": directory, "data": snapshot})
    tables = {str(index): [] for index in range(1, 10)}
    with sqlite3.connect(run / "journal.sqlite3") as db:
        for engine, status, count in db.execute("SELECT engine,status,COUNT(*) FROM recognitions GROUP BY engine,status"):
            metrics.append([engine, status, count, "readings"])
        slots = {sid: str(json.loads(geometry)["slot_id"])
                 for sid, geometry in db.execute("SELECT table_session_id,geometry_json FROM table_sessions")}
        raw_by_message = defaultdict(lambda: {"Tesseract": [], "PaddleOCR-VL-1.5": []})
        for mid, ordinal, engine, text in db.execute("""SELECT mf.message_id,mf.ordinal,r.engine,r.raw_text
            FROM message_fragments mf JOIN row_observations ro ON ro.physical_row_id=mf.physical_row_id
            JOIN recognitions r ON r.observation_id=ro.observation_id
            WHERE ro.ordinal=(SELECT MIN(x.ordinal) FROM row_observations x WHERE x.physical_row_id=ro.physical_row_id)
            AND r.engine IN ('Tesseract','PaddleOCR-VL-1.5') AND r.revision=(
                SELECT MAX(x.revision) FROM recognitions x WHERE x.observation_id=r.observation_id AND x.engine=r.engine)
            ORDER BY mf.message_id,mf.ordinal"""):
            raw_by_message[mid][engine].append(text or "")
        for session, eid, mid, event, actor, amount, cards, status, reason, pts, num, den in db.execute("""
            SELECT e.table_session_id,e.event_id,e.message_id,e.event_type,e.actor,e.amount_decimal,
                   e.cards_json,e.status,e.reason,e.observed_source_pts,
                   (SELECT time_base_num FROM frames WHERE source_pts=e.observed_source_pts LIMIT 1),
                   (SELECT time_base_den FROM frames WHERE source_pts=e.observed_source_pts LIMIT 1)
            FROM event_revisions e WHERE e.revision=(SELECT MAX(x.revision) FROM event_revisions x WHERE x.event_id=e.event_id)
            ORDER BY e.table_session_id,e.order_key"""):
            stamp = pts * num / den / 86400 if pts is not None and num and den else None
            raw = raw_by_message[mid]
            detail = "; ".join(str(value) for value in (status, reason, cards if cards != "[]" else None, eid) if value)
            tables[slots[session]].append([stamp, "CHAT", event, actor or "UNKNOWN", amount,
                                          detail, "\n".join(raw["PaddleOCR-VL-1.5"]),
                                          "\n".join(raw["Tesseract"]), "UNKNOWN"])
    payload = {"report": report, "modules": modules, "metrics": metrics,
               "validation": load(args.output.parent / "validation.json"),
               "resource_samples": samples, "chart_samples": chart_samples,
               "comparisons": comparisons,
               "old_full_counts": old_counts, "old_full_path": str(old),
               "stopped_runs": snapshots, "table_events": tables,
               "source_report": str(run / "pilot_run.json"),
               "source_resources": str(run / "resource_samples.csv")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "module_count": len(modules),
                      "metrics_count": len(metrics), "event_count": sum(map(len, tables.values()))}))


if __name__ == "__main__":
    main()
