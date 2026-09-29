"""RAM first stage -> Tiny OCR -> ordered raw timeline -> Excel. No HH stage."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.message_assembly import RowFact, group_rows
from parser_core.normalize_chat import normalize
from parser_core.ram_first_stage import FirstStage
from parser_core.tiny_chat_client import TinyChatClient


class Telemetry:
    def __init__(self):
        self.values = defaultdict(list)
        self.objects = Counter()

    def add(self, name, seconds, objects=1):
        self.values[name].append(seconds)
        self.objects[name] += objects

    def report(self, wall):
        output = []
        for name, values in sorted(self.values.items()):
            ordered = sorted(values)
            total = sum(values)
            count = len(values)
            output.append({"operation": name, "calls": count,
                           "objects": self.objects[name], "total_wall_seconds": total,
                           "average_seconds": total / count,
                           "median_seconds": statistics.median(ordered),
                           "p95_seconds": ordered[min(count - 1, math.ceil(count * .95) - 1)],
                           "max_seconds": ordered[-1],
                           "percent_of_run_wall": 100 * total / wall if wall else None,
                           "objects_per_second": self.objects[name] / total if total else None})
        return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True,
                   help="Read-only historical run for stable slot identifiers")
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tiny-config", type=Path, default=ROOT / "configs/ocr_tiny_local.json")
    p.add_argument("--node", type=Path, required=True)
    args = p.parse_args()
    if args.frames < 1 or args.output.exists():
        p.error("Frames must be positive and output directory must be fresh")
    config = json.loads(args.tiny_config.read_text("utf-8"))
    if config.get("retry_policy") != "NONE_SINGLE_PASS":
        p.error("OCR retry policy must be NONE_SINGLE_PASS")
    sessions = json.loads((args.baseline / "summary.json").read_text("utf-8"))["table_sessions"]
    output = args.output.resolve()
    output.mkdir(parents=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    process = psutil.Process()
    io_start = process.io_counters()
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    ram_peak = process.memory_info().rss
    rss_samples = []
    telemetry = Telemetry()
    status = "FAILED"
    errors = []
    rows = []
    timeline = []
    stage = FirstStage(args.video, sessions, args.frames, root=ROOT, telemetry=telemetry)
    client = TinyChatClient(Path(config["python"]), Path(config["model_dir"]),
                            config["model_name"], config["weights_sha256"],
                            config["device"], output / "tiny_worker.stderr.log")
    metadata = {"entrypoint": str(Path(__file__).resolve()), "command": sys.argv,
                "started_utc": started_utc, "video": str(args.video.resolve()),
                "frames_requested": args.frames, "first_stage": str((ROOT / "parser_core/ram_first_stage.py").resolve()),
                "ocr_transport": "base64 PNG bytes in RAM; worker decodes to numpy; no image files",
                "legacy_imports": [], "row_identity": "short_history_contiguous_pixel_blocks_v1",
                "history_states": 12, "status": "RUNNING"}
    (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), "utf-8")
    try:
        tick = time.perf_counter()
        for row in stage.rows():
            rows.append(row)
            if len(rows) % 100 == 0:
                rss = process.memory_info().rss
                ram_peak = max(ram_peak, rss)
                rss_samples.append(rss)
        first_stage_wall = time.perf_counter() - tick
        telemetry.add("first_stage_total", first_stage_wall, stage.seen_frames)
        ram_peak = max(ram_peak, process.memory_info().rss)
        # Row order is fixed before OCR. The worker's completion order cannot
        # alter chronology because outputs are mapped back to stable row keys.
        tick = time.perf_counter()
        rows.sort(key=lambda r: (r.table_id, r.frame_id, r.source_span_index))
        telemetry.add("ordering_rows", time.perf_counter() - tick, len(rows))
        batch_size = int(config["batch_size"])
        recognized = {}
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            prepared = []
            for row in batch:
                tick = time.perf_counter()
                ok, packed = cv2.imencode(".png", row.pixels)
                if not ok:
                    raise ValueError("RAM row PNG encode failed")
                prepared.append({"item_id": f"{row.table_id}:{row.frame_id}:{row.source_span_index}",
                                 "png_base64": base64.b64encode(packed).decode("ascii"),
                                 "source_sha256": row.pixel_sha256})
                telemetry.add("ocr_ram_input_encode", time.perf_counter() - tick)
            tick = time.perf_counter()
            try:
                response = client.read_batch(prepared, batch_size)
                telemetry.add("ocr_worker_roundtrip", time.perf_counter() - tick, len(batch))
                telemetry.add("ocr_inference_worker_reported", response["inference_seconds"], len(batch))
                by_id = {item["item_id"]: item for item in response["results"]}
            except Exception as exc:
                errors.append({"batch_offset": offset, "error": repr(exc)})
                by_id = {}
            tick = time.perf_counter()
            for item in prepared:
                recognized[item["item_id"]] = by_id.get(item["item_id"], {
                    "status": "ERROR", "raw_text": "", "score": None,
                    "error": "WORKER_BATCH_FAILED_OR_RESULT_MISSING"})
            telemetry.add("ocr_response_map", time.perf_counter() - tick, len(batch))
            ram_peak = max(ram_peak, process.memory_info().rss)
        client.close()
        for row in rows:
            key = f"{row.table_id}:{row.frame_id}:{row.source_span_index}"
            result = recognized[key]
            raw = result.get("raw_text") or ""
            tick = time.perf_counter()
            candidate = normalize([raw])
            telemetry.add("text_normalize_classify", time.perf_counter() - tick)
            status_text = "UNRESOLVED" if row.origin == "UNRESOLVED_IDENTITY" else (
                "ERROR" if result.get("status") == "ERROR" else
                "UNKNOWN" if candidate.event_type == "UNKNOWN_EVENT" else
                "UNREADABLE" if not raw else "RAW_UNREVIEWED")
            timeline.append({"table_id": f"T{row.table_id:02d}", "sequence_id": None,
                             "frame_id": row.frame_id, "video_time": row.video_time,
                             "source_roi_change_id": f"{row.session_id}:{row.frame_id}:chat",
                             "row_index": row.source_span_index, "raw_ocr_text": raw,
                             "normalized_text": raw.strip(), "event_type": candidate.event_type,
                             "confidence": result.get("score"), "status": status_text,
                             "notes_error": result.get("error") or candidate.reason or "",
                             "session_id": row.session_id, "origin": row.origin,
                             "epoch": row.epoch, "row_pixel_sha256": row.pixel_sha256,
                             "stable_row_id": row.stable_row_id})
        tick = time.perf_counter()
        timeline.sort(key=lambda x: (x["table_id"], x["frame_id"], x["row_index"]))
        for index, item in enumerate(timeline, 1):
            item["sequence_id"] = index
        telemetry.add("timeline_sort_and_sequence", time.perf_counter() - tick, len(timeline))
        # Message grouping is diagnostic; every physical row remains in the
        # primary timeline even when grouping or classification is uncertain.
        tick = time.perf_counter()
        message_counts = {}
        for table_id in sorted({x["table_id"] for x in timeline}):
            table_rows = [x for x in timeline if x["table_id"] == table_id]
            facts = [RowFact(f"{x['table_id']}:{x['frame_id']}:{x['row_index']}",
                             x["session_id"], str(x["epoch"]), x["frame_id"],
                             x["row_index"], x["raw_ocr_text"], None, None,
                             x["origin"]) for x in table_rows]
            messages = group_rows(facts)
            message_counts[table_id] = len(messages)
            by_key = {(x["frame_id"], x["row_index"]): x for x in table_rows}
            for message_index, message in enumerate(messages, 1):
                tick_message = time.perf_counter()
                candidate = normalize([r.text for r in message.rows])
                telemetry.add("message_normalize_classify", time.perf_counter() - tick_message)
                for fact in message.rows:
                    item = by_key[(fact.frame_id, int(fact.y1))]
                    item["message_id"] = f"{table_id}:M{message_index:06d}"
                    item["message_event_type"] = candidate.event_type
                    item["message_status"] = message.completeness
                    item["message_reason"] = message.reason or ""
        telemetry.add("message_grouping", time.perf_counter() - tick, len(timeline))
        tick = time.perf_counter()
        with (output / "ordered_timeline.jsonl").open("w", encoding="utf-8") as stream:
            for item in timeline:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        telemetry.add("jsonl_write", time.perf_counter() - tick, len(timeline))
        tick = time.perf_counter()
        subprocess.run([str(args.node), str(ROOT / "scripts/build_smoke14_ram_workbook.mjs"),
                        str(output / "ordered_timeline.jsonl"), str(output / "ordered_timeline.xlsx")],
                       cwd=ROOT, check=True)
        telemetry.add("excel_builder_process", time.perf_counter() - tick, len(timeline))
        status = "COMPLETE"
    except BaseException as exc:
        errors.append({"fatal": repr(exc)})
        raise
    finally:
        client.close()
        wall = time.perf_counter() - wall_start
        io_end = process.io_counters()
        files = [p for p in output.rglob("*") if p.is_file()]
        png_files = [p for p in files if p.suffix.lower() == ".png"]
        summary = {"status": status, "started_utc": started_utc,
                   "finished_utc": datetime.now(timezone.utc).isoformat(),
                   "frames": stage.seen_frames, "video_duration_seconds": stage.last_video_time,
                   "wall_seconds": wall, "fps": stage.seen_frames / wall if wall else None,
                   "rtf": wall / stage.last_video_time if stage.last_video_time else None,
                   "first_stage_wall_seconds": locals().get("first_stage_wall"),
                   "first_stage_fps": stage.seen_frames / first_stage_wall if locals().get("first_stage_wall") else None,
                   "process_cpu_seconds": time.process_time() - cpu_start,
                   "average_rss_mb": statistics.fmean(rss_samples) / 1048576 if rss_samples else None,
                   "peak_rss_mb": ram_peak / 1048576,
                   "disk_read_bytes": io_end.read_bytes - io_start.read_bytes,
                   "disk_write_bytes": io_end.write_bytes - io_start.write_bytes,
                   "files_created": len(files), "png_files": len(png_files),
                   "png_bytes": sum(p.stat().st_size for p in png_files),
                   "table_count": len(sessions), "table_changes": stage.metrics["changed_table_rois"],
                   "chat_changes": stage.metrics["changed_chat_rois"],
                   "physical_rows": len(rows), "ocr_calls": len(rows),
                   "timeline_rows": len(timeline), "message_counts": locals().get("message_counts", {}),
                   "unknown_rows": sum(x["event_type"] == "UNKNOWN_EVENT" for x in timeline),
                   "unresolved_rows": sum(x["status"] == "UNRESOLVED" for x in timeline),
                   "error_rows": sum(x["status"] == "ERROR" for x in timeline),
                   "retry_count": 0, "queue_wait_seconds": 0.0,
                   "queue_wait_note": "single synchronous OCR worker; no application queue",
                   "errors": errors, "first_stage_counts": dict(stage.metrics)}
        (output / "module_timings.json").write_text(json.dumps(telemetry.report(wall), indent=2), "utf-8")
        (output / "final_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
        (output / "errors_retries.json").write_text(json.dumps({"errors": errors, "retries": 0}, ensure_ascii=False, indent=2), "utf-8")
        metadata["status"] = status
        (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    main()
