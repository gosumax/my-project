"""INTERNAL ONLY: registered RAM-first implementation; launch with run_production.py."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import hashlib
import psutil
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.message_assembly import RowFact, group_rows
from parser_core.normalize_chat import normalize
from parser_core.parallel_ocr import ParallelRowOCR, row_key
from parser_core.ram_first_stage import FirstStage
from parser_core.temporal_overlap import TemporalObservations
from parser_core.poker_pipeline import analyze_timeline
from parser_core.tiny_chat_client import TinyChatClient
from parser_core.visual_cards import associate_visual_evidence, project_chat_card_events
from parser_core.layout import load_fixed_layout
from parser_core.journal import stable_id
from parser_core.visual_tile_queue import VisualTileQueue


class Telemetry:
    def __init__(self, production_guard=None):
        self.values = defaultdict(list)
        self.objects = Counter()
        self.production_guard = production_guard
        self.capture_started = None

    def add(self, name, seconds, objects=1):
        self.values[name].append(seconds)
        self.objects[name] += objects
        if (name == "frame_to_ndarray" and self.production_guard is not None
                and self.capture_started is not None):
            self.production_guard.check_early(self.objects[name], time.perf_counter()-self.capture_started)

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


def source_identity(video: Path, root: Path) -> tuple[dict, dict[str, str]]:
    layout_path = root / "configs/nine_table_layout.json"
    profile_path = root / "configs/layout_847x404.json"
    calibration_path = root / "configs/chat_log_bottom_rois.json"
    layout = load_fixed_layout(layout_path, profile_path, calibration_path)
    with video.open("rb") as stream:
        video_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in
              (layout_path, profile_path, calibration_path)]
    source_id = stable_id("source", video_sha, video.stat().st_size)
    sessions = {str(slot.slot_id): stable_id("table", source_id, layout.layout_id,
                *hashes, slot.slot_id, slot.table_box, slot.chat_box) for slot in layout.slots}
    return {"source_id": source_id, "source_video_sha256": video_sha,
            "layout_id": layout.layout_id, "layout_sha256": hashes[0],
            "profile_sha256": hashes[1], "calibration_sha256": hashes[2]}, sessions


def main(*, production_guard=None):
    if production_guard is None:
        raise RuntimeError("INTERNAL ONLY / LEGACY DIRECT LAUNCH: use python run_production.py")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--baseline", type=Path,
                   help="Optional read-only identity comparison; images are never read from it")
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tiny-config", type=Path, default=ROOT / "configs/ocr_tiny_local.json")
    p.add_argument("--ocr-workers", type=int, default=2)
    p.add_argument("--ocr-queue-size", type=int, default=32)
    p.add_argument("--vl-config", type=Path, default=ROOT / "configs/ocr_vl15_local.json")
    p.add_argument("--vl-workers", type=int, default=1)
    p.add_argument("--vl-batch-size", type=int, default=8)
    p.add_argument("--visual-workers", type=int, default=8)
    p.add_argument("--card-workers", type=int, default=2)
    p.add_argument("--vl-queue-size", type=int, default=256)
    p.add_argument("--skip-cards", "--skip-vl", dest="skip_vl", action="store_true",
                   help="Skip pixel card recognition; --skip-vl is a legacy alias")
    p.add_argument("--card-hybrid-config", type=Path,
                   default=ROOT / "configs/card_hybrid_local.json")
    p.add_argument("--suit-backup-config", type=Path,
                   default=ROOT / "configs/suit_backup_local.json")
    p.add_argument("--rank-ncc-config", type=Path,
                   default=ROOT / "configs/rank_ncc_production.json")
    p.add_argument("--layout-profile", type=Path,
                   default=ROOT / "configs/layout_847x404.json")
    p.add_argument("--node", type=Path, required=True)
    args = p.parse_args()
    production_guard.configure(args)
    if min(args.frames, args.ocr_workers, args.ocr_queue_size,
           args.vl_workers, args.vl_queue_size, args.vl_batch_size,
           args.visual_workers, args.card_workers) < 1 or args.output.exists():
        p.error("Frames and worker/queue counts must be positive; output must be fresh")
    if args.vl_workers != 1 or args.vl_batch_size > 8:
        p.error("This GPU profile requires one persistent VL worker and batch <= 8")
    config = json.loads(args.tiny_config.read_text("utf-8"))
    if config.get("retry_policy") != "NONE_SINGLE_PASS":
        p.error("OCR retry policy must be NONE_SINGLE_PASS")
    launch_started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    process = psutil.Process()
    io_start = process.io_counters()
    cpu_start = time.process_time()
    source_metadata, sessions = source_identity(args.video, ROOT)
    if args.baseline is not None:
        old_sessions = json.loads((args.baseline / "summary.json").read_text("utf-8"))["table_sessions"]
        if old_sessions != sessions:
            p.error("Baseline slot identity does not match current source/layout")
        with sqlite3.connect(args.baseline / "journal.sqlite3") as db:
            source_hashes = {row[0] for row in db.execute("SELECT DISTINCT sha256 FROM sources")}
        if source_hashes != {source_metadata["source_video_sha256"]}:
            p.error("Baseline source does not match the requested video")
    output = args.output.resolve()
    output.mkdir(parents=True)
    wall_start = launch_started
    ram_peak = process.memory_info().rss
    pipeline_rss_peak = ram_peak
    resource_peak = {"rss_bytes": ram_peak, "cpu_percent": 0.0}
    resource_samples = []
    runtime_state = {"phase": "startup"}
    gpu_command = shutil.which("nvidia-smi")
    resource_stop = threading.Event()

    def sample_resources():
        sample_index = 0
        last_gpu = (None, None, None)
        member_cache, lifetime_io = {}, {}
        while not resource_stop.is_set():
            members = [process] + process.children(recursive=True)
            rss = cpu = 0.0
            read_bytes = write_bytes = 0
            for member in members:
                try:
                    key = (member.pid, member.create_time())
                    member = member_cache.setdefault(key, member)
                    rss += member.memory_info().rss
                    cpu += member.cpu_percent(interval=None)
                    io = member.io_counters()
                    lifetime_io[key] = (max(0, io.read_bytes-(io_start.read_bytes if member.pid == process.pid else 0)),
                                        max(0, io.write_bytes-(io_start.write_bytes if member.pid == process.pid else 0)))
                except psutil.Error:
                    pass
            read_bytes = sum(x[0] for x in lifetime_io.values())
            write_bytes = sum(x[1] for x in lifetime_io.values())
            if gpu_command and sample_index % 4 == 0:
                try:
                    sampled = subprocess.run([gpu_command,
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True,
                        timeout=2, check=True)
                    values = sampled.stdout.splitlines()[0].split(",")
                    last_gpu = tuple(float(value.strip()) for value in values[:3])
                except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                    last_gpu = (None, None, None)
            resource_peak["rss_bytes"] = max(resource_peak["rss_bytes"], rss)
            resource_peak["cpu_percent"] = max(resource_peak["cpu_percent"], cpu)
            resource_samples.append({"elapsed_seconds": time.perf_counter()-wall_start,
                "pipeline_rss_mb": rss/1048576, "cpu_percent_sum": cpu,
                "read_bytes_sum": read_bytes, "write_bytes_sum": write_bytes,
                "gpu_util_percent": last_gpu[0], "gpu_memory_used_mb": last_gpu[1],
                "gpu_memory_total_mb": last_gpu[2]})
            if sample_index % 60 == 0:
                capture = runtime_state.get("stage")
                tiny = runtime_state.get("ocr")
                visual = runtime_state.get("vl")
                progress = {"phase": runtime_state["phase"], "wall_seconds": time.perf_counter()-wall_start,
                    "frames": capture.seen_frames if capture is not None else 0,
                    "physical_rows": len(runtime_state.get("rows", [])),
                    "tiny_results": len(tiny.results) if tiny is not None else 0,
                    "vl_results": len(visual.results) if visual is not None else 0,
                    "tiny_queue": tiny.jobs.qsize() if tiny is not None else 0,
                    "vl_queue": visual.jobs.qsize() if visual is not None else 0,
                    "rss_mb": rss/1048576, "gpu_util_percent": last_gpu[0], "gpu_vram_mb": last_gpu[1]}
                (output / "progress.json").write_text(json.dumps(progress, indent=2), "utf-8")
            sample_index += 1
            resource_stop.wait(.5)

    resource_thread = threading.Thread(target=sample_resources, daemon=True)
    resource_thread.start()
    rss_samples = []
    telemetry = Telemetry(production_guard)
    status = "FAILED"
    errors = []
    rows = []
    timeline = []
    rank_policy = json.loads(args.rank_ncc_config.read_text("utf-8"))
    if rank_policy.get("mode") != "PIXEL_ONLY":
        p.error("Production CHAT card recognition requires PIXEL_ONLY")
    vl_enabled = False
    cards_enabled = not args.skip_vl
    temporal = TemporalObservations()
    stage = FirstStage(args.video, sessions, args.frames, root=ROOT, telemetry=telemetry,
                       retain_board=False, identity_observer=temporal.observe)
    runtime_state.update(stage=stage, rows=rows)
    def client_factory(worker_id):
        return TinyChatClient(Path(config["python"]), Path(config["model_dir"]),
                              config["model_name"], config["weights_sha256"],
                              config["device"], output / f"tiny_worker_{worker_id}.stderr.log")
    metadata = {"entrypoint": str(Path(__file__).resolve()), "command": sys.argv,
                "official_entrypoint": production_guard.manifest["entrypoint"],
                "production_version": production_guard.manifest["production_version"],
                "production_manifest_sha256": hashlib.sha256((ROOT/"production_manifest.json").read_bytes()).hexdigest(),
                "production_modified_files": production_guard.modified,
                **source_metadata, "table_sessions": sessions,
                "board_profile": str(args.layout_profile.resolve()),
                "board_profile_sha256": hashlib.sha256(args.layout_profile.read_bytes()).hexdigest(),
                "started_utc": started_utc, "video": str(args.video.resolve()),
                "frames_requested": args.frames, "first_stage": str((ROOT / "parser_core/ram_first_stage.py").resolve()),
                "ocr_transport": "base64 PNG bytes in RAM; worker decodes to numpy; no image files",
                "legacy_imports": [], "row_identity": "short_history_contiguous_pixel_blocks_v1",
                "history_states": 12, "ocr_workers": args.ocr_workers,
                "ocr_queue_size": args.ocr_queue_size,
                "vl_enabled": vl_enabled, "visual_transport": "RAM PNG bytes over IPC; evidence in JSONL",
                "cards_enabled": cards_enabled, "card_recognition_mode": "PIXEL_ONLY",
                "rank_ncc_config": str(args.rank_ncc_config.resolve()),
                "rank_ncc_config_sha256": hashlib.sha256(args.rank_ncc_config.read_bytes()).hexdigest(),
                "vl_input_scope": "DISABLED", "board_roi_vl_enabled": False,
                "vl_workers": args.vl_workers, "vl_queue_size": args.vl_queue_size,
                "vl_batch_size": args.vl_batch_size, "visual_workers": args.visual_workers,
                "card_workers": args.card_workers,
                "card_hybrid_config": str(args.card_hybrid_config.resolve()),
                "suit_backup_config": str(args.suit_backup_config.resolve()),
                "production_code_sha256": {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                    for relative in ("scripts/run_smoke14_ram.py", "parser_core/ram_first_stage.py",
                        "parser_core/parallel_ocr.py", "parser_core/visual_tile_queue.py",
                        "parser_core/temporal_overlap.py",
                        "parser_core/poker_pipeline.py", "parser_core/poker_hand_assembly.py",
                        "parser_core/poker_logic_engine.py", "parser_core/poker_math_engine.py",
                        "parser_core/poker_card_engine.py", "parser_core/poker_diagnostics.py",
                        "parser_core/rank_ncc.py", "parser_core/component_suit.py",
                        "parser_core/card_hybrid.py",
                        "parser_core/board_tiles.py", "parser_core/suit_backup.py",
                        "parser_core/visual_cards.py", "scripts/build_smoke14_ram_workbook.mjs")},
                "ncc_config_sha256": hashlib.sha256(args.suit_backup_config.read_bytes()).hexdigest(),
                "timeline_dedup": "NATIVE_GLYPH_NCC_IMMUTABLE_HAND_CONTEXT",
                "poker_analysis": "STRICT_SOURCE_ASSEMBLY_LOGIC_MATH_PIXEL_CARDS",
                "status": "RUNNING"}
    (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), "utf-8")
    try:
        ocr = ParallelRowOCR(client_factory, workers=args.ocr_workers,
                             queue_size=args.ocr_queue_size,
                             batch_size=int(config["batch_size"]))
        gpu_dispatcher = None
        vl = (VisualTileQueue(None, output, workers=args.visual_workers,
                              queue_size=args.vl_queue_size,
                              card_hybrid_config=args.card_hybrid_config,
                              suit_backup_config=args.suit_backup_config,
                              rank_ncc_config=args.rank_ncc_config,
                              image_transport="ram", card_workers=args.card_workers)
              if cards_enabled else None)
        runtime_state.update(ocr=ocr, vl=vl, phase="capture")
        batch_size = int(config["batch_size"])
        first_table_batch = []
        submitted_first_table = []
        tick = time.perf_counter()
        capture_started = tick
        telemetry.capture_started = capture_started
        for row in stage.rows():
            rows.append(row)
            temporal.emitted(row)
            if vl is not None:
                vl.submit(row)
            if row.table_id == 1:
                first_table_batch.append(row)
                if len(first_table_batch) == batch_size:
                    ocr.submit_batch(first_table_batch)
                    submitted_first_table.extend(first_table_batch)
                    first_table_batch = []
            if len(rows) % 100 == 0:
                rss = process.memory_info().rss
                ram_peak = max(ram_peak, rss)
                child_rss = 0
                for child in process.children(recursive=True):
                    try:
                        child_rss += child.memory_info().rss
                    except psutil.Error:
                        pass
                pipeline_rss_peak = max(pipeline_rss_peak, rss + child_rss)
                rss_samples.append(rss)
        first_stage_wall = time.perf_counter() - tick
        capture_finished = time.perf_counter()
        runtime_state["phase"] = "tiny_drain"
        telemetry.add("first_stage_total", first_stage_wall, stage.seen_frames)
        if stage.seen_frames != args.frames:
            raise ValueError(f"Source ended after {stage.seen_frames}, expected {args.frames} frames")
        ram_peak = max(ram_peak, process.memory_info().rss)
        # Row order is fixed before OCR. The worker's completion order cannot
        # alter chronology because outputs are mapped back to stable row keys.
        tick = time.perf_counter()
        rows.sort(key=lambda r: (r.table_id, r.frame_id, r.source_span_index))
        telemetry.add("ordering_rows", time.perf_counter() - tick, len(rows))
        # The established model saw globally sorted batches of eight. Dispatch
        # complete first-table batches during capture, then preserve every
        # remaining batch boundary (including the table 1/2 boundary).
        offset = len(submitted_first_table)
        if [row_key(row) for row in rows[:offset]] != [row_key(row) for row in submitted_first_table]:
            raise ValueError("First-table OCR batch order differs from stable row order")
        for start in range(offset, len(rows), batch_size):
            ocr.submit_batch(rows[start:start + batch_size])
        ocr.finish()
        ocr_finished = max((end for _, end, _ in ocr.spans), default=time.perf_counter())
        recognized = ocr.results
        runtime_state["phase"] = "event_assembly"
        errors.extend(ocr.errors)
        with (output / "physical_rows.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps({"row_key": row_key(row), "row_id": row.stable_row_id,
                    "table_id": row.table_id, "table_session_id": row.session_id,
                    "frame_id": row.frame_id, "source_span_index": row.source_span_index,
                    "y1": row.y1, "epoch": row.epoch, "origin": row.origin,
                    "source_pts": row.source_pts, "time_base_num": row.time_base_num,
                    "time_base_den": row.time_base_den, "row_pixel_sha256": row.pixel_sha256,
                    "pixel_shape": list(row.pixels.shape), "pixel_dtype": str(row.pixels.dtype),
                    "row_pixels_base64": base64.b64encode(row.pixels.tobytes()).decode("ascii")},
                    ensure_ascii=False) + "\n")
        with (output / "raw_ocr_results.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps({"row_key": row_key(row),
                    "row_id": row.stable_row_id, "table_id": row.table_id, "epoch": row.epoch,
                    "table_session_id": row.session_id, "frame_id": row.frame_id,
                    "source_span_index": row.source_span_index,
                    "source_pts": row.source_pts, "time_base_num": row.time_base_num,
                    "time_base_den": row.time_base_den,
                    "row_pixel_sha256": row.pixel_sha256,
                    "raw_ocr_response": recognized.get(row_key(row))}, ensure_ascii=False) + "\n")
        ocr_overlap = sum(max(0, min(end, capture_finished) - start)
                          for start, end, _ in ocr.spans)
        overlap_intervals = sorted((max(start, capture_started), min(end, capture_finished))
                                   for start, end, _ in ocr.spans if start < capture_finished)
        overlap_wall = 0.0
        merged_end = capture_started
        for start, end in overlap_intervals:
            if end > merged_end:
                overlap_wall += end - max(start, merged_end)
                merged_end = end
        for row in rows:
            key = row_key(row)
            result = recognized.get(key, {"status": "ERROR", "raw_text": "", "score": None,
                                          "error": "OCR_WORKER_DID_NOT_RETURN"})
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
        timeline_before_dedup = len(timeline)
        dedup_started = time.perf_counter()
        timeline, dedup = temporal.reduce(timeline, sessions, ocr_status={
            row.stable_row_id: recognized.get(row_key(row), {}).get("status", "ERROR")
            for row in rows})
        telemetry.add("temporal_overlap_dedup", time.perf_counter() - dedup_started,
                      timeline_before_dedup)
        for name, evidence in (("removed_duplicates.jsonl", dedup.removals),
                               ("context_decisions.jsonl", dedup.decisions)):
            with (output / name).open("w", encoding="utf-8") as stream:
                for decision in evidence:
                    stream.write(json.dumps(decision, ensure_ascii=False) + "\n")
        for index, item in enumerate(timeline, 1):
            item["sequence_id"] = index
        telemetry.add("timeline_sort_and_sequence", time.perf_counter() - tick, len(timeline))
        # Group and project only the retained timeline; all raw evidence remains.
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
        if vl is not None:
            runtime_state["phase"] = "vl_drain"
            vl.finish()
            if gpu_dispatcher is not None:
                gpu_dispatcher.close()
            associate_visual_evidence(timeline, vl.results)
            project_chat_card_events(timeline, vl.results)
            vl.write_evidence(output / "visual_evidence.jsonl")
        runtime_state["phase"] = "poker_analysis"
        tick = time.perf_counter()
        hand_records, hand_summary = analyze_timeline(timeline, ocr_status={
            row.stable_row_id: recognized.get(row_key(row), {}).get("status", "ERROR")
            for row in rows})
        telemetry.add("poker_analysis", time.perf_counter() - tick, len(hand_records))
        for name, records in (("hand_analysis.jsonl", hand_records),
                              ("validated_hands.jsonl", [r for r in hand_records if r["fully_validated"]])):
            with (output / name).open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        (output / "hand_summary.json").write_text(
            json.dumps(hand_summary, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")
        tick = time.perf_counter()
        with (output / "ordered_timeline.jsonl").open("w", encoding="utf-8") as stream:
            for item in timeline:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        telemetry.add("jsonl_write", time.perf_counter() - tick, len(timeline))
        tick = time.perf_counter()
        runtime_state["phase"] = "excel_export"
        subprocess.run([str(args.node), str(ROOT / "scripts/build_smoke14_ram_workbook.mjs"),
                        str(output / "ordered_timeline.jsonl"), str(output / "ordered_timeline.xlsx"),
                        str(output / "visual_evidence.jsonl") if vl is not None else "",
                        str(output / "hand_analysis.jsonl")],
                       cwd=ROOT, check=True)
        telemetry.add("excel_builder_process", time.perf_counter() - tick, len(timeline))
        if errors or any(item["status"] == "ERROR" for item in recognized.values()):
            status = "PARTIAL_OCR_ERROR"
        elif vl is not None and any(item.get("status") != "RAW_UNREVIEWED"
                                      for item in vl.results.values()):
            status = "PARTIAL_VISUAL"
        else:
            status = "COMPLETE"
    except BaseException as exc:
        errors.append({"fatal": repr(exc)})
        raise
    finally:
        original_exception = sys.exc_info()[1]
        final_guard_failure = None
        if "ocr" in locals() and any(thread.is_alive() for thread in ocr._threads):
            ocr.finish()
        if "vl" in locals() and vl is not None and any(thread.is_alive() for thread in vl._threads):
            vl.finish()
        if "gpu_dispatcher" in locals() and gpu_dispatcher is not None:
            gpu_dispatcher.close()
        runtime_state["phase"] = status
        resource_stop.set()
        resource_thread.join()
        with (output / "resource_samples.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["elapsed_seconds", "pipeline_rss_mb",
                "cpu_percent_sum", "read_bytes_sum", "write_bytes_sum", "gpu_util_percent",
                "gpu_memory_used_mb", "gpu_memory_total_mb"])
            writer.writeheader()
            writer.writerows(resource_samples)
        wall = time.perf_counter() - wall_start
        try:
            production_guard.finish(stage.seen_frames, locals().get("first_stage_wall", wall), wall,
                                    process.io_counters().write_bytes-io_start.write_bytes)
        except RuntimeError as guard_error:
            # Preserve the original abort and still write FAILED/fatal artifacts.
            errors.append({"production_guard": str(guard_error)})
            status = "FAILED"
            final_guard_failure = guard_error
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
                   "peak_pipeline_rss_mb_sampled": pipeline_rss_peak / 1048576,
                   "peak_pipeline_rss_mb": resource_peak["rss_bytes"] / 1048576,
                   "peak_pipeline_cpu_percent_sampled": resource_peak["cpu_percent"],
                   "pipeline_cpu_percent_mean_sampled": statistics.fmean(x["cpu_percent_sum"] for x in resource_samples) if resource_samples else None,
                   "peak_gpu_util_percent_sampled": max((x["gpu_util_percent"] for x in resource_samples
                       if x["gpu_util_percent"] is not None), default=None),
                   "peak_gpu_memory_used_mb_sampled": max((x["gpu_memory_used_mb"] for x in resource_samples
                       if x["gpu_memory_used_mb"] is not None), default=None),
                   "gpu_sample_count": sum(x["gpu_util_percent"] is not None for x in resource_samples),
                   "disk_read_bytes": io_end.read_bytes - io_start.read_bytes,
                   "disk_write_bytes": io_end.write_bytes - io_start.write_bytes,
                   "pipeline_disk_read_bytes_sampled": max((x["read_bytes_sum"] for x in resource_samples), default=0),
                   "pipeline_disk_write_bytes_sampled": max((x["write_bytes_sum"] for x in resource_samples), default=0),
                   "files_created": len(files), "png_files": len(png_files),
                   "png_bytes": sum(p.stat().st_size for p in png_files),
                   "table_count": len(sessions), "table_changes": stage.metrics["changed_table_rois"],
                   "chat_changes": stage.metrics["changed_chat_rois"],
                   "physical_rows": len(rows), "ocr_calls": len(rows),
                   "timeline_rows": len(timeline), "message_counts": locals().get("message_counts", {}),
                   "timeline_rows_before_dedup": locals().get("timeline_before_dedup", len(timeline)),
                   "dedup_removed_rows": len(dedup.removals) if "dedup" in locals() else 0,
                   "dedup_visible_blocks": dedup.block_count if "dedup" in locals() else 0,
                   "dedup_compressed_ram_bytes": temporal.compressed_bytes,
                   "poker_analysis": locals().get("hand_summary"),
                   "unknown_rows": sum(x["event_type"] == "UNKNOWN_EVENT" for x in timeline),
                   "unresolved_rows": sum(x["status"] == "UNRESOLVED" for x in timeline),
                   "error_rows": sum(x["status"] == "ERROR" for x in timeline),
                   "retry_count": 0, "ocr_workers": args.ocr_workers,
                   "ocr_queue_size": args.ocr_queue_size,
                   "max_ocr_queue": ocr.max_queue if "ocr" in locals() else None,
                   "max_ocr_queue_unit": "batches",
                   "ocr_wall_seconds": (ocr_finished - ocr.first_started if "ocr_finished" in locals()
                                        and ocr.first_started is not None else None),
                   "ocr_throughput_rows_per_second": (len(rows) / (ocr_finished - ocr.first_started)
                                                      if "ocr_finished" in locals() and
                                                      ocr.first_started is not None and
                                                      ocr_finished > ocr.first_started else None),
                   "ocr_capture_overlap_worker_seconds": locals().get("ocr_overlap"),
                   "ocr_capture_overlap_wall_seconds": locals().get("overlap_wall"),
                   "ocr_started_during_capture": ("capture_finished" in locals() and
                                                  ocr.first_started is not None and
                                                  ocr.first_started < capture_finished),
                   "vl_workers": args.vl_workers, "vl_batch_size": args.vl_batch_size,
                   "visual_workers": args.visual_workers, "card_workers": args.card_workers,
                   "vl_gpu_startup_seconds": gpu_dispatcher.startup_seconds if "gpu_dispatcher" in locals() and gpu_dispatcher is not None else None,
                   "vl_gpu_batch_sizes": gpu_dispatcher.batch_sizes if "gpu_dispatcher" in locals() and gpu_dispatcher is not None else [],
                   "vl_gpu_batch_worker_seconds": sum(end-start for start, end in gpu_dispatcher.spans) if "gpu_dispatcher" in locals() and gpu_dispatcher is not None else 0,
                   "max_gpu_request_queue": gpu_dispatcher.max_queue if "gpu_dispatcher" in locals() and gpu_dispatcher is not None else 0,
                   "vl_jobs": (sum(not key.startswith("MISSING:") for key in vl.results)
                               if "vl" in locals() and vl is not None else 0),
                   "vl_model_responses": (sum(isinstance(item.get("raw_vl_response"), dict)
                                              for item in vl.results.values())
                                          if "vl" in locals() and vl is not None else 0),
                   "vl_missing_event_evidence": (sum(key.startswith("MISSING:") for key in vl.results)
                                                 if "vl" in locals() and vl is not None else 0),
                   "vl_board_jobs": (sum(item.get("type") == "BOARD" and
                                         not key.startswith("MISSING:")
                                         for key, item in vl.results.items())
                                     if "vl" in locals() and vl is not None else 0),
                   "vl_showdown_jobs": (sum(item.get("type") == "SHOWDOWN" and
                                            not key.startswith("MISSING:")
                                            for key, item in vl.results.items())
                                        if "vl" in locals() and vl is not None else 0),
                   "vl_unresolved_jobs": (sum(item.get("status") == "UNRESOLVED" and
                                               not key.startswith("MISSING:")
                                               for key, item in vl.results.items())
                                          if "vl" in locals() and vl is not None else 0),
                   "max_vl_queue": vl.max_queue if "vl" in locals() and vl is not None else 0,
                   "vl_wall_seconds": (max(end for _, end in vl.spans) - min(start for start, _ in vl.spans)
                                       if "vl" in locals() and vl is not None and vl.spans else None),
                   "vl_requests_per_second": (len(vl.spans) /
                                              (max(end for _, end in vl.spans) - min(start for start, _ in vl.spans))
                                              if "vl" in locals() and vl is not None and len(vl.spans) > 1 else None),
                   "vl_errors": vl.errors if "vl" in locals() and vl is not None else [],
                   "board_crop_worker_seconds": (sum(x.get("board_crop_elapsed_seconds", 0)
                       for x in vl.results.values() if x.get("crop_type") == "BOARD")
                       if "vl" in locals() and vl is not None else 0),
                   "board_vl_worker_seconds": (sum(x.get("board_vl_elapsed_seconds", 0)
                       for x in vl.results.values() if x.get("crop_type") == "BOARD")
                       if "vl" in locals() and vl is not None else 0),
                   "card_hybrid_rows": (sum(bool(x.get("hybrid")) for x in vl.results.values())
                                        if "vl" in locals() and vl is not None else 0),
                   "card_hybrid_proposed_rows": (sum(bool(x.get("hybrid", {}).get("cards"))
                                                 for x in vl.results.values())
                                                 if "vl" in locals() and vl is not None else 0),
                   "card_published_rows": (sum(x.get("card_reader") in {"vl_rank_template_suit_v1", "pixel_rank_ncc_template_suit_v1"}
                       and x.get("status") == "RAW_UNREVIEWED" for x in vl.results.values())
                       if "vl" in locals() and vl is not None else 0),
                   "card_hybrid_unresolved_reasons": (dict(Counter(x["hybrid"].get("reason")
                       for x in vl.results.values() if x.get("hybrid") and x["hybrid"].get("reason")))
                       if "vl" in locals() and vl is not None else {}),
                   "suit_backup_decisions": (dict(Counter(t.get("suit_source")
                       for x in vl.results.values() for t in x.get("hybrid", {}).get("tiles", [])))
                       if "vl" in locals() and vl is not None else {}),
                   "card_tile_count": (sum(len(x.get("hybrid", {}).get("tiles", []))
                       for x in vl.results.values()) if "vl" in locals() and vl is not None else 0),
                   "card_hybrid_worker_seconds": (sum(end-start for start, end in vl.hybrid_spans)
                       if "vl" in locals() and vl is not None else 0),
                   "card_hybrid_rows_per_worker_second": (
                       len(vl.hybrid_spans) / sum(end-start for start, end in vl.hybrid_spans)
                       if "vl" in locals() and vl is not None and vl.hybrid_spans else None),
                   "errors": errors, "first_stage_counts": dict(stage.metrics)}
        (output / "module_timings.json").write_text(json.dumps(telemetry.report(wall), indent=2), "utf-8")
        (output / "final_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
        (output / "errors_retries.json").write_text(json.dumps({"errors": errors, "retries": 0}, ensure_ascii=False, indent=2), "utf-8")
        metadata["status"] = status
        (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), "utf-8")
        if final_guard_failure is not None and original_exception is None:
            raise final_guard_failure


if __name__ == "__main__":
    main()
