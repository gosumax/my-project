"""Overlap video capture, row tracking and batched single-pass tiny OCR."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_nine_tables_chunked import db_snapshot, run_json_stage
from scripts.ocr_chat_pilot import OCR_POLICY, ROW_OCR_RETRY_POLICY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        help="Overrides the configured local SSD run directory")
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--capture-workers", type=int, default=4)
    parser.add_argument("--ocr-batch-size", type=int, default=8)
    parser.add_argument("--db-batch-frames", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tiny-config", type=Path,
                        default=ROOT / "configs/ocr_tiny_local.json")
    args = parser.parse_args()
    if min(args.max_frames, args.capture_workers, args.ocr_batch_size,
           args.db_batch_frames) < 1:
        parser.error("worker, batch and frame counts must be positive")
    if args.output is None:
        storage = json.loads((ROOT / "configs/run_storage_local.json").read_text("utf-8"))
        args.output = Path(storage["data_root"]) / (
            args.video.stem + "_streaming_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output = args.output.resolve()
    if (output / "journal.sqlite3").exists() and not args.resume:
        parser.error("Use a fresh output directory for measured streaming runs")
    if args.resume and not (output / "journal.sqlite3").exists():
        parser.error("Resume requires an existing journal")
    output.mkdir(parents=True, exist_ok=True)
    previous_report = json.loads((output / "pilot_run.json").read_text("utf-8")) if args.resume else {}
    started = previous_report.get("started_utc", datetime.now(timezone.utc).isoformat())
    start_tick = time.perf_counter()
    previous_elapsed = previous_report.get("wall_seconds", 0.0)
    if args.resume:
        with sqlite3.connect(output / "journal.sqlite3") as db:
            last_frame = db.execute("SELECT MAX(frame_id) FROM frames").fetchone()[0]
        next_frame = (last_frame if last_frame is not None else -1) + 1
        if next_frame >= args.max_frames:
            parser.error("Requested frame window is already captured")
        if (output / "capture.done").exists() or (output / "tracking.done").exists():
            parser.error("Completion markers exist; inspect the run before resuming")
    else:
        next_frame = 0
    capture_done = output / "capture.done"
    tracking_done = output / "tracking.done"
    env = {**os.environ, "PYTHONUTF8": "1", "OMP_THREAD_LIMIT": "1", "OMP_NUM_THREADS": "1"}
    running = {}
    stages = []
    if args.resume:
        stages.append({"stage": "capture_before_resume", "wall_seconds": previous_elapsed,
                       "measurement": "approximate_from_runner_elapsed"})
        for stage_name, progress_file in (("track_rows_before_resume", "tracking_progress.json"),
                                          ("ocr_rows_before_resume", "ocr_progress.json")):
            path = output / progress_file
            if path.exists():
                progress = json.loads(path.read_text("utf-8"))
                stages.append({"stage": stage_name,
                               "wall_seconds": progress.get("wall_seconds"),
                               "details": {"performance": progress.get("performance", {}),
                                           "progress": progress}})
    handles = []
    completed_names = set()
    status = "RUNNING"
    report = {}

    def save_report():
        try:
            snapshot = db_snapshot(output / "journal.sqlite3")
        except sqlite3.OperationalError:
            snapshot = {}
        report.update(status=status, started_utc=started,
                      updated_utc=datetime.now(timezone.utc).isoformat(),
                      wall_seconds=round(previous_elapsed + time.perf_counter() - start_tick, 3),
                      counts=snapshot, stages=stages,
                      runner={"mode": "streaming_capture_track_ppocrv6_tiny",
                              "data_directory": str(output),
                              "resumed_after_frame": next_frame - 1 if args.resume else None,
                              "capture_workers": args.capture_workers,
                              "ocr_batch_size": args.ocr_batch_size,
                              "db_batch_frames": args.db_batch_frames},
                      versions={"ocr_policy": OCR_POLICY,
                                "ocr_retry_policy": ROW_OCR_RETRY_POLICY,
                                "tiny_config": str(args.tiny_config.resolve())},
                      hh_status="NO_VERIFIED_EXPORT")
        temporary = output / "pilot_run.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        target = output / "pilot_run.json"
        for attempt in range(10):
            try:
                temporary.replace(target)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05)

    def launch(name, script, arguments):
        stdout = (output / f"{name}.stdout.log").open("w", encoding="utf-8")
        stderr = (output / f"{name}.stderr.log").open("w", encoding="utf-8")
        handles.extend([stdout, stderr])
        process = subprocess.Popen([sys.executable, str(ROOT / "scripts" / script),
                                    *map(str, arguments)], cwd=ROOT, env=env,
                                   stdout=stdout, stderr=stderr)
        running[name] = (process, time.perf_counter())

    try:
        launch("capture", "capture_nine_tables.py", [args.video, "--output", output,
               "--start-frame", next_frame,
               "--max-frames", args.max_frames - next_frame,
               "--capture-workers", args.capture_workers,
               "--db-batch-frames", args.db_batch_frames])
        # Initialize the schema through the producer before opening consumer connections.
        def schema_ready():
            if not (output / "journal.sqlite3").exists():
                return False
            with sqlite3.connect(output / "journal.sqlite3") as db:
                return db.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
                                  "AND name IN ('observations','recognitions','physical_rows',"
                                  "'row_decisions','row_suppressions','exports','gaps')").fetchone()[0] == 7

        while not schema_ready():
            if running["capture"][0].poll() is not None:
                raise RuntimeError("Capture failed before journal initialization")
            time.sleep(0.1)
        launch("track_rows", "track_chat_rows.py", [output, "--watch-until", capture_done,
                                                      *(["--resume"] if args.resume else [])])
        launch("ocr_rows", "ocr_chat_pilot.py", [output, "--roi-type", "CHAT_ROW",
               "--watch-until", tracking_done, "--tiny-config", args.tiny_config,
               "--batch-size", args.ocr_batch_size, "--summary-only"])
        tracked_processes = {}
        psutil.cpu_percent()
        sample_tick = 0.0
        sample_file = output / "resource_samples.csv"
        with sample_file.open("a" if args.resume else "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            if not args.resume:
                writer.writerow(["elapsed_seconds", "system_cpu_percent", "pipeline_cpu_percent",
                                 "pipeline_rss_mb", "process_count", "tiny_worker_process_count",
                                 "gpu_percent", "gpu_memory_mb"])
            while len(completed_names) < len(running):
                for name, (process, tick) in running.items():
                    code = process.poll()
                    if code is None or name in completed_names:
                        continue
                    completed_names.add(name)
                    stages.append({"stage": name, "wall_seconds": round(time.perf_counter() - tick, 3),
                                   "exit_code": code})
                    if code != 0:
                        raise RuntimeError(f"{name} failed with exit code {code}; see {name}.stderr.log")
                    if name == "capture":
                        capture_done.write_text("OK", encoding="ascii")
                    elif name == "track_rows":
                        tracking_done.write_text("OK", encoding="ascii")
                if time.perf_counter() - sample_tick >= 2:
                    sample_tick = time.perf_counter()
                    children = psutil.Process().children(recursive=True)
                    cpu_sum = rss = tiny_count = 0
                    for process in children:
                        try:
                            identity = (process.pid, process.create_time())
                            cached = tracked_processes.setdefault(identity, process)
                            cpu_sum += cached.cpu_percent() / psutil.cpu_count()
                            rss += cached.memory_info().rss / 1024**2
                            command = " ".join(cached.cmdline()).lower()
                            tiny_count += "tiny_chat_worker.py" in command
                        except psutil.Error:
                            pass
                    gpu = [None, None]
                    try:
                        raw = subprocess.check_output(["nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                            text=True, timeout=3).strip().splitlines()[0]
                        gpu = [float(value.strip()) for value in raw.split(",")]
                    except (OSError, ValueError, subprocess.SubprocessError):
                        pass
                    writer.writerow([round(previous_elapsed + time.perf_counter() - start_tick, 3), psutil.cpu_percent(),
                                     round(cpu_sum, 2), round(rss, 1), len(children), tiny_count, *gpu])
                    stream.flush()
                    save_report()
                time.sleep(0.2)
        for stage, script in [("assemble_messages", "assemble_messages.py"),
                              ("normalize_messages", "normalize_messages.py"),
                              ("reduce_hands", "reduce_hands.py"),
                              ("export_diagnostics", "export_diagnostics.py")]:
            stages.append(run_json_stage(stage, [sys.executable,
                          str(ROOT / "scripts" / script), str(output)], ROOT))
            save_report()
        status = "PARTIAL_PIPELINE"
    except BaseException:
        status = "FAILED"
        (output / "runner_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        for process, _ in running.values():
            if process.poll() is None:
                try:
                    descendants = psutil.Process(process.pid).children(recursive=True)
                    for child in reversed(descendants):
                        child.terminate()
                    process.terminate()
                except psutil.Error:
                    pass
        for process, _ in running.values():
            process.wait()
        raise
    finally:
        for handle in handles:
            handle.close()
        for name, filename in [("capture", "summary.json"),
                               ("track_rows", "row_tracking_summary.json"),
                               ("ocr_rows", "ocr_summary.json")]:
            if (output / filename).exists():
                details = json.loads((output / filename).read_text("utf-8"))
                for stage in stages:
                    if stage["stage"] == name:
                        stage["details"] = details
        save_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
