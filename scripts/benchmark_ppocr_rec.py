"""Benchmark PaddleOCR line-recognition models on persisted row crops."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

import psutil


MODEL_ALIASES = {
    "PP-OCRv6_tiny": "PP-OCRv6_tiny_rec",
    "PP-OCRv6_small": "PP-OCRv6_small_rec",
    "PP-OCRv5_mobile_rec": "PP-OCRv5_mobile_rec",
}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def allocate_counts(group_sizes: dict[str, int], total: int) -> dict[str, int]:
    available = sum(group_sizes.values())
    if total > available:
        raise ValueError(f"Requested {total} rows but only {available} are available")
    exact = {key: total * size / available for key, size in group_sizes.items()}
    allocated = {key: min(size, math.floor(exact[key])) for key, size in group_sizes.items()}
    remaining = total - sum(allocated.values())
    order = sorted(group_sizes, key=lambda key: (exact[key] - allocated[key], group_sizes[key]), reverse=True)
    for key in order:
        if remaining == 0:
            break
        if allocated[key] < group_sizes[key]:
            allocated[key] += 1
            remaining -= 1
    return allocated


def systematic_sample(rows: list[dict], count: int) -> list[dict]:
    if count >= len(rows):
        return rows
    indexes = [min(len(rows) - 1, math.floor((i + 0.5) * len(rows) / count)) for i in range(count)]
    return [rows[index] for index in indexes]


def latest_recognitions(connection: sqlite3.Connection, observation_ids: list[str]) -> dict[tuple[str, str], str]:
    wanted = set(observation_ids)
    output: dict[tuple[str, str], tuple[int, str]] = {}
    for observation_id, revision, raw_text, engine in connection.execute(
        "SELECT observation_id, revision, raw_text, engine FROM recognitions "
        "WHERE engine IN ('Tesseract', 'PaddleOCR-VL-1.5')"
    ):
        if observation_id not in wanted:
            continue
        key = (observation_id, engine)
        previous = output.get(key)
        if previous is None or revision > previous[0]:
            output[key] = (revision, raw_text or "")
    return {key: value[1] for key, value in output.items()}


def create_manifest(run_dir: Path, output_path: Path, count: int) -> list[dict]:
    database = run_dir / "journal.sqlite3"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    query = """
        WITH candidates AS (
            SELECT
                p.physical_row_id,
                p.table_session_id,
                p.first_source_pts,
                p.last_source_pts,
                o.observation_id,
                o.frame_id,
                o.crop_path,
                o.crop_sha256,
                d.decision,
                ROW_NUMBER() OVER (
                    PARTITION BY p.physical_row_id
                    ORDER BY
                        CASE d.decision
                            WHEN 'INITIAL_VISIBLE' THEN 0
                            WHEN 'NEW_CANDIDATE' THEN 1
                            WHEN 'TRACKED' THEN 2
                            ELSE 3
                        END,
                        o.frame_id,
                        o.observation_id
                ) AS choice
            FROM physical_rows p
            JOIN row_decisions d ON d.physical_row_id = p.physical_row_id
            JOIN observations o ON o.observation_id = d.observation_id
            WHERE o.roi_type = 'CHAT_ROW'
        )
        SELECT * FROM candidates WHERE choice = 1
        ORDER BY table_session_id, frame_id, physical_row_id
    """
    by_table: dict[str, list[dict]] = defaultdict(list)
    for row in connection.execute(query):
        item = dict(row)
        crop_path = Path(item["crop_path"])
        if not crop_path.is_absolute():
            crop_path = run_dir / crop_path
        if crop_path.is_file():
            item["crop_path"] = str(crop_path.resolve())
            by_table[item["table_session_id"]].append(item)

    allocations = allocate_counts({key: len(rows) for key, rows in by_table.items()}, count)
    selected: list[dict] = []
    for table_id in sorted(by_table):
        selected.extend(systematic_sample(by_table[table_id], allocations[table_id]))
    selected.sort(key=lambda item: (item["frame_id"], item["table_session_id"], item["physical_row_id"]))

    recognition_map = latest_recognitions(connection, [item["observation_id"] for item in selected])
    connection.close()
    for sample_id, item in enumerate(selected, start=1):
        item["sample_id"] = sample_id
        item["existing_tesseract"] = recognition_map.get((item["observation_id"], "Tesseract"), "")
        item["existing_vl"] = recognition_map.get((item["observation_id"], "PaddleOCR-VL-1.5"), "")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "physical_row_id", "table_session_id", "observation_id", "frame_id",
        "first_source_pts", "last_source_pts", "decision", "crop_path", "crop_sha256",
        "existing_tesseract", "existing_vl",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: item.get(field, "") for field in fields} for item in selected])
    return selected


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class ResourceMonitor:
    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.samples: list[dict] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _gpu(self) -> tuple[float | None, float | None]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, check=True,
            )
            utilization, memory = result.stdout.strip().split(",", 1)
            return float(utilization.strip()), float(memory.strip())
        except Exception:
            return None, None

    def _run(self) -> None:
        process = psutil.Process()
        process.cpu_percent(None)
        logical_cpus = psutil.cpu_count(logical=True) or 1
        while not self.stop_event.wait(self.interval):
            gpu_util, gpu_memory = self._gpu()
            process_cpu = process.cpu_percent(None)
            self.samples.append({
                "elapsed_s": time.perf_counter(),
                "system_cpu_pct": psutil.cpu_percent(None),
                "process_cpu_pct_one_core_100": process_cpu,
                "process_cpu_pct_total_capacity": process_cpu / logical_cpus,
                "process_rss_mb": process.memory_info().rss / (1024 * 1024),
                "system_ram_pct": psutil.virtual_memory().percent,
                "gpu_util_pct_shared": gpu_util,
                "gpu_memory_mb_shared": gpu_memory,
            })

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> list[dict]:
        self.stop_event.set()
        self.thread.join(timeout=5)
        return self.samples


def resource_summary(samples: list[dict]) -> dict:
    output = {}
    for key in (
        "system_cpu_pct", "process_cpu_pct_one_core_100", "process_cpu_pct_total_capacity",
        "process_rss_mb", "system_ram_pct", "gpu_util_pct_shared", "gpu_memory_mb_shared",
    ):
        values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
        output[key] = {
            "mean": statistics.fmean(values) if values else None,
            "p95": percentile(values, 0.95),
            "max": max(values) if values else None,
        }
    return output


def run_model(manifest: list[dict], model_label: str, output_dir: Path, batch_size: int) -> dict:
    from paddleocr import TextRecognition

    official_name = MODEL_ALIASES[model_label]
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    init_started = time.perf_counter()
    model = TextRecognition(model_name=official_name, device="cpu")
    init_seconds = time.perf_counter() - init_started

    paths = [item["crop_path"] for item in manifest]
    warmup_paths = paths[: min(16, len(paths))]
    warmup_started = time.perf_counter()
    list(model.predict(warmup_paths, batch_size=batch_size))
    warmup_seconds = time.perf_counter() - warmup_started

    monitor = ResourceMonitor()
    monitor.start()
    measured_started = time.perf_counter()
    rows: list[dict] = []
    errors = 0
    output_path = output_dir / f"results_{model_label}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for offset in range(0, len(manifest), batch_size):
            batch_items = manifest[offset : offset + batch_size]
            batch_paths = [item["crop_path"] for item in batch_items]
            batch_started = time.perf_counter()
            try:
                results = list(model.predict(batch_paths, batch_size=batch_size))
                batch_seconds = time.perf_counter() - batch_started
                by_path = {str(Path(result["input_path"]).resolve()).lower(): result for result in results}
                for item in batch_items:
                    result = by_path.get(str(Path(item["crop_path"]).resolve()).lower())
                    if result is None:
                        errors += 1
                        row = {**item, "model": model_label, "official_model": official_name,
                               "raw_text": "", "score": None, "status": "MISSING_RESULT",
                               "batch_seconds": batch_seconds, "batch_size": len(batch_items)}
                    else:
                        row = {**item, "model": model_label, "official_model": official_name,
                               "raw_text": result.get("rec_text") or "",
                               "score": float(result.get("rec_score", 0.0)), "status": "OK",
                               "batch_seconds": batch_seconds, "batch_size": len(batch_items)}
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as exc:
                batch_seconds = time.perf_counter() - batch_started
                for item in batch_items:
                    errors += 1
                    row = {**item, "model": model_label, "official_model": official_name,
                           "raw_text": "", "score": None, "status": "ERROR",
                           "error": repr(exc), "batch_seconds": batch_seconds,
                           "batch_size": len(batch_items)}
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if len(rows) % 200 < batch_size:
                elapsed = time.perf_counter() - measured_started
                print(json.dumps({"model": model_label, "done": len(rows), "total": len(manifest),
                                  "elapsed_s": round(elapsed, 3)}, ensure_ascii=False), flush=True)
    measured_seconds = time.perf_counter() - measured_started
    samples = monitor.stop()

    scores = [row["score"] for row in rows if row.get("score") is not None]
    nonempty = [row for row in rows if (row.get("raw_text") or "").strip()]
    summary = {
        "model": model_label,
        "official_model": official_name,
        "device": "cpu",
        "batch_size": batch_size,
        "images": len(rows),
        "init_seconds": init_seconds,
        "warmup_images": len(warmup_paths),
        "warmup_seconds": warmup_seconds,
        "measured_seconds": measured_seconds,
        "images_per_second": len(rows) / measured_seconds if measured_seconds else None,
        "milliseconds_per_image": measured_seconds * 1000 / len(rows) if rows else None,
        "nonempty_count": len(nonempty),
        "empty_count": len(rows) - len(nonempty),
        "empty_rate": (len(rows) - len(nonempty)) / len(rows) if rows else None,
        "error_count": errors,
        "score_mean": statistics.fmean(scores) if scores else None,
        "score_p50": percentile(scores, 0.50),
        "score_p95": percentile(scores, 0.95),
        "resources": resource_summary(samples),
        "resource_samples": len(samples),
        "gpu_note": "GPU metrics are shared with the concurrent production VL process and are not attributable to this CPU benchmark.",
    }
    (output_dir / f"summary_{model_label}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODEL_ALIASES), required=True)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest_3000.csv"
    if manifest_path.exists():
        manifest = read_manifest(manifest_path)
        if len(manifest) != args.count:
            raise ValueError(f"Existing manifest has {len(manifest)} rows; expected {args.count}")
    else:
        manifest = create_manifest(args.run_dir.resolve(), manifest_path, args.count)
    summary = run_model(manifest, args.model, args.output_dir, args.batch_size)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
