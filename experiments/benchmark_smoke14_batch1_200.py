"""Isolated 200-crop production-worker batch=1 speed and stability probe."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.tiny_chat_client import TinyChatClient


def main() -> None:
    benchmark = Path(sys.argv[1]).resolve(strict=True)
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/ocr_tiny_local.json").read_text("utf-8"))
    if config["device"] != "cpu" or config["retry_policy"] != "NONE_SINGLE_PASS":
        raise ValueError("The production Tiny CPU, single-pass config is required")
    original = json.loads((benchmark / "manifest.json").read_text("utf-8"))
    problem_ids = set()
    for size in (8, 16, 32):
        problem_ids.update(row["observation_id"] for row in json.loads(
            (benchmark / f"mismatches_batch{size}.json").read_text("utf-8")))
    if len(original) != 2000 or len(problem_ids) != 47:
        raise ValueError("Unexpected benchmark source size or problematic crop count")
    problems = [row for row in original if row["observation_id"] in problem_ids]
    others = [row for row in original if row["observation_id"] not in problem_ids]
    selected_others = [others[i * len(others) // (200 - len(problems))]
                       for i in range(200 - len(problems))]
    selected_ids = {row["observation_id"] for row in problems + selected_others}
    selected = [row for row in original if row["observation_id"] in selected_ids]
    if len(selected) != 200 or len({row["observation_id"] for row in selected}) != 200:
        raise ValueError("The fixed sample is not 200 unique crops")
    for row in selected:
        if hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Crop provenance mismatch: {row['observation_id']}")
    (output / "sample_manifest.json").write_text(
        json.dumps({"source_manifest": str(benchmark / "manifest.json"),
                    "selection": "all 47 mismatches plus 153 systematic remainder",
                    "rows": selected}, ensure_ascii=False, indent=2), "utf-8")

    client = TinyChatClient(Path(config["python"]), Path(config["model_dir"]),
                            config["model_name"], config["weights_sha256"],
                            "cpu", output / "worker.stderr.log")
    parent = psutil.Process()
    monitor_stop = threading.Event()
    samples = []

    def monitor():
        tracked = {}
        while not monitor_stop.is_set():
            processes = [parent]
            if client.process is not None:
                try:
                    worker = psutil.Process(client.process.pid)
                    processes.extend([worker, *worker.children(recursive=True)])
                except psutil.Error:
                    pass
            rss = cpu_percent = 0.0
            seen = set()
            for process in processes:
                try:
                    if process.pid in seen:
                        continue
                    seen.add(process.pid)
                    identity = (process.pid, process.create_time())
                    if identity not in tracked:
                        tracked[identity] = process
                        process.cpu_percent(None)
                    else:
                        cpu_percent += process.cpu_percent(None)
                    rss += process.memory_info().rss / 1024**2
                except psutil.Error:
                    pass
            samples.append({"wall_offset_seconds": time.perf_counter() - started,
                            "cpu_percent_one_core": cpu_percent, "rss_mb": rss,
                            "process_count": len(seen)})
            monitor_stop.wait(0.2)

    def recognize(row):
        response = client.read_batch([{"item_id": row["observation_id"],
                                       "path": row["path"],
                                       "source_sha256": row["sha256"]}], 1)
        result = response["results"][0]
        if result["item_id"] != row["observation_id"] or result["status"] == "ERROR":
            raise RuntimeError(f"Tiny failed for {row['observation_id']}: {result}")
        return result

    started = time.perf_counter()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    first = {}
    second = {}
    repeated = {}
    first_end = None
    worker_cpu_first = None
    parent_cpu_first = None
    try:
        parent_cpu_start = parent.cpu_times()
        for number, row in enumerate(selected, 1):
            first[row["observation_id"]] = recognize(row)
            if number % 50 == 0:
                print(json.dumps({"phase": "sample", "done": number,
                                  "wall_seconds": round(time.perf_counter() - started, 3)}), flush=True)
        first_end = time.perf_counter()
        parent_cpu_end = parent.cpu_times()
        parent_cpu_first = ((parent_cpu_end.user + parent_cpu_end.system) -
                            (parent_cpu_start.user + parent_cpu_start.system))
        if client.process is not None:
            worker = psutil.Process(client.process.pid)
            worker_cpu_first = sum(cpu.user + cpu.system for cpu in
                                   (process.cpu_times() for process in
                                    [worker, *worker.children(recursive=True)]))
        for row in selected:
            second[row["observation_id"]] = recognize(row)
        for number, row in enumerate(problems, 1):
            repeated[row["observation_id"]] = [recognize(row) for _ in range(5)]
            if number % 10 == 0:
                print(json.dumps({"phase": "problem_5x", "done": number}), flush=True)
    finally:
        monitor_stop.set()
        thread.join(timeout=2)
        client.close()
    if first_end is None:
        raise RuntimeError("First pass did not finish")
    first_wall = first_end - started
    first_samples = [row for row in samples if row["wall_offset_seconds"] <= first_wall]
    stable_twice = sum(first[row["observation_id"]]["raw_text"] ==
                       second[row["observation_id"]]["raw_text"] for row in selected)
    stable_five = sum(len({result["raw_text"] for result in repeated[row["observation_id"]]}) == 1
                      for row in problems)
    stable_first_plus_five = sum(len({first[row["observation_id"]]["raw_text"]} |
                                 {result["raw_text"] for result in repeated[row["observation_id"]]}) == 1
                                 for row in problems)
    details = []
    for row in problems:
        obs = row["observation_id"]
        details.append({"observation_id": obs, "crop_path": row["path"],
                        "crop_sha256": row["sha256"], "baseline_raw_text": row["baseline_text"],
                        "sample_raw_text": first[obs]["raw_text"],
                        "five_raw_texts": [result["raw_text"] for result in repeated[obs]],
                        "stable_5_of_5": len({result["raw_text"] for result in repeated[obs]}) == 1,
                        "stable_sample_plus_5": first[obs]["raw_text"] == repeated[obs][0]["raw_text"]
                                                   and len({result["raw_text"] for result in repeated[obs]}) == 1})
    (output / "problematic_results.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), "utf-8")
    with (output / "problematic_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["observation_id", "crop_path", "crop_sha256",
                                                  "baseline_raw_text", "sample_raw_text", "repeat_1",
                                                  "repeat_2", "repeat_3", "repeat_4", "repeat_5",
                                                  "stable_5_of_5", "stable_sample_plus_5"])
        writer.writeheader()
        for row in details:
            writer.writerow({**{key: value for key, value in row.items() if key != "five_raw_texts"},
                             **{f"repeat_{i}": value for i, value in enumerate(row["five_raw_texts"], 1)}})
    sample_rows = [{"observation_id": row["observation_id"], "crop_path": row["path"],
                    "crop_sha256": row["sha256"], "raw_text_first": first[row["observation_id"]]["raw_text"],
                    "raw_text_second": second[row["observation_id"]]["raw_text"],
                    "stable_2_of_2": first[row["observation_id"]]["raw_text"] ==
                                     second[row["observation_id"]]["raw_text"]}
                   for row in selected]
    (output / "sample_results.json").write_text(json.dumps(sample_rows, ensure_ascii=False, indent=2), "utf-8")
    summary = {"sample_crops": 200, "problematic_crops": len(problems),
               "model_name": config["model_name"], "model_version": config["model_version"],
               "device": "cpu", "batch_size": 1, "retry_policy": config["retry_policy"],
               "sample_wall_seconds_including_model_startup": first_wall,
               "model_startup_seconds": client.ready.get("startup_seconds"),
               "sample_rows_per_second": 200 / first_wall,
               "sample_rows_per_second_excluding_model_startup":
                   200 / (first_wall - client.ready["startup_seconds"]),
               "cpu_seconds_parent_first_pass": parent_cpu_first,
               "cpu_seconds_worker_first_pass": worker_cpu_first,
               "cpu_percent_one_core_average_first_pass":
                   100 * (parent_cpu_first + worker_cpu_first) / first_wall,
               "cpu_percent_machine_average_first_pass":
                   100 * (parent_cpu_first + worker_cpu_first) / first_wall / psutil.cpu_count(),
               "cpu_percent_one_core_peak_sample_first_pass":
                   max(row["cpu_percent_one_core"] for row in first_samples),
               "process_count_peak_first_pass": max(row["process_count"] for row in first_samples),
               "rss_mb_peak_first_pass": max(row["rss_mb"] for row in first_samples),
               "rss_mb_average_first_pass": sum(row["rss_mb"] for row in first_samples) / len(first_samples),
               "stable_all_sample_2_of_2": stable_twice,
               "stable_problematic_5_of_5": stable_five,
               "stable_problematic_sample_plus_5": stable_first_plus_five,
               "unstable_problematic": [row["observation_id"] for row in details if not row["stable_5_of_5"]]}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
