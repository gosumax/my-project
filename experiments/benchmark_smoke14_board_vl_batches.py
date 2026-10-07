"""Isolated, one-model CUDA batch benchmark on saved Smoke14 BOARD PNGs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gpu_mib() -> tuple[int | None, int | None, int | None]:
    try:
        raw = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True, timeout=5).strip().splitlines()[0]
        return tuple(int(part.strip()) for part in raw.split(","))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, None, None


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * q)]


class Monitor:
    def __init__(self):
        self.samples: list[dict] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        process = psutil.Process()
        process.cpu_percent(None)
        while not self.stop.is_set():
            used, total, util = gpu_mib()
            self.samples.append({"gpu_used_mib": used, "gpu_total_mib": total,
                                 "gpu_util_percent": util,
                                 "rss_mib": process.memory_info().rss / 1048576,
                                 "cpu_percent": process.cpu_percent(None)})
            self.stop.wait(0.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=6)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--allow-batch8", action="store_true")
    parser.add_argument("--allow-batch16", action="store_true")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--previous-peak-gpu-mib", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text("utf-8"))
    records = [json.loads(line) for line in args.evidence.read_text("utf-8").splitlines()]
    boards = [r for r in records if r.get("crop_type") == "BOARD" and r.get("crop_sha256")]
    unique = {}
    for row in boards:
        unique.setdefault(row["crop_sha256"], row)
    # Round robin over tables gives a bounded, varied subset and stable ordering.
    by_table = {}
    for row in sorted(unique.values(), key=lambda r: (r["frame_id"], r["table_id"])):
        by_table.setdefault(row["table_id"], []).append(row)
    selected = []
    while len(selected) < min(args.limit, len(unique)) and any(by_table.values()):
        for table in sorted(by_table):
            if by_table[table] and len(selected) < args.limit:
                selected.append(by_table[table].pop(0))
    for row in selected:
        path = Path(row["crop_path"])
        if sha(path) != row["crop_sha256"]:
            raise ValueError(f"Crop SHA mismatch: {path}")
    manifest = [{"sha256": r["crop_sha256"], "path": r["crop_path"],
                 "table_id": r["table_id"], "frame_id": r["frame_id"],
                 "street": r["street"]} for r in selected]
    (args.output / "dataset.json").write_text(json.dumps({
        "available_unique_board_crops": len(unique), "selected": manifest,
    }, ensure_ascii=False, indent=2), "utf-8")

    import torch
    from PIL import Image
    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              StoppingCriteria, StoppingCriteriaList)
    from parser_core.visual_cards import parse_cards

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    model_dir = Path(config["model_dir"])
    if sha(model_dir / "model.safetensors") != config["weights_sha256"]:
        raise ValueError("Model weights SHA mismatch")
    before_load = gpu_mib()
    load_start = time.perf_counter()
    torch.set_num_threads(3)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.float16, attn_implementation="sdpa",
        local_files_only=True).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True,
                                              use_fast=False)
    torch.cuda.synchronize()
    after_load = gpu_mib()
    load_seconds = time.perf_counter() - load_start
    devices = sorted({str(p.device) for p in model.parameters()})
    if devices != ["cuda:0"]:
        raise RuntimeError(f"Model is not wholly on CUDA: {devices}")
    (args.output / "hardware.json").write_text(json.dumps({
        "before_load_gpu": before_load, "after_load_gpu": after_load,
        "load_seconds": load_seconds, "model_devices": devices,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "model_revision": config.get("model_revision"),
        "weights_sha256": config["weights_sha256"],
        "ram_total_mib": psutil.virtual_memory().total / 1048576,
    }, indent=2), "utf-8")

    def infer(rows):
        class GpuFreeGuard(StoppingCriteria):
            def __init__(self):
                self.hit = False

            def __call__(self, input_ids, scores, **kwargs):
                self.hit = torch.cuda.mem_get_info()[0] < 768 * 1048576
                return self.hit

        messages = []
        for row in rows:
            with Image.open(row["crop_path"]) as opened:
                image = opened.convert("RGB")
            messages.append([{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "OCR:"}]}])
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, padding=True,
            return_dict=True, return_tensors="pt",
            images_kwargs={"size": {"shortest_edge": processor.image_processor.min_pixels,
                                     "longest_edge": 1280 * 28 * 28}}).to(model.device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        guard = GpuFreeGuard() if len(rows) >= 16 else None
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                     use_cache=True,
                                     stopping_criteria=StoppingCriteriaList([guard]) if guard else None)
        torch.cuda.synchronize()
        if guard and guard.hit:
            raise RuntimeError("GPU free memory fell below 768 MiB; batch aborted")
        duration = time.perf_counter() - started
        if len(outputs) != len(rows):
            raise RuntimeError("Batch output count differs from input count")
        texts = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
        return duration, [text.strip() for text in texts]

    baseline = {}
    if args.baseline_results:
        for line in args.baseline_results.read_text("utf-8").splitlines():
            item = json.loads(line)
            baseline[item["sha256"]] = item
    summary = []
    for batch_size in args.batch_sizes:
        if batch_size == 8 and not args.allow_batch8:
            continue
        if batch_size == 16 and not args.allow_batch16:
            continue
        if batch_size in (8, 16) and args.previous_peak_gpu_mib is not None:
            if after_load[1] - args.previous_peak_gpu_mib < 1024:
                summary.append({"batch_size": batch_size, "status": "SKIPPED_MEMORY_GUARD",
                                "previous_peak_gpu_used_mib": args.previous_peak_gpu_mib})
                break
        if summary:
            previous = summary[-1]
            if previous.get("status") != "OK":
                break
            peak = previous.get("peak_gpu_used_mib")
            if peak is not None and (after_load[1] - peak) < (1024 if batch_size == 8 else 768):
                summary.append({"batch_size": batch_size, "status": "SKIPPED_MEMORY_GUARD",
                                "previous_peak_gpu_used_mib": peak})
                break
        try:
            # Warm up this exact batch shape outside measured time.
            for _ in range(2):
                infer(selected[:batch_size])
            torch.cuda.reset_peak_memory_stats()
            results = []
            batch_times = []
            with Monitor() as monitor:
                wall_start = time.perf_counter()
                for offset in range(0, len(selected), batch_size):
                    rows = selected[offset:offset + batch_size]
                    elapsed, texts = infer(rows)
                    batch_times.append(elapsed)
                    for position, (row, raw) in enumerate(zip(rows, texts)):
                        expected = {"FLOP": 3, "TURN": 4, "RIVER": 5}[row["street"]]
                        cards = parse_cards(raw, expected, row["street"])
                        item = {"sha256": row["crop_sha256"], "path": row["crop_path"],
                                "street": row["street"], "batch_size": batch_size,
                                "actual_batch_size": len(rows), "batch_position": position,
                                "batch_seconds": elapsed, "raw_text": raw,
                                "cards": cards, "status": "OK" if raw else "EMPTY"}
                        if batch_size == 1:
                            baseline[item["sha256"]] = item
                        else:
                            other = baseline[item["sha256"]]
                            item["parity"] = ("EXACT_SAME" if raw == other["raw_text"] else
                                              "NORMALIZED_SAME" if cards is not None and cards == other["cards"] else
                                              "INVALID" if cards is None else "DIFFERENT")
                        results.append(item)
                wall = time.perf_counter() - wall_start
            (args.output / f"results_batch{batch_size}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), "utf-8")
            gpu_samples = [s["gpu_used_mib"] for s in monitor.samples if s["gpu_used_mib"] is not None]
            util_samples = [s["gpu_util_percent"] for s in monitor.samples if s["gpu_util_percent"] is not None]
            entry = {"batch_size": batch_size, "status": "OK", "crops": len(results),
                     "wall_seconds": wall, "crops_per_second": len(results) / wall,
                     "seconds_per_crop": wall / len(results),
                     "median_batch_seconds": statistics.median(batch_times),
                     "p95_batch_seconds": percentile(batch_times, .95),
                     "batch_seconds": batch_times,
                     "last_actual_batch_size": len(selected) % batch_size or batch_size,
                     "peak_gpu_used_mib": max(gpu_samples) if gpu_samples else None,
                     "peak_torch_allocated_mib": torch.cuda.max_memory_allocated() / 1048576,
                     "peak_ram_rss_mib": max(s["rss_mib"] for s in monitor.samples),
                     "mean_gpu_util_percent": statistics.mean(util_samples) if util_samples else None,
                     "mean_cpu_percent": statistics.mean(s["cpu_percent"] for s in monitor.samples),
                     "parity_counts": {k: sum(r.get("parity") == k for r in results)
                                       for k in ("EXACT_SAME", "NORMALIZED_SAME", "DIFFERENT", "INVALID")}}
        except Exception as exc:
            entry = {"batch_size": batch_size, "status": "ERROR", "error": repr(exc),
                     "cuda_oom": isinstance(exc, torch.cuda.OutOfMemoryError)}
        summary.append(entry)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        if entry["status"] != "OK":
            raise SystemExit(1)


if __name__ == "__main__":
    run()
