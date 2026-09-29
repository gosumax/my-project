"""Isolated, same-crop PP-OCRv6 tiny CPU batch benchmark for Smoke14."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import statistics
import time
from pathlib import Path

import psutil

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/ocr_tiny_local.json").read_text("utf-8"))
    with sqlite3.connect(run / "journal.sqlite3") as db:
        source = db.execute("""SELECT o.observation_id,o.crop_path,o.crop_sha256,r.raw_text
            FROM physical_rows p JOIN row_observations ro ON ro.physical_row_id=p.physical_row_id
            JOIN observations o ON o.observation_id=ro.observation_id
            JOIN recognitions r ON r.observation_id=o.observation_id
            WHERE ro.ordinal=(SELECT MIN(ro2.ordinal) FROM row_observations ro2
                WHERE ro2.physical_row_id=p.physical_row_id)
              AND r.engine='PP-OCRv6_tiny_rec'
              AND r.revision=(SELECT MAX(r2.revision) FROM recognitions r2
                WHERE r2.observation_id=r.observation_id AND r2.engine=r.engine)
            ORDER BY o.frame_id,o.table_session_id,o.observation_id""").fetchall()
    if len(source) < args.count:
        raise ValueError(f"Only {len(source)} baseline rows, need {args.count}")
    indices = [i * len(source) // args.count for i in range(args.count)]
    manifest = []
    for i in indices:
        obs_id, relative, digest, baseline = source[i]
        path = (run / relative).resolve(strict=True)
        if not path.is_relative_to(run) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Crop provenance mismatch: {obs_id}")
        manifest.append({"observation_id": obs_id, "path": str(path),
                         "sha256": digest, "baseline_text": baseline or ""})
    manifest_path = output / "manifest.json"
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2)
    if manifest_path.exists() and manifest_path.read_text("utf-8") != serialized:
        raise ValueError("Fixed manifest differs from previous benchmark")
    manifest_path.write_text(serialized, "utf-8")

    from paddleocr import TextRecognition

    model = TextRecognition(model_name=config["model_name"], device="cpu")
    process = psutil.Process()
    process.cpu_percent()
    results = []
    for batch_size in args.batch_sizes:
        list(model.predict([item["path"] for item in manifest[:batch_size]], batch_size=batch_size))
        begun = time.perf_counter()
        cpu_begun = time.process_time()
        rss_peak = process.memory_info().rss
        batches = 0
        mismatches = []
        errors = []
        outputs = []
        for offset in range(0, len(manifest), batch_size):
            batch = manifest[offset:offset + batch_size]
            try:
                predictions = list(model.predict([item["path"] for item in batch], batch_size=batch_size))
                by_path = {str(Path(p["input_path"]).resolve()).lower(): p for p in predictions}
                for item in batch:
                    prediction = by_path.get(item["path"].lower())
                    if prediction is None:
                        errors.append({"observation_id": item["observation_id"], "error": "MISSING_RESULT"})
                        continue
                    raw = (prediction.get("rec_text") or "").strip()
                    outputs.append({"observation_id": item["observation_id"], "raw_text": raw,
                                    "score": prediction.get("rec_score")})
                    if raw != item["baseline_text"]:
                        mismatches.append({"observation_id": item["observation_id"],
                                           "baseline": item["baseline_text"], "actual": raw})
            except Exception as exc:
                errors.append({"offset": offset, "error": repr(exc)})
            batches += 1
            rss_peak = max(rss_peak, process.memory_info().rss)
            if batches % 50 == 0:
                print(json.dumps({"batch_size": batch_size, "done": min(offset + batch_size, len(manifest)),
                                  "elapsed": round(time.perf_counter() - begun, 2)}), flush=True)
        wall = time.perf_counter() - begun
        cpu = time.process_time() - cpu_begun
        (output / f"results_batch{batch_size}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs), "utf-8")
        (output / f"mismatches_batch{batch_size}.json").write_text(
            json.dumps(mismatches, ensure_ascii=False, indent=2), "utf-8")
        result = {"device": "cpu", "batch_size": batch_size, "images": len(outputs),
                  "batches": batches, "average_actual_batch": len(manifest) / batches,
                  "wall_seconds": wall, "cpu_seconds": cpu,
                  "cpu_percent_one_core": 100 * cpu / wall,
                  "cpu_percent_machine": 100 * cpu / wall / psutil.cpu_count(),
                  "rss_peak_mb": rss_peak / 1024**2,
                  "images_per_second": len(outputs) / wall,
                  "mismatches_vs_production": len(mismatches), "errors": errors}
        results.append(result)
        (output / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), "utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
