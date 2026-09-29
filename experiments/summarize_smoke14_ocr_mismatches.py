"""Join reproducible OCR tests into one reviewable per-crop comparison."""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    benchmark = Path(sys.argv[1]).resolve(strict=True)
    diagnosis = Path(sys.argv[2]).resolve(strict=True)
    metadata = json.loads((diagnosis / "crop_metadata.json").read_text("utf-8"))
    results = json.loads((diagnosis / "results.json").read_text("utf-8"))
    partial = list(csv.DictReader((diagnosis / "partial_batch_results.csv").open(encoding="utf-8")))
    by_crop = defaultdict(list)
    by_partial = defaultdict(list)
    for record in results:
        by_crop[record["observation_id"]].append(record)
    for record in partial:
        by_partial[record["observation_id"]].append(record)
    benchmark_results = {}
    for size in (8, 16, 32):
        benchmark_results[size] = {row["observation_id"]: row for row in map(
            json.loads, (benchmark / f"results_batch{size}.jsonl").read_text("utf-8").splitlines())}
    rows = []
    for item in metadata:
        obs = item["observation_id"]
        variant = by_crop[obs]
        partial_variant = by_partial[obs]
        variants = defaultdict(set)
        for record in variant:
            key = (record["api"], record["batch_size"], record["batch_position"])
            variants[key].add(record["raw_text"])
        repeat_variation = any(len(values) > 1 for values in variants.values())
        reproduced = any(str(record["matches_baseline"]).lower() == "true"
                         for record in partial_variant)
        width_repro = diagnosis / f"padding_{obs}.json"
        if width_repro.exists():
            reproduced |= any(record["raw_text"] == item["baseline_raw_text"]
                              for record in json.loads(width_repro.read_text("utf-8")))
        rows.append({
            "observation_id": obs,
            "crop_path": item["crop_path"],
            "crop_sha256": item["png_sha256"],
            "dimensions_hwc": json.dumps(item["shape"]),
            "dtype": item["dtype"],
            "resize_shape": json.dumps(item["resize_shape"]),
            "resize_sha256": item["resize_sha256"],
            "normalized_dtype": item["normalized_dtype"],
            "normalized_min": item["normalized_min"],
            "normalized_max": item["normalized_max"],
            "normalized_sha256": item["normalized_sha256"],
            "preprocessing_json": json.dumps(item["production_preprocessing"], ensure_ascii=False),
            "production_batch_position": "UNKNOWN_NOT_LOGGED",
            "recognizer_model_version": item["production_model_version"],
            "production_raw_text": item["baseline_raw_text"],
            "benchmark_batch8_raw_text": benchmark_results[8][obs]["raw_text"],
            "benchmark_batch16_raw_text": benchmark_results[16][obs]["raw_text"],
            "benchmark_batch32_raw_text": benchmark_results[32][obs]["raw_text"],
            "fast_path_raw_text": item["fast_raw_text"] or "",
            "fast_path_mismatch": item["fast_path_mismatch"],
            "tested_api_size_position_text": json.dumps(
                {f"{api}:{size}:{position}": sorted(values)
                 for (api, size, position), values in variants.items()}, ensure_ascii=False),
            "tested_actual_batch_size_position_text": json.dumps(
                {f"{record['actual_batch_length']}:{record['position']}": record["raw_text"]
                 for record in partial_variant}, ensure_ascii=False),
            "baseline_reproduced_in_control": reproduced,
            "nondeterministic_under_identical_test": repeat_variation,
        })
    path = diagnosis / "mismatch_comparison.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"crops": len(rows),
               "baseline_reproduced": sum(r["baseline_reproduced_in_control"] for r in rows),
               "repeat_variation": sum(r["nondeterministic_under_identical_test"] for r in rows),
               "fast_path_mismatches": sum(r["fast_path_mismatch"] for r in rows)}
    (diagnosis / "comparison_summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
