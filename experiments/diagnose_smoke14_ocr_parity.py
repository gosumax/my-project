"""Reproduce Smoke14 Tiny OCR disagreements without modifying production data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.tiny_chat_client import TinyChatClient

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


def source_data(run: Path, previous: Path, benchmark: Path):
    manifest = json.loads((benchmark / "manifest.json").read_text("utf-8"))
    by_id = {row["observation_id"]: row for row in manifest}
    ids = set()
    for size in (8, 16, 32):
        ids.update(row["observation_id"] for row in json.loads(
            (benchmark / f"mismatches_batch{size}.json").read_text("utf-8")))
    query = """SELECT o.observation_id,o.crop_path,o.crop_sha256,r.raw_text,
        r.preprocessing_json,r.model_version
        FROM observations o JOIN recognitions r ON r.observation_id=o.observation_id
        WHERE r.engine='PP-OCRv6_tiny_rec' AND r.revision=(
            SELECT MAX(r2.revision) FROM recognitions r2
            WHERE r2.observation_id=r.observation_id AND r2.engine=r.engine)"""
    with sqlite3.connect(run / "journal.sqlite3") as db:
        baseline = {row[0]: row for row in db.execute(query)}
    with sqlite3.connect(previous / "journal.sqlite3") as db:
        fast = {row[0]: row for row in db.execute(query)}
    fast_differences = {obs for obs in baseline.keys() & fast.keys()
                        if baseline[obs][3] != fast[obs][3]}
    ids |= fast_differences
    extra = ids - by_id.keys()
    if extra:
        raise ValueError(f"Missing benchmark manifest entries: {extra}")
    return manifest, by_id, baseline, fast, sorted(ids), fast_differences


def image_metadata(path: Path):
    data = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    height, width = image.shape[:2]
    target_width = min(3200, max(320, int(48 * max(320 / 48, width / height))))
    resized_width = min(target_width, math.ceil(48 * width / height))
    resized = cv2.resize(image, (resized_width, 48))
    normalized = resized.astype("float32").transpose((2, 0, 1)) / 255
    normalized -= 0.5
    normalized /= 0.5
    padded = np.zeros((3, 48, target_width), dtype=np.float32)
    padded[:, :, :resized_width] = normalized
    return {"png_sha256": hashlib.sha256(data).hexdigest(), "shape": list(image.shape),
            "dtype": str(image.dtype), "resize_shape": list(resized.shape),
            "resize_sha256": hashlib.sha256(resized.tobytes()).hexdigest(),
            "normalized_dtype": str(normalized.dtype),
            "normalized_min": float(normalized.min()), "normalized_max": float(normalized.max()),
            "normalized_sha256": hashlib.sha256(normalized.tobytes()).hexdigest(),
            "padded_shape": list(padded.shape),
            "padded_sha256": hashlib.sha256(padded.tobytes()).hexdigest(),
            "color_order": "BGR", "resize_height": 48,
            "resize_interpolation": "cv2.INTER_LINEAR", "normalization": "(uint8/255-0.5)/0.5"}


def make_batch(manifest, target_index, size, position):
    if size == 1:
        return [manifest[target_index]]
    target = manifest[target_index]
    neighbors = [row for i, row in enumerate(manifest)
                 if i != target_index and abs(i - target_index) <= size]
    if len(neighbors) < size - 1:
        neighbors = [row for i, row in enumerate(manifest) if i != target_index][:size - 1]
    neighbors = neighbors[:size - 1]
    return neighbors[:position] + [target] + neighbors[position:]


def main():
    run = Path(sys.argv[1]).resolve(strict=True)
    previous = Path(sys.argv[2]).resolve(strict=True)
    benchmark = Path(sys.argv[3]).resolve(strict=True)
    output = Path(sys.argv[4]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/ocr_tiny_local.json").read_text("utf-8"))
    manifest, by_id, baseline, fast, ids, fast_differences = source_data(
        run, previous, benchmark)
    model_dir = Path(config["model_dir"])
    import paddle
    import paddleocr
    import paddlex
    from paddleocr import TextRecognition
    import yaml

    model_yml = model_dir / "inference.yml"
    model_config = yaml.safe_load(model_yml.read_text("utf-8"))
    environment = {"python": sys.version, "paddle": paddle.__version__,
                   "paddleocr": paddleocr.__version__, "paddlex": paddlex.__version__,
                   "opencv": cv2.__version__, "device": config["device"],
                   "model_version": config["model_version"],
                   "weights_sha256": hashlib.sha256((model_dir / "inference.pdiparams").read_bytes()).hexdigest(),
                   "inference_json_sha256": hashlib.sha256((model_dir / "inference.json").read_bytes()).hexdigest(),
                   "inference_yml_sha256": hashlib.sha256(model_yml.read_bytes()).hexdigest(),
                   "character_dict_sha256": hashlib.sha256(json.dumps(
                       model_config["PostProcess"]["character_dict"], ensure_ascii=False).encode()).hexdigest(),
                   "character_dict_length": len(model_config["PostProcess"]["character_dict"]),
                   "transform_ops": model_config["PreProcess"]["transform_ops"],
                   "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                   "omp_thread_limit": os.environ.get("OMP_THREAD_LIMIT")}
    (output / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), "utf-8")
    items = []
    for obs in ids:
        entry = by_id[obs]
        path = Path(entry["path"])
        meta = image_metadata(path)
        base = baseline[obs]
        if meta["png_sha256"] != base[2]:
            raise ValueError(f"Baseline crop hash changed: {obs}")
        items.append({"observation_id": obs, "crop_path": str(path),
                      "baseline_raw_text": base[3], "fast_raw_text": fast.get(obs, (None,)*4)[3],
                      "fast_path_mismatch": obs in fast_differences,
                      "production_preprocessing": json.loads(base[4]),
                      "production_model_version": base[5], **meta})
    (output / "crop_metadata.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), "utf-8")

    worker = TinyChatClient(Path(config["python"]), model_dir, config["model_name"],
                            config["weights_sha256"], "cpu", output / "worker.stderr.log")
    direct = TextRecognition(model_name=config["model_name"], device="cpu")
    manifest_index = {row["observation_id"]: i for i, row in enumerate(manifest)}
    records = []
    started = time.perf_counter()
    try:
        for number, obs in enumerate(ids, 1):
            index = manifest_index[obs]
            for size in (1, 8, 16, 32):
                positions = [0] if size == 1 else [0, size - 1]
                for position in positions:
                    batch = make_batch(manifest, index, size, position)
                    paths = [item["path"] for item in batch]
                    for repeat in range(3):
                        result = list(direct.predict(paths, batch_size=size))
                        by_path = {str(Path(value["input_path"]).resolve()).lower(): value for value in result}
                        value = by_path[str(Path(by_id[obs]["path"]).resolve()).lower()]
                        records.append({"observation_id": obs, "api": "benchmark_direct",
                                        "batch_size": size, "batch_position": position,
                                        "neighbor_ids": [item["observation_id"] for item in batch if item["observation_id"] != obs],
                                        "repeat": repeat, "raw_text": (value.get("rec_text") or "").strip(),
                                        "score": float(value.get("rec_score", 0.0)),
                                        "baseline_text": baseline[obs][3]})
            # Exercise the actual production subprocess/path, with the identical input PNG.
            for repeat in range(3):
                entry = by_id[obs]
                response = worker.read_batch([{"item_id": obs, "path": entry["path"],
                                               "source_sha256": entry["sha256"]}], 1)
                value = response["results"][0]
                records.append({"observation_id": obs, "api": "production_worker",
                                "batch_size": 1, "batch_position": 0, "neighbor_ids": [],
                                "repeat": repeat, "raw_text": value["raw_text"],
                                "score": value["score"], "baseline_text": baseline[obs][3]})
            if number % 5 == 0:
                print(json.dumps({"done": number, "total": len(ids),
                                  "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
                (output / "partial_results.json").write_text(json.dumps(records, ensure_ascii=False), "utf-8")
    finally:
        worker.close()
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "neighbor_ids": json.dumps(record["neighbor_ids"])})
    (output / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), "utf-8")
    summary = {"crops": len(ids), "fast_path_mismatches": len(fast_differences),
               "runs": len(records), "wall_seconds": time.perf_counter() - started,
               "variants_per_crop": len(records) // len(ids)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
