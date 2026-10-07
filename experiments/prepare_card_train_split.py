"""Freeze source-held-out CARD train/validation manifests from verified CHAT tiles."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/card_train_smoke16_test_20260930"
SPLIT_OUT = OUT / "split_v2"
POOL = ROOT / "runs/paddle_vl15_poker_sft_20260930/combined_verified_chat_card_tiles.jsonl"
TRAIN_SOURCES = {"smoke_12.mp4", "smoke_13.mp4"}
VALID_SOURCES = {"smoke_14.mp4", "smoke_15.mp4"}
CLASSES = [rank + suit for rank in "23456789TJQKA" for suit in "cdhs"]

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def write_once(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"Frozen artifact differs: {path}")
    path.write_bytes(payload)

def main() -> None:
    SPLIT_OUT.mkdir(parents=True, exist_ok=True)
    source = POOL.read_bytes()
    tiles = [json.loads(line) for line in source.decode("utf-8").splitlines()]
    if len(tiles) != 877 or set(x["ground_truth"] for x in tiles) != set(CLASSES):
        raise ValueError("Verified pool/count changed")
    split = {"TRAIN": [], "VALIDATION": []}
    pixel_owner = {}
    image_data = {}
    for row in tiles:
        src = row["source_video"]
        bucket = "TRAIN" if src in TRAIN_SOURCES else "VALIDATION" if src in VALID_SOURCES else None
        if bucket is None:
            raise ValueError(f"Unexpected source: {src}")
        path = Path(row["tile_png"])
        raw = path.read_bytes()
        if sha(raw) != row["tile_sha256"]:
            raise ValueError(f"PNG SHA mismatch: {path}")
        with Image.open(path) as opened:
            rgb = opened.convert("RGB")
            pixels = np.asarray(rgb.convert("L"))
        if sha(rgb.tobytes()) != row["tile_pixel_sha256"]:
            raise ValueError(f"Pixel SHA mismatch: {path}")
        prior = pixel_owner.get(row["tile_pixel_sha256"])
        if prior and (prior[0] != bucket or prior[1] != row["ground_truth"]):
            raise ValueError(f"Exact-pixel leakage or conflicting label: {path}")
        pixel_owner[row["tile_pixel_sha256"]] = (bucket, row["ground_truth"])
        item = {**row, "split": bucket, "source_pool_sha256": sha(source)}
        split[bucket].append(item)
        image_data.setdefault(src, []).append((row["record_id"], row["ground_truth"], pixels))
    # Quarantine whole validation rows with any near-identical TRAIN tile.
    close = collections.Counter()
    quarantine_ids = set()
    near_label_conflicts = []
    for left_src in TRAIN_SOURCES:
        for right_src in VALID_SOURCES:
            for _, label_a, a in image_data[left_src]:
                for valid_id, label_b, b in image_data[right_src]:
                    if a.shape != b.shape:
                        continue
                    mae = float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))
                    if mae <= 1:
                        close["all_label_mae_le_1"] += 1
                    if mae <= 3:
                        close["all_label_mae_le_3"] += 1
                        quarantine_ids.add(valid_id)
                        if label_a != label_b:
                            near_label_conflicts.append({"validation_record_id": valid_id,
                                                         "train_label": label_a, "validation_label": label_b,
                                                         "mae": mae})
    quarantine = [row for row in split["VALIDATION"] if row["record_id"] in quarantine_ids]
    split["VALIDATION"] = [row for row in split["VALIDATION"] if row["record_id"] not in quarantine_ids]
    if not split["VALIDATION"]:
        raise ValueError("Near-duplicate quarantine emptied validation")
    for _, label_a, a in image_data["smoke_12.mp4"] + image_data["smoke_13.mp4"]:
        for valid_id, label_b, b in image_data["smoke_14.mp4"] + image_data["smoke_15.mp4"]:
            if valid_id in quarantine_ids or label_a != label_b:
                continue
            if a.shape == b.shape and float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)))) <= 3:
                raise ValueError("Near-identical TRAIN/VALIDATION tiles remain")
    for name, rows in split.items():
        payload = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n").encode("utf-8")
        write_once(SPLIT_OUT / f"{name.lower()}_manifest.jsonl", payload)
    write_once(SPLIT_OUT / "quarantined_validation_rows.jsonl",
               ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in quarantine) + "\n").encode("utf-8"))
    counts = {name: dict(sorted(collections.Counter(row["ground_truth"] for row in rows).items())) for name, rows in split.items()}
    summary = {"status": "FROZEN_BEFORE_SMOKE16_GT", "source_pool_sha256": sha(source),
               "source_split": {"TRAIN": sorted(TRAIN_SOURCES), "VALIDATION": sorted(VALID_SOURCES),
                                "TEST": ["smoke_16.mp4"]},
               "counts": {name: len(rows) for name, rows in split.items()},
               "classes": CLASSES, "class_counts": counts,
               "missing_validation_classes": sorted(set(CLASSES) - set(counts["VALIDATION"])),
               "exact_duplicate_pixels_within_pool": len(tiles) - len(pixel_owner),
               "cross_split_exact_duplicate_pixels": 0,
               "cross_source_near_pairs_before_quarantine": dict(close),
               "near_label_conflicts": near_label_conflicts,
               "quarantined_validation_tiles": len(quarantine),
               "quarantined_validation_rows": len(quarantine_ids),
               "near_match_rule": "same-size grayscale mean absolute pixel error <=3; quarantine entire validation parent row",
               "supersedes": "../train_manifest.jsonl and ../validation_manifest.jsonl (split v1 rejected for near-duplicate leakage)",
               "source_group_limit": "Only four source videos; within-video episode identity is incomplete."}
    write_once(SPLIT_OUT / "split_and_class_counts.json", (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({k: summary[k] for k in ("counts", "missing_validation_classes", "quarantined_validation_rows", "quarantined_validation_tiles", "near_label_conflicts")}, ensure_ascii=False))

if __name__ == "__main__":
    main()
