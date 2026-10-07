"""Deterministic card-image comparison with source-held-out validation.

The template bank contains only VERIFIED TRAIN CHAT crops. Smoke16 evaluation is
explicitly post-baseline exploratory and cannot replace its first blind result.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs/card_train_smoke16_test_20260930"
SPLIT = DATA / "split_v2"
OUT = DATA / "stage2_template"
METHODS = ("raw_equal", "raw_suit2", "binary_equal", "binary_suit2")

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

def pixels(row: dict) -> np.ndarray:
    path = Path(row["tile_png"])
    data = path.read_bytes()
    if sha(data) != row["tile_sha256"]:
        raise ValueError(f"Template/input PNG changed: {path}")
    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
        if sha(rgb.tobytes()) != row["tile_pixel_sha256"]:
            raise ValueError(f"Template/input pixels changed: {path}")
        gray = np.asarray(rgb.convert("L"))
    if gray.shape[0] != 16 or gray.shape[1] not in (20, 21):
        raise ValueError(f"Unexpected tile geometry: {path}")
    return np.ascontiguousarray(gray[:, :20])

def predict(bank: np.ndarray, labels: list[str], query: np.ndarray, mode: str) -> tuple[str, float, float, str]:
    if mode.startswith("binary"):
        x = (bank < 180).astype(np.int16)
        y = (query < 180).astype(np.int16)
    else:
        x, y = bank.astype(np.int16), query.astype(np.int16)
    diff = np.abs(x - y)
    # Skip top/bottom border and strongly weight suit area in two candidate rules.
    diff = diff[:, 2:14, 1:19]
    if mode.endswith("suit2"):
        diff[:, :, 9:] *= 2
    distances = diff.mean(axis=(1, 2))
    order = np.argsort(distances, kind="stable")
    best_idx = int(order[0])
    best_label = labels[best_idx]
    best = float(distances[best_idx])
    other = next((int(i) for i in order[1:] if labels[int(i)] != best_label), None)
    if other is None:
        raise ValueError("Template bank lacks other classes")
    second = float(distances[other])
    if second == best:
        return "UNRESOLVED", best, second, best_label
    return best_label, best, second, best_label

def run(target: list[dict], bank: np.ndarray, labels: list[str], mode: str) -> list[dict]:
    result = []
    for row in target:
        pred, distance, runner_up, nearest_label = predict(bank, labels, pixels(row), mode)
        result.append({"ground_truth": row["ground_truth"], "prediction": pred,
                       "distance": distance, "runner_up_other_class_distance": runner_up,
                       "nearest_label": nearest_label, "tile_png": row["tile_png"],
                       "tile_sha256": row["tile_sha256"], "source_video": row["source_video"],
                       "method": mode})
    return result

def metrics(records: list[dict]) -> dict:
    n = len(records)
    correct = sum(row["prediction"] == row["ground_truth"] for row in records)
    unresolved = sum(row["prediction"] == "UNRESOLVED" for row in records)
    rank = sum(row["prediction"] != "UNRESOLVED" and row["prediction"][0] == row["ground_truth"][0] for row in records)
    suit = sum(row["prediction"] != "UNRESOLVED" and row["prediction"][1] == row["ground_truth"][1] for row in records)
    by_suit = {s: {"test": sum(x["ground_truth"][1] == s for x in records),
                   "correct_suit": sum(x["ground_truth"][1] == s and x["prediction"] != "UNRESOLVED" and x["prediction"][1] == s for x in records)} for s in "cdhs"}
    return {"total": n, "correct": correct, "wrong": n-correct-unresolved, "unresolved": unresolved,
            "exact_accuracy_all": correct/n, "rank_accuracy_all": rank/n, "suit_accuracy_all": suit/n,
            "by_gt_suit": by_suit}

def write_json(path: Path, value):
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"Frozen output differs: {path}")
    path.write_bytes(payload)

def write_jsonl(path: Path, value):
    payload = ("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in value) + "\n").encode("utf-8")
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"Frozen output differs: {path}")
    path.write_bytes(payload)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("validation", "smoke16_exploratory"), required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    train_path = SPLIT / "train_manifest.jsonl"
    train = rows(train_path)
    labels = [row["ground_truth"] for row in train]
    if len(train) != 431 or len(set(labels)) != 52:
        raise ValueError("Training template bank changed")
    bank = np.stack([pixels(row) for row in train])
    if args.phase == "validation":
        target_path = SPLIT / "validation_manifest.jsonl"
        target = rows(target_path)
        evaluations = {}
        for mode in METHODS:
            prediction = run(target, bank, labels, mode)
            evaluation = metrics(prediction)
            evaluations[mode] = evaluation
            write_jsonl(OUT / f"validation_{mode}_predictions.jsonl", prediction)
        selected = max(METHODS, key=lambda m: (evaluations[m]["exact_accuracy_all"],
                                                evaluations[m]["suit_accuracy_all"], -METHODS.index(m)))
        config = {"status": "SELECTED_ON_VALIDATION_ONLY", "source": "VERIFIED_TRAIN_CHAT_CROPS",
                  "train_manifest_sha256": sha(train_path.read_bytes()),
                  "validation_manifest_sha256": sha(target_path.read_bytes()),
                  "train_tiles": len(train), "validation_tiles": len(target),
                  "candidate_methods": list(METHODS), "validation_results": evaluations,
                  "selected_method": selected,
                  "normalization": "grayscale first 20 columns, interior y=2:14 x=1:19; no Smoke16 pixels or labels used for selection",
                  "test_status": "POST_BASELINE_EXPLORATORY_ONLY"}
        write_json(OUT / "validation_selection.json", config)
        print(json.dumps({"selected": selected, "validation": evaluations}, ensure_ascii=False))
    else:
        config = json.loads((OUT / "validation_selection.json").read_text(encoding="utf-8"))
        if config["train_manifest_sha256"] != sha(train_path.read_bytes()) or config["validation_manifest_sha256"] != sha((SPLIT / "validation_manifest.jsonl").read_bytes()):
            raise ValueError("Template selection inputs changed")
        if not (DATA / "smoke16_blind_prediction_receipt.json").exists():
            raise ValueError("First blind baseline must be frozen before exploratory comparison")
        target_path = DATA / "smoke16_test_manifest.jsonl"
        target = rows(target_path)
        pred = run(target, bank, labels, config["selected_method"])
        score = metrics(pred)
        score.update({"status": "POST_BASELINE_EXPLORATORY_NOT_BLIND", "selected_method": config["selected_method"],
                      "train_manifest_sha256": config["train_manifest_sha256"],
                      "test_manifest_sha256": sha(target_path.read_bytes())})
        write_jsonl(OUT / "smoke16_exploratory_predictions.jsonl", pred)
        write_json(OUT / "smoke16_exploratory_metrics.json", score)
        print(json.dumps(score, ensure_ascii=False))

if __name__ == "__main__":
    main()
