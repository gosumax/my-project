"""Compare CHAT suit pixels with PokerDom's installed micro suit sprite atlas."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs/card_train_smoke16_test_20260930"
OUT = DATA / "stage2_template"
ATLAS = Path(r"C:\Program Files (x86)\Pokerdom.com\commonTG\data\images\cards\deck\move\suit_0micro.png")
SUITS = "cdhs"
SIZES = ((5, 7), (6, 8), (7, 9), (8, 10))
THRESHOLDS = (120, 160, 200, 230)

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def read(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

def targets(rows):
    result = []
    for row in rows:
        path = Path(row["tile_png"])
        if sha(path.read_bytes()) != row["tile_sha256"]:
            raise ValueError(f"Tile SHA changed: {path}")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if sha(rgb.tobytes()) != row["tile_pixel_sha256"]:
                raise ValueError(f"Tile pixels changed: {path}")
            gray = np.asarray(rgb.convert("L"))
        if gray.shape[0] != 16 or gray.shape[1] not in (20, 21):
            raise ValueError(f"Unexpected tile size: {path}")
        result.append(gray[3:13, 10:19])
    return np.stack(result)

def masks(atlas, half: str, size: tuple[int, int]):
    x0 = 0 if half == "small" else 10
    width, height = size
    templates = []
    owners = []
    for row, suit in enumerate(SUITS):
        alpha = atlas[row*14+1:row*14+14, x0:x0+10, 3]
        yy, xx = np.where(alpha > 64)
        if not len(xx):
            raise ValueError(f"Missing sprite for {suit}")
        cropped = alpha[yy.min():yy.max()+1, xx.min():xx.max()+1]
        resized = Image.fromarray(cropped).resize((width, height), Image.Resampling.BILINEAR)
        shape = np.asarray(resized) > 64
        for y in range(10-height+1):
            for x in range(9-width+1):
                template = np.zeros((10, 9), dtype=np.uint8)
                template[y:y+height, x:x+width] = shape
                templates.append(template)
                owners.append(suit)
    return np.stack(templates), np.array(owners)

def classify(patches, sprites, owners, threshold):
    target = (patches < threshold).astype(np.uint8)
    dist = np.count_nonzero(target[:, None] != sprites[None], axis=(2, 3))
    suit_scores = np.stack([dist[:, owners == suit].min(axis=1) for suit in SUITS], axis=1)
    best = suit_scores.argmin(axis=1)
    tied = (suit_scores == suit_scores.min(axis=1, keepdims=True)).sum(axis=1) > 1
    return ["UNRESOLVED" if tied[i] else SUITS[int(best[i])] for i in range(len(best))], suit_scores

def write_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"Frozen artifact differs: {path}")
    path.write_bytes(data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("validation", "smoke16_exploratory"), required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    atlas_bytes = ATLAS.read_bytes()
    atlas = np.asarray(Image.open(ATLAS).convert("RGBA"))
    if atlas.shape != (406, 20, 4):
        raise ValueError("PokerDom micro suit atlas geometry changed")
    if args.phase == "validation":
        target_path = DATA / "split_v2/validation_manifest.jsonl"
        records = read(target_path)
        patch = targets(records)
        results = []
        for half in ("small", "large"):
            for size in SIZES:
                sprite, owners = masks(atlas, half, size)
                for threshold in THRESHOLDS:
                    pred, _ = classify(patch, sprite, owners, threshold)
                    correct = sum(x["ground_truth"][1] == p for x, p in zip(records, pred))
                    unresolved = sum(p == "UNRESOLVED" for p in pred)
                    results.append({"half": half, "size_wh": list(size), "target_threshold": threshold,
                                    "correct_suit": correct, "unresolved": unresolved,
                                    "suit_accuracy_all": correct/len(records)})
        chosen = max(results, key=lambda x: (x["correct_suit"], -x["unresolved"],
                                             -SIZES.index(tuple(x["size_wh"])),
                                             x["half"] == "large"))
        summary = {"status": "INSTALLED_ATLAS_SUIT_RULE_SELECTED_ON_VALIDATION",
                   "atlas_path": str(ATLAS), "atlas_sha256": sha(atlas_bytes),
                   "validation_manifest_sha256": sha(target_path.read_bytes()),
                   "validation_tiles": len(records), "configurations": results,
                   "selected": chosen, "test_status": "POST_BASELINE_EXPLORATORY_ONLY"}
        write_json(OUT / "installed_atlas_validation_selection.json", summary)
        print(json.dumps({"selected": chosen, "validation_tiles": len(records)}, ensure_ascii=False))
    else:
        summary = json.loads((OUT / "installed_atlas_validation_selection.json").read_text(encoding="utf-8"))
        if sha(atlas_bytes) != summary["atlas_sha256"] or sha((DATA / "split_v2/validation_manifest.jsonl").read_bytes()) != summary["validation_manifest_sha256"]:
            raise ValueError("Atlas or validation selection input changed")
        if not (DATA / "smoke16_blind_prediction_receipt.json").exists():
            raise ValueError("First blind baseline not frozen")
        records = read(DATA / "smoke16_test_manifest.jsonl")
        chosen = summary["selected"]
        sprite, owners = masks(atlas, chosen["half"], tuple(chosen["size_wh"]))
        pred, scores = classify(targets(records), sprite, owners, chosen["target_threshold"])
        output = [{"tile_sha256": row["tile_sha256"], "ground_truth": row["ground_truth"],
                   "suit_prediction": p, "suit_scores_c_d_h_s": score.tolist()}
                  for row, p, score in zip(records, pred, scores)]
        payload = ("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in output) + "\n").encode("utf-8")
        destination = OUT / "installed_atlas_smoke16_exploratory_suits.jsonl"
        if destination.exists() and destination.read_bytes() != payload:
            raise ValueError("Exploratory atlas predictions differ")
        destination.write_bytes(payload)
        correct = sum(row["ground_truth"][1] == p for row, p in zip(records, pred))
        result = {"status": "POST_BASELINE_EXPLORATORY_NOT_BLIND", "test_tiles": len(records),
                  "correct_suit": correct, "suit_accuracy_all": correct/len(records),
                  "unresolved": sum(p == "UNRESOLVED" for p in pred),
                  "predictions_sha256": sha(payload), "selected": chosen,
                  "atlas_sha256": summary["atlas_sha256"]}
        write_json(OUT / "installed_atlas_smoke16_exploratory_metrics.json", result)
        print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
