"""Read a single CARD tile with the validation-selected CHAT-image matcher.

The optional rank must come from a separately validated recognizer. This tool
does not infer a rank from PokerDom card rules or neighboring cards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.compare_card_templates import predict, pixels, rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs/card_train_smoke16_test_20260930"
OUT = DATA / "stage2_template"

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tile_png", type=Path)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--rank", choices=list("23456789TJQKA"))
    args = ap.parse_args()
    selection = json.loads((OUT / "hybrid_validation_selection.json").read_text(encoding="utf-8"))
    train_path = DATA / "split_v2/train_manifest.jsonl"
    if sha(train_path.read_bytes()) != json.loads((OUT / "validation_selection.json").read_text(encoding="utf-8"))["train_manifest_sha256"]:
        raise ValueError("Template bank changed after validation selection")
    raw = args.tile_png.read_bytes()
    if sha(raw) != args.expected_sha256.lower():
        raise ValueError("Input tile SHA256 mismatch")
    with Image.open(args.tile_png) as opened:
        rgb = opened.convert("RGB")
        gray = np.asarray(rgb.convert("L"))
    if gray.shape[0] != 16 or gray.shape[1] not in (20, 21):
        result = {"status": "UNRESOLVED", "reason": "UNSUPPORTED_TILE_GEOMETRY",
                  "tile_png": str(args.tile_png.resolve()), "tile_sha256": sha(raw),
                  "size_wh": [rgb.width, rgb.height]}
        print(json.dumps(result, ensure_ascii=False))
        return
    train = rows(train_path)
    bank = np.stack([pixels(row) for row in train])
    labels = [row["ground_truth"] for row in train]
    method = selection["selected_method"]
    query = np.ascontiguousarray(gray[:, :20])
    nearest, best, second, nearest_label = predict(bank, labels, query, method)
    suit = nearest[1] if nearest != "UNRESOLVED" else None
    card = args.rank + suit if args.rank and suit else None
    result = {"status": "OK" if suit else "UNRESOLVED",
              "reason": None if suit else "EQUAL_BEST_DISTANCES",
              "suit": suit if suit else "UNRESOLVED", "rank_source": "CALLER" if args.rank else "UNRESOLVED",
              "card": card if card else "UNRESOLVED", "nearest_train_card": nearest_label,
              "best_distance": best, "runner_up_other_card_distance": second,
              "method": method, "method_selection_status": selection["status"],
              "tile_png": str(args.tile_png.resolve()), "tile_sha256": sha(raw),
              "tile_pixel_sha256": sha(rgb.tobytes()), "size_wh": [rgb.width, rgb.height],
              "train_manifest_sha256": sha(train_path.read_bytes()),
              "note": "Template distance is not a calibrated probability; no hand context used."}
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
