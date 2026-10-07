"""Select a VL-rank / image-template-suit hybrid using VALIDATION only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs/card_train_smoke16_test_20260930"
OUT = DATA / "stage2_template"
METHODS = ("raw_equal", "raw_suit2", "binary_equal", "binary_suit2")

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

def combine(vl: list[dict], template: list[dict]):
    if len(vl) != len(template):
        raise ValueError("Prediction count differs")
    joined = []
    for index, (left, right) in enumerate(zip(vl, template)):
        if left["tile_sha256"] != right["tile_sha256"] or left["ground_truth"] != right["ground_truth"]:
            raise ValueError(f"Prediction order/provenance differs: {index}")
        rank = left["prediction"][0] if left["prediction"] != "UNRESOLVED" else None
        suit = right["prediction"][1] if right["prediction"] != "UNRESOLVED" else None
        prediction = rank + suit if rank and suit else "UNRESOLVED"
        joined.append({"tile_sha256": left["tile_sha256"], "ground_truth": left["ground_truth"],
                       "vl_prediction": left["prediction"], "template_prediction": right["prediction"],
                       "hybrid_prediction": prediction})
    return joined

def score(rows):
    n = len(rows)
    correct = sum(x["hybrid_prediction"] == x["ground_truth"] for x in rows)
    unresolved = sum(x["hybrid_prediction"] == "UNRESOLVED" for x in rows)
    rank = sum(x["hybrid_prediction"] != "UNRESOLVED" and x["hybrid_prediction"][0] == x["ground_truth"][0] for x in rows)
    suit = sum(x["hybrid_prediction"] != "UNRESOLVED" and x["hybrid_prediction"][1] == x["ground_truth"][1] for x in rows)
    return {"total": n, "correct": correct, "wrong": n-correct-unresolved,
            "unresolved": unresolved, "exact_accuracy_all": correct/n,
            "rank_accuracy_all": rank/n, "suit_accuracy_all": suit/n}

def write_once(path: Path, value):
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"Frozen output differs: {path}")
    path.write_bytes(payload)

def main():
    vl_path = DATA / "postbaseline_validation_predictions.jsonl"
    validation_path = DATA / "split_v2/validation_manifest.jsonl"
    vl = read(vl_path)
    validation = read(validation_path)
    if len(vl) != 318 or len(validation) != len(vl):
        raise ValueError("Full VALIDATION predictions required")
    if [x["tile_sha256"] for x in vl] != [x["tile_sha256"] for x in validation]:
        raise ValueError("VALIDATION order differs")
    results = {}
    template_hashes = {}
    for method in METHODS:
        path = OUT / f"validation_{method}_predictions.jsonl"
        template_hashes[method] = sha(path.read_bytes())
        results[method] = score(combine(vl, read(path)))
    selected = max(METHODS, key=lambda name: (results[name]["correct"],
                                              -results[name]["unresolved"],
                                              -METHODS.index(name)))
    result = {"status": "POST_BASELINE_HYBRID_SELECTED_ON_VALIDATION_ONLY",
              "selected_method": selected, "selection_rule": "maximum exact hybrid accuracy, then fewer unresolved, then fixed method order",
              "validation_manifest_sha256": sha(validation_path.read_bytes()),
              "vl_validation_predictions_sha256": sha(vl_path.read_bytes()),
              "template_validation_predictions_sha256": template_hashes,
              "methods": results, "smoke16_used_for_selection": False,
              "smoke16_evaluation_status": "POST_BASELINE_EXPLORATORY_ONLY"}
    write_once(OUT / "hybrid_validation_selection.json", result)
    print(json.dumps({"selected_method": selected, "methods": results}, ensure_ascii=False))

if __name__ == "__main__":
    main()
