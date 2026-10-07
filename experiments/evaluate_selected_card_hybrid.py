"""Apply validation-selected hybrid to already revealed Smoke16, once."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.compare_card_templates import pixels, predict, rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs/card_train_smoke16_test_20260930"
OUT = DATA / "stage2_template"

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def write_once(path: Path, data: bytes):
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"Frozen result differs: {path}")
    path.write_bytes(data)

def main():
    selection = json.loads((OUT / "hybrid_validation_selection.json").read_text(encoding="utf-8"))
    train_path = DATA / "split_v2/train_manifest.jsonl"
    valid_path = DATA / "split_v2/validation_manifest.jsonl"
    baseline_path = DATA / "smoke16_blind_predictions.jsonl"
    receipt = json.loads((DATA / "smoke16_blind_prediction_receipt.json").read_text(encoding="utf-8"))
    if selection["validation_manifest_sha256"] != sha(valid_path.read_bytes()) or sha(baseline_path.read_bytes()) != receipt["predictions_sha256"]:
        raise ValueError("Frozen validation or first blind baseline changed")
    if selection["smoke16_used_for_selection"] is not False:
        raise ValueError("Selection is not validation-only")
    method = selection["selected_method"]
    train = rows(train_path)
    test = rows(DATA / "smoke16_test_manifest.jsonl")
    vl = rows(baseline_path)
    if len(test) != len(vl) or len(train) != 431:
        raise ValueError("Manifest count differs")
    bank = np.stack([pixels(row) for row in train])
    labels = [row["ground_truth"] for row in train]
    output = []
    for index, (row, old) in enumerate(zip(test, vl)):
        if row["tile_sha256"] != old["tile_sha256"] or row["ground_truth"] != old["ground_truth"]:
            raise ValueError(f"TEST alignment differs at {index}")
        nearest, distance, second, nearest_label = predict(bank, labels, pixels(row), method)
        rank = old["prediction"][0] if old["prediction"] != "UNRESOLVED" else None
        suit = nearest[1] if nearest != "UNRESOLVED" else None
        result = rank + suit if rank and suit else "UNRESOLVED"
        verdict = "UNRESOLVED" if result == "UNRESOLVED" else "CORRECT" if result == row["ground_truth"] else "WRONG"
        output.append({"prediction_index": index, "ground_truth": row["ground_truth"],
                       "vl_prediction": old["prediction"], "template_nearest_label": nearest_label,
                       "template_prediction": nearest, "hybrid_prediction": result,
                       "verdict": verdict, "distance": distance, "runner_up_other_class_distance": second,
                       "tile_sha256": row["tile_sha256"], "tile_png": row["tile_png"],
                       "source_row_png": row["source_row_png"], "source_row_sha256": row["source_row_sha256"],
                       "blind_test_index": row["blind_test_index"], "card_index": row["card_index"],
                       "frame_index_by_pts_512": row["frame_index_by_pts_512"],
                       "table_id": row["table_id"], "source_span_index": row["source_span_index"]})
    payload = ("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in output) + "\n").encode("utf-8")
    write_once(OUT / "selected_hybrid_smoke16_diagnostic_predictions.jsonl", payload)
    counts = collections.Counter(x["verdict"] for x in output)
    result = {"status": "POST_BASELINE_DIAGNOSTIC_NOT_BLIND", "selection_method": method,
              "selection_source": "VALIDATION_ONLY", "test_tiles": len(output),
              "correct": counts["CORRECT"], "wrong": counts["WRONG"],
              "unresolved": counts["UNRESOLVED"],
              "exact_accuracy_all": counts["CORRECT"]/len(output),
              "first_blind_predictions_sha256": receipt["predictions_sha256"],
              "diagnostic_predictions_sha256": sha(payload)}
    write_once(OUT / "selected_hybrid_smoke16_diagnostic_metrics.json",
               (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
