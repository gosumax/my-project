"""Append user visual adjudication without altering frozen GT or predictions."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs/card_train_smoke16_test_20260930"
OUT = DATA / "stage2_template"

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

def write_once(path, payload):
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"Adjudication artifact differs: {path}")
    path.write_bytes(payload)

def main():
    test_path = DATA / "smoke16_test_manifest.jsonl"
    old_path = DATA / "smoke16_blind_predictions.jsonl"
    hybrid_path = OUT / "selected_hybrid_smoke16_diagnostic_predictions.jsonl"
    test, old, hybrid = read(test_path), read(old_path), read(hybrid_path)
    if len(test) != len(old) or len(test) != len(hybrid) or len(test) != 225:
        raise ValueError("Prediction manifests differ")
    decisions = [
        {"review_label": "A", "blind_test_index": 9, "card_index": 3,
         "user_visual_reading": "Qc", "decision": "CORRECT_GT_POST_BASELINE"},
        {"review_label": "B", "blind_test_index": 34, "card_index": 0,
         "user_visual_reading": "Qd", "decision": "CONFIRM_GT_POST_BASELINE"},
    ]
    for decision in decisions:
        matches = [x for x in test if x["blind_test_index"] == decision["blind_test_index"] and x["card_index"] == decision["card_index"]]
        if len(matches) != 1:
            raise ValueError("User-reviewed tile identity differs")
        decision["tile_sha256"] = matches[0]["tile_sha256"]
        decision["original_ground_truth"] = matches[0]["ground_truth"]
        decision["source_row_sha256"] = matches[0]["source_row_sha256"]
        decision["provenance"] = "USER_CHAT_REPLY_A_Qc_B_Qd_AFTER_FIRST_BLIND_PREDICTIONS"
    if decisions[0]["original_ground_truth"] != "Tc" or decisions[1]["original_ground_truth"] != "Qd":
        raise ValueError("Original frozen labels differ")
    correction = {decisions[0]["tile_sha256"]: "Qc"}
    for i, (row, baseline, selected) in enumerate(zip(test, old, hybrid)):
        if row["tile_sha256"] != baseline["tile_sha256"] or row["tile_sha256"] != selected["tile_sha256"]:
            raise ValueError(f"Prediction alignment differs: {i}")
    metadata = {"status": "USER_VISUAL_ADJUDICATION_AFTER_FIRST_BLIND_RUN",
                "test_manifest_sha256": sha(test_path.read_bytes()),
                "first_blind_predictions_sha256": sha(old_path.read_bytes()),
                "selected_hybrid_predictions_sha256": sha(hybrid_path.read_bytes()),
                "decisions": decisions,
                "original_artifacts_untouched": True}
    write_once(OUT / "user_adjudication_A_B.json", (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    def metric(records, prediction_field):
        n = len(records)
        counts = collections.Counter()
        rank = suit = 0
        changed = []
        for row in records:
            gt = correction.get(row["tile_sha256"], row["ground_truth"])
            pred = row[prediction_field]
            verdict = "UNRESOLVED" if pred == "UNRESOLVED" else "CORRECT" if pred == gt else "WRONG"
            counts[verdict] += 1
            if pred != "UNRESOLVED":
                rank += pred[0] == gt[0]
                suit += pred[1] == gt[1]
            if gt != row["ground_truth"]:
                changed.append({"tile_sha256": row["tile_sha256"], "original_gt": row["ground_truth"],
                                "adjudicated_gt": gt, "prediction": pred, "adjudicated_verdict": verdict})
        return {"correct": counts["CORRECT"], "wrong": counts["WRONG"], "unresolved": counts["UNRESOLVED"],
                "exact_accuracy_all": counts["CORRECT"]/n, "rank_accuracy_all": rank/n,
                "suit_accuracy_all": suit/n, "changed_tiles": changed}
    result = {"status": "POST_BASELINE_USER_ADJUDICATED_METRICS_NOT_NEW_BLIND_TEST",
              "total": 225, "correction_count": 1, "confirmation_count": 1,
              "first_blind_predictions_against_adjudicated_gt": metric(old, "prediction"),
              "validation_selected_hybrid_against_adjudicated_gt": metric(hybrid, "hybrid_prediction"),
              "original_prediction_and_gt_files_untouched": True,
              "adjudication_sha256": sha((OUT / "user_adjudication_A_B.json").read_bytes())}
    write_once(OUT / "user_adjudicated_metrics.json", (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    classes = [rank + suit for rank in "23456789TJQKA" for suit in "cdhs"]
    for name, records, field in (("first_blind", old, "prediction"),
                                 ("selected_hybrid", hybrid, "hybrid_prediction")):
        matrix = {gt: {pred: 0 for pred in classes + ["UNRESOLVED"]} for gt in classes}
        per_class = {gt: {"test": 0, "correct": 0, "wrong": 0, "unresolved": 0} for gt in classes}
        errors = []
        for row in records:
            gt = correction.get(row["tile_sha256"], row["ground_truth"])
            pred = row[field]
            if gt not in classes or pred not in classes + ["UNRESOLVED"]:
                raise ValueError("Unexpected adjudicated class or prediction")
            verdict = "unresolved" if pred == "UNRESOLVED" else "correct" if pred == gt else "wrong"
            matrix[gt][pred] += 1
            per_class[gt]["test"] += 1
            per_class[gt][verdict] += 1
            if verdict != "correct":
                errors.append({"ground_truth_adjudicated": gt, "prediction": pred,
                               "tile_sha256": row["tile_sha256"],
                               "blind_test_index": row["blind_test_index"],
                               "card_index": row["card_index"], "verdict": verdict.upper()})
        prefix = OUT / f"user_adjudicated_{name}"
        write_once(prefix.with_name(prefix.name + "_confusion_matrix.json"),
                   (json.dumps(matrix, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        write_once(prefix.with_name(prefix.name + "_per_class.json"),
                   (json.dumps(per_class, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        write_once(prefix.with_name(prefix.name + "_errors.jsonl"),
                   ("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in errors) + "\n").encode("utf-8"))
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
