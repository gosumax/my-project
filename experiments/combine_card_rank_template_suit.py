"""Post-baseline hybrid: frozen VL rank plus validation-selected template suit."""
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
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]

def write_once(path, payload):
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"Output differs: {path}")
    path.write_bytes(payload)

def main():
    baseline_path = DATA / "smoke16_blind_predictions.jsonl"
    template_path = OUT / "smoke16_exploratory_predictions.jsonl"
    receipt = json.loads((DATA / "smoke16_blind_prediction_receipt.json").read_text(encoding="utf-8"))
    if sha(baseline_path.read_bytes()) != receipt["predictions_sha256"]:
        raise ValueError("First blind baseline changed")
    selection = json.loads((OUT / "validation_selection.json").read_text(encoding="utf-8"))
    baseline, template = read(baseline_path), read(template_path)
    if len(baseline) != 225 or len(template) != 225:
        raise ValueError("Expected paired 225 predictions")
    combined = []
    for i, (vl, cmp) in enumerate(zip(baseline, template)):
        if vl["tile_sha256"] != cmp["tile_sha256"] or vl["ground_truth"] != cmp["ground_truth"]:
            raise ValueError(f"Prediction alignment failed: {i}")
        rank = vl["prediction"][0] if vl["prediction"] != "UNRESOLVED" else None
        suit = cmp["prediction"][1] if cmp["prediction"] != "UNRESOLVED" else None
        pred = rank + suit if rank and suit else "UNRESOLVED"
        verdict = "UNRESOLVED" if pred == "UNRESOLVED" else "CORRECT" if pred == vl["ground_truth"] else "WRONG"
        combined.append({"prediction_index": i, "ground_truth": vl["ground_truth"],
                         "vl_prediction": vl["prediction"], "template_prediction": cmp["prediction"],
                         "hybrid_prediction": pred, "verdict": verdict,
                         "tile_sha256": vl["tile_sha256"], "tile_png": vl["tile_png"],
                         "source_video": vl["source_video"], "frame_index_by_pts_512": vl["frame_index_by_pts_512"],
                         "table_id": vl["table_id"], "source_span_index": vl["source_span_index"]})
    payload = ("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in combined) + "\n").encode("utf-8")
    write_once(OUT / "hybrid_smoke16_exploratory_predictions.jsonl", payload)
    counts = collections.Counter(x["verdict"] for x in combined)
    n = len(combined)
    conf = collections.Counter((x["ground_truth"][1], x["hybrid_prediction"][1] if x["hybrid_prediction"] != "UNRESOLVED" else "U") for x in combined)
    summary = {"status": "POST_BASELINE_EXPLORATORY_NOT_BLIND", "total": n,
               "correct": counts["CORRECT"], "wrong": counts["WRONG"], "unresolved": counts["UNRESOLVED"],
               "exact_accuracy_all": counts["CORRECT"]/n,
               "rank_accuracy_all": sum(x["hybrid_prediction"] != "UNRESOLVED" and x["hybrid_prediction"][0] == x["ground_truth"][0] for x in combined)/n,
               "suit_accuracy_all": sum(x["hybrid_prediction"] != "UNRESOLVED" and x["hybrid_prediction"][1] == x["ground_truth"][1] for x in combined)/n,
               "suit_confusion": {f"{a}->{b}": value for (a,b), value in sorted(conf.items())},
               "baseline_predictions_sha256": receipt["predictions_sha256"],
               "template_selection_method": selection["selected_method"],
               "template_train_manifest_sha256": selection["train_manifest_sha256"],
               "hybrid_predictions_sha256": sha(payload),
               "interpretation": "Diagnostic reuse of already revealed Smoke16; cannot be called independent accuracy."}
    write_once(OUT / "hybrid_smoke16_exploratory_metrics.json", (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
