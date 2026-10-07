"""Audit saved BOARD evidence before any poker SFT split or training.

This deliberately does not promote diagnostic labels into an independent test.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/smoke14_parallel_vl_full_20260929/visual_evidence.jsonl"
SELECTED = ROOT / "runs/smoke14_board_vl_batch_benchmark_20260930_v2/dataset.json"
GT = ROOT / "runs/board_vl_quality_audit_20260930/ground_truth.json"
MODEL_CONFIG = ROOT / "configs/ocr_vl15_local.json"
OUT = ROOT / "runs/paddle_vl15_poker_sft_20260930"
RANKS = set("23456789TJQKA")
SUITS = set("cdhs")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dhash(path):
    with Image.open(path) as im:
        gray = im.convert("L").resize((9, 8))
        values = list(gray.get_flattened_data())
    return f"{sum((values[y * 9 + x] > values[y * 9 + x + 1]) << (y * 8 + x) for y in range(8) for x in range(8)):016x}"


def valid_cards(cards, street):
    expected = {"FLOP": 3, "TURN": 4, "RIVER": 5}[street]
    return (isinstance(cards, list) and len(cards) == expected
            and len(set(cards)) == expected
            and all(isinstance(c, str) and len(c) == 2 and c[0] in RANKS and c[1] in SUITS for c in cards))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    source = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line]
    boards = [r for r in source if r.get("crop_type") == "BOARD"]
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))["selected"]
    gt_doc = json.loads(GT.read_text(encoding="utf-8"))
    labels = gt_doc["cards_by_manifest_index"]
    if gt_doc["status"] != "VISUALLY_REVIEWED_DIAGNOSTIC" or len(selected) != len(labels):
        raise ValueError("Diagnostic ground truth and selection disagree")
    by_sha = {}
    for index, (entry, cards) in enumerate(zip(selected, labels, strict=True), 1):
        if not valid_cards(cards, entry["street"]):
            raise ValueError(f"Invalid label at manifest index {index}")
        by_sha[entry["sha256"]] = {"cards": cards, "manifest_index": index}

    records = []
    seen = set()
    for row in boards:
        digest = row["crop_sha256"]
        if digest in seen:
            continue
        seen.add(digest)
        path = Path(row["crop_path"])
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"Missing or changed crop: {path}")
        with Image.open(path) as im:
            size = list(im.size)
        label = by_sha.get(digest)
        records.append({
            "image": str(path), "png_sha256": digest,
            "dhash64": dhash(path), "width": size[0], "height": size[1],
            "table_id": row["table_id"], "table_session_id": row["table_session_id"],
            "frame_id": row["frame_id"], "street": row["street"],
            "hand_id": row.get("hand_id"), "source_video_sha256": row.get("source_video_sha256"),
            "source_row_key": row["row_key"],
            "ground_truth": " ".join(label["cards"]) if label else None,
            "ground_truth_status": "VISUALLY_REVIEWED_DIAGNOSTIC" if label else "UNREVIEWED",
            "diagnostic_manifest_index": label["manifest_index"] if label else None,
            "split": None,
        })
    if set(by_sha) - seen:
        raise ValueError("Some labeled crops are absent from source evidence")

    street = collections.Counter(r["street"] for r in records)
    tables = collections.Counter(str(r["table_id"]) for r in records)
    labeled = [r for r in records if r["ground_truth"]]
    cards = [card for r in labeled for card in r["ground_truth"].split()]
    distribution = {
        "candidate_board_crops": len(records), "visually_reviewed_diagnostic_boards": len(labeled),
        "independent_test_boards": 0, "train_boards": 0, "val_boards": 0,
        "card_crops": 0, "board_street": dict(sorted(street.items())),
        "board_table_id": dict(sorted(tables.items())),
        "diagnostic_rank": dict(sorted(collections.Counter(c[0] for c in cards).items())),
        "diagnostic_suit": dict(sorted(collections.Counter(c[1] for c in cards).items())),
        "diagnostic_card": dict(sorted(collections.Counter(cards).items())),
        "diagnostic_missing_classes": sorted(a + b for a in RANKS for b in SUITS if a + b not in cards),
        "size": dict(sorted(collections.Counter(f'{r["width"]}x{r["height"]}' for r in records).items())),
    }
    cfg = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    manifest = {
        "status": "AUDIT_ONLY_GT_AND_INDEPENDENT_TEST_INSUFFICIENT",
        "source": str(SOURCE), "source_sha256": sha(SOURCE),
        "diagnostic_selection": str(SELECTED), "diagnostic_selection_sha256": sha(SELECTED),
        "diagnostic_gt": str(GT), "diagnostic_gt_sha256": sha(GT),
        "base_model": {k: cfg[k] for k in ("model_dir", "model_revision", "weights_sha256")},
        "records": records,
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "distribution.json").write_text(json.dumps(distribution, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prompt = "Read the poker board. Return only cards left to right using rank+suit notation."
    with (out / "diagnostic_erniekit.jsonl").open("w", encoding="utf-8") as stream:
        for row in labeled:
            example = {"image_info": [{"matched_text_index": 0, "image_url": row["image"]}],
                       "text_info": [{"text": prompt, "tag": "mask"},
                                     {"text": row["ground_truth"], "tag": "no_mask"}]}
            stream.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), **distribution}, ensure_ascii=False))


if __name__ == "__main__":
    main()
