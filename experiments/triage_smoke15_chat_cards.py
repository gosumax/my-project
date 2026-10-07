"""Rank Smoke15 CHAT card tiles by visual similarity to verified tiles.

Nearest-template suggestions are for review priority only, never ground truth.
"""

import hashlib
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.build_verified_chat_card_tiles import OUT as DATASET, spans_like_production


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "runs/smoke15_chat_card_sample_v2_20260930"
RARE = {"6d", "2d", "5s", "Jd", "3d", "4d", "7d", "8c", "Td", "Jh"}


def read(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]


def checked_image(path, expected_sha):
    image_path = Path(path)
    if hashlib.sha256(image_path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError(f"Image SHA mismatch: {image_path}")
    return Image.open(image_path).convert("RGB")


def normal(image):
    return np.asarray(image.convert("L").resize((22, 16)), dtype=np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, default=SAMPLE)
    args = ap.parse_args()
    sample = args.sample
    templates = read(DATASET / "verified_chat_card_tiles.jsonl")
    x = np.stack([normal(checked_image(r["tile_png"], r["tile_sha256"])) for r in templates])
    labels = [r["ground_truth"] for r in templates]
    candidates = read(sample / "candidates.jsonl")
    rows = []
    for parent in candidates:
        gray = np.asarray(checked_image(parent["chat_row_png"], parent["chat_row_sha256"]).convert("L"))[:, :-12]
        spans = spans_like_production(gray)
        if [list(s) for s in spans] != parent["detected_card_spans_inclusive"]:
            raise ValueError("Stored Smoke15 spans changed")
        suggestions = []
        for index, (a, b) in enumerate(spans):
            tile = normal(Image.fromarray(gray[:, a:b+1]))
            distances = np.mean(np.abs(x - tile), axis=(1, 2))
            nearest_by_class = {}
            for label, distance in zip(labels, distances):
                nearest_by_class[label] = min(nearest_by_class.get(label, float("inf")), float(distance))
            best = sorted(nearest_by_class.items(), key=lambda item: item[1])[:3]
            suggestions.append({"index": index, "box_xyxy_exclusive": [a, 0, b+1, 16],
                                "nearest_classes": [{"card": card, "mae": round(score, 3)} for card, score in best]})
        rows.append({"chat_row_png": parent["chat_row_png"],
                     "chat_row_sha256": parent["chat_row_sha256"],
                     "sample_index": parent["sample_index"], "decoded_time_s": parent["decoded_time_s"],
                     "source_pts": parent["source_pts"], "table_id": parent["table_id"],
                     "source_span_index": parent["source_span_index"],
                     "suggestions_not_gt": suggestions,
                     "rare_best": [s for s in suggestions if s["nearest_classes"][0]["card"] in RARE],
                     "gt_status": "UNREVIEWED"})
    rows.sort(key=lambda r: (not bool(r["rare_best"]), r["decoded_time_s"], r["table_id"], r["source_span_index"]))
    with (sample / "visual_triage_not_gt.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = Counter(s["nearest_classes"][0]["card"] for r in rows for s in r["suggestions_not_gt"])
    summary = {"status": "VISUAL_SIMILARITY_HINTS_NOT_GT", "candidate_rows": len(rows),
               "best_match_class_counts": dict(sorted(counts.items())),
               "rare_best_match_counts": {c: counts[c] for c in sorted(RARE)},
               "note": "No labels accepted; source-linked crops require independent visual review."}
    (sample / "triage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "best_match_class_counts"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
