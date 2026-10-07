"""Validate a user-exported Smoke15 CHAT sequence review against frozen pixels."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.build_verified_chat_card_tiles import spans_like_production


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "runs/smoke15_chat_card_sample_5s_20260930"
CARD = re.compile(r"^[2-9TJQKA][cdhs]$")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--sample", type=Path, default=SAMPLE)
    args = ap.parse_args()
    sample = args.sample
    manifest = sample / "candidates.jsonl"
    review = json.loads(args.review.read_text(encoding="utf-8"))
    if (review.get("schema") != "poker.smoke15-chat-review.v1" or
            review.get("sample_manifest_sha256") != sha(manifest.read_bytes())):
        raise ValueError("Review schema or sample manifest SHA mismatch")
    candidates = read_jsonl(manifest)
    by_sha = {}
    for row in candidates:
        by_sha.setdefault(row["chat_row_sha256"], []).append(row)
    decisions = review.get("rows")
    if not isinstance(decisions, list) or len({r.get("chat_row_sha256") for r in decisions}) != len(decisions):
        raise ValueError("Review rows are missing or duplicate")
    verified = []
    for decision in decisions:
        digest = decision["chat_row_sha256"]
        occurrences = by_sha.get(digest)
        if not occurrences or decision.get("source_png") not in {r["chat_row_png"] for r in occurrences}:
            raise ValueError(f"Unknown source row: {digest}")
        gt = decision["ground_truth"]
        cards = gt.split(" ")
        if (not 1 <= len(cards) <= 5 or gt != " ".join(cards) or
                any(not CARD.fullmatch(c) for c in cards) or len(set(cards)) != len(cards)):
            raise ValueError(f"Invalid card sequence: {digest}")
        raw = Path(decision["source_png"]).read_bytes()
        if sha(raw) != digest:
            raise ValueError(f"Source PNG changed: {digest}")
        with Image.open(decision["source_png"]) as opened:
            image = opened.convert("RGB")
        if image.height != 16 or image.width not in (224, 226, 228, 230):
            raise ValueError(f"Nonproduction CHAT geometry: {digest}")
        gray = np.asarray(image.convert("L"))[:, :-12]
        spans = spans_like_production(gray)
        if any([list(x) for x in spans] != r["detected_card_spans_inclusive"] for r in occurrences):
            raise ValueError(f"Saved spans changed: {digest}")
        gap_ok = all(2 <= spans[i+1][0] - spans[i][1] - 1 <= 5 for i in range(len(spans)-1))
        geometry_status = "ACCEPTED" if len(cards) == len(spans) and gap_ok else "UNRESOLVED"
        verified.append({"task_kind": "CHAT_CARD_SEQUENCE", "gt_status": "USER_VERIFIED_SMOKE15",
                         "ground_truth": gt, "chat_row_sha256": digest,
                         "source_png": decision["source_png"], "source_video_sha256": occurrences[0]["source_video_sha256"],
                         "production_right_trim_px": 12,
                         "detected_card_spans_inclusive": spans,
                         "card_tile_geometry_status": geometry_status,
                         "card_tile_geometry_reason": None if geometry_status == "ACCEPTED" else "SPAN_COUNT_OR_GAP_MISMATCH",
                         "occurrences": [{k: r[k] for k in ("sample_index", "decoded_time_s", "source_pts",
                                                              "table_id", "source_span_index", "chat_row_png")}
                                         for r in occurrences],
                         "review_export_sha256": sha(args.review.read_bytes())})
    out = sample / "verified_smoke15_chat_card_sequences.jsonl"
    with out.open("w", encoding="utf-8") as stream:
        for row in verified:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"verified_unique_rows": len(verified), "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
