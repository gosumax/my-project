"""Validate exported human decisions against the frozen CHAT crop manifest."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/paddle_vl15_poker_sft_20260930"
CARD = re.compile(r"^[2-9TJQKA][cdhs]$")


def validate_gt(gt):
    cards = gt.split(" ")
    if gt != " ".join(cards) or any(not CARD.fullmatch(c) for c in cards) or len(set(cards)) != len(cards):
        raise ValueError("GT must be unique rank+suit cards separated by one space")
    n = len(cards)
    if n not in {1, 2, 3, 4, 5}:
        raise ValueError("Card sequence requires 1–5 cards")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=Path, default=OUT / "review_progress.json")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    manifest = OUT / "candidates_manifest.jsonl"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    records = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x]
    by_id = {r["candidate_id"]: r for r in records}
    progress = json.loads(args.progress.read_text(encoding="utf-8"))
    if progress.get("schema") != "poker.chat-card-review.v1" or progress.get("manifest_sha256") != digest:
        raise ValueError("Review progress does not match current frozen candidate manifest")
    audit_path = OUT / "chat_card_visual_audit.csv"
    with audit_path.open(encoding="utf-8-sig", newline="") as stream:
        audited = {r["candidate_id"]: r for r in csv.DictReader(stream, delimiter=";")}
    verified, rejected, disputed = [], [], []
    for candidate_id, decision in progress.get("decisions", {}).items():
        source = by_id.get(candidate_id)
        if source is None or decision.get("image_sha256") != source["image_sha256"]:
            raise ValueError(f"Unknown candidate or changed image: {candidate_id}")
        status = decision.get("status")
        if status == "REJECTED":
            rejected.append(candidate_id)
        elif status == "VERIFIED":
            gt = decision.get("verified_gt")
            validate_gt(gt)
            if candidate_id in audited and gt != audited[candidate_id]["visual_gt"]:
                disputed.append(candidate_id)
                continue
            image = Path(source["artifact_path"])
            if hashlib.sha256(image.read_bytes()).hexdigest() != source["image_sha256"]:
                raise ValueError(f"Image changed since review: {candidate_id}")
            verified.append({"candidate_id": candidate_id, "task_kind": "CHAT_CARD_SEQUENCE", "ground_truth": gt,
                             "source_user_kind": decision.get("kind"),
                             "gt_status": "VERIFIED", "human_reviewed_at": decision.get("reviewed_at"),
                             "source_video": source["source_video"], "source_video_sha256": source["source_video_sha256"],
                             "artifact_path": source["artifact_path"], "image_sha256": source["image_sha256"],
                             "frame_id": source["frame_id"], "table_id": source["table_id"],
                             "row_id": source.get("row_id"), "hand_id": source.get("hand_id"),
                             "event_type": source.get("event_type"), "street": source.get("street"),
                             "production_raw_geometry": source["production_raw_geometry"],
                             "production_model_input_size": source["production_model_input_size"]})
        elif status != "UNREVIEWED":
            raise ValueError(f"Invalid decision status: {candidate_id}")
    args.output.mkdir(parents=True, exist_ok=True)
    compatible = [r for r in verified if r["production_raw_geometry"]]
    with (args.output / "verified_chat_card_sequences.jsonl").open("w", encoding="utf-8") as stream:
        for row in compatible:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"status": "WAITING_FOR_VISUAL_REVIEW" if not verified else "REVIEW_PARTIAL",
               "decisions": len(progress.get("decisions", {})), "verified": len(verified),
               "verified_chat_card_sequences_production_geometry": len(compatible),
               "quarantined_visual_disagreement": len(disputed),
               "quarantined_candidate_ids": disputed,
               "verified_legacy_geometry_excluded": sum(not r["production_raw_geometry"] for r in verified),
               "rejected": len(rejected), "standalone_card_crops": 0,
               "manifest_sha256": digest}
    (args.output / "review_adjudication.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
