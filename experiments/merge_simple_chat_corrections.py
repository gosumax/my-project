"""Validate the downloaded correction sheet and merge it with prior decisions."""

import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/paddle_vl15_poker_sft_20260930"
CARD = re.compile(r"^[2-9TJQKA][cdhs]$")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_gt(gt):
    cards = gt.split(" ")
    if not 1 <= len(cards) <= 5 or cards != gt.split() or any(not CARD.fullmatch(c) for c in cards) or len(set(cards)) != len(cards):
        raise ValueError(f"Invalid card sequence: {gt!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrections", type=Path, required=True)
    args = ap.parse_args()
    manifest_path = OUT / "candidates_manifest.jsonl"
    manifest_sha = sha(manifest_path.read_bytes())
    manifest = {r["candidate_id"]: r for r in (json.loads(s) for s in manifest_path.read_text(encoding="utf-8").splitlines() if s)}
    with (OUT / "chat_card_correction_table.csv").open(encoding="utf-8-sig", newline="") as stream:
        audit = {r["candidate_id"]: r for r in csv.DictReader(stream, delimiter=";")}
    expected = {k for k, r in audit.items() if r["audit_result"] in
                {"USER_GT_DISAGREES_WITH_VISUAL", "PROVISIONAL_VISUAL_DRAFT"}}
    payload = json.loads(args.corrections.read_text(encoding="utf-8"))
    if payload.get("schema") != "poker.chat-card-corrections.v1" or payload.get("manifest_sha256") != manifest_sha:
        raise ValueError("Downloaded corrections have wrong schema or manifest")
    incoming = payload.get("rows")
    if not isinstance(incoming, list) or len(incoming) != len(expected):
        raise ValueError("Downloaded corrections are incomplete")
    if len({r["candidate_id"] for r in incoming}) != len(incoming) or {r["candidate_id"] for r in incoming} != expected:
        raise ValueError("Downloaded corrections contain missing or duplicate candidate IDs")
    original = json.loads((OUT / "review_progress_user_20260930.json").read_text(encoding="utf-8"))
    if original["manifest_sha256"] != manifest_sha:
        raise ValueError("Original user progress has wrong manifest")
    decisions = dict(original["decisions"])
    changed_from_draft = 0
    for row in incoming:
        candidate_id, gt = row["candidate_id"], row["ground_truth"]
        validate_gt(gt)
        source = manifest[candidate_id]
        raw = Path(source["artifact_path"]).read_bytes()
        if sha(raw) != source["image_sha256"]:
            raise ValueError(f"Source PNG changed: {candidate_id}")
        with Image.open(io.BytesIO(raw)) as opened:
            rgb = opened.convert("RGB")
        derived = rgb.crop((0, 0, rgb.width - 12, rgb.height))
        saved_path = OUT / "production_trim12_chat_crops" / f"{candidate_id}.png"
        with Image.open(saved_path) as opened:
            saved = opened.convert("RGB")
        if saved.size != derived.size or saved.tobytes() != derived.tobytes() or sha(saved_path.read_bytes()) != row["derived_image_sha256"]:
            raise ValueError(f"Displayed/derived crop changed: {candidate_id}")
        if gt != audit[candidate_id]["visual_gt"]:
            changed_from_draft += 1
        existing = decisions.get(candidate_id)
        decisions[candidate_id] = {
            "status": "VERIFIED", "verified_gt": gt, "kind": "CARD_SEQUENCE",
            "prior_kind": existing.get("kind") if existing else None,
            "reviewed_at": payload["exported_at"],
            "review_source": "USER_CONFIRMED_SIMPLE_VISUAL_REVIEW",
            "image_sha256": source["image_sha256"],
            "derived_image_sha256": row["derived_image_sha256"],
        }
    merged = {"schema": "poker.chat-card-review.v1", "manifest_sha256": manifest_sha,
              "merged_at": payload["exported_at"],
              "source_original_sha256": sha((OUT / "review_progress_user_20260930.json").read_bytes()),
              "source_corrections_sha256": sha(args.corrections.read_bytes()),
              "decisions": decisions}
    output = OUT / "review_progress_merged.json"
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "review_progress.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"correction_rows": len(incoming), "changed_from_visual_draft": changed_from_draft,
               "verified": sum(d["status"] == "VERIFIED" for d in decisions.values()),
               "rejected": sum(d["status"] == "REJECTED" for d in decisions.values()),
               "merged_progress": str(output), "source_corrections_sha256": merged["source_corrections_sha256"]}
    (OUT / "simple_review_merge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
