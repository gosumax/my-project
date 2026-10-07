"""Join visually read CHAT card sequences with the frozen candidate inventory."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/paddle_vl15_poker_sft_20260930"
PROGRESS = OUT / "review_progress_user_20260930.json"

# Two disagreements recorded during a separate visual pass over all 75 user rows.
USER_VISUAL_CORRECTIONS = {
    29: "Jh Kh 3s Ah",
    57: "Kc 9s 7s 4c Js",
}
REJECTED_VISUAL_READS = {
    59: "As Js 9s 8s 7s",
    62: "Th Jd Qd Kd Ac",
    69: "8s 8c Ah Qh 4s",
}


def main():
    manifest_path = OUT / "candidates_manifest.jsonl"
    manifest = {r["candidate_id"]: r for r in (json.loads(s) for s in manifest_path.read_text(encoding="utf-8").splitlines() if s)}
    progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
    if progress["manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("User export is for another candidate manifest")
    user_index = json.loads((OUT / "sheets_user/index.json").read_text(encoding="utf-8"))
    candidate_index = json.loads((OUT / "sheets_candidates/index.json").read_text(encoding="utf-8"))
    labels = {}
    for line in (OUT / "visual_labels_150.txt").read_text(encoding="utf-8").splitlines():
        number, gt = line.split("|", 1)
        labels[int(number)] = gt
    if set(labels) != set(range(1, 151)) or len(candidate_index) != 150 or len(user_index) != 75:
        raise ValueError("Visual sheet index and transcription must cover 75 + 150 rows")
    rows = []
    for origin, entries in (("USER_REVIEW_AUDIT", user_index), ("ASSISTANT_VISUAL_DRAFT", candidate_index)):
        for entry in entries:
            number, candidate_id = entry["number"], entry["candidate_id"]
            source = manifest[candidate_id]
            decision = progress["decisions"].get(candidate_id) if origin == "USER_REVIEW_AUDIT" else None
            if decision and decision["image_sha256"] != source["image_sha256"]:
                raise ValueError(f"Image SHA mismatch for {candidate_id}")
            user_gt = decision.get("verified_gt", "") if decision else ""
            if origin == "USER_REVIEW_AUDIT":
                visual_gt = USER_VISUAL_CORRECTIONS.get(number, REJECTED_VISUAL_READS.get(number, user_gt))
            else:
                visual_gt = labels[number]
            cards = visual_gt.split(" ")
            if not 1 <= len(cards) <= 5 or len(set(cards)) != len(cards):
                raise ValueError(f"Invalid visual sequence for {candidate_id}: {visual_gt}")
            if decision and decision["status"] == "REJECTED":
                result = "REJECTED_READABLE_DUPLICATE"
            elif decision and user_gt != visual_gt:
                result = "USER_GT_DISAGREES_WITH_VISUAL"
            elif decision:
                result = "USER_GT_VISUAL_MATCH"
            else:
                result = "PROVISIONAL_VISUAL_DRAFT"
            rows.append({
                "candidate_id": candidate_id, "review_origin": origin, "review_number": number,
                "audit_result": result, "visual_gt": visual_gt, "user_gt": user_gt,
                "user_status": decision["status"] if decision else "UNREVIEWED",
                "user_kind_original": decision.get("kind", "") if decision else "",
                "event_type": source.get("event_type") or "UNKNOWN",
                "street": source.get("street") or "UNKNOWN",
                "source_video": source["source_video"], "frame_id": source["frame_id"],
                "table_id": source["table_id"], "image_sha256": source["image_sha256"],
                "source_png": source["artifact_path"], "raw_size": f"{source['width']}x{source['height']}",
                "trimmed_png": str(OUT / "production_trim12_chat_crops" / f"{candidate_id}.png") if source["production_raw_geometry"] else "",
                "production_trim12_size": "x".join(map(str, source["production_model_input_size"])),
                "production_geometry": source["production_raw_geometry"],
                "ocr_hint_unverified": source.get("proposed_gt") or "",
                "sheet": f"{origin}:{entry['sheet']}:{number:03}",
            })
    target = OUT / "chat_card_visual_audit.csv"
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    order = {"USER_GT_DISAGREES_WITH_VISUAL": 0, "PROVISIONAL_VISUAL_DRAFT": 1,
             "REJECTED_READABLE_DUPLICATE": 2, "USER_GT_VISUAL_MATCH": 3}
    short_fields = ["audit_result", "review_number", "candidate_id", "source_video",
                    "frame_id", "table_id", "event_type", "user_gt", "visual_gt", "trimmed_png"]
    short_path = OUT / "chat_card_correction_table.csv"
    with short_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=short_fields, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (order[r["audit_result"]], r["review_origin"], r["review_number"])))
    summary = {
        "status": "PROVISIONAL_VISUAL_AUDIT_USER_CORRECTION_PENDING",
        "source_user_progress_sha256": hashlib.sha256(PROGRESS.read_bytes()).hexdigest(),
        "total_rows": len(rows), "user_decisions": len(user_index),
        "user_gt_visual_match": sum(r["audit_result"] == "USER_GT_VISUAL_MATCH" for r in rows),
        "user_gt_visual_disagreement": sum(r["audit_result"] == "USER_GT_DISAGREES_WITH_VISUAL" for r in rows),
        "rejected_readable_duplicate": sum(r["audit_result"] == "REJECTED_READABLE_DUPLICATE" for r in rows),
        "assistant_visual_drafts": len(candidate_index),
        "trim_right_px": 12,
        "ocr_hint_used_as_ground_truth": False,
    }
    (OUT / "chat_card_visual_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"table": str(target), "correction_table": str(short_path), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
