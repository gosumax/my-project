"""Consolidate saved CHAT-row card candidates; never assign visual ground truth."""

import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from inventory_saved_card_rows import OUT, ROOT, file_hash, perceptual_hash, proposed


CURRENT = OUT / "current_card_row_candidates.jsonl"
VISUAL = ROOT / "runs/smoke14_parallel_vl_full_20260929/visual_evidence.jsonl"
V2_ROWS = Path(r"D:\Проэкты\V2.0 PD\outputs\chat_only_inventory_smoke11_20260927\chat_event_candidates.csv")
V2_ENTITIES = Path(r"D:\Проэкты\V2.0 PD\outputs\chat_entities_smoke11_20260927\chat_entity_mentions.csv")
V2_CAPTURE = Path(r"D:\Проэкты\V2.0 PD\outputs\nine_table_eventlog_smoke11_final_20260927\capture_manifest.json")
CARD_EVENTS = {"FLOP", "TURN", "RIVER", "SHOW", "COMBINATION_FRAGMENT"}
MISSING = {"2c", "2h", "6d", "7s", "8c", "Ad", "Ah", "As", "Kd", "Qh", "Tc"}
RAW_SIZES = {(224, 16), (226, 16), (228, 16), (230, 16)}


def common(path, expected, *, hash_mode="png"):
    if not path.is_file():
        raise ValueError(f"Missing or changed source image: {path}")
    png_hash = file_hash(path)
    if hash_mode == "bgr":
        with Image.open(path) as opened:
            actual = hashlib.sha256(np.asarray(opened.convert("RGB"))[:, :, ::-1].tobytes()).hexdigest()
    else:
        actual = png_hash
    if actual != expected:
        raise ValueError(f"Source image hash mismatch: {path}")
    phash, width, height = perceptual_hash(path)
    return {"artifact_path": str(path), "image_sha256": png_hash,
            "source_recorded_hash": expected, "source_hash_mode": hash_mode,
            "perceptual_hash": phash, "width": width, "height": height,
            "production_raw_geometry": (width, height) in RAW_SIZES,
            "production_model_input_size": [width - 12, height] if (width, height) in RAW_SIZES else None}


def rank(record):
    hint = record.get("proposed_gt")
    cards = set(hint.split()) if hint else set()
    score = 4 if len(cards) in {3, 4, 5} else 2 if len(cards) in {1, 2} else 0
    if record.get("event_type") in {"FLOP", "TURN", "RIVER", "SHOW"}:
        score += 2
    if record.get("artifact_type") == "CHAT_ROW_CARDS":
        score += 2
    if cards & MISSING:
        score += 2
    if record["production_raw_geometry"]:
        score += 2
    return score


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    current = [json.loads(line) for line in CURRENT.read_text(encoding="utf-8").splitlines() if line]
    smoke14_sha = next(r["source_video_sha256"] for r in current if r["source_video"] == "smoke_14.mp4")
    records = []
    # Current Smoke12/13 CHAT_ROW images. Smoke14 uses its canonical card-row selector below.
    for row in current:
        if row["source_video"] == "smoke_14.mp4":
            continue
        if not (row.get("proposed_gt") or row.get("event_type")):
            continue
        path = Path(row["artifact_path"])
        row.update(common(path, row["image_sha256"]))
        row["gt_status"] = "UNREVIEWED"
        row["verified_gt"] = None
        records.append(row)

    # Production VisualTileQueue already chose these card-containing row images.
    for line in VISUAL.read_text(encoding="utf-8").splitlines():
        evidence = json.loads(line)
        if evidence.get("crop_type") != "CHAT_ROW_CARDS" or not evidence.get("crop_path"):
            continue
        path = Path(evidence["crop_path"])
        hint = proposed(evidence.get("vl_raw_text") or "")
        record = {
            "candidate_id": "cand_" + hashlib.sha256((smoke14_sha + evidence["row_key"]).encode()).hexdigest()[:24],
            "source_video": "smoke_14.mp4", "source_video_sha256": smoke14_sha,
            "source_project": "Новый парсер", "source_run": str(VISUAL.parent),
            "artifact_type": "CHAT_ROW_CARDS", "frame_id": evidence["frame_id"],
            "table_id": evidence["table_id"], "table_session_id": evidence["table_session_id"],
            "row_id": evidence["row_key"], "hand_id": evidence.get("hand_id"),
            "event_type": evidence.get("type"), "street": evidence.get("street"),
            "ocr_text": None, "existing_prediction": evidence.get("vl_raw_text"),
            "proposed_gt": hint, "proposed_gt_source": "VL_UNVERIFIED" if hint else None,
            "gt_status": "UNREVIEWED", "verified_gt": None,
        }
        record.update(common(path, evidence["crop_sha256"]))
        records.append(record)

    # V2 Smoke11 legacy rows: useful for review; different crop geometry is explicit.
    video_sha = json.loads(V2_CAPTURE.read_text(encoding="utf-8"))["video_sha256"]
    with V2_ENTITIES.open(encoding="utf-8-sig", newline="") as stream:
        card_row_ids = {r["row_id"] for r in csv.DictReader(stream) if r["entity_type"] == "CARD_TEXT_CANDIDATE"}
    with V2_ROWS.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["event_type"] not in CARD_EVENTS and row["row_id"] not in card_row_ids:
                continue
            path = Path(row["source_crops"])
            if not path.is_file():
                continue
            hint = proposed(row["raw_ocr"])
            record = {
                "candidate_id": "cand_" + hashlib.sha256((video_sha + row["row_id"]).encode()).hexdigest()[:24],
                "source_video": "smoke_11.mp4", "source_video_sha256": video_sha,
                "source_project": "V2.0 PD", "source_run": str(V2_ROWS.parent),
                "artifact_type": "CHAT_ROW", "frame_id": int(row["frame"]),
                "table_id": int(row["table"]), "table_session_id": None,
                "row_id": row["row_id"], "hand_id": None,
                "event_type": row["event_type"],
                "street": row["event_type"] if row["event_type"] in {"FLOP", "TURN", "RIVER"} else None,
                "ocr_text": row["raw_ocr"], "existing_prediction": None,
                "proposed_gt": hint, "proposed_gt_source": "OCR_UNVERIFIED" if hint else None,
                "gt_status": "UNREVIEWED", "verified_gt": None,
            }
            record.update(common(path, row["crop_sha256"], hash_mode="bgr"))
            records.append(record)

    for record in records:
        record["review_priority"] = rank(record)
        record["training_eligible"] = False  # Requires visual review, grouping, and split.
    # Duplicate pixels can occur at different frames; retain every occurrence for lineage.
    records.sort(key=lambda r: (-r["review_priority"], r["source_video"], r["table_id"] or 0,
                                r["frame_id"], r["candidate_id"]))
    ids = [r["candidate_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate_id")
    with (OUT / "candidates_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "status": "UNREVIEWED_CHAT_ONLY", "candidate_count": len(records),
        "unique_image_sha256": len({(r["source_video_sha256"], r["image_sha256"]) for r in records}),
        "by_source": dict(collections.Counter(r["source_video"] for r in records)),
        "production_raw_geometry": sum(r["production_raw_geometry"] for r in records),
        "nonproduction_raw_geometry": sum(not r["production_raw_geometry"] for r in records),
        "ocr_or_vl_gt_hints": sum(bool(r["proposed_gt"]) for r in records),
        "verified_gt": 0,
        "raw_sizes": {f"{w}x{h}": n for (w, h), n in sorted(collections.Counter((r["width"], r["height"]) for r in records).items())},
    }
    (OUT / "candidate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hinted = [card for r in records if r["proposed_gt"] for card in r["proposed_gt"].split()]
    distribution = {
        "status": "UNVERIFIED_HINT_DISTRIBUTION_NOT_GT", "candidates": len(records),
        "by_source_video": summary["by_source"],
        "by_event_type": dict(collections.Counter(r["event_type"] or "UNKNOWN" for r in records)),
        "by_street": dict(collections.Counter(r["street"] or "UNKNOWN" for r in records)),
        "by_table": dict(collections.Counter(f'{r["source_video"]}:T{r["table_id"]}' for r in records)),
        "by_raw_size": summary["raw_sizes"],
        "hint_rank": dict(sorted(collections.Counter(c[0] for c in hinted).items())),
        "hint_suit": dict(sorted(collections.Counter(c[1] for c in hinted).items())),
        "hint_card_class": dict(sorted(collections.Counter(hinted).items())),
        "priority_missing_class_hint_count": {c: hinted.count(c) for c in sorted(MISSING)},
        "verified_board_count": 0, "verified_card_count": 0,
    }
    (OUT / "chat_candidate_distribution.json").write_text(json.dumps(distribution, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
