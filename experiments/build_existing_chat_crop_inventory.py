"""Summarize saved CHAT/row image sets from the three audited projects."""

import collections
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/paddle_vl15_poker_sft_20260930"
V2_OUTPUTS = Path(r"D:\Проэкты\V2.0 PD\outputs")
V1_PACKAGE = Path(r"D:\Проэкты\Парсер пд v1.0\baseline\smoke_04\equivalence\hand_0002\request_package")


def images(path):
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})


def distinct_png_files(files):
    return len({hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sets = json.loads((OUT / "current_crop_inventory.json").read_text(encoding="utf-8"))
    for row in sets:
        row.setdefault("source_video_sha256", None)
        row.setdefault("source_project", "Новый парсер")
    # The canonical production card-row selector is an index over saved rows,
    # not an additional independent video source.
    visual = ROOT / "runs/smoke14_parallel_vl_full_20260929/visual_evidence.jsonl"
    card_rows = [json.loads(line) for line in visual.read_text(encoding="utf-8").splitlines()
                 if json.loads(line).get("crop_type") == "CHAT_ROW_CARDS" and json.loads(line).get("crop_path")]
    sets.append({"source_video": "smoke_14.mp4", "source_video_sha256": None,
                 "source_project": "Новый парсер", "artifact_path": str(visual),
                 "artifact_type": "CARD_ROW", "image_count": len(card_rows),
                 "unique_hash_count": len({r["crop_sha256"] for r in card_rows}),
                 "unique_hash_basis": "recorded PNG SHA; each file checked by candidate builder",
                 "has_frame_id": True, "has_table_id": True, "has_row_id": True,
                 "has_event_type": True, "has_gt": False, "has_ocr_text": False,
                 "has_existing_prediction": True, "usable_for_board": True,
                 "usable_for_card": True,
                 "notes": "Production CHAT_ROW_CARDS selector; overlaps Smoke14 journal rows; 20 entries without saved crop excluded"})

    for directory in sorted(V2_OUTPUTS.glob("nine_table_eventlog_smoke11*")):
        crop_dir = directory / "crops"
        if not crop_dir.is_dir():
            continue
        found = images(crop_dir)
        if not found:
            continue
        capture = directory / "capture_manifest.json"
        meta = json.loads(capture.read_text(encoding="utf-8")) if capture.is_file() else {}
        sets.append({"source_video": "smoke_11.mp4", "source_video_sha256": meta.get("video_sha256"),
                     "source_project": "V2.0 PD", "artifact_path": str(crop_dir),
                     "artifact_type": "CHAT_ROW", "image_count": len(found),
                     "unique_hash_count": distinct_png_files(found),
                     "unique_hash_basis": "computed PNG byte SHA", "has_frame_id": True,
                     "has_table_id": True, "has_row_id": True, "has_event_type": (directory / "recognized.json").is_file(),
                     "has_gt": False, "has_ocr_text": (directory / "recognized.json").is_file(),
                     "has_existing_prediction": (directory / "recognized.json").is_file(),
                     "usable_for_board": True, "usable_for_card": True,
                     "notes": "Legacy row geometry; no exact production 224/226/228/230x16 match. Smoke11 variants overlap; counts are not additive."})

    v2_index = V2_OUTPUTS / "chat_only_inventory_smoke11_20260927/chat_event_candidates.csv"
    with v2_index.open(encoding="utf-8-sig", newline="") as stream:
        indexed = list(csv.DictReader(stream))
    sets.append({"source_video": "smoke_11.mp4", "source_video_sha256": None,
                 "source_project": "V2.0 PD", "artifact_path": str(v2_index),
                 "artifact_type": "EVENT_CANDIDATE", "image_count": len(indexed),
                 "unique_hash_count": len({r["crop_sha256"] for r in indexed}),
                 "unique_hash_basis": "recorded BGR pixel SHA", "has_frame_id": True,
                 "has_table_id": True, "has_row_id": True, "has_event_type": True,
                 "has_gt": False, "has_ocr_text": True, "has_existing_prediction": True,
                 "usable_for_board": True, "usable_for_card": True,
                 "notes": "Index into Smoke11 final crops; 373 narrow card-related rows selected, not additional images."})

    chat_image = V1_PACKAGE / "02_chat_baseline_pixels.png"
    if chat_image.is_file():
        sets.append({"source_video": "smoke_04.mp4", "source_video_sha256": None,
                     "source_project": "Парсер пд v1.0", "artifact_path": str(chat_image),
                     "artifact_type": "LONGCHAT", "image_count": 1, "unique_hash_count": 1,
                     "unique_hash_basis": "one saved PNG", "has_frame_id": False,
                     "has_table_id": False, "has_row_id": False, "has_event_type": False,
                     "has_gt": False, "has_ocr_text": False, "has_existing_prediction": False,
                     "usable_for_board": False, "usable_for_card": False,
                     "notes": "One-hand composite chat evidence, not a production-sized physical row."})

    keys = ["source_video", "source_video_sha256", "source_project", "artifact_path", "artifact_type",
            "image_count", "unique_hash_count", "unique_hash_basis", "has_frame_id", "has_table_id",
            "has_row_id", "has_event_type", "has_gt", "has_ocr_text", "has_existing_prediction",
            "usable_for_board", "usable_for_card", "notes"]
    result = {"schema": "poker.existing-chat-crop-inventory.v1", "scope": "Saved CHAT/row artifacts only; variant counts overlap",
              "sets": sets, "candidate_summary": json.loads((OUT / "candidate_summary.json").read_text(encoding="utf-8"))}
    (OUT / "existing_crop_inventory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "existing_crop_inventory.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(sets)
    print(json.dumps({"sets": len(sets), "by_project": dict(collections.Counter(r["source_project"] for r in sets)),
                      "card_related_candidates": result["candidate_summary"]["candidate_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
