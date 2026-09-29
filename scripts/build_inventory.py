"""Copy selected legacy assets and record their provenance without changing sources."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path(r"D:\Проэкты\Парсер пд v1.0\experiments\yolo_pd_eventlog_20260926")
CHAT = Path(r"D:\Проэкты\V2.0 PD\outputs\chat_only_inventory_smoke11_20260927")
ENTITIES = Path(r"D:\Проэкты\V2.0 PD\outputs\chat_entities_smoke11_20260927")
TESSERACT_DATA = Path(r"C:\Program Files\Tesseract-OCR\tessdata")

ASSETS = {
    "table_chat_detector": (LEGACY / "train_occupied_table/weights/seat_coverage.pt", "models/table_chat.pt"),
    "board_presence_detector": (LEGACY / "train_verified_board/weights/best.pt", "models/board_presence.pt"),
    "shown_pair_classifier": (LEGACY / "train_shown_state_v2/best.pt", "models/shown_pair.pt"),
    "tesseract_rus_language": (Path(r"D:\Проэкты\V2.0 PD\outputs\ocr_paddle_tesseract_10rows_20260927\tessdata\rus.traineddata"), "models/tessdata/rus.traineddata"),
    "tesseract_eng_language": (TESSERACT_DATA / "eng.traineddata", "models/tessdata/eng.traineddata"),
    "nine_table_detector": (Path(r"D:\Проэкты\V2.0 PD\assets\models\pokerdom_table_chat\best.pt"), "models/nine_table.pt"),
}
CONFIGS = {
    "nine_table_layout_profile": (Path(r"D:\Проэкты\V2.0 PD\config\proven\layout_847x404.json"),
                                  "configs/layout_847x404.json"),
    "nine_table_event_rois": (Path(r"D:\Проэкты\V2.0 PD\config\nine_table_event_rois.json"),
                              "configs/nine_table_event_rois.json"),
    "nine_table_split_layout": (Path(r"D:\Проэкты\V2.0 PD\config\proven\nine_table_layout.json"),
                                "configs/nine_table_layout.json"),
    "chat_log_bottom_calibration": (Path(r"D:\Проэкты\V2.0 PD\config\chat_log_bottom_rois.json"),
                                    "configs/chat_log_bottom_rois.json"),
}
CATALOGS = {
    "chat_event_types": CHAT / "chat_event_types.md",
    "chat_event_candidates": CHAT / "chat_event_candidates.csv",
    "all_chat_rows_audit": CHAT / "all_chat_rows_audit.csv",
    "unresolved_chat_views": CHAT / "unresolved_chat_views.csv",
    "chat_entity_types": ENTITIES / "chat_entity_types.md",
    "chat_entity_mentions": ENTITIES / "chat_entity_mentions.csv",
    "nickname_ocr_variants": ENTITIES / "nickname_ocr_variants.csv",
    "money_amounts": ENTITIES / "money_amounts.csv",
    "system_messages": ENTITIES / "system_messages.csv",
    "chat_entity_gaps": ENTITIES / "chat_entity_gaps.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    entries = []
    for name, (source, relative_destination) in {**ASSETS, **CONFIGS}.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = ROOT / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_hash = sha256(source)
        if destination.exists() and sha256(destination) != source_hash:
            raise RuntimeError(f"Existing asset has a different hash: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
        if sha256(destination) != source_hash:
            raise RuntimeError(f"Copy verification failed: {destination}")
        entries.append({"name": name,
                        "kind": "config" if name in CONFIGS else "model",
                        "source": str(source),
                        "local": relative_destination, "bytes": source.stat().st_size,
                        "sha256": source_hash})
    for name, source in CATALOGS.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        entries.append({"name": name, "kind": "reference_catalog", "source": str(source),
                        "local": None, "bytes": source.stat().st_size, "sha256": sha256(source)})
    manifest = {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
                "entries": entries}
    target = ROOT / "provenance" / "manifest.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(entries)} assets in {target}")


if __name__ == "__main__":
    main()
