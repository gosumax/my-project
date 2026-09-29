"""Build visual review sheets for every saved raw table and chat ROI."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    out = run / "roi_review"
    out.mkdir(exist_ok=True)
    with sqlite3.connect(run / "journal.sqlite3") as db:
        rows = db.execute("""SELECT o.frame_id,json_extract(s.geometry_json,'$.slot_id'),
            o.roi_type,o.bbox_json,o.crop_path,o.crop_sha256
            FROM observations o JOIN table_sessions s USING(table_session_id)
            WHERE o.roi_type IN ('table','chat')
            ORDER BY o.frame_id,json_extract(s.geometry_json,'$.slot_id'),o.roi_type""").fetchall()
    groups = defaultdict(dict)
    records = []
    for frame_id, slot_id, roi_type, box_json, relative, digest in rows:
        box = json.loads(box_json)
        path = (run / relative).resolve(strict=True)
        if not path.is_relative_to(run):
            raise ValueError(f"ROI path outside run: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"ROI SHA-256 mismatch: {path}")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        if image.size != (box[2]-box[0], box[3]-box[1]):
            raise ValueError(f"ROI dimensions mismatch: {path}")
        key = (frame_id, roi_type)
        if slot_id in groups[key]:
            raise ValueError(f"Duplicate ROI for frame {frame_id}, slot {slot_id}, {roi_type}")
        groups[key][slot_id] = image
        records.append({"frame_id": frame_id, "slot_id": slot_id,
                        "roi_type": roi_type, "bbox_source_xyxy": box,
                        "path": str(path), "sha256": digest})
    sheets = []
    for (frame_id, roi_type), slots in sorted(groups.items()):
        if set(slots) != set(range(1, 10)):
            raise ValueError(f"Expected nine ROIs in frame {frame_id}, {roi_type}")
        cell_width = max(image.width for image in slots.values())
        cell_height = max(image.height for image in slots.values()) + 24
        sheet = Image.new("RGB", (cell_width * 3, cell_height * 3), "white")
        draw = ImageDraw.Draw(sheet)
        for slot_id, image in sorted(slots.items()):
            x = ((slot_id - 1) % 3) * cell_width
            y = ((slot_id - 1) // 3) * cell_height
            draw.text((x + 4, y + 4), f"Frame {frame_id} / Table {slot_id:02d} / {roi_type}",
                      fill="black")
            sheet.paste(image, (x, y + 24))
        target = out / f"frame_{frame_id:09d}_{roi_type}_all9.png"
        sheet.save(target)
        sheets.append(str(target))
    review = {"status": "VISUAL_REVIEW_ONLY", "raw_roi_count": len(records),
              "raw_rois": records, "sheets": sheets}
    (out / "roi_review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(records)} raw ROIs; wrote {len(sheets)} contact sheets to {out}")


if __name__ == "__main__":
    main()
