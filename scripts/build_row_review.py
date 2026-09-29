"""Show every saved CHAT_ROW strip, grouped by source frame and table slot."""

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
    out = run / "row_review"
    out.mkdir(exist_ok=True)
    with sqlite3.connect(run / "journal.sqlite3") as db:
        rows = db.execute("""SELECT o.frame_id,
            json_extract(s.geometry_json,'$.slot_id'),o.bbox_json,
            o.crop_path,o.crop_sha256,o.observation_id,rd.decision,o.quality_json
            FROM observations o JOIN table_sessions s USING(table_session_id)
            LEFT JOIN row_decisions rd ON rd.observation_id=o.observation_id
            WHERE o.roi_type='CHAT_ROW'
            ORDER BY o.frame_id,json_extract(s.geometry_json,'$.slot_id'),
                     json_extract(o.bbox_json,'$[1]')""").fetchall()
    groups = defaultdict(list)
    records = []
    for frame_id, slot_id, box_json, relative, digest, obs_id, decision, quality_json in rows:
        path = (run / relative).resolve(strict=True)
        if not path.is_relative_to(run):
            raise ValueError(f"Row path outside run: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Row SHA-256 mismatch: {path}")
        box = json.loads(box_json)
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        if image.size != (box[2]-box[0], box[3]-box[1]):
            raise ValueError(f"Row dimensions mismatch: {path}")
        quality = json.loads(quality_json)
        source_index = quality.get("source_span_index")
        if source_index is None:
            source_index = int(path.stem.rsplit("_", 1)[-1])
        groups[(frame_id, slot_id)].append((source_index, image, box, decision))
        records.append({"frame_id": frame_id, "slot_id": slot_id,
                        "bbox_source_xyxy": box, "path": str(path),
                        "sha256": digest, "observation_id": obs_id,
                        "tracking_decision": decision,
                        "source_span_index": source_index})

    by_frame = defaultdict(dict)
    individual = []
    for (frame_id, slot_id), strips in sorted(groups.items()):
        width = max(image.width for _, image, _, _ in strips) + 70
        height = 30 + sum(max(20, image.height) + 5 for _, image, _, _ in strips)
        sheet = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 5), f"Frame {frame_id} / Table {slot_id:02d} / {len(strips)} row ROIs", fill="black")
        y = 30
        for source_index, image, box, decision in strips:
            draw.text((4, y + 2), f"{source_index:02d}", fill="black")
            sheet.paste(image, (35, y))
            draw.rectangle((34, y-1, 35+image.width, y+image.height),
                           outline="gray", width=1)
            y += max(20, image.height) + 5
        target = out / f"frame_{frame_id:09d}_table_{slot_id:02d}_rows.png"
        sheet.save(target)
        by_frame[frame_id][slot_id] = sheet
        individual.append(str(target))

    contact = []
    for frame_id, tables in sorted(by_frame.items()):
        if set(tables) != set(range(1, 10)):
            raise ValueError(f"Frame {frame_id} does not have all nine table row sheets")
        cell_width = max(sheet.width for sheet in tables.values())
        cell_height = max(sheet.height for sheet in tables.values())
        sheet = Image.new("RGB", (cell_width * 3, cell_height * 3), "#dddddd")
        for slot_id, image in sorted(tables.items()):
            x = ((slot_id-1) % 3) * cell_width
            y = ((slot_id-1) // 3) * cell_height
            sheet.paste(image, (x, y))
        target = out / f"frame_{frame_id:09d}_all9_row_strips.png"
        sheet.save(target)
        contact.append(str(target))
    report = {"status": "ROW_STRIP_VISUAL_REVIEW", "row_roi_count": len(records),
              "row_rois": records, "all9_sheets": contact,
              "per_table_sheets": individual}
    (out / "row_review.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(records)} row ROIs; wrote {len(contact)} all-table sheets "
          f"and {len(individual)} per-table sheets to {out}")


if __name__ == "__main__":
    main()
