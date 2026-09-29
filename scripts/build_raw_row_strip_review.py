"""Show physical chat-row strips from immutable raw chat PNGs, without OCR."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.annotation import sha256, write_json
from parser_core.chat_rows import (ROW_TRACKER_VERSION, exclude_first_visible_span,
                                   padded_row_span, physical_spans)


def build(run: Path, output: Path, selected: list[int]) -> dict:
    run = run.resolve(strict=True)
    output = output.resolve()
    if output.exists() or not selected or len(selected) != len(set(selected)):
        raise ValueError("Choose a new output directory and unique frame IDs")
    with sqlite3.connect((run / "journal.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute("""SELECT o.frame_id,json_extract(s.geometry_json,'$.slot_id'),
            o.bbox_json,o.crop_path,o.crop_sha256,f.source_pts,f.time_base_num,f.time_base_den
            FROM observations o JOIN table_sessions s USING(table_session_id)
            JOIN frames f USING(source_id,frame_id)
            WHERE o.roi_type='chat' AND o.frame_id IN (""" +
            ",".join("?" for _ in selected) +
            ") ORDER BY o.frame_id,json_extract(s.geometry_json,'$.slot_id')", selected).fetchall()
    by_frame: dict[int, dict[int, Image.Image]] = {}
    evidence = []
    suppressions = []
    output.mkdir(parents=True)
    (output / "strips").mkdir()
    for frame_id, slot, bbox_json, relative, digest, pts, num, den in rows:
        source = (run / relative).resolve(strict=True)
        if not source.is_relative_to(run) or sha256(source) != digest:
            raise ValueError(f"Raw chat PNG changed: {source}")
        with Image.open(source) as image:
            gray = np.asarray(image.convert("L"))
        spans = physical_spans(gray)
        included, suppressed = exclude_first_visible_span(spans)
        source_box = json.loads(bbox_json)
        for top, bottom in suppressed:
            suppressions.append({"frame_id": frame_id, "slot_id": slot,
                "bbox_chat_xyxy": [0, top, gray.shape[1], bottom],
                "bbox_source_xyxy": [source_box[0], source_box[1] + top,
                                     source_box[2], source_box[1] + bottom],
                "reason": "FIRST_VISIBLE_ROW_EXCLUDED", "chat_sha256": digest})
        strips = []
        for index, _ in enumerate(included, start=len(suppressed)):
            top, bottom = padded_row_span(spans, index, gray.shape[0])
            strip = Image.fromarray(gray[top:bottom].copy())
            path = output / "strips" / f"frame_{frame_id:09d}_table_{slot:02d}_row_{index:02d}.png"
            strip.save(path)
            record = {"frame_id": frame_id, "slot_id": slot, "source_span_index": index,
                      "chat_png": str(source), "chat_sha256": digest,
                      "source_pts": pts, "time_base": [num, den],
                      "bbox_chat_xyxy": [0, top, gray.shape[1], bottom],
                      "bbox_source_xyxy": [source_box[0], source_box[1] + top,
                                           source_box[2], source_box[1] + bottom],
                      "strip_png": str(path), "strip_sha256": sha256(path)}
            evidence.append(record)
            strips.append((index, strip))
        cell_width = gray.shape[1] * 2 + 80
        cell_height = max(50, 34 + len(strips) * 39)
        sheet = Image.new("RGB", (cell_width, cell_height), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((5, 5), f"Frame {frame_id} / Table {slot:02d} / {len(strips)} strips", fill="black")
        for row_number, (index, strip) in enumerate(strips):
            y = 30 + row_number * 39
            draw.text((5, y + 7), f"{index:02d}", fill="black")
            sheet.paste(strip.resize((strip.width * 2, strip.height * 2),
                                     Image.Resampling.NEAREST).convert("RGB"), (38, y))
        by_frame.setdefault(frame_id, {})[slot] = sheet
    contacts = []
    for frame_id in selected:
        slots = by_frame.get(frame_id, {})
        if set(slots) != set(range(1, 10)):
            raise ValueError(f"Frame {frame_id} lacks one or more of nine chats")
        width = max(x.width for x in slots.values())
        height = max(x.height for x in slots.values())
        all9 = Image.new("RGB", (width * 3, height * 3), "#dddddd")
        for slot, sheet in slots.items():
            all9.paste(sheet, (((slot - 1) % 3) * width, ((slot - 1) // 3) * height))
        path = output / f"frame_{frame_id:09d}_all9_row_strips.png"
        all9.save(path)
        contacts.append(str(path))
    report = {"status": "ROW_STRIPS_VISUAL_REVIEW_ONLY", "tracker_version": ROW_TRACKER_VERSION,
              "first_visible_row": "EXCLUDED_FROM_STRIPS", "source_run": str(run),
              "selected_frames": selected, "row_roi_count": len(evidence),
              "row_rois": evidence, "row_suppressions": suppressions,
              "all9_sheets": contacts}
    write_json(output / "row_strip_review.json", report)
    return {"frames": len(selected), "row_rois": len(evidence), "contacts": contacts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--frame", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run, args.output, args.frame), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
