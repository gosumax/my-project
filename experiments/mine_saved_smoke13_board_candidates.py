"""Extract unreviewed BOARD candidates from saved Smoke13 table PNGs.

No video decoding, OCR labeling, or production writes are performed.
"""

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.visual_crops import crop_saved_table  # noqa: E402


DEFAULT_SOURCE = Path(r"C:\ParserData\runs\smoke13_full_streaming_ssd_20260929_v1")
DEFAULT_OUT = ROOT / "runs/paddle_vl15_poker_sft_20260930/smoke13_candidates"
PROFILE = ROOT / "configs/layout_847x404.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--offsets", default="0,10,30")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    offsets = tuple(int(s) for s in args.offsets.split(","))
    if not offsets or any(x < 0 or x > 60 for x in offsets):
        raise ValueError("Offsets must be 0..60 frames")
    with sqlite3.connect(source / "journal.sqlite3") as db:
        sessions = {sid: json.loads(geometry)["slot_id"] for sid, geometry in
                    db.execute("SELECT table_session_id, geometry_json FROM table_sessions")}
        available = set(db.execute("SELECT table_session_id, frame_id FROM observations WHERE roi_type='table'").fetchall())
    with (source / "events_diagnostics.csv").open(encoding="utf-8-sig", newline="") as stream:
        events = [r for r in csv.DictReader(stream) if r["event_type"] in {"FLOP", "TURN", "RIVER"}]
    output.mkdir(parents=True, exist_ok=True)
    records, missing = [], []
    used = set()
    for event in events:
        session = event["table_session_id"]
        frame = int(event["order_key"].split(":", 1)[0])
        street = event["event_type"]
        for offset in offsets:
            candidate_frame = frame + offset
            key = (session, candidate_frame, street)
            if key in used:
                continue
            used.add(key)
            if (session, candidate_frame) not in available:
                missing.append({"event_id": event["event_id"], "frame_id": candidate_frame})
                continue
            target = output / "board" / f"T{sessions[session]:02d}_{candidate_frame:08d}_{street}.png"
            item = crop_saved_table(source, PROFILE, target,
                                    session_id=session, table_id=sessions[session],
                                    frame_id=candidate_frame, kind="BOARD", street=street)
            with Image.open(target) as opened:
                gray = opened.convert("L").resize((9, 8))
                values = list(gray.get_flattened_data())
            item["dhash64"] = f"{sum((values[y * 9 + x] > values[y * 9 + x + 1]) << (y * 8 + x) for y in range(8) for x in range(8)):016x}"
            item.update({"source_event_id": event["event_id"],
                         "source_event_status": event["status"], "frame_offset": offset,
                         "ground_truth": None, "ground_truth_status": "UNREVIEWED",
                         "split": None})
            records.append(item)
    summary = {
        "status": "UNREVIEWED_CANDIDATES_ONLY", "source": str(source),
        "event_count": len(events), "crop_count": len(records),
        "unique_png_sha256": len({r["crop_sha256"] for r in records}),
        "unique_rgb_sha256": len({r["board_sha"] for r in records}),
        "unique_dhash64": len({r["dhash64"] for r in records}),
        "street": dict(Counter(r["street"] for r in records)),
        "missing_frame_count": len(missing), "offsets": offsets,
    }
    (output / "candidate_manifest.json").write_text(json.dumps({"summary": summary, "records": records,
                                                                  "missing": missing}, ensure_ascii=False, indent=2) + "\n",
                                                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
