"""Read-only parity and speed probe for RAM-only chat change detection."""

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.layout import load_fixed_layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    layout = load_fixed_layout(root / "configs/nine_table_layout.json",
                               root / "configs/layout_847x404.json",
                               root / "configs/chat_log_bottom_rois.json")
    with sqlite3.connect(args.baseline / "journal.sqlite3") as db:
        expected = {(frame_id, session_id): json.loads(quality)["native_pixel_sha256"]
                    for frame_id, session_id, quality in db.execute(
                        "SELECT frame_id,table_session_id,quality_json FROM observations "
                        "WHERE roi_type='chat' AND frame_id<?", (args.frames,))}
    summary = {"frames": 0, "chats": 0, "changed": 0, "unchanged": 0,
               "hash_mismatches": 0, "conversion_wall_seconds": 0.0,
               "compare_wall_seconds": 0.0, "hash_wall_seconds": 0.0}
    last = {}
    started = time.perf_counter()
    sessions = json.loads((args.baseline / "summary.json").read_text("utf-8"))["table_sessions"]
    with av.open(str(args.video)) as video:
        for frame_id, frame in enumerate(video.decode(video.streams.video[0])):
            if frame_id >= args.frames:
                break
            tick = time.perf_counter()
            rgb = frame.to_ndarray(format="rgb24")
            summary["conversion_wall_seconds"] += time.perf_counter() - tick
            for slot in layout.slots:
                x1, y1, x2, y2 = slot.chat_box
                crop = np.ascontiguousarray(rgb[y1:y2, x1:x2])
                tick = time.perf_counter()
                unchanged = slot.slot_id in last and np.array_equal(crop, last[slot.slot_id])
                summary["compare_wall_seconds"] += time.perf_counter() - tick
                summary["unchanged" if unchanged else "changed"] += 1
                if not unchanged:
                    tick = time.perf_counter()
                    digest = hashlib.sha256(crop.tobytes()).hexdigest()
                    summary["hash_wall_seconds"] += time.perf_counter() - tick
                    last[slot.slot_id] = crop
                else:
                    digest = last[(slot.slot_id, "digest")]
                last[(slot.slot_id, "digest")] = digest
                if digest != expected[(frame_id, sessions[str(slot.slot_id)])]:
                    summary["hash_mismatches"] += 1
                summary["chats"] += 1
            summary["frames"] += 1
    summary["wall_seconds"] = time.perf_counter() - started
    summary["frames_per_second"] = summary["frames"] / summary["wall_seconds"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
