"""Capture every decoded frame and configured ROIs for one fixed-layout table.

This is an evidence-capture pilot. It does not infer chat messages or hands.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import av

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.journal import Journal, stable_id


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bbox(value: str) -> list[int]:
    result = [int(part) for part in value.split(",")]
    if len(result) != 4 or result[0] < 0 or result[1] < 0 or result[2] <= result[0] or result[3] <= result[1]:
        raise argparse.ArgumentTypeError("ROI must be x1,y1,x2,y2 in source pixels")
    return result


def save_crop(image, box: list[int], destination: Path) -> str:
    output = io.BytesIO()
    image.crop(tuple(box)).save(output, format="PNG")
    data = output.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != digest:
            raise RuntimeError(f"Existing crop differs: {destination}")
    else:
        temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
        temporary.write_bytes(data)
        temporary.replace(destination)
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--table-roi", type=bbox, required=True)
    parser.add_argument("--chat-roi", type=bbox, required=True)
    parser.add_argument("--layout-id", default="configured_fixed_roi_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chat-only", action="store_true",
                        help="Capture chat ROI only; table ROI remains session geometry")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Decode from source start, but capture only at or after this frame")
    parser.add_argument("--max-frames", type=int, help="Stop after N frames and mark source PARTIAL")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.start_frame < 0:
        parser.error("--start-frame must be nonnegative")
    source_path = args.video.resolve(strict=True)
    source_hash = sha256(source_path)
    source_id = stable_id("source", source_hash, source_path.stat().st_size)
    session_id = stable_id("table", source_id, args.layout_id, args.table_roi, args.chat_roi)
    output = args.output.resolve()
    journal = Journal(output / "journal.sqlite3")
    journal.add_source(source_id, str(source_path), source_hash, source_path.stat().st_size)
    journal.add_session(session_id, source_id, args.layout_id,
                        {"table_roi": args.table_roi, "chat_roi": args.chat_roi})
    count = 0
    limited = False
    final_state = "FAILED"
    active_rois = (("chat", args.chat_roi),) if args.chat_only else (
        ("table", args.table_roi), ("chat", args.chat_roi))
    try:
        with av.open(str(source_path)) as container:
            stream = container.streams.video[0]
            for frame_id, frame in enumerate(container.decode(stream)):
                if frame_id < args.start_frame:
                    continue
                if args.max_frames is not None and count >= args.max_frames:
                    limited = True
                    break
                width, height = frame.width, frame.height
                for name, box in active_rois:
                    if box[2] > width or box[3] > height:
                        raise ValueError(f"{name} ROI {box} exceeds frame {width}x{height}")
                time_base = frame.time_base or stream.time_base
                journal.add_frame(source_id, frame_id, frame.pts,
                                  time_base.numerator if frame.pts is not None else None,
                                  time_base.denominator if frame.pts is not None else None,
                                  width, height)
                image = frame.to_image()
                for name, box in active_rois:
                    obs_id = stable_id("obs", source_id, frame_id, session_id, name, box)
                    relative = Path("crops") / session_id / f"{frame_id:09d}_{name}.png"
                    crop_hash = save_crop(image, box, output / relative)
                    journal.add_observation(obs_id, source_id, frame_id, session_id, name, box,
                                            str(relative), crop_hash, "RAW")
                count += 1
        if count == 0:
            raise ValueError("Selected source window contains no frames")
        final_state = "PARTIAL" if limited or args.start_frame > 0 else "EOF"
        journal.set_source_state(source_id, final_state)
    except Exception:
        journal.set_source_state(source_id, "FAILED")
        raise
    finally:
        summary = {"source_id": source_id, "table_session_id": session_id,
                   "start_frame": args.start_frame, "frames_this_run": count,
                   "state": final_state,
                   "counts": journal.counts(), "interpretation": "RAW_ONLY",
                   "capture_mode": "CHAT_ONLY" if args.chat_only else "TABLE_AND_CHAT"}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        journal.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
