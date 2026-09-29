"""Freeze a stratified smoke11 development review queue with source hashes."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path(r"D:\Проэкты\V2.0 PD\outputs\chat_only_inventory_smoke11_20260927")
CAPTURE = Path(r"D:\Проэкты\V2.0 PD\outputs\nine_table_eventlog_smoke11_final_20260927")
VERSION = "dev_queue_v1"
SNAPSHOT = re.compile(r"T(?P<table>\d{2})_f(?P<frame>\d{8})_\d+\.png$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_checked(source: Path, destination: Path, expected_hash: str | None = None) -> dict:
    source = source.resolve(strict=True)
    digest = sha256(source)
    pixel_hash = None
    if expected_hash:
        image = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode source crop: {source}")
        pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
        if pixel_hash != expected_hash:
            raise ValueError(f"Catalog pixel hash differs from source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != digest:
            raise ValueError(f"Existing review file differs: {destination}")
    else:
        shutil.copy2(source, destination)
    return {"source": str(source), "local": str(destination.relative_to(ROOT)),
            "file_sha256": digest, "catalog_pixel_sha256": pixel_hash,
            "bytes": source.stat().st_size}


def stable_order(row: dict) -> str:
    return hashlib.sha256(row.get("row_id", row.get("crop", "")).encode()).hexdigest()


def choose_messages(rows: list[dict], limit: int) -> list[dict]:
    selected: dict[str, dict] = {}
    multi = sorted((row for row in rows if " | " in row["source_crops"]),
                   key=stable_order)
    for row in multi:
        if len(selected) < limit:
            selected[row["row_id"]] = row
    initial = sorted((row for row in rows if row["initially_visible"] == "True"),
                     key=stable_order)
    for row in initial[:20]:
        if len(selected) < limit:
            selected[row["row_id"]] = row
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["event_type"]].append(row)
    for group in by_type.values():
        group.sort(key=stable_order)
    offsets = {kind: 0 for kind in by_type}
    while len(selected) < limit:
        progress = False
        for kind in sorted(by_type):
            group = by_type[kind]
            while offsets[kind] < len(group) and group[offsets[kind]]["row_id"] in selected:
                offsets[kind] += 1
            if offsets[kind] < len(group):
                row = group[offsets[kind]]
                selected[row["row_id"]] = row
                offsets[kind] += 1
                progress = True
                if len(selected) >= limit:
                    break
        if not progress:
            raise ValueError(f"Only {len(selected)} message candidates available")
    return sorted(selected.values(), key=lambda row: (int(row["table"]),
                                                       int(row["frame"]), row["row_id"]))


def choose_transitions(rows: list[dict], limit: int) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["table"], row["status"])].append(row)
    for group in groups.values():
        group.sort(key=stable_order)
    offsets = {key: 0 for key in groups}
    selected = []
    while len(selected) < limit:
        progress = False
        for key in sorted(groups):
            index = offsets[key]
            if index < len(groups[key]):
                selected.append(groups[key][index])
                offsets[key] += 1
                progress = True
                if len(selected) >= limit:
                    break
        if not progress:
            raise ValueError(f"Only {len(selected)} unresolved transitions available")
    return sorted(selected, key=lambda row: (int(row["table"]), int(row["frame"])))


def snapshot_index() -> dict[str, tuple[list[int], dict[int, Path]]]:
    grouped: dict[str, dict[int, Path]] = defaultdict(dict)
    for path in (CAPTURE / "snapshots").glob("*.png"):
        match = SNAPSHOT.match(path.name)
        if match:
            grouped[str(int(match.group("table")))].setdefault(int(match.group("frame")), path)
    return {table: (sorted(frames), frames) for table, frames in grouped.items()}


def extract_table_video_frame(table: str, frame: int, destination: Path,
                              video_hash_cache: dict[str, str]) -> dict:
    video = CAPTURE / "table_videos" / f"table_{int(table):02d}.mp4"
    if not video.is_file():
        raise FileNotFoundError(video)
    capture = cv2.VideoCapture(str(video))
    try:
        if frame >= int(capture.get(cv2.CAP_PROP_FRAME_COUNT)):
            raise ValueError(f"Frame {frame} beyond {video}")
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame):
            raise ValueError(f"Cannot seek to frame {frame} in {video}")
        ok, image = capture.read()
        if not ok or image is None:
            raise ValueError(f"Cannot decode frame {frame} in {video}")
        if round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1:
            raise ValueError(f"Unexpected decoded frame position in {video}")
        image = image[:, :847]
        encoded_ok, encoded = cv2.imencode(".png", image)
        if not encoded_ok:
            raise ValueError(f"Cannot encode frame {frame} in {video}")
        data = encoded.tobytes()
    finally:
        capture.release()
    digest = hashlib.sha256(data).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != digest:
            raise ValueError(f"Existing derived context differs: {destination}")
    else:
        destination.write_bytes(data)
    video_hash_cache.setdefault(table, sha256(video))
    return {"source_video": str(video), "source_video_sha256": video_hash_cache[table],
            "source_frame_id": frame, "local": str(destination.relative_to(ROOT)),
            "file_sha256": digest, "bytes": len(data),
            "derivation": "DECODED_LOSSY_TABLE_VIDEO_FRAME"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=int, default=300)
    parser.add_argument("--transitions", type=int, default=30)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "annotation" / "dev_smoke11_queue_20260927")
    args = parser.parse_args()
    if args.messages < 1 or args.transitions < 1:
        parser.error("Selection counts must be positive")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("Output must remain inside the new parser project")
    message_catalog = CATALOG / "chat_event_candidates.csv"
    transition_catalog = CATALOG / "unresolved_chat_views.csv"
    all_message_rows = list(csv.DictReader(message_catalog.open(encoding="utf-8-sig")))
    messages = choose_messages(all_message_rows, args.messages)
    transitions = choose_transitions(list(csv.DictReader(transition_catalog.open(encoding="utf-8-sig"))),
                                     args.transitions)
    index = snapshot_index()
    video_hash_cache: dict[str, str] = {}
    message_entries = []
    for row in messages:
        crop_paths = row["source_crops"].split(" | ")
        crop_hashes = row["crop_sha256"].split(" | ")
        if len(crop_paths) != len(crop_hashes):
            raise ValueError(f"Crop/hash mismatch in {row['row_id']}")
        copies = [copy_checked(Path(path), output / "messages" /
                               f"{row['row_id']}_{ordinal:02d}.png", crop_hashes[ordinal])
                  for ordinal, path in enumerate(crop_paths)]
        snapshot = CAPTURE / "snapshots" / Path(crop_paths[0]).name
        message_entries.append({"row_id": row["row_id"], "table": row["table"],
                                "frame": row["frame"], "catalog_type_hint": row["event_type"],
                                "raw_ocr_hint": row["raw_ocr"],
                                "initially_visible": row["initially_visible"] == "True",
                                "review_status": "UNREVIEWED", "crops": copies,
                                "context_snapshot_source": str(snapshot),
                                "context_snapshot_sha256": sha256(snapshot)})
    transition_entries = []
    for ordinal, row in enumerate(transitions):
        table, frame = row["table"], int(row["frame"])
        crop = copy_checked(Path(row["crop"]), output / "transitions" /
                            f"{ordinal:03d}_row.png", row["crop_sha256"])
        frame_ids, by_frame = index[table]
        at = bisect.bisect_left(frame_ids, frame)
        nearby = {}
        for label, position in (("before", at - 1), ("after", bisect.bisect_right(frame_ids, frame))):
            if 0 <= position < len(frame_ids):
                nearby[label] = copy_checked(by_frame[frame_ids[position]],
                                             output / "transitions" /
                                             f"{ordinal:03d}_{label}.png")
        current_source = CAPTURE / "snapshots" / Path(row["crop"]).name
        current_selection = "EXACT_ROW_SNAPSHOT"
        if not current_source.is_file():
            current_source = by_frame.get(frame)
            current_selection = "SAME_FRAME_SNAPSHOT"
        if current_source is None:
            current_selection = "DECODED_TABLE_VIDEO_FRAME"
            nearby["current"] = extract_table_video_frame(
                table, frame, output / "transitions" /
                f"{ordinal:03d}_current.png", video_hash_cache)
        else:
            nearby["current"] = copy_checked(current_source,
                                             output / "transitions" /
                                             f"{ordinal:03d}_current.png")
        transition_entries.append({"table": table, "frame": frame,
                                   "catalog_status_hint": row["status"],
                                   "review_status": "UNREVIEWED",
                                   "row_crop": crop, "context_snapshots": nearby,
                                   "current_snapshot_selection": current_selection})
    headers_by_table: dict[str, list[dict]] = defaultdict(list)
    for row in all_message_rows:
        if row["event_type"] == "HAND_HEADER":
            headers_by_table[row["table"]].append(row)
    hand_entries = []
    for table, headers in sorted(headers_by_table.items()):
        headers.sort(key=lambda row: int(row["frame"]))
        table_events = [row for row in all_message_rows if row["table"] == table]
        for before, after in zip(headers, headers[1:]):
            first, last = int(before["frame"]), int(after["frame"])
            index_no = len(hand_entries)
            start_image = extract_table_video_frame(
                table, first, output / "hands" / f"{index_no:03d}_start.png",
                video_hash_cache)
            next_image = extract_table_video_frame(
                table, last, output / "hands" / f"{index_no:03d}_next_header.png",
                video_hash_cache)
            in_window = [row for row in table_events if first <= int(row["frame"]) < last]
            hand_entries.append({"table": table, "start_frame": first,
                                 "next_header_frame": last,
                                 "start_header_row_id": before["row_id"],
                                 "next_header_row_id": after["row_id"],
                                 "start_header_ocr_hint": before["raw_ocr"],
                                 "initially_visible_at_source_start": before["initially_visible"] == "True",
                                 "catalog_rows_between_headers": len(in_window),
                                 "review_status": "UNREVIEWED_COMPLETENESS",
                                 "start_image": start_image,
                                 "next_header_image": next_image})
    manifest = {"selection_version": VERSION, "role": "DEVELOPMENT_REVIEW_ONLY",
                "independent_test": False,
                "source_catalogs": {str(message_catalog): sha256(message_catalog),
                                    str(transition_catalog): sha256(transition_catalog)},
                "messages": message_entries, "transitions": transition_entries,
                "hand_windows": hand_entries}
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                       indent=2) + "\n", encoding="utf-8")
    with (output / "messages_to_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("row_id", "table", "frame", "catalog_type_hint", "raw_ocr_hint",
                         "initially_visible", "crop_files", "context_snapshot_source",
                         "review_status", "verified_text", "verified_type",
                         "message_boundary", "reviewer", "review_note"))
        for entry in message_entries:
            writer.writerow((entry["row_id"], entry["table"], entry["frame"],
                             entry["catalog_type_hint"], entry["raw_ocr_hint"],
                             entry["initially_visible"],
                             " | ".join(crop["local"] for crop in entry["crops"]),
                             entry["context_snapshot_source"], "UNREVIEWED",
                             "", "", "", "", ""))
    with (output / "transitions_to_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("table", "frame", "catalog_status_hint", "row_crop",
                         "before_context", "current_context", "after_context",
                         "current_context_kind", "review_status", "verified_alignment",
                         "verified_new_rows", "reviewer", "review_note"))
        for entry in transition_entries:
            contexts = entry["context_snapshots"]
            writer.writerow((entry["table"], entry["frame"], entry["catalog_status_hint"],
                             entry["row_crop"]["local"],
                             contexts.get("before", {}).get("local", ""),
                             contexts["current"]["local"],
                             contexts.get("after", {}).get("local", ""),
                             entry["current_snapshot_selection"], "UNREVIEWED",
                             "", "", "", ""))
    with (output / "hands_to_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("table", "start_frame", "next_header_frame", "start_header_row_id",
                         "next_header_row_id", "start_header_ocr_hint",
                         "initially_visible_at_source_start", "catalog_rows_between_headers",
                         "start_image", "next_header_image", "table_video_source",
                         "table_video_sha256", "review_status", "verified_complete",
                         "verified_hand_number", "reviewer", "review_note"))
        for entry in hand_entries:
            writer.writerow((entry["table"], entry["start_frame"],
                             entry["next_header_frame"], entry["start_header_row_id"],
                             entry["next_header_row_id"], entry["start_header_ocr_hint"],
                             entry["initially_visible_at_source_start"],
                             entry["catalog_rows_between_headers"],
                             entry["start_image"]["local"],
                             entry["next_header_image"]["local"],
                             entry["start_image"]["source_video"],
                             entry["start_image"]["source_video_sha256"],
                             "UNREVIEWED_COMPLETENESS", "", "", "", ""))
    summary = {"messages": len(message_entries), "transitions": len(transition_entries),
               "hand_windows_for_review": len(hand_entries),
               "message_types": len({row["catalog_type_hint"] for row in message_entries}),
               "status": "UNREVIEWED_DEVELOPMENT_QUEUE", "independent_test": False}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False,
                                                     indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
