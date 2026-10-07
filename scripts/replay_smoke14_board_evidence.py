"""Rebuild BOARD projections from saved Smoke14 OCR/VL evidence, without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.visual_cards import (associate_visual_evidence, parse_cards,
                                      project_board_events)
from parser_core.board_roundtrip import associate_board_crops, board_triggers
from parser_core.visual_crops import exact_pixel_sha256
from PIL import Image


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--board-responses", type=Path,
                        help="Fresh raw VL responses for exact saved BOARD crop IDs")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output directory must be fresh")
    timeline_path = source / "ordered_timeline.jsonl"
    visual_path = source / "visual_evidence.jsonl"
    baseline = read_jsonl(timeline_path)
    prior = read_jsonl(visual_path)
    timeline = [dict(item) for item in baseline]
    readings = {}
    for item in prior:
        if item.get("crop_type") != "CHAT_ROW_CARDS" or not item.get("crop_path"):
            continue
        row = dict(item)
        for field in ("type", "street", "player", "hand_id", "trigger_row_key",
                      "continuity_status"):
            row.pop(field, None)
        row["status"] = "RAW_UNREVIEWED" if row.get("cards") else "UNRESOLVED"
        readings[row["row_key"]] = row
    board_by_row = {item["row_key"]: item for item in board_triggers(timeline)}
    board_records = []
    for item in prior:
        if item.get("crop_type") != "BOARD":
            continue
        row = dict(item)
        row["table_id"] = f"T{int(row['table_id']):02d}" if isinstance(row.get("table_id"), int) else row.get("table_id")
        trigger = board_by_row.get(row.get("trigger_row_key"))
        if trigger is not None:
            row["board_evidence_id"] = trigger["board_evidence_id"]
            with Image.open(row["crop_path"]) as image:
                row["board_sha"] = exact_pixel_sha256(image)
        board_records.append(row)
    override_count = 0
    if args.board_responses is not None:
        overrides = read_jsonl(args.board_responses)
        by_id = {row.get("board_evidence_id"): row for row in board_records}
        if len(by_id) != len(board_records):
            raise ValueError("Duplicate BOARD evidence IDs in source")
        seen = set()
        for override in overrides:
            evidence_id = override.get("board_evidence_id")
            if evidence_id in seen or evidence_id not in by_id:
                raise ValueError("Duplicate or foreign BOARD response override")
            seen.add(evidence_id)
            target = by_id[evidence_id]
            if (override.get("board_sha") != target.get("board_sha") or
                    override.get("crop_sha256") != target.get("crop_sha256") or
                    override.get("crop_path") != target.get("crop_path")):
                raise ValueError("BOARD response override does not match saved crop")
            target["raw_vl_response"] = override["raw_vl_response"]
            target["vl_status"] = override["raw_vl_response"].get("status")
            target["vl_raw_text"] = override["raw_vl_response"].get("raw_text", "")
            target["vl_worker_id"] = "board_probe"
            override_count += 1
    old_board = [item for item in prior if item.get("type") == "BOARD" and
                 item.get("crop_type") == "CHAT_ROW_CARDS"]
    old_raw_valid = sum(bool(item.get("crop_path")) and
                        parse_cards(item.get("vl_raw_text", ""),
                                    {"FLOP": 3, "TURN": 4, "RIVER": 5}.get(item.get("street"), -1))
                        is not None for item in old_board)
    old_accepted = sum(item.get("status") == "RAW_UNREVIEWED" for item in old_board)
    associate_visual_evidence(timeline, readings)
    board_counts = associate_board_crops(timeline, board_records)
    for row in board_records:
        readings["BOARD:" + str(row.get("board_evidence_id", row.get("row_key")))] = row
    projected = project_board_events(timeline, readings)
    if len(timeline) != len(baseline):
        raise AssertionError("Physical timeline row count changed")
    for old, new in zip(baseline, timeline):
        for field in ("table_id", "frame_id", "row_index", "source_roi_change_id",
                      "raw_ocr_text", "row_pixel_sha256", "session_id", "epoch"):
            if old.get(field) != new.get(field):
                raise AssertionError(f"Physical timeline changed at {field}")
    keys = [(item["source_roi_change_id"], item["row_index"]) for item in timeline]
    if len(keys) != len(set(keys)):
        raise AssertionError("Timeline source row key collision")
    boards = [item for item in timeline if item.get("event_type") == "BOARD"]
    if projected != len(boards):
        raise AssertionError("BOARD projection count mismatch")
    output.mkdir(parents=True)
    with (output / "ordered_timeline.jsonl").open("w", encoding="utf-8") as stream:
        for item in timeline:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (output / "visual_evidence.jsonl").open("w", encoding="utf-8") as stream:
        for item in sorted(readings.values(), key=lambda row: (str(row.get("table_id", "")),
                                                        row.get("frame_id", -1),
                                                        row.get("source_span_index", -1))):
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "source_timeline": str(timeline_path), "source_timeline_sha256": sha256(timeline_path),
        "source_visual_evidence": str(visual_path), "source_visual_evidence_sha256": sha256(visual_path),
        "source_full_frame_count": json.loads((source / "final_summary.json").read_text("utf-8"))["frames"],
        "table_count": len({item["table_id"] for item in timeline}),
        "physical_rows": len(timeline),
        "unique_hand_ids": len({item["raw_ocr_text"].strip() for item in timeline
                                if item.get("ocr_event_type", item["event_type"]) == "HAND_ID"}),
        "unique_source_row_keys": len(set(keys)),
        "raw_board_evidence_before": len(old_board),
        "board_crop_evidence": len(board_records),
        "new_board_responses": override_count,
        "board_crop_sha_verified": sum(row.get("reason") not in
            ("BOARD_SHA_MISMATCH", "BOARD_CROP_UNAVAILABLE") for row in board_records),
        "board_crop_results": board_counts,
        "raw_text_complete_before": old_raw_valid,
        "board_accepted_before": old_accepted,
        "board_accepted_after": projected,
        "streets": dict(Counter(item["street"] for item in boards)),
        "hands_with_board": len({(item["table_id"], item["session_id"], item["hand_id"])
                                 for item in boards}),
        "unresolved_reasons": dict(Counter(item.get("reason", "UNKNOWN") for item in readings.values()
                                           if item.get("status") == "UNRESOLVED")),
        "missing_board_evidence": sum(item.get("type") == "BOARD" and
                                      item.get("crop_path") is None for item in readings.values()),
    }
    (output / "board_replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
