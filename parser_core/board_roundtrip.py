"""Bind a BOARD VL answer to the exact submitted crop and its source event."""

from __future__ import annotations

import hashlib
import base64
import io
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from parser_core.visual_cards import board_status, parse_cards, visual_event
from parser_core.visual_crops import exact_pixel_sha256


def board_evidence_id(item: dict) -> str:
    return (f"{item['table_id']}:{item['session_id']}:{item['frame_id']}:"
            f"{item['row_index']}:{item['street']}")


def board_triggers(timeline: list[dict]) -> list[dict]:
    """Find explicit BOARD messages and the active hand in physical order."""
    hands: dict[str, tuple[tuple, str]] = {}
    found = []
    for item in timeline:
        table = item["table_id"]
        context = (item["session_id"], item.get("epoch"))
        if table in hands and hands[table][0] != context:
            hands.pop(table)
        if item.get("event_type") in ("HAND_ID", "NEW_HAND") or \
                item.get("message_event_type") == "NEW_HAND":
            hands.pop(table, None)
        if item.get("event_type") == "HAND_ID":
            match = re.match(r"\s*#\s*(\d+)", item.get("raw_ocr_text", ""))
            if match:
                hands[table] = (context, match.group(1))
        event = visual_event(item.get("raw_ocr_text", ""))
        if event is None or event[0] != "BOARD":
            continue
        trigger = {"table_id": table, "session_id": item["session_id"],
                   "epoch": item.get("epoch"), "frame_id": item["frame_id"],
                   "row_index": item["row_index"], "street": event[1],
                   "hand_id": hands.get(table, (None, None))[1],
                   "row_key": f"{int(table[1:])}:{item['frame_id']}:{item['row_index']}"}
        trigger["board_evidence_id"] = board_evidence_id(trigger)
        found.append(trigger)
    return found


def associate_board_crops(timeline: list[dict], evidence: list[dict]) -> dict:
    """Verify every crop/response, then validate hand and street separately."""
    triggers = {item["board_evidence_id"]: item for item in board_triggers(timeline)}
    trigger_order = {evidence_id: index for index, evidence_id in enumerate(triggers)}
    duplicate_events = Counter(row.get("board_evidence_id") for row in evidence)
    counts = defaultdict(int)
    response_ids = set()
    boards = defaultdict(dict)
    for row in sorted(evidence, key=lambda item: trigger_order.get(
            item.get("board_evidence_id"), len(trigger_order))):
        evidence_id = row.get("board_evidence_id")
        trigger = triggers.get(evidence_id)
        row["type"] = "BOARD"
        row["status"] = "UNRESOLVED"
        row.pop("cards", None)
        if evidence_id is not None and duplicate_events[evidence_id] > 1:
            row["reason"] = "DUPLICATE_BOARD_EVIDENCE_ID"
            counts[row["reason"]] += 1
            continue
        if trigger is None or any(row.get(field) != trigger.get(field) for field in
                                  ("table_id", "frame_id", "street")) or \
                row.get("table_session_id") != trigger["session_id"] or \
                row.get("trigger_row_key") != trigger["row_key"]:
            row["reason"] = "FOREIGN_BOARD_EVIDENCE"
            counts[row["reason"]] += 1
            continue
        row["hand_id"] = trigger["hand_id"]
        path = Path(row.get("crop_path") or "")
        try:
            data = (base64.b64decode(row["crop_png_base64"], validate=True)
                    if row.get("image_transport") == "ram" else path.read_bytes())
            file_sha = hashlib.sha256(data).hexdigest()
            with Image.open(io.BytesIO(data)) as image:
                pixel_sha = exact_pixel_sha256(image)
        except (OSError, ValueError):
            row["reason"] = "BOARD_CROP_UNAVAILABLE"
            counts[row["reason"]] += 1
            continue
        if file_sha != row.get("crop_sha256") or pixel_sha != row.get("board_sha"):
            row["reason"] = "BOARD_SHA_MISMATCH"
            counts[row["reason"]] += 1
            continue
        response = row.get("raw_vl_response")
        if not isinstance(response, dict) or response.get("status") != "OK":
            row["reason"] = "BOARD_VL_RESPONSE_UNAVAILABLE"
            counts[row["reason"]] += 1
            continue
        response_key = (row.get("vl_worker_id"), response.get("request_id"))
        if response.get("request_id") is None or response_key in response_ids:
            row["reason"] = "DUPLICATE_BOARD_VL_RESPONSE"
            counts[row["reason"]] += 1
            continue
        response_ids.add(response_key)
        if (response.get("board_sha") != pixel_sha or
                response.get("source_sha256") != file_sha or
                response.get("board_evidence_id") != evidence_id):
            row["reason"] = "BOARD_RESPONSE_SHA_MISMATCH"
            counts[row["reason"]] += 1
            continue
        row["identity_status"] = "SHA_ROUNDTRIP_VERIFIED"
        expected = {"FLOP": 3, "TURN": 4, "RIVER": 5}[trigger["street"]]
        cards = parse_cards(response.get("raw_text", ""), expected, trigger["street"])
        if cards is None:
            row["reason"] = "BOARD_CARDS_UNREADABLE"
            counts[row["reason"]] += 1
            continue
        if trigger["hand_id"] is None:
            row["reason"] = "HAND_ID_NOT_VISIBLE"
            counts[row["reason"]] += 1
            continue
        previous_street = {"FLOP": None, "TURN": "FLOP", "RIVER": "TURN"}[trigger["street"]]
        key = (trigger["table_id"], trigger["session_id"], trigger["hand_id"])
        previous = boards[key].get(previous_street) if previous_street else None
        status = board_status(trigger["street"], cards, previous)
        if status != "RAW_UNREVIEWED":
            row["reason"] = status
            counts[row["reason"]] += 1
            continue
        row["cards"] = cards
        row["status"] = status
        row["continuity_status"] = ("PREVIOUS_STREET_NOT_AVAILABLE"
                                    if previous_street and previous is None else status)
        row["vl_raw_text"] = response["raw_text"]
        row["row_key"] = trigger["row_key"]
        boards[key][trigger["street"]] = cards
        counts["accepted"] += 1
    return dict(counts)
