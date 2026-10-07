"""Strict card readings and continuity checks for separate visual evidence."""

from __future__ import annotations

import re
import hashlib
from collections import defaultdict


_SUITS = {"s": "s", "h": "h", "d": "d", "c": "c",
          "♠": "s", "♥": "h", "♦": "d", "♣": "c"}
_CARD = re.compile(r"(10|[2-9TJQKA])\s*([shdc♠♥♦♣])", re.IGNORECASE)
_SEPARATOR = re.compile(r"^[\s,;|/\[\](){}:.-]*$")
_BOARD_LABEL = re.compile(r"^\s*(?:board|flop|turn|river)\s*[:=]\s*", re.IGNORECASE)


def _chat_recognition_response(reading: dict) -> dict:
    if reading.get("recognition_mode") == "PIXEL_ONLY":
        raw = reading.get("raw_pixel_response") or {}
        if (raw.get("recognizer") != "FROZEN_NATIVE_PIXEL_CARD_READER" or
                not all(isinstance(raw.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", raw[key])
                        for key in ("rank_config_sha256", "suit_config_sha256"))):
            return {}
        return raw
    return reading.get("raw_vl_response") or {}


def _verified_chat_identity(reading: dict) -> bool:
    crop_sha = reading.get("crop_sha256")
    pixel_sha = reading.get("row_pixel_sha256")
    raw = _chat_recognition_response(reading)
    if (reading.get("identity_status") != "CHAT_ROW_SHA_ROUNDTRIP_VERIFIED" or
            not isinstance(crop_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", crop_sha) or
            not isinstance(pixel_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", pixel_sha) or
            not reading.get("row_id") or not raw.get("request_id") or raw.get("status") != "OK" or
            raw.get("source_sha256") != crop_sha):
        return False
    identity = f"{reading.get('table_session_id')}:{reading.get('frame_id')}:{reading.get('source_span_index')}:{crop_sha}"
    return reading.get("card_evidence_id") == "chatcard_"+hashlib.sha256(identity.encode()).hexdigest()[:24]


def visual_event(raw_text: str) -> tuple[str, str | None] | None:
    """Route only explicit street/show messages; leave ordinary actions alone."""
    text = raw_text.strip()
    lower = text.casefold()
    for phrase, street in (("dealing flop", "FLOP"), ("dealing turn", "TURN"),
                           ("dealing river", "RIVER"), ("раздаётся флоп", "FLOP"),
                           ("раздаётся терн", "TURN"), ("раздаётся ривер", "RIVER")):
        if phrase in lower:
            return "BOARD", street
    if "doesn't show" in lower or "does not show" in lower or "не показывает" in lower:
        return None
    if re.search(r"\bshows\b", lower) or "показывает" in lower:
        return "SHOWDOWN", None
    return None


def parse_cards(raw_text: str, expected: int, street: str | None = None) -> list[str] | None:
    """Accept only a complete card sequence, never repair missing or extra text."""
    raw_text = raw_text.strip()
    label = _BOARD_LABEL.match(raw_text)
    if label:
        name = label.group(0).strip().split(":")[0].split("=")[0].strip().upper()
        if street is not None and name != "BOARD" and name != street:
            return None
        raw_text = raw_text[label.end():]
    matches = list(_CARD.finditer(raw_text))
    if len(matches) != expected:
        return None
    rest = raw_text[:matches[0].start()] if matches else raw_text
    for left, right in zip(matches, matches[1:]):
        rest += raw_text[left.end():right.start()]
    if matches:
        rest += raw_text[matches[-1].end():]
    if not _SEPARATOR.fullmatch(rest):
        return None
    cards = [("T" if match.group(1) == "10" else match.group(1).upper()) +
             _SUITS[match.group(2).lower()] for match in matches]
    return cards if len(set(cards)) == expected else None


def board_status(street: str, cards: list[str] | None,
                 previous: list[str] | None) -> str:
    required = {"FLOP": 3, "TURN": 4, "RIVER": 5}[street]
    if cards is None or len(cards) != required:
        return "UNRESOLVED_READING"
    if street == "FLOP":
        return "RAW_UNREVIEWED"
    if previous is None:
        return "RAW_UNREVIEWED"
    if cards[:len(previous)] != previous or len(previous) != required - 1:
        return "CONFLICT_BOARD_CONTINUITY"
    return "RAW_UNREVIEWED"


def associate_visual_evidence(timeline: list[dict], readings: dict[str, dict]) -> None:
    """Attach card-tile readings by physical order, never VL completion order."""
    hand_ids = {}
    pending = {}
    boards = defaultdict(dict)
    contexts = {}

    def missing(trigger):
        if trigger["type"] == "BOARD":
            return
        key = "MISSING:" + trigger["source_row_key"]
        readings[key] = {"type": trigger["type"], "street": trigger["street"],
                         "player": trigger["player"], "table_id": trigger["table_id"],
                         "table_session_id": trigger["session_id"],
                         "frame_id": trigger["frame_id"],
                         "hand_id": trigger["hand_id"],
                         "trigger_row_key": trigger["source_row_key"],
                         "crop_type": "CHAT_ROW_CARDS", "crop_path": None,
                         "raw_vl_response": None, "status": "UNRESOLVED",
                         "reason": "NO_VALID_ADJACENT_CARD_TILE"}

    for item in timeline:
        table = item["table_id"]
        context = (item["session_id"], item.get("epoch"))
        if table in contexts and contexts[table] != context:
            if table in pending:
                missing(pending.pop(table))
            hand_ids.pop(table, None)
        contexts[table] = context
        if item.get("event_type") == "NEW_HAND":
            if table in pending:
                missing(pending.pop(table))
            hand_ids.pop(table, None)
        if item.get("event_type") == "HAND_ID":
            number = re.match(r"\s*#\s*(\d+)", item.get("raw_ocr_text", ""))
            if number:
                if table in pending:
                    missing(pending.pop(table))
                hand_ids[table] = number.group(1)
        event = visual_event(item.get("raw_ocr_text", ""))
        if event:
            if table in pending:
                missing(pending.pop(table))
            kind, street = event
            actor = None
            if kind == "SHOWDOWN":
                match = re.search(r"(?:Dealer|Дилер)\s*:\s*(.*?)\s+(?:shows|показывает)\b",
                                  item["raw_ocr_text"], re.IGNORECASE)
                actor = match.group(1) if match else None
            pending[table] = {"type": kind, "street": street,
                              "table_id": table,
                              "frame_id": item["frame_id"],
                              "row_index": item["row_index"],
                              "epoch": item.get("epoch"),
                              "session_id": item["session_id"],
                              "hand_id": hand_ids.get(table),
                              "origin": item.get("origin"),
                              "row_status": item.get("status"),
                              "source_row_key": f"{int(table[1:])}:{item['frame_id']}:{item['row_index']}",
                              "player": actor}
        key = f"{int(table[1:])}:{item['frame_id']}:{item['row_index']}"
        reading = readings.get(key)
        if reading is None:
            continue
        if reading.get("hybrid_authoritative_rejection"):
            reading["cards"] = None
            reading.setdefault("hybrid_rejection_reason", reading.get("reason"))
        if item.get("origin") == "UNRESOLVED_IDENTITY":
            reading.update(status="UNRESOLVED", reason="ROW_IDENTITY_UNRESOLVED")
            continue
        if reading.get("identity_status") == "CHAT_ROW_SHA_ROUNDTRIP_VERIFIED" and (
                not _verified_chat_identity(reading) or
                reading.get("table_id") not in {int(table[1:]), table} or
                reading.get("table_session_id") != item["session_id"] or
                reading.get("frame_id") != item["frame_id"] or
                reading.get("epoch") != item.get("epoch") or
                reading.get("row_pixel_sha256") != item.get("row_pixel_sha256") or
                reading.get("row_id") != item.get("stable_row_id") or
                _chat_recognition_response(reading).get("source_sha256") != reading.get("crop_sha256")):
            reading.update(status="UNRESOLVED", reason="FOREIGN_CHAT_CARD_EVIDENCE")
            continue
        trigger = pending.get(table)
        reading["hand_id"] = hand_ids.get(table)
        if (trigger is None or trigger["session_id"] != item["session_id"] or
                trigger["epoch"] != item.get("epoch") or
                item["frame_id"] != trigger["frame_id"] or
                not 1 <= item["row_index"] - trigger["row_index"] <= 2):
            reading["status"] = "UNRESOLVED"
            if reading.get("hybrid_authoritative_rejection"):
                reading["context_reason"] = "NO_NEARBY_CARD_EVENT"
            else:
                reading["reason"] = "NO_NEARBY_CARD_EVENT"
            continue
        if trigger.get("origin") == "UNRESOLVED_IDENTITY" or trigger.get("row_status") == "UNRESOLVED":
            reading.update(status="UNRESOLVED", reason="TRIGGER_ROW_IDENTITY_UNRESOLVED")
            pending.pop(table, None)
            continue
        if reading.get("hybrid_authoritative_rejection") or reading.get("reason") in {"SUIT_CONFLICT", "PRIMARY_VL_RANK_CONFLICT",
                                     "VL_RANK_COUNT_MISMATCH", "DUPLICATE_CARD_CONFLICT",
                                     "UNSUPPORTED_TILE_GEOMETRY"}:
            pending.pop(table, None)
            reading["type"] = trigger["type"]
            reading["street"] = trigger["street"]
            reading["player"] = trigger["player"]
            reading["trigger_row_key"] = trigger["source_row_key"]
            reading["status"] = "UNRESOLVED"
            continue
        expected = {"FLOP": 3, "TURN": 4, "RIVER": 5}.get(trigger["street"], 2)
        cards = reading.get("cards")
        if (trigger["type"] == "BOARD" and reading.get("vl_status") == "OK" and
                reading.get("identity_status") != "CHAT_ROW_SHA_ROUNDTRIP_VERIFIED"):
            raw_cards = parse_cards(reading.get("vl_raw_text", ""), expected,
                                    trigger["street"])
            if raw_cards is not None:
                if cards is not None and cards != raw_cards:
                    reading["status"] = "UNRESOLVED"
                    reading["reason"] = "VL_TEXT_PIXEL_CARD_CONFLICT"
                    continue
                if cards is None:
                    cards = raw_cards
                    reading["cards"] = raw_cards
                    reading["card_reader"] = "vl_text_strict_v1"
        if cards is None or parse_cards(" ".join(cards), expected) != cards:
            reading["status"] = "UNRESOLVED"
            reading["reason"] = "CARD_COUNT_OR_READING_UNRESOLVED"
            continue
        pending.pop(table, None)
        reading["type"] = trigger["type"]
        reading["street"] = trigger["street"]
        reading["player"] = trigger["player"]
        reading["trigger_row_key"] = trigger["source_row_key"]
        if trigger["type"] == "BOARD":
            hand = reading["hand_id"]
            if hand is None:
                if reading.get("identity_status") == "CHAT_ROW_SHA_ROUNDTRIP_VERIFIED":
                    reading["status"] = "RAW_UNREVIEWED"
                    reading["continuity_status"] = "PARTIAL_HAND_CONTEXT"
                    reading["context_reason"] = "HAND_ID_NOT_VISIBLE"
                    continue
                reading["status"] = "UNRESOLVED"
                reading["reason"] = "HAND_ID_NOT_VISIBLE"
                continue
            previous_street = {"FLOP": None, "TURN": "FLOP", "RIVER": "TURN"}[trigger["street"]]
            board_key = (table, item["session_id"], item.get("epoch"), hand)
            previous = boards[board_key].get(previous_street) if previous_street else None
            reading["status"] = board_status(trigger["street"], cards, previous)
            if previous_street and previous is None:
                reading["continuity_status"] = "PREVIOUS_STREET_NOT_AVAILABLE"
            else:
                reading["continuity_status"] = reading["status"]
            if reading["status"] == "RAW_UNREVIEWED":
                boards[board_key][trigger["street"]] = cards
        elif trigger["player"] is None:
            reading["status"] = "UNRESOLVED"
            reading["reason"] = "SHOW_PLAYER_UNREADABLE"
    for trigger in pending.values():
        missing(trigger)


def project_board_events(timeline: list[dict], readings: dict[str, dict]) -> int:
    """Add accepted visual BOARD fields to the existing physical tile row."""
    by_key = {f"{int(item['table_id'][1:])}:{item['frame_id']}:{item['row_index']}": item
              for item in timeline}
    count = 0
    for key, reading in readings.items():
        if (reading.get("type") != "BOARD" or reading.get("crop_type") != "BOARD" or
                reading.get("identity_status") != "SHA_ROUNDTRIP_VERIFIED" or
                reading.get("status") != "RAW_UNREVIEWED" or
                reading.get("hand_id") is None or not reading.get("cards")):
            continue
        item = by_key.get(reading.get("trigger_row_key"))
        if item is None or item["session_id"] != reading.get("table_session_id"):
            continue
        item["ocr_event_type"] = item["event_type"]
        item["event_type"] = "BOARD"
        item["hand_id"] = reading["hand_id"]
        item["street"] = reading["street"]
        item["cards"] = reading["cards"]
        item["raw_vl_response"] = reading.get("vl_raw_text")
        item["evidence_status"] = reading["status"]
        item["continuity_status"] = reading.get("continuity_status")
        item["visual_crop_path"] = reading.get("crop_path")
        item["visual_crop_sha256"] = reading.get("crop_sha256")
        item["board_sha"] = reading.get("board_sha")
        item["board_evidence_id"] = reading.get("board_evidence_id")
        item["trigger_row_key"] = reading.get("trigger_row_key")
        count += 1
    return count


def project_chat_card_events(timeline: list[dict], readings: dict[str, dict]) -> int:
    """Project verified CHAT readings as raw events, including partial hand context."""
    by_key = {f"{int(item['table_id'][1:])}:{item['frame_id']}:{item['row_index']}": item
              for item in timeline}
    count = 0
    for reading in readings.values():
        reading["projected_card_count"] = 0
        kind = reading.get("type")
        if (reading.get("hybrid_authoritative_rejection") or
                kind not in {"BOARD", "SHOWDOWN"} or reading.get("crop_type") != "CHAT_ROW_CARDS" or
                not _verified_chat_identity(reading) or
                reading.get("status") != "RAW_UNREVIEWED" or
                _chat_recognition_response(reading).get("source_sha256") != reading.get("crop_sha256")):
            continue
        trigger = by_key.get(reading.get("trigger_row_key"))
        physical = by_key.get(reading.get("row_key"))
        if trigger is None or physical is None:
            continue
        if any(item.get("origin") == "UNRESOLVED_IDENTITY" or item.get("status") == "UNRESOLVED"
               for item in (trigger, physical)):
            continue
        if visual_event(trigger.get("raw_ocr_text", "")) != (kind, reading.get("street")):
            continue
        if (trigger["table_id"] != physical["table_id"] or
                reading.get("table_id") not in {int(physical["table_id"][1:]), physical["table_id"]} or
                trigger["session_id"] != reading.get("table_session_id") or
                physical["session_id"] != reading.get("table_session_id") or
                trigger["frame_id"] != reading.get("frame_id") or
                physical["frame_id"] != reading.get("frame_id") or
                trigger.get("epoch") != reading.get("epoch") or
                physical.get("epoch") != reading.get("epoch") or
                physical.get("stable_row_id") != reading.get("row_id") or
                physical.get("row_pixel_sha256") != reading.get("row_pixel_sha256") or
                not 1 <= physical["row_index"]-trigger["row_index"] <= 2):
            continue
        expected = {"FLOP": 3, "TURN": 4, "RIVER": 5}.get(reading.get("street"), 2)
        cards = reading.get("cards")
        if not cards or parse_cards(" ".join(cards), expected) != cards:
            continue
        trigger.update(ocr_event_type=trigger["event_type"], event_type=kind,
            hand_id=reading.get("hand_id"), street=reading.get("street"),
            player=reading.get("player"), cards=cards,
            status="RAW_UNREVIEWED", raw_vl_response=reading.get("vl_raw_text"),
            evidence_status="RAW_UNREVIEWED", continuity_status=reading.get("continuity_status"),
            evidence_context_status="HAND_CONTEXT_AVAILABLE" if reading.get("hand_id") else "PARTIAL_HAND_CONTEXT",
            visual_crop_path=reading.get("crop_path"), visual_crop_sha256=reading.get("crop_sha256"),
            visual_crop_type="CHAT_ROW_CARDS", identity_status=reading.get("identity_status"),
            card_evidence_id=reading.get("card_evidence_id"), card_row_key=reading.get("row_key"),
            card_row_id=reading.get("row_id"), card_crop_pixel_sha256=reading.get("row_pixel_sha256"),
            trigger_row_key=reading.get("trigger_row_key"))
        trigger.update(recognition_mode=reading.get("recognition_mode", "VL"),
                       raw_pixel_response=reading.get("raw_pixel_response"),
                       development_only=(reading.get("hybrid") or {}).get("development_only", False),
                       rank_mode=(reading.get("hybrid") or {}).get("rank_mode", "OFF"),
                       rank_ncc_config_sha256=(reading.get("hybrid") or {}).get("rank_ncc_config_sha256"))
        if reading.get("recognition_mode") == "PIXEL_ONLY":
            trigger["raw_vl_response"] = None
        reading["projected_card_count"] = len(cards)
        count += 1
    return count
