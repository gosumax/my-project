"""Join exact-frame seat crops with raw chat rows for START STACK proof.

PokerDom's first numeric stack is post-forced-bets. Raw chat first appearance
does not prove voluntary-action timing when it coincides with a stack frame.
"""

from __future__ import annotations

import re
from collections import defaultdict

from parser_core.start_stack import ChatAction, HandStackTracker, StackReading, money


POST = re.compile(r"^Dealer:\s*(.+?)\s+posts\s+(Ante|Small Blind|Big Blind)(?:\s+([\d ]+))?\s*$", re.I)
ACTION = re.compile(r"^Dealer:\s*(.+?)\s+(calls|bets|raises to)\s+([\d ]+)\s*$", re.I)
NUMBER = re.compile(r"^\s*\d+(?:[ ]\d+)*\s*$")
KIND = {"ante": "ANTE", "small blind": "SMALL_BLIND",
        "big blind": "BIG_BLIND", "calls": "CALL", "bets": "BET",
        "raises to": "RAISE_TO"}
FORCED = {"ANTE", "SMALL_BLIND", "BIG_BLIND"}


def hand_windows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["table_id"]].append(row)
    windows = []
    for table_id, items in grouped.items():
        items.sort(key=lambda row: (row["frame_id"], row["row_index"]))
        starts = []
        seen = set()
        for index, row in enumerate(items):
            if row.get("event_type") != "HAND_ID":
                continue
            text = row.get("raw_ocr_text", "").strip()
            if not text.startswith("#") or not text[1:].isdigit():
                continue
            key = text[1:]
            if key in seen:
                continue
            seen.add(key)
            starts.append((index, key))
        for pos, (index, number) in enumerate(starts):
            next_index = starts[pos + 1][0] if pos + 1 < len(starts) else len(items)
            windows.append({"table_id": table_id, "hand_number": number,
                            "start_frame": items[index]["frame_id"],
                            "end_frame": items[next_index]["frame_id"] if next_index < len(items) else None,
                            "rows": items[index:next_index]})
    return sorted(windows, key=lambda w: (w["table_id"], w["start_frame"]))


def parse_actions(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    actions, unparsed = [], []
    i = 0
    while i < len(rows):
        row = rows[i]
        text = (row.get("raw_ocr_text") or "").strip()
        match = POST.fullmatch(text) or ACTION.fullmatch(text)
        if match:
            actor, label, amount = match.groups()
            if not amount and i + 1 < len(rows):
                next_row = rows[i + 1]
                if (next_row["frame_id"] == row["frame_id"] and
                        NUMBER.fullmatch((next_row.get("raw_ocr_text") or "").strip())):
                    amount = next_row["raw_ocr_text"]
                    i += 1
            if amount:
                actions.append({"player": actor, "event_type": KIND[label.lower()],
                    "amount": amount.replace(" ", ""), "frame_index": row["frame_id"],
                    "source_ref": row["source_roi_change_id"] + ":" + str(row["row_index"]),
                    "raw_text": text})
            else:
                unparsed.append(row)
        elif text.startswith("Dealer:") and not text.lower().startswith("dealer: dealing cards"):
            unparsed.append(row)
        i += 1
    return actions, unparsed


def prove_hand(window: dict, observations: list[dict], review: dict | None) -> dict:
    table_id = window["table_id"]
    start = window["start_frame"]
    rows = window["rows"]
    hand_id = window["hand_number"]
    relevant = sorted((x for x in observations if x["table_id"] == table_id and
                       x["frame_index"] >= start),
                      key=lambda x: (x["frame_index"], x["seat"], x["roi_type"]))
    if window["end_frame"] is not None:
        relevant = [x for x in relevant if x["frame_index"] < window["end_frame"]]
    actions, unparsed = parse_actions(rows)
    if review is not None:
        reviewed_pairs = {(seat_review[f"{prefix}_frame"],
                           seat_review[f"{prefix}_table_sha256"])
            for seat_review in review.get("seats", {}).values()
            for prefix in ("overlay", "stack") if f"{prefix}_frame" in seat_review}
        for supplemental in review.get("visual_chat_supplement", []):
            match = POST.fullmatch(supplemental.get("raw_visible_text", ""))
            if (match is None or (supplemental.get("frame_index"),
                    supplemental.get("table_sha256")) not in reviewed_pairs or
                    match.group(1) != supplemental.get("player") or
                    KIND[match.group(2).lower()] != supplemental.get("event_type") or
                    money(match.group(3)) != money(supplemental.get("amount")) or
                    supplemental["frame_index"] < start or
                    any(x["player"] == supplemental["player"] and
                        x["event_type"] == supplemental["event_type"] and
                        x["frame_index"] == supplemental["frame_index"] for x in actions)):
                raise ValueError("Visual chat supplement is inconsistent with reviewed frame or raw OCR")
            actions.append({"player": supplemental["player"],
                "event_type": supplemental["event_type"],
                "amount": str(supplemental["amount"]),
                "frame_index": supplemental["frame_index"],
                "source_ref": "VISUAL_TABLE_SHA256:" + supplemental["table_sha256"],
                "raw_text": supplemental["raw_visible_text"],
                "evidence_origin": "VISUALLY_REVIEWED_TABLE_FRAME"})
        actions.sort(key=lambda x: (x["frame_index"], x["source_ref"]))
    by_seat = defaultdict(list)
    for item in relevant:
        by_seat[item["seat"]].append(item)
    players = []
    for seat, seat_obs in sorted(by_seat.items()):
        stacks = [x for x in seat_obs if x["roi_type"] == "stack"]
        nicks = [x for x in seat_obs if x["roi_type"] == "nick"]
        numeric = [x for x in stacks if x.get("visual_kind") == "STACK" and
                   x.get("confidence") is not None and x["confidence"] >= .90]
        candidate = numeric[0] if numeric else None
        if candidate is None:
            players.append({"seat": seat, "status": "UNKNOWN", "reason": "STACK_OCR_UNKNOWN"})
            continue
        frame = candidate["frame_index"]
        nickname_obs = next((x for x in nicks if x["frame_index"] == frame and
            x.get("visual_kind") == "NICKNAME" and x.get("confidence", 0) >= .90), None)
        nickname = nickname_obs["raw_recognition"] if nickname_obs else None
        if not nickname or nickname_obs["table_crop_sha256"] != candidate["table_crop_sha256"]:
            players.append({"seat": seat, "stack_frame": frame,
                "status": "UNKNOWN", "reason": "NICKNAME_OCR_UNKNOWN"})
            continue
        player_actions = [x for x in actions if x["player"] == nickname]
        forced_player_prefix = [x for x in player_actions if x["event_type"] in FORCED
                                and x["frame_index"] <= frame]
        role = "SB" if any(x["event_type"] == "SMALL_BLIND" for x in forced_player_prefix) else (
            "BB" if any(x["event_type"] == "BIG_BLIND" for x in forced_player_prefix) else "NORMAL")
        seat_review = review.get("seats", {}).get(seat) if review else None
        if seat_review is None:
            players.append({"seat": seat, "nickname": nickname, "stack_frame": frame,
                "status": "UNKNOWN", "reason": "EVIDENCE_REVIEW_MISSING"})
            continue
        if (seat_review.get("stack_frame") != frame or
                seat_review.get("stack_table_sha256") != candidate["table_crop_sha256"]):
            players.append({"seat": seat, "nickname": nickname, "stack_frame": frame,
                "status": "UNKNOWN", "reason": "STACK_FRAME_REVIEW_MISMATCH"})
            continue
        if not (review.get("forced_post_chat_visually_checked") is True and
                review.get("no_other_action_or_bet_chips_visible_through_stack") is True and
                review.get("reviewed_rows_override_unresolved_tracking") is True):
            players.append({"seat": seat, "nickname": nickname, "stack_frame": frame,
                "status": "UNKNOWN", "reason": "EVIDENCE_REVIEW_MISSING"})
            continue
        voluntary = [x for x in actions if x["event_type"] not in FORCED]
        if any(x["frame_index"] <= frame for x in voluntary):
            players.append({"seat": seat, "nickname": nickname, "stack_frame": frame,
                "status": "UNKNOWN", "reason": "VOLUNTARY_ACTION_TIMING_UNKNOWN"})
            continue
        # Raw rows later in a broad HAND_ID window may belong to a missed hand.
        # The reviewed stack frame supplies the forced-prefix boundary.
        forced = [x for x in actions if x["event_type"] in FORCED and
                  x["frame_index"] <= frame]
        forced_pot = sum((money(x["amount"]) for x in forced), start=money(0))
        if forced_pot != money(review.get("stack_frame_pot_visible")):
            players.append({"seat": seat, "nickname": nickname, "stack_frame": frame,
                "status": "UNKNOWN", "reason": "ACTION_ACCOUNTING_UNKNOWN"})
            continue
        attested = seat_review.get("reviewed_player_actions_before_stack")
        actual = sorted((x["event_type"], str(money(x["amount"])), x["source_ref"])
                        for x in forced if x["player"] == nickname)
        expected = sorted((x["event_type"], str(money(x["amount"])), x["source_ref"])
                          for x in attested) if isinstance(attested, list) else None
        if expected is None or actual != expected:
            players.append({"seat": seat, "nickname": nickname, "stack_frame": frame,
                "status": "UNKNOWN", "reason": "ACTION_ACCOUNTING_UNKNOWN"})
            continue
        if any(x["frame_id"] <= frame for x in unparsed):
            players.append({"seat": seat, "nickname": nickname, "stack_frame": frame,
                "status": "UNKNOWN", "reason": "ACTION_ACCOUNTING_UNKNOWN"})
            continue
        if (review.get("ante_policy", "UNKNOWN") != "ABSENT" and
                not any(x["event_type"] == "ANTE" for x in forced if
                        x["player"] == nickname)):
            players.append({"seat": seat, "nickname": nickname, "stack_frame": frame,
                "status": "UNKNOWN", "reason": "ACTION_ACCOUNTING_UNKNOWN"})
            continue
        tracker = HandStackTracker(table_id, candidate["table_context"], hand_id, start, False)
        tracker.bind_seat(seat, nickname, role, identity_proven=True,
                          ante_policy=review.get("ante_policy", "UNKNOWN"))
        tracker.observe(StackReading(table_id, candidate["table_context"], seat,
            frame, candidate["source_pts"], candidate["timestamp"],
            candidate["crop_sha256"], candidate["crop_path"],
            candidate["raw_recognition"], candidate["confidence"]))
        timed = [ChatAction(x["player"], x["event_type"], x["amount"],
            None if x["event_type"] in FORCED or x["frame_index"] == frame
            else x["frame_index"] < frame, x["source_ref"])
            for x in actions if x["player"] == nickname and
            (x["event_type"] not in FORCED or x in forced)]
        result = tracker.result(seat, timed, chat_complete=True, timing_proven=True)
        result["certification_basis"] = "POKERDOM_POST_FORCED_FIRST_STACK_AND_HASH_PINNED_VISUAL_REVIEW"
        result["stack_frame_pot_visible"] = str(forced_pot)
        if seat_review.get("require_prehand_stack_match"):
            prehand = [x for x in observations if x["table_id"] == table_id and
                       x["seat"] == seat and x["roi_type"] == "stack" and
                       x["frame_index"] < start and x.get("visual_kind") == "STACK" and
                       x.get("confidence", 0) >= .90]
            prehand.sort(key=lambda x: x["frame_index"])
            previous = prehand[-1] if prehand else None
            result["prehand_visible_stack"] = previous["visible_stack"] if previous else None
            result["prehand_frame"] = previous["frame_index"] if previous else None
            if previous is None or money(previous["visible_stack"]) != money(result["start_stack"]):
                result["status"] = "UNKNOWN"
                result["reason"] = "PREHAND_CONTINUITY_MISMATCH"
                result["start_stack"] = None
        result["nickname_crop_path"] = nickname_obs["crop_path"]
        result["nickname_crop_sha256"] = nickname_obs["crop_sha256"]
        result["overlay_frame"] = seat_review.get("overlay_frame")
        result["forced_bets_already_posted"] = [x for x in forced if x["player"] == nickname]
        result["voluntary_actions_before_stack"] = [x for x in voluntary if x["player"] == nickname and x["frame_index"] < frame]
        result["voluntary_actions_after_stack"] = [x for x in voluntary if x["player"] == nickname and x["frame_index"] > frame]
        result["chat_events_before_stack"] = [x for x in player_actions if x["frame_index"] < frame]
        result["chat_events_after_stack"] = [x for x in player_actions if x["frame_index"] > frame]
        result["reviewed_table_frames"] = sorted({frame, *([seat_review["overlay_frame"]]
            if "overlay_frame" in seat_review else [])})
        players.append(result)
    timeline = ([{"frame_index": row["frame_id"], "kind": "CHAT_ROW",
                  "source_roi_change_id": row["source_roi_change_id"],
                  "row_index": row["row_index"], "raw_text": row.get("raw_ocr_text"),
                  "event_type": row.get("event_type")}
                 for row in rows] +
                [{"frame_index": item["frame_index"], "kind": "SEAT_ROI",
                  "seat": item["seat"], "roi_type": item["roi_type"],
                  "raw_recognition": item.get("raw_recognition"),
                  "visual_kind": item.get("visual_kind"),
                  "table_crop_sha256": item["table_crop_sha256"]}
                 for item in relevant] +
                [{"frame_index": action["frame_index"], "kind": "PARSED_ACTION",
                  "player": action["player"], "event_type": action["event_type"],
                  "amount": action["amount"], "source_ref": action["source_ref"]}
                 for action in actions])
    timeline.sort(key=lambda item: (item["frame_index"],
                                    {"CHAT_ROW": 0, "PARSED_ACTION": 1, "SEAT_ROI": 2}[item["kind"]]))
    return {"table_id": table_id, "hand_id": hand_id, "hand_start_frame": start,
            "hand_start_status": "PARTIAL_START" if start == 0 else "OBSERVED_HAND_ID",
            "reviewed": review is not None, "players": players,
            "timeline": timeline,
            "chat_rows": rows, "parsed_actions": actions,
            "unparsed_rows": unparsed}
