"""Offline evidence edge classification, not a promoted production component.

Reuse the existing experimental message parser; never rewrite its source rows.
KEEP proves observed edges only, not poker validity, accounting or certification.
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import dataclass

from parser_core.poker_hand_assembly import assemble_messages

STATUSES = {"KEEP", "PARTIAL_START", "PARTIAL_END", "UNRESOLVED_EDGE"}
VERSION = "offline-edge-gate-v1"


@dataclass
class EdgeResult:
    rows: list[dict]
    sessions: list[dict]


def _identity(row):
    return row.get("stable_row_id") or row.get("row_id") or row.get("sequence_id")


def _safe(message):
    return all(row.get("origin") != "UNRESOLVED_IDENTITY" and
               row.get("status") not in {"UNRESOLVED", "ERROR", "UNREADABLE"} and
               row.get("notes_error", "") in {None, "", "TYPE_NOT_RECOGNIZED"} and
               bool(row.get("raw_ocr_text") or row.get("raw_text"))
               for row in message["source_rows"])


def confirmed_hand_starts(messages):
    """Explicit server NEW_HAND or contiguous [ANTE*] SB BB DEAL.

    A number/header alone is never a start. Do not bridge unknown messages,
    epoch changes or unsafe provenance to complete a forced-bet pattern.
    """
    starts = {}
    pending = None
    phase = None
    def forced(message):
        return bool(_safe(message) and message["player"] and message["amount"] and
                    int(message["amount"]) > 0)
    for index, message in enumerate(messages):
        action = message["action"]
        safe = _safe(message)
        epoch = message["source_rows"][0].get("epoch")
        if pending is not None and epoch != messages[pending]["source_rows"][0].get("epoch"):
            pending = phase = None
        if action == "NEW_HAND" and safe:
            start = index
            # A preceding visible header belongs to the same explicit start,
            # but cannot establish a start by itself.
            if index and messages[index-1]["action"] == "HAND_HEADER" and _safe(messages[index-1]):
                prior = messages[index-1]
                if prior["source_rows"][0].get("epoch") == epoch and (
                        not message["hand_number"] or not prior["hand_number"] or
                        message["hand_number"] == prior["hand_number"]):
                    start -= 1
            starts[start] = "EXPLICIT_NEW_HAND"
            pending = phase = None
        elif forced(message) and action == "ANTE" and phase in {None, "ANTE"}:
            pending = index if pending is None else pending
            phase = "ANTE"
        elif forced(message) and action == "SMALL_BLIND" and phase in {None, "ANTE"}:
            pending = index if pending is None else pending
            phase = "SB"
        elif forced(message) and action == "BIG_BLIND" and phase == "SB":
            phase = "BB"
        elif safe and action == "DEAL" and phase == "BB":
            # Do not create a second start for the blinds of an explicit start.
            last_start = max(starts, default=-1)
            intervening = messages[last_start+1:pending] if last_start >= 0 else []
            if last_start < 0 or any(m["action"] in {"DEAL", "FOLD", "CHECK", "CALL", "BET",
                                                     "RAISE", "FLOP", "TURN", "RIVER", "SHOW", "WIN"}
                                    for m in intervening):
                starts[pending] = "FORCED_BETS_AND_DEAL"
            pending = phase = None
        else:
            pending = phase = None
    # A repeated number is unverified OCR/observation, not proof of the next
    # hand. Leave its messages inside the uncertain window for classification.
    seen_numbers = set()
    accepted = {}
    for index in sorted(starts):
        number = None
        for message in messages[index:]:
            if message["action"] not in {"HAND_HEADER", "HAND_ID", "NEW_HAND"}:
                break
            number = number or message["hand_number"]
        if number and number in seen_numbers:
            continue
        if number:
            seen_numbers.add(number)
        accepted[index] = starts[index]
    return accepted


def _confirmed_end(messages):
    # Existing assembly treats WIN/HAND_END as completion. Restrict WIN to a
    # source-linked, parsed positive payout; a SHOW or lone FOLD is insufficient.
    for index in range(len(messages)-1, -1, -1):
        message = messages[index]
        if not _safe(message):
            continue
        # Only explicitly identified post-hand messages may follow completion.
        tail_safe = all(_safe(m) and m["action"] in {
            "WIN", "SHOW", "NO_SHOW", "UNCALLED_RETURN", "TABLE_INFO", "BLIND_LEVEL", "HAND_END"}
            for m in messages[index+1:])
        if message["action"] == "WIN" and message["player"] and message["amount"]:
            if int(message["amount"]) > 0 and tail_safe:
                return "OBSERVED_WIN_PAYOUT"
        if message["action"] == "HAND_END" and tail_safe and all(
                r.get("status") in {"CONFIRMED", "VERIFIED", "CERTIFIED"}
                for r in message["source_rows"]):
            return "VERIFIED_HAND_END"
    return None


def classify_edges(rows: list[dict]) -> EdgeResult:
    """Classify in caller order, independently per table/session, without edits."""
    output = copy.deepcopy(rows)
    groups = defaultdict(list)
    for position, row in enumerate(rows):
        output[position]["edge_status"] = "UNRESOLVED_EDGE"
        key = (row.get("table_id"), row.get("session_id"))
        if all(value is not None and value != "" for value in key):
            groups[key].append(position)
    summaries = []
    for (table, session), positions in groups.items():
        source = [rows[p] for p in positions]
        identities = [_identity(row) for row in source]
        orders = [(row.get("frame_id"), row.get("row_index")) for row in source]
        valid = (all(identity is not None for identity in identities) and
                 len(set(identities)) == len(identities) and
                 all(isinstance(f, int) and isinstance(r, int) for f, r in orders) and
                 len(set(orders)) == len(orders) and
                 orders == sorted(orders))
        messages = assemble_messages(source) if valid else []
        starts = confirmed_hand_starts(messages)
        start_indices = sorted(starts)
        repeated_windows = set()
        number_windows = {}
        for n, begin in enumerate(start_indices):
            end = start_indices[n+1] if n+1 < len(start_indices) else len(messages)
            for number in {m["hand_number"] for m in messages[begin:end]
                           if m["action"] in {"HAND_HEADER", "HAND_ID", "NEW_HAND"} and m["hand_number"]}:
                if number in number_windows:
                    repeated_windows.update((number_windows[number], n))
                else:
                    number_windows[number] = n
        position_by_id = dict(zip(identities, positions))
        hands = []
        for message in messages:
            for row in message["source_rows"]:
                output[position_by_id[_identity(row)]]["edge_status"] = "PARTIAL_START"
        for n, begin in enumerate(start_indices):
            end = start_indices[n+1] if n+1 < len(start_indices) else len(messages)
            window = messages[begin:end]
            end_reason = "NEXT_CONFIRMED_HAND_START" if n+1 < len(start_indices) else _confirmed_end(window)
            # A conflicting numbered marker or a rejected explicit start inside
            # a window prevents us from silently combining different hands.
            numbers = {m["hand_number"] for m in window
                       if m["action"] in {"HAND_HEADER", "HAND_ID", "NEW_HAND"} and m["hand_number"]}
            epochs = {r.get("epoch") for m in window for r in m["source_rows"]}
            uncertain = (n in repeated_windows or len(numbers) > 1 or len(epochs) > 1 or
                         sum(m["action"] == "NEW_HAND" for m in window) > 1 or
                         any(m["action"] == "NEW_HAND" and not _safe(m) for m in window))
            status = ("UNRESOLVED_EDGE" if uncertain else
                      "KEEP" if end_reason else "PARTIAL_END")
            for message in window:
                for row in message["source_rows"]:
                    output[position_by_id[_identity(row)]]["edge_status"] = status
            hands.append({"start_frame": window[0]["frame_id"],
                          "start_row_id": _identity(window[0]["source_rows"][0]),
                          "hand_number_candidate": next(iter(numbers)) if len(numbers) == 1 else None,
                          "start_reason": starts[begin], "end_reason": end_reason,
                          "edge_status": status})
        counts = Counter(output[p]["edge_status"] for p in positions)
        summaries.append({"table_id": table, "session_id": session, "total_rows": len(source),
                          **{status: counts[status] for status in sorted(STATUSES)},
                          "first_confirmed_start_frame": hands[0]["start_frame"] if hands else None,
                          "keep_hands": sum(hand["edge_status"] == "KEEP" for hand in hands),
                          "last_started_hand": hands[-1] if hands else None,
                          "last_hand_end_proven": bool(hands and hands[-1]["end_reason"]),
                          "downstream_rows": counts["KEEP"], "hands": hands,
                          "input_order_identity_valid": valid})
    return EdgeResult(output, summaries)


def rows_for_downstream(classified_rows: list[dict]) -> list[dict]:
    """The single admission point before reconstruction/accounting/math."""
    if any(row.get("edge_status") not in STATUSES for row in classified_rows):
        raise ValueError("Downstream requires classified edge_status on every row")
    return [copy.deepcopy(row) for row in classified_rows if row["edge_status"] == "KEEP"]
