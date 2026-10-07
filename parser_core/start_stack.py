"""Fail-closed, per-seat PokerDom START STACK reconstruction.

The first readable numeric stack is post-forced-bets. ANTE/SB/BB are always
added back when their actor and amount are proven. Voluntary debits still need
independent timing evidence relative to that stack observation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation


OVERLAYS = {"ANTE", "SB", "BB"}
FORCED_EVENTS = {"ANTE", "SMALL_BLIND", "BIG_BLIND"}
ROLE_OVERLAY = {"NORMAL": "ANTE", "SB": "SB", "BB": "BB"}
NUMBER = re.compile(r"^\s*(\d{1,3}(?:[ \u00a0]\d{3})+|\d+)\s*$")


def money(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        amount = Decimal(str(value).replace(" ", "").replace("\u00a0", ""))
    except InvalidOperation:
        return None
    return amount if amount.is_finite() and amount >= 0 else None


def classify_stack_text(raw: str | None) -> tuple[str, Decimal | None]:
    """Accept only an exact overlay token or an unambiguous whole-number stack."""
    value = (raw or "").strip().upper()
    if value in OVERLAYS:
        return value, None
    if NUMBER.fullmatch(value):
        return "STACK", money(value)
    return "UNREADABLE", None


@dataclass(frozen=True)
class StackReading:
    table_id: str
    table_context: str
    seat: str
    frame_index: int
    source_pts: int | None
    timestamp: float | None
    crop_sha256: str
    crop_path: str | None
    raw_recognition: str | None
    confidence: float | None
    recognition_status: str = "OK"


@dataclass(frozen=True)
class ChatAction:
    player: str
    event_type: str
    amount: object = None
    # Applies to voluntary actions. PokerDom forced posts are postulated to
    # precede the first readable numeric stack, independent of OCR row time.
    before_stack: bool | None = None
    source_ref: str | None = None
    evidence_status: str = "PROVEN"


def contribution_before_stack(actions: list[ChatAction], player: str) -> tuple[Decimal | None, str | None]:
    """Add forced posts regardless of OCR frame; time voluntary debits."""
    total = Decimal(0)
    street = Decimal(0)
    for action in actions:
        if action.event_type in {"FLOP", "TURN", "RIVER"}:
            if action.before_stack is None or action.evidence_status != "PROVEN":
                return None, "CONTRIBUTION_AMBIGUOUS"
            if action.before_stack:
                street = Decimal(0)
            continue
        if action.player != player:
            continue
        kind = action.event_type
        if action.evidence_status != "PROVEN":
            return None, "CONTRIBUTION_AMBIGUOUS"
        if kind not in FORCED_EVENTS and action.before_stack is None:
            return None, "CONTRIBUTION_AMBIGUOUS"
        if kind not in FORCED_EVENTS and not action.before_stack:
            continue
        if kind in {"FOLD", "CHECK", "SHOW", "NO_SHOW"}:
            continue
        amount = money(action.amount)
        if amount is None:
            return None, "CONTRIBUTION_AMBIGUOUS"
        if kind == "ANTE":
            total += amount
        elif kind in {"SMALL_BLIND", "BIG_BLIND", "BET", "CALL"}:
            total += amount
            street += amount
        elif kind in {"RAISE_TO", "RAISE"}:
            additional = amount - street
            if additional < 0:
                return None, "CONTRIBUTION_AMBIGUOUS"
            total += additional
            street = amount
        else:
            return None, "CONTRIBUTION_AMBIGUOUS"
    return total, None


@dataclass
class SeatState:
    nickname: str | None
    role: str | None
    identity_proven: bool
    ante_policy: str = "UNKNOWN"
    phase: str = "WAIT_FOR_FIRST_NUMERIC_STACK"
    overlay_before_stack: str | None = None
    reading: StackReading | None = None
    visible_stack: Decimal | None = None
    reason: str | None = None
    observations: int = 0


@dataclass
class HandStackTracker:
    table_id: str
    table_context: str
    hand_id: str
    start_frame: int
    partial_start: bool
    seats: dict[str, SeatState] = field(default_factory=dict)

    def bind_seat(self, seat: str, nickname: str | None, role: str | None,
                  *, identity_proven: bool, ante_policy: str = "UNKNOWN") -> None:
        if seat in self.seats:
            raise ValueError("Seat binding cannot change inside a hand")
        if ante_policy not in {"UNKNOWN", "REQUIRED", "ABSENT"}:
            raise ValueError("Unknown ante policy")
        self.seats[seat] = SeatState(nickname, role, identity_proven, ante_policy)

    def observe(self, reading: StackReading) -> None:
        if (reading.table_id != self.table_id or
                reading.table_context != self.table_context or
                reading.frame_index < self.start_frame or reading.seat not in self.seats):
            raise ValueError("Stack observation crosses a hand, seat or table context")
        state = self.seats[reading.seat]
        if state.phase == "STACK_CAPTURED":
            return
        state.observations += 1
        if reading.recognition_status != "OK":
            state.reason = "STACK_OCR_FAILED"
            return
        kind, value = classify_stack_text(reading.raw_recognition)
        expected = ROLE_OVERLAY.get(state.role)
        if kind in OVERLAYS:
            if reading.confidence is None or reading.confidence < 0.90:
                state.reason = "STACK_OCR_FAILED"
                return
            if expected is not None and kind != expected:
                state.reason = "ROLE_OVERLAY_CONFLICT"
                return
            state.overlay_before_stack = kind
            return
        if kind == "STACK":
            if reading.confidence is None or reading.confidence < 0.90:
                state.reason = "STACK_OCR_FAILED"
                return
            state.reading = reading
            state.visible_stack = value
            state.phase = "STACK_CAPTURED"
            state.reason = None
        elif kind == "UNREADABLE":
            state.reason = "STACK_OCR_FAILED"

    def result(self, seat: str, actions: list[ChatAction], *,
               chat_complete: bool, timing_proven: bool) -> dict:
        state = self.seats[seat]
        reading = state.reading
        base = {"table_id": self.table_id, "table_context": self.table_context,
                "hand_id": self.hand_id, "seat": seat, "nickname": state.nickname,
                "role": state.role, "overlay_before_stack": state.overlay_before_stack,
                "stack_frame": reading.frame_index if reading else None,
                "stack_source_pts": reading.source_pts if reading else None,
                "stack_timestamp": reading.timestamp if reading else None,
                "raw_visible_stack": str(state.visible_stack) if state.visible_stack is not None else None,
                "crop_path": reading.crop_path if reading else None,
                "crop_sha256": reading.crop_sha256 if reading else None,
                "raw_recognition": reading.raw_recognition if reading else None,
                "confidence": reading.confidence if reading else None,
                "ante_amount": None, "sb_amount": None, "bb_amount": None,
                "player_contribution_before_stack_frame": None,
                "start_stack": None, "status": "UNKNOWN", "reason": None}
        if self.partial_start:
            base["reason"] = "PARTIAL_START"
            return base
        if not state.identity_proven or not state.nickname:
            base["reason"] = "PLAYER_IDENTITY_UNKNOWN"
            return base
        if state.role not in ROLE_OVERLAY:
            base["reason"] = "ROLE_UNKNOWN"
            return base
        if state.reason == "ROLE_OVERLAY_CONFLICT":
            base["reason"] = state.reason
            return base
        if reading is None:
            base["reason"] = state.reason or "STACK_NEVER_VISIBLE"
            return base
        if not chat_complete or not timing_proven:
            base["reason"] = "CONTRIBUTION_AMBIGUOUS"
            return base
        contribution, reason = contribution_before_stack(actions, state.nickname)
        if reason:
            base["reason"] = reason
            return base
        assert contribution is not None and state.visible_stack is not None
        forced = {}
        for event_type, key in (("ANTE", "ante_amount"),
                                ("SMALL_BLIND", "sb_amount"),
                                ("BIG_BLIND", "bb_amount")):
            found = [money(action.amount) for action in actions
                     if action.player == state.nickname and action.event_type == event_type
                     and action.evidence_status == "PROVEN"]
            if any(value is None for value in found) or len(found) > 1:
                base["reason"] = "CONTRIBUTION_AMBIGUOUS"
                return base
            forced[key] = found[0] if found else None
            base[key] = str(forced[key]) if forced[key] is not None else None
        if ((forced["ante_amount"] is None and state.ante_policy != "ABSENT") or
                (forced["ante_amount"] is not None and state.ante_policy == "ABSENT") or
                (state.role == "SB" and forced["sb_amount"] is None) or
                (state.role == "BB" and forced["bb_amount"] is None) or
                (state.role == "NORMAL" and
                 (forced["sb_amount"] is not None or forced["bb_amount"] is not None))):
            base["reason"] = "CONTRIBUTION_AMBIGUOUS"
            return base
        start = state.visible_stack + contribution
        if start < state.visible_stack or start - state.visible_stack != contribution:
            base["reason"] = "CONTRIBUTION_MISMATCH"
            return base
        base["player_contribution_before_stack_frame"] = str(contribution)
        base["start_stack"] = str(start)
        base["status"] = "CERTIFIED"
        return base
