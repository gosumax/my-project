"""Conservative grouping of physical chat rows into candidate messages."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


ASSEMBLER_VERSION = "message_assembly_v13_tiny_only"
TIMESTAMP = re.compile(r"^\s*\[\d{1,2}:\d{2}\]")
HAND_HEADER = re.compile(r"^\s*(?:Рука\s*#|#)\s*\d", re.IGNORECASE)
DEALER_PREFIX = re.compile(r"^\s*Дилер\s*:", re.IGNORECASE)
BARE_HAND_NUMBER = re.compile(r"^\s*#\s*\d+\s*$")
NEW_HAND_WRAP_START = re.compile(
    r"^\s*(?:Дилер\s*:\s*|\[\d{1,2}:\d{2}\]\s*)"
    r"Начинается\s+новая\s+рука[.!]?\s*$", re.IGNORECASE)


def has_text(text: str | None) -> bool:
    return bool(text and text.strip())


@dataclass(frozen=True)
class RowFact:
    row_id: str
    session_id: str
    epoch: str
    frame_id: int
    y1: int
    text: str | None
    recognition_id: str | None
    right_margin: int | None
    origin_state: str = "TRACKED"
    ocr_retry: bool = False
    vl_nickname_fallback: bool = False
    vl_board_fallback: bool = False


@dataclass
class MessageCandidate:
    rows: list[RowFact] = field(default_factory=list)
    start_status: str = "UNRESOLVED_START"
    end_status: str = "SOURCE_PARTIAL"
    inferred_continuation: bool = False

    @property
    def completeness(self) -> str:
        if (self.start_status == "VISIBLE_START" and
                self.end_status == "NEXT_START_OBSERVED" and
                not self.inferred_continuation and all(has_text(row.text) for row in self.rows) and
                not any(row.ocr_retry for row in self.rows) and
                not any(row.vl_nickname_fallback for row in self.rows) and
                not any(row.vl_board_fallback for row in self.rows) and
                all(row.origin_state != "UNRESOLVED_IDENTITY" for row in self.rows)):
            return "CANDIDATE"
        return "PARTIAL"

    @property
    def reason(self) -> str | None:
        reasons = []
        if self.start_status != "VISIBLE_START":
            reasons.append(self.start_status)
        if self.end_status != "NEXT_START_OBSERVED":
            reasons.append(self.end_status)
        if self.inferred_continuation:
            reasons.append("CONTINUATION_WITHOUT_VISIBLE_START")
        if not all(has_text(row.text) for row in self.rows):
            reasons.append("OCR_MISSING")
        if any(row.ocr_retry for row in self.rows):
            reasons.append("OCR_RETRY_UNVERIFIED")
        if any(row.vl_nickname_fallback for row in self.rows):
            reasons.append("VL_NICKNAME_UNVERIFIED")
        if any(row.vl_board_fallback for row in self.rows):
            reasons.append("VL_BOARD_UNVERIFIED")
        if any(row.origin_state == "UNRESOLVED_IDENTITY" for row in self.rows):
            reasons.append("ROW_IDENTITY_UNRESOLVED")
        return ";".join(reasons) or None


def visible_start(text: str | None) -> bool:
    if not text:
        return False
    return bool(TIMESTAMP.match(text) or HAND_HEADER.match(text) or
                DEALER_PREFIX.match(text))


def group_rows(rows: list[RowFact]) -> list[MessageCandidate]:
    messages: list[MessageCandidate] = []
    active: MessageCandidate | None = None
    for row in rows:
        explicit = visible_start(row.text)
        # A bare number immediately following a new-hand start is its wrapped
        # continuation. A standalone number or a different epoch stays distinct.
        if (active is not None and len(active.rows) == 1 and
                row.session_id == active.rows[0].session_id and
                row.epoch == active.rows[0].epoch and
                NEW_HAND_WRAP_START.fullmatch(active.rows[0].text or "") and
                BARE_HAND_NUMBER.fullmatch(row.text or "")):
            explicit = False
        if active is None:
            active = MessageCandidate([row], "VISIBLE_START" if explicit else "UNRESOLVED_START")
            continue
        prior = active.rows[-1]
        # A row without a visible prefix continues only within the same chat epoch.
        # Its relationship is inferred even when the right edge has free space.
        if has_text(row.text) and not explicit and row.session_id == prior.session_id and row.epoch == prior.epoch:
            active.rows.append(row)
            active.inferred_continuation = True
            continue
        active.end_status = "NEXT_START_OBSERVED" if explicit else "NEXT_ROW_UNRESOLVED"
        messages.append(active)
        active = MessageCandidate([row], "VISIBLE_START" if explicit else "UNRESOLVED_START")
    if active is not None:
        messages.append(active)
    return messages
