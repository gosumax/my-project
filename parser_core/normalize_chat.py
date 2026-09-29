"""Strict candidate extraction from Russian PokerDom chat text."""

from __future__ import annotations

import re
from dataclasses import dataclass


NORMALIZER_VERSION = "candidate_ru_v7"
TIME_PREFIX = re.compile(r"^\s*\[\d{1,2}:\d{2}\]\s*")
HAND_HEADER = re.compile(r"^\s*Рука\s*#\s*(\d+)", re.IGNORECASE)
HAND_ID = re.compile(r"^\s*#\s*(\d+)")
NEW_HAND_NUMBER = re.compile(r"^[.!]?\s*#\s*(\d+)\s*[.!]?\s*$")
AMOUNT = re.compile(r"(?<!\d)(\d+(?:[ \u00a0]\d{3})*)(?!\d)")
BLIND_ACTIONS = (
    (re.compile(r"\bставит\s+малый\W+блайнд(?=\d|\W|$)", re.IGNORECASE),
     "SMALL_BLIND"),
    (re.compile(r"\bставит\s+большой\W+блайнд(?=\d|\W|$)", re.IGNORECASE),
     "BIG_BLIND"),
)
NEGATED_SHOW = re.compile(r"\bне\s+показывает\b", re.IGNORECASE)
PLAYER_ACTIONS = (
    ("ставит анте", "ANTE", "ANTE"),
    ("повышает до", "RAISE", "RAISE_TO"),
    ("уравнивает", "CALL", "CALL"),
    ("сбрасывает", "FOLD", None),
    ("делает чек", "CHECK", None),
    ("проверяет", "CHECK", None),
    ("выигрывает", "WIN", "WIN_PAYOUT"),
    ("показывает", "SHOW", None),
    ("ставит", "BET", "BET"),
)
SYSTEM = (
    ("начинается новая рука", "NEW_HAND"),
    ("раздаются карты", "DEAL"),
    ("раздаётся флоп", "FLOP"),
    ("раздаётся терн", "TURN"),
    ("раздаётся ривер", "RIVER"),
)


@dataclass(frozen=True)
class Candidate:
    event_type: str
    actor: str | None = None
    amount_decimal: str | None = None
    amount_role: str | None = None
    hand_number: str | None = None
    reason: str | None = None


def _player_candidate(text: str, start: int, end: int,
                      event_type: str, role: str | None) -> Candidate:
    actor = text[:start].strip(" @©) |:;") or None
    actor_reason = ("ACTOR_OCR_MISSING" if actor is None else
                    "ACTOR_OCR_UNREADABLE" if not any(character.isalnum() for character in actor)
                    else "ACTOR_OCR_AMBIGUOUS" if "?" in actor or "\ufffd" in actor
                    else None)
    tail = text[end:]
    if role is None:
        return Candidate(event_type, actor=actor, reason=actor_reason)
    matches = AMOUNT.findall(tail)
    if len(matches) != 1:
        return Candidate(event_type, actor=actor, amount_role=role,
                         reason=";".join(filter(None, (actor_reason,
                              "AMOUNT_AMBIGUOUS_OR_MISSING"))))
    amount = matches[0].replace(" ", "").replace("\u00a0", "")
    return Candidate(event_type, actor=actor, amount_decimal=amount,
                     amount_role=role, reason=actor_reason)


def normalize(raw_fragments: list[str | None]) -> Candidate:
    # The chat viewport border is often OCR'd as a trailing vertical bar.
    text = " ".join(fragment.strip().rstrip("|").strip()
                    for fragment in raw_fragments if fragment).strip()
    if not text:
        return Candidate("UNKNOWN_EVENT", reason="OCR_MISSING")
    header = HAND_HEADER.match(text)
    if header:
        return Candidate("HAND_HEADER", hand_number=header.group(1))
    hand_id = HAND_ID.match(text)
    if hand_id:
        return Candidate("HAND_ID", hand_number=hand_id.group(1))
    text = TIME_PREFIX.sub("", text)
    text = re.sub(r"^[^\w#]+", "", text, flags=re.UNICODE).strip()
    if text.casefold().startswith("дилер:"):
        text = text[len("дилер:"):].strip()
    lower = text.casefold()
    for phrase, event_type in SYSTEM:
        if phrase in lower:
            if event_type == "NEW_HAND":
                tail = text[lower.find(phrase) + len(phrase):].strip()
                number = NEW_HAND_NUMBER.fullmatch(tail)
                return Candidate(event_type, hand_number=number.group(1) if number else None)
            return Candidate(event_type)
    for pattern, event_type in BLIND_ACTIONS:
        match = pattern.search(text)
        if match:
            return _player_candidate(text, match.start(), match.end(),
                                     event_type, event_type)
    # Negation belongs to the action, not to the nickname. Check it before
    # the positive "показывает" phrase, including a wrap between the words.
    negative_show = NEGATED_SHOW.search(text)
    if negative_show:
        return _player_candidate(text, negative_show.start(), negative_show.end(),
                                 "NO_SHOW", None)
    for phrase, event_type, role in PLAYER_ACTIONS:
        position = lower.find(phrase)
        if position < 0:
            continue
        if event_type == "BET" and re.match(r"\s+(?:малый|большой)\b",
                                             text[position + len(phrase):],
                                             re.IGNORECASE):
            return Candidate("UNKNOWN_EVENT", reason="BLIND_TYPE_UNREADABLE")
        return _player_candidate(text, position, position + len(phrase),
                                 event_type, role)
    return Candidate("UNKNOWN_EVENT", reason="TYPE_NOT_RECOGNIZED")
