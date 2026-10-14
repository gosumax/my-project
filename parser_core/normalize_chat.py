"""Strict candidate extraction from Russian and English PokerDom chat text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from parser_core.table_service_events import describe_service
from parser_core.service_text_normalizer import normalize_service, ServiceText


NORMALIZER_VERSION = "candidate_bilingual_v8"
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

ENGLISH_ACTIONS = (
    (r'posts (?:the )?ante', 'ANTE', 'ANTE'),
    (r'posts (?:the )?small blind', 'SMALL_BLIND', 'SMALL_BLIND'),
    (r'posts (?:the )?big blind', 'BIG_BLIND', 'BIG_BLIND'),
    (r'raises to', 'RAISE', 'RAISE_TO'),
    (r'calls', 'CALL', 'CALL'), (r'checks', 'CHECK', None),
    (r'folds', 'FOLD', None), (r'bets', 'BET', 'BET'),
    (r'wins', 'WIN', 'WIN_PAYOUT'),
    (r'gets uncalled bet', 'UNCALLED_RETURN', 'UNCALLED_RETURN'),
    (r'does not show|doesn.t show', 'NO_SHOW', None),
    (r'shows', 'SHOW', None),
)


def english_candidate(text: str) -> Candidate | None:
    """Shared literal grammar; amount suffixes cannot supply extra numbers."""
    service = describe_service(text)
    if service and service['kind'] == 'REBUY':
        return Candidate('TABLE_INFO')
    header = re.match(r'^Hand\s*#\s*(\d+)', text, re.I)
    if header:
        return Candidate('HAND_HEADER', hand_number=header.group(1))
    new_hand = re.match(r'^Starting a new hand\b', text, re.I)
    if new_hand:
        number = NEW_HAND_NUMBER.fullmatch(text[new_hand.end():].strip())
        return Candidate('NEW_HAND', hand_number=number.group(1) if number else None)
    for pattern, kind in [(r'Dealing cards[.!●]?', 'DEAL'),
                          (r'(?:Dealing (?:the )?)?Flop\b.*', 'FLOP'),
                          (r'(?:Dealing (?:the )?)?Turn\b.*', 'TURN'),
                          (r'(?:Dealing (?:the )?)?River\b.*', 'RIVER'),
                          (r'Showdown[.!]?', 'SHOWDOWN'), (r'Hand end[.!]?', 'HAND_END')]:
        if re.fullmatch(pattern, text, re.I):
            return Candidate(kind)
    if re.fullmatch(r'(?:.+[ _](?:uses Time Bank|sits out|sits in|finished the tournament|has joined the table|has left the table)[.!]?|Waiting for .+ to(?: (?:reconnect|rebuy)\.{0,3})?|(?:reconnect|rebuy)\.{0,3}|.+ is reconnected[.!]?)', text, re.I):
        return Candidate('TABLE_INFO')
    if re.fullmatch(r'Limits going up: stakes(?:\s+\d+(?:[ \u00a0]\d{3})*/\d+(?:[ \u00a0]\d{3})*\s+ante\s+\d+(?:[ \u00a0]\d{3})*)?[.!]?', text, re.I):
        return Candidate('BLIND_LEVEL')
    matches = []
    for pattern, kind, role in ENGLISH_ACTIONS:
        boundary=r'(?![^\W\d])' if kind=='UNCALLED_RETURN' else r'(?!\w)'
        match = re.search(r'(?<![^\W_])(?:' + pattern + r')'+boundary, text, re.I)
        if match:
            matches.append((match, kind, role))
    if len([m for m, _, _ in matches if m.start() > 0]) > 1:
        return Candidate('UNKNOWN_EVENT', reason='MULTIPLE_ACTION_CLAUSES')
    for match, kind, role in matches:
        actor = text[:match.start()].strip(' :') or None
        reason = ('ACTOR_OCR_MISSING' if actor is None else
                  'ACTOR_OCR_AMBIGUOUS' if '?' in actor or '\ufffd' in actor else None)
        if role is None:
            return Candidate(kind, actor=actor, reason=reason)
        tail = text[match.end():].strip()
        number = AMOUNT.match(tail)
        amount = None
        if number:
            suffix = tail[number.end():].strip()
            if (not suffix or suffix in {'.', '!', '●'} or
                    re.fullmatch(r'(?:and\s*(?:is All[ -]?in)?|is All[ -]?in|back)\s*[.!●]?', suffix, re.I) or
                    kind == 'WIN' and re.match(r'from\b', suffix, re.I)):
                amount = number.group().replace(' ', '').replace('\u00a0', '')
        if amount is None:
            reason = ';'.join(filter(None, (reason, 'AMOUNT_AMBIGUOUS_OR_MISSING')))
        return Candidate(kind, actor=actor, amount_decimal=amount, amount_role=role, reason=reason)
    return None


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
    return normalize_with_service(raw_fragments)[0]


def normalize_with_service(raw_fragments: list[str | None], *, apply_service=True):
    """Return the existing candidate plus a separate derived service receipt."""
    # The chat viewport border is often OCR'd as a trailing vertical bar.
    text = " ".join(fragment.strip().rstrip("|").strip()
                    for fragment in raw_fragments if fragment).strip()
    service = normalize_service(text) if apply_service else ServiceText(text, text, ())
    if service.changes:
        originals = [fragment for fragment in raw_fragments if fragment]
        raw = ' '.join(originals)
        changes = tuple(dict(change, raw_ocr=raw, source_raw_fragments=originals,
                             prepared_service_input=text,
                             source_span_basis='PREPARED_SERVICE_INPUT')
                        for change in service.changes)
        service = ServiceText(raw, service.normalized_text, changes)
    return _normalize_prepared(service.normalized_text), service


def _normalize_prepared(text: str) -> Candidate:
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
    text = re.sub(r'^Dealer\s*:\s*', '', text, flags=re.I)
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
    english = english_candidate(text)
    if english is not None:
        return english
    return Candidate("UNKNOWN_EVENT", reason="TYPE_NOT_RECOGNIZED")
