"""Exact saved-corpus service variants; RAW pixels/text never change.

This is a derived CHAT view before the existing grammar. Numeric/player/card
data cannot be corrected here. Unknown variants remain literal unknowns.
"""
from dataclasses import dataclass
import re

VERSION = 'POKERDOM_SERVICE_EXACT_V1'
ALL_IN_VARIANTS = ('All-n', 'Al-in', 'All-in0', 'Al-in0', 'Allin')
_VARIANTS = ALL_IN_VARIANTS + ('rebuy...0', 'table0', 'potO')
_NUMBER = r'\d+(?:[ \u00a0]\d{3})*'
_ACTION = r'(?:bets|calls|raises to|posts (?:the )?(?:Ante|Small Blind|Big Blind))'
_ALL_IN = re.compile(
    r'^.+?\s+' + _ACTION + r'\s+' + _NUMBER +
    r'\s+(?:and\s+)?is\s+(?P<token>' + '|'.join(map(re.escape, ALL_IN_VARIANTS)) +
    r')(?P<end>[.!●]?)$')
_REBUY = re.compile(r'^(?:Dealer:\s*)?Waiting for .+ to\s+(?P<token>rebuy\.\.\.0)$')
_TABLE = re.compile(r'^.+ (?:has joined the|has left the)\s+(?P<token>table0)(?P<end>[.!]?)$')
_TIME_BANK_UI = re.compile(
    r'^(?:\[\d{1,2}:\d{2}\]\s*)?(?:(?:Dealer|Дилер)\s*[:;]\s*)?'
    r'[^\s:#;]+[ _]uses Time Bank[.!]?(?P<token>●)$', re.I)
_WIN_MAIN_POT = re.compile(
    r'^(?:\[\d{1,2}:\d{2}\]\s*)?(?:(?:Dealer|Дилер)\s*[:;]\s*)?'
    r'[^\s:#;]+ wins ' + _NUMBER + r' from (?:the )?main (?P<token>potO)$')
_RULES = ((_ALL_IN, 'All-in', 'SERVICE_ALL_IN_EXACT_SUFFIX_V1'),
          (_REBUY, 'rebuy...', 'SERVICE_REBUY_EXACT_SUFFIX_V1'),
          (_TABLE, 'table', 'SERVICE_TABLE_EXACT_SUFFIX_V1'),
          (_TIME_BANK_UI, '', 'SERVICE_TIME_BANK_TERMINAL_UI_MARKER_V1'),
          (_WIN_MAIN_POT, 'pot', 'SERVICE_WIN_MAIN_POT_TERMINAL_SPELLING_V1'))


@dataclass(frozen=True)
class ServiceText:
    raw_ocr: str
    normalized_text: str
    changes: tuple


def normalize_service(raw_ocr: str, *, previous_clause: str | None = None) -> ServiceText:
    """Canonicalize only a whole terminal service token in a known clause.

The previous clause is usable only after the caller verifies source adjacency.
Standalone strings cannot acquire a service interpretation through a label.
"""
    time_bank_marker = raw_ocr.rstrip().endswith('●') and 'time bank' in raw_ocr.casefold()
    if not time_bank_marker and not any(variant in raw_ocr for variant in _VARIANTS):
        return ServiceText(raw_ocr, raw_ocr, ())
    stripped = raw_ocr.strip()
    offset = len(raw_ocr) - len(raw_ocr.lstrip())
    candidates = [(stripped, offset, 'COMPLETE_SERVICE_CLAUSE')]
    if previous_clause:
        prefix = previous_clause.strip() + ' '
        candidates.append((prefix + stripped, offset - len(prefix), 'PHYSICAL_ADJACENT_SERVICE_CONTINUATION'))
    for text, base, context in candidates:
        for pattern, canonical, rule in _RULES:
            match = pattern.fullmatch(text)
            if not match:
                continue
            start, end = match.span('token')
            start += base; end += base
            if start < 0 or end > len(raw_ocr):
                continue  # A current fragment cannot rewrite its parent row.
            variant = raw_ocr[start:end]
            change = dict(raw_ocr=raw_ocr, canonical_token=canonical,
                          observed_variant=variant, rule_id=rule,
                          source_span=[start, end], context=context)
            normalized = raw_ocr[:start] + canonical + raw_ocr[end:]
            change['normalized_text'] = normalized
            return ServiceText(raw_ocr, normalized, (change,))
    return ServiceText(raw_ocr, raw_ocr, ())


def same_physical_neighbor(previous: dict, current: dict) -> bool:
    """A new spelling rule cannot grant cross-frame/occupant/hand joins."""
    session = lambda row: row.get('session_id', row.get('table_session_id'))
    for left, right in ((previous.get('table_id'), current.get('table_id')),
                        (session(previous), session(current)),
                        (previous.get('epoch'), current.get('epoch')),
                        (previous.get('frame_id'), current.get('frame_id'))):
        if left is None or right is None or left != right:
            return False
    a, b = previous.get('row_index', previous.get('source_span_index')), current.get('row_index', current.get('source_span_index'))
    if not isinstance(a, int) or not isinstance(b, int) or b != a + 1:
        return False
    for field in ('hand_id', 'hand_number', 'source_file'):
        if previous.get(field) is not None and current.get(field) is not None and previous[field] != current[field]:
            return False
    return True


def physical_service_group(rows) -> bool:
    """Every parent fragment must belong to the same adjacent source group."""
    return all(same_physical_neighbor(a, b) for a, b in zip(rows, rows[1:]))


def literal_win_pot_continuation(rows) -> bool:
    """One known two-row WIN spelling; grants no pixel/card/UI authority."""
    if len(rows) != 2 or not physical_service_group(rows):
        return False
    text = lambda row: str(row.get('raw_ocr_text') or row.get('raw_text') or row.get('text') or '').strip()
    if text(rows[1]) != 'main potO':
        return False
    normalized = normalize_service(text(rows[0]) + ' ' + text(rows[1]))
    return any(c['rule_id'] == 'SERVICE_WIN_MAIN_POT_TERMINAL_SPELLING_V1' for c in normalized.changes)
