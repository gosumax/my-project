"""Bounded alternatives for a malformed literal chip token, never OCR edits."""
import re
from dataclasses import replace

NUMBER = r'\d+(?:[ \u00a0]\d{3})*'
TOKEN = r'[0-9]+(?:[ \u00a0]+[0-9]+)*'


def numeric_alternatives(event):
    """Keep original digits and every one-adjacent-duplicate deletion.

    Only malformed digit grouping on an explicit money action is eligible.
    Correct numbers are never opportunistically replaced to close the pot.
    The caller must exhaust legal histories and prove economic uniqueness.
    """
    rows = event.source.get('rows', [])
    if event.amount is not None or event.kind not in {'RAISE', 'CALL', 'BET'} or not rows:
        return []
    if any(r.get('canonical_conflict') or r.get('canonical_duplicate_of') for r in rows):
        return []
    for a, b in zip(rows, rows[1:]):
        if (a.get('frame_id') is None or a.get('frame_id') != b.get('frame_id') or
                not isinstance(a.get('row_index'), int) or b.get('row_index') != a['row_index'] + 1 or
                any(a.get(k) != b.get(k) for k in ('table_id', 'session_id', 'epoch'))):
            return []
    text = event.source.get('raw_text', '')
    match = re.fullmatch(r'(?:Dealer\s*:\s*)?(.+?)\s+(raises to|calls|bets)\s+(' + TOKEN +
                         r')(?:\s+(and is All[ -]?in|is All[ -]?in))?[.!●]?', text.strip(), re.I)
    if not match or match[1] != event.player:
        return []
    token = match[3]
    if re.fullmatch(NUMBER, token):
        return []
    digits = re.sub(r'[ \u00a0]', '', token)
    if len(digits) > 12 or digits.startswith('0'):
        return []
    original = int(digits)
    values = {original}
    for i in range(1, len(digits)):
        if digits[i] == digits[i - 1]:
            candidate = int(digits[:i] + digits[i + 1:])
            if candidate > 0:
                values.add(candidate)
    return [replace(event, amount=value, source=dict(event.source, numeric_amount_proof=dict(
        raw_amount_text=token, parsed_amount=original, resolved_amount=value,
        candidates=sorted(values), rule='MALFORMED_GROUPING_ONE_ADJACENT_DUPLICATE_DIGIT',
        resolution_reason='ORIGINAL_DIGITS' if value == original else 'CANDIDATE_DUPLICATE_DIGIT_DELETION',
        source='OCR' if value == original else 'POKER_MATH_INFERENCE',
        source_row_ids=[r.get('stable_row_id', r.get('row_id')) for r in rows],
        source_pixel_hashes=[r.get('row_pixel_sha256') for r in rows]))) for value in sorted(values)]
