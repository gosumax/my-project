"""Deterministic source-preserving downstream production hand analysis.

No external resolver is invoked and no monetary semantics are inferred.
Validated hand candidates are distinct from certified/exported hand histories.
"""
from __future__ import annotations
import copy
import time
from collections import Counter
from parser_core.poker_hand_assembly import assemble_hands
from parser_core.poker_logic_engine import validate_hand
from parser_core.poker_diagnostics import diagnose_hand


def analyze_timeline(timeline, *, ocr_status, join_show_description=False):
    started = time.perf_counter()
    rows = []
    for original in timeline:
        row = copy.deepcopy(original)
        row['row_id'] = row['stable_row_id']
        row['raw_text'] = row['raw_ocr_text']
        stamp = row.get('video_time')
        row['timestamp_ms'] = stamp * 1000 if stamp is not None else None
        row['ocr_status'] = ocr_status.get(row['stable_row_id'],
            row.get('ocr_status', 'ERROR') if row.get('canonical_stream_version') else 'ERROR')
        rows.append(row)
    hands = assemble_hands(rows, require_pixel_cards=True, require_adjacency=True,
                           pixel_sources={row['row_id']: row for row in rows},
                           join_show_description=join_show_description)
    observed_ids = [row['row_id'] for row in rows]
    covered_ids = [rid for hand in hands for msg in hand['messages'] for rid in msg['source_row_ids']]
    if Counter(covered_ids) != Counter(observed_ids) or len(set(observed_ids)) != len(observed_ids):
        raise ValueError('Hand assembly lost, duplicated or invented source rows')
    records = []
    for index, hand in enumerate(hands, 1):
        checked = validate_hand(hand)
        diagnostic = diagnose_hand(hand)
        diagnostic['raw_final_status'] = diagnostic['final_status']
        ledger = diagnostic['math']
        diagnostic['raw_math_status'] = diagnostic['math_status']
        if (not ledger.get('conservation_test_applicable') or
                ledger.get('ledger_certainty') != 'OBSERVED_COMPLETE' or
                not hand['start_observed'] or not hand['end_observed']):
            diagnostic['math_status'] = 'NOT_VERIFIABLE'
            ledger['conservation_conclusion'] = 'NOT_PROVEN_INCOMPLETE_OR_UNCERTAIN_LEDGER'
            if diagnostic['final_status'] == 'MATH_FAIL':
                diagnostic['final_status'] = 'UNRESOLVED'
        else:
            ledger['conservation_conclusion'] = 'PASS' if ledger['pass'] else 'FAIL'
        fully = (checked['status'] == 'FULLY_VALID_HAND' and
                 diagnostic['final_status'] == 'FULLY_VALID' and
                 checked['complete_candidate'] and not checked['certification_blockers'])
        records.append({'segment_id': f"{hand['table_id']}:{hand['session_id']}:S{index:06d}",
                        'observed_hand': hand, 'validation': checked, 'diagnostics': diagnostic,
                        'fully_validated': bool(fully), 'certified_hh': False})
    summary = {
        'source_rows': len(rows), 'source_rows_covered': len(covered_ids),
        'assembled_segments': len(hands),
        'numbered_segments': sum(h['hand_id'] is not None for h in hands),
        'unique_numbered_hands': len({(h['table_id'], h['session_id'], h['hand_id'])
                                     for h in hands if h['hand_id'] is not None}),
        'observed_start_segments': sum(bool(h['start_observed']) for h in hands),
        'observed_end_segments': sum(bool(h['end_observed']) for h in hands),
        'complete_candidates': sum(bool(r['validation']['complete_candidate']) for r in records),
        'fully_validated_hands': sum(r['fully_validated'] for r in records),
        'analysis_status': ('NO_HANDS' if not records else
                            'ALL_VALIDATED' if all(r['fully_validated'] for r in records)
                            else 'PARTIAL_UNRESOLVED'),
        'certified_hh': 0,
        'validation_statuses': dict(Counter(r['validation']['status'] for r in records)),
        'logic_statuses': dict(Counter(r['diagnostics']['logic_status'] for r in records)),
        'math_statuses': dict(Counter(r['diagnostics']['math_status'] for r in records)),
        'card_statuses': dict(Counter(r['diagnostics']['card_status'] for r in records)),
        'first_breaks': dict(Counter(r['diagnostics']['first_break']['code'] for r in records
                                  if r['diagnostics']['first_break'])),
        'blocking_issues': dict(Counter(i['code'] for r in records
                                       for i in r['validation']['certification_blockers'])),
        'monetary_semantics': 'UNPROVEN_UNLESS_EXPLICIT_SOURCE_CONTRACT',
        'card_authority': 'VERIFIED_PIXEL_PROJECTION_ONLY',
        'resolver': 'DISABLED_NO_EXTERNAL_HYPOTHESES',
        'wall_seconds': time.perf_counter() - started,
    }
    return records, summary
