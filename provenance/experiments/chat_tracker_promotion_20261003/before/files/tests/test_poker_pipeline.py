"""Production integration contracts: no text cards or invented continuations."""
import copy
import json
from pathlib import Path
import pytest
from parser_core.poker_hand_assembly import assemble_messages, make_message
from parser_core.poker_pipeline import analyze_timeline


def rows(*texts):
    return [{'stable_row_id': f's:R{i:06d}', 'row_id': f's:R{i:06d}',
             'table_id': 'T01', 'session_id': 's', 'frame_id': 0,
             'row_index': i, 'epoch': 1, 'origin': 'NEW',
             'raw_ocr_text': text, 'status': 'RAW_UNREVIEWED',
             'video_time': 0, 'row_pixel_sha256': 'a' * 64}
            for i, text in enumerate(texts)]


def analyze(source, errors=()):
    return analyze_timeline(source, ocr_status={r['stable_row_id']:
        'ERROR' if r['stable_row_id'] in errors else 'RAW_UNREVIEWED' for r in source})


def test_strict_production_does_not_use_ocr_card_facts():
    source = rows('Dealer: Dealing Flop 2h 3d 4s')
    msg = make_message(source, require_pixel_cards=True)
    assert msg['action'] == 'FLOP' and msg['cards'] == []
    assert msg['source_rows'] == source and msg['raw_text'] == source[0]['raw_ocr_text']


@pytest.mark.parametrize('gap', ['frame', 'span', 'session'])
def test_ambiguous_numeric_continuation_not_joined(gap):
    source = rows('Dealer: A calls', '5')
    if gap == 'frame':
        source[1]['frame_id'] = 1000
    elif gap == 'span':
        source[1]['row_index'] = 3
    else:
        source[1]['session_id'] = 'other'
    result = assemble_messages(source, require_pixel_cards=True, require_adjacency=True)
    assert len(result) == 2 and result[0]['amount'] is None


def test_proven_adjacent_amount_continuation_keeps_both_sources():
    source = rows('Dealer: A calls', '5')
    result = assemble_messages(source, require_pixel_cards=True, require_adjacency=True)
    assert len(result) == 1 and result[0]['amount'] == '5'
    assert result[0]['source_row_ids'] == [r['stable_row_id'] for r in source]


def test_full_fold_hand_runs_all_engines_without_altering_timeline():
    source = rows('Hand #123456', 'Dealer: Starting a new hand',
                  'Dealer: A posts Small Blind 5', 'Dealer: B posts Big Blind 10',
                  'Dealer: A folds', 'Dealer: B gets uncalled bet 5 back', 'Dealer: B wins 10')
    original = copy.deepcopy(source)
    records, summary = analyze(source)
    assert source == original and summary['source_rows_covered'] == 7
    assert summary['assembled_segments'] == summary['fully_validated_hands'] == 1
    assert summary['certified_hh'] == 0
    assert records[0]['diagnostics']['math_status'] == 'PASS'


def test_failed_ocr_with_valid_text_blocks_full_hand():
    source = rows('Hand #123456', 'Dealer: Starting a new hand',
                  'Dealer: A posts Small Blind 5', 'Dealer: B posts Big Blind 10',
                  'Dealer: A folds', 'Dealer: B gets uncalled bet 5 back', 'Dealer: B wins 10')
    records, summary = analyze(source, [source[2]['stable_row_id']])
    assert summary['fully_validated_hands'] == 0
    assert 'OCR_EVIDENCE_ERROR' in summary['blocking_issues']
    assert records[0]['diagnostics']['math_status'] == 'NOT_VERIFIABLE'


def test_partial_ledger_delta_is_not_a_proved_conservation_failure():
    records, summary = analyze(rows('Hand #123456', 'Dealer: A posts Small Blind 5'))
    assert summary['fully_validated_hands'] == 0
    assert records[0]['diagnostics']['math_status'] == 'NOT_VERIFIABLE'
    assert not records[0]['diagnostics']['math']['conservation_test_applicable']


def test_unproved_math_failure_is_reported_as_unresolved():
    source = rows('Hand #123456', 'Dealer: Starting a new hand',
                  'Dealer: A posts Small Blind 5', 'Dealer: B posts Big Blind 10',
                  'Dealer: A folds', 'Dealer: B gets uncalled bet 5 back', 'Dealer: B wins 11')
    source[2]['origin'] = 'UNRESOLVED_IDENTITY'
    records, summary = analyze(source)
    diagnostic = records[0]['diagnostics']
    assert diagnostic['math_status'] == 'NOT_VERIFIABLE'
    assert diagnostic['final_status'] != 'MATH_FAIL'
    assert summary['fully_validated_hands'] == 0


def test_existing_pixel_projection_binds_its_exact_source_row():
    path = Path('runs/production_temporal_smoke18_full_candidate_20261002/ordered_timeline.jsonl')
    source = [json.loads(line) for line in path.read_text('utf-8').splitlines()]
    trigger = next(r for r in source if r.get('cards') and r.get('card_row_id'))
    physical = next(r for r in source if r['stable_row_id'] == trigger['card_row_id'])
    pair = copy.deepcopy([trigger, physical])
    for row in pair:
        row['row_id'] = row['stable_row_id']
    result = make_message(pair, require_pixel_cards=True)
    assert result['cards'] == trigger['cards']
    assert result['field_sources']['cards'] == 'OBSERVED_PIXEL'
    pair[1]['row_pixel_sha256'] = 'b' * 64
    rejected = make_message(pair, require_pixel_cards=True)
    assert rejected['cards'] == [] and rejected['card_binding_conflict']


def test_full_pipeline_binds_offset_two_without_merging_intervening_text():
    path = Path('runs/production_temporal_smoke18_full_candidate_20261002/ordered_timeline.jsonl')
    source = [json.loads(line) for line in path.read_text('utf-8').splitlines()]
    records, summary = analyze(source)
    target = next(r for r in source if r['table_id'] == 'T04' and r['frame_id'] == 576
                  and r['row_index'] == 12)
    message = next(m for r in records for m in r['observed_hand']['messages']
                   if target['stable_row_id'] in m['source_row_ids'])
    assert message['cards'] == target['cards'] and not message.get('card_binding_conflict')
    assert message['source_row_ids'] == [target['stable_row_id']]
    assert message['card_provenance']['support_row']['stable_row_id'] == target['card_row_id']
    assert summary['source_rows_covered'] == len(source)
