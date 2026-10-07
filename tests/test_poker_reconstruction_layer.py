import copy
import json
from pathlib import Path

from parser_core.poker_pipeline import analyze_timeline
from parser_core.poker_reconstruction_layer import reconstruct_production_hands, write_reconstruction
from parser_core.poker_reconstruction_layer import proven_start_stacks

CONFIG = {'enabled': True, 'allow_logic_math_output': False, 'max_nodes': 20000}


def test_smoke19_stack_snapshot_is_joined_to_actor_but_never_authoritative():
    fixture=json.loads((Path(__file__).parent/'fixtures/smoke19_stack_join.json').read_text('utf-8'))
    fact,events=fixture['fact'],fixture['events']
    hand=dict(table_id='T09',session_id=fact['session'],hand_id=fact['hand_id'],
              messages=[dict(source_rows=events)])
    original=copy.deepcopy(fact)
    stacks,receipts,rejected=proven_start_stacks(hand,[fact])
    assert stacks=={} and receipts[0]['actor']=='u4itel88'
    assert receipts[0]['START_STACK_CANDIDATE']=='59140' and receipts[0]['START_STACK'] is None
    assert rejected[0]['reason']=='CLEAN_WINDOW_TIMING_UNPROVEN' and fact==original
    for changed in [dict(fact,occupancy_epoch='wrong'),dict(fact,seat='seat1'),
                    dict(fact,session='other'),dict(fact,hand_id='other'),dict(fact,START_STACK='1')]:
        assert proven_start_stacks(hand,[changed])[1]==[]


def test_literal_amount_role_survives_without_granting_a_validator_contract():
    from parser_core.poker_hand_assembly import parse_text
    assert parse_text('Dealer: A raises to 300')['amount_role']=='RAISE_TO'
    assert parse_text('Dealer: A calls 100')['amount_role']=='CALL'
    source=records('Hand #123456','Dealer: A posts Small Blind 5','Dealer: B posts Big Blind 10',
                   'Dealer: A calls 5','Dealer: B checks','Dealer: A folds','Dealer: B wins 20')
    assert any(i['code']=='AMOUNT_SEMANTICS_UNPROVEN' for i in source[0]['validation']['issues'])


def records(*texts, failed=None):
    rows = [{'stable_row_id': f's:R{i}', 'table_id': 'T01', 'session_id': 's',
             'frame_id': 1, 'row_index': i, 'epoch': 1, 'origin': 'NEW',
             'raw_ocr_text': text, 'video_time': 0} for i, text in enumerate(texts)]
    return analyze_timeline(rows, ocr_status={r['stable_row_id']:
        'ERROR' if i == failed else 'RAW_UNREVIEWED' for i, r in enumerate(rows)})[0]


FOLD = ('Hand #123456', 'Dealer: Starting a new hand',
        'Dealer: A posts Small Blind 5', 'Dealer: B posts Big Blind 10',
        'Dealer: A folds', 'Dealer: B gets uncalled bet 5 back', 'Dealer: B wins 10')


def test_exact_additional_validation_preserves_existing_output_and_sources(tmp_path):
    original = records(*FOLD)
    original[0]['observed_hand']['messages'][0]['source_rows'][0]['source_index'] = 777
    frozen = copy.deepcopy(original)
    summary = write_reconstruction(tmp_path, original, config=CONFIG)
    result = json.loads((tmp_path / 'poker_reconstruction.jsonl').read_text())
    assert original == frozen
    assert summary['hands_seen'] == summary['output_eligible'] == 1
    assert result['reconstruction_status'] == 'AUTO_RECOVERED'
    assert result['calculated_pot'] == result['observed_payout'] == 10
    assert result['returns'] == {'B': 5}
    assert result['player_contributions'] == {'A': 5, 'B': 10}
    assert result['raw_events'] == [r for m in original[0]['observed_hand']['messages'] for r in m['source_rows']]
    assert not result['dropped_stale'] and not result['unresolved_rows']


def test_disabled_has_no_engine_results():
    out, summary = reconstruct_production_hands(records(*FOLD), config=dict(CONFIG, enabled=False))
    assert out == [] and not summary['enabled']


def test_failed_ocr_cannot_be_promoted_by_exact_pot():
    out, _ = reconstruct_production_hands(records(*FOLD, failed=2), config=CONFIG)
    assert out[0]['status'] == 'AUTO_RECOVERED'
    assert not out[0]['output_eligible'] and out[0]['unresolved_rows']
    assert 'EXISTING_PRODUCTION_VALIDATION_UNRESOLVED' in out[0]['output_blockers']


def test_logic_math_inference_is_candidate_only():
    source = list(FOLD)
    source[5] = 'Dealer: B gets uncalled bet'
    out, _ = reconstruct_production_hands(records(*source), config=CONFIG)
    assert out[0]['status'] == 'RECOVERED_BY_LOGIC_MATH'
    assert not out[0]['output_eligible']
    assert 'LOGIC_MATH_OUTPUT_DISABLED' in out[0]['output_blockers']
    assert out[0]['inferred_amounts']


def test_unknown_and_partial_remain_unresolved():
    for source, status in [(('unreadable',), 'TRUNCATED_START'),
                           (('Hand #123456', 'Dealer: A posts Small Blind 5'), 'TRUNCATED_END')]:
        out, _ = reconstruct_production_hands(records(*source), config=CONFIG)
        assert out[0]['status'] == status
        assert not out[0]['output_eligible'] and out[0]['unresolved_rows']


def test_repeated_source_event_never_removed_from_old_output():
    source = list(FOLD)
    source.insert(3, source[2])
    old = records(*source)
    frozen = copy.deepcopy(old)
    out, _ = reconstruct_production_hands(old, config=CONFIG)
    assert old == frozen
    assert not out[0]['output_eligible']
    assert len(out[0]['raw_events']) == len(source)


def test_repeated_check_cannot_be_admitted():
    source = ('Hand #123456', 'Dealer: A posts Small Blind 5',
              'Dealer: B posts Big Blind 10', 'Dealer: A calls 5',
              'Dealer: B checks', 'Dealer: B checks', 'Dealer: A folds',
              'Dealer: B wins 20')
    old = records(*source)
    frozen = copy.deepcopy(old)
    out, _ = reconstruct_production_hands(old, config=CONFIG)
    assert old == frozen and not out[0]['output_eligible']
    assert len(out[0]['raw_events']) == len(source)


def test_cross_frame_assembly_is_candidate_only():
    source = list(FOLD)
    source[2:3] = ['Dealer: A posts Small Blind', '5']
    old = records(*source)
    old[0]['observed_hand']['messages'][2]['source_rows'][1]['frame_id'] = 99
    out, _ = reconstruct_production_hands(old, config=CONFIG)
    assert not out[0]['output_eligible']
    assert len(out[0]['raw_events']) == len(source)
