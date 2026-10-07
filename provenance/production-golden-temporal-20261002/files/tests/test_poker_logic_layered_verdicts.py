"""Regressions for proved Smoke17 diagnostic false positives."""
import copy

from parser_core.poker_hand_assembly import assemble_messages
from parser_core.poker_logic_engine import validate_hand
from test_poker_logic_math_jev import hand, simple, runout


def test_unresolved_deal_is_noncritical_without_losing_observation():
    value = hand('Hand #1', 'Dealing cards', 'A posts Small Blind 5',
                 'B posts Big Blind 10', 'A folds', 'B gets uncalled bet 5 back', 'B wins 10')
    value['messages'][1]['source_rows'][0]['origin'] = 'UNRESOLVED_IDENTITY'
    original = copy.deepcopy(value)
    result = validate_hand(value)
    assert result['action_logic_status'] == 'PASS'
    assert result['identity_status'] == 'IDENTITY_NONCRITICAL'
    assert result['status'] == 'FULLY_VALID_HAND'
    assert result['issues'][0]['severity'] == 'WARNING'
    assert value == original


def test_unresolved_money_is_certification_blocker_but_separate_from_action_rules():
    value = simple()
    value['messages'][2]['source_rows'][0]['origin'] = 'UNRESOLVED_IDENTITY'
    result = validate_hand(value)
    assert result['action_logic_status'] == 'PASS'
    assert result['identity_status'] == 'IDENTITY_CRITICAL'
    assert result['status'] == 'UNRESOLVED_HAND'
    assert any(i['code'] == 'ROW_IDENTITY_UNRESOLVED' for i in result['certification_blockers'])


def test_missing_show_cards_is_unknown_not_observed_contradiction():
    value = runout('A calls 5', 'B checks')
    value['messages'] = [m for m in value['messages'] if m['action'] != 'SHOW']
    result = validate_hand(value)
    assert result['action_logic_status'] == 'PASS'
    assert result['card_status'] == 'NOT_VERIFIABLE'
    assert result['side_pot_status'] == 'NOT_VERIFIABLE'
    assert result['math']['pass']
    assert result['status'] != 'FULLY_VALID_HAND'
    assert not any(i['code'] == 'CARD_LOGIC_CONFLICT' for i in result['issues'])


def test_observed_wrong_winner_still_blocks_card_certification():
    result = validate_hand(runout('A calls 5', 'B checks', payouts=('B wins 20',)))
    assert result['action_logic_status'] == 'PASS'
    assert result['card_status'] == 'CONFLICT'
    assert result['status'] == 'CARD_CONFLICT'


def test_incomplete_ledger_does_not_prove_sidepot_conflict():
    value = simple()
    value['messages'].insert(4, assemble_messages([{'row_id': 'gap', 'raw_text': 'unreadable action'}])[0])
    value['messages'][-1]['amount'] = '100'
    result = validate_hand(value)
    assert result['side_pot_status'] == 'NOT_VERIFIABLE'
    assert not any(i['code'] == 'SIDE_POT_ELIGIBILITY_CONFLICT' for i in result['issues'])
    assert not result['math']['pass']
    assert result['status'] != 'FULLY_VALID_HAND'


def test_same_explicit_header_is_retained_redundant_metadata():
    value = simple()
    value['messages'].insert(1, copy.deepcopy(value['messages'][0]))
    result = validate_hand(value)
    assert result['status'] == 'FULLY_VALID_HAND'
    assert result['issues'][0]['code'] == 'CHAT_OVERLAP_DUPLICATE'
    assert result['issues'][0]['severity'] == 'WARNING'


def test_unproven_new_hand_marker_remains_ambiguous_blocker():
    value = simple()
    value['messages'].insert(4, copy.deepcopy(value['messages'][1]))
    result = validate_hand(value)
    assert result['action_logic_status'] == 'NOT_VERIFIABLE'
    assert result['status'] != 'FULLY_VALID_HAND'


def test_server_notices_are_informational_and_levels_require_linked_prefix():
    texts = ['Waiting for antivea to reconnect...', 'antivea is reconnected',
             'Limits going up: stakes', '3 000/6 000 ante 700', 'unreadable action',
             '4 000/8 000 ante 900']
    rows = [{'row_id': str(i), 'raw_text': text} for i, text in enumerate(texts)]
    messages = assemble_messages(rows)
    assert [m['action'] for m in messages] == ['TABLE_INFO', 'TABLE_INFO', 'BLIND_LEVEL', 'UNKNOWN_EVENT', 'UNKNOWN_EVENT']
    assert messages[2]['source_row_ids'] == ['2', '3']
    assert messages[2]['raw_text'] == 'Limits going up: stakes 3 000/6 000 ante 700'
    assert all(m['amount'] is None for m in messages[:3])


def test_server_notice_does_not_hide_a_missing_player_response():
    value = hand('Hand #1', 'A posts Small Blind 5', 'B posts Big Blind 10',
                 'Waiting for A to reconnect...', 'A is reconnected',
                 'Dealing Flop 2h 3d 4s', 'B wins 15')
    result = validate_hand(value)
    assert any(i['code'] == 'BETTING_ROUND_INCOMPLETE' for i in result['issues'])
    assert result['action_logic_status'] == 'FAIL'
    assert result['status'] != 'FULLY_VALID_HAND'


def test_informational_origin_is_critical_when_used_as_financial_evidence():
    value = simple()
    notice = assemble_messages([{'row_id': 'level', 'raw_text': 'Limits going up: stakes',
                                'origin': 'UNRESOLVED_IDENTITY'}])[0]
    value['messages'].insert(1, notice)
    for evidence in ({'blind_level': {'source_row_ids': ['level']}},
                     {'inferences': [{'decision': {'evidence_rows': ['level']}}]}):
        candidate = copy.deepcopy(value)
        candidate.update(evidence)
        result = validate_hand(candidate)
        assert result['identity_status'] == 'IDENTITY_CRITICAL'
        assert result['status'] != 'FULLY_VALID_HAND'
        assert any(i['code'] == 'ROW_IDENTITY_UNRESOLVED' for i in result['certification_blockers'])


def test_wrapped_and_truncated_server_pause_keeps_all_source_rows():
    rows = [{'row_id': str(i), 'raw_text': text} for i, text in enumerate([
        'Dealer: Waiting for antivea to', 'reconnect...',
        'Dealer: Waiting for antivea to', 'Dealer: antivea is reconnected'])]
    original = copy.deepcopy(rows)
    messages = assemble_messages(rows)
    assert [m['action'] for m in messages] == ['TABLE_INFO']*3
    assert messages[0]['source_row_ids'] == ['0', '1']
    assert messages[1]['source_row_ids'] == ['2']
    assert rows == original
