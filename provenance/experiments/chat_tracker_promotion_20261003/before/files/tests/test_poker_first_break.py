"""Sequential experimental diagnostics and immutable candidate evidence."""
import copy
import pytest
from parser_core.poker_diagnostics import diagnose_hand, revalidate_candidate
from parser_core.poker_hand_assembly import assemble_messages, assemble_hands
from test_poker_logic_math_jev import hand, basic, simple, runout


@pytest.mark.parametrize('texts,trigger', [
    (['Hand #1', 'B posts Big Blind 10', 'A folds'], 'MISSING_BLINDS_BEFORE_ACTION'),
    (['Hand #1', 'A posts Small Blind 5', 'A folds'], 'MISSING_BLINDS_BEFORE_ACTION'),
    (['Hand #1', 'A posts Small Blind', 'B posts Big Blind 10'], 'MISSING_AMOUNT'),
])
def test_missing_forced_sequence(texts, trigger):
    result = diagnose_hand(hand(*texts))
    assert result['first_break']['RULE_TRIGGER'] == trigger
    assert result['first_break']['PROVEN_CAUSE'] == 'UNKNOWN'


@pytest.mark.parametrize('action,tail,kind,amount', [
    ('A posts Small Blind', '5', 'SMALL_BLIND', '5'),
    ('A calls', '10', 'CALL', '10'),
    ('A raises to', '30 and is All-in', 'RAISE', '30'),
    ('A posts Big Blind', '10 and is All-in', 'BIG_BLIND', '10'),
    ('A gets uncalled bet', '5 back', 'UNCALLED_RETURN', '5'),
    ('_rinrus_posts Small Blind', '5', 'SMALL_BLIND', '5'),
])
def test_source_supported_continuations(action, tail, kind, amount):
    rows = [{'row_id': str(i), 'raw_text': text, 'epoch': 1} for i, text in enumerate([action, tail])]
    before = copy.deepcopy(rows)
    result = assemble_messages(rows)
    assert len(result) == 1 and result[0]['action'] == kind and result[0]['amount'] == amount
    assert result[0]['source_row_ids'] == ['0', '1'] and rows == before
    if 'All-in' in tail:
        assert result[0]['all_in']


@pytest.mark.parametrize('actions,trigger', [
    (['A folds', 'A calls 5'], 'ACTION_AFTER_FOLD'),
    (['A raises to 30 and is All-in', 'A calls 10'], 'ACTION_AFTER_ALLIN'),
    (['A checks'], 'CHECK_FACING_BET'),
    (['A calls 3'], 'INVALID_CALL'),
    (['A raises to 5'], 'INVALID_RAISE'),
    (['A calls 5', 'A calls 5'], 'CALL_WITH_NOTHING_TO_CALL'),
    (['B wins 15'], 'OLD_WIN'),
    (['Hand #2'], 'HAND_BOUNDARY_CONFLICT'),
    (['A. calls 5'], 'PLAYER_IDENTITY_CONFLICT'),
    (['Starting a new hand'], 'CHAT_OVERLAP_DUPLICATE'),
])
def test_exact_first_rule_and_no_cascade(actions, trigger):
    value = basic(*actions, 'unreadable', 'A checks', 'B wins 999')
    # Keep conflicting boundary inside a single candidate to test validator.
    if actions == ['Hand #2']:
        from parser_core.poker_hand_assembly import make_message
        value['messages'].append(make_message([{'row_id': 'other', 'raw_text': 'Hand #2'}]))
    original = copy.deepcopy(value)
    result = diagnose_hand(value)
    assert result['first_break']['RULE_TRIGGER'] == trigger
    first = next(t for t in result['trace'] if t['status'] == 'FIRST_BREAK')
    assert first['state_after'] == first['state_before']
    assert sum(t['is_first_break'] for t in result['trace']) == 1
    assert all(t['status'] == 'AFTER_BREAK' and not t['issues'] for t in result['trace'][first['message_index']+1:])
    assert original == value


def test_valid_second_call_after_raise():
    result = diagnose_hand(runout('A calls 5', 'B raises to 30', 'A calls 20', payouts=('A wins 60',)))
    assert result['first_break'] is None and result['math']['delta'] == '0'


def test_raise_to_financial_semantics():
    result = diagnose_hand(runout('A raises to 30', 'B calls 20', payouts=('A wins 60',)))
    assert result['math']['players']['A']['preflop_contribution'] == '25'
    assert result['final_status'] == 'FULLY_VALID'


def test_partial_first_and_last_are_explicit():
    assert diagnose_hand(hand('A folds'))['first_break']['code'] == 'MISSING_BLINDS_BEFORE_ACTION'
    result = diagnose_hand(basic('A calls 5', 'B checks'))
    assert result['first_break']['virtual_boundary_check']
    assert result['first_break']['source_row_ids'] == []
    assert all(t['status'] == 'OK' for t in result['trace'])


def test_card_missing_does_not_break_betting():
    value = runout('A calls 5', 'B checks')
    value['messages'] = [m for m in value['messages'] if m['action'] != 'SHOW']
    result = diagnose_hand(value)
    assert result['first_break'] is None and result['logic_status'] == 'PASS'
    assert result['card_status'] == 'NOT_VERIFIABLE' and result['final_status'] != 'FULLY_VALID'


def test_candidate_requires_full_revalidation():
    value = simple()
    assert revalidate_candidate(value, copy.deepcopy(value))['accepted']
    candidate = copy.deepcopy(value)
    candidate['messages'][-1]['amount'] = '11'
    result = revalidate_candidate(value, candidate)
    assert not result['accepted']


def test_hallucinated_jev_action_rejected():
    value = simple()
    candidate = copy.deepcopy(value)
    candidate['messages'][4]['action'] = 'CALL'
    candidate['messages'][4]['amount'] = '5'
    assert not revalidate_candidate(value, candidate)['accepted']


def test_candidate_new_first_break_restarts_from_beginning():
    from parser_core.jev_poker_resolver import candidate_view
    from parser_core.poker_logic_engine import validate_hand
    from parser_core.poker_hand_assembly import make_message
    value = basic('A calls', 'unreadable action')
    value['messages'][4]['source_rows'][0]['epoch'] = 0
    tail = make_message([{'row_id': 'amount', 'raw_text': '5', 'epoch': 0, 'table_id': 'T01', 'session_id': 's'}])
    value['messages'].insert(5, tail)
    decision = {'decision': 'LIKELY_MULTILINE_CONTINUATION',
                'target_rows': value['messages'][4]['source_row_ids']+tail['source_row_ids'],
                'evidence_rows': value['messages'][4]['source_row_ids']+tail['source_row_ids']}
    candidate, rejection = candidate_view(value, validate_hand(value), decision)
    assert not rejection
    result = revalidate_candidate(value, candidate)
    assert not result['accepted']
    assert result['revalidation']['first_break']['code'] == 'MISSING_ACTION'


def test_candidate_cannot_change_observed_payout_with_self_citation():
    value = simple()
    value['messages'][-1]['amount'] = '11'
    candidate = copy.deepcopy(value)
    candidate['messages'][-1]['amount'] = '10'
    candidate['inferences'] = [{'decision': {'evidence_rows': candidate['messages'][-1]['source_row_ids']}}]
    assert not revalidate_candidate(value, candidate)['accepted']


def test_candidate_cannot_mask_duplicate_without_proof():
    value = basic('B posts Big Blind 10', 'A folds', 'B gets uncalled bet 5 back', 'B wins 10')
    candidate = copy.deepcopy(value, )
    candidate['excluded_message_indices'] = [4]
    assert not revalidate_candidate(value, candidate)['accepted']


def test_after_break_folded_show_is_not_independent_card_contradiction():
    result = diagnose_hand(basic('A checks', 'B folds', 'B shows Ah Ac'))
    assert not any(i['detail'] == 'Folded player showdown' for i in result['card_issues'])


def test_repeated_deal_is_visible_first_break():
    result = diagnose_hand(basic('Dealing cards', 'Dealing cards', 'A folds', 'B wins 15'))
    assert result['first_break']['code'] == 'CHAT_OVERLAP_DUPLICATE'


@pytest.mark.parametrize('field,value', [('epoch', 2), ('table_id', 'T02'), ('session_id', 'other')])
def test_jev_multiline_cannot_cross_source_boundary(field, value):
    from parser_core.jev_poker_resolver import candidate_view
    from parser_core.poker_logic_engine import validate_hand
    from parser_core.poker_hand_assembly import make_message
    observed = simple()
    rows = [{'row_id': 'a', 'raw_text': 'B gets uncalled bet', 'table_id': 'T01', 'session_id': 's', 'epoch': 1},
            {'row_id': 'b', 'raw_text': '5 back', 'table_id': 'T01', 'session_id': 's', 'epoch': 1}]
    rows[1][field] = value
    observed['messages'][5:6] = [make_message([r]) for r in rows]
    decision = {'decision': 'LIKELY_MULTILINE_CONTINUATION', 'target_rows': ['a', 'b'], 'evidence_rows': ['a', 'b']}
    candidate, rejection = candidate_view(observed, validate_hand(observed), decision)
    assert candidate is None and rejection == 'MULTILINE_SOURCE_BOUNDARY_MISMATCH'


def test_observed_blind_level_is_valid_cross_row_evidence():
    from parser_core.jev_poker_resolver import candidate_view
    from parser_core.poker_logic_engine import validate_hand
    value = hand('Hand #1', 'Blind level 5/10', 'A posts Small Blind 5', 'B posts Big Blind',
                 'A folds', 'B gets uncalled bet 5 back', 'B wins 10')
    value['blind_level'] = {'source_type': 'OBSERVED', 'source_row_ids': ['r1'], 'small_blind': '5', 'big_blind': '10'}
    candidate, rejection = candidate_view(value, validate_hand(value), {'decision': 'LIKELY_MISSING_AMOUNT',
        'target_rows': ['r3'], 'evidence_rows': ['r1'], 'candidate_amount': '10'})
    assert not rejection
    result = revalidate_candidate(value, candidate)
    assert result['revalidation'] is not None


def test_decorated_deal_phrase_retains_observed_glyph():
    messages = assemble_messages([{'row_id': 'decorated', 'raw_text': 'Dealer: Dealing cards●'}])
    assert messages[0]['action'] == 'DEAL'
    assert messages[0]['raw_text'] == 'Dealer: Dealing cards●'
    assert messages[0]['source_row_ids'] == ['decorated']


def test_mcp_transport_uses_jev_decide_and_native_questions():
    import json
    from parser_core.jev_poker_resolver import JevMCPClient
    client = JevMCPClient({})
    payload, options = client.prepare_request({'hand': simple(), 'conflicts': []})
    body = json.loads(payload)
    assert client.config['endpoint'] == 'https://www.jevai.org/api/mcp'
    assert body['method'] == 'tools/call' and body['params']['name'] == 'jev_decide'
    assert body['params']['arguments']['questions']['hypothesis']['type'] == 'choice'
    assert not options
    assert client.unwrap_response({'result': {'structuredContent': {'answers': {}}}}) == {'code': 0, 'data': {'answers': {}}}


def test_mcp_error_is_not_successful_hypothesis():
    from parser_core.jev_poker_resolver import JevMCPClient
    with pytest.raises(RuntimeError, match='JEV_MCP_TOOL_ERROR'):
        JevMCPClient({}).unwrap_response({'result': {'isError': True, 'content': [{'type': 'text', 'text': 'upstream failed'}]}})


@pytest.mark.parametrize('text,amount,all_in', [('A calls 25 145●', '25145', False),
                                              ('A calls 71 950 and is All-in ●', '71950', True)])
def test_observed_amount_before_decorative_glyph(text, amount, all_in):
    msg = assemble_messages([{'row_id': 'suffix', 'raw_text': text}])[0]
    assert msg['amount'] == amount and msg['all_in'] == all_in
    assert msg['raw_text'] == text


def test_split_allin_suffix_requires_complete_source_and_same_epoch():
    rows = [{'row_id': 'a', 'raw_text': 'A calls 10 and is', 'epoch': 1},
            {'row_id': 'b', 'raw_text': 'All-in', 'epoch': 1}]
    msg = assemble_messages(rows)[0]
    assert msg['amount'] == '10' and msg['all_in'] and msg['source_row_ids'] == ['a', 'b']
    rows[1]['epoch'] = 2
    assert len(assemble_messages(rows)) == 2


def test_truncated_small_is_not_invented_blind():
    result = assemble_messages([{'row_id': 'partial', 'raw_text': 'Dealer: A posts Small'}])
    assert result[0]['action'] == 'UNKNOWN_EVENT' and result[0]['amount'] is None


def test_table_isolation_all_nine():
    rows = [dict(r, table_id=f'T{i:02}', row_id=f'T{i:02}:{r["row_id"]}')
            for i in range(1, 10) for m in simple()['messages'] for r in m['source_rows']]
    combined = assemble_hands(rows)
    for h in combined:
        h['semantics'] = {'call': 'ADDED', 'bet': 'ADDED'}
        tables = {r['table_id'] for m in h['messages'] for r in m['source_rows']}
        assert tables == {h['table_id']}
        single = assemble_hands([r for r in rows if r['table_id'] == h['table_id']])[0]
        single['semantics'] = h['semantics']
        assert diagnose_hand(single)['first_break'] == diagnose_hand(h)['first_break']


def test_observed_math_continues_after_logic_break_without_fake_localization():
    result = diagnose_hand(basic('A checks', 'A folds', 'B gets uncalled bet 5 back', 'B wins 10'))
    assert result['math']['delta'] == '0'
    assert result['logic_status'] == 'FAIL' and result['final_status'] != 'FULLY_VALID'
    assert result['math']['ledger_certainty'] == 'UNPROVEN'
