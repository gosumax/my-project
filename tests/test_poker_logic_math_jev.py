"""Experimental evidence-gated contracts; production modules remain frozen."""
import copy
import hashlib
import pytest
from parser_core.poker_hand_assembly import assemble_hands, assemble_messages, make_message
from parser_core.poker_logic_engine import validate_hand, resolve_hand
from parser_core.jev_poker_resolver import JevResolver
from parser_core.poker_card_engine import best_rank


def hand(*texts, semantics=True):
    rows = [{'table_id': 'T01', 'session_id': 's', 'row_id': f'r{i}', 'frame_id': i,
             'timestamp_ms': i*10, 'raw_text': t, 'row_pixel_sha256': hashlib.sha256(t.encode()).hexdigest()}
            for i, t in enumerate(texts)]
    result = assemble_hands(rows)[0]
    if semantics:
        result['semantics'] = {'call': 'ADDED', 'bet': 'ADDED', 'evidence': 'SYNTHETIC_TEST_CONTRACT'}
    return result


def basic(*actions):
    return hand('Hand #1', 'Starting a new hand', 'A posts Small Blind 5', 'B posts Big Blind 10', *actions)


def codes(value):
    return {i['code'] for i in validate_hand(value)['issues']}


def simple():
    return basic('A folds', 'B gets uncalled bet 5 back', 'B wins 10')


def runout(*actions, payouts=('A wins 20',)):
    return basic(*actions, 'Dealing Flop 2h 3d 4s', 'A checks', 'B checks',
                 'Dealing Turn 2h 3d 4s 8c', 'A checks', 'B checks',
                 'Dealing River 2h 3d 4s 8c 9h', 'A checks', 'B checks',
                 'A shows Ah Ac', 'B shows Kh Kc', *payouts)


def test_normal_fold_sequence():
    assert validate_hand(simple())['status'] == 'FULLY_VALID_HAND'


def test_check_check_and_cards():
    assert validate_hand(runout('A calls 5', 'B checks'))['status'] == 'FULLY_VALID_HAND'


def test_bet_fold():
    value = basic('A calls 5', 'B checks', 'Dealing Flop 2h 3d 4s',
                  'A bets 20', 'B folds', 'A gets uncalled bet 20 back', 'A wins 20')
    assert validate_hand(value)['status'] == 'FULLY_VALID_HAND'


def test_bet_call():
    value = runout('A calls 5', 'B checks')
    value['messages'][7:9] = [make_message([{'row_id': 'bet', 'raw_text': 'A bets 20'}]),
                               make_message([{'row_id': 'call', 'raw_text': 'B calls 20'}])]
    value['messages'][-1]['amount'] = '60'
    assert validate_hand(value)['status'] == 'FULLY_VALID_HAND'


def test_raise_call_delta():
    value = runout('A raises to 30', 'B calls 20', payouts=('A wins 60',))
    result = validate_hand(value)
    assert result['status'] == 'FULLY_VALID_HAND'
    assert result['math']['players']['A']['preflop_contribution'] == '25'
    assert result['math']['players']['B']['preflop_contribution'] == '20'


def test_reraise():
    assert validate_hand(runout('A raises to 30', 'B raises to 70', 'A calls 40', payouts=('A wins 140',)))['status'] == 'FULLY_VALID_HAND'


def allin(*actions, payouts=('A wins 30',)):
    return basic(*actions, 'A shows Ah Ac', 'B shows Kh Kc', 'Dealing Flop 2h 3d 4s',
                 'Dealing Turn 2h 3d 4s 8c', 'Dealing River 2h 3d 4s 8c 9h', *payouts)


def test_allin_call():
    value = allin('A raises to 30 and is All-in', 'B calls 5 and is All-in', 'A gets uncalled bet 15 back')
    assert validate_hand(value)['status'] == 'FULLY_VALID_HAND'


def test_allin_raise():
    assert validate_hand(allin('A raises to 30 and is All-in', 'B calls 20 and is All-in', payouts=('A wins 60',)))['status'] == 'FULLY_VALID_HAND'


def test_uncalled_return():
    assert validate_hand(simple())['math']['players']['B']['uncalled_return'] == '5'


def test_split_pot():
    value = runout('A calls 5', 'B checks', payouts=('A wins 10', 'B wins 10'))
    for msg in value['messages']:
        if msg['action'] == 'SHOW' and msg['player'] == 'B':
            msg['cards'] = ['Ad', 'As']
    assert validate_hand(value)['status'] == 'FULLY_VALID_HAND'


def test_sidepot_conservation():
    value = hand('Hand #1', 'A posts Small Blind 5', 'B posts Big Blind 10',
                 'C raises to 30 and is All-in', 'A calls 5 and is All-in', 'B calls 20 and is All-in',
                 'A shows Ah Ac', 'B shows Kh Kc', 'C shows Qh Qc',
                 'Dealing Flop 2h 3d 4s', 'Dealing Turn 2h 3d 4s 8c', 'Dealing River 2h 3d 4s 8c 9h',
                 'A wins 30', 'B wins 40')
    assert validate_hand(value)['status'] == 'FULLY_VALID_HAND'
    value['messages'][-1]['amount'] = '35'
    assert 'MATH_DELTA' in codes(value)


@pytest.mark.parametrize('actions,expected', [
    (['A folds', 'A calls 5'], 'ACTION_AFTER_FOLD'),
    (['A calls 5 and is All-in', 'A bets 5'], 'ACTION_AFTER_ALLIN'),
    (['A checks'], 'CHECK_FACING_BET'),
    (['B calls 10'], 'CALL_WITH_NOTHING_TO_CALL'),
    (['A raises to 10'], 'INVALID_RAISE'),
    (['A posts Ante 1', 'A posts Ante 1'], 'CHAT_OVERLAP_DUPLICATE'),
    (['B posts Big Blind 10'], 'CHAT_OVERLAP_DUPLICATE'),
    (['Starting a new hand'], 'CHAT_OVERLAP_DUPLICATE'),
    (['A wins 15', 'B raises to 30'], 'OLD_WIN'),
    (['Dealing Turn 2h 3d 4s 8c'], 'BROKEN_STREET_ORDER'),
    (['A calls 5', 'B checks', 'Dealing Flop 2h 3d 4s', 'Dealing Flop 2h 3d 4s'], 'CHAT_OVERLAP_DUPLICATE'),
    (['A calls 5', 'B checks', 'Dealing Flop 2h 2h 4s'], 'CARD_LOGIC_CONFLICT'),
    (['A folds', 'B gets uncalled bet 100 back'], 'RETURN_EXCEEDS_CONTRIBUTION'),
])
def test_forbidden_transition(actions, expected):
    assert expected in codes(basic(*actions))


def test_multiline_blind():
    value = hand('Hand #1', 'A posts Small Blind', '5', 'B posts Big Blind', '10',
                 'A folds', 'B gets uncalled bet 5 back', 'B wins 10')
    assert validate_hand(value)['status'] == 'FULLY_VALID_HAND'
    assert value['messages'][1]['source_row_ids'] == ['r1', 'r2']


def test_multiline_uncalled():
    value = basic('A folds', 'B gets uncalled bet', '5 back', 'B wins 10')
    assert validate_hand(value)['status'] == 'FULLY_VALID_HAND'


def test_multiline_allin():
    value = basic('A raises to 30', 'and is All-in', 'B folds', 'A gets uncalled bet 20 back', 'A wins 20')
    assert value['messages'][4]['all_in'] is True
    assert validate_hand(value)['status'] == 'FULLY_VALID_HAND'


def test_nickname_punctuation_never_silent_merge():
    value = basic('A. folds', 'B gets uncalled bet 5 back', 'B wins 10')
    assert validate_hand(value)['status'] != 'FULLY_VALID_HAND'
    assert 'A.' in validate_hand(value)['players']


def test_incomplete_first_hand():
    value = hand('A posts Small Blind 5', 'B posts Big Blind 10', 'A folds', 'B gets uncalled bet 5 back', 'B wins 10')
    assert validate_hand(value)['status'] == 'INCOMPLETE_START'


def test_incomplete_last_hand():
    assert validate_hand(basic('A calls 5', 'B checks'))['status'] == 'INCOMPLETE_END'


def missing_blind():
    value = basic('A folds', 'B gets uncalled bet 5 back', 'B wins 10')
    value['messages'][3]['amount'] = None
    value['blind_level'] = {'source_type': 'OBSERVED', 'small_blind': '5', 'big_blind': '10', 'source_row_ids': ['r0']}
    # Explicit source text, not merely a trusted structured value.
    value['messages'][0]['source_rows'][0]['raw_text'] = 'Hand #1; Blind level 5/10'
    value['messages'][0]['raw_text'] = 'Hand #1; Blind level 5/10'
    return value


def response_for(value, **updates):
    return dict({'decision': 'LIKELY_MISSING_AMOUNT', 'target_rows': ['r3'], 'candidate_amount': '10',
                 'evidence_rows': ['r0'], 'reason': 'Observed level', 'confidence': .9}, **updates)


def test_jev_accepted_missing_blind_inference():
    value = missing_blind()
    before = copy.deepcopy(value)
    result = resolve_hand(value, JevResolver(lambda request: response_for(value)))
    assert result['final']['status'] == 'JEV_ASSISTED_CERTIFIED'
    assert result['final']['accepted_inference'] is True
    assert value == before
    assert result['resolution']['affected_rows'] == ['r3']


def test_jev_rejected_inference():
    value = missing_blind()
    result = resolve_hand(value, JevResolver(lambda req: response_for(value, candidate_amount='11')))
    assert result['final']['status'] == 'UNRESOLVED_HAND'


def test_jev_hallucinated_action_rejected():
    value = missing_blind()
    result = resolve_hand(value, JevResolver(lambda req: response_for(value, decision='CREATE_ACTION')))
    assert result['resolution']['accepted'] is False


def test_jev_math_fixed_logic_broken_rejected():
    value = missing_blind()
    value['messages'].insert(-1, make_message([{'row_id': 'bad', 'raw_text': 'A checks'}]))
    result = resolve_hand(value, JevResolver(lambda req: response_for(value)))
    assert result['resolution']['revalidation']['math']['pass'] is True
    assert result['final']['status'] == 'UNRESOLVED_HAND'


def test_jev_logic_fixed_math_broken_rejected():
    value = missing_blind()
    value['messages'][-1]['amount'] = '9'
    result = resolve_hand(value, JevResolver(lambda req: response_for(value)))
    assert result['resolution']['revalidation']['logic_pass'] is True
    assert result['final']['status'] == 'UNRESOLVED_HAND'


def test_jev_cache_exact_input(tmp_path):
    value = missing_blind()
    resolver = JevResolver(lambda req: response_for(value), tmp_path)
    resolve_hand(value, resolver)
    resolve_hand(value, resolver)
    assert resolver.live_calls == 1 and resolver.cache_hits == 1
    changed = copy.deepcopy(value)
    changed['messages'][-1]['source_rows'][0]['frame_id'] += 1
    resolve_hand(changed, resolver)
    assert resolver.live_calls == 2


@pytest.mark.parametrize('bad', [{}, {'decision': 'x', 'confidence': .9, 'target_rows': [{}]},
                                 {'decision': 'x', 'confidence': True, 'target_rows': ['r3']},
                                 {'decision': 'x', 'confidence': .9, 'target_rows': ['r3'], 'evidence_rows': [{}]}])
def test_malformed_jev_fails_closed(bad):
    assert resolve_hand(missing_blind(), JevResolver(lambda req: bad))['final']['status'] == 'UNRESOLVED_HAND'


def test_jev_not_called_on_pass_or_partial():
    def forbidden(req):
        raise AssertionError('Unexpected Jev call')
    resolver = JevResolver(forbidden)
    resolve_hand(simple(), resolver)
    resolve_hand(basic('A folds'), resolver)
    assert resolver.live_calls == 0


def test_stale_duplicate_requires_pixels_and_conflict():
    value = simple()
    duplicate = copy.deepcopy(value['messages'][3])
    duplicate['source_row_ids'] = ['duplicate']
    duplicate['source_rows'][0]['row_id'] = 'duplicate'
    value['messages'].insert(4, duplicate)
    response = {'decision': 'LIKELY_STALE_DUPLICATE', 'target_rows': ['duplicate'], 'evidence_rows': ['r3'], 'confidence': .99}
    assert resolve_hand(value, JevResolver(lambda req: response))['final']['status'] == 'JEV_ASSISTED_CERTIFIED'
    value['messages'][4]['source_rows'][0]['row_pixel_sha256'] = 'different'
    assert resolve_hand(value, JevResolver(lambda req: response))['final']['status'] == 'UNRESOLVED_HAND'


def test_card_description_conflict():
    value = runout('A calls 5', 'B checks')
    value['messages'][-3]['showdown_class'] = 'FLUSH'
    assert 'CARD_LOGIC_CONFLICT' in codes(value)


def test_hole_before_board_collision():
    value = allin('A raises to 30 and is All-in', 'B calls 20 and is All-in', payouts=('A wins 60',))
    for m in value['messages']:
        if m['action'] in {'FLOP', 'TURN', 'RIVER'}:
            m['cards'][0] = 'Ah'
    assert 'CARD_LOGIC_CONFLICT' in codes(value)


def test_wrong_winner_conserved_chips():
    value = runout('A calls 5', 'B checks', payouts=('B wins 20',))
    assert validate_hand(value)['math']['pass'] is True
    assert 'CARD_LOGIC_CONFLICT' in codes(value)


def test_boundary_conflict_retained():
    rows = [{'table_id': 'T01', 'session_id': 's', 'row_id': str(i), 'raw_text': t} for i, t in
            enumerate(['Hand #1', 'A posts Small Blind 5', 'Hand #2', 'B posts Big Blind 10'])]
    hands = assemble_hands(rows)
    assert len(hands) == 2
    assert 'HAND_BOUNDARY_CONFLICT' in codes(hands[0])


def test_unique_call_effect_needs_no_predeclared_amount_contract():
    value = runout('A calls 5', 'B checks')
    value.pop('semantics')
    before = copy.deepcopy(value)
    result = validate_hand(value)
    assert result['status'] == 'FULLY_VALID_HAND'
    assert result['amount_resolution_proofs'][0]['resolved_semantics'] == 'ADDED'
    assert result['amount_resolution_proofs'][0]['delta'] == '5'
    assert value == before


def test_ambiguous_allin_call_effect_still_blocks_certification():
    value = basic('B raises to 30', 'A calls 10 and is All-in')
    value.pop('semantics')
    result = validate_hand(value)
    assert 'AMOUNT_SEMANTICS_UNPROVEN' in {i['code'] for i in result['issues']}
    assert result['amount_resolution_proofs'][0]['status'] == 'AMBIGUOUS'
    assert result['status'] != 'FULLY_VALID_HAND'


def test_holdem_wheel_and_fullhouse():
    assert best_rank(['Ah', '2d', '3s', '4c', '5h', 'Kd', 'Qc']) == (4, 5)
    assert best_rank(['Ah', 'Ad', 'As', 'Kc', 'Kh', '2d', 'Qc']) == (6, 14, 13)


def test_uncalled_excess_is_not_winnings():
    assert 'MISSING_UNCALLED_RETURN' in codes(basic('A raises to 100', 'B folds', 'A wins 110'))


def test_unresolved_header_identity_blocks_certification():
    value = simple()
    value['messages'][0]['source_rows'][0]['origin'] = 'UNRESOLVED_IDENTITY'
    assert 'ROW_IDENTITY_UNRESOLVED' in codes(value)


def test_projected_cards_need_nonempty_verified_hashes():
    row = {'raw_text': 'A shows', 'row_id': 'a', 'card_row_id': 'a', 'cards': ['Ah', 'Ac'],
           'card_evidence_id': 'e', 'identity_status': 'CHAT_ROW_SHA_ROUNDTRIP_VERIFIED',
           'evidence_status': 'RAW_UNREVIEWED'}
    msg = make_message([row])
    assert msg.get('card_binding_conflict') is True
    assert msg['cards'] == []


def test_payout_class_digits_are_not_amount():
    msg = make_message([{'row_id': 'x', 'raw_text': 'Dealer: A wins 1 234 from the main pot with Two pairs, 6633'}])
    assert msg['amount'] == '1234'
    assert msg['showdown_class'] == 'TWO_PAIR'


def test_jev_multiline_candidate_revalidated():
    value = simple()
    value['messages'][5:6] = [make_message([{'row_id': 'first', 'raw_text': 'B gets uncalled bet'}]),
                               make_message([{'row_id': 'second', 'raw_text': '5 back'}])]
    decision = {'decision': 'LIKELY_MULTILINE_CONTINUATION', 'target_rows': ['first', 'second'],
                'evidence_rows': ['first', 'second'], 'confidence': .95}
    result = resolve_hand(value, JevResolver(lambda req: decision))
    assert result['final']['status'] == 'JEV_ASSISTED_CERTIFIED'
    assert result['resolution']['candidate_view']['messages'][5]['source_row_ids'] == ['first', 'second']


def test_jev_stale_raise_on_flop():
    value = basic('A raises to 30', 'B calls 20', 'Dealing Flop 2h 3d 4s',
                  'A raises to 30', 'A bets 20', 'B folds', 'A gets uncalled bet 20 back', 'A wins 60')
    decision = {'decision': 'LIKELY_STALE_DUPLICATE', 'target_rows': ['r7'], 'evidence_rows': ['r4'], 'confidence': .95}
    assert resolve_hand(value, JevResolver(lambda req: decision))['final']['status'] == 'JEV_ASSISTED_CERTIFIED'


def test_jev_previous_hand_payout_needs_boundary_proof():
    value = basic('A wins 15', 'A calls 5', 'B checks', 'B wins 20')
    decision = {'decision': 'STALE_PREVIOUS_HAND_ROW', 'target_rows': ['r4'], 'evidence_rows': ['previous:r99'], 'confidence': .99}
    result = resolve_hand(value, JevResolver(lambda req: decision))
    assert result['final']['status'] == 'UNRESOLVED_HAND'
    assert result['resolution']['rejection'] == 'UNSUPPORTED_OR_UNPROVEN_HYPOTHESIS'


def test_jev_card_conflict_cannot_edit_cards():
    value = runout('A calls 5', 'B checks')
    value['messages'][-3]['cards'] = ['2h', 'Ac']
    decision = {'decision': 'CARD_EVIDENCE_CONFLICT', 'target_rows': ['r15'], 'confidence': .99}
    before = copy.deepcopy(value)
    assert resolve_hand(value, JevResolver(lambda req: decision))['final']['status'] == 'UNRESOLVED_HAND'
    assert value == before


def test_parallel_same_hash_has_one_live_call(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from time import sleep
    value = missing_blind()
    def client(req):
        sleep(.01)
        return response_for(value)
    resolver = JevResolver(client, tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda ignored: resolve_hand(value, resolver), range(4)))
    assert resolver.live_calls == 1 and resolver.cache_hits == 3
    assert all(r['final']['status'] == 'JEV_ASSISTED_CERTIFIED' for r in results)


def test_native_request_preserves_whole_hand_without_key():
    from parser_core.jev_poker_resolver import JevHTTPClient
    from parser_core.poker_logic_engine import validate_hand
    value = missing_blind()
    initial = validate_hand(value)
    request = {'hand': value, 'conflicts': initial['issues'], 'math': initial['math'], 'players': initial['players']}
    encoded, options = JevHTTPClient({}).prepare_request(request)
    import json
    payload = json.loads(encoded)
    assert payload['model'] == 'typesafe-ai/jev'
    assert len(payload['state']['hand']['ordered_actions']) == len(value['messages'])
    assert payload['state']['hand']['ordered_actions'][0][5] == value['messages'][0]['raw_text']
    assert options[0]['decision'] == 'LIKELY_MISSING_AMOUNT'
    assert 'Authorization' not in encoded.decode()


def test_whole_hand_over_limit_is_not_truncated():
    from parser_core.jev_poker_resolver import JevHTTPClient
    value = missing_blind()
    value['messages'][0]['raw_text'] = 'x'*40000
    with pytest.raises(ValueError, match='WHOLE_HAND_EXCEEDS'):
        JevHTTPClient({}).prepare_request({'hand': value, 'conflicts': []})
