import copy
import json
import pytest
from decimal import Decimal as D
from pathlib import Path

from parser_core.poker_amount_semantics import resolve_call, with_call_context, call_start_stack
from parser_core.poker_hand_assembly import assemble_messages
from parser_core.poker_math_engine import audit_observed_ledger
from parser_core.poker_logic_engine import validate_hand
from parser_core.poker_reconstruction.assembly import candidate_assemblies
from parser_core.poker_reconstruction.reconstruct import reconstruct_hand
from parser_core.poker_reconstruction.timeline import parse_rows
from parser_core.poker_evidence_axes import evidence_axes
from parser_core.poker_reconstruction.accounting import pot_layers, payout_feasible
from parser_core.poker_reconstruction.models import Player

FIXTURE = json.loads((Path(__file__).parent/'fixtures/poker_math_smoke19_source.json').read_text('utf-8'))


def source(hand_id):
    data=copy.deepcopy(FIXTURE['hands'][hand_id])
    rows=[r for m in data['messages'] for r in m['source_rows']]
    for i,r in enumerate(rows):r['source_index']=i
    return rows


def hand(hand_id):
    rows=source(hand_id)
    messages=assemble_messages(rows,require_adjacency=True)
    return dict(hand_id=hand_id,messages=messages,start_observed=True,end_observed=True)


def economic(hand_id, *, require_sb=True):
    rows=source(hand_id)
    assemblies,exhausted,_=candidate_assemblies(rows)
    return reconstruct_hand(parse_rows(rows),hand_id=hand_id,assemblies=assemblies,
        assembly_exhausted=exhausted,strict_source_uniqueness=True,require_small_blind=require_sb)


def test_real_ante_and_raise_to_do_not_share_street_balance():
    h=hand('2878365630');raw=copy.deepcopy(h)
    ledger=audit_observed_ledger(h)
    raise_step=next(x for x in ledger['steps'] if x['action']=='RAISE' and x['raw_amount']=='10000')
    assert raise_step['street_contribution_after']=='10000'
    assert raise_step['ante_contribution']=='625'
    assert raise_step['total_hand_contribution']=='10625'
    assert h==raw


def test_real_bb5000_and_call5000_is_additional_payment():
    ledger=audit_observed_ledger(hand('2878365630'))
    call=next(x for x in ledger['steps'] if x['action']=='CALL' and
              x['raw_amount']=='5000' and x['street_contribution_before']=='5000')
    assert call['delta']=='5000' and call['street_contribution_after']=='10000'
    assert call['resolved_semantics']=='ADDED'


def test_ambiguous_short_allin_is_not_resolved_by_preference():
    p=resolve_call(D(10),D(5),D(30),all_in=True)
    assert p['status']=='AMBIGUOUS' and p['delta'] is None
    assert set(p['alternatives'].values())=={'5','10'}


def test_ordinary_call_contradiction_and_equivalent_zero_prior():
    assert resolve_call(D(5),D(5),D(20))['status']=='CONTRADICTION'
    assert resolve_call(D(10),D(0),D(10))['delta']=='10'


def test_real_t04_missing_ante_is_not_silently_filled():
    ledger=audit_observed_ledger(hand('2878365485'))
    assert ledger['delta']=='-70'
    assert not any(m['action']=='ANTE' and m['player']=='111' for m in hand('2878365485')['messages'])


def test_real_t07_numeric_candidates_cannot_bypass_unresolved_source():
    rows=source('2878367230');before=copy.deepcopy(rows)
    assemblies,exhausted,_=candidate_assemblies(rows)
    result=reconstruct_hand(parse_rows(rows),hand_id='2878367230',assemblies=assemblies,
        assembly_exhausted=exhausted,strict_source_uniqueness=True)
    assert not result['exhausted']
    candidates=[e for a in assemblies for e in a if e.source.get('numeric_amount_proof')]
    assert {e.amount for e in candidates}=={10000,100000}
    assert all(e.source['numeric_amount_proof']['parsed_amount']==100000 for e in candidates)
    # Real source has an unproved opaque payout tail. Pot closure alone must
    # not authorize dropping that row or promote the numeric hypothesis.
    assert result['solution_count']==0 and not result['kept_events']
    assert any(e.kind=='UNKNOWN' and e.source['raw_text']=='0+03030K'
               for a in assemblies for e in a)
    assert rows==before


def test_numeric_repair_never_crosses_source_frames():
    rows=source('2878367230')
    tail=next(r for r in rows if r['raw_ocr_text']=='10 0000')
    tail['frame_id']+=1
    assemblies,_,_=candidate_assemblies(rows)
    assert not any(e.source.get('numeric_amount_proof') for a in assemblies for e in a)


def test_unknown_stack_does_not_make_ordinary_money_action_fail():
    h=hand('2878365630')
    result=validate_hand(h)
    assert result['action_logic_pass'] and result['math']['pass']
    assert all('AMOUNT_SEMANTICS_UNPROVEN'!=i['code'] for i in result['issues'])


def test_real_no_sb_money_assessment_does_not_prove_dead_blind_or_full_hh():
    strict=economic('2878365655')
    money=economic('2878365655',require_sb=False)
    assert strict['solution_count']==0 and money['solution_count']==1
    assert money['calculated_pot']==money['observed_payout']==2820
    assert sum(money['uncalled_returns'].values())==1110
    h=hand('2878365655')
    record=dict(observed_hand=h,validation=validate_hand(h),diagnostics=dict(math=audit_observed_ledger(h)))
    axes=evidence_axes(record,dict(money_only_assessment=dict(money, full_hh_authority=False)))
    assert axes['pot_accounting']['status']=='PASS'
    assert axes['forced_blind_state']['status']=='UNKNOWN'
    assert axes['full_hh']['certified'] is False
    assert axes['seats']['status']==axes['start_stacks']['status']=='UNKNOWN'


@pytest.mark.parametrize('return_text,solutions', [('9 900',1),('',2)])
def test_numeric_repair_requires_unique_entire_trace(return_text,solutions):
    texts=['Dealer: A posts Small Blind 50','Dealer: B posts Big Blind 100',
           'Dealer: A raises to 10 0000','Dealer: B folds',
           'Dealer: A gets uncalled bet '+return_text+' back','Dealer: A wins 200']
    rows=[dict(source_index=i,stable_row_id=str(i),raw_ocr_text=text,frame_id=1,
               row_index=i,table_id='T01',session_id='test',epoch=1) for i,text in enumerate(texts)]
    assemblies,exhausted,_=candidate_assemblies(rows)
    result=reconstruct_hand(parse_rows(rows),hand_id='numeric-contract',assemblies=assemblies,
        assembly_exhausted=exhausted,strict_source_uniqueness=True)
    assert not result['exhausted'] and result['solution_count']==solutions
    assert result['status']==('RECOVERED_BY_LOGIC_MATH' if solutions==1 else 'AMBIGUOUS')
    if solutions==1:
        repaired=next(e for e in result['kept_events'] if e['source'].get('numeric_amount_proof'))
        assert repaired['amount']==10000
        assert result['calculated_pot']==result['observed_payout']==200


@pytest.mark.parametrize('hand_id,returned,layers', [
    ('2878366772',25000,[4375,51375,54500]),
    ('2878367004',108652,[3750,12500,10000,16875,331324])])
def test_real_allin_return_and_folded_money_layers(hand_id,returned,layers):
    h=hand(hand_id);ledger=audit_observed_ledger(h)
    assert ledger['delta']=='0' and int(ledger['total_returns'])==returned
    assert any(m['action']=='CALL' and m['all_in'] for m in h['messages'])
    folded={m['player'] for m in h['messages'] if m['action']=='FOLD'}
    players={n:Player(total_commit=int(p['gross_contribution']),returned=int(p['uncalled_return']),
                     status='FOLDED' if n in folded else 'ACTIVE') for n,p in ledger['players'].items()}
    pots=pot_layers(players)
    assert [p['amount'] for p in pots]==layers
    assert all(not (set(p['eligible']) & folded) for p in pots)
    assert payout_feasible(pots,{n:int(p['payout']) for n,p in ledger['players'].items()})


def contract_hand(amount=10, returned=15, payout=30):
    texts = ['Dealer: A posts Small Blind 5', 'Dealer: B posts Big Blind 30',
             f'Dealer: A calls {amount} and is All-in',
             f'Dealer: B gets uncalled bet {returned} back', f'Dealer: B wins {payout}']
    rows = [dict(stable_row_id=str(i), raw_ocr_text=t, frame_id=1, row_index=i,
                 table_id='T01', session_id='contract', epoch=1) for i, t in enumerate(texts)]
    return dict(table_id='T01', session_id='contract', hand_id='contract',
                start_observed=True, end_observed=True, messages=assemble_messages(rows),
                semantics=dict(call='UNPROVEN'))


def contract_receipt(value=15, **changes):
    return dict(dict(status='PROVEN', actor='A', seat='seat1', table_id=1,
                     session_id='contract', hand_id='contract', calculated_start_stack=value), **changes)


@pytest.mark.parametrize('meaning,returned,payout', [('ADDED',15,30), ('TOTAL',20,20)])
def test_call_contract_proven_not_reguessed(meaning, returned, payout):
    h = contract_hand(returned=returned, payout=payout)
    msg = h['messages'][2]
    msg['amount_proof'] = dict(rule='EXHAUSTIVE_UNIQUE_ORDERED_ECONOMIC_TRACE', kind='CALL',
        actor='A', source_row_ids=msg['source_row_ids'][:], derived_amount=10, call_contract=meaning)
    before = copy.deepcopy(h)
    checked = validate_hand(h)
    proof = checked['amount_resolution_proofs'][0]
    assert proof['status'] == 'RESOLVED' and proof['resolved_semantics'] == meaning
    assert proof['source'] == 'OBSERVED_CONTRACT' and list(proof['alternatives']) == [meaning]
    assert checked['math']['pass'] and audit_observed_ledger(h)['pass']
    assert not any(i['code'] in {'INVALID_CALL','AMOUNT_SEMANTICS_UNPROVEN'} for i in checked['issues'])
    assert h == before


@pytest.mark.parametrize('stack,meaning,delta', [(15,'ADDED','10'), (10,'TOTAL','5')])
def test_call_contract_unproven_unique_stack_context(stack, meaning, delta):
    h = with_call_context(contract_hand(), [contract_receipt(stack)])
    before = copy.deepcopy(h)
    proof = validate_hand(h)['amount_resolution_proofs'][0]
    step = audit_observed_ledger(h)['steps'][2]
    assert proof['status'] == 'RESOLVED' and proof['resolved_semantics'] == meaning
    assert proof['delta'] == step['delta'] == delta
    assert proof['stack_context']['remaining_before'] == delta
    assert h == before


def test_call_contract_two_legal_effects_stays_unproven():
    h = contract_hand()
    checked = validate_hand(h)
    assert checked['amount_resolution_proofs'][0]['status'] == 'AMBIGUOUS'
    assert any(i['code'] == 'AMOUNT_SEMANTICS_UNPROVEN' for i in checked['issues'])
    assert audit_observed_ledger(h)['suspicious_rows'][0]['RULE_TRIGGER'] == 'AMOUNT_SEMANTICS_UNPROVEN'
    # A known capacity alone does not select one of two short all-in effects;
    # exact exhaustion is required. With no all-in flag both short effects fail.
    p = resolve_call(D(10), D(5), D(30), all_in=True, start_stack=100)
    assert p['status'] == 'AMBIGUOUS' and set(p['alternatives']) == {'ADDED','TOTAL'}


def test_call_contract_return_context_and_exact_exhaustion():
    p = resolve_call(D(10), D(5), D(30), all_in=True,
                     start_stack=20, total_contribution=20, returned=10)
    assert p['status'] == 'RESOLVED' and p['resolved_semantics'] == 'ADDED'
    assert p['stack_context']['remaining_before'] == '10'
    h = with_call_context(contract_hand(), [contract_receipt(100)])
    assert any(i['code'] == 'INVALID_CALL' for i in validate_hand(h)['issues'])


@pytest.mark.parametrize('change', [dict(seat='seat2'), dict(calculated_start_stack=16),
    dict(identity_proof=dict(actor='B',seat='seat1'))])
def test_call_contract_conflicting_receipts_cannot_choose(change):
    h = with_call_context(contract_hand(), [contract_receipt(), contract_receipt(**change)])
    assert call_start_stack(h, 'A') is None
    assert validate_hand(h)['amount_resolution_proofs'][0]['status'] == 'AMBIGUOUS'


def test_call_contract_stale_proof_cannot_reguess():
    h = hand('2878365630')
    msg = next(m for m in h['messages'] if m['action'] == 'CALL')
    msg['amount_proof'] = dict(rule='EXHAUSTIVE_UNIQUE_ORDERED_ECONOMIC_TRACE', kind='CALL',
        actor='wrong', source_row_ids=msg['source_row_ids'][:], derived_amount=msg['amount'], call_contract='ADDED')
    proof = validate_hand(h)['amount_resolution_proofs'][0]
    assert proof['status'] == 'CONTRADICTION' and not proof['alternatives']


def test_call_contract_existing_ready_call_unchanged():
    h = hand('2878365630'); before = copy.deepcopy(h)
    checked = validate_hand(h)
    assert checked['action_logic_pass'] and checked['math']['pass']
    proof = next(p for p in checked['amount_resolution_proofs'] if p['street_contribution_before'] == '5000')
    assert proof['resolved_semantics'] == 'ADDED' and proof['delta'] == '5000'
    assert h == before


def test_call_contract_fallback_retains_proved_candidate():
    from parser_core.reconstruction_validation import validate_call_fallback
    derived = dict(accepted=False, candidate_hand=contract_hand(), validation={'sentinel':'proof'},
                   diagnostics={'sentinel':'proof'}, reason='OTHER_BLOCKER')
    result = validate_call_fallback(contract_hand(), dict(derived_final_validation=derived))
    assert result == dict(validation=derived['validation'], diagnostics=derived['diagnostics'],
                         diagnostics_ledger_view='SOURCE_PROVEN_DERIVED')


def test_call_contract_fallback_passes_saved_context_without_trace():
    from parser_core.reconstruction_validation import validate_call_fallback
    h = contract_hand(); before = copy.deepcopy(h)
    result = validate_call_fallback(h, dict(simple_start_stack_join_receipts=[contract_receipt()]))
    proof = result['validation']['amount_resolution_proofs'][0]
    assert proof['resolved_semantics'] == 'ADDED'
    assert result['diagnostics']['math']['steps'][2]['delta'] == '10'
    assert h == before
