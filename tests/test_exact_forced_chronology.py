"""Real immutable Smoke19 exact-label chronology and fail-closed mutations."""
import copy
import json
from pathlib import Path

import pytest

from parser_core.exact_forced_chronology import RULE, prove
from parser_core.seat_epoch_state import SeatEpochMachine


@pytest.fixture(scope='module')
def source_case():
    return json.loads((Path(__file__).parent/'fixtures/exact_forced_chronology_source.json').read_text('utf-8'))


def run(case):
    return prove(case['fact'], case['snapshot'], case['matched'], case['inputs'])


def target_rows(case):
    receipt=case['matched'][0]['canonical_provenance']['binding_receipt']
    frames=receipt['label_evidence'][0]['frames']
    return [r for r in case['inputs']['nick_rows'] if r['seat']==receipt['seat'] and r['source_frame'] in frames]


def test_real_t01_exact_chronology(source_case):
    before=copy.deepcopy(source_case)
    proof=run(source_case)
    assert proof and proof['rule']==RULE
    assert (proof['actor'],proof['seat'],proof['epoch'])==('ibnemelian','seat3','T01_SEAT3_EPOCH_1')
    assert proof['forced_source_frame']==2108
    assert proof['first_numeric_source_frame']==2132
    assert proof['binding_anchor_frame']==2134
    assert source_case==before


@pytest.mark.parametrize('mutation',[
    'nick_hash','forced_hash','context','own_posts','frame_pts','epoch',
    'replacement','missing_triple','ambiguous_label','first_visible','unknown_state',
])
def test_real_source_mutations_fail_closed(source_case, mutation):
    case=copy.deepcopy(source_case)
    inputs=case['inputs'];fact=case['fact']
    if mutation=='nick_hash':
        target_rows(case)[0]['pixel_sha256']='0'*64
    elif mutation=='forced_hash':
        inputs['forced_rows'][0]['source_ref']='0'*64
    elif mutation=='context':
        inputs['context'][2]='DIFFERENT_HAND'
    elif mutation=='own_posts':
        own=next(p for p in case['snapshot']['posts'] if p['actor']=='ibnemelian')
        own['amount']+=1
    elif mutation=='frame_pts':
        inputs['forced_rows'][0]['source_pts']+=1
    elif mutation=='epoch':
        fact['occupancy_epoch']='T01_SEAT3_EPOCH_2'
    elif mutation in {'replacement','unknown_state'}:
        e=dict(table=fact['table'],session=fact['session'],seat=fact['seat'],kind='REPLACE',
               frame=2109,actor='replacement_actor',hand_id=fact['hand_id'],
               evidence_status='PROVEN' if mutation=='replacement' else 'UNKNOWN',
               source_ref='audit_negative_only',source_index=100000)
        machine=SeatEpochMachine()
        for t in inputs['hand_triggers']:
            machine.hand(t['table_id'],t['session_id'],t['hand_id'],t['source_frame'])
        for s in [*inputs['state_receipts'],e]:
            machine.observe(s['table'],s['session'],s['seat'],s['kind'],s['frame'],
                            **{k:s[k] for k in ('actor','hand_id','evidence_status','source_ref','source_index')})
        machine.build();inputs['state_receipts']=machine.audit
    elif mutation=='missing_triple':
        missing=target_rows(case)[0]
        inputs['nick_rows'].remove(missing)
    elif mutation=='ambiguous_label':
        receipt=case['matched'][0]['canonical_provenance']['binding_receipt']
        frames=receipt['label_evidence'][0]['frames']
        rows=[r for r in inputs['nick_rows'] if r['seat']=='seat5' and r['source_frame'] in frames]
        assert len(rows)==3
        for r in rows:
            r['ocr_response'].update(status='OK',raw_text=receipt['actor'],score=.999)
    elif mutation=='first_visible':
        fact['first_visible_proof']['status']='UNKNOWN'
    assert run(case) is None

@pytest.mark.parametrize('mutation',['snapshot_context','event_context','post_context','post_epoch','post_seat','late_occupied','deal_pixel'])
def test_source_context_and_deal_occupancy_fail_closed(source_case, mutation):
    case=copy.deepcopy(source_case)
    if mutation=='snapshot_context':case['snapshot']['hand_id']='DIFFERENT_HAND'
    elif mutation=='event_context':case['matched'][0]['hand_id']='DIFFERENT_HAND'
    elif mutation=='post_context':case['matched'][0]['canonical_provenance']['post']['hand_id']='DIFFERENT_HAND'
    elif mutation=='post_epoch':case['matched'][0]['canonical_provenance']['post']['epoch']='WRONG_EPOCH'
    elif mutation=='post_seat':case['matched'][0]['canonical_provenance']['post']['seat']='seat8'
    elif mutation=='deal_pixel':case['inputs']['deal']['table_pixel_sha256']='0'*64
    else:
        inputs=case['inputs'];fact=case['fact'];deal=inputs['deal']
        for e in inputs['state_receipts']:
            if (e['table'],e['session'],e['hand_id'],e['seat'],e['kind'],e['frame'])==(fact['table'],fact['session'],fact['hand_id'],fact['seat'],'OCCUPIED',deal['source_frame']):
                e['frame']=deal['source_frame']+1
        machine=SeatEpochMachine()
        for t in inputs['hand_triggers']:machine.hand(t['table_id'],t['session_id'],t['hand_id'],t['source_frame'])
        for e in inputs['state_receipts']:
            machine.observe(e['table'],e['session'],e['seat'],e['kind'],e['frame'],
                **{k:e[k] for k in ('actor','hand_id','evidence_status','source_ref','source_index')})
        machine.build();inputs['state_receipts']=machine.audit
    assert run(case) is None


@pytest.mark.parametrize('conflict',[False,True])
def test_load_sources_duplicate_deal_is_order_independent(source_case,tmp_path,conflict):
    from parser_core.exact_forced_chronology import load_sources
    case=copy.deepcopy(source_case);i=case['inputs'];deal=i['deal'];duplicate=copy.deepcopy(deal)
    if conflict:duplicate['table_pixel_sha256']='0'*64
    rows={
        'seat_forced_inputs.jsonl':[*i['nick_rows'],*i['forced_rows']],
        'dealing_cards_triggers.jsonl':i['hand_triggers'],
        'start_stack_live_receipts.jsonl':[deal,duplicate,deal],
    }
    for name,values in rows.items():
        (tmp_path/name).write_text(''.join(json.dumps(v)+'\n' for v in values),'utf-8')
    (tmp_path/'seat_forced_summary.json').write_text(json.dumps(dict(state_receipts=i['state_receipts'],unresolved_state_events=i['unresolved_events'])),'utf-8')
    loaded=load_sources(tmp_path);case['inputs']=loaded[tuple(i['context'])]
    if conflict:
        assert case['inputs']['deal'] is None
        assert run(case) is None
    else:assert run(case)


def test_exact_reconcile_and_start_stack_preserve_raw_sources(source_case):
    from decimal import Decimal
    from parser_core.canonical_events import reconcile_stack_facts
    from parser_core.poker_math_engine import calculate_start_stack
    case=copy.deepcopy(source_case);before=copy.deepcopy(case)
    args=dict(forced_snapshots=[case['snapshot']],first_visible_contract=True)
    without=reconcile_stack_facts(case['fact'],case['matched'],**args)
    assert without['status']=='UNKNOWN'
    fixed=reconcile_stack_facts(case['fact'],case['matched'],exact_binding_sources=case['inputs'],**args)
    assert fixed['status']=='PROVEN_FOR_MATH'
    assert fixed['forced_chronology_proof']['rule']==RULE
    assert fixed['vision_fact']==case['fact']
    value=calculate_start_stack(fixed,first_visible_contract=True)
    expected=Decimal(case['fact']['displayed_stack'])+sum(Decimal(str(v)) for v in fixed['forced_bets'].values())
    assert value['status']=='PASS' and Decimal(value['START_STACK'])==expected
    assert value['vision_fact']==case['fact'] and case==before
    bad=copy.deepcopy(case);bad['inputs']['context'][2]='DIFFERENT_HAND'
    withheld=reconcile_stack_facts(bad['fact'],bad['matched'],exact_binding_sources=bad['inputs'],**args)
    assert withheld['status']=='UNKNOWN'

@pytest.mark.parametrize('mutation',['proof_anchor','proof_context','forced_bets','fact_context','canonical_ids'])
def test_start_stack_reproduces_exact_proof_at_consumption(source_case,mutation):
    from parser_core.canonical_events import reconcile_stack_facts
    from parser_core.poker_math_engine import calculate_start_stack
    from parser_core.exact_forced_chronology import verify
    fixed=reconcile_stack_facts(source_case['fact'],source_case['matched'],
        forced_snapshots=[source_case['snapshot']],first_visible_contract=True,exact_binding_sources=source_case['inputs'])
    assert verify(fixed)
    bad=copy.deepcopy(fixed)
    if mutation=='proof_anchor':bad['forced_chronology_proof']['binding_anchor_frame']+=1
    elif mutation=='proof_context':bad['forced_chronology_proof']['exact_binding_inputs']['context'][2]='OTHER_HAND'
    elif mutation=='forced_bets':bad['forced_bets']['ANTE']+=1
    elif mutation=='fact_context':bad['hand_id']='OTHER_HAND'
    else:bad['canonical_forced_event_ids'].append('invented_event')
    assert not verify(bad)
    result=calculate_start_stack(bad,first_visible_contract=True)
    assert result['status']=='UNKNOWN' and result['START_STACK'] is None
    assert result['blocker']=='EXACT_FORCED_CHRONOLOGY_NOT_REPRODUCIBLE'

@pytest.mark.parametrize('score',[1.1,float('inf'),True])
def test_forced_ocr_scores_must_be_finite_probability(source_case,score):
    case=copy.deepcopy(source_case)
    row=case['inputs']['forced_rows'][0]
    row['score']=score;row['ocr_response']['score']=score
    assert run(case) is None


@pytest.mark.parametrize('mutation',['missing_proof','unknown_rule','renamed_old_rule','truth_only','correction_only','chronology_only'])
def test_exact_discriminators_withhold_missing_or_unproved_chronology(source_case,mutation):
    from parser_core.canonical_events import reconcile_stack_facts
    from parser_core.poker_math_engine import calculate_start_stack
    fixed=reconcile_stack_facts(source_case['fact'],source_case['matched'],
        forced_snapshots=[source_case['snapshot']],first_visible_contract=True,exact_binding_sources=source_case['inputs'])
    assert fixed['forced_truth']=='CANONICAL_COMPLETE_ACTOR_POSTS_EXACT_LABEL'
    assert fixed['correction_proof']['rule']=='COMPLETE_PHYSICAL_ACTOR_POST_SET_EXACT_LABEL'
    bad=copy.deepcopy(fixed)
    if mutation=='missing_proof':bad.pop('forced_chronology_proof')
    elif mutation=='unknown_rule':bad['forced_chronology_proof']['rule']='UNKNOWN'
    elif mutation=='renamed_old_rule':bad['forced_chronology_proof']['rule']='COMPLETE_FORCED_PREFIX_OBSERVED_BY_FIRST_POSTFORCED_NUMBER_V1'
    elif mutation=='truth_only':
        bad.pop('correction_proof');bad.pop('forced_chronology_proof')
    elif mutation=='correction_only':
        bad['forced_truth']='CANONICAL_EVENTS';bad.pop('forced_chronology_proof')
    else:
        bad['forced_truth']='CANONICAL_EVENTS';bad.pop('correction_proof')
        bad['forced_chronology_proof']['binding_anchor_frame']+=1
    result=calculate_start_stack(bad,first_visible_contract=True)
    assert result['status']=='UNKNOWN' and result['START_STACK'] is None
    assert result['blocker']=='EXACT_FORCED_CHRONOLOGY_NOT_REPRODUCIBLE'

@pytest.mark.parametrize('score',[1.1,float('inf'),True])
def test_nickname_ocr_scores_must_be_finite_probability(source_case,score):
    case=copy.deepcopy(source_case)
    target_rows(case)[0]['ocr_response']['score']=score
    assert run(case) is None
