"""HH presentation/axis contract units; these do not certify native SitOut."""
import copy
import json
from types import SimpleNamespace
import pytest
from parser_core.hh_text import render_complete
from parser_core.hh_export import export_hand_histories
from parser_core.poker_evidence_axes import evidence_axes
from parser_core.sitout_hand_evidence import HH_CONTRACT,RULE


def example():
    receipt=dict(rule=RULE,proof_id='unit-schema-only',actor='A',seat='seat1',table_id=1,
        session_id='s',hand_id='123456',hh_contract=HH_CONTRACT,numeric_stack_authority=False,
        synthetic_fold_authority=False,player_status='SIT_OUT')
    return dict(status='COMPLETE',certified=True,blockers=[],table='T01',session='s',
        hand_number='123456',hh_contract=HH_CONTRACT,final_pot=10,
        players=[dict(player='A',seat='seat1',START_STACK=None,START_STACK_status='UNKNOWN',
            player_status='SIT_OUT',stack_requirement='NOT_REQUIRED_SIT_OUT',sitout_binding_receipt=receipt),
            dict(player='B',seat='seat2',START_STACK=1000,START_STACK_status='PROVEN')],
        actions=[dict(action='SMALL_BLIND',player='A',amount=5),dict(action='BIG_BLIND',player='B',amount=10),
            dict(action='DEAL'),dict(action='FOLD',player='A'),
            dict(action='UNCALLED_RETURN',player='B',amount=5),dict(action='WIN',player='B',amount=10)])


def test_sitout_text_has_no_amount_or_invented_action():
    hand=example();before=copy.deepcopy(hand);text=render_complete(hand)
    assert 'Seat 1: A (sitting out)\n' in text
    assert 'Seat 2: B (1000 in chips)\n' in text
    assert 'A: posts small blind 5\n' in text and text.count('A: folds\n')==1
    assert 'Seat 1: A (small blind) (sitting out) folded before Flop' in text
    assert 'UNKNOWN' not in text and '(None in chips)' not in text and '(0 in chips)' not in text
    assert hand==before


@pytest.mark.parametrize('mutation',['no_contract','normal_unknown','zero_stack','numeric_authority','synthetic_fold',
    'foreign_actor','foreign_seat','foreign_hand','foreign_table','foreign_session','missing_fold','voluntary_call','all_in'])
def test_sitout_cannot_disguise_unknown_player_or_action(mutation):
    hand=example();p=hand['players'][0];r=p['sitout_binding_receipt']
    if mutation=='no_contract':hand.pop('hh_contract')
    elif mutation=='normal_unknown':p.pop('player_status')
    elif mutation=='zero_stack':p['START_STACK']=0
    elif mutation=='numeric_authority':r['numeric_stack_authority']=True
    elif mutation=='synthetic_fold':r['synthetic_fold_authority']=True
    elif mutation=='foreign_actor':r['actor']='B'
    elif mutation=='foreign_seat':r['seat']='seat2'
    elif mutation=='foreign_hand':r['hand_id']='654321'
    elif mutation=='foreign_table':r['table_id']=2
    elif mutation=='foreign_session':r['session_id']='other'
    elif mutation=='missing_fold':hand['actions']=[a for a in hand['actions'] if a['action']!='FOLD']
    elif mutation=='voluntary_call':hand['actions'].insert(3,dict(action='CALL',player='A',amount=5))
    elif mutation=='all_in':hand['actions'][0]['all_in']=True
    before=copy.deepcopy(hand)
    with pytest.raises(ValueError,match='PROVEN_SEAT_AND_STACK_REQUIRED'):render_complete(hand)
    assert hand==before


def record_for(hand):
    messages=copy.deepcopy(hand['actions'])
    for action in messages:
        action['raw_text']='UNIT_SCHEMA_FIXTURE '+action['action']
    return dict(segment_id='s:1',lifecycle=dict(status='COMPLETE'),fully_validated=True,
        observed_hand=dict(table_id='T01',session_id='s',hand_id=hand['hand_number'],
            messages=messages,start_observed=True,end_observed=True,boundary_conflicts=[]),
        validation=dict(action_logic_status='PASS',card_status='PASS',identity_status='PASS'),
        diagnostics=dict(math=dict(**{'pass':True})))


def test_seat_proof_is_separate_from_unknown_numeric_stack_and_does_not_certify_itself():
    record=record_for(example());numeric={'B':dict(seat_id='seat2',calculated_start_stack=1000)}
    seats={**numeric,'A':dict(seat_id='seat1',calculated_start_stack=None)}
    axes=evidence_axes(record,{},proven_stacks=numeric,proven_seats=seats)
    assert axes['seats']['status']=='PASS' and axes['start_stacks']['status']=='UNKNOWN'
    assert axes['full_hh']['status']=='UNKNOWN' and axes['full_hh']['certified'] is False
    axes=evidence_axes(record,{},proven_stacks=numeric,proven_seats=seats,certified=True,hh_contract=HH_CONTRACT)
    assert axes['full_hh']==dict(status='PASS',certified=True,contract=HH_CONTRACT)
    assert axes['start_stacks']['status']=='UNKNOWN'


@pytest.mark.parametrize('provider',[None,SimpleNamespace(for_hand=lambda hand:[])])
def test_saved_sitout_receipt_without_fresh_source_authority_cannot_certify(tmp_path,provider):
    hand=example();record=record_for(hand);receipt=hand['players'][0]['sitout_binding_receipt']
    row=dict(segment_id=record['segment_id'],player='A',seat_id='seat1',table_id='T01',session_id='s',
        hand_id=hand['hand_number'],status='START_STACK_UNKNOWN',calculated_start_stack=None,
        observed_stack=None,source_observation_id=None,source_state_id=None,occupancy_epoch=None,
        player_status='SIT_OUT',stack_requirement='NOT_REQUIRED_SIT_OUT',sitout_binding_receipt=receipt)
    recon=dict(segment_id=record['segment_id'],output_eligible=True,output_blockers=[],
        payout_eligible=True,calculated_pot=10)
    for name,value in [('stack_start_stack.jsonl',row),('poker_reconstruction.jsonl',recon)]:
        (tmp_path/name).write_text(json.dumps(value)+'\n','utf-8')
    summary=export_hand_histories(tmp_path,[record],sitout_provider=provider)
    saved=json.loads((tmp_path/'hand_histories.jsonl').read_text('utf-8'))
    assert not saved['certified'] and 'hh_contract' not in saved
    assert 'ALL_PLAYER_SEATS_AND_START_STACKS_NOT_PROVEN' in saved['blockers']
    assert saved['players'][0]['START_STACK'] is None
    assert summary['ready_hh_text_count']==0 and (tmp_path/'hand_histories.txt').read_text('utf-8')==''
