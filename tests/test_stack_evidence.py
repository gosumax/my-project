import copy
import json
from pathlib import Path

import pytest

from parser_core.stack_evidence import StackEvidenceProvider, evaluate_action, evaluate_start_stack
from parser_core.poker_reconstruction.models import Event, Player
from parser_core.poker_reconstruction.state_machine import State
from parser_core.poker_reconstruction_layer import write_reconstruction


def point(value=59400, frame=1005, **kwargs):
    return dict(table_id=1, seat_id='seat1', frame_index=frame, stack_value=value,
                status='CONFIRMED_NUMERIC', unknown_between=0, **kwargs)


def binding(**kwargs):
    return dict(source_identity=True, hand_membership=True, occupancy_epoch=True,
                single_action_interval=True, visibility_complete=True, player='A', action_frame=1004, **kwargs)


def state():
    return State(players={'A': Player(street_commit=2000), 'B': Player(street_commit=6000)},
                 phase='PREFLOP', current_bet=6000, min_raise=2000, pending={'A'})


def test_clean_call_and_state_immutable():
    s=state(); before=copy.deepcopy(s)
    r=evaluate_action(s, Event('CALL','A',None), point(100000,1000), point(96000), binding=binding())
    assert r['status']=='CONFIRMED' and r['paid_delta']==4000 and r['new_street_contribution']==6000
    assert s==before


def test_raise_to_uses_contribution():
    r=evaluate_action(state(), Event('RAISE','A',None,semantics='TO'),point(100000,1000),point(94000),binding=binding())
    assert r['candidate_amount']==8000 and r['event_kind']=='RAISE'


@pytest.mark.parametrize('case,reason', [('unknown','AGGREGATE_ONLY'),('increase','NO_TYPED_DEBIT'),
    ('missing','NUMERIC_UNKNOWN'),('wrong_table','ROUTE_MISMATCH'),('wrong_seat','ROUTE_MISMATCH'),
    ('late_frame','FRAME_ALIGNMENT_UNPROVEN'),('unbound','ACTION_BINDING_UNPROVEN'),('actor','ACTOR_MISMATCH')])
def test_delta_rejection(case,reason):
    a,b,k=point(100000,1000),point(96000),binding()
    if case=='unknown': b['unknown_between']=1
    if case=='increase': b['stack_value']=110000
    if case=='missing': b['stack_value']=None
    if case=='wrong_table': b['table_id']=2
    if case=='wrong_seat': b['seat_id']='seat2'
    if case=='late_frame': k['action_frame']=1010
    if case=='unbound': k['occupancy_epoch']=False
    if case=='actor': k['player']='B'
    assert evaluate_action(state(),Event('CALL','A',None),a,b,binding=k)['reason']==reason


def test_chat_conflict_and_invalid_call():
    for amount,value in [(5000,96000),(None,95000)]:
        assert evaluate_action(state(),Event('CALL','A',amount),point(100000,1000),point(value),binding=binding())['status']=='STACK_CONFLICT'


def test_occluded_number_and_ambiguous_status_cannot_confirm():
    proof=binding();proof['visibility_complete']=False
    assert evaluate_action(state(),Event('CALL','A',None),point(100000,1000),point(96000),binding=proof)['status']=='UNKNOWN'
    current=point(96000);current['status']='AMBIGUOUS'
    assert evaluate_action(state(),Event('CALL','A',None),point(100000,1000),current,binding=binding())['reason']=='NUMERIC_AMBIGUOUS'


def proof(debits=()):
    return dict(source_identity=True,boundary=True,occupancy_epoch=True,hand_membership=True,
                ordered_complete_debits=True,visibility_complete=True,hand_start_frame=1000,
                debits=list(debits))


def debit(kind,amount):
    return dict(kind=kind,amount=amount,frame=1001,source_refs=['independent-test-source'])


@pytest.mark.parametrize('posts,expected', [([debit('ANTE',100)],59500),
    ([debit('ANTE',100),debit('SB',250)],59750),([debit('ANTE',100),debit('BB',500)],60000),
    ([debit('ANTE',100),debit('CALL',5000)],64500)])
def test_start_stack_debits(posts,expected):
    p=proof(posts);p['accounting']=dict(complete=True,final_stack=expected-10000,debits=10000,returns=0,wins=0)
    r=evaluate_start_stack(point(),proof=p)
    assert r['calculated_start_stack']==expected and r['status']=='START_STACK_PROVEN'


@pytest.mark.parametrize('case', ['initial','unknown','midhand','late','unknown_call','accounting_conflict'])
def test_start_stack_rejects_uncertainty(case):
    p=proof([debit('ANTE',100)]);obs=point()
    if case=='initial': obs['origin']='INITIAL';p['boundary']=False
    if case=='unknown': p['unknown_before_observation']=True
    if case=='midhand': p['video_starts_mid_hand']=True
    if case=='late': p['hand_start_frame']=1010
    if case=='unknown_call': p['debits'].append(debit('CALL',None))
    if case=='accounting_conflict': p['accounting']=dict(complete=True,final_stack=1,debits=1,returns=0,wins=0)
    r=evaluate_start_stack(obs,proof=p)
    assert r['status']==('START_STACK_CONFLICT' if case=='accounting_conflict' else 'START_STACK_UNKNOWN')


def test_feature_off_does_not_load_or_emit_evidence(tmp_path):
    cfg=dict(enabled=True,allow_logic_math_output=False,max_nodes=100)
    write_reconstruction(tmp_path,[],config=cfg)
    assert not (tmp_path/'stack_evidence_summary.json').exists()
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    write_reconstruction(tmp_path,[],config=cfg,stack_provider=None)
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}


def test_real_sqlite_source_identity_and_no_actor_authority(tmp_path):
    root=Path(__file__).resolve().parents[1]
    db=root/'runs/stack_engine_integration_20261004/final_stack_evidence_v2/timeline.sqlite'
    if not db.exists(): pytest.skip('local recorded evidence unavailable')
    sha='27c0ba10544e0f6025e2d563baf7693d36c185cdbbfeb4517d3486116e9f8baf'
    provider=StackEvidenceProvider(db,video_sha256=sha)
    assert len(provider.points)==850
    report=provider.write_audit(tmp_path,[],start_stack=True)
    assert report['actor_bound_observations']==0 and report['recovered_proven']==0
    with pytest.raises(ValueError,match='identity mismatch'): StackEvidenceProvider(db,video_sha256='wrong')
