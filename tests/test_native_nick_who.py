"""Intrinsic native guards and explicitly synthetic resealed interval controls.

Resealed in-memory fixtures exercise guards; they are not new native proofs.
"""
import copy,json
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
import pytest
from parser_core.native_nick_who import NativeNickWhoSource,validate_native_nick,hand_key
from parser_core.chat_nick_actor import RULE,CROSS_RULE,SOURCE_KEY,literal_constraint
INPUT=json.loads((Path(__file__).parent/'fixtures/native_who_guard_inputs.json').read_text('utf8'))

def inventory(point,hand=None):
    point=copy.deepcopy(point);key=hand_key(point);anchor=point['anchor_frame'];native=NativeNickWhoSource.__new__(NativeNickWhoSource)
    hand=copy.deepcopy(hand or dict(table_id=f'T{key[0]:02d}',session_id=key[1],hand_id=key[2],messages=[]))
    native.hands={key:hand};native.hashes=copy.deepcopy(point['source_files_sha256']);native.video_sha256=INPUT['new_native']['source_video_sha256']
    native.bounds={key:(anchor-100,max(anchor+300,point.get('identity_interval',[0,0])[1]))}
    native.nicks=defaultdict(list,{key:[(copy.deepcopy(point['nickname_anchor_source']),copy.deepcopy(point['match_constraint']))]})
    native.errors=defaultdict(list);native.records=defaultdict(list,{key:copy.deepcopy(point['chat_actor_sources'])})
    native.checkpoints=defaultdict(list,{key[:2]:[dict(frame=native.bounds[key][0],hand_id=key[2])]})
    native.summary={name:[] for name in ['forced_snapshots','bound_posts','state_receipts','unresolved_state_events']};native.services=[]
    native.source_lines={(*key,point['seat'],anchor):copy.deepcopy(point.get('native_nick_source',{}))}
    native._inventory_seals={key:native._inventory_digest(key)}
    context=dict(table_id=key[0],session_id=key[1],hand_id=key[2],header_frame=native.bounds[key][0],capture_end_inclusive=anchor+64)
    context[SOURCE_KEY]=dict(policy=RULE,cross_source_policy=CROSS_RULE,source_sha256=copy.deepcopy(native.hashes))
    return native,key,context,point

def reseal(native,key):native._inventory_seals[key]=native._inventory_digest(key)

@pytest.mark.parametrize('policy,expected_rule',[(None,CROSS_RULE),
    ('FIRST_QUALIFIED_NICK_FRAME_PER_HAND_V1',CROSS_RULE),
    ('FIRST_QUALIFIED_NICK_FRAME_PER_HAND_V2',RULE)])
def test_explicit_nick_policy_keeps_legacy_floor_and_accepts_v2_90(policy,expected_rule):
    from parser_core.nickname_policy import nickname_witness_floor
    native,key,context,point=inventory(INPUT['new_native'],INPUT['new_hand'])
    native.nickname_witness_policy=policy
    native.nickname_confidence=nickname_witness_floor(policy)
    native.nicks[key][0][0]['ocr_response']['score']=.90
    reseal(native,key)
    result=native.for_actor(context,point['actor'])
    assert result['status']=='PROVEN',result
    assert result['nick_matching_rule']==expected_rule
    if policy is None:
        assert 'nickname_witness_policy' not in result
    else:
        assert result['nickname_witness_policy']==policy
        assert result['nickname_confidence_min']==nickname_witness_floor(policy)

def state(native,key,point,kind,frame,actor=None,epoch=None):
    row=dict(table=key[0],session=key[1],hand_id=key[2],seat=point['seat'],kind=kind,frame=frame,status='APPLIED',evidence_status='PROVEN',actor=point['actor'] if actor is None else actor)
    if epoch is not None:row['epoch']=epoch
    native.summary['state_receipts'].append(row);reseal(native,key)

@pytest.mark.parametrize('field,value',[('table_id',99),('session_id','FOREIGN'),('hand_id','FOREIGN')])
def test_foreign_native_context_has_no_who_authority(field,value):
    native,key,c,p=inventory(INPUT['new_native'],INPUT['new_hand']);c[field]=value
    assert native.for_actor(c,p['actor'])['status']=='UNKNOWN'

@pytest.mark.parametrize('field',['policy','cross_source_policy','source_sha256'])
def test_explicit_policy_and_original_parent_required(field):
    native,key,c,p=inventory(INPUT['new_native'],INPUT['new_hand'])
    if field=='source_sha256':c[SOURCE_KEY][field][next(iter(native.hashes))]='0'*64
    else:c[SOURCE_KEY][field]='FOREIGN_POLICY'
    assert native.for_actor(c,p['actor'])['status']=='UNKNOWN'

def test_new_rule_without_fresh_inventory_cannot_bind_numeric():
    from parser_core.chat_nick_actor import single_numeric_identity_binding
    native,key,c,p=inventory(INPUT['new_native'],INPUT['new_hand'])
    with pytest.raises(ValueError):single_numeric_identity_binding(c,p,p['anchor_frame'],start_frame=p['anchor_frame'],end_frame=p['anchor_frame'])

def test_post_replacement_synthetic_actor_action_denies_hand_wide_scope():
    native,key,c,p=inventory(INPUT['new_native'],INPUT['new_hand']);barrier=native.bounds[key][1]-10
    state(native,key,p,'REPLACE',barrier)
    message=copy.deepcopy(next(m for m in native.hands[key]['messages'] if m.get('player')==p['actor']))
    message['source_rows'][0]['frame_id']=barrier+1;native.hands[key]['messages'].append(message);reseal(native,key)
    scope=native.scope_for_point(native.hands[key],p)
    assert scope['status']=='UNKNOWN' and scope['reason']=='ACTOR_ACTION_OUTSIDE_IDENTITY_INTERVAL'

@pytest.mark.parametrize('mutation',['pixels','roi','seat','item_id','frame','pts','time_base','table'])
def test_intrinsic_native_row_rejects_independent_corruption(mutation):
    row=copy.deepcopy(INPUT['new_native']['nickname_anchor_source'])
    if mutation=='pixels':row['pixel_sha256']='0'*64
    elif mutation=='roi':row['roi'][0]+=1
    elif mutation=='seat':row['seat']='seat1'
    elif mutation=='item_id':row['ocr_response']['item_id']='foreign'
    elif mutation=='frame':row['source_frame']=True
    elif mutation=='pts':row['source_pts']=-1
    elif mutation=='time_base':row['time_base']=[1,0]
    else:row['table_id']=True
    with pytest.raises((ValueError,KeyError,TypeError,ZeroDivisionError)):validate_native_nick(row)

@pytest.mark.parametrize('text',['PENSEAT','EMPTY','Open Seat','Ante 70'])
def test_ui_and_overlay_have_no_nickname_authority(text):
    row=copy.deepcopy(INPUT['new_native']['nickname_anchor_source']);row['ocr_response']['raw_text']=text
    assert validate_native_nick(row)[1] is None

@pytest.mark.parametrize('field',['nicks','summary','services','checkpoints','bounds','hashes','hand'])
def test_cached_parent_mutations_rejected_without_reseal(field):
    native,key,c,p=inventory(INPUT['new_native'],INPUT['new_hand'])
    if field=='nicks':native.nicks[key][0][0]['source_pts']+=1
    elif field=='summary':native.summary['state_receipts'].append(dict(table=key[0],session=key[1],frame=p['anchor_frame'],seat=p['seat']))
    elif field=='services':native.services.append(dict(table_id=key[0],session_id=key[1],hand_id=key[2],actor=p['actor'],frame=p['anchor_frame'],kind='LEAVE'))
    elif field=='checkpoints':native.checkpoints[key[:2]][0]['hand_id']='FOREIGN'
    elif field=='bounds':native.bounds[key]=(native.bounds[key][0]+1,native.bounds[key][1])
    elif field=='hashes':native.hashes[next(iter(native.hashes))]='0'*64
    else:native.hands[key]['hand_id']='FOREIGN'
    assert native.for_actor(c,p['actor'])['status']=='UNKNOWN'
    assert native.scope_for_point(native.hands[key],p)['status']=='UNKNOWN'

@pytest.mark.parametrize('which',['old_direct','old_cross'])
@pytest.mark.parametrize('kind',['SIT_IN','SIT_OUT'])
def test_same_actor_same_seat_status_preserves_old_numeric_interval(which,kind):
    native,key,c,p=inventory(INPUT[which]);a=p['anchor_frame'];state(native,key,p,kind,a+70)
    native.assert_use_interval(c,p,[a+80],start_frame=a,end_frame=a+80)

@pytest.mark.parametrize('which',['old_direct','old_cross'])
@pytest.mark.parametrize('kind,actor',[('REPLACE',None),('has left the table',None),('SIT_IN','different_player')])
def test_identity_replacement_blocks_later_numeric_interval(which,kind,actor):
    native,key,c,p=inventory(INPUT[which]);a=p['anchor_frame'];state(native,key,p,kind,a+70,actor=actor)
    native.assert_use_interval(c,p,[a+69],start_frame=a,end_frame=a+69)
    with pytest.raises(ValueError):native.assert_use_interval(c,p,[a+80],start_frame=a,end_frame=a+80)

@pytest.mark.parametrize('which',['old_direct','old_cross'])
def test_prehand_epoch_change_is_identity_barrier(which):
    native,key,c,p=inventory(INPUT[which]);a=p['anchor_frame'];low=native.bounds[key][0]
    state(native,key,p,'OBSERVATION',low-1,epoch='A');state(native,key,p,'OBSERVATION',a+70,epoch='B')
    with pytest.raises(ValueError):native.assert_use_interval(c,p,[a+80],start_frame=a,end_frame=a+80)

@pytest.mark.parametrize('which',['old_direct','old_cross'])
@pytest.mark.parametrize('conflict',['other_nickname','second_seat','prefix_collision'])
def test_late_qualified_nickname_conflict_blocks_old_interval(which,conflict):
    native,key,c,p=inventory(INPUT[which]);a=p['anchor_frame'];row=copy.deepcopy(p['nickname_anchor_source']);row['source_frame']=a+70;row['ocr_response']['score']=.99
    if conflict=='other_nickname':text='different_player'
    elif conflict=='second_seat':row['seat']='seat9' if p['seat']!='seat9' else 'seat1';text=p['actor']
    else:
        text=p['actor'][:3]+'..';other=copy.deepcopy(native.records[key][0]);other['actor']=p['actor'][:3]+'OTHER';native.records[key].append(other)
    row['ocr_response']['raw_text']=text;native.nicks[key].append((row,literal_constraint(text)));reseal(native,key)
    with pytest.raises(ValueError):native.assert_use_interval(c,p,[a+80],start_frame=a,end_frame=a+80)

@pytest.mark.parametrize('which',['old_direct','old_cross'])
def test_weak_unmatched_nickname_does_not_block_interval(which):
    native,key,c,p=inventory(INPUT[which]);a=p['anchor_frame'];row=copy.deepcopy(p['nickname_anchor_source']);row['source_frame']=a+70;row['ocr_response'].update(score=.4,raw_text='unmatched')
    native.nicks[key].append((row,literal_constraint('unmatched')));reseal(native,key)
    native.assert_use_interval(c,p,[a+80],start_frame=a,end_frame=a+80)

@pytest.mark.parametrize('which',['old_direct','old_cross'])
def test_conflict_after_replacement_does_not_poison_prior_segment(which):
    native,key,c,p=inventory(INPUT[which]);a=p['anchor_frame'];state(native,key,p,'REPLACE',a+70)
    row=copy.deepcopy(p['nickname_anchor_source']);row['source_frame']=a+80;row['ocr_response'].update(score=.99,raw_text='different_player')
    native.nicks[key].append((row,literal_constraint('different_player')));reseal(native,key)
    native.assert_use_interval(c,p,[a+69],start_frame=a,end_frame=a+69)

def test_hh_scope_denial_removes_actor_and_numeric_authority(tmp_path):
    from parser_core.hh_export import export_hand_histories
    record=dict(segment_id='synthetic:1',lifecycle=dict(status='COMPLETE'),fully_validated=True,observed_hand=dict(table_id='T01',session_id='s',hand_id='100',start_observed=True,end_observed=True,boundary_conflicts=[],messages=[dict(player='A',action='FOLD',raw_text='Dealer: A folds')]),validation=dict(action_logic_status='PASS',card_status='PASS',identity_status='PASS'),diagnostics=dict(math={'pass':True}))
    context=dict(player='A',seat_id='seat1',table_id='T01',session_id='s',hand_id='100',occupancy_epoch=1)
    stack=dict(context,segment_id='synthetic:1',status='START_STACK_PROVEN',calculated_start_stack=100,source_observation_id=1,source_state_id=1,binding_receipt=dict(context))
    for name,value in [('poker_reconstruction.jsonl',dict(segment_id='synthetic:1',output_eligible=True,output_blockers=[])),('stack_start_stack.jsonl',stack)]:
        (tmp_path/name).write_text(json.dumps(value)+'\n','utf8')
    proof=dict(actor='A',seat='seat1',rule=RULE);scope=dict(status='UNKNOWN',reason='ACTOR_ACTION_OUTSIDE_IDENTITY_INTERVAL')
    provider=SimpleNamespace(for_hand_identity_uses=lambda hand:[(proof,scope)])
    summary=export_hand_histories(tmp_path,[record],actor_provider=provider)
    saved=json.loads((tmp_path/'hand_histories.jsonl').read_text('utf8'))
    assert 'ACTOR_IDENTITY_INTERVAL_UNKNOWN' in saved['blockers'] and 'START_STACK_CONFLICT' not in saved['blockers']
    assert saved['players'][0]['ACTOR_status']=='UNKNOWN' and saved['players'][0]['START_STACK_status']=='UNKNOWN'
    assert summary['ready_hh_text_count']==0
