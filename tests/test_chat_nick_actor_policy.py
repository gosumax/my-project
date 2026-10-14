"""Practical identity policy and independent current-hand barrier controls."""
import hashlib
import json
import copy
import base64

import pytest
import numpy as np
from types import SimpleNamespace

from parser_core import chat_nick_actor as policy
from parser_core import current_chat_sources as bridge


@pytest.mark.parametrize('raw,actor', [
    ('krismoltisa..','krismoltisanta'), ('stoleyourb..','stoleyourblind'),
    ('romashka..','romashka91808'), ('nick.','nickname'),
    ('nick...','nickname'), ('nick…','nickname'), ('nickname','nickname')])
def test_user_literal_examples(raw, actor):
    assert policy.matches(actor, policy.literal_constraint(raw))


@pytest.mark.parametrize('raw', ['PENSEAT','PENSEAT..','OPENSEAT','OPEN SEAT',
    'OPEN SEAT...','FOLD.','SIT OUT..','ANTE 20','…','..','???'])
def test_observed_ui_is_not_a_nickname(raw):
    assert policy.literal_constraint(raw) is None


def case(monkeypatch, labels=None, actors=('nickname',)):
    if labels is None:
        labels=[(dict(seat='seat2',source_frame=11,source_pts=110),dict(kind='PREFIX',text='nick'))]
    monkeypatch.setattr(policy,'qualified_nicks',lambda c:labels)
    return dict(table_id=1,session_id='s',hand_id='100',**{policy.SOURCE_KEY:dict(
        policy=policy.RULE,source_sha256={},chat_records=[dict(table_id=1,session_id='s',
            hand_id='100',actor=a) for a in actors],conflict_inventory=dict(
                bound_posts=[],state_receipts=[],forced_snapshots=[]))})


def test_one_observation_is_proven_without_roster_other_seats_or_money(monkeypatch):
    c=case(monkeypatch)
    p=policy.prove_direct_actor(c,'nickname')
    assert (p['status'],p['seat'],p['anchor_frame'])==('PROVEN','seat2',11)
    assert 'nickname_sources' not in p
    assert p['start_stack_authority'] is False
    with pytest.raises(ValueError,match='THREE_FRAME'):
        policy.numeric_nickname_sources(c,p)


def test_observed_second_seat_is_a_conflict(monkeypatch):
    c=case(monkeypatch,[(dict(seat=s,source_frame=11,source_pts=110),
        dict(kind='PREFIX',text='nick')) for s in ('seat2','seat3')])
    assert policy.prove_direct_actor(c,'nickname')['reason']=='OBSERVED_SECOND_SEAT_MATCH'


def test_observed_chat_collision_is_a_conflict(monkeypatch):
    c=case(monkeypatch,actors=('nickname','nickother'))
    assert policy.prove_direct_actor(c,'nickname')['reason']=='OBSERVED_CHAT_PREFIX_COLLISION'


def test_full_exact_name_disambiguates_known_prefix_peers(monkeypatch):
    c=case(monkeypatch,[(dict(seat='seat2',source_frame=11,source_pts=110),
        dict(kind='EXACT',text='nickname'))],actors=('nickname','nickother'))
    assert policy.prove_direct_actor(c,'nickname')['status']=='PROVEN'


def test_foreign_chat_context_is_rejected(monkeypatch):
    c=case(monkeypatch);c[policy.SOURCE_KEY]['chat_records'][0]['session_id']='foreign'
    assert policy.prove_direct_actor(c,'nickname')['reason']=='CHAT_ACTOR_CONTEXT_CONFLICT'


@pytest.mark.parametrize('frames,accepted', [([11,12,13],True),([11],False),([11,13,14],False)])
def test_numeric_three_frame_contract_remains_separate(monkeypatch,frames,accepted):
    c=case(monkeypatch,[(dict(seat='seat2',source_frame=f,source_pts=f*10),
        dict(kind='PREFIX',text='nick')) for f in frames])
    p=policy.prove_direct_actor(c,'nickname')
    assert p['status']=='PROVEN'
    if accepted:
        assert [r['source_frame'] for r in policy.numeric_nickname_sources(c,p)]==frames
    else:
        with pytest.raises(ValueError,match='THREE_FRAME'):
            policy.numeric_nickname_sources(c,p)


def test_foreign_hand_label_does_not_hide_observed_target_transition(monkeypatch):
    c=case(monkeypatch)
    c[policy.SOURCE_KEY]['conflict_inventory']['state_receipts']=[dict(
        status='APPLIED',evidence_status='PROVEN',seat='seat2',kind='REPLACE',hand_id='101')]
    assert policy.prove_direct_actor(c,'nickname')['reason']=='OBSERVED_STATE_HAND_CONTEXT_CONFLICT'


@pytest.mark.parametrize('score,expected',[(.95,1),(.949999,0),(.85,0)])
def test_nick_confidence_is_not_lowered(monkeypatch,score,expected):
    monkeypatch.setattr(policy,'frame_times',lambda c:None)
    monkeypatch.setattr(policy,'_seat_rois',lambda:dict(slot_01=dict(seats=dict(seat2=dict(nick=[0,0,2,2])))))
    monkeypatch.setattr(policy,'_verify_gray_source',lambda r:np.zeros((2,2),dtype=np.uint8))
    c=dict(table_id=1,session_id='s',hand_id='100',nickname_observations_all_within_window=[dict(
        table_id=1,session_id='s',hand_id='100',kind='NICK',seat='seat2',source_frame=11,
        roi=[0,0,2,2],ocr_response=dict(status='OK',score=score,raw_text='nick..'))])
    assert len(policy.qualified_nicks(c))==expected


def test_identity_only_export_cannot_certify_unknown_stack(tmp_path):
    from parser_core.hh_export import export_hand_histories
    record=dict(segment_id='s:1',lifecycle=dict(status='COMPLETE'),fully_validated=True,
        observed_hand=dict(table_id='T01',session_id='s',hand_id='100',start_observed=True,
            end_observed=True,boundary_conflicts=[],messages=[dict(player='nickname',action='FOLD',raw_text='Dealer: nickname folds')]),
        validation=dict(action_logic_status='PASS',card_status='PASS',identity_status='PASS'),
        diagnostics=dict(math={'pass':True}))
    recon=dict(segment_id='s:1',output_eligible=True,output_blockers=[])
    (tmp_path/'poker_reconstruction.jsonl').write_text(json.dumps(recon)+'\n','utf8')
    provider=SimpleNamespace(for_hand=lambda h:[dict(actor='nickname',seat='seat2',rule=policy.RULE)])
    summary=export_hand_histories(tmp_path,[record],actor_provider=provider)
    saved=json.loads((tmp_path/'hand_histories.jsonl').read_text('utf8'))
    assert saved['players'][0]['seat']=='seat2'
    assert saved['players'][0]['ACTOR_status']=='PROVEN'
    assert saved['players'][0]['START_STACK'] is None
    assert saved['players'][0]['START_STACK_status']=='UNKNOWN'
    assert saved['evidence_axes']['seats']['status']=='PASS'
    assert saved['evidence_axes']['start_stacks']['status']=='UNKNOWN'
    assert not saved['certified'] and summary['ready_hh_text_count']==0


def test_cross_provider_duplicate_seat_quarantines_both_actors(tmp_path):
    from parser_core.hh_export import export_hand_histories
    record=dict(segment_id='s:1',lifecycle=dict(status='COMPLETE'),fully_validated=True,
        observed_hand=dict(table_id='T01',session_id='s',hand_id='100',start_observed=True,
            end_observed=True,boundary_conflicts=[],messages=[dict(player=a,action='FOLD',raw_text='Dealer: '+a+' folds')
                for a in ('alpha','beta')]),
        validation=dict(action_logic_status='PASS',card_status='PASS',identity_status='PASS'),
        diagnostics=dict(math={'pass':True}))
    recon=dict(segment_id='s:1',output_eligible=True,output_blockers=[])
    context=dict(player='beta',seat_id='seat2',table_id='T01',session_id='s',hand_id='100',occupancy_epoch=1)
    stack=dict(context,segment_id='s:1',status='START_STACK_PROVEN',calculated_start_stack=100,
        source_observation_id=1,source_state_id=1,binding_receipt=dict(context))
    for name,value in [('poker_reconstruction.jsonl',recon),('stack_start_stack.jsonl',stack)]:
        (tmp_path/name).write_text(json.dumps(value)+'\n','utf8')
    provider=SimpleNamespace(for_hand=lambda h:[dict(actor='alpha',seat='seat2',rule=policy.RULE)])
    summary=export_hand_histories(tmp_path,[record],actor_provider=provider)
    saved=json.loads((tmp_path/'hand_histories.jsonl').read_text('utf8'))
    assert 'PLAYER_SEAT_CONFLICT' in saved['blockers']
    assert all(p['ACTOR_status']=='UNKNOWN' and p['seat'] is None for p in saved['players'])
    assert saved['evidence_axes']['seats']['status']=='UNKNOWN'
    assert summary['ready_hh_text_count']==0


def bridge_fixture(tmp_path,monkeypatch,timeline,headers,states):
    def source_rows(packet):
        return [(r,r,dict(status='OK',raw_text=r['raw_ocr_text'],score=.99),None)
                for r in packet['ordinary_prefix_timeline']]
    monkeypatch.setattr(bridge,'observation_source_rows',source_rows)
    monkeypatch.setattr(bridge,'confirmed_header',lambda row,*args:dict(
        hand_id=bridge.HEADER.fullmatch(row['raw_ocr_text'])[1],row=row))
    admissions=[dict(table=1,session_id='s',frame_after=f) for f in sorted({r['frame_id'] for r in timeline})]
    parents=dict(zip(bridge.FILES,[timeline,[],[],admissions,headers,states]))
    pins={}
    for name,rows in parents.items():
        raw=''.join(json.dumps(r)+'\n' for r in rows).encode()
        (tmp_path/name).write_bytes(raw);pins[name]=hashlib.sha256(raw).hexdigest()
    return bridge.load_current_chat(tmp_path,pins)


def row(frame,index,text):
    return dict(table_id='T01',session_id='s',frame_id=frame,row_index=index,
        raw_ocr_text=text,confidence=.99,stable_row_id=f's:F{frame}:R{index}')


def header(frame,number,index=0):
    base=dict(table_id=1,session_id='s',source_frame=frame,source_span_index=index)
    return [dict(base,status='ADMITTED'),dict(base,status='HAND_HEADER_CONFIRMED',
        current_hand_id=number,ocr_response=dict(status='OK',raw_text='Hand #'+number,score=.99))]


def test_sidecar_only_new_header_blocks_previous_hand_assignment(tmp_path,monkeypatch):
    result=bridge_fixture(tmp_path,monkeypatch,
        [row(0,0,'Hand #100'),row(0,1,'Dealer: alpha folds'),row(2,0,'Dealer: beta folds')],
        header(0,'100')+header(1,'101'),[])
    assert [(r['actor'],r['hand_id']) for r in result['records']]==[('alpha','100')]
    assert result['unassigned_actor_rows'][0]['actor']=='beta'


def test_same_frame_native_header_cannot_override_pending_barrier(tmp_path,monkeypatch):
    pending=dict(table_id=1,session_id='s',source_frame=0,source_span_index=9,status='ADMITTED')
    result=bridge_fixture(tmp_path,monkeypatch,
        [row(0,0,'Hand #100'),row(0,1,'Dealer: alpha folds')],header(0,'100')+[pending],[])
    assert result['records']==[]
    assert result['context_checkpoints'][-1]['hand_id'] is None


def test_native_header_reopens_matching_new_context(tmp_path,monkeypatch):
    result=bridge_fixture(tmp_path,monkeypatch,
        [row(0,0,'Hand #100'),row(0,1,'Dealer: alpha folds'),
         row(2,0,'Hand #101'),row(2,1,'Dealer: beta folds')],
        header(0,'100')+header(1,'101')+header(2,'101'),[])
    assert [(r['actor'],r['hand_id']) for r in result['records']]==[('alpha','100'),('beta','101')]


def test_low_confidence_same_number_repeat_cannot_erase_proven_current_context(tmp_path,monkeypatch):
    repeated=header(1,'100');repeated[1]['ocr_response']['score']=.94
    result=bridge_fixture(tmp_path,monkeypatch,
        [row(0,0,'Hand #100'),row(0,1,'Dealer: alpha folds'),row(2,0,'Dealer: beta folds')],
        header(0,'100')+repeated,[])
    assert [(r['actor'],r['hand_id']) for r in result['records']]==[('alpha','100'),('beta','100')]


def observation_packet():
    pixels=np.arange(32,dtype=np.uint8).reshape(4,8)
    digest=hashlib.sha256(pixels.tobytes()).hexdigest()
    r=row(10,3,'Dealer: nickname folds')
    r.update(stable_row_id='s:R000001',video_time=1.0,epoch=5,origin='NEW_CANDIDATE',row_pixel_sha256=digest)
    p=dict(row_id=r['stable_row_id'],row_key='1:10:3',table_id=1,table_session_id='s',
        frame_id=10,source_span_index=3,source_pts=10,time_base_num=1,time_base_den=10,
        epoch=5,origin='NEW_CANDIDATE',row_pixel_sha256=digest,pixel_dtype='uint8',
        pixel_shape=[4,8],row_pixels_base64=base64.b64encode(pixels.tobytes()).decode())
    raw={k:v for k,v in p.items() if k not in ('pixel_dtype','pixel_shape','row_pixels_base64','origin')}
    raw['raw_ocr_response']=dict(item_id='1:10:3',status='OK',raw_text=r['raw_ocr_text'],score=.99)
    return dict(table_id=1,session_id='s',header_frame=10,ordinary_prefix_timeline=[r],
        ordinary_prefix_physical_rows=[p],ordinary_prefix_raw_ocr=[raw])


def test_repeated_native_observation_needs_no_new_event_admission():
    packet=observation_packet();before=copy.deepcopy(packet)
    joined=bridge.observation_source_rows(packet)
    assert joined[0][0]['stable_row_id']=='s:R000001'
    assert bridge.chat_actor_candidates(joined)[0]['actor']=='nickname'
    assert packet==before and 'ordinary_prefix_admission' not in packet


@pytest.mark.parametrize('mutation',['clock','hash','raw_text','confidence','callback','seat_context','raw_frame'])
def test_repeat_observation_keeps_exact_provenance_guards(mutation):
    packet=observation_packet();r=packet['ordinary_prefix_timeline'][0]
    raw=packet['ordinary_prefix_raw_ocr'][0];response=raw['raw_ocr_response']
    if mutation=='clock':r['video_time']=2.0
    elif mutation=='hash':r['row_pixel_sha256']='0'*64
    elif mutation=='raw_text':response['raw_text']='Dealer: foreign folds'
    elif mutation=='confidence':response['score']=.96
    elif mutation=='callback':response['item_id']='1:11:3'
    elif mutation=='seat_context':raw['table_id']=2
    elif mutation=='raw_frame':raw['frame_id']=11
    with pytest.raises(ValueError):bridge.observation_source_rows(packet)


def test_same_epoch_repeat_can_supply_identity_without_modifying_actions(tmp_path,monkeypatch):
    first=row(0,1,'Dealer: alpha folds');first['stable_row_id']='s:R000001'
    repeat=dict(first,frame_id=1,row_index=8)
    timeline=[row(0,0,'Hand #100'),first,repeat];before=copy.deepcopy(timeline)
    result=bridge_fixture(tmp_path,monkeypatch,timeline,header(0,'100'),[])
    assert len(result['records'])==2 and {r['hand_id'] for r in result['records']}=={'100'}
    assert all(r['poker_action_authority'] is False and r['new_event_source_required'] is False
               for r in result['records'])
    assert timeline==before


def test_repeat_cannot_move_from_proven_previous_hand_to_current(tmp_path,monkeypatch):
    first=row(0,1,'Dealer: alpha folds');first['stable_row_id']='s:R000001'
    repeat=dict(first,frame_id=2,row_index=18)
    result=bridge_fixture(tmp_path,monkeypatch,
        [row(0,0,'Hand #100'),first,row(1,0,'Hand #101'),repeat],
        header(0,'100')+header(1,'101'),[])
    assert [(r['actor'],r['hand_id']) for r in result['records']]==[('alpha','100')]
    assert result['unassigned_actor_rows'][-1]['reason']=='OBSERVED_FOREIGN_HAND_ROW_OWNER'


def test_reobserved_closed_header_cannot_reopen_a_prior_hand(tmp_path,monkeypatch):
    old_header=row(0,0,'Hand #100');old_header['stable_row_id']='s:R000001'
    repeat=dict(old_header,frame_id=2,row_index=0)
    result=bridge_fixture(tmp_path,monkeypatch,
        [old_header,row(1,0,'Hand #101'),repeat,row(2,1,'Dealer: beta folds')],
        header(0,'100')+header(1,'101')+header(2,'100'),[])
    assert result['context_checkpoints'][-1]['hand_id']=='101'
    assert result['records']==[]
    assert result['unassigned_actor_rows'][-1]['reason']=='HISTORICAL_CHAT_REGION'


def test_event_duplicate_annotation_does_not_erase_identity_source():
    r=row(10,3,'Dealer: nickname folds');h=row(0,0,'Hand #100')
    proof=dict(rule=policy.RULE,table_id=1,session_id='s',hand_id='100',chat_actor_sources=[dict(
        row_id=r['stable_row_id'],header_row_id=h['stable_row_id'],source_row=r,header_source=dict(row=h))])
    current=dict(r,canonical_duplicate_of='earlier-event-source')
    hand=dict(table_id='T01',session_id='s',hand_id='100',messages=[dict(source_rows=[h,current])])
    assert policy.matches_hand(proof,hand)
    current['raw_ocr_text']='Dealer: another folds'
    assert not policy.matches_hand(proof,hand)


@pytest.mark.parametrize('repeated_source',['actor','header'])
def test_later_weak_repeat_does_not_hide_qualified_occurrence(repeated_source):
    r=row(10,3,'Dealer: nickname folds');h=row(0,0,'Hand #100')
    proof=dict(rule=policy.RULE,table_id=1,session_id='s',hand_id='100',chat_actor_sources=[dict(
        row_id=r['stable_row_id'],header_row_id=h['stable_row_id'],source_row=r,header_source=dict(row=h))])
    repeated=dict(r if repeated_source=='actor' else h,frame_id=20,row_index=8,confidence=.94)
    hand=dict(table_id='T01',session_id='s',hand_id='100',messages=[dict(source_rows=[h,r,repeated])])
    assert policy.matches_hand(proof,hand)
    # Contradictory copies of this exact occurrence remain a factual conflict.
    hand['messages'][0]['source_rows'].append(dict(r,raw_ocr_text='Dealer: foreign folds'))
    assert not policy.matches_hand(proof,hand)
