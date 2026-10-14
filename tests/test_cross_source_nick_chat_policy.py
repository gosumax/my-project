import copy
import pytest
import numpy as np
from parser_core import chat_nick_actor as policy


def source_row(frame,index,text,score=.99):
    return dict(table_id='T01',session_id='s',frame_id=frame,row_index=index,
        stable_row_id=f's:R{frame:06d}_{index}',raw_ocr_text=text,confidence=score,
        row_pixel_sha256=f'pixels:{frame}:{index}',epoch=0,video_time=frame/10)


def chat(actor,score=.99,index=1):
    h=source_row(0,0,'Hand #100')
    r=source_row(12,index,f'Dealer: {actor} folds',score)
    return dict(table_id=1,session_id='s',hand_id='100',actor=actor,confidence=score,
        source_row=r,header_source=dict(row=h),row_id=r['stable_row_id'],header_row_id=h['stable_row_id'])


def nick(raw,score,seat='seat2',frame=11):
    return dict(table_id=1,session_id='s',hand_id='100',kind='NICK',seat=seat,
        source_frame=frame,source_pts=frame*10,roi=[0,0,2,2],
        ocr_response=dict(status='OK',score=score,raw_text=raw))


def context(monkeypatch,raw='bop_',score=.912,actor='bop_',chat_score=.99):
    monkeypatch.setattr(policy,'frame_times',lambda c:None)
    monkeypatch.setattr(policy,'_seat_rois',lambda:dict(slot_01=dict(seats={
        seat:dict(nick=[0,0,2,2]) for seat in ('seat2','seat3')})))
    monkeypatch.setattr(policy,'_verify_gray_source',lambda r:np.zeros((2,2),dtype=np.uint8))
    return dict(table_id=1,session_id='s',hand_id='100',
        nickname_observations_all_within_window=[nick(raw,score)],
        **{policy.SOURCE_KEY:dict(policy=policy.RULE,cross_source_policy=policy.CROSS_RULE,
            source_sha256={},chat_records=[chat(actor,chat_score)],
            context_checkpoints=[dict(frame=0,hand_id='100')],
            conflict_inventory=dict(bound_posts=[],state_receipts=[],forced_snapshots=[]))})


@pytest.mark.parametrize('raw,score,actor',[
    ('bop_',.912,'bop_'),('gosteeer',.946,'gosteeer'),('vasyajin',.949,'vasyajin'),
    ('bashersha...',.938,'bashershadow'),('retre1',.932,'retre1'),
    ('nick.',.8,'nickname'),('nick..',.8,'nickname'),('nick…',.8,'nickname')])
def test_weak_literal_nick_and_strong_current_chat_prove_who(monkeypatch,raw,score,actor):
    c=context(monkeypatch,raw,score,actor);before=copy.deepcopy(c)
    p=policy.prove_direct_actor(c,actor)
    assert (p['status'],p['rule'],p['seat'])==('PROVEN',policy.CROSS_RULE,'seat2')
    assert p['cross_source_proof']['nick_confidence']==score
    assert p['nickname_anchor_source']['ocr_response']['raw_text']==raw
    assert c==before
    assert not p['poker_action_authority'] and not p['event_uniqueness_authority']
    assert not p['start_stack_authority'] and not p['historical_occupancy_authority']
    assert not p['new_event_source_required']


@pytest.mark.parametrize('score',[0.0,.1,.949999])
def test_named_cross_source_rule_has_no_new_nick_score_floor(monkeypatch,score):
    c=context(monkeypatch,score=score)
    assert policy.prove_direct_actor(c,'bop_')['status']=='PROVEN'
    assert policy.qualified_nicks(c)==[]


@pytest.mark.parametrize('score',[.949999,.85,float('nan'),True])
def test_weak_chat_cannot_corroborate_weak_nick(monkeypatch,score):
    c=context(monkeypatch,chat_score=score)
    assert policy.prove_direct_actor(c,'bop_')['status']=='UNKNOWN'


def test_weak_nick_without_chat_or_with_roster_only_stays_unknown(monkeypatch):
    c=context(monkeypatch);s=c[policy.SOURCE_KEY]
    s['chat_records']=[];s['conflict_inventory']['forced_snapshots']=[dict(actors=['bop_'])]
    assert policy.prove_direct_actor(c,'bop_')['status']=='UNKNOWN'


@pytest.mark.parametrize('raw',['wrong','retrel','bop_x','bop','PENSEAT','PENSEAT..',
    'EMPTY','EMPTY..','OPEN SEAT','OpenSeat','FOLD','SIT OUT'])
def test_no_literal_match_and_ui_remain_unknown(monkeypatch,raw):
    c=context(monkeypatch,raw=raw)
    assert policy.prove_direct_actor(c,'bop_')['status']=='UNKNOWN'


def test_weak_full_name_is_not_an_implicit_prefix(monkeypatch):
    c=context(monkeypatch,raw='bop',actor='bop_')
    assert policy.prove_direct_actor(c,'bop_')['status']=='UNKNOWN'


@pytest.mark.parametrize('mutation',['table','session','hand','roi','kind','context_checkpoint'])
def test_cross_source_keeps_fixed_roi_and_current_context_guards(monkeypatch,mutation):
    c=context(monkeypatch);r=c['nickname_observations_all_within_window'][0]
    if mutation=='table':r['table_id']=2
    elif mutation=='session':r['session_id']='foreign'
    elif mutation=='hand':r['hand_id']='101'
    elif mutation=='roi':r['roi']=[1,0,3,2]
    elif mutation=='kind':r['kind']='STACK'
    else:c[policy.SOURCE_KEY]['context_checkpoints'][0]['hand_id']='101'
    assert policy.prove_direct_actor(c,'bop_')['status']=='UNKNOWN'


def test_two_observed_actors_for_one_seat_are_not_guessed(monkeypatch):
    c=context(monkeypatch)
    c['nickname_observations_all_within_window'].append(nick('another',.9,frame=12))
    c[policy.SOURCE_KEY]['chat_records'].append(chat('another',index=2))
    assert all(policy.prove_direct_actor(c,a)['status']=='UNKNOWN' for a in ('bop_','another'))


def test_same_actor_matching_two_observed_seats_is_a_conflict(monkeypatch):
    c=context(monkeypatch)
    c['nickname_observations_all_within_window'].append(nick('bop_',.9,seat='seat3'))
    assert policy.prove_direct_actor(c,'bop_')['reason']=='OBSERVED_SECOND_SEAT_MATCH'


def test_unmatched_weak_text_does_not_veto_an_existing_strong_binding(monkeypatch):
    c=context(monkeypatch,score=.99)
    before=policy.prove_direct_actor(c,'bop_')
    c['nickname_observations_all_within_window'].append(nick('noise',.2,frame=12))
    assert policy.prove_direct_actor(c,'bop_')==before


def test_existing_strong_anchor_is_preferred_and_keeps_direct_rule(monkeypatch):
    c=context(monkeypatch)
    c['nickname_observations_all_within_window'].append(nick('bop_',.99,frame=13))
    p=policy.prove_direct_actor(c,'bop_')
    assert p['rule']==policy.RULE and p['anchor_frame']==13
    assert 'cross_source_proof' not in p


def test_global_and_numeric_three_frame_thresholds_remain_095(monkeypatch):
    c=context(monkeypatch)
    c['nickname_observations_all_within_window']=[nick('bop_',.912,frame=f) for f in (11,12,13)]
    p=policy.prove_direct_actor(c,'bop_')
    assert p['status']=='PROVEN' and p['rule']==policy.CROSS_RULE
    assert policy.qualified_nicks(c)==[]
    with pytest.raises(ValueError,match='NUMERIC_NICKNAME_THREE_FRAME_WITNESS_UNKNOWN'):
        policy.numeric_nickname_sources(c,p)


def test_current_hand_join_is_required_for_cross_source_rule(monkeypatch):
    c=context(monkeypatch);p=policy.prove_direct_actor(c,'bop_');r=p['chat_actor_sources'][0]
    hand=dict(table_id='T01',session_id='s',hand_id='100',messages=[dict(source_rows=[
        r['header_source']['row'],dict(r['source_row'],canonical_duplicate_of='older-action')])])
    assert policy.point_matches_hand(p,hand)
    hand['hand_id']='101'
    assert not policy.point_matches_hand(p,hand)
    assert policy.outer_rule('NUMERIC',p)=='NUMERIC__DIRECT_CHAT_NICK_V1'


def test_legacy_context_cannot_silently_inherit_cross_source_rule(monkeypatch):
    c=context(monkeypatch);c[policy.SOURCE_KEY].pop('cross_source_policy')
    assert policy.prove_direct_actor(c,'bop_')['status']=='UNKNOWN'
