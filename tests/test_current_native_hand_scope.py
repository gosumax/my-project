"""Synthetic transport around a retained native NICK; no video/OCR/solver."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from parser_core.simple_saved_policy import LineageActorProvider,hand_key,confirmed_hand_associations
from parser_core.simple_seat_who import SeatIdentityLineage

INPUT=json.loads((Path(__file__).parent/'fixtures/native_who_guard_inputs.json').read_text('utf8'))

def fixture(tmp_path, *, extra=(), state=()):
    point=copy.deepcopy(INPUT['new_native']);raw=point['nickname_anchor_source'];actor=point['actor']
    key=hand_key(point);frame=raw['source_frame'];low,high=frame-100,frame+100
    path=tmp_path/'seat_forced_inputs.jsonl';path.write_text(json.dumps(raw)+'\n','utf8')
    response=raw['ocr_response']
    nick=dict(table_id=key[0],session_id=key[1],hand_id=key[2],seat=point['seat'],frame=frame,
        raw_text=response['raw_text'],confidence=response['score'],ocr_status=response['status'],
        roi=raw['roi'],pixel_sha256=raw['pixel_sha256'],source_pts=raw['source_pts'],time_base=raw['time_base'],
        source_file=str(path),source_line=1,item_id=response['item_id'])
    weak=[dict(nick,frame=low+1,raw_text='NUIN',confidence=.57),
          dict(nick,frame=low+2,raw_text='NUIN',confidence=.58)]
    observations=weak+[nick]+[dict(nick,**r) for r in extra]
    hand=dict(table_id=f'T{key[0]:02d}',session_id=key[1],hand_id=key[2],messages=[
        dict(action='ANTE',player=actor,source_rows=[dict(frame_id=low,confidence=.99,raw_ocr_text=f'Dealer: {actor} posts Ante 137')]),
        dict(action='NO_SHOW',player=actor,source_rows=[dict(frame_id=high,confidence=.99,raw_ocr_text=f"Dealer: {actor} doesn't show cards")])])
    states=[dict(table=key[0],session=key[1],seat=point['seat'],evidence_status='PROVEN',**r) for r in state]
    lineage=SeatIdentityLineage(observations,states)
    provider=LineageActorProvider(lineage,{key:(low,high)},[hand],native_source=tmp_path)
    return provider,hand,actor,nick,path

def test_late_native_name_covers_ante_and_no_show_without_ocr_interval_rewrite(tmp_path):
    provider,hand,actor,nick,path=fixture(tmp_path)
    before=copy.deepcopy(provider.lineage.segments)
    proof=provider.for_actor(hand,actor)
    assert proof['status']=='PROVEN' and proof['rule_used']=='CURRENT_HAND_NATIVE_NICK_CHAT'
    assert proof['identity_interval']==list(provider.bounds[hand_key(hand)])
    assert provider.lineage.segments==before
    assert proof['anchor']['pixel_sha256']==nick['pixel_sha256']

@pytest.mark.parametrize('kind',['EMPTY','REPLACE','LEFT','OPEN SEAT'])
def test_physical_identity_clear_blocks_hand_scope(tmp_path,kind):
    frame=INPUT['new_native']['nickname_anchor_source']['source_frame']
    provider,hand,actor,_,_=fixture(tmp_path,state=[dict(kind=kind,frame=frame-1)])
    assert provider._hand_scope(hand,actor) is None

def test_participant_name_is_locked_despite_sustained_qualified_ocr(tmp_path):
    frame=INPUT['new_native']['nickname_anchor_source']['source_frame']
    provider,hand,actor,_,_=fixture(tmp_path,extra=[
        dict(frame=frame+20,raw_text='different_player',confidence=.99),
        dict(frame=frame+21,raw_text='different_player',confidence=.99)])
    proof=provider._hand_scope(hand,actor)
    assert proof['status']=='PROVEN'
    assert proof['current_hand_identity']['participant_nickname_locked'] is True
    assert [r['raw_text'] for r in proof['current_hand_identity']['uncertain_observations']][-2:]==['different_player']*2

def test_single_anomaly_does_not_become_replacement(tmp_path):
    frame=INPUT['new_native']['nickname_anchor_source']['source_frame']
    provider,hand,actor,_,_=fixture(tmp_path,extra=[dict(frame=frame+20,raw_text='sw',confidence=.951)])
    assert provider._hand_scope(hand,actor)['status']=='PROVEN'

def test_competing_physical_seat_blocks_hand_scope(tmp_path):
    point=INPUT['new_native'];frame=point['nickname_anchor_source']['source_frame']
    provider,hand,actor,_,_=fixture(tmp_path,extra=[dict(frame=frame+20,seat='seat9',
        raw_text=point['nickname_anchor_source']['ocr_response']['raw_text'],confidence=.9999)])
    assert provider._hand_scope(hand,actor) is None

def test_changed_native_anchor_is_not_accepted(tmp_path):
    provider,hand,actor,_,path=fixture(tmp_path)
    raw=json.loads(path.read_text('utf8'));raw['pixel_sha256']='0'*64
    path.write_text(json.dumps(raw)+'\n','utf8')
    assert provider._hand_scope(hand,actor) is None


def test_empty_native_png_withholds_instead_of_crashing(tmp_path):
    provider,hand,actor,_,path=fixture(tmp_path)
    raw=json.loads(path.read_text('utf8'));raw['png_base64']=''
    path.write_text(json.dumps(raw)+'\n','utf8')
    assert provider._hand_scope(hand,actor) is None

def test_later_registered_ocr_spelling_does_not_rename_participant(tmp_path):
    frame=INPUT['new_native']['nickname_anchor_source']['source_frame']
    provider,hand,actor,_,_=fixture(tmp_path,extra=[dict(frame=frame+20,raw_text='registered_player',confidence=.7)])
    provider.player_registry=SimpleNamespace(aliases={},resolve=lambda raw,kind:
        dict(canonical='registered_player') if raw=='registered_player' else None)
    assert provider._hand_scope(hand,actor)['actor']==actor


def test_post_win_tail_cannot_supply_best_anchor_or_rename_winner(tmp_path):
    frame=INPUT['new_native']['nickname_anchor_source']['source_frame']
    raw=INPUT['new_native']['nickname_anchor_source']['ocr_response']['raw_text']
    provider,hand,actor,nick,_=fixture(tmp_path,extra=[
        dict(frame=frame+20,raw_text=raw,confidence=1.0),
        dict(frame=frame+30,raw_text='WINNER',confidence=.99),
        dict(frame=frame+31,raw_text='WINNER',confidence=.99)])
    hand['messages'][-1]['source_rows'][0]['frame_id']=frame+10
    proof=provider.for_actor(hand,actor)
    assert proof['status']=='PROVEN'
    assert proof['anchor']['frame']==nick['frame']
    assert proof['identity_interval'][1]==frame+10
    assert proof['rule_used']=='CURRENT_HAND_NATIVE_NICK_CHAT'


def test_competing_current_chat_player_on_same_seat_is_not_merged(tmp_path):
    frame=INPUT['new_native']['nickname_anchor_source']['source_frame']
    provider,hand,actor,_,_=fixture(tmp_path,extra=[
        dict(frame=frame+20,raw_text='other_chat_player',confidence=.99)])
    hand['messages'].append(dict(action='FOLD',player='other_chat_player',
        source_rows=[dict(frame_id=frame+21)]))
    assert provider._hand_scope(hand,actor) is None


def test_hand_lock_does_not_grant_identity_to_next_hand(tmp_path):
    provider,hand,actor,_,_=fixture(tmp_path)
    following=copy.deepcopy(hand);following['hand_id']='different_hand'
    provider.bounds[hand_key(following)]=provider.bounds[hand_key(hand)]
    assert provider._hand_scope(following,actor) is None


def test_first_qualified_native_frame_wins_over_later_higher_confidence(tmp_path):
    raw=INPUT['new_native']['nickname_anchor_source'];frame=raw['source_frame']
    provider,hand,actor,nick,_=fixture(tmp_path,extra=[dict(frame=frame+10,
        raw_text=raw['ocr_response']['raw_text'],confidence=1.0)])
    proof=provider.for_actor(hand,actor)
    assert proof['status']=='PROVEN' and proof['anchor']['frame']==frame
    assert proof['scope_nick_observations']==[nick]


def test_low_confidence_lineage_does_not_bypass_current_native_095_policy(tmp_path):
    provider,hand,actor,nick,path=fixture(tmp_path)
    nick['confidence']=.94
    for row in provider.lineage.observations:
        if row['frame']==nick['frame']:row['confidence']=.94
    raw=json.loads(path.read_text('utf8'));raw['ocr_response']['score']=.94
    path.write_text(json.dumps(raw)+'\n','utf8')
    proof=provider.for_actor(hand,actor)
    assert proof['status']=='UNKNOWN'
    assert proof['reason']=='CURRENT_HAND_FIRST_QUALIFIED_NICK_UNKNOWN'

def test_historical_association_requires_video_and_exact_bounds(tmp_path,monkeypatch):
    import parser_core.simple_saved_policy as policy
    config=tmp_path/'configs';config.mkdir();source=tmp_path/'source';source.mkdir()
    record=dict(table_id=1,session_id='s',hand_id='h',seat='seat1',aliases=['A','B'],
        frame_start=10,frame_end=20,authority='USER_EXPLICIT_SAME_OCCUPANT')
    original=tmp_path/'history.jsonl';original.write_text(json.dumps(record)+'\n','utf8')
    import hashlib
    spec=dict(schema='poker.confirmed-hand-associations.v1',source_file=str(original),
        source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),source_video_sha256='1'*64,records=[record])
    (config/'confirmed_nickname_associations.json').write_text(json.dumps(spec),'utf8')
    monkeypatch.setattr(policy,'ROOT',tmp_path)
    metadata=source/'capture_metadata.json';metadata.write_text(json.dumps(dict(source_video_sha256='2'*64)),'utf8')
    bounds={(1,'s','h'):(10,20)}
    assert confirmed_hand_associations(source,bounds)==[]
    saved=original.read_bytes();original.unlink()
    assert confirmed_hand_associations(source,bounds)==[]
    original.write_bytes(saved)
    metadata.write_text(json.dumps(dict(source_video_sha256='1'*64)),'utf8')
    assert len(confirmed_hand_associations(source,bounds))==1
    assert confirmed_hand_associations(source,{(1,'s','h'):(10,21)})==[]
    original.write_text(json.dumps(dict(record,aliases=['A','FOREIGN']))+'\n','utf8')
    with pytest.raises(ValueError,match='HISTORY_CHANGED'):confirmed_hand_associations(source,bounds)
