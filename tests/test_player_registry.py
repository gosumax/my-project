import base64
import copy
import hashlib
import json
import threading
from http.client import HTTPConnection

import cv2
import numpy as np
import pytest

from parser_core.player_registry import PlayerRegistry, digest
from parser_core.nickname_review import _image, _near, build_cases
from parser_core.simple_seat_who import SeatIdentityLineage
from parser_core.simple_saved_policy import LineageActorProvider
from scripts.review_nicknames import ReviewServer, verify_case


def case(nick='retre1', actor='retrel'):
    result = dict(context=dict(table_id=1, session_id='s', hand_id='h', seat='seat1'),
        identity_interval=[0, None], source_pins={}, observations=[
            dict(id='nick', kind='NICK', text=nick, source={'pixel_sha256':'a'*64}),
            dict(id='chat', kind='CHAT', text=actor, source={'pixel_sha256':'b'*64})])
    result['case_id'] = digest(result)
    return result


def registry(tmp_path):
    path = tmp_path/'players.sqlite'
    PlayerRegistry(path).confirm(case(), 'retrel', ['nick', 'chat'])
    return PlayerRegistry(path)


def nick(frame=10, raw='retre1', seat='seat1'):
    return dict(table_id=1,session_id='s',seat=seat,frame=frame,raw_text=raw,confidence=.99)


def message(actor='retrel', frame=15, action='FOLD'):
    return dict(player=actor, action=action, source_rows=[dict(table_id='T01',session_id='s',
        frame_id=frame,stable_row_id=f's:{frame}',row_pixel_sha256='b'*64,raw_ocr_text=f'Dealer: {actor} folds',confidence=.98)])


def evidence(db, observations=None, messages=None, states=()):
    lineage=SeatIdentityLineage(observations or [nick(),nick(20)],states)
    hand=dict(table_id='T01',session_id='s',hand_id='h',messages=messages or [message()])
    bounds={(1,'s','h'):(10,30)}
    return lineage,hand,bounds,db.hand_evidence(lineage,[hand],bounds)


def test_missing_database_read_does_not_create_files(tmp_path):
    path=tmp_path/'missing'/'players.sqlite'
    db=PlayerRegistry(path)
    assert not path.exists() and db.resolve('retrel','CHAT') is None


def test_human_confirmation_is_typed_and_retains_raw_provenance(tmp_path):
    path=tmp_path/'players.sqlite';original=case();before=copy.deepcopy(original)
    PlayerRegistry(path).confirm(original,'retrel',['nick'])
    db=PlayerRegistry(path)
    assert db.resolve('retre1','NICK')['canonical']=='retrel'
    assert db.resolve('retre1','CHAT') is None
    assert db.resolve('retrel','CHAT')['canonical']=='retrel'
    assert db.resolve('retreI','NICK') is None
    assert original==before
    assert next(iter(db.reviews.values()))['observations'][0]['source']['pixel_sha256']=='a'*64


def test_conflicting_alias_is_not_overwritten_or_merged(tmp_path):
    db=registry(tmp_path)
    second=case();second.pop('case_id');second['context']['hand_id']='another';second['case_id']=digest(second)
    result=db.confirm(second,'differentPlayer',['nick'])
    fresh=PlayerRegistry(db.path)
    assert fresh.resolve('retre1','NICK') is None
    assert len(fresh.players)==2 and result['conflicts']


def test_same_case_can_be_corrected_without_erasing_review_history(tmp_path):
    db=registry(tmp_path)
    result=db.confirm(case(),'correctedPlayer',['nick','chat'])
    fresh=PlayerRegistry(db.path)
    assert fresh.resolve('retre1','NICK')['canonical']=='correctedPlayer'
    assert fresh.resolve('retrel','CHAT')['canonical']=='correctedPlayer'
    assert len(fresh.reviews)==2 and not result['conflicts']
    assert next(r for r in fresh.reviews.values() if r['canonical']=='correctedPlayer')['supersedes']


def test_deselected_chat_alias_loses_authority_for_same_case(tmp_path):
    db=registry(tmp_path)
    db.confirm(case(),'retrel',['nick'])
    fresh=PlayerRegistry(db.path)
    assert fresh.resolve('retre1','NICK')['canonical']=='retrel'
    # Canonical manual text remains valid; an independently selected typo does not.
    alternate=case(actor='retreI')
    fresh.confirm(alternate,'retrel',['nick','chat'])
    db=PlayerRegistry(db.path);db.confirm(alternate,'retrel',['nick'])
    db=PlayerRegistry(db.path)
    assert db.resolve('retreI','CHAT') is None and db.resolve('retre1','NICK')


def test_registry_reuses_only_exact_confirmed_spellings_through_current_seat(tmp_path):
    db=registry(tmp_path)
    lineage,hand,bounds,(associations,manual,rejected)=evidence(db)
    assert len(associations)==len(manual)==1 and not rejected
    provider=LineageActorProvider(lineage,bounds,[hand],associations,manual)
    result=provider.for_actor(hand,'retrel')
    assert result['status']=='PROVEN' and result['seat']=='seat1'
    assert result['rule_used']=='CONFIRMED_PLAYER_REGISTRY_SAME_OCCUPANT'
    assert result['canonical_actor']=='retrel'
    canonical=provider.canonical_hand(hand)
    assert canonical['messages'][0]['source_rows']==hand['messages'][0]['source_rows']
    assert canonical['messages'][0]['original_player']=='retrel'
    assert hand['messages'][0]['player']=='retrel'


@pytest.mark.parametrize('observations,messages,states',[
    ([nick(),nick(11,seat='seat2')],None,()),
    ([nick(),nick(18,'other'),nick(19,'other')],None,()),
    ([nick(),nick(18)],None,[dict(table_id=1,session_id='s',seat='seat1',frame=17,kind='EMPTY')]),
    ([nick(),nick(12,'unreviewed')],None,()),
    (None,[message(frame=9)],()),
    (None,[message(),dict(player='retrel',action='ANTE',source_rows=[])],()),
    (None,[dict(message(),source_rows=[dict(message()['source_rows'][0],table_id='T02')])],()),
])
def test_registry_rejects_seat_replacement_temporal_unknown_and_source_conflicts(tmp_path,observations,messages,states):
    _,_,_,(associations,manual,rejected)=evidence(registry(tmp_path),observations,messages,states)
    assert not associations and not manual and rejected


def test_two_actor_spellings_require_current_review_even_for_call_raise(tmp_path):
    db=registry(tmp_path)
    db.confirm(case(actor='retreI'),'retrel',['nick','chat'])
    db=PlayerRegistry(db.path)
    _,_,_,(associations,manual,rejected)=evidence(db,messages=[message(action='CALL'),message('retreI',16,'RAISE')])
    assert not associations and not manual
    assert rejected[0]['reason']=='REGISTRY_MULTIPLE_CHAT_SPELLINGS_REQUIRE_CURRENT_REVIEW'


def test_raw_gray_chat_and_native_png_previews_preserve_pixel_hash():
    pixels=np.arange(128,dtype=np.uint8).reshape(8,16)
    sha=hashlib.sha256(pixels.tobytes()).hexdigest()
    raw=base64.b64encode(pixels.tobytes()).decode()
    url=_image(raw,sha,list(pixels.shape))
    image=cv2.imdecode(np.frombuffer(base64.b64decode(url.split(',')[1]),np.uint8),cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(image,pixels)
    _,png=cv2.imencode('.png',pixels)
    assert _image(base64.b64encode(png).decode(),sha).startswith('data:image/png;base64,')
    with pytest.raises(ValueError):_image(raw,'0'*64,list(pixels.shape))


def test_similarity_is_only_a_suggestion(tmp_path):
    assert _near('retrel','retre1')
    assert registry(tmp_path).resolve('retreI','NICK') is None


def test_ignored_native_anomaly_is_visible_unchecked_and_can_be_reviewed(tmp_path):
    from types import SimpleNamespace
    from parser_core.nickname_roster_binding import _seat_rois
    db=registry(tmp_path)
    source=tmp_path/'source';source.mkdir()
    roi=_seat_rois()['slot_01']['seats']['seat1']['nick']
    pixels=np.arange((roi[3]-roi[1])*(roi[2]-roi[0]),dtype=np.uint8).reshape(roi[3]-roi[1],roi[2]-roi[0])
    sha=hashlib.sha256(pixels.tobytes()).hexdigest();_,png=cv2.imencode('.png',pixels)
    native=[];observations=[]
    for line,(frame,raw) in enumerate([(10,'retre1'),(12,'retreI'),(20,'retre1')],1):
        native.append(dict(table_id=1,session_id='s',hand_id='h',seat='seat1',kind='NICK',source_frame=frame,
            source_pts=frame,time_base=[1,30],roi=roi,pixel_sha256=sha,source_ref=sha,png_base64=base64.b64encode(png).decode(),
            ocr_response=dict(status='OK',raw_text=raw,score=.99)))
        observations.append(dict(nick(frame,raw),source_file=str(source/'seat_forced_inputs.jsonl'),source_line=line,
            pixel_sha256=sha,roi=roi,source_pts=frame,time_base=[1,30]))
    (source/'seat_forced_inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in native))
    chat_pixels=np.arange(64,dtype=np.uint8).reshape(4,16);chat_sha=hashlib.sha256(chat_pixels.tobytes()).hexdigest()
    physical=dict(row_id='s:15',frame_id=15,table_id=1,table_session_id='s',pixel_dtype='uint8',
        pixel_shape=[4,16],row_pixel_sha256=chat_sha,row_pixels_base64=base64.b64encode(chat_pixels.tobytes()).decode())
    (source/'physical_rows.jsonl').write_text(json.dumps(physical)+'\n')
    msg=message();msg['source_rows'][0]['row_pixel_sha256']=chat_sha
    lineage,hand,bounds,(associations,manual,rejected)=evidence(db,observations,[msg])
    assert rejected[0]['reason']=='REGISTRY_UNREVIEWED_NICK_ANOMALY'
    actors=LineageActorProvider(lineage,bounds,[hand],associations,manual)
    provider=SimpleNamespace(hands={(1,'s','h'):hand},bounds=bounds,lineage=lineage,actor_sources=actors,player_registry=db)
    cases=build_cases(provider,source)
    assert len(cases)==1 and 'error' not in cases[0]
    choices={r['text']:r for r in cases[0]['observations'] if r['kind']=='NICK'}
    assert set(choices)=={'retre1','retreI'}
    assert choices['retre1']['selected'] and not choices['retreI']['selected']
    assert all(r['image'].startswith('data:image/png;base64,') for r in cases[0]['observations'])
    db.confirm(cases[0],'retrel',[r['id'] for r in cases[0]['observations']])
    _,_,_,(associations,manual,rejected)=evidence(PlayerRegistry(db.path),observations,[msg])
    assert len(associations)==1 and not rejected


@pytest.mark.parametrize('confidence,anomaly,expected_cases',[(.99,False,0),(.87,False,1),(.99,True,0)])
def test_review_does_not_join_other_interval_ui_or_amount_only_chat(tmp_path,confidence,anomaly,expected_cases):
    from types import SimpleNamespace
    from parser_core.nickname_roster_binding import _seat_rois
    db=PlayerRegistry(tmp_path/'players.sqlite');source=tmp_path/'source';source.mkdir()
    roi=_seat_rois()['slot_01']['seats']['seat1']['nick']
    pixels=np.zeros((roi[3]-roi[1],roi[2]-roi[0]),dtype=np.uint8)
    sha=hashlib.sha256(pixels.tobytes()).hexdigest();_,png=cv2.imencode('.png',pixels)
    native=[];observations=[]
    samples=[(10,'sP'),(11,'SR'),(12,'SR'),(20,'gretchin'),(21,'gretchin')]
    if anomaly:samples.append((22,'unrelatedNoise'))
    for line,(frame,raw) in enumerate(samples,1):
        score=.60 if raw=='unrelatedNoise' else confidence
        native.append(dict(table_id=1,session_id='s',hand_id='h',seat='seat1',kind='NICK',source_frame=frame,
            source_pts=frame,time_base=[1,30],roi=roi,pixel_sha256=sha,source_ref=sha,png_base64=base64.b64encode(png).decode(),
            ocr_response=dict(status='OK',raw_text=raw,score=score)))
        observations.append(dict(nick(frame,raw),confidence=score,source_file=str(source/'seat_forced_inputs.jsonl'),source_line=line,
            pixel_sha256=sha,roi=roi,source_pts=frame,time_base=[1,30]))
    (source/'seat_forced_inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in native))
    physical=[]
    ante=message('gretchin',15,'SMALL_BLIND');fold=message('gretchin',25)
    for m in [ante,fold]:
        actor=m['source_rows'][0];actor['confidence']=.90;actor['row_pixel_sha256']=sha
        amount=dict(actor,raw_ocr_text='3000',stable_row_id=actor['stable_row_id']+':amount',confidence=.999)
        m['source_rows'].append(amount)
        for row in m['source_rows']:
            physical.append(dict(row_id=row['stable_row_id'],frame_id=row['frame_id'],table_id=1,table_session_id='s',
                pixel_dtype='uint8',pixel_shape=list(pixels.shape),row_pixel_sha256=sha,
                row_pixels_base64=base64.b64encode(pixels.tobytes()).decode()))
    (source/'physical_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in physical))
    lineage,hand,bounds,_=evidence(db,observations,[ante,fold])
    actors=LineageActorProvider(lineage,bounds,[hand])
    provider=SimpleNamespace(hands={(1,'s','h'):hand},bounds=bounds,lineage=lineage,actor_sources=actors,player_registry=db)
    cases=build_cases(provider,source)
    assert len(cases)==expected_cases
    if not expected_cases:
        if anomaly:
            assert any(r['raw_text']=='unrelatedNoise' for r in lineage.observations)
        return
    assert cases[0]['identity_interval']==[20,None]
    assert {o['text'] for o in cases[0]['observations'] if o['kind']=='NICK'}==({'gretchin','unrelatedNoise'} if anomaly else {'gretchin'})
    if anomaly:
        assert not next(o for o in cases[0]['observations'] if o['text']=='unrelatedNoise')['selected']
    chat=next(o for o in cases[0]['observations'] if o['kind']=='CHAT')
    assert chat['source']['frame']==25 and chat['source']['raw_text']=='Dealer: gretchin folds'
    assert 'error' not in cases[0]


def test_case_hash_and_source_freshness_are_required(tmp_path):
    source=tmp_path/'source';source.mkdir();path=source/'physical_rows.jsonl';path.write_text('{}')
    item=case();item.pop('case_id');item['source_pins']={str(path):hashlib.sha256(path.read_bytes()).hexdigest()}
    item['case_id']=digest(item);verify_case(item,source)
    altered=copy.deepcopy(item);altered['context']['seat']='seat2'
    with pytest.raises(ValueError,match='modified'):verify_case(altered,source)
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='изменились'):verify_case(item,source)


def test_local_review_server_rejects_cross_origin_missing_token_and_forged_case(tmp_path):
    manifest=dict(source=str(tmp_path),cases=[case(actor='retreI')])
    with ReviewServer(('127.0.0.1',0),manifest,tmp_path/'players.sqlite',tmp_path) as server:
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def request(method,path,body=None,headers=None):
            conn=HTTPConnection('127.0.0.1',server.server_port)
            conn.request(method,path,body=body,headers=headers or {})
            response=conn.getresponse();result=(response.status,response.read());conn.close();return result
        try:
            assert request('GET','/cases')[0]==403
            headers={'X-Review-Token':server.token,'Content-Type':'application/json'}
            assert request('GET','/cases',headers=headers)[0]==200
            assert request('GET','/cases',headers=dict(headers,Origin='https://example.com'))[0]==403
            assert request('POST','/confirm',json.dumps(dict(case_id='forged',canonical='retrel',selected=['nick'])),headers)[0]==400
            assert not (tmp_path/'players.sqlite').exists()
            status,body=request('POST','/confirm',json.dumps(dict(case_id=manifest['cases'][0]['case_id'],canonical='retrel',selected=['nick','chat'])),headers)
            assert status==200 and json.loads(body)['canonical']=='retrel'
            assert PlayerRegistry(tmp_path/'players.sqlite').resolve('retre1','NICK')['canonical']=='retrel'
            status,body=request('GET','/cases',headers=headers)
            assert json.loads(body)['confirmations'][manifest['cases'][0]['case_id']]['selected']==['nick','chat']
            status,_=request('POST','/confirm',json.dumps(dict(case_id=manifest['cases'][0]['case_id'],canonical='retrel',selected=['nick'])),headers)
            assert status==200
            status,body=request('GET','/cases',headers=headers)
            assert json.loads(body)['confirmations'][manifest['cases'][0]['case_id']]['selected']==['nick']
            assert PlayerRegistry(tmp_path/'players.sqlite').resolve('retreI','CHAT') is None
        finally:server.shutdown();thread.join(timeout=2)
