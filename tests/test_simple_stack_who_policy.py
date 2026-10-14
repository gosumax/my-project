"""Focused contracts for independent stack values and fixed-seat lineage."""
import pytest

from parser_core.simple_stack_policy import parse_stack, normalize_stack_text, first_displayed_stack, corrected_start_stack
from parser_core.simple_seat_who import SeatIdentityLineage, nick_constraint


def active_identity_fixture(tmp_path,score=.90,clear_frame=30,messages=None):
    import base64,copy,hashlib,json
    import cv2
    import numpy as np
    from parser_core.nickname_roster_binding import _seat_rois
    from parser_core.simple_saved_policy import LineageActorProvider
    roi=_seat_rois()['slot_01']['seats']['seat1']['nick']
    pixels=np.arange((roi[3]-roi[1])*(roi[2]-roi[0]),dtype=np.uint8).reshape(roi[3]-roi[1],roi[2]-roi[0])
    sha=hashlib.sha256(pixels.tobytes()).hexdigest();_,png=cv2.imencode('.png',pixels)
    path=tmp_path/'seat_forced_inputs.jsonl';raw=[];observations=[]
    for line,(frame,confidence) in enumerate([(10,score),(11,.99)],1):
        response=dict(status='OK',raw_text='Alice',score=confidence,item_id=f'1:{frame}:100')
        raw.append(dict(kind='NICK',table_id=1,session_id='s',hand_id='h',seat='seat1',source_frame=frame,
            source_pts=frame,time_base=[1,30],roi=roi,source_ref=sha,pixel_sha256=sha,
            png_base64=base64.b64encode(png).decode(),ocr_response=response))
        observations.append(dict(table_id=1,session_id='s',hand_id='h',seat='seat1',frame=frame,
            source_pts=frame,time_base=[1,30],roi=roi,pixel_sha256=sha,raw_text='Alice',confidence=confidence,
            ocr_status='OK',item_id=response['item_id'],source_file=str(path),source_line=line))
    path.write_text(''.join(json.dumps(r)+'\n' for r in raw),encoding='utf-8')
    def message(action,frame,actor=None):
        return dict(action=action,player=actor,source_rows=[dict(frame_id=frame)])
    messages=messages or [message('ANTE',1,'Alice'),message('WIN',20,'Alice'),message('TABLE_INFO',40)]
    hand=dict(table_id='T01',session_id='s',hand_id='h',messages=messages)
    events=[] if clear_frame is None else [dict(table=1,session='s',hand_id='h',seat='seat1',
        frame=clear_frame,kind='EMPTY',evidence_status='PROVEN')]
    lineage=SeatIdentityLineage(observations,events)
    provider=LineageActorProvider(lineage,{(1,'s','h'):(0,50)},[hand],native_source=tmp_path)
    return provider,hand,copy.deepcopy(hand),lineage


@pytest.mark.parametrize('score,first_frame',[(.8999,11),(.90,10),(.9499,10),(.95,10)])
def test_active_nick90_uses_first_actual_qualified_source(tmp_path,score,first_frame):
    provider,hand,before,lineage=active_identity_fixture(tmp_path,score)
    proof=provider._hand_scope(hand,'Alice')
    assert proof['status']=='PROVEN' and proof['anchor']['frame']==first_frame
    assert len(proof['scope_nick_observations'])==1 and hand==before


def test_post_win_elimination_retains_prior_participant_identity(tmp_path):
    provider,hand,before,lineage=active_identity_fixture(tmp_path)
    proof=provider._hand_scope(hand,'Alice')
    assert proof['status']=='PROVEN' and proof['seat']=='seat1'
    assert proof['identity_interval']==[0,20] and hand==before
    assert any(b['kind']=='EMPTY' and b['frame']==30 for b in lineage.boundaries[(1,'s','seat1')])


def test_clear_before_final_payout_still_blocks_identity(tmp_path):
    messages=[dict(action=a,player='Alice',source_rows=[dict(frame_id=f)]) for a,f in
              [('ANTE',1),('WIN',20),('WIN',35),('UNCALLED_RETURN',36),('SHOW',37),('TABLE_INFO',40)]]
    provider,hand,_,_=active_identity_fixture(tmp_path,messages=messages,clear_frame=30)
    assert provider._hand_scope(hand,'Alice') is None


def test_all_sidepot_payouts_returns_show_and_no_win_partial_keep_material_bounds(tmp_path):
    messages=[dict(action=a,player='Alice',source_rows=[dict(frame_id=f)]) for a,f in
              [('ANTE',1),('WIN',20),('WIN',35),('UNCALLED_RETURN',36),('NO_SHOW',37),('TABLE_INFO',40)]]
    provider,hand,_,_=active_identity_fixture(tmp_path,messages=messages,clear_frame=39)
    assert provider._hand_scope(hand,'Alice')['identity_interval']==[0,37]
    hand['messages']=[dict(action='ANTE',player='Alice',source_rows=[dict(frame_id=1)]),
                      dict(action='FOLD',player='Alice',source_rows=[dict(frame_id=20)])]
    provider.scope_observations.clear();provider.scope_labels.clear()
    assert provider._hand_scope(hand,'Alice')['identity_interval']==[0,20]


@pytest.mark.parametrize('raw,expected', [('_6332',6332),('_28 039',28039),('_ABC',None),('__6332',None),('63_32',None),('28\u00a0039',28039)])
def test_stack_only_leading_underscore_numeric(raw, expected):
    assert parse_stack(raw)==expected
    if expected is None:
        assert normalize_stack_text(raw)==raw


def test_first_qualified_numeric_survives_diagnostics_and_unknown_who():
    rows=[dict(table_id=1,session_id='session',seat='seat4',frame=99,raw_text='999'),
          dict(table_id=1,session_id='session',seat='seat4',frame=101,raw_text='ANTE'),
          dict(table_id=1,session_id='session',seat='seat4',frame=170,raw_text='_6332',confidence=.91,
               WHO_status='UNKNOWN',old_source_status='UNKNOWN',sql_context='conflict'),
          dict(table_id=1,session_id='session',seat='seat4',frame=171,raw_text='6400',confidence=.99)]
    first=first_displayed_stack(rows,table_id=1,session_id='session',seat='seat4',hand_start=100,hand_end=200)
    assert first['displayed_stack']==6332
    assert first['raw_text']=='_6332' and first['confidence']==.91
    assert corrected_start_stack(first['displayed_stack'],{'ANTE':None,'SB':None,'BB':None}) is None
    assert first['displayed_stack']==6332


@pytest.mark.parametrize('raw,confidence,expected', [
    ('6332', .849, None), ('6332', .850, 6332), ('6332', .90, 6332), ('6332', .95, 6332),
    ('18544', .862634, 18544), ('28039', .868614, 28039), ('1', .566404, None),
    ('_6332', .85, 6332), ('_6332', .80, None),
    ('_28 039', .90, 28039), ('6332', None, None), ('6332', float('nan'), None),
    ('6332', float('inf'), None),
])
def test_simple_stack_numeric_confidence_threshold(raw, confidence, expected):
    row = dict(table_id=1, session_id='session', seat='seat4', frame=110,
               raw_text=raw, confidence=confidence)
    result = first_displayed_stack([row], table_id=1, session_id='session',
        seat='seat4', hand_start=100, hand_end=200)
    assert (result['displayed_stack'] if result else None) == expected


def test_low_confidence_early_junk_is_skipped_for_later_numeric():
    rows = [dict(table_id=1, session_id='session', seat='seat4', frame=110,
                 raw_text='1', confidence=.56),
            dict(table_id=1, session_id='session', seat='seat4', frame=111,
                 raw_text='_28 039', confidence=.85, WHO_status='UNKNOWN',
                 old_source_status='UNKNOWN', sql_context='conflict')]
    first = first_displayed_stack(rows[::-1], table_id=1, session_id='session',
        seat='seat4', hand_start=100, hand_end=200)
    assert first['frame'] == 111 and first['displayed_stack'] == 28039
    assert first['raw_text'] == '_28 039' and first['confidence'] == .85


def test_stack_route_and_hand_filters_prevent_cross_seat_or_hand_leak():
    rows=[dict(table_id=1,session_id='session',seat='seat5',frame=110,raw_text='111'),
          dict(table_id=2,session_id='session',seat='seat4',frame=110,raw_text='222'),
          dict(table_id=1,session_id='other',seat='seat4',frame=110,raw_text='333'),
          dict(table_id=1,session_id='session',seat='seat4',frame=201,raw_text='444')]
    assert first_displayed_stack(rows,table_id=1,session_id='session',seat='seat4',hand_start=100,hand_end=200) is None


def test_sitout_without_numeric_has_no_fabricated_stack():
    rows=[dict(table_id=1,session_id='session',seat='seat7',frame=110,raw_text='Sit out',confidence=1)]
    assert first_displayed_stack(rows,table_id=1,session_id='session',seat='seat7',hand_start=100,hand_end=200) is None
    assert corrected_start_stack(None,{'ANTE':70,'SB':0,'BB':0}) is None
    assert corrected_start_stack(6332,{'ANTE':70,'SB':0,'BB':300})==6702


def nick(frame, text, seat='seat4', session='session'):
    return dict(table_id=1,session_id=session,seat=seat,frame=frame,raw_text=text,confidence=.97)


def resolve(lineage, actor='romashka91808',low=100,high=200,actors=()):
    return lineage.for_actor(1,'session',actor,low,high,actors=actors)


def test_exact_nick_in_fixed_seat_roi():
    result=resolve(SeatIdentityLineage([nick(110,'romashka91808')]))
    assert result['status']=='PROVEN' and result['seat']=='seat4'


def test_late_nick_after_header64_and_after_hand_can_bind_same_lineage():
    result=resolve(SeatIdentityLineage([nick(500,'romashka91808')]))
    assert result['status']=='PROVEN' and result['rule_used']=='LATE_NATIVE_NICK'


def test_prefix_nick_chat_actor_preserves_roi_seat():
    result=resolve(SeatIdentityLineage([nick(110,'romashka...')]),actors=['romashka91808'])
    assert result['status']=='PROVEN' and result['rule_used']=='PREFIX_NICK_CHAT'
    assert result['seat']=='seat4'


def test_prefix_cannot_select_two_literal_chat_actors():
    result=resolve(SeatIdentityLineage([nick(110,'romashka...')]),actors=['romashka91808','romashka999'])
    assert result['status']=='UNKNOWN'


def test_same_occupant_sitout_sitin_carries_across_hands():
    events=[dict(table=1,session='session',seat='seat4',frame=120,kind='SIT_OUT',evidence_status='PROVEN'),
            dict(table=1,session='session',seat='seat4',frame=180,kind='SIT_IN',evidence_status='PROVEN')]
    result=resolve(SeatIdentityLineage([nick(10,'romashka91808')],events))
    assert result['status']=='PROVEN' and result['rule_used']=='CARRIED_OCCUPANT'


def test_one_bad_ocr_does_not_replace_occupant():
    lineage=SeatIdentityLineage([nick(10,'romashka91808'),nick(110,'romashka9I808'),nick(120,'romashka91808')])
    assert resolve(lineage)['status']=='PROVEN'
    assert resolve(lineage,'romashka9I808')['status']=='UNKNOWN'


def test_open_seat_clears_occupant_and_prevents_future_backfill_across_empty():
    lineage=SeatIdentityLineage([nick(10,'romashka91808'),nick(90,'OPEN SEAT'),nick(220,'newplayer')])
    assert resolve(lineage)['status']=='UNKNOWN'
    assert resolve(lineage,'newplayer',low=20,high=80)['status']=='UNKNOWN'
    assert resolve(lineage,'newplayer')['status']=='PROVEN'


def test_two_new_nick_frames_replace_existing_occupant_at_first_new_frame():
    lineage=SeatIdentityLineage([nick(10,'romashka91808'),nick(110,'newplayer'),nick(120,'newplayer')])
    assert resolve(lineage,low=130,high=200)['status']=='UNKNOWN'
    result=resolve(lineage,'newplayer',low=130,high=200)
    assert result['status']=='PROVEN' and result['identity_interval'][0]==110


@pytest.mark.parametrize('status',['CALL','RAISE','BET','CHECK','FOLD','ALL-IN','SIT OUT','TIME','OPEN SEAT'])
def test_ui_status_is_not_nickname(status):
    assert nick_constraint(status) is None


def test_other_session_does_not_supply_identity():
    assert resolve(SeatIdentityLineage([nick(110,'romashka91808',session='other')]))['status']=='UNKNOWN'


def test_saved_provider_keeps_unbound_numeric_and_separate_sitout_status(tmp_path):
    from sitout_status_fixture import status_source
    from parser_core.simple_saved_policy import SimpleSavedPolicyProvider, SIT_OUT_GATE_AUTHORITY
    hand=dict(table_id=1,session_id='session',hand_id='hand',messages=[])
    observations={(1,'session','seat4'):[dict(table_id=1,session_id='session',seat='seat4',
                    frame=110,raw_text='_6332',confidence=.91)]}
    provider=SimpleSavedPolicyProvider([hand],{(1,'session','hand'):(100,200)},observations,[],
        status_observations=[status_source(tmp_path,session='session',hand='hand',seat='seat7',frame=120)],
        layout={'slot_01':{'seats':{'seat4':{'stack':[1,2,3,4]},'seat7':{'stack':[5,6,7,8]}}}})
    byseat={r['seat']:r for r in provider.seat_rows(hand)}
    assert byseat['seat4']['DISPLAYED_STACK']==6332
    assert byseat['seat4']['WHO_status']=='UNKNOWN' and byseat['seat4']['START_STACK'] is None
    assert byseat['seat7']['DISPLAYED_STACK'] is None
    assert byseat['seat7']['player_status'] is None
    assert byseat['seat7']['reason']=='SIT_OUT_ACTOR_SEAT_UNPROVEN'
    assert byseat['seat7']['sitout_authority'] is None


def test_sitout_and_valid_numeric_are_explicit_conflict_not_status_exemption(tmp_path):
    from sitout_status_fixture import status_source
    from parser_core.simple_saved_policy import SimpleSavedPolicyProvider, SIT_OUT_GATE_AUTHORITY
    hand=dict(table_id=1,session_id='session',hand_id='hand',messages=[])
    observations={(1,'session','seat7'):[dict(table_id=1,session_id='session',seat='seat7',
        frame=120,raw_text='6332',confidence=.91)]}
    status=status_source(tmp_path,session='session',hand='hand',seat='seat7',frame=120)
    provider=SimpleSavedPolicyProvider([hand],{(1,'session','hand'):(100,200)},observations,[],
        status_observations=[status],layout={'slot_01':{'seats':{'seat7':{'stack':[5,6,7,8]}}}})
    row=provider.seat_rows(hand)[0]
    assert row['player_status'] is None and row['sitout_sources']==[status]
    assert row['sitout_numeric_conflict'] is True and row['reason']=='SIT_OUT_NUMERIC_CONFLICT'


def test_same_actor_reappears_after_repeated_overlay_ocr_fragments():
    lineage=SeatIdentityLineage([nick(10,'romashka91808'),nick(110,'ANTF'),nick(111,'ANTF'),
                                nick(120,'romashka91808'),nick(121,'romashka91808')])
    result=resolve(lineage)
    assert result['status']=='PROVEN' and result['seat']=='seat4'
    assert len(result['identity_intervals'])==2


def test_two_real_actors_claiming_one_seat_in_one_hand_stay_unknown():
    from parser_core.simple_saved_policy import LineageActorProvider
    lineage=SeatIdentityLineage([nick(10,'firstplayer'),nick(110,'secondplayer'),nick(111,'secondplayer')])
    hand=dict(table_id=1,session_id='session',hand_id='hand',messages=[
        dict(player='firstplayer',source_rows=[]),dict(player='secondplayer',source_rows=[])])
    provider=LineageActorProvider(lineage,{(1,'session','hand'):(100,200)},[hand])
    assert provider.for_actor(hand,'firstplayer')['status']=='UNKNOWN'
    assert provider.for_actor(hand,'secondplayer')['status']=='UNKNOWN'


def test_late_nick_capture_uses_fake_transport_beyond_deal64():
    from collections import Counter
    from types import SimpleNamespace
    import threading
    import numpy as np
    from parser_core.seat_forced_binding import SeatForcedBinding
    key=(1,'session','hand')
    packets=[]
    binding=SeatForcedBinding.__new__(SeatForcedBinding)
    binding.sidecar=SimpleNamespace(lock=threading.RLock(),current={1:{'session_id':'session','hand_id':'hand'}},
        windows={key:SimpleNamespace(deal={'source_frame':10,'occupied_seats':['seat4']},closed=None)},
        triggers=[dict(table_id=1,session_id='session',hand_id='hand',source_frame=1)],
        flags={'capture_frames':64},profile={'slots':{'slot_01':{'seats':{'seat4':{'nick':[0,0,10,5]}}}}})
    binding.lock=threading.RLock()
    binding.occupancy=None
    binding.machine=SimpleNamespace(hand=lambda *args:None)
    binding.seen_deals={key}
    binding.seen_prefix_candidates=set()
    binding.roster_prefix_binding=True
    binding.nick_schedule={}
    binding.jobs={}
    binding.counts=Counter()
    binding.ocr=SimpleNamespace(submit_batch=lambda batch:packets.extend(batch))
    binding.observe_frame(1,'session',100,51200,[1,15360],np.zeros((10,20,3),dtype=np.uint8),np.zeros((5,10),dtype=np.uint8))
    assert len(packets)==1
    meta=next(iter(binding.jobs.values()))[1]
    assert meta['kind']=='NICK' and meta['seat']=='seat4' and meta['source_frame']==100


def test_simple_stack_receipt_is_consumed_by_hh_export_without_raw_changes(tmp_path):
    import copy
    import json
    from parser_core.simple_saved_policy import SimpleSavedPolicyProvider
    from parser_core.hh_export import export_hand_histories
    def message(player, action, amount, text):
        return dict(player=player,action=action,amount=str(amount),raw_text=text,
                    source_rows=[dict(confidence=.99,raw_ocr_text=text,frame_id=100)],source_row_ids=[text])
    hand=dict(table_id='T01',session_id='session',hand_id='hand',start_observed=True,end_observed=True,
              boundary_conflicts=[],messages=[message('A','ANTE',2,'Dealer: A posts Ante 2'),
                  message('B','ANTE',2,'Dealer: B posts Ante 2'),
                  message('A','SMALL_BLIND',5,'Dealer: A posts Small Blind 5'),
                  message('B','BIG_BLIND',10,'Dealer: B posts Big Blind 10')])
    original=copy.deepcopy(hand)
    nicks=[dict(nick(100,'A',seat='seat1'),confidence=.99),dict(nick(100,'B',seat='seat2'),confidence=.99)]
    observations={(1,'session',seat):[dict(table_id=1,session_id='session',seat=seat,frame=100,raw_text=str(value),confidence=.91)]
                  for seat,value in [('seat1',93),('seat2',88)]}
    provider=SimpleSavedPolicyProvider([hand],{(1,'session','hand'):(100,200)},observations,nicks,
        layout={'slot_01':{'seats':{'seat1':{'stack':[1,2,3,4]},'seat2':{'stack':[5,6,7,8]}}}})
    record=dict(segment_id='segment',observed_hand=hand,lifecycle={'status':'COMPLETE'},fully_validated=True,
        validation={'action_logic_status':'PASS','card_status':'PASS','identity_status':'PASS'},diagnostics={'math':{'pass':True}})
    stackrows=provider.player_rows(hand,segment_id='segment')
    (tmp_path/'stack_start_stack.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in stackrows),encoding='utf8')
    (tmp_path/'poker_reconstruction.jsonl').write_text(json.dumps({'segment_id':'segment','output_eligible':False,'output_blockers':[]})+'\n',encoding='utf8')
    export_hand_histories(tmp_path,[record],simple_provider=provider,actor_provider=provider.actor_sources)
    saved=json.loads((tmp_path/'hand_histories.jsonl').read_text('utf8'))
    assert all(p['START_STACK_status']=='PROVEN' and p['START_STACK']==100 for p in saved['players'])
    assert saved['observed_actions']==hand['messages'] and hand==original


def test_exact_native_nick_seat_identity_does_not_require_chat_actor_coverage():
    from parser_core.simple_saved_policy import SimpleSavedPolicyProvider
    hand=dict(table_id=1,session_id='session',hand_id='hand',messages=[])
    provider=SimpleSavedPolicyProvider([hand],{(1,'session','hand'):(100,200)},
        {(1,'session','seat4'):[dict(table_id=1,session_id='session',seat='seat4',frame=110,raw_text='171250',confidence=.91)]},
        [nick(110,'temazm')],layout={'slot_01':{'seats':{'seat4':{'stack':[1,2,3,4]}}}})
    row=provider.seat_rows(hand)[0]
    assert row['WHO_status']=='PROVEN' and row['player']=='temazm'
    assert row['DISPLAYED_STACK']==171250 and row['START_STACK'] is None


def test_standalone_nick_does_not_choose_high_confidence_across_real_replacement():
    from parser_core.simple_saved_policy import SimpleSavedPolicyProvider
    hand=dict(table_id=1,session_id='session',hand_id='hand',messages=[])
    nicks=[dict(nick(110,'Alice'),confidence=.99),dict(nick(150,'Bob'),confidence=.97),
           dict(nick(151,'Bob'),confidence=.97)]
    provider=SimpleSavedPolicyProvider([hand],{(1,'session','hand'):(100,200)}, {},nicks,
        layout={'slot_01':{'seats':{'seat4':{'stack':[1,2,3,4]}}}})
    row=provider.seat_rows(hand)[0]
    assert row['WHO_status']=='MANUAL_REVIEW' and row['player'] is None
    selection=row['identity_proof']['selection']
    assert selection['canonical_nick'] is None
    assert selection['association_status']=='MULTIPLE_OCCUPANT_INTERVALS'
    assert {'Alice','Bob'}=={r['label']['text'] for r in selection['occupant_interval_sources']}


def test_standalone_manual_override_preserves_auto_raw_and_review_anomaly():
    from parser_core.simple_saved_policy import SimpleSavedPolicyProvider
    hand=dict(table_id=1,session_id='session',hand_id='hand',messages=[])
    nicks=[dict(nick(110,'Alice'),confidence=.99),dict(nick(115,'AliceI'),confidence=.50),
           dict(nick(120,'Alice'),confidence=.99)]
    manual=dict(table=1,session='session',hand='hand',seat='seat4',
                manual_ground_truth='ConfirmedAlice',manual_confirmed=True)
    provider=SimpleSavedPolicyProvider([hand],{(1,'session','hand'):(100,200)}, {},nicks,
        manual_reviews=[manual],layout={'slot_01':{'seats':{'seat4':{'stack':[1,2,3,4]}}}})
    row=provider.seat_rows(hand)[0]
    assert row['WHO_status']=='PROVEN' and row['player']=='ConfirmedAlice'
    selection=row['identity_proof']['selection']
    assert selection['auto_selected_nick']=='Alice'
    assert selection['nick_roi_raw_text']=='Alice' and selection['manual_confirmed'] is True
    queue=provider.actor_sources.review_queue()
    assert len(queue)==1 and queue[0]['canonical_nick']=='ConfirmedAlice'
    assert any(r['raw_text']=='AliceI' for r in queue[0]['anomalous_nick_observations'])
    assert nicks[1]['raw_text']=='AliceI'


# Native saved-pixel regressions: no OCR, resizing, or generated pixels.
import copy
import json
from pathlib import Path
from parser_core.chat_nick_actor import prove_nick_ui_overlay
from parser_core.simple_seat_who import nick_observation_constraint


def overlay_source(frame):
    return json.loads((Path(__file__).parent/'fixtures/nick_ui_overlay_native_sources.json').read_text('utf-8'))[str(frame)]


def overlay_observation(frame):
    source=overlay_source(frame)
    response=source['ocr_response']
    return dict(table_id=source['table_id'],session_id=source['session_id'],seat=source['seat'],
        hand_id=source['hand_id'],frame=frame,raw_text=response['raw_text'],confidence=response['score'],
        ocr_status=response['status'],item_id=response['item_id'],roi=source['roi'],
        pixel_sha256=source['pixel_sha256'],source_pts=source['source_pts'],time_base=source['time_base'],
        native_nick_source=source)


@pytest.mark.parametrize('frame,canonical',[(1185,'FOLD'),(1180,'ANTE'),(1168,'ANTE')])
def test_native_ui_alias_excluded_without_rewriting_raw(frame,canonical):
    row=overlay_observation(frame); original=copy.deepcopy(row)
    assert prove_nick_ui_overlay(row)['canonical_ui']==canonical
    assert nick_observation_constraint(row) is None
    assert row==original


@pytest.mark.parametrize('raw',['FOLD','ANTE'])
def test_exact_ui_is_not_nickname(raw):
    assert nick_constraint(raw) is None


def test_overlay_preserves_lineage_and_real_replacement():
    rows=[overlay_observation(f) for f in (520,1168,1180,1185,1220,1760)]
    original=copy.deepcopy(rows)
    route=(6,rows[0]['session_id'],'seat2')
    lineage=SeatIdentityLineage(rows)
    assert lineage.observations==original
    assert len(lineage.segments[route])==1
    assert not lineage.boundaries[route]
    assert {r['source']['frame'] for r in lineage.ui_overlay_rejections}=={1168,1180,1185,1220}
    canonical_fold=next(r for r in lineage.ui_overlay_rejections if r['source']['frame']==1220)
    assert canonical_fold['source']['raw_text']=='FOLD'
    assert canonical_fold['proof']['canonical_ui']=='FOLD'
    assert lineage.for_seat(*route,1185)['label']['text'].startswith('vlad1994q')
    newer=[dict(table_id=6,session_id=route[1],seat='seat2',frame=f,raw_text='RealNewNick') for f in (1770,1771)]
    after=SeatIdentityLineage(rows+newer)
    assert after.boundaries[route][0]['kind']=='STABLE_NEW_NICK'
    assert after.boundaries[route][0]['frame']==1770
    assert after.for_seat(*route,1771)['label']['text']=='RealNewNick'


@pytest.mark.parametrize('raw',['5OLD','ANTt','5OLD...','FOLDman','ANTtHero'])
def test_similar_real_nickname_requires_positive_native_ui_proof(raw):
    assert nick_constraint(raw) is not None
    source=overlay_source(520)
    source['ocr_response']['raw_text']=raw
    assert prove_nick_ui_overlay(source) is None
    assert nick_observation_constraint(dict(raw_text=raw)) is not None


@pytest.mark.parametrize('field,value',[('pixel_sha256','bad'),('roi',[0,0,83,17]),('source_pts',None),('raw_text','ANTt')])
def test_ui_source_wrapper_tamper_rejected(field,value):
    row=overlay_observation(1185);row[field]=value
    with pytest.raises(ValueError):prove_nick_ui_overlay(row)


def test_alias_without_native_source_retains_literal_candidate():
    row=overlay_observation(1185);del row['native_nick_source']
    assert nick_observation_constraint(row)==dict(kind='EXACT',text='5OLD')



def test_native_pixel_twin_other_geometry_and_changed_pixels():
    import base64
    import hashlib
    import cv2
    import numpy as np
    source=json.loads((Path(__file__).parent/'fixtures/nick_ui_overlay_native_sources.json').read_text('utf-8'))['twin81x19']
    assert prove_nick_ui_overlay(source)['rule']=='POKERDOM_NATIVE_NICK_UI_PIXEL_TWIN_V1'
    pixels=cv2.imdecode(np.frombuffer(base64.b64decode(source['png_base64']),np.uint8),cv2.IMREAD_GRAYSCALE)
    pixels[0,0]^=1
    source['png_base64']=base64.b64encode(cv2.imencode('.png',pixels)[1]).decode()
    source['pixel_sha256']=source['source_ref']=hashlib.sha256(pixels.tobytes()).hexdigest()
    assert prove_nick_ui_overlay(source) is None
