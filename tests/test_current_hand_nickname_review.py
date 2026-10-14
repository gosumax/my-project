"""Manual spelling review may bind early CHAT only to its current saved hand."""
import base64
import copy
import hashlib
import json
import shutil
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from parser_core.nickname_review import build_cases
from parser_core.nickname_roster_binding import _seat_rois
from parser_core.player_registry import PlayerRegistry, CurrentHandReviewAssociation, digest
from parser_core.simple_saved_policy import LineageActorProvider, saved_hand_bounds, project_economic_actor_aliases
from parser_core.simple_seat_who import SeatIdentityLineage
from scripts.review_nicknames import verify_case, load_cached_cases


def fixture(tmp_path,extra_nicks=(),chat_actor='retre1'):
    source=tmp_path/'source';source.mkdir()
    roi=_seat_rois()['slot_01']['seats']['seat1']['nick']
    pixels=np.arange((roi[3]-roi[1])*(roi[2]-roi[0]),dtype=np.uint8).reshape(roi[3]-roi[1],roi[2]-roi[0])
    sha=hashlib.sha256(pixels.tobytes()).hexdigest();_,png=cv2.imencode('.png',pixels)
    observations=[];native=[]
    samples=[(10,'retre1',.82),(11,'retre1',.82),(20,'retrel',.82),(21,'retrel',.82),*extra_nicks]
    for line,(frame,raw,confidence) in enumerate(sorted(samples),1):
        native.append(dict(table_id=1,session_id='s',hand_id='h',seat='seat1',kind='NICK',source_frame=frame,
            source_pts=frame,time_base=[1,30],roi=roi,pixel_sha256=sha,source_ref=sha,
            png_base64=base64.b64encode(png).decode(),ocr_response=dict(status='OK',raw_text=raw,score=confidence)))
        observations.append(dict(table_id=1,session_id='s',hand_id='h',seat='seat1',frame=frame,
            raw_text=raw,confidence=confidence,source_file=str(source/'seat_forced_inputs.jsonl'),source_line=line,
            pixel_sha256=sha,roi=roi,source_pts=frame,time_base=[1,30]))
    (source/'seat_forced_inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in native),encoding='utf-8')
    messages=[];physical=[]
    chat_pixels=np.arange(64,dtype=np.uint8).reshape(4,16)
    chat_sha=hashlib.sha256(chat_pixels.tobytes()).hexdigest()
    for actor,frame,action,text in [(chat_actor,1,'ANTE','posts Ante 700'),('retrel',25,'FOLD','folds')]:
        raw=f'{actor} {text}' if actor.startswith('Dealer; ') else f'Dealer: {actor} {text}'
        row_id=f's:{frame}'
        messages.append(dict(player=actor,action=action,source_rows=[dict(table_id='T01',session_id='s',
            frame_id=frame,stable_row_id=row_id,row_pixel_sha256=chat_sha,raw_ocr_text=raw,confidence=.99)]))
        physical.append(dict(table_id=1,table_session_id='s',frame_id=frame,row_id=row_id,
            pixel_dtype='uint8',pixel_shape=[4,16],row_pixel_sha256=chat_sha,
            row_pixels_base64=base64.b64encode(chat_pixels.tobytes()).decode(),raw_ocr_text=raw))
    (source/'physical_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in physical),encoding='utf-8')
    lineage=SeatIdentityLineage(observations)
    hand=dict(table_id='T01',session_id='s',hand_id='h',messages=messages)
    bounds={(1,'s','h'):(0,30)};db=PlayerRegistry(tmp_path/'players.sqlite')
    actors=LineageActorProvider(lineage,bounds,[hand])
    provider=SimpleNamespace(hands={(1,'s','h'):hand},bounds=bounds,lineage=lineage,
                             actor_sources=actors,player_registry=db)
    cases=build_cases(provider,source)
    assert len(cases)==1 and cases[0]['review_scope']=='HAND_SEAT' and 'error' not in cases[0]
    return source,lineage,hand,bounds,db,cases[0]


def confirm(db,case,selected=None):
    db.confirm(case,'retrel',selected if selected is not None else [o['id'] for o in case['observations']])
    return PlayerRegistry(db.path)


def single_spelling_fixture(tmp_path,score,with_chat=True,raw='retrel'):
    source,lineage,hand,bounds,db,_=fixture(tmp_path)
    native=[json.loads(t) for t in (source/'seat_forced_inputs.jsonl').read_text('utf-8').splitlines()]
    for row in native:
        row['ocr_response'].update(raw_text=raw,score=score)
    (source/'seat_forced_inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in native),encoding='utf-8')
    observations=[dict(r,raw_text=raw,confidence=score) for r in lineage.observations]
    lineage=SeatIdentityLineage(observations)
    hand['messages']=hand['messages'][1:]
    if with_chat:
        hand['messages'][0]['source_rows'][0]['frame_id']=1
        physical=[json.loads(t) for t in (source/'physical_rows.jsonl').read_text('utf-8').splitlines()]
        physical[1]['frame_id']=1
        (source/'physical_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in physical),encoding='utf-8')
    else:
        hand['messages']=[]
    actors=LineageActorProvider(lineage,bounds,[hand],native_source=source)
    provider=SimpleNamespace(hands={(1,'s','h'):hand},bounds=bounds,lineage=lineage,actor_sources=actors,player_registry=db)
    return source,lineage,hand,bounds,db,provider


@pytest.mark.parametrize('score',[.85,.87,.8999])
def test_single_weak_nick_with_early_identical_chat_requires_human_not_auto_binding(tmp_path,score):
    from parser_core.nickname_review import unreviewed_cases
    source,lineage,hand,bounds,db,provider=single_spelling_fixture(tmp_path,score)
    cases=build_cases(provider,source)
    assert len(cases)==1 and cases[0]['review_scope']=='HAND_SEAT' and 'error' not in cases[0]
    assert unreviewed_cases(cases,db)==cases
    assert provider.actor_sources._proofs(hand)['retrel']['status']=='UNKNOWN'
    verify_case(cases[0],source)
    db=confirm(db,cases[0])
    associations,manual,rejected=db.hand_evidence(lineage,[hand],bounds,source=source)
    assert len(associations)==1 and not rejected


@pytest.mark.parametrize('score',[.8499,.90,.998])
def test_identical_spelling_outside_new_manual_band_not_added_by_new_rule(tmp_path,score):
    source,_,_,_,_,provider=single_spelling_fixture(tmp_path,score,with_chat=False)
    assert build_cases(provider,source)==[]


def test_uncorroborated_native_score_does_not_make_ui_text_a_player(tmp_path):
    from parser_core.nickname_review import unreviewed_cases
    source,lineage,hand,bounds,db,provider=single_spelling_fixture(tmp_path,.87,with_chat=False)
    cases=build_cases(provider,source)
    assert cases==[] and unreviewed_cases(cases,db)==[]
    assert not db.path.exists()


def test_service_text_does_not_enter_new_native_manual_band(tmp_path):
    source,_,_,_,_,provider=single_spelling_fixture(tmp_path,.87,with_chat=False,raw='SB')
    assert build_cases(provider,source)==[]


def test_early_chat_requires_explicit_human_review_and_retains_native_intervals(tmp_path):
    source,lineage,hand,bounds,db,case=fixture(tmp_path)
    before=copy.deepcopy(hand);segments=copy.deepcopy(dict(lineage.segments))
    verify_case(case,source)
    assert db.hand_evidence(lineage,[hand],bounds,source=source)==([],[],[])
    db=confirm(db,case)
    associations,manual,rejected=db.hand_evidence(lineage,[hand],bounds,source=source)
    assert len(associations)==len(manual)==1 and not rejected
    assert isinstance(associations[0],CurrentHandReviewAssociation)
    assert associations[0]['registry_evidence']['early_chat_before_first_nick']
    assert associations[0]['registry_evidence']['native_intervals']==[[0,19],[20,None]]
    actors=LineageActorProvider(lineage,bounds,[hand],associations,manual)
    assert actors.actor_map(hand)=={'retre1':'retrel','retrel':'retrel'}
    assert all(actors.for_actor(hand,a)['rule_used']=='USER_CONFIRMED_CURRENT_HAND_SEAT'
               for a in ('retre1','retrel'))
    canonical=actors.canonical_hand(hand)
    assert [m['player'] for m in canonical['messages']]==['retrel','retrel']
    assert canonical['messages'][0]['source_rows']==hand['messages'][0]['source_rows']
    assert hand==before and dict(lineage.segments)==segments


def test_serialized_authority_cannot_recreate_current_hand_capability(tmp_path):
    source,lineage,hand,bounds,db,case=fixture(tmp_path);db=confirm(db,case)
    associations,manual,_=db.hand_evidence(lineage,[hand],bounds,source=source)
    copied=json.loads(json.dumps(associations))
    provider=LineageActorProvider(lineage,bounds,[hand],copied,manual)
    assert not provider.associations and not provider.manual_reviews
    assert all(p.get('rule_used')!='USER_CONFIRMED_CURRENT_HAND_SEAT' for p in provider._proofs(hand).values())


def rerun_source(tmp_path,source,lineage):
    """Same physical CHAT; unrelated native bookkeeping shifts JSONL ordinals."""
    fresh=tmp_path/'rerun';fresh.mkdir()
    shutil.copyfile(source/'physical_rows.jsonl',fresh/'physical_rows.jsonl')
    native=(source/'seat_forced_inputs.jsonl').read_text('utf-8')
    (fresh/'seat_forced_inputs.jsonl').write_text('\n'+native,encoding='utf-8')
    rows=[dict(row,source_file=str(fresh/'seat_forced_inputs.jsonl'),
               source_line=row['source_line']+1) for row in lineage.observations]
    return fresh,SeatIdentityLineage(rows)


def test_review_reuses_exact_native_parents_in_rerun_without_reconfirming(tmp_path):
    source,lineage,hand,bounds,db,case=fixture(tmp_path);db=confirm(db,case)
    reviews=copy.deepcopy(db.reviews);db_bytes=db.path.read_bytes()
    fresh,current=rerun_source(tmp_path,source,lineage)
    raw=copy.deepcopy(current.observations)
    associations,manual,rejected=db.hand_evidence(current,[hand],bounds,source=fresh)
    assert len(associations)==len(manual)==1 and not rejected
    assert isinstance(associations[0],CurrentHandReviewAssociation)
    evidence=associations[0]['registry_evidence'];binding=evidence['source_rebinding']
    assert binding['rule']=='SAME_PHYSICAL_CHAT_EXACT_REVIEWED_NATIVE_PARENTS_V1'
    assert binding['physical_chat_byte_parity'] is True
    assert binding['reviewed_source_pins']==case['source_pins']
    assert set(evidence['source_pins'])=={str(fresh/name) for name in
        ('physical_rows.jsonl','seat_forced_inputs.jsonl')}
    assert all(parent['current_source']['source_line']==parent['reviewed_source']['line']+1
               for parent in binding['native_parents'])
    actors=LineageActorProvider(current,bounds,[hand],associations,manual)
    assert actors.actor_map(hand)=={'retre1':'retrel','retrel':'retrel'}
    assert db.reviews==reviews and db.path.read_bytes()==db_bytes
    assert current.observations==raw


@pytest.mark.parametrize('change,reason',[
    ('changed_chat','CURRENT_REVIEW_SOURCE_CHANGED'),
    ('original_pin','CURRENT_REVIEW_SOURCE_CHANGED'),
    ('pixel','CURRENT_REVIEW_NICK_PARENT_CONFLICT'),
    ('clock','CURRENT_REVIEW_NICK_PARENT_CONFLICT'),
    ('roi','CURRENT_REVIEW_NICK_PARENT_CONFLICT'),
    ('duplicate','CURRENT_REVIEW_NICK_PARENT_CONFLICT'),
    ('interval','CURRENT_REVIEW_NATIVE_INTERVAL_CHANGED'),
    ('clear','CURRENT_REVIEW_PHYSICAL_CLEAR_OR_REPLACEMENT'),
    ('other_seat','CURRENT_REVIEW_ACTOR_AT_OTHER_SEAT'),
    ('unselected','CURRENT_REVIEW_UNSELECTED_STABLE_NICK'),
])
def test_rerun_rebinding_preserves_source_parent_and_occupant_guards(tmp_path,change,reason):
    extra=[(28,'anotherPlayer',.99),(29,'anotherPlayer',.99)] if change=='unselected' else []
    source,lineage,hand,bounds,db,case=fixture(tmp_path,extra);db=confirm(db,case)
    fresh,current=rerun_source(tmp_path,source,lineage)
    if change=='changed_chat':
        with (fresh/'physical_rows.jsonl').open('a',encoding='utf-8') as stream:stream.write('\n')
    elif change=='original_pin':
        with (source/'seat_forced_inputs.jsonl').open('a',encoding='utf-8') as stream:stream.write('\n')
    elif change=='pixel':current.observations[0]['pixel_sha256']='0'*64
    elif change=='clock':current.observations[0]['source_pts']+=1
    elif change=='roi':current.observations[0]['roi']=[1,2,3,4]
    elif change=='duplicate':current.observations.append(dict(current.observations[0]))
    elif change=='interval':current.segments[(1,'s','seat1')][0]['start']=1
    elif change=='clear':current.boundaries[(1,'s','seat1')].append(dict(frame=17,kind='EMPTY'))
    elif change=='other_seat':
        row=dict(current.observations[0],seat='seat2')
        current.segments[(1,'s','seat2')]=[dict(start=0,end=None,label=dict(kind='EXACT',text='retre1'),
            observations=[row],ignored=[])]
    associations,manual,rejected=db.hand_evidence(current,[hand],bounds,source=fresh)
    assert not associations and not manual
    assert any(row['reason']==reason for row in rejected)


@pytest.mark.parametrize('change,reason',[
    ('pin','CURRENT_REVIEW_SOURCE_CHANGED'),
    ('bounds','CURRENT_REVIEW_HAND_BOUNDS_CHANGED'),
    ('interval','CURRENT_REVIEW_NATIVE_INTERVAL_CHANGED'),
    ('chat_parent','CURRENT_REVIEW_CHAT_PARENT_CONFLICT'),
    ('missing_parent','CURRENT_REVIEW_CHAT_CONTEXT_CONFLICT'),
    ('duplicate_ante','CURRENT_REVIEW_MULTIPLE_ACTORS_SAME_SINGLE_ACTION'),
    ('clear','CURRENT_REVIEW_PHYSICAL_CLEAR_OR_REPLACEMENT'),
    ('other_seat','CURRENT_REVIEW_ACTOR_AT_OTHER_SEAT'),
    ('before_unselected','CURRENT_REVIEW_UNSELECTED_STABLE_NICK'),
    ('after_unselected','CURRENT_REVIEW_UNSELECTED_STABLE_NICK'),
    ('malformed','CURRENT_REVIEW_SOURCE_INVALID'),
])
def test_current_hand_review_rejects_source_and_occupant_conflicts(tmp_path,change,reason):
    extra=[]
    if change=='before_unselected':extra=[(0,'anotherPlayer',.99),(2,'anotherPlayer',.99)]
    if change=='after_unselected':extra=[(28,'anotherPlayer',.99),(29,'anotherPlayer',.99)]
    source,lineage,hand,bounds,db,case=fixture(tmp_path,extra)
    if change=='interval':case['observations'][0]['source']['native_interval']=[0,None]
    if change=='malformed':case['observations'][0]['source'].pop('line')
    db=confirm(db,case)
    if change=='pin':
        with (source/'physical_rows.jsonl').open('a',encoding='utf-8') as stream:stream.write('\n')
    elif change=='bounds':bounds[(1,'s','h')]=(0,31)
    elif change=='chat_parent':hand['messages'][0]['source_rows'][0]['raw_ocr_text']='Dealer: other posts Ante 700'
    elif change=='missing_parent':hand['messages'].append(dict(player='retrel',action='CALL',source_rows=[]))
    elif change=='duplicate_ante':hand['messages'][1]['action']='ANTE'
    elif change=='clear':lineage.boundaries[(1,'s','seat1')].append(dict(frame=17,kind='EMPTY'))
    elif change=='other_seat':
        row=dict(lineage.observations[0],seat='seat2')
        lineage.segments[(1,'s','seat2')]=[dict(start=0,end=None,label=dict(kind='EXACT',text='retre1'),observations=[row],ignored=[])]
    associations,manual,rejected=db.hand_evidence(lineage,[hand],bounds,source=source)
    assert not associations and not manual
    assert any(r['reason']==reason for r in rejected)


def test_conflicting_registered_alias_does_not_gain_current_hand_authority(tmp_path):
    source,lineage,hand,bounds,db,case=fixture(tmp_path)
    other=copy.deepcopy(case);other['case_id']='different-review';other['review_scope']='INTERVAL'
    other['review_scope_key']=None;other['context']['hand_id']='other'
    db.confirm(other,'differentPlayer',[o['id'] for o in other['observations']])
    db=confirm(PlayerRegistry(db.path),case)
    associations,manual,rejected=db.hand_evidence(lineage,[hand],bounds,source=source)
    assert not associations and not manual
    assert any(r['reason']=='CURRENT_REVIEW_CONFLICTING_REGISTERED_PLAYER' for r in rejected)


def test_scope_correction_supersedes_current_hand_authority(tmp_path):
    source,lineage,hand,bounds,db,case=fixture(tmp_path);db=confirm(db,case)
    corrected=copy.deepcopy(case);corrected['case_id']='changed-display-case'
    db.confirm(corrected,'correctedPlayer',[o['id'] for o in case['observations'] if o['kind']=='NICK'])
    db=PlayerRegistry(db.path)
    assert len(db.reviews)==2 and len(db.active_reviews)==1
    associations,manual,rejected=db.hand_evidence(lineage,[hand],bounds,source=source)
    assert not associations and not manual
    assert any(r['reason']=='CURRENT_REVIEW_NICK_OR_CHAT_MISSING' for r in rejected)


def test_typed_exact_canonical_sources_have_separate_provenance(tmp_path):
    source,lineage,hand,bounds,db,case=fixture(tmp_path)
    explicit=[o['id'] for o in case['observations'] if o['text']=='retre1']
    db=confirm(db,case,explicit)
    record=next(iter(db.active_reviews.values()))
    assert record['selected_explicit']==explicit
    assert record['canonical_observations']==[o['id'] for o in case['observations'] if o['text']=='retrel']
    associations,manual,rejected=db.hand_evidence(lineage,[hand],bounds,source=source)
    assert len(associations)==len(manual)==1 and not rejected


def test_canonical_source_adaptation_keeps_original_human_record(tmp_path):
    source,lineage,hand,bounds,db,case=fixture(tmp_path)
    legacy_case=copy.deepcopy(case)
    legacy_case['observations']=[o for o in case['observations'] if o['text']=='retre1']
    explicit=[o['id'] for o in legacy_case['observations']]
    result=db.confirm(legacy_case,'retrel',explicit)
    db=PlayerRegistry(db.path);original=copy.deepcopy(db.reviews[result['review_id']])
    db.confirm(case,original['canonical'],explicit,source_review_id=result['review_id'])
    db=PlayerRegistry(db.path)
    assert db.reviews[result['review_id']]==original
    active=next(iter(db.active_reviews.values()))
    assert active['canonical_sources_derived_from_review']==result['review_id']
    assert active['human_confirmed_at']==original['confirmed_at']
    assert active['selected_explicit']==explicit and len(active['observations'])==4
    with pytest.raises(ValueError,match='original human decision'):
        db.confirm(case,'differentPlayer',explicit,source_review_id=result['review_id'])


def test_display_only_canonical_does_not_implicitly_select_raw_chat(tmp_path):
    source,lineage,hand,bounds,db,case=fixture(tmp_path)
    actor=next(o for o in case['observations'] if o['kind']=='CHAT' and o['text']=='retrel')
    actor['text']='Dealer; retrel';actor['display_text']='retrel'
    db=confirm(db,case,[o['id'] for o in case['observations'] if o['kind']=='NICK'])
    record=next(iter(db.active_reviews.values()))
    assert all(o['kind']=='NICK' for o in record['observations'])
    assert actor['id'] not in record['canonical_observations']


def test_unreviewed_native_blocks_binding_but_not_spelling_queue(tmp_path):
    source,lineage,hand,bounds,db,case=fixture(tmp_path,[(28,'unresolved',.99),(29,'unresolved',.99)])
    provider=SimpleNamespace(hands={(1,'s','h'):hand},bounds=bounds,lineage=lineage,
        actor_sources=LineageActorProvider(lineage,bounds,[hand]),player_registry=db)
    cases=build_cases(provider,source)
    assert len(cases)==1 and cases[0]['binding_blockers'][0]['raw_text']=='unresolved'
    assert {o['text'] for o in cases[0]['observations']}=={'retre1','retrel'}
    db=confirm(db,cases[0])
    associations,manual,rejected=db.hand_evidence(lineage,[hand],bounds,source=source)
    assert not associations and not manual
    assert any(r['reason']=='CURRENT_REVIEW_UNSELECTED_STABLE_NICK' for r in rejected)


def test_uncertain_unmatched_ocr_is_not_a_second_player_after_human_spelling_review(tmp_path):
    source,lineage,hand,bounds,db,case=fixture(tmp_path,[(28,'unreadable',.52),(29,'unreadable',.53)])
    raw=copy.deepcopy(lineage.observations);db=confirm(db,case)
    associations,manual,rejected=db.hand_evidence(lineage,[hand],bounds,source=source)
    assert len(associations)==1 and not rejected
    assert len(associations[0]['registry_evidence']['unmatched_uncertain_nick_candidates'])==2
    assert lineage.observations==raw
    actors=LineageActorProvider(lineage,bounds,[hand],associations,manual)
    assert actors.actor_map(hand)=={'retre1':'retrel','retrel':'retrel'}


def test_selected_nick_cannot_admit_an_unselected_chat_actor_posting_another_ante(tmp_path):
    source,lineage,hand,bounds,db,case=fixture(tmp_path)
    selected=[o['id'] for o in case['observations']
              if o['kind']=='NICK' or o['kind']=='CHAT' and o['text']=='retrel']
    db=confirm(db,case,selected)
    hand['messages'][1]['action']='ANTE'
    associations,manual,rejected=db.hand_evidence(lineage,[hand],bounds,source=source)
    assert not associations and not manual
    assert any(r['reason']=='CURRENT_REVIEW_OTHER_OBSERVED_PLAYER' for r in rejected)


def test_known_different_player_blocks_even_with_low_ocr_confidence(tmp_path):
    source,lineage,hand,bounds,db,case=fixture(tmp_path,[(28,'anotherPlayer',.52),(29,'anotherPlayer',.53)])
    unrelated=copy.deepcopy(case);unrelated['case_id']='another-user-review';unrelated['review_scope']='INTERVAL'
    unrelated['review_scope_key']=None;unrelated['context']['hand_id']='previous'
    unrelated['observations']=[dict(id='other',kind='NICK',text='anotherPlayer',source={})]
    db.confirm(unrelated,'anotherPlayer',['other'])
    db=confirm(PlayerRegistry(db.path),case)
    associations,manual,rejected=db.hand_evidence(lineage,[hand],bounds,source=source)
    assert not associations and not manual
    assert any(r['reason']=='CURRENT_REVIEW_OTHER_REGISTERED_PLAYER' for r in rejected)


def test_reviewed_source_actor_reaches_economic_parser_without_virtual_duplicate(tmp_path):
    from parser_core.poker_reconstruction.timeline import parse_rows
    source,lineage,hand,bounds,db,case=fixture(tmp_path,chat_actor='Dealer; retre1')
    raw=copy.deepcopy(hand);db=confirm(db,case)
    associations,manual,rejected=db.hand_evidence(lineage,[hand],bounds,source=source)
    assert len(associations)==1 and not rejected
    actors=LineageActorProvider(lineage,bounds,[hand],associations,manual)
    mapping=actors.actor_map(hand)
    assert mapping=={'Dealer; retre1':'retrel','retre1':'retrel','retrel':'retrel'}
    events=parse_rows([r for m in hand['messages'] for r in m['source_rows']])
    assert [mapping.get(e.player,e.player) for e in events]==['retrel','retrel']
    assert hand==raw


@pytest.mark.parametrize('actorless',[False,True])
def test_economic_actor_projection_cannot_swallow_another_observed_player(actorless):
    hand=dict(messages=[
        dict(player='Dealer; first',source_rows=[dict(raw_ocr_text='Dealer; first posts Ante 700')]),
        dict(player='first',source_rows=[dict(raw_ocr_text='Dealer: first folds')])])
    if actorless:hand['messages'][1].pop('player')
    mapping={'Dealer; first':'confirmed'}
    assert project_economic_actor_aliases(hand,mapping)==mapping
    hand['messages'].pop()
    assert project_economic_actor_aliases(hand,mapping)==dict(mapping,first='confirmed')


def test_saved_hand_bounds_uses_next_hand_even_when_review_filters_one_hand():
    def record(number,first,last):
        return dict(observed_hand=dict(table_id='T01',session_id='s',hand_id=number),
                    lifecycle=dict(first_source_frame=first,last_source_frame=last))
    records=[record('one',10,17),record('two',25,45)]
    full=saved_hand_bounds(records)
    assert full[(1,'s','one')]==(10,24)
    assert saved_hand_bounds(records[:1])[(1,'s','one')]==(10,17)


def test_reopened_review_rechecks_original_case_and_current_sources(tmp_path):
    source,_,_,_,_,case=fixture(tmp_path)
    path=tmp_path/'cases.json'
    manifest=dict(schema='poker.nickname-review.v1',source=str(source),cases=[case])
    path.write_text(json.dumps(manifest),encoding='utf-8')
    assert load_cached_cases(path,source)==manifest
    with pytest.raises(ValueError,match='source/schema'):
        load_cached_cases(path,tmp_path/'other')
    manifest['cases'][0]['context']['seat']='seat2'
    path.write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(ValueError,match='modified'):
        load_cached_cases(path,source)


def test_reopened_review_rejects_changed_native_source(tmp_path):
    source,_,_,_,_,case=fixture(tmp_path)
    path=tmp_path/'cases.json'
    path.write_text(json.dumps(dict(schema='poker.nickname-review.v1',source=str(source),cases=[case])),encoding='utf-8')
    with (source/'seat_forced_inputs.jsonl').open('a',encoding='utf-8') as stream:stream.write('\n')
    with pytest.raises(ValueError,match='изменились'):
        load_cached_cases(path,source)
