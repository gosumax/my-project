"""Strict physical snapshots and source-bound enriched replay using real fixtures."""
import json
import threading
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from parser_core.dealing_start_stack import CleanStackWindow
from parser_core.seat_forced_binding import SeatForcedBinding,parse_forced_snapshot
from parser_core.poker_math_engine import write_start_stack_math, calculate_start_stack

ROOT=Path(__file__).resolve().parents[1]


def snapshot(hand='12345678',wrap=False):
    lines=[f'Hand #{hand}']+[f'Dealer: p{i} posts Ante 100' for i in range(8)]
    lines+=['Dealer: p0 posts Small Blind','500','Dealer: p1 posts Big Blind','1 000'] if wrap else ['Dealer: p0 posts Small Blind 500','Dealer: p1 posts Big Blind 1 000']
    lines+=['Dealer: Dealing cards']
    return [dict(index=i,text=s,score=.99,source_ref=f'pixel{i}',source_frame=20) for i,s in enumerate(lines)]


@pytest.mark.parametrize('wrap',[False,True])
def test_complete_physical_snapshot_exact_roles_and_adjacent_amounts(wrap):
    result,block=parse_forced_snapshot(snapshot(wrap=wrap),'12345678',8)
    assert block is None and len(result['posts'])==10
    assert next(p for p in result['posts'] if p['role']=='BB')['amount']==1000
    assert len(next(p for p in result['posts'] if p['role']=='SB')['source_refs'])==(2 if wrap else 1)


def test_duplicate_money_row_is_not_counted_twice():
    rr=snapshot();rr.insert(3,dict(rr[2],index=2.5,source_ref='newphysicalduplicate'))
    assert parse_forced_snapshot(rr,'12345678',8)[1]=='FORCED_DUPLICATE_ROW_RISK'


def test_conflicting_duplicate_amount_is_unknown():
    rr=snapshot();rr.insert(3,dict(rr[2],index=2.5,text='Dealer: p1 posts Ante 200'))
    assert parse_forced_snapshot(rr,'12345678',8)[0] is None


@pytest.mark.parametrize('missing',[1,9,10])
def test_missing_ante_or_blind_role_cannot_prove_forced_snapshot(missing):
    rr=snapshot();rr.pop(missing)
    assert parse_forced_snapshot(rr,'12345678',8)[1]=='FORCED_SNAPSHOT_COVERAGE_UNKNOWN'


def test_unadjacent_amount_wrap_is_unknown():
    rr=snapshot(wrap=True);rr[10]['index']=10.5
    assert parse_forced_snapshot(rr,'12345678',8)[1]=='FORCED_AMOUNT_WRAP_UNKNOWN'


@pytest.mark.parametrize('index',[1,9,10])
def test_low_confidence_post_or_wrapped_amount_is_unknown(index):
    rr=snapshot(wrap=True);rr[index]['score']=.949999
    assert parse_forced_snapshot(rr,'12345678',8)[0] is None


def test_mixed_source_frames_are_not_one_snapshot():
    rr=snapshot();rr[2]['source_frame']=21
    assert parse_forced_snapshot(rr,'12345678',8)[1]=='FORCED_SOURCE_FRAME_CONFLICT'


def test_other_header_inside_forced_window_conflicts():
    rr=snapshot();rr.insert(5,dict(rr[0],index=4.5,text='Hand #12345679'))
    assert parse_forced_snapshot(rr,'12345678',8)[1]=='PHYSICAL_HAND_HEADER_CONFLICT'


def test_same_actor_cannot_post_both_blinds():
    rr=snapshot();rr[10]['text']='Dealer: p0 posts Big Blind 1 000'
    assert parse_forced_snapshot(rr,'12345678',8)[1]=='FORCED_ROLE_CONFLICT'


def test_second_deal_row_is_ambiguous():
    rr=snapshot();rr.append(dict(rr[-1],index=20))
    assert parse_forced_snapshot(rr,'12345678',8)[1]=='PHYSICAL_DEAL_ROW_AMBIGUOUS'


def test_snapshot_money_row_after_deal_is_preserved_as_timing_fence():
    rr=snapshot();rr.append(dict(index=12,text='Dealer: p0 calls 1 000',score=.99,source_ref='callpixel',source_frame=20))
    result,block=parse_forced_snapshot(rr,'12345678',8)
    assert block is None and result['money_rows_after_deal']


def test_real_smoke19_all_in_ante_continuation_is_preserved_without_uniform_ante():
    fixture=json.loads((ROOT/'tests/fixtures/smoke19_forced_continuations.json').read_text('utf-8'))
    group=next(g for g in fixture['groups'] if g['table']==8)
    # Six is an explicitly supplied hypothetical occupancy prerequisite here;
    # the CHAT actor count does not authorize occupancy in production.
    result,block=parse_forced_snapshot(group['rows'],group['hand_id'],6)
    assert block is None and result['standard_ante'] is None and not result['standard_ante_proven']
    post=next(p for p in result['posts'] if p['actor']=='rinatromanov19')
    assert post['amount']==137 and post['all_in'] is True and len(post['source_refs'])==2
    assert [r['saved_input_line'] for r in post['source_rows']]==[569,570]
    assert parse_forced_snapshot(group['rows'],group['hand_id'],8)[1]=='FORCED_SNAPSHOT_COVERAGE_UNKNOWN'


def test_real_smoke19_big_blind_wrap_does_not_bypass_other_low_confidence_ante():
    fixture=json.loads((ROOT/'tests/fixtures/smoke19_forced_continuations.json').read_text('utf-8'))
    group=next(g for g in fixture['groups'] if g['table']==7)
    assert parse_forced_snapshot(group['rows'],group['hand_id'],7)[1]=='FORCED_ROW_CONFIDENCE_UNKNOWN'


def big_wrap_snapshot():
    rr=snapshot()
    rr[10]['text']='Dealer: p1 posts Big'
    rr.insert(11,dict(rr[10],index=11,text='Blind 1 000',source_ref='big-tail'))
    rr[-1]['index']=12
    return rr


def test_big_blind_exact_adjacent_wrap_preserves_two_sources():
    rr=big_wrap_snapshot()
    result,block=parse_forced_snapshot(rr,'12345678',8)
    assert block is None
    post=next(p for p in result['posts'] if p['role']=='BB')
    assert post['amount']==1000 and post['source_rows']==rr[10:12]


@pytest.mark.parametrize('change',['confidence','frame','index','actor','session','hand','ambiguous'])
def test_big_blind_wrap_rejects_unproven_join(change):
    rr=big_wrap_snapshot()
    if change=='confidence':rr[11]['score']=.9448
    if change=='frame':rr[11]['source_frame']=21
    if change=='index':rr[11]['index']=11.5
    if change=='actor':rr[11]['text']='Dealer: other Blind 1 000'
    if change=='session':rr[11]['session_id']='OTHER'
    if change=='hand':rr[11]['hand_id']='OTHER'
    if change=='ambiguous':rr[11]['text']='Blind 1 000 or 2 000'
    assert parse_forced_snapshot(rr,'12345678',8)[0] is None


@pytest.mark.parametrize('change',['confidence','frame','index','session','hand','actor','amount'])
def test_all_in_continuation_rejects_unproven_join(change):
    rr=snapshot()
    rr.insert(2,dict(rr[1],index=2,text='and is All-in',source_ref='all-in-tail'))
    for r in rr[3:]:r['index']+=1
    if change=='confidence':rr[2]['score']=.9499
    if change=='frame':rr[2]['source_frame']=21
    if change=='index':rr[2]['index']=2.5
    if change=='session':rr[2]['session_id']='OTHER'
    if change=='hand':rr[2]['hand_id']='OTHER'
    if change=='actor':rr[2]['text']='Dealer: other is All-in'
    if change=='amount':rr[1]['text']='Dealer: p0 posts Ante'
    assert parse_forced_snapshot(rr,'12345678',8)[0] is None


class FakeOCR:
    def __init__(self,*a,**kw):self.errors=[];self.batches=[]
    def submit_batch(self,b):self.batches.append(b)
    def finish(self):pass


@pytest.fixture
def binder(tmp_path,monkeypatch):
    monkeypatch.setattr('parser_core.seat_forced_binding.ParallelRowOCR',FakeOCR)
    window=CleanStackWindow(.95);key=(3,'S','12345678')
    window.confirm_deal(dict(evidence_status='PROVEN',table_pixel_sha256='table',occupied_seats=['seat1'],source_frame=20))
    log=(tmp_path/'start_stack_readings.jsonl').open('w',encoding='utf-8')
    sidecar=SimpleNamespace(lock=threading.RLock(),flags={'capture_frames':64,'stack_confidence':.95},windows={key:window},
        triggers=[dict(table_id=3,session_id='S',hand_id=key[2],source_frame=10)],logs={'start_stack_readings':log},
        histories=defaultdict(list),active_sessions={3:'S'},current={3:dict(session_id='S',hand_id=key[2])})
    def read(k,r):
        value,status=sidecar.windows[k].reading(r['seat'],r['source_frame'],r['raw_text'],r['confidence'],r['forced_bets'],r['occupancy_epoch'],r['forced_bets_proven'],r['occupancy_proven'])
        log.write(json.dumps(dict(r,status=status,result=value))+'\n')
        return value,status
    sidecar.stack_reading=read
    stack=SimpleNamespace(states={key:{'numeric_triples':{}}})
    obj=SeatForcedBinding(ROOT,tmp_path,sidecar,stack,{})
    obj.machine.hand(*key,10)
    obj.machine.observe(3,'S','seat1','OCCUPIED',20,hand_id=key[2])
    obj.machine.build()
    # Synthetic POST fixtures below represent physical pre-deal debits.
    original_finish=obj.finish
    def finish_fixture():
        for snap in obj.forced.values():
            for post in snap['posts']:
                post.setdefault('phase','BEFORE_DEAL');post.setdefault('source_frame',20)
        return original_finish()
    obj.finish=finish_fixture
    obj.test_key=key
    yield obj
    if not obj.closed:obj.finish()
    log.close()


def test_unique_actor_mapping_requires_nickname_at_or_before_stack_frame(binder):
    key=binder.test_key;binder.nick_labels[(*key,'seat1')].append(dict(actor='p',frames=[21,22,23]))
    assert binder.seats_for_actor(key,'p',22)==[]
    assert binder.seats_for_actor(key,'p',23)==['seat1']


def test_confirmed_marker_preserves_candidate_chat_before_deal_without_admission(binder,monkeypatch):
    key=binder.test_key;binder.sidecar.windows[key].deal=None
    pixels=np.arange(96,dtype=np.uint8).reshape(4,24)
    monkeypatch.setattr('parser_core.seat_forced_binding.physical_spans',lambda chat:[(0,4)])
    monkeypatch.setattr('parser_core.seat_forced_binding.padded_row_span',lambda *a:(0,4))
    binder.observe_frame(3,'S',11,11,[1,30],np.zeros((10,24,3),dtype=np.uint8),pixels)
    batch=binder.ocr.batches[0];assert len(batch)==1
    assert np.array_equal(batch[0].pixels,pixels)
    binder.completed(batch[0],dict(status='OK',raw_text='Dealer: p posts Ante 100',score=.99))
    binder.log.flush()
    receipt=json.loads((binder.output/'seat_forced_inputs.jsonl').read_text('utf-8').splitlines()[0])
    assert receipt['kind']=='CHAT_CANDIDATE' and receipt['authority']=='CANDIDATE_ONLY'
    assert not binder.forced and not binder.snapshots and not binder.nick_labels
    binder.observe_frame(3,'S',12,12,[1,30],np.zeros((10,24,3),dtype=np.uint8),pixels)
    assert len(binder.ocr.batches)==1
    binder.observe_frame(3,'wrong',13,13,[1,30],np.zeros((10,24,3),dtype=np.uint8),pixels)
    assert len(binder.ocr.batches)==1


def test_numeric_nickname_from_fixed_nick_roi_keeps_exact_three_frame_label(binder):
    key=binder.test_key
    source=snapshot(hand=key[2])
    for row in source:row['text']=row['text'].replace('p0','7697')
    binder.forced[key]=parse_forced_snapshot(source,key[2],8)[0]
    for frame in [21,22,23]:
        pixels=np.full((2,10),frame,np.uint8)
        row=binder.row(3,'S',frame,frame,[1,30],106,0,pixels,'NICK_ONLY')
        from parser_core.parallel_ocr import row_key
        binder.jobs[row_key(row)]=(key,dict(table_id=3,session_id='S',hand_id=key[2],source_frame=frame,
                                          kind='NICK',seat='seat1',source_ref=row.pixel_sha256))
        binder.completed(row,dict(status='OK',raw_text='7697',score=.99998))
    labels=binder.nick_labels[(*key,'seat1')]
    assert labels[0]['actor']=='7697' and labels[0]['frames']==[21,22,23]
    assert binder.seats_for_actor(key,'7697',23)==['seat1']


@pytest.mark.parametrize('snapshot_proof',[None,dict(actors=['7697'],forced_snapshot_complete=False),
                                         dict(actors=['other'],forced_snapshot_complete=True)])
def test_numeric_nickname_without_exact_complete_snapshot_cannot_be_admitted(binder,snapshot_proof):
    key=binder.test_key
    if snapshot_proof:binder.forced[key]=dict(snapshot_proof,posts=[])
    for frame in [21,22,23]:
        row=binder.row(3,'S',frame,frame,[1,30],106,0,np.ones((2,10),np.uint8),'NICK_ONLY')
        from parser_core.parallel_ocr import row_key
        binder.jobs[row_key(row)]=(key,dict(kind='NICK',seat='seat1',source_frame=frame))
        binder.completed(row,dict(status='OK',raw_text='7697',score=.99998))
    assert not binder.nick_labels[(*key,'seat1')]


def test_same_actor_two_seats_does_not_have_unique_mapping(binder):
    key=binder.test_key;binder.sidecar.windows[key].deal['occupied_seats'].append('seat2')
    for s in ('seat1','seat2'):binder.nick_labels[(*key,s)].append(dict(actor='p',frames=[21,22,23]))
    binder.machine.observe(3,'S','seat2','OCCUPIED',20,hand_id=key[2]);binder.machine.build()
    assert len(binder.seats_for_actor(key,'p',23))==2


def test_two_forced_actors_mapped_to_one_seat_cannot_bind_any_posts(binder):
    key=binder.test_key
    binder.machine.observe(3,'S','seat1','OCCUPIED',20,hand_id=key[2])
    for actor in ('A','B'):binder.nick_labels[(*key,'seat1')].append(dict(actor=actor,frames=[21,22,23]))
    binder.forced[key]=dict(actors=['A','B'],posts=[dict(actor=a,role='ANTE',amount=100,source_refs=[a]) for a in ('A','B')])
    binder.finish()
    result=combined_math_summary(binder.output)
    assert result['bound_posts']==[] and result['forced_bets_bound']==0


def test_overlays_and_low_confidence_nickname_are_not_actor_labels(binder):
    key=binder.test_key
    for f in (21,22,23):
        row=binder.row(3,'S',f,f*512,[1,15360],100,0,np.full((2,2),80,np.uint8),'NICK_ONLY')
        from parser_core.parallel_ocr import row_key
        binder.jobs[row_key(row)]=(key,dict(kind='NICK',seat='seat1',source_frame=f))
        binder.completed(row,dict(status='OK',raw_text='ANTE',score=.99))
    assert not binder.nick_labels[(*key,'seat1')]


def install_raw_triplet(binder,key,seat,frames,text='1000'):
    binder.stack_live.states[key]['numeric_triples'][seat]=dict(displayed_stack=text,frames=frames)
    for f in frames:
        binder.sidecar.logs['start_stack_readings'].write(json.dumps(dict(table_id=key[0],session_id=key[1],hand_id=key[2],seat=seat,source_frame=f,raw_text=text,confidence=.99,forced_bets=None,forced_bets_proven=False,occupancy_epoch=None,occupancy_proven=False,status='IDENTITY_UNKNOWN',result=None))+'\n')


def test_late_nickname_source_does_not_retroactively_bind_old_stack(binder):
    key=binder.test_key;install_raw_triplet(binder,key,'seat1',[21,22,23])
    binder.machine.observe(3,'S','seat1','OCCUPIED',20,hand_id=key[2])
    binder.forced[key]=dict(actors=['p'],posts=[dict(actor='p',role='ANTE',amount=100,source_refs=['ante'])])
    binder.nick_labels[(*key,'seat1')].append(dict(actor='p',frames=[24,25,26]))
    binder.finish()
    result=combined_math_summary(binder.output)
    assert result['start_stack_pass']==0
    assert result['results'][0]['blocker']=='FORCED_ACTOR_TO_SEAT_UNKNOWN'


def test_sitout_forced_bet_binding_does_not_require_visible_numeric_stack(binder):
    key=binder.test_key
    # Occupancy must cover all three nickname sources, not only confirmation.
    binder.machine.observe(3,'S','seat1','SIT_OUT',7,actor='p')
    binder.machine.observe(3,'S','seat1','OCCUPIED',20,hand_id=key[2])
    binder.nick_labels[(*key,'seat1')].append(dict(actor='p',frames=[7,8,9]))
    binder.forced[key]=dict(actors=['p'],posts=[dict(actor='p',role='ANTE',amount=100,source_refs=['ante'])])
    binder.finish()
    result=combined_math_summary(binder.output)
    assert result['states']=={'SITOUT':1}
    assert result['start_stack_pass']==0
    assert any(r['seat']=='seat1' and r['role']=='ANTE' and r['amount']==100 for r in result['bound_posts'])


@pytest.mark.parametrize('later_observation',[False,True])
def test_absent_or_late_money_observation_never_supplies_clean_window_proof(binder,later_observation):
    key=binder.test_key;install_raw_triplet(binder,key,'seat1',[21,22,23])
    binder.machine.observe(3,'S','seat1','OCCUPIED',20,hand_id=key[2])
    binder.forced[key]=dict(actors=['p'],posts=[dict(actor='p',role='ANTE',amount=100,source_refs=['ante'])])
    binder.nick_labels[(*key,'seat1')].append(dict(actor='p',frames=[21,22,23]))
    if later_observation:binder.sidecar.windows[key].action('CALL',100)
    binder.finish()
    result=combined_math_summary(binder.output)
    assert result['results'][0]['forced_bets']=={'ANTE':100}
    assert result['results'][0]['blocker']=='FIRST_VISIBLE_STACK_TRIPLE_UNPROVEN'
    assert result['start_stack_pass']==0 and not binder.sidecar.windows[key].results


def test_visible_money_in_snapshot_blocks_certification_of_arithmetic_candidate(binder):
    key=binder.test_key;install_raw_triplet(binder,key,'seat1',[21,22,23])
    binder.machine.observe(3,'S','seat1','OCCUPIED',20,hand_id=key[2])
    binder.forced[key]=dict(actors=['p'],posts=[dict(actor='p',role='ANTE',amount=100,source_refs=['ante'])],money_rows_after_deal=[dict(text='Dealer: p calls 500',source_ref='callpixel')])
    binder.nick_labels[(*key,'seat1')].append(dict(actor='p',frames=[21,22,23]))
    binder.finish()
    result=combined_math_summary(binder.output)
    assert result['results'][0]['blocker']=='FIRST_VISIBLE_STACK_TRIPLE_UNPROVEN'
    assert result['start_stack_pass']==0 and not binder.sidecar.windows[key].results


def test_old_actor_departure_after_empty_and_new_occupant_cannot_empty_new_epoch(binder):
    old=binder.test_key;new=(3,'S','12345679')
    w=CleanStackWindow(.95);w.confirm_deal(dict(evidence_status='PROVEN',table_pixel_sha256='newtable',occupied_seats=['seat1'],source_frame=30))
    binder.sidecar.windows[new]=w;binder.stack_live.states[new]={'numeric_triples':{}}
    binder.sidecar.triggers.append(dict(table_id=3,session_id='S',hand_id=new[2],source_frame=28))
    binder.machine.observe(3,'S','seat1','OCCUPIED',20,actor='old',hand_id=old[2])
    binder.machine.observe(3,'S','seat1','EMPTY',25)
    binder.machine.observe(3,'S','seat1','OCCUPIED',30,actor='new',hand_id=new[2])
    binder.nick_labels[(*old,'seat1')].append(dict(actor='old',frames=[21,22,23]))
    binder.nick_labels[(*new,'seat1')].append(dict(actor='new',frames=[30,31,32]))
    binder.state_events.append(dict(table=3,session='S',hand_id=new[2],frame=33,source_index=1,actor='old',kind='has left the table',source_ref='lateoldleave'))
    binder.finish()
    result=combined_math_summary(binder.output)
    assert result['unresolved_state_events'][0]['actor']=='old'
    assert binder.machine.epoch_at(3,'S','seat1',34).endswith('_2')
    assert binder.machine.rows[3,'S','seat1']['state']!='EMPTY'


@pytest.mark.parametrize('bets,expected',[({'ANTE':100},'1100'),({'ANTE':100,'SB':500},'1600'),({'ANTE':100,'BB':1000},'2100')])
def test_existing_window_forced_only_regular_sb_bb_math(bets,expected):
    w=CleanStackWindow(.95);w.confirm_deal(dict(evidence_status='PROVEN',table_pixel_sha256='p',occupied_seats=['seat1'],source_frame=20))
    for f in (21,22,23):v,status=w.reading('seat1',f,'1000',.99,bets,'E1',True,True)
    assert status=='START_STACK_RECOVERED' and v['start_stack']==expected


def test_real_probe_replay_has_seven_arithmetic_candidates_and_zero_live_pass(tmp_path,monkeypatch):
    monkeypatch.setattr('parser_core.seat_forced_binding.ParallelRowOCR',FakeOCR)
    probe=ROOT/'runs/seat_forced_binding_20261003/input_probe/raw_probe_jobs.jsonl'
    analysis=ROOT/'runs/start_stack_live_path_20261003/connection_acceptance_analysis.json'
    if not probe.exists() or not analysis.exists():pytest.skip('bounded real probe fixtures unavailable')
    jobs=[json.loads(l) for l in probe.read_text('utf-8').splitlines()]
    seat_rows=json.loads(analysis.read_text('utf-8'))['seat_rows']
    windows={};states={};triggers=[]
    for hand,frame in [('2876719109',884),('2876719777',2530)]:
        key=(3,'S',hand);w=CleanStackWindow(.95)
        w.confirm_deal(dict(evidence_status='PROVEN',table_pixel_sha256='p',occupied_seats=[f'seat{i}' for i in range(1,9)],source_frame=frame))
        windows[key]=w;states[key]={'numeric_triples':{}};triggers.append(dict(table_id=3,session_id='S',hand_id=hand,source_frame=frame-13))
    log=(tmp_path/'start_stack_readings.jsonl').open('w',encoding='utf-8')
    sidecar=SimpleNamespace(lock=threading.RLock(),flags={'capture_frames':64,'stack_confidence':.95},windows=windows,triggers=triggers,logs={'start_stack_readings':log})
    def read(key,r):
        v,s=windows[key].reading(r['seat'],r['source_frame'],r['raw_text'],r['confidence'],r['forced_bets'],r['occupancy_epoch'],r['forced_bets_proven'],r['occupancy_proven'])
        log.write(json.dumps(dict(r,status=s,result=v))+'\n');return v,s
    sidecar.stack_reading=read;b=SeatForcedBinding(ROOT,tmp_path,sidecar,SimpleNamespace(states=states),{})
    for key,w in windows.items():
        b.machine.hand(*key,w.deal['source_frame']-13)
        for seat in w.deal['occupied_seats']:b.machine.observe(key[0],key[1],seat,'OCCUPIED',w.deal['source_frame'],hand_id=key[2])
        chat=[dict(index=r['row_index'],text=r['response']['raw_text'],score=r['response']['score'],source_ref=r['pixel_sha256'],source_frame=r['frame']) for r in jobs if r['field']=='CHAT' and r['hand_id']==key[2]]
        snap,block=parse_forced_snapshot(chat,key[2],8);assert block is None;b.forced[key]=snap
        for post in snap['posts']:post['source_frame']=chat[0]['source_frame']
        for seat in w.deal['occupied_seats']:
            rr=[r for r in jobs if r['field']=='NICK' and r['hand_id']==key[2] and r['seat']==seat]
            texts=[r['response']['raw_text'] for r in rr]
            if len(set(texts))==1 and all(r['response']['score']>=.95 for r in rr):b.nick_labels[(*key,seat)].append(dict(actor=texts[0],frames=[r['frame'] for r in rr]))
        for r in seat_rows:
            if r['table']==3 and r['hand_id']==key[2] and r['stable_numeric_stack_frames']:
                install_raw_triplet(b,key,r['seat'],r['stable_numeric_stack_frames'],r['displayed_stack'])
    b.finish();log.close();result=combined_math_summary(tmp_path)
    assert result['numeric_triples']==13 and result['arithmetic_candidates']==7 and result['start_stack_pass']==0 and result['start_stack_unknown']==16
    assert result['forced_bets_bound']==13
    for r in result['results']:
        if 'forced_bets' in r:
            assert r['START_STACK'] is None and r['blocker']=='FIRST_VISIBLE_STACK_TRIPLE_UNPROVEN'
    evidence=ROOT/'runs/seat_forced_binding_20261003/tests'
    evidence.mkdir(parents=True,exist_ok=True)
    (evidence/'real_fixture_replay.json').write_text(json.dumps(dict(scope='OFFLINE_REAL_INPUT_REPLAY_NOT_LIVE_PASS',summary=result),indent=2),'utf-8')


def canonical(binder,frames=(21,22,23)):
    key=binder.test_key
    install_raw_triplet(binder,key,'seat1',list(frames))
    r=binder.stack_live.states[key]['numeric_triples']['seat1']
    r.update(clean_window_rule='FIRST_VISIBLE_STABLE_STACK_TRIPLE_AFTER_DEAL_COMPLETE',status='PASS',first_numeric_visible_frame=frames[0])
    return key


@pytest.mark.parametrize('bets,expected',[({'ANTE':100},'1100'),({'ANTE':100,'SB':500},'1600'),({'ANTE':100,'BB':1000},'2100')])
def test_first_canonical_triple_keeps_arithmetic_candidate_without_action_onset(binder,bets,expected):
    key=canonical(binder)
    binder.nick_labels[(*key,'seat1')].append(dict(actor='p',frames=[21,22,23]))
    binder.forced[key]=dict(actors=['p'],posts=[dict(actor='p',role=k,amount=v,source_refs=[k]) for k,v in bets.items()],money_rows_after_deal=[dict(text='Dealer: p calls 500')])
    binder.sidecar.windows[key].action('CALL',21)
    binder.finish()
    result=combined_math_summary(binder.output)
    assert result['start_stack_pass']==0 and result['results'][0]['START_STACK'] is None
    assert result['results'][0]['START_STACK_CANDIDATE']==expected
    assert result['results'][0]['blocker']=='CLEAN_WINDOW_TIMING_UNPROVEN'
    assert not binder.sidecar.windows[key].results
    facts=json.loads((binder.output/'seat_forced_summary.json').read_text('utf-8'))['results'][0]
    assert 'START_STACK' not in facts and 'START_STACK_CANDIDATE' not in facts
    binder.finish()
    assert not binder.sidecar.windows[key].results
    assert combined_math_summary(binder.output)['snapshot_arithmetic_pass']==1


def test_new_epoch_cannot_receive_old_label_or_cross_epoch_triple(binder):
    key=canonical(binder)
    binder.nick_labels[(*key,'seat1')].append(dict(actor='p',frames=[18,19,20]))
    binder.forced[key]=dict(actors=['p'],posts=[dict(actor='p',role='ANTE',amount=100,source_refs=['ante'])])
    binder.machine.observe(3,'S','seat1','EMPTY',21)
    binder.machine.observe(3,'S','seat1','OCCUPIED',22,hand_id=key[2])
    binder.finish()
    r=combined_math_summary(binder.output)['results'][0]
    assert r['status']=='UNKNOWN' and r['blocker']=='STACK_TRIPLE_CROSSES_EPOCH'


def test_two_actors_one_seat_canonical_fails_closed(binder):
    key=canonical(binder)
    binder.forced[key]=dict(actors=['p','q'],posts=[dict(actor=a,role='ANTE',amount=100,source_refs=[a]) for a in ('p','q')])
    for a in ('p','q'):binder.nick_labels[(*key,'seat1')].append(dict(actor=a,frames=[21,22,23]))
    binder.finish()
    r=combined_math_summary(binder.output)
    assert r['start_stack_pass']==0 and r['results'][0]['blocker']=='FORCED_ACTOR_TO_SEAT_UNKNOWN'


def test_one_actor_two_seats_canonical_fails_closed(binder):
    key=canonical(binder)
    binder.sidecar.windows[key].deal['occupied_seats'].append('seat2')
    binder.machine.observe(3,'S','seat2','OCCUPIED',20,hand_id=key[2])
    binder.forced[key]=dict(actors=['p'],posts=[dict(actor='p',role='ANTE',amount=100,source_refs=['ante'])])
    for seat in ('seat1','seat2'):binder.nick_labels[(*key,seat)].append(dict(actor='p',frames=[21,22,23]))
    binder.finish()
    r=combined_math_summary(binder.output)
    assert r['start_stack_pass']==0 and not r['bound_posts']


def test_ambiguous_second_actor_is_not_dropped_from_reverse_seat_check(binder):
    key=canonical(binder)
    binder.sidecar.windows[key].deal['occupied_seats'].append('seat2')
    binder.machine.observe(3,'S','seat2','OCCUPIED',20,hand_id=key[2])
    binder.forced[key]=dict(actors=['A','B'],posts=[dict(actor=a,role='ANTE',amount=100,source_refs=[a]) for a in ('A','B')])
    for actor,seat in [('A','seat1'),('B','seat1'),('B','seat2')]:
        binder.nick_labels[(*key,seat)].append(dict(actor=actor,frames=[21,22,23]))
    binder.finish()
    r=combined_math_summary(binder.output)
    assert r['start_stack_pass']==0 and r['results'][0]['blocker']=='FORCED_ACTOR_TO_SEAT_UNKNOWN'


def test_unrelated_conflicting_label_cannot_prove_forced_binding(binder):
    key=canonical(binder)
    binder.forced[key]=dict(actors=['A'],posts=[dict(actor='A',role='ANTE',amount=100,source_refs=['ante'])])
    for actor in ('A','unrelated'):
        binder.nick_labels[(*key,'seat1')].append(dict(actor=actor,frames=[21,22,23]))
    binder.finish()
    r=combined_math_summary(binder.output)
    assert r['start_stack_pass']==0 and r['bound_posts']==[]


def standard_snapshot(binder):
    key=canonical(binder)
    binder.sidecar.windows[key].deal['occupied_seats']+=['seat2','seat3']
    for seat in ('seat2','seat3'):binder.machine.observe(3,'S',seat,'OCCUPIED',20,hand_id=key[2])
    binder.forced[key]=dict(actors=['ordinary','small','big'],
        posts=[dict(actor=a,role='ANTE',amount=100,source_refs=[a]) for a in ('ordinary','small','big')]
            +[dict(actor='small',role='SB',amount=500,source_refs=['sb']),dict(actor='big',role='BB',amount=1000,source_refs=['bb'])],
        standard_ante=100,standard_ante_proven=True,forced_snapshot_complete=True)
    binder.nick_labels[(*key,'seat2')].append(dict(actor='small',frames=[21,22,23]))
    binder.nick_labels[(*key,'seat3')].append(dict(actor='big',frames=[21,22,23]))
    return key


def test_standard_ante_ordinary_seat_passes_without_any_nickname(binder):
    key=standard_snapshot(binder)
    binder.finish()
    r=combined_math_summary(binder.output)['results'][0]
    assert r['status']=='UNKNOWN' and r['START_STACK'] is None and r['START_STACK_CANDIDATE']=='1100'
    assert r['forced_bets']=={'ANTE':100} and 'actor' not in r


def test_standard_ante_source_late_ordinary_nick_does_not_block_fixed_roi(binder):
    key=standard_snapshot(binder)
    binder.nick_labels[(*key,'seat1')].append(dict(actor='ordinary',frames=[24,25,26]))
    binder.finish()
    r=combined_math_summary(binder.output)['results'][0]
    assert r['snapshot_arithmetic_status']=='PASS' and r['displayed_stack']=='1000'
    assert r['START_STACK'] is None and r['blocker']=='CLEAN_WINDOW_TIMING_UNPROVEN'


def test_unknown_special_seat_keeps_numeric_and_ante_candidate(binder):
    key=standard_snapshot(binder)
    binder.nick_labels[(*key,'seat3')].clear()
    binder.finish()
    r=combined_math_summary(binder.output)['results'][0]
    assert r['status']=='UNKNOWN' and r['displayed_stack']=='1000'
    assert r['displayed_stack_status']=='VALID_FIXED_SEAT_ROI'
    assert r['standard_ante']==100 and r['blocker']=='SPECIAL_FORCED_SEAT_UNKNOWN'


def test_auto_all_in_without_actual_actor_binding_cannot_use_standard_ante(binder):
    key=standard_snapshot(binder)
    binder.stack_live.states[key]['source_readings']={'seat1':{20:dict(raw_text='AUTO ALL-IN')}}
    binder.finish()
    r=combined_math_summary(binder.output)['results'][0]
    assert r['status']=='UNKNOWN' and r['blocker']=='AUTO_ALL_IN_ACTUAL_FORCED_UNKNOWN'
    assert r['displayed_stack']=='1000'


def test_nonuniform_ante_does_not_prove_standard_ante():
    rr=snapshot();rr[1]['text']='Dealer: p0 posts Ante 50'
    result,block=parse_forced_snapshot(rr,'12345678',8)
    assert block is None and result['standard_ante'] is None and not result['standard_ante_proven']


@pytest.mark.parametrize('seat,expected',[('seat2','1600'),('seat3','2100')])
def test_standard_ante_special_seat_adds_only_bound_blind(binder,seat,expected):
    key=standard_snapshot(binder)
    install_raw_triplet(binder,key,seat,[21,22,23])
    r=binder.stack_live.states[key]['numeric_triples'][seat]
    r.update(clean_window_rule='FIRST_VISIBLE_STABLE_STACK_TRIPLE_AFTER_DEAL_COMPLETE',status='PASS')
    binder.finish()
    results=combined_math_summary(binder.output)['results']
    r=next(r for r in results if r['seat']==seat)
    assert r['snapshot_arithmetic_status']=='PASS' and r['START_STACK_CANDIDATE']==expected
    assert r['START_STACK'] is None and r['blocker']=='CLEAN_WINDOW_TIMING_UNPROVEN'


def test_standard_ante_special_actor_on_two_seats_blocks_adjustment(binder):
    key=standard_snapshot(binder)
    binder.nick_labels[(*key,'seat1')].append(dict(actor='small',frames=[21,22,23]))
    binder.finish()
    r=combined_math_summary(binder.output)['results'][0]
    assert r['status']=='UNKNOWN' and r['blocker']=='SPECIAL_FORCED_SEAT_UNKNOWN'
    assert r['displayed_stack']=='1000'


def test_standard_ante_two_actors_on_blind_seat_block_adjustment(binder):
    key=standard_snapshot(binder)
    binder.nick_labels[(*key,'seat2')].append(dict(actor='other',frames=[21,22,23]))
    binder.finish()
    r=combined_math_summary(binder.output)['results'][0]
    assert r['status']=='UNKNOWN' and r['blocker']=='SPECIAL_FORCED_SEAT_UNKNOWN'


def test_unresolved_ordinary_join_cannot_grant_standard_ante_membership(binder):
    key=standard_snapshot(binder)
    binder.state_events.append(dict(table=3,session='S',hand_id=key[2],frame=21,source_index=1,
        actor='unknown_joiner',kind='has joined the table',source_ref='joinpixel'))
    binder.finish()
    r=combined_math_summary(binder.output)['results'][0]
    assert r['status']=='UNKNOWN' and r['blocker']=='SEAT_EVENT_LINEAGE_OR_BINDING_UNKNOWN'
    assert r['displayed_stack']=='1000' and r['standard_ante']==100


def combined_math_summary(output):
    facts=json.loads((output/'seat_forced_summary.json').read_text('utf-8'))
    math=write_start_stack_math(output)
    return dict(facts,**math)


def test_math_never_mutates_vision_facts(binder):
    key=standard_snapshot(binder)
    binder.finish()
    p=binder.output/'start_stack_facts.jsonl'
    before=p.read_bytes()
    facts=json.loads(before.splitlines()[0])
    r=calculate_start_stack(facts)
    assert r['START_STACK_CANDIDATE']=='1100' and 'START_STACK' not in facts
    assert r['START_STACK'] is None and r['blocker']=='CLEAN_WINDOW_TIMING_UNPROVEN'
    assert p.read_bytes()==before
    assert not binder.sidecar.windows[key].results


def test_unproven_pre_snapshot_debit_blocks_math_only(binder):
    key=standard_snapshot(binder)
    binder.finish()
    r=json.loads((binder.output/'start_stack_facts.jsonl').read_text('utf-8').splitlines()[0])
    r['forced_bets_debited_before_snapshot_proven']=False
    result=calculate_start_stack(r)
    assert result['status']=='UNKNOWN' and result['blocker']=='FORCED_DEBIT_AT_SNAPSHOT_UNPROVEN'
    assert result['displayed_stack']=='1000'


def test_vision_binding_module_contains_no_start_stack_arithmetic():
    import ast
    tree=ast.parse((ROOT/'parser_core/seat_forced_binding.py').read_text('utf-8'))
    assert not any(isinstance(n,ast.BinOp) and isinstance(n.op,ast.Add) and any(
        isinstance(x,ast.Subscript) and isinstance(x.slice,ast.Constant) and x.slice.value=='displayed_stack'
        for x in ast.walk(n)) for n in ast.walk(tree))


def historical_fixture(binder):
    key=binder.test_key
    binder.forced[key]=dict(actors=['bankiir'],posts=[dict(actor='bankiir',role='ANTE',amount=700,
        source_frame=20,phase='BEFORE_DEAL',source_refs=['physical_ante'])])
    binder.nick_labels[(*key,'seat1')].append(dict(actor='bankiir',frames=[21,22,23],source_ref='exact_nick'))
    return key


def test_historical_binding_preserves_late_conflict_without_alias_or_backdating(binder):
    key=historical_fixture(binder)
    binder.nick_labels[(*key,'seat1')].append(dict(actor='bankir',frames=[30,31,32],source_ref='late_literal'))
    binder.finish()
    summary=json.loads((binder.output/'seat_forced_summary.json').read_text('utf-8'))
    assert summary['forced_bets_bound']==1 and summary['bound_posts'][0]['actor']=='bankiir'
    assert summary['bound_posts'][0]['source_frame']==20
    receipt=summary['historical_binding_receipts'][0]
    assert receipt['binding_anchor_frame']==receipt['label_confirmation_frame']==23
    assert receipt['forced_source_frame']==20
    assert receipt['later_literal_conflicts']==[dict(actor='bankir',seat='seat1',frames=[30,31,32],source_ref='late_literal')]
    assert not binder.stack_live.states[key]['numeric_triples']


@pytest.mark.parametrize('case',[
    'reverse_before','reverse_same','actor_other_seat','replacement_between','replacement_inside',
    'unknown_occupancy','unresolved_lineage','bad_first_anchor','alias_only','unknown_source','source_before_deal'])
def test_historical_binding_conflicts_fail_closed_without_later_retry(binder,case):
    key=historical_fixture(binder)
    if case in {'reverse_before','reverse_same'}:
        frames=[18,19,20] if case=='reverse_before' else [21,22,23]
        binder.nick_labels[(*key,'seat1')].append(dict(actor='other',frames=frames))
    elif case=='actor_other_seat':
        binder.sidecar.windows[key].deal['occupied_seats'].append('seat2')
        binder.machine.observe(*key[:2],'seat2','OCCUPIED',20,hand_id=key[2])
        binder.nick_labels[(*key,'seat2')].append(dict(actor='bankiir',frames=[21,22,23]))
    elif case.startswith('replacement_'):
        binder.machine.observe(*key[:2],'seat1','REPLACE',21 if case.endswith('between') else 22,hand_id=key[2])
    elif case=='unknown_occupancy':
        binder.machine.observe(*key[:2],'seat2','EMPTY',21,evidence_status='UNKNOWN')
    elif case=='unresolved_lineage':
        binder.state_events.append(dict(table=3,session='S',hand_id=key[2],frame=21,source_index=0,
            actor='unmapped_actor',kind='has joined the table'))
    elif case=='bad_first_anchor':
        # The earliest literal is outside the proven epoch; later valid evidence cannot replace it.
        binder.nick_labels[(*key,'seat1')].append(dict(actor='bankiir',frames=[14,15,16]))
    elif case=='alias_only':
        binder.nick_labels[(*key,'seat1')]=[dict(actor='bankir',frames=[21,22,23])]
    elif case=='unknown_source':binder.forced[key]['posts'][0]['source_frame']=None
    elif case=='source_before_deal':binder.forced[key]['posts'][0]['source_frame']=19
    binder.finish()
    summary=json.loads((binder.output/'seat_forced_summary.json').read_text('utf-8'))
    assert summary['bound_posts']==[] and summary['historical_binding_receipts']==[]
    assert summary['historical_binding_blockers']


def test_historical_sitout_label_before_post_keeps_real_confirmation_time(binder):
    key=historical_fixture(binder)
    binder.machine.observe(*key[:2],'seat1','SIT_OUT',7,actor='bankiir')
    binder.nick_labels[(*key,'seat1')]=[dict(actor='bankiir',frames=[7,8,9])]
    binder.finish()
    summary=json.loads((binder.output/'seat_forced_summary.json').read_text('utf-8'))
    assert summary['forced_bets_bound']==1
    receipt=summary['historical_binding_receipts'][0]
    assert receipt['label_confirmation_frame']==9 and receipt['binding_anchor_frame']==20


def test_historical_anchor_uses_source_order_of_admitted_labels(binder):
    key=historical_fixture(binder)
    binder.nick_labels[(*key,'seat1')].insert(0,dict(actor='bankiir',frames=[40,41,42]))
    binder.finish()
    summary=json.loads((binder.output/'seat_forced_summary.json').read_text('utf-8'))
    assert summary['forced_bets_bound']==1
    assert summary['historical_binding_receipts'][0]['label_confirmation_frame']==23


def test_historical_late_forward_conflict_retained_without_new_binding(binder):
    key=historical_fixture(binder)
    binder.sidecar.windows[key].deal['occupied_seats'].append('seat2')
    binder.machine.observe(*key[:2],'seat2','OCCUPIED',20,hand_id=key[2])
    binder.nick_labels[(*key,'seat2')].append(dict(actor='bankiir',frames=[30,31,32],source_ref='late_other_seat'))
    binder.finish()
    summary=json.loads((binder.output/'seat_forced_summary.json').read_text('utf-8'))
    assert len(summary['bound_posts'])==1 and summary['bound_posts'][0]['seat']=='seat1'
    receipt=summary['historical_binding_receipts'][0]
    assert receipt['binding_anchor_frame']==23
    assert receipt['later_literal_conflicts']==[dict(actor='bankiir',seat='seat2',frames=[30,31,32],source_ref='late_other_seat')]
