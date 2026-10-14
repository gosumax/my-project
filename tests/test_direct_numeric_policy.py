"""Calibrated first-number policy; native pixels and synthetic guard mutations."""
import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest
from parser_core.first_stack_ocr import direct_numeric_policy,effective_numeric,proof_config,scale_pixels,RULE
from parser_core.start_stack_live import first_visible_stack_triple
from parser_core.start_stack_visibility import first_visible_proof,verify_first_visible_fact

ROOT=Path(__file__).resolve().parents[1]


def native_context():
    data=json.loads((ROOT/'tests/fixtures/first_visible_stack_smoke19.json').read_text('utf-8'))
    return next(r for r in data['contexts'] if r['seat']=='seat1')


def candidate_context(score=.85):
    c=native_context()
    for r in c['readings'][-3:]:
        r['confidence']=score
        r['ocr_response']['score']=score
    return c


def check(c):
    return first_visible_proof({r['source_frame']:r for r in c['readings']},[284,285,286],
        start_frame=258,confidence=.85,numeric_policy=direct_numeric_policy())


def attach_bound_scale(row,text,score=.99):
    model,slots=proof_config()
    item=f"{row['table_id']}:{row['source_frame']}:{list(slots[f'slot_{row['table_id']:02d}']['seats']).index(row['seat'])}"
    gray,enlarged=scale_pixels(row['png_base64'],row['pixel_sha256'])
    row['ocr_response']['item_id']=item
    row['ocr_response']['same_source_scale_evidence']=dict(rule=RULE,status='OBSERVED',source_item_id=item,
        source_pixel_sha256=row['pixel_sha256'],source_shape=list(gray.shape),scaled_shape=list(enlarged.shape),
        scaled_pixel_sha256=hashlib.sha256(enlarged.tobytes()).hexdigest(),model_name=model['model_name'],
        weights_sha256=model['weights_sha256'],primary_response=dict(status=row['ocr_status'],raw_text=row['raw_text'],score=row['confidence']),
        response=dict(item_id=item,status='OK',raw_text=text,score=score))


def test_boundary_new_policy_and_legacy_replay_are_separate():
    c=candidate_context();p=check(c)
    assert p['status']=='PROVEN' and p['numeric_ocr_policy']==direct_numeric_policy()
    f=copy.deepcopy(c['fact']);f['evidence']=c['readings'][-3:];f['first_visible_proof']=p
    assert verify_first_visible_fact(f)
    f['first_visible_proof'].pop('numeric_ocr_policy')
    assert not verify_first_visible_fact(f)
    historical=native_context();old=first_visible_proof({r['source_frame']:r for r in historical['readings']},[284,285,286],start_frame=258)
    assert old['status']=='PROVEN' and 'numeric_ocr_policy' not in old
    historical['fact']['first_visible_proof']=old
    assert verify_first_visible_fact(historical['fact'])
    assert check(candidate_context(.849999))['status']=='UNKNOWN'


def single_attempt(readings):
    return first_visible_stack_triple(readings,.85,start_frame=258,allow_scaled=True,
        numeric_policy=direct_numeric_policy(),single_frame=True)


def test_first_single_stack_boundary_without_two_more_numeric_frames():
    c=candidate_context(.85);row=c['readings'][-3];frame=row['source_frame']
    result=single_attempt({frame:row})
    assert result['status']=='PASS' and result['frames']==[frame] and result['evidence']==[row]
    row['confidence']=.8499;row['ocr_response']['score']=.8499
    assert single_attempt({frame:row}) is None


def test_single_stack_waits_for_earlier_pending_and_keeps_first_source():
    c=candidate_context(.85);early,later=c['readings'][-3:-1];f=early['source_frame']
    later.update(raw_text='999999',confidence=.999)
    rows={f:None,later['source_frame']:later}
    assert single_attempt(rows) is None
    rows[f]=early
    result=single_attempt(rows)
    assert result['frames']==[f] and result['displayed_stack']!= '999999'


def test_single_native_frame_reaches_math_and_reconstruction_without_overlay_sequence():
    from parser_core.start_stack_visibility import FIRST_QUALIFIED_CONTRACT
    from parser_core.poker_math_engine import calculate_start_stack
    from parser_core.poker_reconstruction_layer import proven_start_stacks
    c=candidate_context(.85);row=c['readings'][-3];rows={row['source_frame']:row}
    attempt=single_attempt(rows);f=copy.deepcopy(c['fact']);f.update(attempt)
    f['status']='PROVEN_FOR_MATH'
    f['first_visible_proof']=first_visible_proof(rows,f['frames'],start_frame=258,
        confidence=.85,numeric_policy=direct_numeric_policy(),qualified_single=True)
    assert verify_first_visible_fact(f)
    result=calculate_start_stack(f,first_visible_contract=True)
    assert result['status']=='PASS' and result['initial_stack_contract']==FIRST_QUALIFIED_CONTRACT
    assert len(result['frames'])==1
    data=json.loads((ROOT/'tests/fixtures/first_visible_stack_smoke19.json').read_text('utf-8'))
    hand=dict(table_id='T05',session_id=f['session'],hand_id=f['hand_id'],
        messages=[dict(source_rows=data['canonical_events'])])
    stacks,receipts,rejected=proven_start_stacks(hand,[result],first_visible_contract=True)
    assert list(stacks.values())==[17786] and not rejected and len(receipts)==1
    assert c['fact']['frames']==[284,285,286]


@pytest.mark.parametrize('fault',['boundary_equal','source_hash','context','clock','invented_evidence','earlier_good'])
def test_single_stack_source_and_close_boundary_are_reproduced(fault):
    c=candidate_context(.85);row=c['readings'][-3];frame=row['source_frame'];rows={frame:row}
    end=frame if fault=='boundary_equal' else None
    if fault=='source_hash':row['pixel_sha256']='0'*64
    elif fault=='context':row['hand_id']=None
    elif fault=='clock':row['time_base']=[1,0]
    elif fault=='earlier_good':
        earlier=copy.deepcopy(row);earlier.update(source_frame=frame-1,source_pts=row['source_pts']-512)
        rows[frame-1]=earlier
    p=first_visible_proof(rows,[frame],start_frame=258,confidence=.85,numeric_policy=direct_numeric_policy(),
        qualified_single=True,clean_window_end_frame=end)
    f=copy.deepcopy(c['fact']);f.update(single_attempt({frame:row}) or {})
    f['first_visible_proof']=p
    if fault=='invented_evidence':f['evidence']=[copy.deepcopy(row),copy.deepcopy(row)]
    assert not verify_first_visible_fact(f)


@pytest.mark.parametrize('value',[True,False,None,'0.95',float('nan'),float('inf'),-.1,1.01])
def test_invalid_scores_never_admit_numeric_or_triple(value):
    c=candidate_context(value);readings={r['source_frame']:r for r in c['readings'][-3:]}
    for r in readings.values():assert effective_numeric(r,.85,allow_scaled=True,numeric_policy=direct_numeric_policy()) is None
    assert first_visible_stack_triple(readings,.85,final=True,start_frame=284,allow_scaled=True,numeric_policy=direct_numeric_policy()) is None


@pytest.mark.parametrize('value',['-100','+100','100 chips','Sit out','CHECK','1 00'])
def test_numeric_grammar_does_not_admit_overlay_or_signed_value(value):
    r=candidate_context()['readings'][-1];r['raw_text']=value
    assert effective_numeric(r,.85,allow_scaled=True,numeric_policy=direct_numeric_policy()) is None


@pytest.mark.parametrize('score',[.85,.90,.9499])
def test_bound_scale_conflict_withholds_new_primary_and_full_firstness(score):
    c=candidate_context(score)
    for r in c['readings'][-3:]:
        attach_bound_scale(r,'999999')
        assert effective_numeric(r,.85,allow_scaled=True)==r['raw_text'].replace(' ','')
        assert effective_numeric(r,.85,allow_scaled=True,numeric_policy=direct_numeric_policy()) is None
        assert effective_numeric(r,.85,allow_scaled=False,numeric_policy=direct_numeric_policy()) is None
    readings={r['source_frame']:r for r in c['readings'][-3:]}
    assert first_visible_stack_triple(readings,.85,final=True,start_frame=284,allow_scaled=True,numeric_policy=direct_numeric_policy()) is None
    assert check(c)['status']=='UNKNOWN'


@pytest.mark.parametrize('field',['source_item_id','source_pixel_sha256','scaled_pixel_sha256','model_name','weights_sha256'])
def test_unbound_scale_cannot_supply_numeric_conflict(field):
    r=candidate_context(.90)['readings'][-1];attach_bound_scale(r,'999999')
    r['ocr_response']['same_source_scale_evidence'][field]='foreign'
    assert effective_numeric(r,.85,allow_scaled=True,numeric_policy=direct_numeric_policy())==r['raw_text'].replace(' ','')
    r['confidence']=.80
    assert effective_numeric(r,.85,allow_scaled=True,numeric_policy=direct_numeric_policy()) is None


@pytest.mark.parametrize('field,value',[('policy_id','other'),('min_confidence',.84),('scope','SQL'),('bound_scaled_numeric_conflict','IGNORE')])
def test_policy_mutations_rejected_at_proof_and_consumer(field,value):
    c=candidate_context();p=check(c);policy=direct_numeric_policy();policy[field]=value
    assert first_visible_proof({r['source_frame']:r for r in c['readings']},[284,285,286],start_frame=258,
        confidence=.85,numeric_policy=policy)['status']=='UNKNOWN'
    f=copy.deepcopy(c['fact']);f['evidence']=c['readings'][-3:];f['first_visible_proof']=p
    p['numeric_ocr_policy']=policy
    assert not verify_first_visible_fact(f)


def test_shared_identity_occupancy_and_sql_thresholds_unchanged():
    config=json.loads((ROOT/'configs/dealing_start_stack.json').read_text('utf-8'))
    assert config['stack_confidence']==.95
    from parser_core.stack_timeline_algorithms import numeric
    assert numeric('12345',.90) is None


def test_actual_full_live_factory_callback_and_finish_use_one_numeric_policy(tmp_path,monkeypatch):
    import threading
    from collections import defaultdict
    from types import SimpleNamespace
    import cv2
    import numpy as np
    from parser_core.dealing_start_stack import CleanStackWindow
    from parser_core.first_stack_ocr import FirstStackOCR
    from parser_core.start_stack_live import StartStackLiveConnection
    from parser_core.ram_first_stage import RamRow
    from parser_core.parallel_ocr import row_key
    c=candidate_context(.85);first=c['readings'][0]
    table,session,hand,seat=(first[k] for k in ('table_id','session_id','hand_id','seat'))
    key=(table,session,hand);window=CleanStackWindow(.95)
    window.deal=dict(source_frame=257,occupied_seats=[seat])
    sidecar=SimpleNamespace(lock=threading.RLock(),active_sessions={table:session},
        windows={key:window},histories=defaultdict(list,{key[:2]:[dict(hand_id=hand,source_frame=257)]}),
        flags=dict(stack_confidence=.95,capture_frames=64),stack_reading=lambda *args:(None,'UNKNOWN'))
    class NeverReadClient:
        def __init__(self,*args,**kwargs):pass
        def read_batch(self,*args,**kwargs):raise AssertionError('No OCR allowed in source replay test')
    class QueueFixture:
        def __init__(self,factory,**kwargs):self.client=factory(0);self.errors=[]
        def finish(self):pass
        def submit_batch(self,*args):raise AssertionError('No new capture in callback fixture')
    monkeypatch.setattr('parser_core.start_stack_live.TinyChatClient',NeverReadClient)
    monkeypatch.setattr('parser_core.start_stack_live.ParallelRowOCR',QueueFixture)
    ocr=json.loads((ROOT/'configs/ocr_tiny_local.json').read_text('utf-8'))
    obj=StartStackLiveConnection(ROOT,tmp_path,sidecar,ocr,first_visible_contract=True)
    assert isinstance(obj.ocr.client,FirstStackOCR) and obj.ocr.client.confidence==.95
    assert sidecar.flags['stack_confidence']==.95 and obj.numeric_confidence==.85
    assert obj.numeric_policy==direct_numeric_policy()
    state=dict(deal_frame=257,source_readings={seat:{}},stop=set(),canonical_attempts={},numeric_triples={})
    obj.states[key]=state
    for original in c['readings']:
        r=copy.deepcopy(original);frame=r['source_frame']
        pixels=cv2.imdecode(np.frombuffer(base64.b64decode(r['png_base64']),np.uint8),0)
        row=RamRow(frame,None,table,session,0,r['roi'][1],'STACK_ONLY',0,pixels,r['pixel_sha256'],
            f'{session}:STACK:{hand}:{seat}:{frame}',r['source_pts'],*r['time_base'])
        obj.jobs[row_key(row)]=(key,seat,r)
        state['source_readings'][seat][frame]=None
        response=dict(r['ocr_response'],status=r['ocr_status'],raw_text=r['raw_text'],score=r['confidence'])
        obj.completed(row,response)
    incremental=copy.deepcopy(state['canonical_attempts'][seat])
    assert incremental['status']=='PASS' and incremental['frames']==[284]
    obj.finish()
    final=state['canonical_attempts'][seat]
    assert final['frames']==incremental['frames'] and final['displayed_stack']==incremental['displayed_stack']
    assert final['first_visible_proof']['status']=='PROVEN'
    assert final['first_visible_proof']['numeric_ocr_policy']==direct_numeric_policy()
    fact=copy.deepcopy(c['fact']);fact.update(final)
    assert verify_first_visible_fact(fact)
    summary=json.loads((tmp_path/'start_stack_live_summary.json').read_text('utf-8'))
    assert summary['hands'][0]['canonical_attempts'][seat]['first_visible_proof']==final['first_visible_proof']
