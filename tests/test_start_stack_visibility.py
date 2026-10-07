import base64
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from parser_core.start_stack_visibility import VisibilitySource, first_visible_proof, verify_first_visible_fact
from parser_core.poker_math_engine import calculate_start_stack
from parser_core.poker_reconstruction_layer import proven_start_stacks
from parser_core.ram_first_stage import png_compatible_gray
from parser_core.stack_evidence import StackEvidenceProvider


@pytest.fixture
def real():
    return json.loads((Path(__file__).parent/'fixtures/first_visible_stack_smoke19.json').read_text('utf-8'))


def context(real,seat='seat4'):
    return next(r for r in real['contexts'] if r['seat']==seat)


def proof(c):
    return first_visible_proof({r['source_frame']:r for r in c['readings']},[284,285,286],start_frame=258)


def fact(c):
    f=copy.deepcopy(c['fact'])
    f['first_visible_proof']=proof(c)
    return f


@pytest.mark.parametrize('seat,total',[('seat1','17786'),('seat4','11125')])
def test_real_forced_fade_blank_and_first_numeric_has_full_only_authority(real,seat,total):
    c=context(real,seat);f=fact(c)
    assert f['first_visible_proof']['status']=='PROVEN'
    assert f['first_visible_proof']['phases'][-4]['phase']=='BLANK_WITHIN_CODEC_NOISE'
    assert verify_first_visible_fact(f)
    old=calculate_start_stack(f)
    assert old['START_STACK'] is None and old['blocker']=='CLEAN_WINDOW_TIMING_UNPROVEN'
    result=calculate_start_stack(f,first_visible_contract=True)
    assert result['status']=='PASS' and result['START_STACK']==total
    hand=dict(table_id='T05',session_id=f['session'],hand_id=f['hand_id'],messages=[dict(source_rows=real['canonical_events'])])
    stacks,receipts,rejections=proven_start_stacks(hand,[result],first_visible_contract=True)
    assert list(stacks.values())==[int(total)] and not rejections
    assert receipts[0]['START_STACK']==total
    assert proven_start_stacks(hand,[result])[0]=={}


def test_real_first_numeric_low_confidence_cannot_be_replaced_by_later_value(real):
    c=context(real,'seat5')
    assert proof(c)['status']=='UNKNOWN'


@pytest.mark.parametrize('corrupt', ['gap','hash','context','pts','roi','earlier_high_confidence_number'])
def test_first_source_proof_rejects_missing_or_contradictory_evidence(real,corrupt):
    c=context(real)
    r=c['readings'][8]
    if corrupt=='gap':c['readings'].pop(8)
    elif corrupt=='hash':r['pixel_sha256']='0'*64
    elif corrupt=='context':r['hand_id']='prior-hand'
    elif corrupt=='pts':r['source_pts']=0
    elif corrupt=='roi':r['visibility_source']['timeline_roi'][0]+=1
    else:r.update(raw_text='999',ocr_status='OK',confidence=.999)
    assert proof(c)['status']=='UNKNOWN'


def replace_pixels(reading,rgb):
    x,y,X,Y=reading['roi']
    table=np.zeros((404,847,3),np.uint8);table[y:Y,x:X]=rgb
    gray=png_compatible_gray(rgb)
    reading['visibility_source']=VisibilitySource().capture(table,reading['table_id'],reading['seat'],reading['roi'])
    reading['pixel_sha256']=hashlib.sha256(gray.tobytes()).hexdigest()
    reading['png_base64']=base64.b64encode(cv2.imencode('.png',gray)[1]).decode('ascii')


@pytest.mark.parametrize('scale',[.1,.2,.5,1.])
@pytest.mark.parametrize('seat',['seat1','seat2','seat4'])
def test_dim_earlier_number_is_not_proven_absent_by_negative_digit_gate(real,scale,seat):
    c=context(real,seat)
    r=c['readings'][12]
    number=c['readings'][-1]['visibility_source']
    rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(number['source_rgb_png_base64']),np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    replace_pixels(r,np.clip(rgb.astype(float)*scale,0,255).astype(np.uint8))
    r.update(raw_text='',ocr_status='EMPTY',confidence=0.)
    assert proof(c)['status']=='UNKNOWN'
    assert proof(c)['reason']=='UNEXPLAINED_PRE_NUMERIC_PIXELS'


def test_late_immediate_numeric_capture_has_no_first_visibility_authority(real):
    c=context(real)
    assert first_visible_proof({r['source_frame']:r for r in c['readings'][-3:]},[284,285,286],start_frame=284)['status']=='UNKNOWN'


def provider_for(f,value=None):
    provider=StackEvidenceProvider.__new__(StackEvidenceProvider)
    r=f['evidence'][0]
    provider.points=[dict(table_id=f['table'],session_id=f['session'],seat_id=f['seat'],frame_index=r['source_frame'],
        source_pts=r['source_pts'],source_time_base=json.dumps(r['time_base']),exact_hash=r['visibility_source']['timeline_pixel_sha256'],
        stack_value=value,status='UNKNOWN' if value is None else 'CONFIRMED_NUMERIC',observation_id=123,source_state_id='state123')]
    return provider


def test_distinct_rois_join_only_through_reproduced_same_rgb_source(real):
    f=fact(context(real))
    assert f['evidence'][0]['pixel_sha256']!=f['evidence'][0]['visibility_source']['timeline_pixel_sha256']
    provider=provider_for(f)
    assert provider.match_snapshot(f)['status']=='PROVEN'
    assert provider.points[0]['status']=='UNKNOWN'  # SQLite OCR stays independent.
    provider.points[0]['source_pts']+=1
    assert provider.match_snapshot(f)['status']=='UNKNOWN'


def test_independent_confirmed_numeric_conflict_never_reaches_solver(real):
    f=calculate_start_stack(fact(context(real)),first_visible_contract=True)
    provider=provider_for(f,'999')
    assert provider.match_snapshot(f)['status']=='CONFLICT'
    hand=dict(table_id='T05',session_id=f['session'],hand_id=f['hand_id'],messages=[dict(source_rows=real['canonical_events'])])
    stacks,receipts,rejected=proven_start_stacks(hand,[f],first_visible_contract=True,stack_provider=provider)
    assert not stacks and not receipts and rejected[0]['reason']=='INDEPENDENT_STACK_OCR_CONFLICT'


def test_conflict_withheld_from_solver_is_still_explicit_in_export_audit(real,tmp_path):
    f=calculate_start_stack(fact(context(real)),first_visible_contract=True)
    provider=provider_for(f,'999');provider.sha256='test';provider.metadata={}
    provider.points[0]['delta_kind']='INITIAL'
    hand=dict(table_id='T05',session_id=f['session'],hand_id=f['hand_id'],messages=[dict(source_rows=real['canonical_events'])])
    _,receipts,rejected=proven_start_stacks(hand,[f],first_visible_contract=True,stack_provider=provider)
    result=dict(segment_id='real-hand',start_stack_join_receipts=receipts,start_stack_join_rejections=rejected)
    summary=provider.write_audit(tmp_path,[dict(segment_id='real-hand',observed_hand=hand)],start_stack=True,reconstruction_results=[result])
    assert summary['start_stack_conflicts']==1
    rows=[json.loads(line) for line in (tmp_path/'stack_start_stack.jsonl').read_text('utf-8').splitlines()]
    assert next(r for r in rows if r['player']=='leopold7024')['status']=='START_STACK_CONFLICT'


def test_verified_proof_is_recomputed_not_trusted_as_boolean(real):
    f=fact(context(real));f['first_visible_proof']['status']='PROVEN'
    f['first_visible_proof']['phases'][4]['phase']='INVENTED'
    assert not verify_first_visible_fact(f)
    assert calculate_start_stack(f,first_visible_contract=True)['START_STACK'] is None
