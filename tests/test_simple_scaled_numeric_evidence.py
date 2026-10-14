"""Recorded false scaled1, retained dim positives and source-bound scale guards."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from parser_core.first_stack_ocr import (RULE,proof_config,scale_pixels,
    simple_scaled_numeric_evidence)
from parser_core.simple_saved_policy import saved_stack_observations
from parser_core.simple_stack_policy import first_displayed_stack

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=json.loads((ROOT/'tests/fixtures/simple_scaled_native_smoke18.json').read_text('utf8'))


def retained(frame):
    return copy.deepcopy(next(r['reading'] for r in FIXTURE['records'] if r['reading']['source_frame']==frame))


def positive_scaled(primary_score=.8):
    """Mutate confidence only on retained numeric pixels/recorded scaled OCR."""
    contexts=json.loads((ROOT/'tests/fixtures/first_visible_stack_smoke19.json').read_text('utf8'))
    r=copy.deepcopy(next(c for c in contexts['contexts'] if c['seat']=='seat5')['readings'][-3])
    data=json.loads((ROOT/'tests/fixtures/first_stack_scaled_smoke19.json').read_text('utf8'))
    alt=next(v for v in data['measured_results'] if v['variant']=='direct_nearest2' and
        all(v['source'][k]==r[k] for k in ('table_id','seat','source_frame')))
    r['confidence']=primary_score; r['ocr_response']['score']=primary_score
    model,slots=proof_config()
    item=f"{r['table_id']}:{r['source_frame']}:{list(slots[f'slot_{r['table_id']:02d}']['seats']).index(r['seat'])}"
    gray,enlarged=scale_pixels(r['png_base64'],r['pixel_sha256'])
    assert hashlib.sha256(enlarged.tobytes()).hexdigest()==alt['derived_pixel_sha256']
    r['ocr_response']['item_id']=item
    r['ocr_response']['same_source_scale_evidence']=dict(rule=RULE,status='OBSERVED',
        source_item_id=item,source_pixel_sha256=r['pixel_sha256'],source_shape=list(gray.shape),
        scaled_shape=list(enlarged.shape),scaled_pixel_sha256=alt['derived_pixel_sha256'],
        model_name=model['model_name'],weights_sha256=model['weights_sha256'],
        primary_response=dict(status=r['ocr_status'],raw_text=r['raw_text'],score=primary_score),
        response=dict(alt['response'],item_id=item))
    return r


def test_recorded_false_scaled_one_stays_unknown_without_absence_or_empty_claim():
    r=retained(85); before=copy.deepcopy(r)
    proof=simple_scaled_numeric_evidence(r)
    assert proof['scaled']['status']=='UNKNOWN'
    assert proof['scaled']['reason']=='SCALED_ONLY_WITHOUT_POSITIVE_NATIVE_DIGITS'
    assert proof['scaled']['same_source_bound'] is True
    assert proof['scaled']['visibility_status']=='NOT_DIGITS_VISIBLE'
    assert proof['primary_conflict'] is False and r==before
    assert 'EMPTY' not in json.dumps(proof)


def test_retained_positive_pixels_still_allow_an_agreeing_scaled_rescue():
    r=positive_scaled(); before=copy.deepcopy(r)
    proof=simple_scaled_numeric_evidence(r)
    assert proof['scaled']['status']=='ADMITTED'
    assert proof['scaled']['scaled_value']=='28260'
    assert proof['scaled']['visibility_status']=='DIGITS_VISIBLE'
    assert r==before


@pytest.mark.parametrize('mutation',['parent_item','source_hash','scaled_hash','shape','weights','model','rule','png'])
def test_unbound_scale_supplies_neither_promotion_nor_conflict(mutation):
    r=positive_scaled(); p=r['ocr_response']['same_source_scale_evidence']
    if mutation=='parent_item':p['source_item_id']='other'
    elif mutation=='source_hash':p['source_pixel_sha256']='0'*64
    elif mutation=='scaled_hash':p['scaled_pixel_sha256']='0'*64
    elif mutation=='shape':p['source_shape']=[1,2]
    elif mutation=='weights':p['weights_sha256']='0'*64
    elif mutation=='model':p['model_name']='other'
    elif mutation=='rule':p['rule']='other'
    else:r['png_base64']='corrupt'
    proof=simple_scaled_numeric_evidence(r)
    assert proof['scaled']['status']=='UNKNOWN' and not proof['primary_conflict']


@pytest.mark.parametrize('mutation',['missing_visibility','rgb_hash','gate_claim'])
def test_unverified_positive_gate_cannot_rescue_weak_primary(mutation):
    r=positive_scaled()
    if mutation=='missing_visibility':r.pop('visibility_source')
    elif mutation=='rgb_hash':r['visibility_source']['source_rgb_sha256']='0'*64
    else:r['visibility_source']['digit_gate']['status']='NOT_DIGITS_VISIBLE'
    proof=simple_scaled_numeric_evidence(r)
    assert proof['scaled']['status']=='UNKNOWN' and not proof['primary_conflict']


def test_bound_confident_disagreement_withholds_both_readings():
    r=positive_scaled(.9)
    r['ocr_response']['same_source_scale_evidence']['response']['raw_text']='999999'
    proof=simple_scaled_numeric_evidence(r)
    assert proof['primary_conflict'] and proof['scaled']['status']=='UNKNOWN'
    assert proof['scaled']['reason']=='BOUND_PRIMARY_SCALED_NUMERIC_CONFLICT'


def test_high_scaled_confidence_cannot_replace_missing_positive_pixels():
    r=retained(85)
    r['ocr_response']['same_source_scale_evidence']['response']['score']=.99999
    assert simple_scaled_numeric_evidence(r)['scaled']['status']=='UNKNOWN'


def test_numeric_scale_cannot_rescue_a_nonnumeric_primary():
    r=positive_scaled()
    r['raw_text']='CHECK'
    r['ocr_response']['same_source_scale_evidence']['primary_response']['raw_text']='CHECK'
    assert simple_scaled_numeric_evidence(r)['scaled']['status']=='UNKNOWN'


@pytest.fixture(scope='module')
def observations():
    source=Path(FIXTURE['source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==FIXTURE['source_sha256']
    for row in FIXTURE['records']:
        actual=json.loads(source.read_text('utf8').splitlines()[row['source_line']-1])
        assert actual==row['reading']
    return saved_stack_observations(source.parent)


def test_actual_simple_selection_skips_false_one_and_keeps_raw_candidate(observations):
    r=retained(85); route=(8,r['session_id'],'seat4')
    all_rows=observations[route]
    false=[x for x in all_rows if x.get('source_line')==621 and x['source_kind']=='SAVED_DIRECT_SCALED']
    assert len(false)==1 and false[0]['raw_text']=='1'
    assert false[0]['numeric_admission']['status']=='UNKNOWN'
    first=first_displayed_stack(all_rows,table_id=8,session_id=r['session_id'],seat='seat4',hand_start=35,hand_end=87)
    assert first['displayed_stack']==91463 and first['frame']==86
    assert first['source_kind']=='SAVED_DIRECT_PRIMARY'


def test_negative_digit_gate_does_not_veto_retained_strong_dim_primary(observations):
    r=retained(1748); route=(2,r['session_id'],'seat3')
    assert r['visibility_source']['digit_gate']['status']=='NOT_DIGITS_VISIBLE'
    first=first_displayed_stack(observations[route],table_id=2,session_id=r['session_id'],seat='seat3',hand_start=1748,hand_end=1750)
    assert first['displayed_stack']==104950 and first['source_kind']=='SAVED_DIRECT_PRIMARY'
