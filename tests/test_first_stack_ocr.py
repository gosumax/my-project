import base64
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from parser_core.first_stack_ocr import FirstStackOCR,effective_numeric,proof_config
from parser_core.start_stack_live import first_visible_stack_triple
from parser_core.start_stack_visibility import first_visible_proof

ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'tests/fixtures/first_stack_scaled_smoke19.json').read_text('utf-8'))


class RecordedOCR:
    def __init__(self,row,alt):self.row=row;self.alt=alt;self.calls=[]
    def read_batch(self,items,batch_size):
        self.calls.append(items)
        row=self.row
        if len(self.calls)==1:
            result=dict(status=row['ocr_status'],raw_text=row['raw_text'],score=row['confidence'])
        else:result={k:self.alt['response'][k] for k in ('status','raw_text','score')}
        return dict(results=[dict(item_id=items[0]['item_id'],**result)],inference_seconds=.01)


def scaled(row):
    alt=next(r for r in DATA['measured_results'] if r['variant']=='direct_nearest2' and
        all(r['source'][k]==row[k] for k in ('table_id','seat','source_frame')))
    client=RecordedOCR(row,alt);model,slots=proof_config()
    item_id=f"{row['table_id']}:{row['source_frame']}:{list(slots[f'slot_{row["table_id"]:02d}']['seats']).index(row['seat'])}"
    wrapper=FirstStackOCR(client,model_name=model['model_name'],weights_sha256=model['weights_sha256'])
    result=wrapper.read_batch([dict(item_id=item_id,png_base64=row['png_base64'],source_sha256=row['pixel_sha256'])],8)
    derived=copy.deepcopy(row);derived['ocr_response']=result['results'][0]
    if len(client.calls)==2:
        second=client.calls[1][0]
        assert second['source_sha256']==alt['derived_pixel_sha256']
    return derived,client


def test_saved_three_source_low_confidence_numbers_and_non_numeric_controls():
    groups={}
    for row in DATA['source_records']:
        r,client=scaled(row)
        alt=next(v for v in DATA['measured_results'] if v['variant']=='direct_nearest2' and
            all(v['source'][k]==row[k] for k in ('table_id','seat','source_frame')))
        expected=alt['expected_numeric']
        assert effective_numeric(r,allow_scaled=True)==expected
        assert (r['raw_text'],r['confidence'],r['pixel_sha256'])==(row['raw_text'],row['confidence'],row['pixel_sha256'])
        if expected and row['confidence']<.95:
            assert effective_numeric(r) is None
            groups.setdefault((r['table_id'],r['seat']),{})[r['source_frame']]=r
    assert len(groups)==4
    for readings in groups.values():
        result=first_visible_stack_triple(readings,.95,final=True,start_frame=min(readings),allow_scaled=True)
        assert result['status']=='PASS'
        assert result['first_numeric_visible_frame']==min(readings)
        assert first_visible_stack_triple(readings,.95,final=True,start_frame=min(readings)) is None


@pytest.mark.parametrize('corrupt',['source_sha','scaled_sha','shape','weights','rule','wrong_number','low_confidence',
    'item_id','frame','seat','primary','png','missing'])
def test_scale_proof_rejects_corruption_and_disagreement(corrupt):
    row=next(r for r in DATA['source_records'] if r['table_id']==5 and r['seat']=='seat5' and r['source_frame']==284)
    r,_=scaled(row);p=r['ocr_response']['same_source_scale_evidence']
    assert effective_numeric(r,allow_scaled=True)=='28260'
    if corrupt=='source_sha':p['source_pixel_sha256']='0'*64
    elif corrupt=='scaled_sha':p['scaled_pixel_sha256']='0'*64
    elif corrupt=='shape':p['scaled_shape']=[1,2]
    elif corrupt=='weights':p['weights_sha256']='0'*64
    elif corrupt=='rule':p['rule']='other'
    elif corrupt=='wrong_number':p['response']['raw_text']='28 280'
    elif corrupt=='low_confidence':p['response']['score']=.949
    elif corrupt=='item_id':p['response']['item_id']='other'
    elif corrupt=='frame':r['source_frame']+=1
    elif corrupt=='seat':r['seat']='seat6'
    elif corrupt=='primary':r['confidence']+=.01
    elif corrupt=='png':r['png_base64']='corrupt'
    else:r['ocr_response'].pop('same_source_scale_evidence')
    assert effective_numeric(r,allow_scaled=True) is None


def test_full_first_visibility_replays_scaled_evidence_without_later_substitute():
    fixture=json.loads((ROOT/'tests/fixtures/first_visible_stack_smoke19.json').read_text('utf-8'))
    context=next(r for r in fixture['contexts'] if r['seat']=='seat5')
    readings={r['source_frame']:r for r in context['readings']}
    assert first_visible_proof(readings,[284,285,286],start_frame=258)['status']=='UNKNOWN'
    for frame in (284,285,286):readings[frame]=scaled(readings[frame])[0]
    proof=first_visible_proof(readings,[284,285,286],start_frame=258)
    assert proof['status']=='PROVEN' and proof['displayed_stack']=='28260'
    readings.pop(285)
    assert first_visible_proof(readings,[284,285,286],start_frame=258)['status']=='UNKNOWN'


def test_failed_additional_inference_retains_primary_response():
    row=next(r for r in DATA['source_records'] if r['seat']=='seat5' and r['table_id']==5)
    model,_=proof_config()
    class Failure(RecordedOCR):
        def read_batch(self,items,batch_size):
            if self.calls:raise RuntimeError('test worker failure')
            return super().read_batch(items,batch_size)
    client=Failure(row,None)
    wrapper=FirstStackOCR(client,model_name=model['model_name'],weights_sha256=model['weights_sha256'])
    result=wrapper.read_batch([dict(item_id='5:284:4',png_base64=row['png_base64'],source_sha256=row['pixel_sha256'])],8)['results'][0]
    assert result['raw_text']==row['raw_text'] and result['score']==row['confidence']
    assert result['same_source_scale_evidence']['status']=='UNKNOWN'
