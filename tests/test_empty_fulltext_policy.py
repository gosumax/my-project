"""Calibrated native EMPTY acceptance and source-bound receipt regression."""
import base64
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from parser_core.empty_seat_evidence import EmptySeatButtonModel

ROOT=Path(__file__).resolve().parents[1]
FIXTURES=json.loads((ROOT/'tests/fixtures/empty_fulltext_calibration.json').read_text('utf-8'))


@pytest.mark.parametrize('row',FIXTURES,ids=lambda r:r['id'])
def test_retained_open_seat_passes_and_reproduces_source_receipt(row):
    model=EmptySeatButtonModel(ROOT)
    gray=np.frombuffer(base64.b64decode(row['pixels_base64']),np.uint8).reshape(row['shape'])
    assert hashlib.sha256(gray.tobytes()).hexdigest()==row['pixel_sha256']
    result=model.classify(gray)
    if row['expected'] is None:
        assert result==dict(status='UNKNOWN',reason='COMPLETE_EMPTY_BUTTON_NOT_PROVEN',model_sha256=model.sha256)
        return
    assert result==dict(status='EMPTY',**row['expected'],model_sha256=model.sha256)
    assert result['border_ncc']>=.08
    source={k:row[k] for k in ('table_id','hand_id','source_frame','source_pts','time_base')}
    source['session_id']='RETAINED_REGRESSION'
    ok,png=cv2.imencode('.png',gray);assert ok
    receipt=dict(source,**result,seat=row['seat'],roi=row['roi'],shape=row['shape'],
        pixel_sha256=row['pixel_sha256'],png_base64=base64.b64encode(png).decode())
    assert model.verify(receipt,source)
    for field,value in [('border_ncc',0.),('text_ncc',0.),('full_ncc',0.),
                        ('source_frame',row['source_frame']+1),('model_sha256','0'*64)]:
        damaged=dict(receipt);damaged[field]=value
        assert not model.verify(damaged,source)


@pytest.mark.parametrize('change',[
    {'schema':'unknown'}, {'full_ncc_min':.90}, {'text_ncc_min':.94},
    {'full_ncc_min':float('nan')}, {'text_ncc_min':float('nan')},
    {'border_ncc_min':.07}, {'border_check_enabled':False}, {'border_check_enabled':1},
    {'translation_radius':3},
])
def test_unreviewed_v2_policy_is_rejected(tmp_path,change):
    model=json.loads((ROOT/'models/empty_seat_buttons.json').read_text('utf-8'))
    model.update(change);path=tmp_path/'model.json';path.write_text(json.dumps(model),'utf-8')
    with pytest.raises(ValueError,match='EMPTY_BUTTON_MODEL_CONTRACT_UNKNOWN'):
        EmptySeatButtonModel(ROOT,path)


def test_legacy_schema_keeps_its_three_gates(tmp_path):
    config=json.loads((ROOT/'models/empty_seat_buttons.json').read_text('utf-8'))
    config.update(schema='native-full-empty-seat-button-v1',full_ncc_min=.98,text_ncc_min=.98,border_ncc_min=.98)
    del config['border_check_enabled']
    path=tmp_path/'legacy.json';path.write_text(json.dumps(config),'utf-8')
    legacy=EmptySeatButtonModel(ROOT,path)
    for row in FIXTURES:
        gray=np.frombuffer(base64.b64decode(row['pixels_base64']),np.uint8).reshape(row['shape'])
        assert legacy.classify(gray)['status']=='UNKNOWN'
    config['border_check_enabled']=False;path.write_text(json.dumps(config),'utf-8')
    with pytest.raises(ValueError,match='EMPTY_BUTTON_MODEL_CONTRACT_UNKNOWN'):
        EmptySeatButtonModel(ROOT,path)
