"""Native EMPTY labels and proof rejection, separate from card OCR/identity."""
import base64
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from parser_core.empty_seat_evidence import EmptySeatButtonModel, frame_partition

ROOT=Path(__file__).resolve().parents[1]
FIXTURES=json.loads((ROOT/'tests/fixtures/empty_button_smoke19.json').read_text('utf-8'))


@pytest.fixture(scope='module')
def model():return EmptySeatButtonModel(ROOT)


def pixels(row):
    return np.frombuffer(base64.b64decode(row['pixels_base64']),np.uint8).reshape(row['shape']).copy()


@pytest.mark.parametrize('row',FIXTURES,ids=lambda r:f"T{r['table']}_{r['seat']}_{r['label']}")
def test_native_full_buttons_and_occupied_regions(model,row):
    gray=pixels(row)
    assert hashlib.sha256(gray.tobytes()).hexdigest()==row['pixel_sha256']
    result=model.classify(gray)
    assert (result['status']=='EMPTY')==(row['label']=='EMPTY')


def proof(model):
    row=next(r for r in FIXTURES if r['label']=='EMPTY')
    source=dict(table_id=row['table'],session_id='native',hand_id='1234567890',
                source_frame=row['frame'],source_pts=row['frame']*512,time_base=[1,15360])
    gray=pixels(row);ok,png=cv2.imencode('.png',gray);assert ok
    return source,dict(source,seat=row['seat'],roi=row['roi'],shape=row['shape'],
        png_base64=base64.b64encode(png).decode(),pixel_sha256=row['pixel_sha256'],**model.classify(gray))


@pytest.mark.parametrize('field,value',[
    ('session_id','different'),('hand_id','other'),('source_frame',-1),('source_pts',-1),
    ('time_base',[1,30]),('roi',[0,0,128,48]),('shape',[1,1]),
    ('pixel_sha256','0'*64),('png_base64','corrupt'),('model_sha256','0'*64),
    ('translation',[2,2]),('template_id','wrong'),('full_ncc',0.)])
def test_button_receipt_reproduces_source_pixels_geometry_and_scores(model,field,value):
    source,row=proof(model)
    assert model.verify(row,source)
    row[field]=value
    assert not model.verify(row,source)


@pytest.mark.parametrize('damage',['blank','only_text','hide_first_word','hide_last_word','crop','number'])
def test_partial_button_or_numeric_text_is_not_empty(model,damage):
    gray=pixels(next(r for r in FIXTURES if r['label']=='EMPTY'))
    if damage=='blank':gray[:]=30
    elif damage=='only_text':
        text=gray[12:36,20:108].copy();gray[:]=30;gray[12:36,20:108]=text
    elif damage=='hide_first_word':gray[12:36,20:64]=30
    elif damage=='hide_last_word':gray[12:36,64:108]=30
    elif damage=='crop':gray=gray[:,20:]
    else:
        gray[12:36,20:108]=30
        cv2.putText(gray,'10 000',(22,31),cv2.FONT_HERSHEY_SIMPLEX,.45,220,1)
    assert model.classify(gray)['status']=='UNKNOWN'


def test_partition_requires_positive_cards_for_every_other_seat(model):
    source,empty=proof(model);seats=['seat1','seat2',empty['seat']]
    observed={empty['seat']:empty};scores=dict(seat1=.99,seat2=.98)
    result=frame_partition(source,seats,scores,.95,observed)
    assert result['status']=='PROVEN_FRAME'
    assert result['occupied_seats']==['seat1','seat2']
    assert frame_partition(source,seats,dict(seat1=.99),.95,observed)['status']=='UNKNOWN'
    assert frame_partition(source,seats,scores,.95,{})['status']=='UNKNOWN'
    assert frame_partition(source,seats,dict(scores,**{empty['seat']:.99}),.95,observed)['status']=='CONFLICT'
    label=dict(source,state='OCCUPIED')
    assert frame_partition(source,seats,scores,.95,observed,{empty['seat']:label})['status']=='CONFLICT'
    label['hand_id']='wrong'
    assert frame_partition(source,seats,scores,.95,observed,{empty['seat']:label})['status']=='UNKNOWN'
