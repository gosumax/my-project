import copy
import subprocess
import sys

import numpy as np
import pytest

from experiments.nickname_occupancy.core import consensus, label, prefilter, build_epochs, bind
from experiments.nickname_occupancy.provider import OccupancyTransitions


def reads(names):
    return [dict(read_id=f'R{i}',frame=i*15,crop_hash=f'h{i}',normalized_text=n,confidence=.99) for i,n in enumerate(names)]


def group(names):
    return dict(group_id=1,table=1,seat='seat1',reads=reads(names),end_frame=100,coverage_segment=1)


@pytest.mark.parametrize('strategy,names,expected', [(1,['Alice'],'Alice'),(2,['Alice','Alice'],'Alice'),
    (2,['Alice','A1ice'],None),(3,['Alice','Alice','Alice'],'Alice'),(3,['Alice','Alice','A1ice'],'Alice'),
    (5,['Alice','A1ice','Al1ce','Alice','Alice'],'Alice'),(5,['A','B','C','D','E'],None)])
def test_exact_consensus(strategy,names,expected):
    assert consensus(reads(names),strategy)['nickname']==expected


def test_independent_source_frames_and_hashes():
    for mutation in ['frame','crop_hash']:
        r=reads(['Alice']*3);r[1][mutation]=r[0][mutation]
        assert consensus(r,3)['nickname'] is None
    r=reads(['Alice']*3);r[1]['frame']=1
    assert consensus(r,3)['nickname'] is None


@pytest.mark.parametrize('text',['SIT OUT','OPEN SEAT','WINNER','PENSEAT','CALL','FOLD','CHECK','RAISE 123','Alice...','Alice…'])
def test_overlay_and_truncation(text):
    value,status=label(text,.99)
    assert value is None or status=='EMPTY_LABEL'


def test_prefilter_rejects_empty_and_block():
    assert not prefilter(np.zeros((15,90),np.uint8))[0]
    assert not prefilter(np.full((15,90),255,np.uint8))[0]


def test_conflicting_reads_do_not_create_binding_epoch():
    assert build_epochs([group(['Alice','Alice','A1ice'])],[[0,100]],'sha')==[]


def test_confirmation_and_coverage_are_closed():
    epochs=build_epochs([group(['Alice']*3)],[[0,100]],'sha')
    assert epochs[0]['start_frame']==30
    epochs[0]['manually_verified']=True
    p=dict(table_id=1,seat_id='seat1',frame_index=31,coverage_segment=1)
    before=copy.deepcopy(p)
    assert bind(p,epochs,source_sha256='sha')['nickname']=='Alice' and p==before
    for field,value in [('table_id',2),('seat_id','seat2'),('frame_index',29),('frame_index',101),('coverage_segment',2)]:
        assert bind(dict(p,**{field:value}),epochs,source_sha256='sha')['nickname'] is None
    assert bind(p,epochs,source_sha256='other')['nickname'] is None
    assert build_epochs([group(['Alice']*3)],[[0,20]],'sha')==[]


def test_manual_pending_is_not_authority():
    epochs=build_epochs([group(['Alice']*3)],[[0,100]],'sha')
    assert bind(dict(table_id=1,seat_id='seat1',frame_index=35,coverage_segment=1),epochs,source_sha256='sha')['binding_status']=='CANDIDATE'


def test_noise_and_sitout_do_not_replace_occupant():
    m=OccupancyTransitions();old=m.observe(1,'seat1','Alice',30,confirmed=True)
    for nick in [None,'Bob','SIT OUT','OPEN SEAT']:
        assert m.observe(1,'seat1',nick,40,confirmed=False)==old
    assert m.observe(1,'seat1','Alice',70,confirmed=True)==old
    assert m.observe(1,'seat1','SIT OUT',75,confirmed=True)==old
    assert m.observe(1,'seat1','WINNER',76,confirmed=True)==old
    assert m.observe(1,'seat1','PENSEAT',77,confirmed=True)==old
    assert len(m.events)==1


def test_proven_replacement_empty_and_context():
    m=OccupancyTransitions();a=m.observe(1,'seat1','Alice',30,confirmed=True)
    assert m.events[-1]['kind']=='FIRST_OBSERVED_PLAYER'
    b=m.observe(1,'seat1','Bob',80,confirmed=True)
    assert b['epoch_id']!=a['epoch_id'] and m.events[-1]['kind']=='NICKNAME_REPLACEMENT'
    assert m.observe(1,'seat1','OPEN SEAT',90,confirmed=True) is None
    assert m.observe(1,'seat1','Bob',100,confirmed=True)['epoch_id']!=b['epoch_id']
    assert m.events[-1]['kind']=='EMPTY_TO_PLAYER'
    m.observe(1,'seat1','Bob',110,confirmed=True,context='new-table')
    assert m.events[-1]['kind']=='TABLE_REPLACEMENT'


def test_off_creates_no_outputs_and_no_ocr(tmp_path):
    output=tmp_path/'off'
    result=subprocess.run([sys.executable,'experiments/nickname_occupancy/run.py','--output',str(output)],capture_output=True,text=True)
    assert result.returncode==0 and '"ocr_calls": 0' in result.stdout and not output.exists()
