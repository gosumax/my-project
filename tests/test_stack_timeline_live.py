import ast
import hashlib
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np

from parser_core import stack_timeline_algorithms as algorithms
from parser_core.stack_evidence import StackEvidenceProvider
from parser_core.stack_timeline_live import StackTimelineLive

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_algorithm_source_parity():
    promoted = ast.parse((ROOT/'parser_core/stack_timeline_algorithms.py').read_text('utf-8'))
    for source, names in [
        ('experiments/stack_visual_tracker/tracker.py', {'metrics','different','Track'}),
        ('experiments/stack_digit_gate/gate.py', {'segments','normalized','DigitGate'}),
        ('experiments/stack_numeric_timeline/sitout_mask.py', {'SitOutMask'}),
        ('experiments/stack_numeric_timeline/timeline.py', {'Timeline','numeric','RawTiny'})]:
        original = ast.parse((ROOT/source).read_text('utf-8'))
        for old in original.body:
            if getattr(old,'name',None) not in names:
                continue
            new = next(n for n in promoted.body if getattr(n,'name',None) == old.name)
            # Separate modules' VERSION globals are retained explicitly in extraction.
            for n in ast.walk(new):
                if isinstance(n,ast.Name) and n.id == 'GATE_VERSION':
                    n.id = 'VERSION'
            assert ast.dump(new,include_attributes=False) == ast.dump(old,include_attributes=False)
    assert (ROOT/'parser_core/raw_worker.py').read_bytes() == (ROOT/'experiments/stack_numeric_timeline/raw_worker.py').read_bytes()
    for old,new in [('runs/stack_digit_gate_20261004_dataset_v1/gate_config_dev.json','configs/stack_digit_gate.json'),
        ('runs/stack_roi_refine_20261004_v3/layout_stack_refined.json','configs/stack_roi_refined.json'),
        ('experiments/stack_numeric_timeline/sitout_mask.json','configs/stack_sitout_mask.json')]:
        assert (ROOT/old).read_bytes() == (ROOT/new).read_bytes()


def test_real_initial_pixels_continuous_writer_and_gate(monkeypatch,tmp_path):
    source = ROOT/'runs/stack_pipeline_integration_20261005/preflight_gate/timeline.sqlite'
    with sqlite3.connect(source) as db:
        rows = db.execute("select table_id,seat_id,raw_gray,width,height,digit_gate_status from visual_states where origin='INITIAL'").fetchall()
    calls = []
    class Tiny:
        def __init__(self,out):pass
        def read(self,states):
            assert all(s['digit_gate_status']=='DIGITS_VISIBLE' for s in states)
            assert 1 <= len(states) <= 8
            calls.extend(s['state_id'] for s in states)
            return dict(results=[dict(item_id=s['state_id'],raw_text='12345',score=.99) for s in states],inference_seconds=.001)
        def close(self):pass
    monkeypatch.setattr('parser_core.stack_timeline_live.RawTiny',Tiny)
    live = StackTimelineLive(tmp_path,video_sha256='a'*64,sessions={},frames=2)
    tables = {t:np.zeros((404,847,3),np.uint8) for t in range(1,10)}
    for table,seat,raw,w,h,status in rows:
        route = next(r for r in live.routes if r['table_id']==table and r['seat_id']==seat)
        x,y,X,Y=route['xyxy']
        gray=np.frombuffer(raw,np.uint8).reshape(h,w)
        tables[table][y:Y,x:X]=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    for frame in range(2):
        for table in range(1,10):
            live.observe_frame(table,f's{table}',frame,frame*512,[1,15360],tables[table],None)
        with sqlite3.connect(live.path) as db:
            assert db.execute('select count(*) from observations').fetchone()[0]==72
            assert db.execute("select count(*) from visual_states where ocr_called=1 and digit_gate_status!='DIGITS_VISIBLE'").fetchone()[0]==0
            assert db.execute('pragma integrity_check').fetchone()[0]=='ok'
    assert live.counts['roi_checks']==144
    assert len(calls)==sum(status=='DIGITS_VISIBLE' for *_,status in rows)
    assert len(calls)<72
    live.finish()
    live.finish()
    provider=StackEvidenceProvider(live.path,video_sha256='a'*64)
    assert len(provider.points)==72
    assert sum(p['stack_value'] is not None for p in provider.points)==len(calls)
    summary=provider.write_audit(tmp_path,[],start_stack=True)
    assert summary['start_stack_enabled'] is True
    assert summary['recovered_proven']==0
    assert not list(tmp_path.rglob('*.png'))
