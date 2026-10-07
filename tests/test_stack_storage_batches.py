"""Bounded storage regression; fake OCR avoids a long video A/B."""
import json
import sqlite3

import cv2
import numpy as np

from parser_core.stack_timeline_live import StackTimelineLive
from parser_core.stack_evidence import StackEvidenceProvider


def test_batch_release_exact_recurrence_and_final_last_seen(monkeypatch,tmp_path):
    batches=[]
    class Tiny:
        def __init__(self,out):pass
        def read(self,states):
            batches.append([s['state_id'] for s in states])
            return dict(results=[dict(item_id=s['state_id'],raw_text='12345',score=.99) for s in states],inference_seconds=.001)
        def close(self):pass
    monkeypatch.setattr('parser_core.stack_timeline_live.RawTiny',Tiny)
    live=StackTimelineLive(tmp_path,video_sha256='a'*64,sessions={},frames=9)
    # Every state is admitted by this test gate; visual confirmation remains real.
    live.timeline.gate.evaluate=lambda image:dict(status='DIGITS_VISIBLE',score=1.,evidence={})
    images={t:np.zeros((404,847,3),np.uint8) for t in range(1,10)}
    route=live.routes[0];x,y,X,Y=route['xyxy']
    rng=np.random.default_rng(12)
    a=rng.integers(0,256,size=(Y-y,X-x),dtype=np.uint8)
    b=255-a
    for frame in range(9):
        gray=b if 1<=frame<=3 else a
        images[1][y:Y,x:X]=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        for table in range(1,10):
            live.observe_frame(table,f's{table}',frame,frame*512,[1,15360],images[table],None)
        # Flush at frame2 while B is pending: first-source PTS must survive.
        if frame in (0,2,3,6):
            live.flush()
            assert not live.timeline.states.items and not live.timeline.observations.items
            assert all(isinstance(value,str) for pool in live.timeline.pools for value in pool.index.values())
    assert len(live.timeline.states)==73
    assert len(live.timeline.observations)==74
    assert sum(len(batch) for batch in batches)==73
    assert live.counts['historical_state_lookups']>0
    live.finish()
    with sqlite3.connect(live.path) as db:
        rows=db.execute("select state_id,frame_index,confirmed_frame,last_seen_frame,previous_observation_id,previous_known_observation_id,observed_delta from observations where table_id=1 and seat_id=? order by observation_id",(route['seat_id'],)).fetchall()
        assert [r[0] for r in rows]==['S00001','S00073','S00001']
        assert [r[1] for r in rows]==[0,1,4]
        assert rows[-1][3]==8
        assert rows[1][3]==3
        assert db.execute("select source_pts from observations where state_id='S00073'").fetchone()[0]==512
        assert db.execute('pragma integrity_check').fetchone()[0]=='ok'
    provider=StackEvidenceProvider(live.path,video_sha256='a'*64)
    assert len(provider.points)==74
    metrics=json.loads((live.output/'progress.json').read_text('utf-8'))
    assert metrics['pending_states']==metrics['pending_observations']==0
    assert metrics['commits']<9
    assert metrics['process']=='CLOSED'
