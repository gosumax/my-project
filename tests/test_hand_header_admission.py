"""Header-only physical admission and async provider regression contracts."""
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from parser_core.hand_header_admission import HeaderPrefixGate, HandHeaderAdmission
from parser_core.dealing_start_stack import DealingStartStackSidecar

ROOT=Path(__file__).resolve().parents[1]

class Queue:
    def __init__(self):self.rows=[];self.callback=None
    def submit_batch(self,rows):
        self.rows.extend(rows)
        if self.callback:
            for row in rows:self.callback(row,{'status':'OK','raw_text':'Hand #2876718828'})

@pytest.fixture
def lane(tmp_path,monkeypatch):
    flags=json.loads((ROOT/'configs/dealing_start_stack.json').read_text('utf-8'))
    provider=DealingStartStackSidecar(ROOT,tmp_path,flags)
    queue=Queue();lane=HandHeaderAdmission(ROOT,tmp_path,provider,queue)
    monkeypatch.setattr('parser_core.hand_header_admission.physical_spans',lambda gray:[(0,16)])
    monkeypatch.setattr('parser_core.hand_header_admission.padded_row_span',lambda spans,i,h:(0,16))
    yield lane,provider,queue
    lane.finish();provider.finish()

def crop(name='Hand_hash',digit=1):
    template=cv2.imdecode(np.frombuffer((ROOT/f'models/hand_header_prefix/{name}.png').read_bytes(),np.uint8),0)
    pixels=np.full((16,228),25,np.uint8)
    pixels[:template.shape[0],8:8+template.shape[1]]=template
    pixels[4:12,60+digit:62+digit]=180
    return pixels

def observe(lane,pixels,frame=10,session='S'):
    lane.observe_frame(1,session,frame,frame*512,[1,15360],None,pixels)

@pytest.mark.parametrize('name',['Hand_hash','hash'])
def test_real_prefix_template_is_candidate(name):
    assert HeaderPrefixGate(ROOT).classify(crop(name))['candidate']

def test_ordinary_chat_and_displaced_body_hash_are_not_candidates():
    gate=HeaderPrefixGate(ROOT)
    for text in ['Dealer: posts Ante','Dealer: Dealing cards','Dealer: gosumax bets']:
        pixels=np.full((16,228),25,np.uint8)
        cv2.putText(pixels,text,(8,12),cv2.FONT_HERSHEY_SIMPLEX,.35,180,1)
        assert not gate.classify(pixels)['candidate']
    assert not gate.classify(np.roll(crop('hash'),90,axis=1))['candidate']

def test_repeats_scroll_dedup_preserves_changed_digit(lane):
    obj,provider,queue=lane
    pixels=crop();observe(obj,pixels)
    observe(obj,pixels,11);observe(obj,np.roll(pixels,1,axis=0),12)
    observe(obj,crop(digit=4),13)
    assert len(queue.rows)==2
    assert obj.counts['REPEATED_CANDIDATE_SUPPRESSED']==2

def test_source_context_and_current_hand(lane):
    obj,provider,queue=lane;observe(obj,crop())
    row=queue.rows[0]
    assert (row.table_id,row.session_id,row.frame_id,row.source_pts,row.time_base_num,row.time_base_den)==(1,'S',10,5120,1,15360)
    obj.ocr_complete(row,{'status':'OK','raw_text':'Hand #2876718828'})
    assert provider.current[1]['hand_id']=='2876718828'
    assert obj.counts['HAND_HEADER_CONFIRMED']==1

def test_replacement_session_before_immediate_callback_and_old_result(lane):
    obj,provider,queue=lane;observe(obj,crop(),session='OLD');old=queue.rows[0]
    queue.callback=obj.ocr_complete
    observe(obj,crop(),11,'NEW')
    assert provider.current[1]['session_id']=='NEW'
    obj.ocr_complete(old,{'status':'OK','raw_text':'#2876719999'})
    assert provider.current[1]['session_id']=='NEW'
    assert obj.counts['STALE_SESSION_IGNORED']==1

def test_late_older_header_does_not_overwrite_newer(lane):
    obj,provider,queue=lane
    observe(obj,crop(),10);observe(obj,crop(digit=4),20)
    obj.ocr_complete(queue.rows[1],{'status':'OK','raw_text':'#2876719999'})
    obj.ocr_complete(queue.rows[0],{'status':'OK','raw_text':'Hand #2876718828'})
    assert provider.current[1]['hand_id']=='2876719999'
    assert obj.counts['STALE_HEADER_IGNORED']==1

def test_independent_pending_token_ordinary_error_cannot_clear_header(lane):
    obj,provider,queue=lane;observe(obj,crop());row=queue.rows[0]
    provider.note_ocr_submission(row)
    provider.ocr_complete(row,{'status':'ERROR','raw_text':''})
    assert provider.outstanding[(1,'S')]=={(10,1000000)}
    obj.ocr_complete(row,{'status':'OK','raw_text':'Hand #2876718828'})
    assert not provider.outstanding[(1,'S')]
    assert provider.source_barriers[(1,'S')]=={(10,0)}

@pytest.mark.parametrize('response',[{'status':'ERROR','raw_text':''},{'status':'OK','raw_text':'Hand #oops'},{'status':'OK','raw_text':'Dealer: posts Ante'}])
def test_unknown_header_candidate_is_barrier(lane,response):
    obj,provider,queue=lane;observe(obj,crop());obj.ocr_complete(queue.rows[0],response)
    assert not provider.current
    assert provider.source_barriers[(1,'S')]=={(10,0)}
    assert obj.counts['HEADER_OCR_UNKNOWN']==1
