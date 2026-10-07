"""Sidecar isolation, asynchronous source order, and clean stack contracts."""
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
import hashlib,json,threading
from pathlib import Path
import numpy as np
import pytest
from parser_core.dealing_start_stack import CleanStackWindow,DealingStartStackSidecar
from parser_core.parallel_ocr import ParallelRowOCR,row_key

ROOT=Path(__file__).resolve().parents[1]

@dataclass
class Row:
    table_id:int=1
    frame_id:int=10
    source_span_index:int=1
    session_id:str='S'
    source_pts:int=5120
    time_base_num:int=1
    time_base_den:int=15360
    pixel_sha256:str='header'
    pixels:object=None

class Model:
    def __init__(self,*args):self.status='cards'
    def classify(self,gray):
        return {'status':self.status,'gate_pass':self.status!='UNKNOWN','pixel_hash':hashlib.sha256(gray.tobytes()).hexdigest(),'scores':{'cards':.99,'Turn':.6,'Flop':.5,'River':.6} if self.status!='UNKNOWN' else None}

@pytest.fixture
def sidecar(tmp_path,monkeypatch):
    monkeypatch.setattr('parser_core.dealing_start_stack.TwoStageModel',Model)
    monkeypatch.setattr('parser_core.dealing_start_stack.physical_spans',lambda gray:[(0,16)])
    flags=json.loads((ROOT/'configs/dealing_start_stack.json').read_text(encoding='utf-8'));flags.update(enable_dealing_cards_hand_state_trigger=True,enable_start_stack_capture=False,candidate_buffer_frames=2)
    obj=DealingStartStackSidecar(ROOT,tmp_path,flags)
    yield obj
    obj.finish()

def observe(obj,frame,session='S'):
    obj.tracker_transition({'table_id':1,'session_id':session,'frame_id':frame,'classification':'GLOBAL_SCROLL'})
    obj.observe_frame(1,session,frame,frame*512,[1,15360],np.zeros((404,847,3),np.uint8),np.full((16,228),100,np.uint8))

def ocr(obj,text,frame,index=1,session='S'):
    obj.ocr_complete(Row(frame_id=frame,source_span_index=index,session_id=session,source_pts=frame*512),{'status':'OK','raw_text':text,'score':.99})

def test_late_old_header_does_not_overwrite_new_hand_or_reopen_latch(sidecar):
    observe(sidecar,200);ocr(sidecar,'Hand #10200000',200,index=-1);observe(sidecar,201)
    assert len(sidecar.triggers)==1
    fired=set(sidecar.fired);ocr(sidecar,'Hand #10100000',100)
    assert sidecar.current[1]['hand_id']=='10200000' and sidecar.fired==fired
    assert sidecar.counts['STALE_HEADER_IGNORED']==1

def test_hand_and_bare_number_same_hand_are_not_conflicts(sidecar):
    ocr(sidecar,'Hand #10200000',10,1);ocr(sidecar,'#10200000',10,3)
    assert sidecar.current[1]['hand_id']=='10200000'
    assert sidecar.counts['HEADER_CONFLICT_UNKNOWN']==0
    assert len(sidecar.histories[(1,'S')])==1

def test_old_session_late_header_is_ignored(sidecar):
    observe(sidecar,100,'A');observe(sidecar,101,'B');ocr(sidecar,'Hand #10200000',101,-1,'B')
    before=dict(sidecar.current[1]);ocr(sidecar,'Hand #99900000',102,1,'A')
    assert sidecar.current[1]==before
    assert sidecar.counts['STALE_SESSION_IGNORED']==1

def test_unknown_hand_cannot_publish_trigger_or_deal(sidecar):
    observe(sidecar,10)
    assert not sidecar.triggers and not sidecar.windows
    assert sidecar.confirm_deal_complete((1,'S','10200000'),{})=='HAND_TRIGGER_UNKNOWN'

def test_first_confirmed_cards_triggers_once_through_later_frames(sidecar):
    observe(sidecar,10);ocr(sidecar,'Hand #10200000',10,-1)
    for frame in range(11,16):observe(sidecar,frame)
    ocr(sidecar,'#10200000',15,0)
    assert len(sidecar.triggers)==1 and sidecar.triggers[0]['trigger_source_frame']==10
    assert sidecar.windows[(1,'S','10200000')].deal is None

def test_expired_candidate_cannot_be_reconstructed_as_a_trigger(sidecar):
    observe(sidecar,10);sidecar.model.status='UNKNOWN';observe(sidecar,11);observe(sidecar,12)
    ocr(sidecar,'Hand #10200000',10,-1)
    assert sidecar.counts['NCC_CANDIDATE_BUFFER_EXPIRED']==1
    assert not sidecar.triggers

def test_pending_new_header_blocks_old_hand_until_callback_completes(sidecar):
    ocr(sidecar,'Hand #10100000',9)
    pending=Row(frame_id=19,source_span_index=1,source_pts=19*512)
    sidecar.note_ocr_submission(pending)
    observe(sidecar,20);observe(sidecar,21)
    assert not sidecar.triggers
    sidecar.ocr_complete(pending,{'status':'OK','raw_text':'Hand #10200000','score':.99})
    assert len(sidecar.triggers)==1
    assert sidecar.triggers[0]['hand_id']=='10200000'
    assert sidecar.triggers[0]['trigger_source_frame']==20

def test_tracker_unknown_context_cannot_admit_a_cards_marker(sidecar):
    ocr(sidecar,'Hand #10200000',9)
    sidecar.tracker_transition({'table_id':1,'session_id':'S','frame_id':10,'classification':'UNRESOLVED'})
    pixels=np.full((16,228),100,np.uint8);rgb=np.zeros((404,847,3),np.uint8)
    sidecar.observe_frame(1,'S',10,5120,[1,15360],rgb,pixels)
    sidecar.observe_frame(1,'S',11,5632,[1,15360],rgb,pixels)
    assert not sidecar.triggers and sidecar.counts['HAND_CONTEXT_COMPLETENESS_UNKNOWN']

def test_failed_new_header_result_does_not_release_old_hand_marker(sidecar):
    ocr(sidecar,'Hand #10100000',9)
    pending=Row(frame_id=19,source_span_index=1,source_pts=19*512)
    sidecar.note_ocr_submission(pending);observe(sidecar,20);observe(sidecar,21)
    assert not sidecar.triggers
    sidecar.ocr_complete(pending,{'status':'ERROR','raw_text':'Hand #10200000','score':None})
    assert not sidecar.triggers

def test_failed_header_requires_independent_later_header_resync(sidecar):
    ocr(sidecar,'Hand #10100000',9);observe(sidecar,10)
    sidecar.ocr_complete(Row(frame_id=10),{'status':'ERROR','raw_text':'','score':None})
    observe(sidecar,11)
    assert not sidecar.triggers
    ocr(sidecar,'Hand #10200000',12,-1);observe(sidecar,12);observe(sidecar,13)
    assert len(sidecar.triggers)==1 and sidecar.triggers[0]['hand_id']=='10200000'

def test_initial_visible_context_is_not_complete(sidecar):
    ocr(sidecar,'Hand #10200000',9)
    pixels=np.full((16,228),100,np.uint8);rgb=np.zeros((404,847,3),np.uint8)
    for frame in (10,11):
        sidecar.tracker_transition({'table_id':1,'session_id':'S','frame_id':frame,'classification':'INITIAL_VISIBLE'})
        sidecar.observe_frame(1,'S',frame,frame*512,[1,15360],rgb,pixels)
    assert not sidecar.triggers

def test_second_cards_episode_captures_before_second_header_ocr(sidecar):
    sidecar.flags.update(enable_start_stack_capture=True,capture_frames=1)
    ocr(sidecar,'Hand #10100000',9)
    observe(sidecar,10);observe(sidecar,11)
    first_crops=sidecar.counts['ROI_CROPS'];first_episode=sidecar.capture_episodes[(1,'S')]
    sidecar.model.status='UNKNOWN';observe(sidecar,12);observe(sidecar,13)
    before=sidecar.counts['ROI_CROPS'];sidecar.model.status='cards';observe(sidecar,14)
    assert sidecar.current[1]['hand_id']=='10100000'
    assert sidecar.counts['ROI_CROPS']>before and first_crops>0
    assert sidecar.capture_episodes[(1,'S')]==first_episode+1

def test_money_action_before_delayed_header_is_replayed_into_window(sidecar):
    observe(sidecar,10)
    ocr(sidecar,'Dealer: Alice calls 1000',12)
    ocr(sidecar,'Hand #10200000',10,-1);observe(sidecar,13)
    assert sidecar.windows[(1,'S','10200000')].closed==12

def test_actual_first_stage_native_tracker_receipt_adapter(sidecar):
    from parser_core.ram_first_stage import FirstStage
    stage=FirstStage(ROOT/'video/smoke_17.mp4',{'1':'S'},2,root=ROOT,tracker_observer=sidecar.tracker_transition)
    pixels=np.zeros((100,228),np.uint8)
    list(stage.chat_rows(pixels,frame_id=0,table_id=1,session_id='S',source_pts=0))
    proof=sidecar.context_proofs[(1,'S')]
    assert proof['frame']==0 and not proof['complete']
    assert proof['classification']=='INITIAL_VISIBLE'
    list(stage.chat_rows(pixels,frame_id=1,table_id=1,session_id='S',source_pts=512))
    assert sidecar.context_proofs[(1,'S')]['frame']==1
    assert not sidecar.triggers

def test_real_registered_ncc_templates_recognize_source_literal():
    import cv2
    from parser_core.dealing_cards_ncc import TwoStageModel
    model=TwoStageModel(ROOT/'configs/dealing_cards_ncc.json')
    source=Path('C:/ParserData/runs/smoke14_full_tiny_ssd_20260929_v3/crops/rows/table_95984db438dc55f85dee9067/000000681_018.png')
    gray=cv2.imdecode(np.fromfile(source,dtype=np.uint8),0)
    result=model.classify(gray)
    assert result['gate_pass'] and result['status']=='cards'
    assert set(result['scores'])=={'cards','Turn','Flop','River'}
    assert result['margin']>=model.config['margin_threshold']

def test_callback_failure_drains_queue_and_cannot_replace_raw_output():
    closed=[]
    class Client:
        def read_batch(self,items,batch_size):return {'results':[{'item_id':i['item_id'],'status':'OK','raw_text':'original','score':.99} for i in items]}
        def close(self):closed.append(True)
    def callback(row,response):
        response['raw_text']='mutated'
        raise RuntimeError('sidecar failure')
    queue=ParallelRowOCR(lambda _:Client(),workers=1,batch_size=1,result_observer=callback)
    row=Row(pixels=np.zeros((16,228),np.uint8));queue.submit(row)
    finish=threading.Thread(target=queue.finish,daemon=True);finish.start();finish.join(2)
    assert not finish.is_alive() and queue.jobs.unfinished_tasks==0
    assert queue.results[row_key(row)]['raw_text']=='original'
    assert any('sidecar_error' in e for e in queue.errors) and closed==[True]

def proof():return {'evidence_status':'PROVEN','table_pixel_sha256':'table-sha','occupied_seats':[1],'source_frame':10}

def opened():
    w=CleanStackWindow(.95);assert w.confirm_deal(proof())=='DEAL_COMPLETE';return w

def read(w,frame,text='10 000',confidence=.99,forced=None,epoch='E1'):
    return w.reading(1,frame,text,confidence,{'ANTE':1750} if forced is None else forced,epoch,forced_bets_proven=True,occupancy_proven=True)

def test_table_proof_required_before_stack_reading():
    w=CleanStackWindow();assert w.confirm_deal({'evidence_status':'UNKNOWN'})=='DEAL_COMPLETE_UNKNOWN'
    assert read(w,11)[1]=='DEAL_COMPLETE_UNKNOWN'

@pytest.mark.parametrize('kind',['CHECK','FOLD'])
def test_check_and_fold_do_not_close_window(kind):
    w=opened();w.action(kind,11)
    for f in (11,12,13):result,status=read(w,f)
    assert status=='START_STACK_RECOVERED' and w.closed is None

@pytest.mark.parametrize('kind',['CALL','BET','RAISE','RAISE_TO','ALL-IN','ALL_IN'])
def test_money_action_closes_at_its_source_frame(kind):
    w=opened();read(w,11);read(w,12);w.action(kind,13)
    assert read(w,13)[1]=='WINDOW_CLOSED_OR_TIMING_UNKNOWN' and not w.results

def test_three_identical_consecutive_reads_strict_confidence_forced_only():
    w=opened();assert read(w,11)[0] is None;assert read(w,12)[0] is None
    result,status=read(w,13,forced={'ANTE':1750,'SB':7000})
    assert status=='START_STACK_RECOVERED'
    assert result['snapshot_frame']==11 and result['confirmation_frame']==13
    assert result['start_stack']=='18750' and result['displayed_stack']=='10000'

def test_low_confidence_nonnumeric_or_frame_gap_resets_stability():
    w=opened();read(w,11);assert read(w,12,confidence=.949)[0] is None
    assert read(w,13,text='ANTE')[0] is None
    assert read(w,14)[0] is None;assert read(w,16)[0] is None;assert read(w,17)[0] is None
    assert read(w,18)[0]['snapshot_frame']==16

@pytest.mark.parametrize('forced',[{}, {'CALL':1000},{'ANTE':-1},{'ANTE':True},{'ANTE':None}])
def test_unknown_or_nonforced_commitments_cannot_recover(forced):
    w=opened();read(w,11);read(w,12)
    assert read(w,13,forced=forced)[1]=='FORCED_BET_UNKNOWN' and not w.results

def test_missing_occupancy_blocks_recovery():
    w=opened()
    assert read(w,11,epoch=None)[1]=='IDENTITY_UNKNOWN'

def test_occupancy_and_forced_bet_values_need_independent_proof():
    w=opened()
    assert w.reading(1,11,'10000',.99,{'ANTE':1750},'E1')[1]=='IDENTITY_UNKNOWN'
    for f in (11,12,13):result,status=w.reading(1,f,'10000',.99,{'ANTE':1750},'E1',occupancy_proven=True)
    assert result is None and status=='FORCED_BET_UNKNOWN'

def test_stability_cannot_cross_player_occupancy_epoch():
    w=opened();read(w,11,epoch='E1');assert read(w,12,epoch='E2')[0] is None
    assert read(w,13,epoch='E2')[0] is None
    assert read(w,14,epoch='E2')[0]['snapshot_frame']==12

def test_late_money_action_invalidates_same_frame_confirmed_result():
    w=opened()
    for f in (11,12,13):read(w,f)
    assert w.results
    w.action('CALL',13)
    assert not w.results and read(w,14)[0] is None
