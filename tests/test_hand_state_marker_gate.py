"""Source-bound hand marker gate tests; no whole CHAT lineage requirement."""
from collections import deque
from types import SimpleNamespace
from pathlib import Path
import hashlib,json
import numpy as np
import pytest
from parser_core.dealing_start_stack import DealingStartStackSidecar
from parser_core.hand_state_marker_gate import HandStateMarkerGate

ROOT=Path(__file__).resolve().parents[1]
class Model:
    def __init__(self,*args):self.status='cards'
    def classify(self,gray):return {'status':self.status,'gate_pass':self.status!='UNKNOWN','pixel_hash':hashlib.sha256(gray.tobytes()).hexdigest()}

@pytest.fixture
def sidecar(tmp_path,monkeypatch):
    monkeypatch.setattr('parser_core.dealing_start_stack.TwoStageModel',Model)
    monkeypatch.setattr('parser_core.dealing_start_stack.physical_spans',lambda gray:[(0,16)])
    flags=json.loads((ROOT/'configs/dealing_start_stack.json').read_text('utf-8'))
    flags.update(enable_dealing_cards_hand_state_trigger=True,enable_start_stack_capture=False,candidate_buffer_frames=3)
    obj=DealingStartStackSidecar(ROOT,tmp_path,flags,hand_state_gate=HandStateMarkerGate())
    yield obj
    obj.finish()

def observe(obj,frame,session='S',classification='UNRESOLVED'):
    obj.tracker_transition({'table':1,'frame_to':frame,'session_id':session,'classification':classification,'new_link_states':['UNKNOWN']*18})
    obj.observe_frame(1,session,frame,frame*512,[1,15360],np.zeros((404,847,3),np.uint8),np.full((16,228),100,np.uint8))

def header(obj,frame=10,hand='10200000',index=-1,session='S'):
    return obj.header(1,session,hand,frame,frame*512,index,[1,15360],'header')

@pytest.mark.parametrize('classification',['UNRESOLVED','LOCAL_CHANGE','INITIAL_VISIBLE'])
def test_confirmed_hand_is_independent_of_unrelated_row_links(sidecar,classification):
    header(sidecar,9);observe(sidecar,10,classification=classification);observe(sidecar,11,classification=classification)
    assert len(sidecar.triggers)==1 and sidecar.triggers[0]['hand_id']=='10200000'
    assert sidecar.counts['HAND_CONTEXT_COMPLETENESS_UNKNOWN']==0

def test_late_header_releases_buffered_source_candidate_only_after_callback(sidecar):
    sidecar.active_sessions[1]='S';sidecar.outstanding[(1,'S')].add((10,1000000))
    observe(sidecar,10);observe(sidecar,11);assert not sidecar.triggers
    header(sidecar,10);sidecar._release(1,'S');assert not sidecar.triggers
    sidecar.outstanding[(1,'S')].discard((10,1000000));sidecar._release(1,'S')
    assert sidecar.triggers[0]['trigger_source_frame']==10 and sidecar.triggers[0]['release_capture_frame']==11

def test_candidate_before_header_source_is_rejected(sidecar):
    observe(sidecar,10);sidecar.model.status='UNKNOWN';observe(sidecar,11);header(sidecar,11);sidecar._release(1,'S')
    assert not sidecar.triggers

def test_same_frame_cards_above_header_are_rejected(sidecar):
    header(sidecar,10,index=1);observe(sidecar,10);sidecar.model.status='UNKNOWN';observe(sidecar,11)
    assert not sidecar.triggers

def test_visible_header_order_is_enforced(sidecar):
    header(sidecar,9);sidecar.active_sessions[1]='S';sidecar.sealed[(1,'S')]=10
    candidate={'row_index':0,'row_bounds':[0,16],'pixel_hash':'cards'}
    sidecar.frames[(1,'S')].append({'source_frame':10,'source_pts':5120,'cards':[candidate],
      'rows':[candidate,{'row_bounds':[20,36],'pixel_hash':'header'}],'context_complete':False})
    sidecar._release(1,'S');assert not sidecar.triggers

def test_late_old_header_cannot_rebind_or_reopen_latch(sidecar):
    header(sidecar,20);observe(sidecar,20);observe(sidecar,21)
    assert header(sidecar,10,'10100000')=='STALE_HEADER_IGNORED'
    sidecar._release(1,'S');assert len(sidecar.triggers)==1 and sidecar.current[1]['hand_id']=='10200000'

def test_old_session_result_cannot_rebind(sidecar):
    observe(sidecar,10,'A');observe(sidecar,11,'B');header(sidecar,11,session='B')
    assert header(sidecar,12,'99900000',session='A')=='STALE_SESSION_IGNORED'
    assert sidecar.current[1]['session_id']=='B'

def test_duplicate_header_does_not_move_earliest_hand_anchor(sidecar):
    sidecar.outstanding[(1,'S')].add((10,1000000))
    header(sidecar,10);observe(sidecar,10);sidecar.model.status='UNKNOWN';observe(sidecar,11)
    assert not sidecar.triggers;header(sidecar,11,index=-1)
    sidecar.outstanding[(1,'S')].discard((10,1000000))
    sidecar._release(1,'S')
    assert sidecar.triggers[0]['source_frame']==10 and sidecar.triggers[0]['trigger_source_frame']==10

def test_ordinary_pending_and_failed_ocr_do_not_block_hand_marker(sidecar):
    header(sidecar,9);sidecar.outstanding[(1,'S')].add((10,2));sidecar.source_barriers[(1,'S')].add((10,3))
    observe(sidecar,10);observe(sidecar,11)
    assert len(sidecar.triggers)==1

def test_unknown_header_barrier_requires_new_confirmed_anchor(sidecar):
    header(sidecar,9);sidecar.hand_state_gate.header_unknown(SimpleNamespace(table_id=1,session_id='S',frame_id=10,source_span_index=1))
    observe(sidecar,10);observe(sidecar,11);assert not sidecar.triggers
    header(sidecar,12,'10300000');observe(sidecar,12);observe(sidecar,13)
    assert len(sidecar.triggers)==1 and sidecar.triggers[0]['hand_id']=='10300000'

def test_unknown_ncc_cannot_trigger(sidecar):
    header(sidecar,9);sidecar.model.status='UNKNOWN';observe(sidecar,10);observe(sidecar,11)
    assert not sidecar.triggers

def test_ambiguous_two_cards_cannot_trigger(sidecar):
    header(sidecar,9);sidecar.active_sessions[1]='S';sidecar.sealed[(1,'S')]=10
    c={'row_index':2,'row_bounds':[40,56],'pixel_hash':'cards'}
    sidecar.frames[(1,'S')].append({'source_frame':10,'source_pts':5120,'cards':[c,dict(c,row_index=3)],'rows':[]})
    sidecar._release(1,'S');assert not sidecar.triggers

def test_once_per_hand_is_unchanged(sidecar):
    header(sidecar,9)
    for f in range(10,16):observe(sidecar,f)
    header(sidecar,15);sidecar._release(1,'S')
    assert len(sidecar.triggers)==1

def test_check_fold_do_not_close_and_call_still_closes_existing_window(sidecar):
    header(sidecar,9);observe(sidecar,10);observe(sidecar,11)
    row=SimpleNamespace(table_id=1,session_id='S',frame_id=12,source_span_index=2,source_pts=6144,
      time_base_num=1,time_base_den=15360,pixel_sha256='action')
    window=sidecar.windows[(1,'S','10200000')]
    for text in ('Dealer: player checks','Dealer: player folds'):
        sidecar.ocr_complete(row,{'status':'OK','raw_text':text});assert window.closed is None
    sidecar.ocr_complete(row,{'status':'OK','raw_text':'Dealer: player calls 100'})
    assert window.closed==12 and len(sidecar.triggers)==1

def test_expired_candidate_is_not_recovered(sidecar):
    observe(sidecar,10);sidecar.model.status='UNKNOWN'
    for f in range(11,15):observe(sidecar,f)
    header(sidecar,10);sidecar._release(1,'S');assert not sidecar.triggers

def test_conflicting_new_header_resets_unknown(sidecar):
    header(sidecar,10);assert header(sidecar,11,'10100000')=='HEADER_CONFLICT_UNKNOWN'
    observe(sidecar,11);observe(sidecar,12);assert not sidecar.triggers

def test_conflict_fence_survives_current_reset_and_late_old_callback(sidecar):
    header(sidecar,30,'10200000')
    assert header(sidecar,31,'10100000')=='HEADER_CONFLICT_UNKNOWN'
    assert 1 not in sidecar.current
    assert header(sidecar,10,'10000000')=='STALE_HEADER_IGNORED'
    assert 1 not in sidecar.current
    assert header(sidecar,32,'10000000')=='HEADER_CONFLICT_UNKNOWN'
    observe(sidecar,32);assert not sidecar.triggers
    assert header(sidecar,33,'10300000')=='HAND_HEADER_CONFIRMED'
    observe(sidecar,33);observe(sidecar,34)
    assert len(sidecar.triggers)==1 and sidecar.triggers[0]['hand_id']=='10300000'

def test_same_id_post_conflict_resync_cannot_reuse_pre_conflict_cards(sidecar):
    header(sidecar,10,'10200000');observe(sidecar,10)
    assert header(sidecar,11,'10100000')=='HEADER_CONFLICT_UNKNOWN'
    sidecar.model.status='UNKNOWN';observe(sidecar,11)
    assert header(sidecar,12,'10200000')=='HAND_HEADER_CONFIRMED'
    sidecar._release(1,'S');assert not sidecar.triggers
    sidecar.model.status='cards';observe(sidecar,12);observe(sidecar,13)
    assert len(sidecar.triggers)==1 and sidecar.triggers[0]['trigger_source_frame']==12
    assert sidecar.triggers[0]['source_frame']==12

@pytest.mark.parametrize('frame,pts',[(10,5121),(11,5000)])
def test_contradictory_pts_frame_source_order_is_rejected(sidecar,frame,pts):
    header(sidecar,10);sidecar.active_sessions[1]='S';sidecar.sealed[(1,'S')]=frame
    c={'row_index':2,'row_bounds':[40,56],'pixel_hash':'cards'}
    sidecar.frames[(1,'S')].append({'source_frame':frame,'source_pts':pts,'time_base':[1,15360],'cards':[c],'rows':[c]})
    sidecar._release(1,'S')
    assert not sidecar.triggers and sidecar.hand_state_gate.counts['SOURCE_ORDER_CONFLICT']==1

def test_equivalent_rational_time_bases_are_accepted(sidecar):
    sidecar.header(1,'S','10200000',10,10240,-1,[1,30720],'header')
    observe(sidecar,10);observe(sidecar,11)
    assert len(sidecar.triggers)==1

def test_38_saved_candidates_with_delayed_source_bound_headers(sidecar):
    audit=ROOT/'runs/hand_context_completeness_audit_20261003/all_38_hands.json'
    if not audit.exists():pytest.skip('saved diagnostic sample absent')
    groups=json.loads(audit.read_text('utf-8'))['hands'];evidence=[]
    for g in sorted(groups,key=lambda g:(g['table'],g['header_source_frame'])):
        t=g['table'];sid=f'T{t}';c=g['first_cards_receipt'];f=c['source_frame'];hf=g['header_source_frame'];idx=g['header_source_span_index']
        sidecar.active_sessions[t]=sid;sidecar.outstanding[(t,sid)].add((hf,1000000+idx))
        sidecar.frames[(t,sid)].append({'source_frame':f,'source_pts':c['source_pts'],'time_base':c['time_base'],'cards':[c],'rows':[c],'context_complete':False})
        delay=2 if g['hand_id']=='2876720526' else 1
        sidecar.sealed[(t,sid)]=f;sidecar.counts['last_capture_frame']=f+delay
        before=len(sidecar.triggers);sidecar._release(t,sid);assert len(sidecar.triggers)==before
        sidecar.header(t,sid,g['hand_id'],hf,g['header_source_pts'],idx,[1,15360],g['header_pixel_sha256'])
        sidecar.outstanding[(t,sid)].discard((hf,1000000+idx));sidecar._release(t,sid)
        assert len(sidecar.triggers)==before+1
        ev=sidecar.triggers[-1];assert ev['hand_id']==g['hand_id'] and ev['trigger_source_frame']==f
        sidecar._release(t,sid);assert len(sidecar.triggers)==before+1
        evidence.append({'table':t,'hand_id':g['hand_id'],'candidate_source_frame':f,'synthetic_callback_frame':f+delay,'result':'HIT'})
    assert len(evidence)==38 and len(sidecar.triggers)==38
    out=ROOT/'runs/hand_state_marker_gate_20261003';out.mkdir(parents=True,exist_ok=True)
    (out/'saved_38_candidate_gate_replay.json').write_text(json.dumps({'scope':'SYNTHETIC_CALLBACK_SCHEDULE_WITH_SAVED_REAL_NCC_CANDIDATES_NOT_LIVE_ACCEPTANCE','HITS':38,'MISSES':0,'FALSE_POSITIVES':'NOT_MEASURED','DUPLICATE_TRIGGERS':0,'hands':evidence},indent=2),'utf-8')
