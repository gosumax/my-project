"""Native-pixel capture, stale-context and lifetime guards for EMPTY routing."""
import copy
import hashlib
import json
import threading
from collections import Counter,defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from parser_core.empty_seat_evidence import EmptySeatButtonModel
from parser_core.empty_fast_capture import install_empty_capture
from parser_core.dealing_start_stack import CleanStackWindow
from parser_core.start_stack_live import StartStackLiveConnection

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def native():
    folder=ROOT/'tests/fixtures/native_empty_deal'
    image=cv2.imdecode(np.frombuffer((folder/'T06_frame483_table.png').read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    assert image is not None
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def owner():
    model=EmptySeatButtonModel(ROOT);k=(6,'NATIVE_TEST','12345678')
    header=dict(table_id=k[0],session_id=k[1],hand_id=k[2],source_frame=480,
        source_position=[480,0],source_pts=480*512,time_base=[1,15360],header_pixel_sha256='a'*64)
    side=SimpleNamespace(lock=threading.RLock(),current={6:header},active_sessions={6:k[1]},
        windows={k:CleanStackWindow(.95)},header_fences={k[:2]:dict(hand_id=k[2],source_position=[480,0],unknown=False)},
        flags=dict(capture_frames=64),outstanding=defaultdict(set),hand_state_gate=SimpleNamespace(failed_headers=defaultdict(set)))
    return SimpleNamespace(root=ROOT,lock=threading.RLock(),sidecar=side,empty_button_model=model,
        closed=False,states={},test_key=k)

def source(obj,frame=483):
    table,session,hand=obj.test_key
    return dict(table_id=table,session_id=session,hand_id=hand,source_frame=frame,
        source_pts=frame*512,time_base=[1,15360])

def capture(obj,pixels,frame=483,meta=None,boxes=None):
    with obj.sidecar.lock,obj.lock:
        return obj.empty_button_model.capture(pixels,meta or source(obj,frame),
            boxes or obj.empty_button_model.slots['slot_06']['seats'])

def baseline(pixels,meta,boxes=None):
    return EmptySeatButtonModel(ROOT).capture(pixels,meta,boxes or EmptySeatButtonModel(ROOT).slots['slot_06']['seats'])

def test_feature_off_is_literal_original_methods(native):
    obj=owner();model=obj.empty_button_model
    assert install_empty_capture(obj,enabled=False) is None
    assert 'capture' not in vars(model) and 'classify' not in vars(model)
    assert capture(obj,native)==baseline(native,source(obj))


def test_capture_does_not_populate_owner_gate_maps(native):
    obj=owner();hook=install_empty_capture(obj,enabled=True)
    try:
        assert not obj.sidecar.outstanding and not obj.sidecar.hand_state_gate.failed_headers
        capture(obj,native)
        assert not obj.sidecar.outstanding and not obj.sidecar.hand_state_gate.failed_headers
    finally:hook.close()

def test_exact_roi_reuse_keeps_fresh_source_png_and_old_verify(native):
    obj=owner();hook=install_empty_capture(obj,enabled=True)
    try:
        first=capture(obj,native);second=capture(obj,native,484)
        assert first==baseline(native,source(obj)) and second==baseline(native,source(obj,484))
        assert hook.dispatcher.counts['exact_hits']==8
        before=hook.dispatcher.counts['observations']
        assert all(obj.empty_button_model.verify(row,source(obj,484)) for row in second.values() if row['status']=='EMPTY')
        assert hook.dispatcher.counts['observations']==before and hook.counts['old_non_capture_calls']>0
        assert all(row['source_frame']==484 for row in second.values())
        assert all(len(values)<=2 for values in hook.dispatcher.history.values())
    finally:hook.close()

def test_constant_gray_sad_reject_and_unknown_are_exact(native):
    obj=owner();hook=install_empty_capture(obj,enabled=True);pixels=np.zeros_like(native)
    try:
        result=capture(obj,pixels)
        assert result==baseline(pixels,source(obj))
        assert all(row['status']=='UNKNOWN' for row in result.values())
        assert hook.dispatcher.counts['bound_rejects']==8 and hook.dispatcher.counts['heavy_old_calls']==0
    finally:hook.close()

def test_changed_positive_roi_falls_back_instead_of_reusing(native):
    obj=owner();hook=install_empty_capture(obj,enabled=True)
    try:
        first=capture(obj,native);changed=native.copy();seat=next(s for s,r in first.items() if r['status']=='EMPTY')
        x,y,X,Y=first[seat]['roi'];changed[y+10,x+10,:]^=255
        from parser_core.ram_first_stage import png_compatible_gray
        assert not np.array_equal(png_compatible_gray(native[y:Y,x:X]),png_compatible_gray(changed[y:Y,x:X]))
        old=hook.dispatcher.counts['heavy_old_calls'];hits=hook.dispatcher.counts['exact_hits']
        result=capture(obj,changed,484)
        assert result==baseline(changed,source(obj,484))
        assert hook.dispatcher.counts['heavy_old_calls']>old
        assert hook.dispatcher.counts['exact_hits']-hits<8
    finally:hook.close()

@pytest.mark.parametrize('change',[
    'wrong_hand','wrong_session','wrong_table','missing_hand','bad_table','missing_pts','bool_frame',
    'bad_timebase','stale_frame','stale_pts','changed_geometry','pending_header','failed_header',
    'unknown_fence','wrong_fence','missing_window','closed_window','deal_window','owner_replaced','unlocked','budget_expired'])
def test_unsupported_source_uses_old_and_discards_stale_cache(native,change):
    obj=owner();hook=install_empty_capture(obj,enabled=True)
    try:
        capture(obj,native);meta=source(obj,484);boxes=copy.deepcopy(obj.empty_button_model.slots['slot_06']['seats'])
        if change=='wrong_hand':meta['hand_id']='12345679'
        if change=='wrong_session':meta['session_id']='OTHER'
        if change=='wrong_table':meta['table_id']=5
        if change=='missing_hand':del meta['hand_id']
        if change=='bad_table':meta['table_id']=[]
        if change=='missing_pts':del meta['source_pts']
        if change=='bool_frame':meta['source_frame']=True
        if change=='bad_timebase':meta['time_base']=[0,15360]
        if change=='stale_frame':meta.update(source_frame=482,source_pts=482*512)
        if change=='stale_pts':meta['source_pts']=479*512
        if change=='changed_geometry':boxes['seat1']['nick'][0]+=1
        if change=='pending_header':obj.sidecar.outstanding[obj.test_key[:2]].add((483,1000000))
        if change=='failed_header':obj.sidecar.hand_state_gate.failed_headers[obj.test_key[:2]].add((481,0))
        if change=='unknown_fence':obj.sidecar.header_fences[obj.test_key[:2]]['unknown']=True
        if change=='wrong_fence':obj.sidecar.header_fences[obj.test_key[:2]]['source_position']=[481,0]
        if change=='missing_window':obj.sidecar.windows.clear()
        if change=='closed_window':obj.sidecar.windows[obj.test_key].closed=484
        if change=='deal_window':obj.sidecar.windows[obj.test_key].deal=dict(source_frame=484)
        if change=='owner_replaced':obj.empty_button_model=EmptySeatButtonModel(ROOT)
        if change=='budget_expired':meta.update(source_frame=545,source_pts=545*512)
        before=hook.dispatcher.counts['observations']
        with obj.sidecar.lock,obj.lock:
            if change=='unlocked':pass
            else:result=hook.model.capture(native,meta,boxes)
        if change=='unlocked':result=hook.model.capture(native,meta,boxes)
        assert result==baseline(native,meta,boxes)
        assert hook.dispatcher.counts['observations']==before
        if change=='wrong_table':
            assert all(slot[0]==6 for slot in hook.dispatcher.entries) and 6 in hook.contexts
            hits=hook.dispatcher.counts['exact_hits']
            assert capture(obj,native,485)==baseline(native,source(obj,485))
            assert hook.dispatcher.counts['exact_hits']==hits+8
        else:assert not hook.dispatcher.entries
    finally:hook.close()

@pytest.mark.parametrize('change',['window_replace','same_hand_header','session_replace','hand_replace'])
def test_context_epoch_change_cannot_reuse_previous_roi(native,change):
    obj=owner();hook=install_empty_capture(obj,enabled=True)
    try:
        capture(obj,native);old=obj.test_key;side=obj.sidecar;header=side.current[6]
        if change=='window_replace':side.windows[old]=CleanStackWindow(.95)
        if change=='same_hand_header':
            header.update(source_frame=484,source_pts=484*512,source_position=[484,0])
            side.header_fences[old[:2]]['source_position']=[484,0]
        if change in ('session_replace','hand_replace'):
            header.update(session_id='NEW' if change=='session_replace' else old[1],
                          hand_id='12345679' if change=='hand_replace' else old[2])
            obj.test_key=(6,header['session_id'],header['hand_id']);side.active_sessions[6]=obj.test_key[1]
            side.windows[obj.test_key]=CleanStackWindow(.95)
            side.header_fences[obj.test_key[:2]]=dict(hand_id=obj.test_key[2],source_position=header['source_position'],unknown=False)
        before=hook.dispatcher.counts['exact_hits'];meta=source(obj,485)
        assert capture(obj,native,485)==baseline(native,meta)
        assert hook.dispatcher.counts['exact_hits']==before
        assert len(hook.contexts)<=1 and len(hook.dispatcher.entries)<=8
    finally:hook.close()

def test_a_b_a_missing_cache_and_result_copy_do_not_grant_stale_hit(native):
    obj=owner();hook=install_empty_capture(obj,enabled=True)
    try:
        first=capture(obj,native);first['seat1']['status']='FABRICATED'
        capture(obj,np.zeros_like(native),484);hits=hook.dispatcher.counts['exact_hits']
        assert capture(obj,native,485)==baseline(native,source(obj,485))
        assert hook.dispatcher.counts['exact_hits']==hits and hook.dispatcher.counts['A_B_A']>0
        hook.dispatcher.entries.clear()
        assert capture(obj,native,486)==baseline(native,source(obj,486))
    finally:hook.close()

@pytest.mark.parametrize('failure',['model_mutation','cached_pixels'])
def test_integrity_violation_is_not_silently_old(native,failure):
    obj=owner();hook=install_empty_capture(obj,enabled=True)
    try:
        capture(obj,native)
        if failure=='model_mutation':hook.model.model['full_ncc_min']+=.001
        else:
            slot=next(iter(hook.dispatcher.entries));entry=hook.dispatcher.entries[slot]
            hook.dispatcher.entries[slot]=replace(entry,pixels=np.zeros_like(entry.pixels))
        with pytest.raises(ValueError,match='MUTATED|CONFLICT'):
            capture(obj,native,484)
    finally:hook.close()

def test_sad_exact_upper_bound_boundary_uses_old():
    from off_empty_bound import BoundedSadDispatcher,ROUNDOFF_SLACK
    gray=np.arange(64,dtype=np.uint8).reshape(8,8);calls=[]
    model=SimpleNamespace(sha256='BOUNDARY_TEST',templates=[('native',gray)],
        model=dict(translation_radius=0,full_ncc_min=1.+ROUNDOFF_SLACK))
    old=lambda m,p: calls.append(p.copy()) or dict(status='OLD_BOUNDARY')
    assert BoundedSadDispatcher(old).decide(model,gray)==dict(status='OLD_BOUNDARY')
    assert len(calls)==1

def test_finish_failure_always_restores_original_and_releases_cache(native):
    obj=owner();hook=install_empty_capture(obj,enabled=True);capture(obj,native)
    obj.empty_fast_hook=hook
    def fail():raise RuntimeError('OCR_DRAIN_TEST_FAILURE')
    obj._finish=fail
    with pytest.raises(RuntimeError,match='OCR_DRAIN_TEST_FAILURE'):
        StartStackLiveConnection.finish(obj)
    assert hook.closed and not hook.contexts and not hook.dispatcher.entries and not hook.dispatcher.history
    assert 'capture' not in vars(hook.model) and 'classify' not in vars(hook.model)
    hook.close()


def test_two_table_cache_isolation(native):
    obj=owner();side=obj.sidecar;key=(5,'SECOND','12345679')
    side.current[5]=dict(side.current[6],table_id=5,session_id=key[1],hand_id=key[2])
    side.active_sessions[5]=key[1];side.windows[key]=CleanStackWindow(.95)
    side.header_fences[key[:2]]=dict(hand_id=key[2],source_position=[480,0],unknown=False)
    hook=install_empty_capture(obj,enabled=True)
    try:
        capture(obj,native)
        meta=dict(source(obj),table_id=5,session_id=key[1],hand_id=key[2])
        boxes=hook.model.slots['slot_05']['seats']
        with side.lock,obj.lock:hook.model.capture(native,meta,boxes)
        assert {slot[0] for slot in hook.dispatcher.entries}=={5,6}
        side.windows[key].closed=484
        with side.lock,obj.lock:hook.sync(5,key[1],484)
        assert {slot[0] for slot in hook.dispatcher.entries}=={6}
        before=hook.dispatcher.counts['exact_hits'];capture(obj,native,484)
        assert hook.dispatcher.counts['exact_hits']==before+8
    finally:hook.close()


def test_capture_exception_releases_tls_before_next_capture(native):
    obj=owner();hook=install_empty_capture(obj,enabled=True);original=hook.original_capture
    try:
        def fail(*args):raise RuntimeError('CAPTURE_TEST_FAILURE')
        hook.original_capture=fail
        with pytest.raises(RuntimeError,match='CAPTURE_TEST_FAILURE'):capture(obj,native)
        assert hook.local.capture is None
        hook.original_capture=original
        assert capture(obj,native,484)==baseline(native,source(obj,484))
    finally:hook.close()


@pytest.mark.parametrize('kind',['shape','dtype','clock_reversal'])
def test_unsupported_pixels_and_reversed_clock_use_old(native,kind):
    obj=owner();hook=install_empty_capture(obj,enabled=True)
    try:
        capture(obj,native);before=hook.dispatcher.counts['observations']
        pixels=native[:403] if kind=='shape' else native.astype(np.float32) if kind=='dtype' else native
        meta=source(obj,484)
        if kind=='clock_reversal':meta['source_pts']=482*512
        assert capture(obj,pixels,meta=meta)==baseline(pixels,meta)
        assert hook.dispatcher.counts['observations']==before and not hook.dispatcher.entries
    finally:hook.close()
