"""Promoted tracker lifecycle and conservative downstream bridge contracts."""
from pathlib import Path
import zlib
import numpy as np
import pytest
from parser_core.chat_tracker import AdjacentChatTracker
from parser_core.ram_first_stage import FirstStage
from parser_core.temporal_overlap import TemporalObservations


def chat():
    rng = np.random.default_rng(104)
    pixels = np.zeros((360,228),np.uint8)
    for y in range(8,360,20):
        pixels[y:y+8,12:190] = rng.integers(80,240,(8,178),dtype=np.uint8)
    return pixels


def scroll(pixels, n):
    result = np.roll(pixels,-20*n,axis=0)
    result[-20*n:] = 0
    rng = np.random.default_rng(200+n)
    for y in range(368-20*n,360,20):
        result[y:y+8,12:190] = rng.integers(80,240,(8,178),dtype=np.uint8)
    return result


@pytest.mark.parametrize('n',[1,2,3])
def test_global_transport_only_bottom_n_and_exact_model_coordinates(n):
    tracker = AdjacentChatTracker(1,'S')
    before = tracker.process(chat(),0)
    after = tracker.process(scroll(chat(),n),1)
    assert after['classification'] == 'GLOBAL_SCROLL'
    assert after['SHIFT_PX'] == 20*n
    assert after['symmetry']['N_OUT'] == after['symmetry']['N_IN'] == n
    assert after['new_ids'][:-n] == before['new_ids'][n:]
    assert after['OCR_queue_row_ids'] == after['new_ids'][-n:]
    for carry in after['carried_rows']:
        assert carry['translated_span'] == [y-20*n for y in carry['old_span']]


def test_initial_seed_has_no_ocr_or_downstream_provenance_requirement():
    temporal = TemporalObservations()
    stage = FirstStage(Path('unused'),{},0,root=Path('.'),identity_observer=temporal.observe)
    assert not list(stage.chat_rows(chat(),frame_id=0,table_id=1,session_id='S'))
    assert not temporal.records
    retained, reducer = temporal.reduce([],{'1':'S'})
    assert retained == [] and not reducer.removals


def test_local_uncertainty_survives_exact_repeat_and_global_carry():
    tracker = AdjacentChatTracker(1,'S')
    original = chat()
    initial = tracker.process(original,0)
    changed = original.copy()
    changed[88:96,12:190] //= 2
    local = tracker.process(changed,1)
    assert local['classification'] == 'LOCAL_CHANGE'
    assert local['new_ids'] == initial['new_ids']
    assert local['new_link_states'][4] == 'LOCAL_LINK_UNRESOLVED'
    repeated = tracker.process(changed,2)
    assert repeated['new_link_states'][4] == 'LOCAL_LINK_UNRESOLVED'
    shifted = tracker.process(scroll(changed,1),3)
    assert shifted['new_link_states'][3] == 'GLOBAL_PARENT_UNRESOLVED'


@pytest.mark.parametrize('workers',[1,2])
def test_uncertain_reused_marker_is_barrier_preserving_original_ocr(workers):
    temporal = TemporalObservations()
    pixels = chat()[:16].copy()
    timeline = [dict(table_id='T01',stable_row_id='S:m',raw_ocr_text='Hand #123456',
                     status='UNRESOLVED',video_time=0),
                dict(table_id='T01',stable_row_id='S:a',raw_ocr_text='ante A',
                     status='UNRESOLVED',video_time=1/30)]
    temporal.record(1,0,'S:m',0,0,'NEW_CANDIDATE',pixels,False)
    temporal.observe(dict(table_id=1,frame_id=1,stable_row_id='S:m',source_span_index=0,
                         y1=0,origin='NEW_CANDIDATE',identity_confirmed=False),pixels)
    assert temporal.records[-1][-1] is True
    assert temporal.records[-1][5] == 'UNRESOLVED_OBJECT_LINK'
    assert zlib.decompress(temporal.records[-1][8]) == pixels.tobytes()
    temporal.record(1,1,'S:a',1,20,'NEW_CANDIDATE',pixels,False)
    retained,reducer = temporal.reduce(timeline,{'1':'S'},workers=workers)
    assert retained == timeline and timeline[0]['raw_ocr_text'] == 'Hand #123456'
    assert not reducer.removals and reducer.birth['S:a'] is None
    assert reducer.resolve('S:m') == 'S:m'
    assert any(d['reason']=='UNRESOLVED_OBJECT_LINK_CONTEXT_BARRIER' for d in reducer.decisions)


def test_missing_physical_slot_does_not_transmit_hand_context():
    temporal = TemporalObservations()
    pixels = chat()[:16].copy()
    timeline = [dict(table_id='T01',stable_row_id=rid,raw_ocr_text=text,
                     status='UNRESOLVED',video_time=0)
                for rid,text in [('S:m','Hand #123456'),('S:a','ante A')]]
    temporal.record(1,0,'S:m',0,0,'NEW_CANDIDATE',pixels,False)
    temporal.record(1,0,'S:a',2,40,'NEW_CANDIDATE',pixels,False)
    retained,reducer = temporal.reduce(timeline,{'1':'S'})
    assert retained == timeline and reducer.birth['S:a'] is None
    assert not reducer.removals
