from dataclasses import FrozenInstanceError,replace
import numpy as np
import pytest
from experiments.cross_track_linker import (TrackSnapshot,motion_proof,decide,SAME,DIFFERENT,AMBIGUOUS,digest)
from parser_core.chat_rows import physical_spans,padded_row_span


def viewport(height=360):
    gray=np.zeros((height,228),np.uint8)
    for i,y in enumerate((15,60,105,150,195,240,285)):
        gray[y:y+8,12:100]=100+i*15
        gray[y+2:y+6,120:131+i*8]=225
    return gray


def snapshot(rid,frame,gray,index,*,first=None,spans=None,session='s',text='same row'):
    bands=physical_spans(gray)
    a,b=padded_row_span(bands,index+1,gray.shape[0])
    first=frame if first is None else first
    return TrackSnapshot(rid,'T01',session,first,frame,first*33.3,frame*33.3,index+1,a,
        digest(gray[a:b]),text,text,.99,((first,frame),) if spans is None else spans,first)


def fixture_scroll():
    before=viewport();after=np.zeros_like(before);after[:-20]=before[20:]
    old_ids=tuple('abcdef');new_ids=tuple('bXdef')
    proof=motion_proof(before,after,2,1,old_ids,new_ids,0,1)
    return before,after,snapshot('old',0,before,2),snapshot('new',1,after,1),proof


def test_simple_scroll_continuity_combines_unique_target_and_neighbors():
    before,after,old,new,proof=fixture_scroll()
    assert proof.shift_px==20 and len(proof.neighbor_anchors)>=2
    assert decide(old,new,proof,1).link_status==SAME


def test_temporary_one_frame_local_raster_loss():
    before=viewport();after=before.copy();bridge=before.copy()
    old=snapshot('old',0,before,2);new=snapshot('new',2,after,2)
    bands=physical_spans(before);a,b=padded_row_span(bands,3,before.shape[0]);bridge[a:b]=0
    proof=motion_proof(before,after,2,2,tuple('abcdef'),tuple('abXdef'),0,2,bridge=bridge,bridge_frame=1)
    assert proof.local_loss_bridge
    assert decide(old,new,proof,2).link_status==SAME


def test_same_text_new_bottom_onset_after_boundary_is_different():
    before=viewport(400);after=before.copy();after[330:338,15:120]=250
    old=snapshot('old',0,before,2);new=snapshot('new',101,after,6)
    onset=motion_proof(before,after,2,6,tuple('abcdef'),tuple('abcdefX'),100,101)
    assert onset.newly_exposed and len(onset.neighbor_anchors)>=2
    assert decide(old,new,None,101,onset_proof=onset,boundary_frames=(50,)).link_status==DIFFERENT
    assert decide(old,new,None,101,onset_proof=onset).link_status==AMBIGUOUS


def test_simultaneous_tracks_need_distinct_position_witness():
    gray=viewport();old=snapshot('old',1,gray,1,first=0);new=snapshot('new',1,gray,3)
    assert decide(old,new,None,1,simultaneous_positions=(2,4)).link_status==DIFFERENT
    assert decide(old,new,None,1).link_status==AMBIGUOUS


def test_same_pixels_different_session():
    before,after,old,new,proof=fixture_scroll()
    assert decide(old,replace(new,session_id='foreign'),proof,1).link_status==DIFFERENT


def test_missing_motion_or_neighbors_stays_ambiguous():
    _,_,old,new,proof=fixture_scroll()
    assert decide(old,new,None,1).link_status==AMBIGUOUS
    assert decide(old,new,replace(proof,neighbor_anchors=()),1).link_status==AMBIGUOUS
    assert decide(old,new,replace(proof,target_unique=False),1).link_status==AMBIGUOUS


def test_multiple_row_motion_cannot_use_isolated_repeated_boilerplate():
    before,after,old,new,proof=fixture_scroll()
    assert len(proof.neighbor_anchors)>=2
    assert decide(old,replace(new,normalized_text='another reading'),proof,1).link_status==AMBIGUOUS


def test_future_frame_and_future_terminal_lifetime_rejected():
    _,_,old,new,proof=fixture_scroll()
    assert decide(old,new,replace(proof,evidence_frames=(0,2)),1).link_reason=='FUTURE_FRAME_EVIDENCE_REJECTED'
    assert decide(replace(old,last_frame=9,visible_spans=((0,9),)),new,proof,1).link_reason=='FUTURE_FRAME_EVIDENCE_REJECTED'


def test_snapshot_immutability_copies_nested_spans():
    gray=viewport();spans=[[0,0]];old=snapshot('old',0,gray,2,spans=spans)
    spans[0][1]=100
    assert old.visible_spans==((0,0),)
    with pytest.raises(FrozenInstanceError):old.last_frame=3


def test_original_pixels_hash_and_row_ids_unchanged():
    before=viewport();after=before.copy();a,b=digest(before),digest(after)
    ids=tuple('abcdef')
    motion_proof(before,after,2,2,ids,ids,0,1)
    assert (digest(before),digest(after),ids)==(a,b,tuple('abcdef'))


def test_stale_motion_endpoints_cannot_certify_other_time():
    _,_,old,new,proof=fixture_scroll()
    assert decide(replace(old,last_frame=100),replace(new,first_frame=101,last_frame=101),proof,101).link_status==AMBIGUOUS


def test_historical_boundary_outside_separation_cannot_prove_onset():
    before=viewport(400);after=before.copy();after[330:338,15:120]=250
    old=snapshot('old',80,before,2);new=snapshot('new',101,after,6)
    onset=motion_proof(before,after,2,6,tuple('abcdef'),tuple('abcdefX'),100,101)
    result=decide(old,new,None,101,onset_proof=onset,boundary_frames=(50,))
    assert result.link_status==AMBIGUOUS and 'OUTSIDE' in result.link_reason


def test_runner_does_not_clip_away_current_coexistence():
    from experiments.run_cross_track_linker import causal_old_frame
    track={'visible_spans':[[0,20]]}
    assert causal_old_frame(track,10,('old','new'),'old')==10
    assert causal_old_frame(track,10,('new',),'old')==9
    gray=viewport();old=snapshot('old',10,gray,1,first=0);new=snapshot('new',10,gray,3)
    assert decide(old,new,None,10,simultaneous_positions=(old.row_index,new.row_index)).link_status==DIFFERENT
