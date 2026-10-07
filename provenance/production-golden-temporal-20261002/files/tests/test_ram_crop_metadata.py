from fractions import Fraction
import gc
import hashlib
import weakref
import cv2
import numpy as np
import pytest
from experiments.ram_crop_metadata import RamMetadata, debug_overlay
from experiments.capture_temporal_evidence import source_ms


def stamp(fid):
    return dict(source_frame=fid,source_pts=fid*1001,time_base_num=1,time_base_den=30000,
                timestamp_ms=source_ms(fid*1001,1,30000))


def meta(fid,rid='s:R1',origin='NEW_CANDIDATE'):
    return dict(frame_id=fid,source_pts=stamp(fid)['source_pts'],session_id='s',table_id=1,
                stable_row_id=rid,source_span_index=2,y1=40,origin=origin)


def observe(layer,fid,rid='s:R1',origin='NEW_CANDIDATE'):
    with layer.borrow(meta(fid,rid,origin),np.full((20,200),77,np.uint8),stamp(fid)) as view:
        assert view.metadata['row_key']==f'1:{fid}:2'
        assert view.metadata['observation_count']>=1
    layer.complete_frame(fid,stamp(fid),['s'])


def test_overlay_preserves_original_hash_and_never_writes_images(monkeypatch,tmp_path):
    def forbidden(*args,**kwargs):
        pytest.fail('Image IO in RAM watermark path')
    monkeypatch.setattr(cv2,'imwrite',forbidden)
    from PIL import Image
    monkeypatch.setattr(Image.Image,'save',forbidden)
    pixels=np.arange(4000,dtype=np.uint8).reshape(20,200)
    digest=hashlib.sha256(pixels.tobytes()).hexdigest()
    layer=RamMetadata()
    with layer.borrow(meta(0),pixels,stamp(0)) as view:
        overlay=debug_overlay(view.image,view.metadata)
        assert not np.shares_memory(overlay,pixels)
        overlay[:]=0
    assert hashlib.sha256(pixels.tobytes()).hexdigest()==digest
    assert list(tmp_path.rglob('*'))==[]


def test_identity_pts_and_unchanged_visibility_frames():
    layer=RamMetadata()
    observe(layer,0)
    layer.complete_frame(1,stamp(1),[])
    observe(layer,2)
    layer.finish()
    assert len(layer.tracks)==1
    t=layer.tracks['s:R1']
    assert t['observation_count']==3 and t['explicit_view_count']==2
    assert t['visible_spans']==[[0,2]]
    assert (t['first_seen_ms'],t['last_seen_ms'])==(0,source_ms(2002,1,30000))
    assert (t['frame_id'],t['source_pts'],t['time_base_num'],t['time_base_den'])==(2,2002,1,30000)


def test_empty_changed_chat_ends_visibility_and_reappearance_has_gap():
    layer=RamMetadata();observe(layer,0)
    layer.complete_frame(1,stamp(1),['s'])
    layer.complete_frame(2,stamp(2),[])
    observe(layer,3);layer.finish()
    assert layer.tracks['s:R1']['visible_spans']==[[0,0],[3,3]]
    assert layer.tracks['s:R1']['observation_count']==2
    assert layer.tracks['s:R1']['visibility_gap_count']==1


def boundary(fid):
    return dict(frame_id=fid,row_index=1,timestamp_ms=stamp(fid)['timestamp_ms'])


def test_existing_classifier_previous_tail_current_and_ambiguous():
    layer=RamMetadata();observe(layer,0)
    layer.complete_frame(1,stamp(1),[])
    observe(layer,2,'s:R2');layer.finish()
    assert layer.temporal('s:R1',boundary(1),boundary(4))[0]=='PREVIOUS_HAND_TAIL'
    assert layer.temporal('s:R2',boundary(1),boundary(4))[0]=='CURRENT_HAND'
    assert layer.temporal('s:R2')[0]=='AMBIGUOUS'
    assert layer.temporal('s:R2',boundary(1),boundary(4),blocked=True)[0]=='AMBIGUOUS'


def test_initial_and_unresolved_tracks_remain_ambiguous():
    for origin in ('INITIAL_VISIBLE','UNRESOLVED_IDENTITY'):
        layer=RamMetadata();observe(layer,0,origin=origin);layer.finish()
        assert layer.temporal('s:R1',boundary(0),boundary(4))[0]=='AMBIGUOUS'


def test_borrow_release_after_exception_has_no_image_reference():
    layer=RamMetadata();pixels=np.ones((20,200),np.uint8);ref=weakref.ref(pixels)
    with pytest.raises(RuntimeError):
        with layer.borrow(meta(0),pixels,stamp(0)) as view:
            raise RuntimeError('consumer failed')
    assert view.image is None and layer.live_images==layer.image_bytes==0
    del pixels;gc.collect()
    assert ref() is None
    layer.complete_frame(0,stamp(0),['s']);layer.finish()
    assert not layer.active and not layer.pending


def test_temporal_missing_pts_stays_ambiguous():
    layer=RamMetadata();m=meta(0);m['source_pts']=None;s=stamp(0);s['source_pts']=None;s['timestamp_ms']=None
    with layer.borrow(m,np.zeros((10,200),np.uint8),s):pass
    layer.complete_frame(0,s,['s']);layer.finish()
    assert layer.temporal('s:R1',boundary(1),boundary(4))[0]=='AMBIGUOUS'


def test_track_cannot_cross_session_and_source_mismatch_rejected():
    layer=RamMetadata();observe(layer,0)
    m=meta(1);m['session_id']='foreign'
    with pytest.raises(ValueError,match='crossed'):
        with layer.borrow(m,np.zeros((20,200),np.uint8),stamp(1)):pass
    with pytest.raises(ValueError,match='PTS mismatch'):
        with layer.borrow(meta(1),np.zeros((20,200),np.uint8),stamp(2)):pass


def test_crop_snapshot_cannot_mutate_track_lifetime():
    layer=RamMetadata()
    with layer.borrow(meta(0),np.zeros((20,200),np.uint8),stamp(0)) as view:
        snapshot=view.metadata
        snapshot['first_seen_ms']=999
    layer.complete_frame(0,stamp(0),['s'])
    layer.complete_frame(1,stamp(1),[])
    assert snapshot['visible_spans']==()
    assert layer.tracks['s:R1']['visible_spans']==[[0,1]]
    assert layer.tracks['s:R1']['first_seen_ms']==0


def test_crop_from_future_frame_rejected_before_track_created():
    layer=RamMetadata()
    with pytest.raises(ValueError,match='Crop frame'):
        with layer.borrow(meta(5),np.zeros((20,200),np.uint8),stamp(5)):pass
    assert layer.tracks=={} and layer.live_images==0


def test_frame_completion_during_borrow_rejected():
    layer=RamMetadata()
    with layer.borrow(meta(0),np.zeros((20,200),np.uint8),stamp(0)):
        with pytest.raises(ValueError,match='active crop'):
            layer.complete_frame(0,stamp(0),['s'])
    layer.complete_frame(0,stamp(0),['s']);layer.finish()
