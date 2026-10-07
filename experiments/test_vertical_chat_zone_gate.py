import numpy as np
import hashlib
import json
import pytest
from experiments.vertical_chat_zone_gate import global_scroll,classify_zones
from experiments.vertical_chat_zone_replay import carried_ids,frozen_queue


def zone(d,accepted=True):return {'shift_px':d,'accepted':accepted}


def test_moving_majority_cannot_override_stationary_bottom():
    result=classify_zones({'top':zone(20),'middle':zone(20),'bottom':zone(0)})
    assert result['classification']=='LOCAL_CHANGE' and result['shift_px'] is None


def test_majority_plus_unobserved_zone_can_establish_scroll():
    result=classify_zones({'top':zone(40),'middle':zone(41),'bottom':zone(None,False)})
    assert result['classification']=='GLOBAL_SCROLL' and result['shift_px']==40


def test_one_moving_zone_does_not_establish_scroll():
    result=classify_zones({'top':zone(20),'middle':zone(None,False),'bottom':zone(None,False)})
    assert result['classification']=='UNRESOLVED'


def test_zero_consensus_never_admits_new_rows_by_formula():
    result=classify_zones({'top':zone(0),'middle':zone(0),'bottom':zone(0)})
    assert result['classification']=='LOCAL_CHANGE' and result['shift_px'] is None


def test_rigid_native_pixel_scroll_20_40_60():
    rng=np.random.default_rng(781)
    old=rng.integers(0,255,(377,228),dtype=np.uint8)
    for d in [20,40,60]:
        new=rng.integers(0,255,old.shape,dtype=np.uint8);new[:-d]=old[d:]
        result=global_scroll(old,new)
        assert result['classification']=='GLOBAL_SCROLL' and result['shift_px']==d
        assert all(z['shift_px']==d for z in result['zones'].values())


def test_partial_upper_redraw_has_stationary_lower_zones():
    rng=np.random.default_rng(13);old=rng.integers(0,255,(377,228),dtype=np.uint8)
    new=old.copy();new[:125]=old[20:145]
    result=global_scroll(old,new)
    assert result['zones']['top']['shift_px']==20
    assert result['zones']['bottom']['shift_px']==0
    assert result['classification']=='LOCAL_CHANGE'


def test_coordinate_carry_keeps_ids_and_unknown_parents_out_of_bottom_queue():
    old=[(10,20),(30,40),(50,60),(70,80)]
    new=[(10,20),(30,40),(50,60),(70,80)]
    ids,carry=carried_ids(['old0',None,'old2','old3'],old,new,20,1)
    assert ids==[None,'old2','old3',None]
    assert all(not row['OCR_submitted'] for row in carry)
    assert all(row['new_index']<3 for row in carry)
    assert all(row['translated_span']==[a-20 for a in row['old_span']] for row in carry)


def test_local_keep_does_not_allocate_fresh_ids():
    spans=[(10,20),(30,40)]
    ids,_=carried_ids(['old0','old1'],spans,[(10,21),(31,40)],0,0)
    assert ids==['old0','old1']


def test_modified_ocr_queue_is_rejected_before_worker_start(tmp_path):
    path=tmp_path/'zonal_ocr_queue.jsonl'
    path.write_text(json.dumps({'item_id':'old','classification':'GLOBAL_SCROLL'})+'\n','utf-8')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path/'zonal_summary.json').write_text(json.dumps({'queue_sha256':digest}),'utf-8')
    frozen_queue(tmp_path)
    path.write_text(json.dumps({'item_id':'replacement','classification':'GLOBAL_SCROLL'})+'\n','utf-8')
    with pytest.raises(ValueError,match='CHANGED_AFTER_PIXEL'):frozen_queue(tmp_path)


def test_duplicate_ocr_row_is_rejected_before_worker_start(tmp_path):
    path=tmp_path/'zonal_ocr_queue.jsonl'
    path.write_text((json.dumps({'item_id':'same','classification':'GLOBAL_SCROLL'})+'\n')*2,'utf-8')
    (tmp_path/'zonal_summary.json').write_text(json.dumps({'queue_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}),'utf-8')
    with pytest.raises(ValueError,match='DUPLICATE_ROW'):frozen_queue(tmp_path)
