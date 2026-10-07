from experiments.vertical_chat_symmetric_tracker import symmetry,transport
from experiments.vertical_chat_symmetric_tracker import replay
import base64
import hashlib
import json
import zlib
import numpy as np


def spans():return [(10+20*i,20+20*i) for i in range(12)]


def test_symmetric_one_two_three_rows_delete_top_and_preserve_remaining_ids():
    old=spans();old_ids=[f'old{i}' for i in range(12)]
    for n in [1,2,3]:
        decision=symmetry(old,old,20*n,20)
        assert decision['passed'] and decision['N_OUT']==decision['N_IN']==decision['N']==n
        ids,boxes=transport(old_ids,old,old,20*n,n,[f'new{i}' for i in range(n)])
        assert ids[:12-n]==old_ids[n:]
        assert all(boxes[j]==[a-20*n for a in old[j+n]] for j in range(12-n))
        assert len(set(ids))==12 and len(ids)==12


def test_visible_count_change_does_not_invent_transfer():
    result=symmetry(spans(),spans()[1:],20,20)
    assert not result['passed'] and result['reason']=='VISIBLE_WINDOW_COUNT_CHANGED'


def test_neighbor_offset_cannot_pass_symmetry():
    # One row predicted, but upper observed window starts two old rows higher.
    new=[(a+20,b+20) for a,b in spans()]
    result=symmetry(spans(),new,20,20)
    assert not result['passed'] and result['reason']=='N_OUT_N_IN_DISAGREE_WITH_SHIFT'


def test_local_coordinate_rearrangement_cannot_pass_global_transfer():
    old=spans();new=spans();new[5]=(115,125)
    result=symmetry(old,new,20,20)
    assert not result['passed'] and result['reason']=='RETAINED_ORDER_OR_COORDINATES_UNRESOLVED'


def test_commit_local_rejected_commit_preserves_checkpoint_ids_and_admission(tmp_path):
    source=tmp_path/'source';source.mkdir();out=tmp_path/'output'
    rng=np.random.default_rng(775)
    old=np.full((360,228),20,np.uint8)
    for y in range(5,351,20):old[y:y+10,4:208]=rng.integers(80,240,(10,204),dtype=np.uint8)
    def moved(image,n):
        d=n*20;new=np.full_like(image,20);new[:-d]=image[d:]
        for y in range(365-d,351,20):new[y:y+10,4:208]=rng.integers(80,240,(10,204),dtype=np.uint8)
        return new
    first=moved(old,2);local=first.copy();local[:120]=first[20:140];last=moved(first,1)
    (source/'shift_rule_summary.json').write_text(json.dumps({'calibration':{'regression_pitch_px':20,'calibration_end_frame':0}}),'utf-8')
    with (source/'states.jsonl').open('w',encoding='utf-8') as f:
        for fid,image in enumerate([old,first,local,last]):
            r={'table':1,'frame':fid,'pts':fid,'time_base':[1,30],'shape':list(image.shape),
               'sha256':hashlib.sha256(image.tobytes()).hexdigest(),
               'zlib_gray':base64.b64encode(zlib.compress(image.tobytes())).decode()}
            f.write(json.dumps(r)+'\n')
    replay(source,out,True)
    records=[json.loads(s) for s in (out/'symmetric_transitions.jsonl').read_text('utf-8').splitlines()]
    assert [r['classification'] for r in records]==['GLOBAL_SCROLL','LOCAL_CHANGE','GLOBAL_SCROLL']
    assert records[1]['checkpoint_committed'] is False and records[1]['OCR_queue_row_ids']==[]
    assert records[2]['frame_from']==1
    assert records[2]['old_ids']==records[0]['new_ids']
    assert records[2]['new_ids'][:-1]==records[0]['new_ids'][1:]
    queue=[json.loads(s) for s in (out/'zonal_ocr_queue.jsonl').read_text('utf-8').splitlines()]
    assert [r['frame'] for r in queue]==[1,1,3]
    assert len({r['item_id'] for r in queue})==3
