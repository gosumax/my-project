import base64
import hashlib
import json
import zlib
import numpy as np
from experiments.vertical_chat_adjacent_safe_tracker import complete_spans,replay,stationary_links,local_object_links
from experiments.vertical_chat_symmetric_tracker import symmetry


def raster(seed=991):
    rng=np.random.default_rng(seed);image=np.full((360,228),20,np.uint8)
    for y in range(5,351,20):image[y:y+10,4:208]=rng.integers(80,240,(10,204),dtype=np.uint8)
    return image


def moved(image,n,seed):
    d=n*20;new=np.full_like(image,20);new[:-d]=image[d:]
    rng=np.random.default_rng(seed)
    for y in range(365-d,351,20):new[y:y+10,4:208]=rng.integers(80,240,(10,204),dtype=np.uint8)
    return new


def run_frames(tmp_path,images,keep_local_objects=False):
    source=tmp_path/'source';source.mkdir();out=tmp_path/'out'
    (source/'shift_rule_summary.json').write_text(json.dumps({'calibration':{'regression_pitch_px':20,'calibration_end_frame':0}}),'utf-8')
    with (source/'states.jsonl').open('w',encoding='utf-8') as f:
        for frame,image in enumerate(images):
            f.write(json.dumps({'table':1,'frame':frame,'pts':frame,'time_base':[1,30],'shape':list(image.shape),
                'sha256':hashlib.sha256(image.tobytes()).hexdigest(),'zlib_gray':base64.b64encode(zlib.compress(image.tobytes())).decode()})+'\n')
    replay(source,out,keep_local_objects)
    records=[json.loads(l) for l in (out/'symmetric_transitions.jsonl').read_text('utf-8').splitlines()]
    queue=[json.loads(l) for l in (out/'zonal_ocr_queue.jsonl').read_text('utf-8').splitlines()]
    return records,queue,out


def test_complete_first_row_is_not_automatically_excluded():
    assert complete_spans([(13,25),(33,43)])==([(13,25),(33,43)],[])
    assert complete_spans([(0,5),(13,25),(33,43)])==([(13,25),(33,43)],[(0,5)])
    assert complete_spans([(1,5),(13,25)])==([(13,25)],[(1,5)])


def test_clipped_top_toggle_does_not_break_complete_row_symmetry():
    core=[(13+20*i,23+20*i) for i in range(18)]
    for old_edge,new_edge in [([(0,5)],[]),([],[ (1,5)])]:
        old,_=complete_spans(old_edge+core);new,_=complete_spans(new_edge+core)
        result=symmetry(old,new,20,20)
        assert result['passed'] and result['N_OUT']==result['N_IN']==1
    # An interior missing band remains a genuine guard failure.
    assert not symmetry(core,core[:8]+core[9:],20,20)['passed']


def test_local_changed_row_has_no_confirmed_id_stationary_row_keeps_it():
    old=raster();new=old.copy();new[5:15,4:208]=100
    spans=[(5+20*i,15+20*i) for i in range(18)];ids=[f'old{i}' for i in range(18)]
    links,model=stationary_links({'gray':old,'observed_spans':spans,'model_spans':spans,'row_ids':ids},new,spans)
    assert links[0] is None and links[1:]==ids[1:]
    assert model[1:]==spans[1:]


def test_global_unresolved_global_does_not_revive_stale_safe_ids(tmp_path):
    old=raster();first=moved(old,2,993);unresolved=raster(201);last=moved(unresolved,1,202)
    records,queue,out=run_frames(tmp_path,[old,first,unresolved,last])
    assert [r['classification'] for r in records]==['GLOBAL_SCROLL','UNRESOLVED','GLOBAL_SCROLL']
    assert [r['frame_from'] for r in records]==[0,1,2]
    assert records[1]['SAFE_ID_STATE_UNCHANGED'] and records[1]['OCR_queue_row_ids']==[]
    assert all(rid is None for rid in records[1]['new_ids'])
    assert all(c['row_id'] is None for c in records[2]['carried_rows'])
    assert records[2]['safe_frame_from']==1
    assert [q['frame'] for q in queue]==[1,1,3]
    journal=json.loads((out/'safe_object_journal.json').read_text('utf-8'))
    assert set(records[0]['old_ids']).issubset(journal)


def test_local_updates_previous_without_new_ids_or_safe_commit(tmp_path):
    old=raster();local=old.copy();local[:120]=old[20:140];last=moved(local,1,333)
    records,queue,_=run_frames(tmp_path,[old,local,last])
    assert [r['classification'] for r in records]==['LOCAL_CHANGE','GLOBAL_SCROLL']
    assert records[0]['SAFE_ID_STATE_UNCHANGED'] and not records[0]['checkpoint_committed']
    assert records[0]['OCR_queue_row_ids']==[] and records[1]['frame_from']==1
    assert len(queue)==1 and queue[0]['frame']==2
    old_known={rid for rid in records[0]['new_ids'] if rid is not None}
    inherited={c['row_id'] for c in records[1]['carried_rows'] if c['row_id'] is not None}
    assert inherited.issubset(old_known)


def test_each_local_frame_advances_ncc_source(tmp_path):
    old=raster();images=[old]
    for i in range(1,9):
        image=old.copy();image[5:15,4:208]=80+i*7;images.append(image)
    records,queue,_=run_frames(tmp_path,images)
    assert all(r['classification']=='LOCAL_CHANGE' for r in records)
    assert [r['frame_from'] for r in records]==list(range(8))
    assert all(r['PREVIOUS_FRAME_UPDATED'] and r['SAFE_ID_STATE_UNCHANGED'] for r in records)
    assert not queue


def test_local_keep_uncertainty_is_not_cured_by_later_exact_pixels():
    old=raster();new=old.copy();new[5:15,4:208]=100
    spans=[(5+20*i,15+20*i) for i in range(18)];ids=[f'old{i}' for i in range(18)]
    previous={'gray':old,'observed_spans':spans,'model_spans':spans,'row_ids':ids,'link_states':['INITIAL_VISIBLE']*18}
    links,model,states=local_object_links(previous,new,spans)
    assert links==ids and states[0]=='LOCAL_LINK_UNRESOLVED' and states[1:] == ['INITIAL_VISIBLE']*17
    previous.update({'gray':new,'row_ids':links,'model_spans':model,'link_states':states})
    links,_,states=local_object_links(previous,new,spans)
    assert links==ids and states[0]=='LOCAL_LINK_UNRESOLVED'


def test_keep_mode_cannot_revive_ids_across_unresolved_motion(tmp_path):
    old=raster();first=moved(old,2,993);unresolved=raster(201);last=moved(unresolved,1,202)
    records,queue,_=run_frames(tmp_path,[old,first,unresolved,last],True)
    assert records[1]['classification']=='UNRESOLVED'
    assert all(i is None for i in records[1]['new_ids'])
    assert all(c['row_id'] is None for c in records[2]['carried_rows'])
    assert all(s=='GLOBAL_PARENT_UNRESOLVED' for s in records[2]['new_link_states'][:-1])
    assert [q['frame'] for q in queue]==[1,1,3]
