"""Transactional visible-window tracker. Commit only GLOBAL + N-out=N-in.

LOCAL/UNRESOLVED cannot mutate confirmed IDs, positions or admit OCR. A later
state is tested against the last confirmed pixel/ID checkpoint; no guessed
carry across a rejected boundary. Independent local pixel reference is audit
data only and never determines N or admission.
"""
import argparse
import base64
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import zlib
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.vertical_chat_zone_gate import global_scroll
from experiments.vertical_chat_shift import row_ncc_reference
from experiments.vertical_chat_zone_replay import recognize
from experiments.vertical_chat_id_transfer_audit import get_metric
from parser_core.chat_rows import physical_spans,exclude_first_visible_span,padded_row_span


def symmetry(old_spans,new_spans,shift,pitch,tolerance=2.):
    n=round(shift/pitch);result={'N':n,'old_count':len(old_spans),'new_count':len(new_spans),'passed':False}
    if n<1 or n>=len(old_spans) or n>len(new_spans):result['reason']='N_OUTSIDE_PROVEN_OVERLAP';return result
    old_centers=np.array([(a+b)/2-shift for a,b in old_spans]);new_centers=np.array([(a+b)/2 for a,b in new_spans])
    nout=int(np.sum(old_centers<new_centers[0]-tolerance));nin=int(np.sum(new_centers>old_centers[-1]+tolerance))
    result.update({'N_OUT':nout,'N_IN':nin})
    if len(old_spans)!=len(new_spans):result['reason']='VISIBLE_WINDOW_COUNT_CHANGED';return result
    if nout!=n or nin!=n:result['reason']='N_OUT_N_IN_DISAGREE_WITH_SHIFT';return result
    residuals=(old_centers[n:]-new_centers[:-n]).tolist();result['retained_center_residuals_px']=residuals
    if max(map(abs,residuals),default=0)>tolerance:result['reason']='RETAINED_ORDER_OR_COORDINATES_UNRESOLVED';return result
    result['passed']=True;result['reason']='N_OUT_EQUALS_N_IN_WITH_ORDERED_COORDINATE_CARRY';return result


def transport(old_ids,old_spans,new_spans,shift,n,new_ids):
    if len(new_ids)!=n:raise ValueError('WRONG_NEW_OBJECT_COUNT')
    kept_ids=old_ids[n:];moved=[[a-shift,b-shift] for a,b in old_spans[n:]]
    return kept_ids+new_ids,moved+[list(s) for s in new_spans[-n:]]


def replay(source,out,observed_geometry=False):
    out.mkdir(parents=True,exist_ok=False)
    rule=json.loads((source/'shift_rule_summary.json').read_text('utf-8'));pitch=rule['calibration']['regression_pitch_px'];cut=rule['calibration']['calibration_end_frame']
    checkpoints={};counters=Counter();statistics=Counter();timings=Counter();transitions=[];windows=[];queue=[]
    for line in (source/'states.jsonl').open(encoding='utf-8'):
        r=json.loads(line);t,fid=r['table'],r['frame']
        gray=np.frombuffer(zlib.decompress(base64.b64decode(r['zlib_gray'])),np.uint8).reshape(r['shape']).copy()
        all_spans=physical_spans(gray);spans,suppressed=exclude_first_visible_span(all_spans)
        if t not in checkpoints:
            ids=[]
            for _ in spans:counters[t]+=1;ids.append(f'SYM:T{t}:R{counters[t]:06d}')
            checkpoint={'frame':fid,'gray':gray,'observed_spans':spans,'model_spans':spans,'row_ids':ids}
            checkpoints[t]=checkpoint;windows.append({'table':t,'frame':fid,'initial':True,'row_ids':ids,'observed_spans':spans,'model_spans':spans});continue
        previous=checkpoints[t];tick=time.perf_counter();gate=global_scroll(previous['gray'],gray);timings['zonal_ncc_seconds']+=time.perf_counter()-tick
        candidate=gate['classification'];guard=None;admitted=[];carried=[];removed=[];proxy=None;actual=None
        if candidate=='GLOBAL_SCROLL':
            guard=symmetry(previous['observed_spans'] if observed_geometry else previous['model_spans'],spans,gate['shift_px'],pitch)
            if not guard['passed']:
                gate['classification']='UNRESOLVED';gate['reason']='SYMMETRY:'+guard['reason']
            else:
                n=guard['N'];shift=gate['shift_px'];new_ids=[]
                for j in range(len(spans)-n,len(spans)):
                    counters[t]+=1;rid=f'SYM:T{t}:R{counters[t]:06d}';new_ids.append(rid)
                    source_index=j+len(suppressed);crop_span=padded_row_span(all_spans,source_index,gray.shape[0])
                    pixels=np.ascontiguousarray(gray[crop_span[0]:crop_span[1]]);ok,packed=cv2.imencode('.png',pixels);assert ok
                    queue.append({'item_id':rid,'table':t,'frame':fid,'frame_from':previous['frame'],
                       'source_pts':r['pts'],'time_base':r['time_base'],'classification':'GLOBAL_SCROLL',
                       'row_index':j,'source_span_index':source_index,'row_span':spans[j],'padded_span':crop_span,
                       'N_OUT':n,'N_IN':n,'new_rows_in_transition':n,'SHIFT_PX':shift,'ROW_STEP_PX':pitch,
                       'source_chat_sha256':r['sha256'],'source_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),
                       'png_base64':base64.b64encode(packed).decode()});admitted.append(rid)
                current_ids,model_spans=transport(previous['row_ids'],previous['model_spans'],spans,shift,n,new_ids)
                for j in range(len(spans)-n):
                    i=j+n;carried.append({'old_index':i,'new_index':j,'row_id':previous['row_ids'][i],
                       'old_span':previous['model_spans'][i],'translated_span':model_spans[j],'observed_span':spans[j],'OCR_submitted':False})
                removed=previous['row_ids'][:n]
                # Audit only AFTER commit decision; this local raster proxy
                # cannot influence classification, N, IDs or queue admission.
                proxy=row_ncc_reference(previous['gray'],gray,previous['observed_spans'],spans)
                if proxy:
                    mp={int(j):i for j,i in proxy['mapping'].items()};bottom=[j for j,i in mp.items() if i==len(previous['observed_spans'])-1]
                    if len(bottom)==1:
                        j=bottom[0];run=0
                        while j-run>=0 and len(previous['observed_spans'])-1-run>=0 and mp.get(j-run)==len(previous['observed_spans'])-1-run:run+=1
                        if run>=6:actual=len(spans)-j-1
                mp={int(j):i for j,i in proxy['mapping'].items()} if proxy else {}
                id_metrics,id_checks=get_metric(previous['row_ids'],current_ids,mp)
                id_metrics['ID_TRANSFER_UNRESOLVED']+=sum(c['new_index'] not in mp for c in carried)
                checkpoints[t]={'frame':fid,'gray':gray,'observed_spans':spans,'model_spans':model_spans,'row_ids':current_ids}
                windows.append({'table':t,'frame':fid,'initial':False,'row_ids':current_ids,'observed_spans':spans,'model_spans':model_spans})
        assert gate['classification']=='GLOBAL_SCROLL' or not admitted
        assert gate['classification']!='GLOBAL_SCROLL' or len(admitted)==len(removed)==guard['N']
        record={'table':t,'frame_from':previous['frame'],'frame_to':fid,'candidate_classification':candidate,
                'classification':gate['classification'],'reason':gate['reason'],'SHIFT_PX':gate['shift_px'],
                'ROW_STEP_PX':pitch,'symmetry':guard,'zones':gate['zones'],'removed_top_ids':removed,'carried_rows':carried,
                'old_ids':previous['row_ids'],'old_model_spans':previous['model_spans'],'old_observed_spans':previous['observed_spans'],
                'new_observed_spans':spans,'new_ids':checkpoints[t]['row_ids'] if admitted else None,
                'new_model_spans':checkpoints[t]['model_spans'] if admitted else None,'OCR_queue_row_ids':admitted,
                'local_pixel_reference':proxy,'actual_new_rows':actual,'evaluation':fid>cut,
                'checkpoint_committed':bool(admitted),'ID_transfer_metrics':dict(id_metrics) if admitted else {},
                'ID_transfer_checks':id_checks if admitted else [],'symmetry_geometry_source':'OBSERVED_RAW_BANDS' if observed_geometry else 'INTEGRATED_OBJECT_POSITION'}
        transitions.append(record)
        if fid>cut:
            statistics['TESTED_STATES']+=1;statistics[gate['classification']+'_COUNT']+=1
            statistics['GLOBAL_CANDIDATES']+=candidate=='GLOBAL_SCROLL'
            statistics['BROKEN_SYMMETRY_COUNT']+=bool(guard and not guard['passed'])
            if admitted:
                statistics['OCR_NEW_BOTTOM_ROWS']+=len(admitted);statistics['ID_CARRY_COUNT']+=len(carried)
                statistics.update(id_metrics)
                if actual is not None:
                    statistics['GLOBAL_COUNT_VALIDATED']+=1;statistics['GLOBAL_COUNT_CORRECT']+=actual==guard['N']
                    statistics['FALSE_GLOBAL_SCROLL']+=actual!=guard['N']
                else:statistics['GLOBAL_COUNT_UNKNOWN']+=1
    for name,data in [('symmetric_transitions',transitions),('confirmed_windows',windows),('zonal_ocr_queue',queue)]:
        with (out/f'{name}.jsonl').open('w',encoding='utf-8') as f:
            for row in data:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    statistics.setdefault('FALSE_GLOBAL_SCROLL',0)
    # summarize reasons without labeling geometry mismatch as a physical cause
    reasons=Counter(r['symmetry']['reason'] for r in transitions if r['evaluation'] and r['symmetry'] and not r['symmetry']['passed'])
    summary={'statistics':dict(statistics),'broken_symmetry_reasons':dict(reasons),'timing_seconds':dict(timings),
             'all_frames_OCR_queued_rows':len(queue),'all_frames_OCR_repeated_row_ids':len(queue)-len({r['item_id'] for r in queue}),
             'queue_sha256':hashlib.sha256((out/'zonal_ocr_queue.jsonl').read_bytes()).hexdigest(),
             'checkpoint_policy':'Rejected/local states do not mutate confirmed object IDs or coordinates. Next state compared to last committed viewport.',
             'symmetry_geometry_source':'OBSERVED_RAW_BANDS' if observed_geometry else 'INTEGRATED_OBJECT_POSITION',
             'contract':'GLOBAL zone consensus + Nout=Nin=N + ordered retained geometry within2px. Delete exactly firstN visible objects (retain source journal); translate remaining by exactSHIFT; allocate exactlyN lower objects/OCR. Otherwise UNRESOLVED and no mutation.',
             'input':str(source.resolve()),'OCR_executed':False,'verdict':'NOT_PROVEN'}
    (out/'zonal_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['replay','ocr']);p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--observed-geometry',action='store_true');a=p.parse_args()
    if a.mode=='replay':replay(a.input,a.output,a.observed_geometry)
    else:recognize(a.output)
