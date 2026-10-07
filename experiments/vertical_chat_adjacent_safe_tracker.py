"""Adjacent motion and a separate immutable-until-commit SAFE_ID_STATE.

Experiment only. A lost observation link is None, never filled from history.
Top border fragments are edge evidence, not complete row inventory.
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
from experiments.vertical_chat_symmetric_tracker import symmetry,transport
from experiments.vertical_chat_zone_gate import global_scroll
from experiments.vertical_chat_shift import row_ncc_reference,coordinate_map
from experiments.vertical_chat_id_transfer_audit import get_metric
from experiments.vertical_chat_zone_replay import recognize
from parser_core.chat_rows import physical_spans,padded_row_span


def complete_spans(all_spans):
    """Only a band touching the upper border is uncertain edge evidence.

    Do not suppress a complete first band merely because it is first.
    Source pixels and production segmentation remain unchanged.
    """
    edge=all_spans[:1] if all_spans and all_spans[0][0]<=1 else []
    return all_spans[len(edge):],edge


def stationary_links(previous,gray,spans):
    """LOCAL: keep only exact same-position pixel links; never create IDs."""
    ids=[None]*len(spans);model=[list(s) for s in spans]
    old_index={tuple(s):i for i,s in enumerate(previous['observed_spans'])}
    for j,(a,b) in enumerate(spans):
        i=old_index.get((a,b))
        if i is None:continue
        if np.array_equal(previous['gray'][a:b,:-12],gray[a:b,:-12]):
            ids[j]=previous['row_ids'][i];model[j]=previous['model_spans'][i]
    return ids,model


def local_object_links(previous,gray,spans):
    """Existing KEEP policy: object continuity is not confirmed pixel lineage."""
    exact,exact_model=stationary_links(previous,gray,spans)
    ids=[None]*len(spans);model=[list(s) for s in spans];states=['UNKNOWN']*len(spans)
    for j,i in coordinate_map(previous['observed_spans'],spans,0).items():
        ids[j]=previous['row_ids'][i];model[j]=previous['model_spans'][i]
        if ids[j] is not None:
            states[j]=previous['link_states'][i] if exact[j]==ids[j] else 'LOCAL_LINK_UNRESOLVED'
    return ids,model,states


def replay(source,out,keep_local_objects=False):
    out.mkdir(parents=True,exist_ok=False)
    cal=json.loads((source/'shift_rule_summary.json').read_text('utf-8'))['calibration']
    pitch=cal['regression_pitch_px'];cut=cal['calibration_end_frame']
    previous_frames={};safe_states={};journal={};counters=Counter();stats=Counter();timings=Counter();records=[];windows=[];queue=[]
    for line in (source/'states.jsonl').open(encoding='utf-8'):
        r=json.loads(line);t,fid=r['table'],r['frame']
        gray=np.frombuffer(zlib.decompress(base64.b64decode(r['zlib_gray'])),np.uint8).reshape(r['shape']).copy()
        all_spans=physical_spans(gray);spans,edge=complete_spans(all_spans)
        if t not in previous_frames:
            ids=[]
            for j,span in enumerate(spans):
                counters[t]+=1;rid=f'ADJ:T{t}:R{counters[t]:06d}';ids.append(rid)
                journal[rid]={'table':t,'birth_frame':fid,'initial':True,'initial_span':span}
            previous_frames[t]={'frame':fid,'gray':gray,'observed_spans':spans,'model_spans':spans,'row_ids':ids,'edge_spans':edge,'link_states':['INITIAL_VISIBLE']*len(ids)}
            safe_states[t]={'frame':fid,'row_ids':ids,'model_spans':spans,'link_states':['INITIAL_VISIBLE']*len(ids)}
            windows.append({'table':t,'frame':fid,'initial':True,'row_ids':ids,'observed_spans':spans,'model_spans':spans,'edge_spans':edge})
            continue
        previous=previous_frames[t];safe_before=safe_states[t]
        tick=time.perf_counter();gate=global_scroll(previous['gray'],gray);timings['zonal_ncc_seconds']+=time.perf_counter()-tick
        candidate=gate['classification'];guard=None;admitted=[];carried=[];removed=[];proxy=None;actual=None;metrics={};checks=[]
        current_ids=[None]*len(spans);model_spans=[list(s) for s in spans];link_states=['UNKNOWN']*len(spans)
        if candidate=='GLOBAL_SCROLL':
            guard=symmetry(previous['observed_spans'],spans,gate['shift_px'],pitch)
            if not guard['passed']:
                gate['classification']='UNRESOLVED';gate['reason']='SYMMETRY:'+guard['reason']
            else:
                n=guard['N'];shift=gate['shift_px'];new_ids=[]
                for j in range(len(spans)-n,len(spans)):
                    counters[t]+=1;rid=f'ADJ:T{t}:R{counters[t]:06d}';new_ids.append(rid)
                    source_index=j+len(edge);crop_span=padded_row_span(all_spans,source_index,gray.shape[0])
                    pixels=np.ascontiguousarray(gray[crop_span[0]:crop_span[1]]);ok,packed=cv2.imencode('.png',pixels);assert ok
                    queue.append({'item_id':rid,'table':t,'frame':fid,'frame_from':previous['frame'],
                        'source_pts':r['pts'],'time_base':r['time_base'],'classification':'GLOBAL_SCROLL',
                        'row_index':j,'source_span_index':source_index,'row_span':spans[j],'padded_span':crop_span,
                        'N_OUT':n,'N_IN':n,'new_rows_in_transition':n,'SHIFT_PX':shift,'ROW_STEP_PX':pitch,
                        'source_chat_sha256':r['sha256'],'source_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),
                        'png_base64':base64.b64encode(packed).decode()});admitted.append(rid)
                    journal[rid]={'table':t,'birth_frame':fid,'initial':False,'initial_span':spans[j]}
                current_ids,model_spans=transport(previous['row_ids'],previous['model_spans'],spans,shift,n,new_ids)
                for j in range(len(spans)-n):
                    i=j+n;parent=previous['link_states'][i]
                    link_states[j]='GLOBAL_CONFIRMED_CARRY' if parent in ['INITIAL_VISIBLE','GLOBAL_NEW_BOTTOM','GLOBAL_CONFIRMED_CARRY'] and current_ids[j] is not None else 'GLOBAL_PARENT_UNRESOLVED'
                    carried.append({'old_index':i,'new_index':j,'row_id':previous['row_ids'][i],'parent_link_state':parent,
                        'old_span':previous['model_spans'][i],'translated_span':model_spans[j],'observed_span':spans[j],'OCR_submitted':False})
                removed=previous['row_ids'][:n]
                link_states[-n:]=['GLOBAL_NEW_BOTTOM']*n
                safe_states[t]={'frame':fid,'row_ids':current_ids,'model_spans':model_spans,'link_states':link_states}
                # Post-decision audit. This reference cannot select IDs or N.
                proxy=row_ncc_reference(previous['gray'],gray,previous['observed_spans'],spans)
                mapping={int(j):i for j,i in proxy['mapping'].items()} if proxy else {}
                bottom=[j for j,i in mapping.items() if i==len(previous['observed_spans'])-1]
                if len(bottom)==1:
                    j=bottom[0];run=0
                    while j-run>=0 and mapping.get(j-run)==len(previous['observed_spans'])-1-run:run+=1
                    if run>=6:actual=len(spans)-j-1
                metrics,checks=get_metric(previous['row_ids'],current_ids,mapping)
                metrics['ID_TRANSFER_UNRESOLVED']+=sum(c['new_index'] not in mapping for c in carried)
                metrics['ID_TRANSFER_CONFIRMED_PARENT_CORRECT']=sum(c['status']=='ID_TRANSFER_CORRECT' and previous['link_states'][c['physical_old_index_proxy']] in ['INITIAL_VISIBLE','GLOBAL_NEW_BOTTOM','GLOBAL_CONFIRMED_CARRY'] for c in checks)
        if gate['classification']=='LOCAL_CHANGE':
            if keep_local_objects:current_ids,model_spans,link_states=local_object_links(previous,gray,spans)
            else:
                current_ids,model_spans=stationary_links(previous,gray,spans)
                old_index={tuple(s):i for i,s in enumerate(previous['observed_spans'])}
                link_states=[previous['link_states'][old_index[tuple(s)]] if rid is not None else 'UNKNOWN' for s,rid in zip(spans,current_ids)]
        assert gate['classification']=='GLOBAL_SCROLL' or not admitted
        assert gate['classification']!='GLOBAL_SCROLL' or len(admitted)==len(removed)==guard['N']
        safe_unchanged=safe_states[t]==safe_before
        assert gate['classification']=='GLOBAL_SCROLL' or safe_unchanged
        # This update is unconditional: SAFE_ID_STATE is never the NCC source.
        previous_frames[t]={'frame':fid,'gray':gray,'observed_spans':spans,'model_spans':model_spans,'row_ids':current_ids,'edge_spans':edge,'link_states':link_states}
        rec={'table':t,'frame_from':previous['frame'],'frame_to':fid,'candidate_classification':candidate,
             'classification':gate['classification'],'reason':gate['reason'],'SHIFT_PX':gate['shift_px'],'ROW_STEP_PX':pitch,
             'symmetry':guard,'zones':gate['zones'],'removed_top_ids':removed,'carried_rows':carried,
             'old_ids':previous['row_ids'],'old_model_spans':previous['model_spans'],'old_observed_spans':previous['observed_spans'],
             'new_observed_spans':spans,'new_ids':current_ids,'new_model_spans':model_spans,'OCR_queue_row_ids':admitted,
             'old_edge_spans':previous['edge_spans'],'new_edge_spans':edge,
             'local_pixel_reference':proxy,'actual_new_rows':actual,'evaluation':fid>cut,'checkpoint_committed':bool(admitted),
             'ID_transfer_metrics':dict(metrics),'ID_transfer_checks':checks,'safe_frame_from':safe_before['frame'],
             'SAFE_ID_STATE_UNCHANGED':safe_unchanged,'PREVIOUS_FRAME_UPDATED':previous_frames[t]['frame']==fid,
             'old_link_states':previous['link_states'],'new_link_states':link_states,
             'visible_unknown_links':sum(i is None for i in current_ids),'symmetry_geometry_source':'COMPLETE_RAW_BANDS_TOP_EDGE_SEPARATE'}
        records.append(rec);windows.append({'table':t,'frame':fid,'initial':False,'row_ids':current_ids,'observed_spans':spans,'model_spans':model_spans,'edge_spans':edge})
        if fid>cut:
            stats['TESTED_STATES']+=1;stats[gate['classification']+'_COUNT']+=1
            stats['GLOBAL_CANDIDATES']+=candidate=='GLOBAL_SCROLL';stats['BROKEN_SYMMETRY_COUNT']+=bool(guard and not guard['passed'])
            stats['PREVIOUS_FRAME_UPDATE_WRONG']+=not rec['PREVIOUS_FRAME_UPDATED']
            stats['NON_GLOBAL_SAFE_STATE_MUTATED']+=gate['classification']!='GLOBAL_SCROLL' and not safe_unchanged
            stats['LOCAL_KNOWN_LINKS_KEPT']+=sum(i is not None for i in current_ids) if gate['classification']=='LOCAL_CHANGE' else 0
            if admitted:
                stats['OCR_NEW_BOTTOM_ROWS']+=len(admitted);stats['ID_CARRY_COUNT']+=len(carried);stats.update(metrics)
                stats['ID_CARRY_PARENT_UNRESOLVED']+=sum(c['parent_link_state'] not in ['INITIAL_VISIBLE','GLOBAL_NEW_BOTTOM','GLOBAL_CONFIRMED_CARRY'] or c['row_id'] is None for c in carried)
                if actual is None:stats['GLOBAL_COUNT_UNKNOWN']+=1
                else:
                    stats['GLOBAL_COUNT_VALIDATED']+=1;stats['GLOBAL_COUNT_CORRECT']+=actual==guard['N'];stats['FALSE_GLOBAL_SCROLL']+=actual!=guard['N']
    for name,data in [('symmetric_transitions',records),('confirmed_windows',windows),('zonal_ocr_queue',queue)]:
        with (out/f'{name}.jsonl').open('w',encoding='utf-8') as f:
            for r in data:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    (out/'safe_object_journal.json').write_text(json.dumps(journal,ensure_ascii=False,indent=2),'utf-8')
    (out/'safe_id_states.json').write_text(json.dumps(safe_states,ensure_ascii=False,indent=2),'utf-8')
    for name in ['FALSE_GLOBAL_SCROLL','ID_TRANSFER_CORRECT','ID_TRANSFER_WRONG','ID_TRANSFER_UNRESOLVED','DUPLICATE_ID','LOST_ID']:stats.setdefault(name,0)
    result={'statistics':dict(stats),'timing_seconds':dict(timings),'all_frames_OCR_queued_rows':len(queue),
            'all_frames_OCR_repeated_row_ids':len(queue)-len({i['item_id'] for i in queue}),
            'queue_sha256':hashlib.sha256((out/'zonal_ocr_queue.jsonl').read_bytes()).hexdigest(),
            'input':str(source.resolve()),'OCR_executed':False,'verdict':'NOT_PROVEN',
            'checkpoint_policy':'NCC always previous adjacent saved state; SAFE_ID_STATE committed GLOBAL only. Exact stationary LOCAL observation links; UNKNOWN after unconfirmed motion. SAFE journal never fills unknown links.',
            'edge_policy':'First raw band at y<=1 is boundary fragment outside complete-row inventory; complete first band retained.',
            'local_object_policy':'KEEP geometrically compatible objects with propagated LOCAL_LINK_UNRESOLVED; exact native pixels do not cure earlier uncertainty' if keep_local_objects else 'Exact stationary known links only; changed LOCAL becomesNone',
            'contract':'GLOBAL+ordered complete-row symmetry => deleteN top slots, exactSHIFT remaining knownIDs, N bottom newIDs/OCR; unknown parents remainNone.'}
    (out/'zonal_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['replay','ocr']);p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--keep-local-objects',action='store_true');a=p.parse_args()
    if a.mode=='replay':replay(a.input,a.output,a.keep_local_objects)
    else:recognize(a.output)
