"""Isolated zonal gate + coordinate carry + bottom-only OCR admission.

Recognition is a separate optional pass over a frozen queue. Its results can
never influence the pixel tracker, geometry, identities or admission decisions.
"""
import argparse
import base64
from collections import Counter,defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import sys
import time
import zlib
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.vertical_chat_zone_gate import global_scroll
from experiments.vertical_chat_shift import reference,coordinate_map
from parser_core.chat_rows import physical_spans,exclude_first_visible_span,padded_row_span


def frozen_queue(out):
    path=out/'zonal_ocr_queue.jsonl';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    receipt=json.loads((out/'zonal_summary.json').read_text('utf-8'))
    if digest!=receipt['queue_sha256']:raise ValueError('OCR_QUEUE_CHANGED_AFTER_PIXEL_TRACKER_REPLAY')
    queue=[json.loads(r) for r in path.open(encoding='utf-8')]
    if len(queue)!=len({r['item_id'] for r in queue}):raise ValueError('OCR_QUEUE_DUPLICATE_ROW_ID')
    if any(r['classification']!='GLOBAL_SCROLL' for r in queue):raise ValueError('OCR_QUEUE_NON_GLOBAL_ITEM')
    return queue,digest


def carried_ids(old_ids,old_spans,new_spans,shift,new_count):
    """Transport existing IDs; unknown parents remain unknown, no revival."""
    current=[None]*len(new_spans);carry=[];mapping=coordinate_map(old_spans,new_spans,shift)
    for j,i in mapping.items():
        if j>=len(new_spans)-new_count:continue
        current[j]=old_ids[i]
        carry.append({'row_id':old_ids[i],'old_index':i,'new_index':j,
                      'old_span':old_spans[i],'translated_span':[old_spans[i][0]-shift,old_spans[i][1]-shift],
                      'observed_span':new_spans[j],'OCR_submitted':False})
    return current,carry


def replay(out):
    rule=json.loads((out/'shift_rule_summary.json').read_text('utf-8'));pitch=rule['calibration']['regression_pitch_px']
    truth={(int(r['table']),int(r['frame_to'])):r for r in csv.DictReader((out/'shift_to_new_rows.csv').open(encoding='utf-8-sig'))}
    previous={};ids={};link_states={};counters=Counter();statistics=Counter();timings=Counter();rows=[];queue=[];views=[];cases=[]
    cut=rule['calibration']['calibration_end_frame'];out_cases=out/'zonal_cases';out_cases.mkdir(exist_ok=True)
    for line in (out/'states.jsonl').open(encoding='utf-8'):
        r=json.loads(line);t,fid=r['table'],r['frame']
        gray=np.frombuffer(zlib.decompress(base64.b64decode(r['zlib_gray'])),np.uint8).reshape(r['shape']).copy()
        all_spans=physical_spans(gray);spans,suppressed=exclude_first_visible_span(all_spans)
        if t not in previous:
            current=[]
            for _ in spans:
                counters[t]+=1;current.append(f'ZONE:T{t}:R{counters[t]:06d}')
            previous[t]=(fid,gray,spans);ids[t]=current;link_states[t]=['INITIAL_VISIBLE']*len(spans);continue
        old_fid,old,os=previous[t];old_ids=ids[t]
        tick=time.perf_counter();gate=global_scroll(old,gray);timings['zonal_ncc_seconds']+=time.perf_counter()-tick
        new_count=None;mapping={};current=[None]*len(spans);admitted=[];carry=[];current_links=['UNKNOWN']*len(spans)
        if gate['classification']=='GLOBAL_SCROLL':
            shift=gate['shift_px'];new_count=round(shift/pitch)
            if new_count<1 or new_count>len(spans):
                gate['classification']='UNRESOLVED';gate['reason']='INSUFFICIENT_DETECTED_BOTTOM_BANDS';new_count=None
            else:
                # No text/canonical history matching: simply translate old
                # row geometry, retaining IDs on the existing upper bands.
                current,carry=carried_ids(old_ids,os,spans,shift,new_count)
                bottom=set(range(len(spans)-new_count,len(spans)))
                for c in carry:
                    status=link_states[t][c['old_index']]
                    c['parent_link_status']=status
                    current_links[c['new_index']]='CARRIED_PARENT_UNRESOLVED' if 'UNRESOLVED' in status or status=='UNKNOWN' else 'GLOBAL_GEOMETRY_CARRIED'
                for j in sorted(bottom):
                    counters[t]+=1;rid=f'ZONE:T{t}:R{counters[t]:06d}';current[j]=rid;current_links[j]='GLOBAL_NEW_BOTTOM_ROW'
                    source_index=j+len(suppressed);crop_span=padded_row_span(all_spans,source_index,gray.shape[0])
                    pixels=np.ascontiguousarray(gray[crop_span[0]:crop_span[1]])
                    ok,packed=cv2.imencode('.png',pixels);assert ok
                    item={'item_id':rid,'table':t,'frame':fid,'source_pts':r['pts'],'time_base':r['time_base'],
                          'row_index':j,'source_span_index':source_index,'row_span':spans[j],'padded_span':crop_span,
                          'SHIFT_PX':shift,'ROW_STEP_PX':pitch,'classification':'GLOBAL_SCROLL',
                          'new_rows_in_transition':new_count,'source_chat_sha256':r['sha256'],
                          'source_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),
                          'png_base64':base64.b64encode(packed).decode()}
                    queue.append(item);admitted.append(rid)
        if gate['classification']!='GLOBAL_SCROLL':
            # No movement or birth by the formula. Keep existing objects at
            # compatible stationary positions, without claiming pixel identity
            # for a local repaint. Exact anchors strengthen only known links.
            current,_=carried_ids(old_ids,os,spans,0,0)
            current_links=['KEPT_OBJECT_UNRESOLVED_LOCAL_LINK' if rid is not None else 'UNKNOWN' for rid in current]
            anchors=reference(old,gray,os,spans)
            if anchors and anchors['shift']==0:
                for j,i in anchors['mapping'].items():
                    current[j]=old_ids[i]
                    status=link_states[t][i]
                    current_links[j]='KEPT_PARENT_UNRESOLVED' if 'UNRESOLVED' in status or status=='UNKNOWN' else 'EXACT_STATIONARY_ANCHOR'
        assert len(admitted)==(new_count or 0)
        assert gate['classification']=='GLOBAL_SCROLL' or not admitted
        record={'table':t,'frame_from':old_fid,'frame_to':fid,'classification':gate['classification'],
                'reason':gate['reason'],'ROW_STEP_PX':pitch,'SHIFT_PX':gate['shift_px'],
                'predicted_new_rows':new_count,'OCR_queue_row_ids':admitted,'carried_rows':carry,'zones':gate['zones'],
                'row_ids':current,'row_link_states':current_links,'unknown_parent_ids':sum(x is None for x in current[:len(spans)-(new_count or 0)])}
        for name,z in gate['zones'].items():record[name+'_shift_px']=z['shift_px'];record[name+'_accepted']=z['accepted'];record[name+'_score']=z['score'];record[name+'_margin']=z['margin']
        if fid>cut:
            statistics['TESTED_TRANSITIONS']+=1;statistics[gate['classification']+'_COUNT']+=1
            expected=truth[(t,fid)]['actual_new_rows'];record['actual_new_rows']=int(expected) if expected else None
            if gate['classification']=='GLOBAL_SCROLL':
                statistics['OCR_QUEUED_ROWS_EVALUATION']+=len(admitted)
                statistics['CARRIED_KNOWN_IDS']+=sum(c['row_id'] is not None for c in carry)
                statistics['CARRIED_WITH_PARENT_UNRESOLVED']+=sum('UNRESOLVED' in c['parent_link_status'] or c['parent_link_status']=='UNKNOWN' for c in carry)
                if expected:
                    statistics['GLOBAL_SCROLL_COUNT_VALIDATED']+=1
                    statistics['GLOBAL_COUNT_CORRECT']+=new_count==int(expected)
                    statistics['GLOBAL_COUNT_ERROR']+=new_count!=int(expected)
                    statistics['FALSE_GLOBAL_SCROLL']+=int(expected)==0 or new_count!=int(expected)
                else:statistics['GLOBAL_SCROLL_ACTUAL_COUNT_UNKNOWN']+=1
            else:
                if expected and int(expected)>0:statistics['NEW_BIRTHS_NOT_ADMITTED_UNDER_UNCONFIRMED_GLOBAL']+=int(expected)
            # Counterexample plus bounded per-class native controls.
            key=f'T{t:02d}_{old_fid:06d}_{fid:06d}'
            prior_count=statistics[gate['classification']+'_COUNT']
            if (t==9 and old_fid==1470 and fid==1471) or prior_count<=3 or (gate['classification']=='GLOBAL_SCROLL' and expected and new_count!=int(expected)):
                cv2.imwrite(str(out_cases/f'{key}_before.png'),old);cv2.imwrite(str(out_cases/f'{key}_after.png'),gray)
                (out_cases/f'{key}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),'utf-8');cases.append((key,record))
            rows.append(record)
        views.append({'table':t,'frame':fid,'classification':gate['classification'],'spans':spans,'row_ids':current,
                      'row_link_states':current_links,'OCR_queue_row_ids':admitted,'carried_rows':carry})
        previous[t]=(fid,gray,spans);ids[t]=current;link_states[t]=current_links
    for name,data in [('zonal_transitions',rows),('zonal_tracker_views',views),('zonal_ocr_queue',queue)]:
        with (out/f'{name}.jsonl').open('w',encoding='utf-8') as f:
            for r in data:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    fields=['table','frame_from','frame_to','classification','reason','ROW_STEP_PX','SHIFT_PX','predicted_new_rows','actual_new_rows',
            'top_shift_px','top_accepted','top_score','top_margin','middle_shift_px','middle_accepted','middle_score','middle_margin',
            'bottom_shift_px','bottom_accepted','bottom_score','bottom_margin','unknown_parent_ids']
    with (out/'zonal_transitions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    sections=[]
    for key,r in cases:sections.append(f'<h2>{key} {r["classification"]}</h2><img src="zonal_cases/{key}_before.png"><img src="zonal_cases/{key}_after.png"><pre>{html.escape(json.dumps(r,ensure_ascii=False,indent=2))}</pre>')
    (out/'zonal_review.html').write_text('<meta charset="utf-8"><style>body{background:#222;color:#eee;font-family:monospace}img{width:456px;image-rendering:pixelated}pre{white-space:pre-wrap}</style>'+''.join(sections),'utf-8')
    statistics.setdefault('FALSE_GLOBAL_SCROLL',0)
    before=json.loads((out/'source_before.json').read_text('utf-8'))
    changed=[p for p,h in before.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    result={'statistics':dict(statistics),'timing_seconds':dict(timings),'all_frames_OCR_queued_rows':len(queue),
            'all_frames_OCR_repeated_row_ids':len(queue)-len({r['item_id'] for r in queue}),
            'queue_sha256':hashlib.sha256((out/'zonal_ocr_queue.jsonl').read_bytes()).hexdigest(),
            'counterexample_T09_1470_1471':next(r for r in rows if r['table']==9 and r['frame_from']==1470 and r['frame_to']==1471),
            'contract':'At least2 confident agreeing shifts>2px within2px, any confident disagreeing zone vetoes. GLOBAL only => bottom round(shift/pitch) rows queued, old IDs coordinate-carried. LOCAL/UNRESOLVED => no formula/no OCR/no newID, compatible existing objects KEEP with UNRESOLVED pixel links. Unknown parents never invented.',
            'zone_min_overlap_px':48,'score_threshold':.90,'margin_threshold':.02,
            'false_global_scope':'Count errors on independent local bottom-context-covered subset only; UNKNOWN global cases not asserted correct.',
            'OCR_executed':False,'production_changed':bool(changed),'changed_registered_files':changed,'verdict':'NOT_PROVEN'}
    (out/'zonal_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


def recognize(out):
    from parser_core.tiny_chat_client import TinyChatClient
    queue_path=out/'zonal_ocr_queue.jsonl';queue,before=frozen_queue(out)
    conf=json.loads((ROOT/'configs/ocr_tiny_local.json').read_text('utf-8'))
    client=TinyChatClient(Path(conf['python']),Path(conf['model_dir']),conf['model_name'],conf['weights_sha256'],conf['device'],out/'zonal_ocr_worker.stderr.log')
    by_table=defaultdict(list)
    for item in queue:
        assert item['classification']=='GLOBAL_SCROLL';by_table[item['table']].append(item)
    started=time.perf_counter();count=0;errors=0
    try:
        with (out/'zonal_raw_ocr_results.jsonl').open('w',encoding='utf-8') as sink:
            for t,items in sorted(by_table.items()):
                for start in range(0,len(items),8):
                    batch=items[start:start+8];response=client.read_batch(batch,8)
                    got=[r['item_id'] for r in response['results']]
                    if len(got)!=len(set(got)) or set(got)!={r['item_id'] for r in batch}:
                        raise ValueError('OCR_RESPONSE_ITEM_ID_PARITY_FAILED')
                    sink.write(json.dumps({'table':t,'results':response},ensure_ascii=False)+'\n')
                    count+=len(response['results']);errors+=sum(r.get('status')=='ERROR' for r in response['results'])
    finally:client.close()
    after=hashlib.sha256(queue_path.read_bytes()).hexdigest();assert before==after
    result={'OCR_results':count,'queue_rows':len(queue),'OCR_errors':errors,'seconds_including_startup':time.perf_counter()-started,
            'queue_before_sha256':before,'queue_after_sha256':after,'queue_unchanged':True,
            'identity_inputs_from_OCR':0,'whole_CHAT_requests':0,'old_row_requests':0,'initial_visible_row_requests':0,
            'only_GLOBAL_new_bottom_rows':True}
    (out/'zonal_ocr_receipt.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['replay','ocr']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    (replay if a.mode=='replay' else recognize)(a.output)
