"""Audit frozen symmetric receipts; pixel proxies never feed tracker decisions."""
import base64
from collections import Counter
import csv
import html
import json
from pathlib import Path
import sys
import zlib
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.vertical_chat_id_transfer_audit import annotated
from experiments.vertical_chat_symmetric_tracker import symmetry

def audit(source,out):
    records=list(map(json.loads,(out/'symmetric_transitions.jsonl').open(encoding='utf-8')))
    adjacent_mode=bool(records and 'PREVIOUS_FRAME_UPDATED' in records[0])
    groups={'evaluation':Counter(),'all_frames_including_calibration':Counter()}
    selected=[];quotas=Counter();max_observed_residual=0.;repeated_rasters=[]
    for r in records:
        if r['classification']!='GLOBAL_SCROLL':continue
        for name in ['all_frames_including_calibration']+(['evaluation'] if r['evaluation'] else []):
            m=groups[name];m['GLOBAL_SCROLL_COUNT']+=1;m.update(r['ID_transfer_metrics'])
            m['SYMMETRY_CHECKED']+=1
            m['SYMMETRY_WRONG']+=not (r['symmetry']['N']==r['symmetry']['N_OUT']==r['symmetry']['N_IN']==len(r['removed_top_ids'])==len(r['OCR_queue_row_ids']))
            for c in r['carried_rows']:
                m['EXACT_COORDINATE_CHECKED']+=1
                m['EXACT_COORDINATE_WRONG']+=c['translated_span']!=[v-r['SHIFT_PX'] for v in c['old_span']]
                m['ID_RECEIPT_WRONG']+=r['old_ids'][c['old_index']]!=r['new_ids'][c['new_index']]
        for c in r['carried_rows']:
            i,j=c['old_index'],c['new_index']
            residual=(sum(r['old_observed_spans'][i])-sum(r['new_observed_spans'][j]))/2-r['SHIFT_PX']
            max_observed_residual=max(max_observed_residual,abs(residual))
    for r in sorted((r for r in records if r['classification']=='GLOBAL_SCROLL'),key=lambda r:(not r['evaluation'],r['actual_new_rows'] is None)):
        n=r['symmetry']['N'];kind=f'{n}_ROWS' if n<=3 else 'FAST_SCROLL'
        if quotas[kind]<(5 if n<=3 else 3):selected.append((kind,r));quotas[kind]+=1
    # LOCAL is a no-commit control. Observed AFTER has no asserted identity;
    # the confirmed old inventory is stored in JSON, never projected onto it.
    for r in records:
        if len(selected)>=20:break
        if r['classification']=='LOCAL_CHANGE' and r['table']==9 and r['frame_to'] in [1471,1067]:selected.append(('LOCAL_NO_COMMIT',r))
    if len(selected)<20:
        for r in records:
            if len(selected)>=20:break
            if r['classification']=='UNRESOLVED' and r['table']==9 and r['frame_to'] in [1471,1067]:selected.append(('UNRESOLVED_NO_COMMIT',r))
    # Separate adjacent-frame symmetry audit preserves the original transition
    # denominator; it does not inherit the stale confirmed-checkpoint policy.
    views={(r['table'],r['frame']):r for r in map(json.loads,(source/'zonal_tracker_views.jsonl').open(encoding='utf-8'))}
    adjacent=[];adj_counts=Counter();failure_examples=[];reason_quotas=Counter()
    adjacent_source=[r for r in records if r['evaluation']] if adjacent_mode else map(json.loads,(source/'zonal_transitions.jsonl').open(encoding='utf-8'))
    for r in adjacent_source:
        if r.get('candidate_classification',r['classification'])!='GLOBAL_SCROLL':continue
        t,f1,f2=r['table'],r['frame_from'],r['frame_to']
        old,new=({'spans':r['old_observed_spans']},{'spans':r['new_observed_spans']}) if adjacent_mode else (views[(t,f1)],views[(t,f2)])
        result=symmetry(old['spans'],new['spans'],r['SHIFT_PX'],r['ROW_STEP_PX'])
        rec={'table':t,'frame_from':f1,'frame_to':f2,'SHIFT_PX':r['SHIFT_PX'],'symmetry':result,
             'classification_after_symmetry':'GLOBAL_SCROLL' if result['passed'] else 'UNRESOLVED',
             'old_spans':old['spans'],'new_spans':new['spans'],'zones':r['zones']}
        adjacent.append(rec);adj_counts['tested_global_candidates']+=1;adj_counts[result['reason']]+=1
        if not result['passed']:
            failure_examples.append(rec);reason_quotas[result['reason']]+=1
    unresolved=[r for r in records if r['evaluation'] and r['classification']=='UNRESOLVED']
    zone_count=Counter(sum(z['accepted'] for z in r['zones'].values()) for r in unresolved)
    gaps=[r['frame_to']-r['frame_from'] for r in unresolved]
    adjacent_records={(r['table'],r['frame_to']):r for r in map(json.loads,(source/'zonal_transitions.jsonl').open(encoding='utf-8'))}
    checkpoint_examples=[r for r in unresolved if (r['table'],r['frame_to']) in [(1,1000),(2,1000),(8,1028)]]
    if adjacent_mode:checkpoint_examples=unresolved
    diagnostic={'UNRESOLVED':len(unresolved),'accepted_zones_distribution':dict(zone_count),'checkpoint_gap_median_frames':float(np.median(gaps)),
                'checkpoint_gap_over100':sum(g>100 for g in gaps),'checkpoint_gap_over500':sum(g>500 for g in gaps),
                'direct_reason':'INSUFFICIENT_CONFIDENT_ZONES; symmetry not reached',
                'structural_reason':'Rejected LOCAL/UNRESOLVED cannot mutate confirmed identity. Frozen checkpoint loses current-content overlap.',
                'examples':[{'checkpoint':r,'adjacent':adjacent_records[(r['table'],r['frame_to'])]} for r in checkpoint_examples]}
    if adjacent_mode:
        diagnostic['comparison_source']='ALWAYS_PREVIOUS_ADJACENT_SAVED_FRAME'
        diagnostic['structural_reason']='Remaining low confidence uses adjacent frames; no frozen SAFE raster enters motion NCC.'
        diagnostic['reasons']=dict(Counter(r['reason'] for r in unresolved))
    (out/'checkpoint_diagnostics.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2),'utf-8')
    needed={(r['table'],f) for _,r in selected for f in [r['frame_from'],r['frame_to']]}
    needed.update((r['table'],f) for r in failure_examples for f in [r['frame_from'],r['frame_to']]);images={}
    needed.update((r['table'],f) for r in checkpoint_examples for f in [r['frame_from'],r['frame_to']])
    for line in (source/'states.jsonl').open(encoding='utf-8'):
        r=json.loads(line);key=(r['table'],r['frame'])
        if key in needed:images[key]=np.frombuffer(zlib.decompress(base64.b64decode(r['zlib_gray'])),np.uint8).reshape(r['shape']).copy()
    gallery=out/'id_transfer_cases';gallery.mkdir(exist_ok=True);sections=[];examples=[]
    for kind,r in selected:
        t,f1,f2=r['table'],r['frame_from'],r['frame_to'];key=f'T{t:02d}_{f1:06d}_{f2:06d}'
        before=annotated(images[(t,f1)],r['old_observed_spans'],r['old_ids'],f'BEFORE {f1}: IDs (? = uncertain)',r.get('old_link_states'))
        ids=r['new_ids'] if r['new_ids'] is not None else [None]*len(r['new_observed_spans'])
        after=annotated(images[(t,f2)],r['new_observed_spans'],ids,f'AFTER {f2}: {kind}, SHIFT {r["SHIFT_PX"]}',r.get('new_link_states'))
        cv2.imwrite(str(gallery/f'{key}.png'),np.hstack([before,after]))
        cv2.imwrite(str(gallery/f'{key}_before_raw.png'),images[(t,f1)])
        cv2.imwrite(str(gallery/f'{key}_after_raw.png'),images[(t,f2)])
        (gallery/f'{key}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),'utf-8')
        examples.append({'key':key,'kind':kind,'evaluation':r['evaluation'],'path':str((gallery/f'{key}.png').resolve())})
        sections.append(f'<h2>{key}: {kind}; calibration={not r["evaluation"]}</h2><p>Same color = same stored row_id. Pairwise physical evidence is separately listed; UNKNOWN is not proof.</p><img src="id_transfer_cases/{key}.png"><pre>{html.escape(json.dumps(r,ensure_ascii=False,indent=2))}</pre>')
    failure_gallery=out/'symmetry_failure_cases';failure_gallery.mkdir(exist_ok=True);failure_sections=[]
    for r in failure_examples:
        t,f1,f2=r['table'],r['frame_from'],r['frame_to'];key=f'T{t:02d}_{f1:06d}_{f2:06d}'
        before=annotated(images[(t,f1)],r['old_spans'],[None]*len(r['old_spans']),f'BEFORE {f1}: raw bands')
        after=annotated(images[(t,f2)],r['new_spans'],[None]*len(r['new_spans']),f'AFTER {f2}: symmetry UNRESOLVED')
        cv2.imwrite(str(failure_gallery/f'{key}.png'),np.hstack([before,after]))
        (failure_gallery/f'{key}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),'utf-8')
        failure_sections.append(f'<h2>{key}: {r["symmetry"]["reason"]}</h2><img src="symmetry_failure_cases/{key}.png"><pre>{html.escape(json.dumps(r,ensure_ascii=False,indent=2))}</pre>')
    for r in checkpoint_examples:
        t,f1,f2=r['table'],r['frame_from'],r['frame_to'];key=f'{"ADJACENT" if adjacent_mode else "CHECKPOINT"}_T{t:02d}_{f1:06d}_{f2:06d}'
        cv2.imwrite(str(failure_gallery/f'{key}.png'),np.hstack([images[(t,f1)],images[(t,f2)]]))
        failure_sections.append(f'<h2>{key}: {"adjacent frame low confidence" if adjacent_mode else "stale checkpoint, insufficient confident zones"}</h2><img src="symmetry_failure_cases/{key}.png">')
        (failure_gallery/f'{key}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),'utf-8')
    (out/'adjacent_symmetry_audit.json').write_text(json.dumps({'statistics':dict(adj_counts),'transitions':adjacent},ensure_ascii=False,indent=2),'utf-8')
    (out/'symmetry_failure_review.html').write_text('<meta charset="utf-8"><style>body{font-family:monospace}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Adjacent GLOBAL candidates rejected by observed-window symmetry</h1>'+''.join(failure_sections),'utf-8')
    fields=['table','frame_from','frame_to','evaluation','classification','ROW_STEP_PX','SHIFT_PX','SHIFT/ROW_STEP','predicted_new_rows','actual_new_rows','error','N_OUT','N_IN','checkpoint_committed','reason']
    with (out/'shift_to_new_rows.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader()
        for r in records:
            row={k:r.get(k) for k in fields};shift=r['SHIFT_PX'];ratio=shift/r['ROW_STEP_PX'] if shift is not None else None
            n=r['symmetry']['N'] if r['classification']=='GLOBAL_SCROLL' else None
            row.update({'SHIFT/ROW_STEP':ratio,'predicted_new_rows':n,'error':n-r['actual_new_rows'] if n is not None and r['actual_new_rows'] is not None else None,
                        'N_OUT':r['symmetry'].get('N_OUT') if r['symmetry'] else None,'N_IN':r['symmetry'].get('N_IN') if r['symmetry'] else None})
            writer.writerow(row)
    for m in groups.values():
        for k in ['ID_TRANSFER_CORRECT','ID_TRANSFER_WRONG','ID_TRANSFER_UNRESOLVED','DUPLICATE_ID','LOST_ID','SYMMETRY_WRONG','EXACT_COORDINATE_WRONG','ID_RECEIPT_WRONG']:m.setdefault(k,0)
    result={'metrics_by_scope':{k:dict(v) for k,v in groups.items()},'observed_glyph_center_max_residual_px':max_observed_residual,
            'examples':examples,'saved_visual_examples':len(examples),'physical_reference':'PAIRWISE_LOCAL_ORDERED_PIXEL_PROXY; NOT_COMPLETE_HUMAN_GROUND_TRUTH',
            'no_commit_control':'No fresh objects, OCR or SAFE-state mutation. Provisional LOCAL object links are labelled?; no stronger identity assertion follows from matching ID strings.',
            'same_text_control':'Synthetic identical-raster rows remain unresolved; OCR text never selects physical identity. Real repeated-text safety is not established.',
            'verdict':'OLD_ROW_ID_TRANSFER_SAFE_NOT_PROVEN'}
    (out/'id_transfer_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    (out/'id_transfer_review.html').write_text('<meta charset="utf-8"><style>body{background:#222;color:#eee;font-family:monospace}img{max-width:100%;image-rendering:pixelated}pre{white-space:pre-wrap}</style><h1>Transactional row_id transfer: raw observed boxes, stored IDs, explicit uncertainty</h1>'+''.join(sections),'utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':audit(Path(sys.argv[1]),Path(sys.argv[2]))
