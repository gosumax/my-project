"""Read-only ID transport audit against independently ordered local pixels.

Correctness is pair-wise proxy evidence, not certified global physical identity.
Raw images and observed glyph boxes are never shifted or corrected for labels.
"""
import argparse
import base64
from collections import Counter
import csv
import html
import hashlib
import json
from pathlib import Path
import sys
import zlib
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from parser_core.chat_rows import physical_spans,exclude_first_visible_span


def annotated(gray,spans,ids,side,link_states=None):
    image=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);w=gray.shape[1]
    image=cv2.copyMakeBorder(image,25,0,0,240,cv2.BORDER_CONSTANT,value=(28,28,28))
    cv2.putText(image,side,(5,17),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
    for j,((a,b),rid) in enumerate(zip(spans,ids)):
        seed=hashlib.sha256((rid or 'UNKNOWN').encode()).digest()
        color=tuple(45+int(v)%190 for v in seed[:3]) if rid else (150,150,150)
        cv2.rectangle(image,(0,a+25),(w-1,b+25),color,1)
        uncertain=link_states is not None and ('UNRESOLVED' in link_states[j] or link_states[j]=='UNKNOWN')
        label=('? ' if uncertain and rid else '')+(rid or 'UNKNOWN')
        cv2.putText(image,f'{j}: {label}',(w+5,int((a+b)/2)+29),cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
    return image


def get_metric(old_ids,new_ids,ref_mapping):
    metrics=Counter();details=[]
    for j,i in ref_mapping.items():
        expected=old_ids[i];actual=new_ids[j]
        if expected is None or actual is None:
            status='ID_TRANSFER_UNRESOLVED';metrics[status]+=1
        elif actual==expected:
            status='ID_TRANSFER_CORRECT';metrics[status]+=1
        else:
            status='ID_TRANSFER_WRONG';metrics[status]+=1
            if expected not in new_ids:metrics['LOST_ID']+=1
        details.append({'new_index':j,'physical_old_index_proxy':i,'before_id':expected,'after_id':actual,'status':status})
    known=[x for x in new_ids if x is not None]
    metrics['DUPLICATE_ID']=len(known)-len(set(known))
    return metrics,details


def audit(out):
    views={(r['table'],r['frame']):r for r in map(json.loads,(out/'zonal_tracker_views.jsonl').open(encoding='utf-8'))}
    refs={(r['table'],r['after_frame']):r for r in map(json.loads,(out/'row_reference_transitions.jsonl').open(encoding='utf-8'))}
    transitions=list(map(json.loads,(out/'zonal_transitions.jsonl').open(encoding='utf-8')))
    totals=Counter();records=[];selection=[];categories=Counter();raster_needed=set();max_residual=0.
    for r in transitions:
        if r['classification']!='GLOBAL_SCROLL':continue
        t,f1,f2=r['table'],r['frame_from'],r['frame_to'];old=views[(t,f1)];new=views[(t,f2)]
        proxy=refs[(t,f2)]['reference'];ref_map={int(j):i for j,i in proxy['mapping'].items()} if proxy else {}
        totals['GLOBAL_SCROLL_TRANSITIONS']+=1
        metrics,detail=get_metric(old['row_ids'],new['row_ids'],ref_map)
        if not proxy:
            # No matching raster proxy: every coordinate transport remains
            # physically unresolved, even if the strings of IDs are equal.
            metrics['ID_TRANSFER_UNRESOLVED']+=len(new['carried_rows'])
        else:
            metrics['ID_TRANSFER_UNRESOLVED']+=sum(c['new_index'] not in ref_map for c in new['carried_rows'])
        for c in new['carried_rows']:
            totals['DECLARED_COORD_TRANSLATION_CHECKED']+=1
            exact=c['translated_span']==[a-r['SHIFT_PX'] for a in c['old_span']]
            totals['DECLARED_COORD_TRANSLATION_WRONG']+=not exact
            if c['new_index'] in ref_map and ref_map[c['new_index']]==c['old_index']:
                residual=(sum(c['old_span'])-sum(c['observed_span']))/2-r['SHIFT_PX']
                max_residual=max(max_residual,abs(residual));c['native_center_residual_px']=residual
        totals.update(metrics)
        rec={'table':t,'frame_from':f1,'frame_to':f2,'SHIFT_PX':r['SHIFT_PX'],
             'predicted_new_rows':r['predicted_new_rows'],'metrics':dict(metrics),
             'local_row_reference':proxy,'physical_transfer_checks':detail,'carried_rows':new['carried_rows'],
             'old_ids':old['row_ids'],'new_ids':new['row_ids'],'old_spans':old['spans'],'new_spans':new['spans'],
             'old_link_states':old.get('row_link_states',[]),'new_link_states':new.get('row_link_states',[])}
        records.append(rec)
        n=r['predicted_new_rows'];kind=f'{n}_ROW_SCROLL' if n<=3 else 'FAST_SCROLL'
        count_correct=metrics['ID_TRANSFER_CORRECT']
        take=False
        if count_correct>=3 and categories[kind] < (5 if n<=3 else 3):
            categories[kind]+=1;take=True
        if metrics['ID_TRANSFER_WRONG'] and categories['WRONG']<2:
            categories['WRONG']+=1;take=True
        if take and len(selection)<18:
            selection.append((kind,rec));raster_needed.update([(t,f1),(t,f2)])
    # Two localized/changed-crop controls demonstrate no new row_id/OCR allocation.
    local=[r for r in transitions if r['classification']=='LOCAL_CHANGE' and r['table']==9 and r['frame_from'] in [1066,1470]]
    for r in local[:2]:
        t,f1,f2=r['table'],r['frame_from'],r['frame_to'];old=views[(t,f1)];new=views[(t,f2)]
        rec={'table':t,'frame_from':f1,'frame_to':f2,'SHIFT_PX':None,'classification':'LOCAL_CHANGE',
             'old_ids':old['row_ids'],'new_ids':new['row_ids'],'old_spans':old['spans'],'new_spans':new['spans'],
             'new_link_states':new.get('row_link_states',[]),'OCR_queue_row_ids':new['OCR_queue_row_ids'],'zones':r['zones']}
        selection.append(('LOCAL_CHANGE',rec));raster_needed.update([(t,f1),(t,f2)])
    images={};repeated_controls=[]
    for line in (out/'states.jsonl').open(encoding='utf-8'):
        r=json.loads(line);key=(r['table'],r['frame'])
        if key not in raster_needed:continue
        images[key]=np.frombuffer(zlib.decompress(base64.b64decode(r['zlib_gray'])),np.uint8).reshape(r['shape']).copy()
    gallery=out/'id_transfer_cases';gallery.mkdir(exist_ok=True);sections=[];cases=[]
    for kind,r in selection:
        t,f1,f2=r['table'],r['frame_from'],r['frame_to'];key=f'T{t:02d}_{f1:06d}_{f2:06d}'
        before=annotated(images[(t,f1)],r['old_spans'],r['old_ids'],f'BEFORE {f1}')
        after=annotated(images[(t,f2)],r['new_spans'],r['new_ids'],f'AFTER {f2}; shift={r["SHIFT_PX"]}')
        maxh=max(before.shape[0],after.shape[0]);assert before.shape[0]==after.shape[0]
        montage=np.hstack([before,after]);cv2.imwrite(str(gallery/f'{key}.png'),montage)
        (gallery/f'{key}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),'utf-8')
        sections.append(f'<h2>{key} {kind}</h2><img src="id_transfer_cases/{key}.png"><pre>{html.escape(json.dumps(r,ensure_ascii=False,indent=2))}</pre>')
        cases.append({'key':key,'kind':kind,'path':str((gallery/f'{key}.png').resolve())})
    for name in ['ID_TRANSFER_CORRECT','ID_TRANSFER_WRONG','ID_TRANSFER_UNRESOLVED','DUPLICATE_ID','LOST_ID','DECLARED_COORD_TRANSLATION_WRONG']:totals.setdefault(name,0)
    result={'metrics':dict(totals),'native_glyph_center_max_residual_px':max_residual,'saved_visual_examples':len(cases),'examples':cases,
            'physical_identity_evidence':'PAIRWISE_LOCAL_ORDERED_PIXEL_NCC_PROXY; not independent complete human physical-line ground truth',
            'coordinate_contract':'stored tracked geometry moves exactly SHIFT; native observed glyph bounds retain measured rounding/antialias differences',
            'LOST_ID_contract':'Known ID missing after physically matched proxy row; IDs leaving top are not lost errors',
            'count_scope':'181 evaluation GLOBAL transitions; not all initial calibration events',
            'verdict':'OLD_ROW_ID_TRANSFER_SAFE_NOT_PROVEN','production_changed':False}
    with (out/'id_transfer_checks.jsonl').open('w',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    (out/'id_transfer_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    (out/'id_transfer_review.html').write_text('<meta charset="utf-8"><style>body{background:#222;color:#eee;font-family:monospace}img{width:960px;image-rendering:pixelated}pre{white-space:pre-wrap}</style><h1>Same object IDs and measured physical pixel proxy; unresolved links stay explicit</h1>'+''.join(sections),'utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.output)
