"""Geometry/row-order audit of SHIFT / pitch, with no OCR or text identity.

Counting suffix requires the old bottom row to be located independently via
local ordered row pixels. The global NCC shift is never used to create labels.
"""
import argparse
import base64
from collections import Counter,defaultdict
import csv
import html
import json
from pathlib import Path
import sys
import zlib
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.vertical_chat_shift import row_ncc_reference
from parser_core.chat_rows import physical_spans,exclude_first_visible_span


def audit(out):
    captures=json.loads((out/'capture.json').read_text('utf-8'))
    pairs={}
    for line in (out/'row_reference_transitions.jsonl').open(encoding='utf-8'):
        r=json.loads(line); pairs[(r['table'],r['after_frame'])]=r
    cut=json.loads((out/'row_reference_summary.json').read_text('utf-8'))['calibration_end_frame']
    unique_geometry={}; gaps=[]; previous={}; data=[]; images={}; diagnostics=[]; temporal_calibration=[]
    for line in (out/'states.jsonl').open(encoding='utf-8'):
        r=json.loads(line); t=r['table']; fid=r['frame']
        gray=np.frombuffer(zlib.decompress(base64.b64decode(r['zlib_gray'])),np.uint8).reshape(r['shape']).copy()
        spans=exclude_first_visible_span(physical_spans(gray))[0]
        centers=np.array([(a+b)/2 for a,b in spans])
        if fid<=cut and len(spans)>=10:
            key=(t,tuple(spans))
            slope=float(np.polyfit(np.arange(len(centers)),centers,1)[0])
            unique_geometry[key]=slope; gaps.extend(np.diff(centers).tolist())
        pair=pairs.get((t,fid))
        if pair and fid<=cut and pair['reference']:
            ref=pair['reference']; mapping={int(j):i for j,i in ref['mapping'].items()}; os=previous[t][2]
            bottom=[j for j,i in mapping.items() if i==len(os)-1]
            if len(bottom)==1:
                j=bottom[0]; run=0
                while j-run>=0 and len(os)-1-run>=0 and mapping.get(j-run)==len(os)-1-run:run+=1
                n=len(spans)-j-1
                if run>=6 and n>0 and ref['shift']>0:
                    temporal_calibration.append({'table':t,'frame_from':previous[t][0],'frame_to':fid,
                       'actual_suffix_rows':n,'row_displacement_px':ref['shift'],'step_from_physical_suffix_px':ref['shift']/n})
        if pair and pair['evaluation']:
            old_fid,old,os=previous[t]
            ref=pair['reference']; alg=pair['algorithms']['NCC_MASK']['algorithm']
            known=False; actual=None; last_j=None; unknown_reason='NO_UNIQUE_LOCAL_ROW_OVERLAP'; bottom_run=0
            if ref:
                mapping={int(j):i for j,i in ref['mapping'].items()}
                bottom=[j for j,i in mapping.items() if i==len(os)-1]
                if len(bottom)==1:
                    last_j=bottom[0]
                    run=0
                    while mapping.get(last_j-run)==len(os)-1-run and last_j-run>=0 and len(os)-1-run>=0:
                        run+=1
                    bottom_run=run;unknown_reason='BOTTOM_CONTEXT_TOO_SHORT_OR_INTERRUPTED'
                    if run>=6:
                        actual=len(spans)-last_j-1; known=True
                else:unknown_reason='BOTTOM_ROW_NOT_LOCATED_BY_LOCAL_PIXELS'
            row={'table':t,'frame_from':old_fid,'frame_to':fid,'SHIFT_PX':alg['shift'],
                 'NCC_score':alg['score'],'NCC_margin':alg['margin'],'NCC_accepted':alg['accepted'],
                 'actual_new_rows':actual,'actual_count_status':'LOCAL_PIXEL_OLD_BOTTOM_MIN6_CONTIGUOUS_ANCHORS' if known else 'UNKNOWN',
                 'unknown_actual_reason':None if known else unknown_reason,'bottom_context_length':bottom_run,
                 'reference_shift':ref['shift'] if ref else None,'row_displacement_range':ref.get('displacement_range') if ref else None,
                 'bottom_old_index':len(os)-1,'bottom_new_index':last_j,'crop_box_fixed':list(r['box']),
                 'before_sha256':pair['before_sha256'],'after_sha256':pair['after_sha256']}
            data.append(row)
            # Retain both crops for ALL unresolved positive shifts, all eventual
            # count failures, and bounded useful 1/2/3/multiple controls.
            need=alg['shift']>0 or not alg['accepted'] or (actual is not None and actual>0)
            if need:
                key=f'T{t:02d}_{old_fid:06d}_{fid:06d}'
                images[key]=(old,gray)
                diagnostics.append((key,pair,os,spans))
        previous[t]=(fid,gray,spans)
    slope_values=list(unique_geometry.values()); pitch=float(np.median(slope_values))
    calibration={'modal_gap_px':float(Counter(gaps).most_common(1)[0][0]),'regression_pitch_px':pitch,
                 'raw_adjacent_gap_count':len(gaps),'unique_table_geometry_states':len(unique_geometry),
                 'fit_slope_p05_p95':np.percentile(slope_values,[5,95]).tolist(),
                 'calibration_end_frame':cut,'per_table':{t:{'median':float(np.median([v for k,v in unique_geometry.items() if k[0]==t])),
                  'n':sum(k[0]==t for k in unique_geometry)} for t in range(1,10)}}
    if temporal_calibration:
        values=[r['step_from_physical_suffix_px'] for r in temporal_calibration]
        calibration['independent_calibration_scroll_samples']=len(values)
        calibration['step_from_physical_suffix_median_px']=float(np.median(values))
        calibration['step_from_physical_suffix_p05_p95_px']=np.percentile(values,[5,95]).tolist()
        calibration['temporal_calibration_samples']=temporal_calibration
    counts=Counter(); positive=Counter(); accepted=Counter(); dist=Counter(); known_dist=Counter(); failures=[]; groups=Counter(); unknown_groups=Counter()
    for row in data:
        shift=row['SHIFT_PX']; ratio=shift/pitch; pred=round(ratio); actual=row['actual_new_rows']
        row.update({'ROW_STEP_PX':pitch,'SHIFT/ROW_STEP':ratio,'predicted_new_rows':pred,
                    'quantization_residual_px':shift-pred*pitch,'tolerance_px':2.,
                    'integer_ratio_within_0.10':abs(ratio-pred)<=.10,
                    'within_tolerance_px':abs(shift-pred*pitch)<=2.,'error':pred-actual if actual is not None else None})
        counts['tested_transitions']+=1;dist[shift]+=1
        accepted['NCC_accepted' if row['NCC_accepted'] else 'NCC_rejected']+=1
        if shift>0:positive['total_positive_NCC_shifts']+=1
        if actual is not None:
            counts['actual_count_established']+=1; counts['actual_new_rows']+=actual; known_dist[shift]+=1
            counts['exact_count_matches']+=pred==actual; counts['count_mismatches']+=pred!=actual
            counts['count_matches_within_tolerance']+=pred==actual and row['within_tolerance_px']
            if row['NCC_accepted']:
                accepted['actual_count_established']+=1;accepted['exact_count_matches']+=pred==actual
                accepted['count_mismatches']+=pred!=actual
                if shift>0 or actual>0:
                    accepted['positive_count_established']+=1;accepted['positive_exact_count_matches']+=pred==actual
                    accepted['positive_count_mismatches']+=pred!=actual;accepted['positive_actual_new_rows']+=actual
                    accepted['positive_count_matches_within_tolerance']+=pred==actual and row['within_tolerance_px']
            if shift>0 or actual>0:
                positive['count_established']+=1;positive['actual_new_rows']+=actual
                positive['exact_count_matches']+=pred==actual;positive['count_mismatches']+=pred!=actual
                positive['count_matches_within_tolerance']+=pred==actual and row['within_tolerance_px']
                positive['new_count_'+str(actual)]+=1
            if pred!=actual:
                reason='NCC_AMBIGUOUS' if not row['NCC_accepted'] else ('LOCAL_CHANGE_WITHOUT_SCROLL' if shift==0 else 'UNKNOWN')
                row['failure_reason']=reason;failures.append(row);groups[reason]+=1
        else:
            counts['actual_count_unknown']+=1;unknown_groups[row['unknown_actual_reason']]+=1
    cases=out/'rule_cases'; cases.mkdir(exist_ok=True); sections=[]
    row_map={(r['table'],r['frame_to']):r for r in data}
    for key,pair,os,ns in diagnostics:
        old,new=images[key];cv2.imwrite(str(cases/f'{key}_before.png'),old);cv2.imwrite(str(cases/f'{key}_after.png'),new)
        row=row_map[(pair['table'],pair['after_frame'])]
        detail={'row':row,'local_pixel_reference':pair['reference'],'algorithms':pair['algorithms'],
                'old_spans':os,'new_spans':ns}
        (cases/f'{key}.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2),'utf-8')
        sections.append(f'<h2>{key}</h2><img src="rule_cases/{key}_before.png"><img src="rule_cases/{key}_after.png"><pre>{html.escape(json.dumps(detail,ensure_ascii=False,indent=2))}</pre>')
    fields=list(data[0])
    for r in data:
        for k in r:
            if k not in fields:fields.append(k)
    with (out/'shift_to_new_rows.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
    (out/'rule_review.html').write_text('<meta charset="utf-8"><style>body{background:#222;color:#eee;font-family:monospace}img{width:456px;image-rendering:pixelated}pre{white-space:pre-wrap}</style><h1>SHIFT / fitted pitch audit; actual count UNKNOWN where no independent bottom-row pixel anchor</h1>'+''.join(sections),'utf-8')
    result={'calibration':calibration,'statistics':dict(counts),'positive_transition_statistics':dict(positive),
            'accepted_only_statistics':dict(accepted),
            'shift_values_all_evaluation_pairs':dict(dist),'shift_values_count_covered':dict(known_dist),
            'maximum_count_error_covered':max((abs(r['error']) for r in data if r['error'] is not None),default=None),
            'maximum_quantization_residual_covered_px':max((abs(r['quantization_residual_px']) for r in data if r['actual_new_rows'] is not None),default=None),
            'failure_reasons':dict(groups),'failure_details':failures,'saved_case_pairs':len(images),'OCR_calls':0,
            'unknown_actual_reasons':dict(unknown_groups),
            'tolerance_px':2.,'verdict':'NOT_PROVEN',
            'evidence_contract':'Local ordered row pixel proxy locates previous bottom row; actual suffix count independently measured from bands below it. Shared band segmentation, no complete human physical-line GT. Unknown not labelled using shift division.',
            'source_video_sha256':captures['video_sha256']}
    (out/'shift_rule_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.output)
