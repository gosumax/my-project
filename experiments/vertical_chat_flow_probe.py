"""Secondary bounded Optical Flow probe, only after documented NCC failure."""
import argparse
import base64
from collections import Counter
import json
from pathlib import Path
import time
import zlib
import cv2
import numpy as np


def main(out, limit):
    pairs=[json.loads(s) for s in (out/'row_reference_transitions.jsonl').open(encoding='utf-8')]
    # Failure-conditioned probe; no claim of general out-of-sample performance.
    cases=[]; counts=Counter()
    for r in pairs:
        if not r['evaluation']: continue
        ref=r['reference']; alg=r['algorithms']['NCC_MASK']['algorithm']
        group='UNKNOWN_REFERENCE' if ref is None else ('NCC_WRONG_OR_ABSTAIN' if abs(alg['shift']-ref['shift'])>1.5 or not alg['accepted'] else ('NCC_POSITIVE_CONTROL' if ref['shift']>0 else 'NCC_ZERO_CONTROL'))
        if counts[group]>=limit: continue
        counts[group]+=1; cases.append((group,r))
    wanted={(r['table'],r[f'{side}_frame']) for _,r in cases for side in ['before','after']}
    states={}
    for line in (out/'states.jsonl').open(encoding='utf-8'):
        r=json.loads(line); key=(r['table'],r['frame'])
        if key in wanted:
            states[key]=np.frombuffer(zlib.decompress(base64.b64decode(r['zlib_gray'])),np.uint8).reshape(r['shape']).copy()
    rows=[]; metrics=Counter(); started=time.perf_counter()
    for group,r in cases:
        old=states[(r['table'],r['before_frame'])][:,:-12]
        new=states[(r['table'],r['after_frame'])][:,:-12]
        tick=time.perf_counter()
        # Current -> old: positive y is old content's upward displacement.
        flow=cv2.calcOpticalFlowFarneback(new,old,None,.5,5,31,5,7,1.5,0)
        mask=new>70; mask[int(new.shape[0]*.7):]=False
        vals=flow[:,:,1][mask]
        shift=float(np.median(vals)) if len(vals) else None
        mad=float(np.median(np.abs(vals-shift))) if len(vals) else None
        dx=float(np.median(flow[:,:,0][mask])) if len(vals) else None
        elapsed=time.perf_counter()-tick
        ref=r['reference']; expected=ref['shift'] if ref else None
        metrics[group+':pairs']+=1
        if expected is not None:
            metrics[group+':within1px']+=abs(shift-expected)<=1
            metrics[group+':within2px']+=abs(shift-expected)<=2
        rows.append({'group':group,'table':r['table'],'before_frame':r['before_frame'],'after_frame':r['after_frame'],
                     'expected_shift_px':expected,'flow_shift_px':shift,'flow_mad':mad,'horizontal_median':dx,
                     'ncc':r['algorithms']['NCC_MASK']['algorithm'],'seconds':elapsed})
    with (out/'optical_flow_probe.jsonl').open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row)+'\n')
    result={'scope':'FAILURE_CONDITIONED_DIAGNOSTIC_NOT_FULL_TRACKER','max_pairs_per_group':limit,
            'counts':dict(metrics),'flow_total_seconds':sum(r['seconds'] for r in rows),'wall_seconds':time.perf_counter()-started,
            'UNKNOWN_reference_not_scored':True,'config':{'method':'Farneback','levels':5,'winsize':31,'iterations':5,'upper_foreground_fraction':.7}}
    (out/'optical_flow_summary.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--limit',type=int,default=60)
    a=p.parse_args(); main(a.output,a.limit)
