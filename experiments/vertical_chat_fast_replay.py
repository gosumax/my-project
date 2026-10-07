"""Parity and overlap-support fix on the exact frozen experiment states."""
import argparse
import base64
from collections import Counter
import json
from pathlib import Path
import time
import zlib
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.vertical_chat_fast_ncc import correlation_fft


def main(out):
    original={}
    for s in (out/'transitions.jsonl').open(encoding='utf-8'):
        r=json.loads(s); original[(r['table'],r['after_frame'])]=r['algorithms']
    counts=Counter(); timings=Counter(); max_score_delta=0.; rows=[]; previous={}
    for line in (out/'states.jsonl').open(encoding='utf-8'):
        r=json.loads(line); t=r['table']; new=np.frombuffer(zlib.decompress(base64.b64decode(r['zlib_gray'])),np.uint8).reshape(r['shape']).copy()
        if t in previous:
            old=previous[t]; algorithms={}
            for name,fixed,min_overlap in [('NCC_RAW',False,48),('NCC_MASK',True,48),('NCC_SUPPORT',True,120)]:
                tick=time.perf_counter(); result=correlation_fft(old,new,fixed,min_overlap); timings[name]+=time.perf_counter()-tick
                algorithms[name]=result
                if name!='NCC_SUPPORT':
                    prior=original[(t,r['frame'])][name]['algorithm']; counts[name+':pairs']+=1
                    counts[name+':shift_differences']+=result['shift']!=prior['shift']
                    counts[name+':acceptance_differences']+=result['accepted']!=prior['accepted']
                    counts[name+':accepted_shift_differences']+=result['shift']!=prior['shift'] and (result['accepted'] or prior['accepted'])
                    max_score_delta=max(max_score_delta,abs(result['score']-prior['score']))
            rows.append({'table':t,'after_frame':r['frame'],'algorithms':algorithms})
        previous[t]=new
    with (out/'fast_correlations.jsonl').open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    result={'comparison_counts':dict(counts),'max_best_score_absolute_delta':max_score_delta,'timing_seconds':dict(timings),
            'parity':'PASS' if all(counts[k]==0 for k in counts if 'acceptance_differences' in k or 'accepted_shift_differences' in k) else 'FAIL',
            'support_fix':'minimum overlap 120 native y pixels (was 48), mask unchanged; explicitly abstain on unsupported large scroll'}
    (out/'fast_ncc_parity.json').write_text(json.dumps(result,indent=2),'utf-8'); print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True); a=p.parse_args();main(a.output)
