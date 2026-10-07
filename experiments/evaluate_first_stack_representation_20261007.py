"""Bounded audit only: preserve raw source, evaluate fixed image representations."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parser_core.start_stack_visibility import VisibilitySource
from parser_core.tiny_chat_client import TinyChatClient


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    # Native sources already reviewed. Expectations are audit labels, never runtime input.
    groups=[(5,'seat5',284,286,'28260'),(9,'seat3',1377,1379,'132026'),
            (9,'seat5',1377,1379,'197744'),(5,'seat2',1965,1967,'35627'),
            (5,'seat1',284,286,'17416'),(5,'seat1',258,260,None),
            (5,'seat1',280,283,None)]
    sources=[]
    for line in (args.source/'start_stack_readings.jsonl').open(encoding='utf-8'):
        r=json.loads(line)
        group=next((g for g in groups if r.get('table_id')==g[0] and r.get('seat')==g[1] and g[2]<=r.get('source_frame',-1)<=g[3]),None)
        if group is not None:sources.append((r,group[-1]))
    if len(sources)>32:raise ValueError('BOUNDED_SOURCE_LIMIT')
    verifier=VisibilitySource();jobs=[];records=[]
    for row,expected in sources:
        projected=verifier.verify(row)
        if projected is None:raise ValueError('SOURCE_REPRODUCTION_FAILED')
        gray=cv2.imdecode(np.frombuffer(base64.b64decode(row['png_base64']),np.uint8),cv2.IMREAD_GRAYSCALE)
        variants={'direct_native':gray,'direct_nearest2':cv2.resize(gray,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST),
            'direct_linear2':cv2.resize(gray,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR),
            'refined_native':projected['gray'],
            'refined_nearest2':cv2.resize(projected['gray'],None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)}
        for variant,pixels in variants.items():
            item_id=str(len(jobs));ok,png=cv2.imencode('.png',pixels);assert ok
            sha=hashlib.sha256(pixels.tobytes()).hexdigest()
            jobs.append(dict(item_id=item_id,png_base64=base64.b64encode(png).decode(),source_sha256=sha))
            records.append(dict(item_id=item_id,variant=variant,expected_numeric=expected,
                source={k:row[k] for k in ('table_id','session_id','hand_id','seat','source_frame','source_pts','time_base','roi','pixel_sha256','raw_text','confidence')},
                derived_pixel_sha256=sha,derived_shape=list(pixels.shape)))
    config=json.loads((ROOT/'configs/ocr_tiny_local.json').read_text('utf-8'))
    client=TinyChatClient(Path(config['python']),Path(config['model_dir']),config['model_name'],
        config['weights_sha256'],config['device'],args.output/'ocr.stderr.log')
    started=time.perf_counter()
    try:
        for start in range(0,len(jobs),8):
            results=client.read_batch(jobs[start:start+8],8)['results']
            for r in results:records[int(r['item_id'])]['response']=r
    finally:client.close()
    (args.output/'source_records.json').write_text(json.dumps([r for r,_ in sources]),'utf-8')
    (args.output/'results.json').write_text(json.dumps(records,indent=2),'utf-8')
    summary={}
    for variant in variants:
        rows=[r for r in records if r['variant']==variant]
        def number(row):
            value=row['response'];text=(value.get('raw_text') or '').strip()
            return text.replace(' ','').replace('\u00a0','') if value.get('status')=='OK' and value.get('score',0)>=.95 and re.fullmatch(r'\d+(?:[ \u00a0]\d{3})*',text) else None
        summary[variant]=dict(total=len(rows),numeric_labels=sum(r['expected_numeric'] is not None for r in rows),
            exact_confident=sum(number(r)==r['expected_numeric'] for r in rows if r['expected_numeric'] is not None),
            wrong_confident=sum(number(r) is not None and number(r)!=r['expected_numeric'] for r in rows),
            unresolved=sum(number(r) is None for r in rows))
    result=dict(status='OFFLINE_REPRESENTATION_AUDIT_ONLY',sources=len(sources),inferences=len(records),
        seconds=time.perf_counter()-started,variants=summary)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
