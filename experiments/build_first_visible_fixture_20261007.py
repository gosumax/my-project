"""Extract immutable real first-stack sources; does not rerun or repair OCR."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parser_core.start_stack_visibility import VisibilitySource
from parser_core.canonical_events import reconcile_stack_facts

run=ROOT/'runs/full_hh_production_20261006/short_v1'
names=['start_stack_live_receipts.jsonl','start_stack_capture.jsonl','start_stack_facts.jsonl',
       'canonical_forced_events.jsonl','seat_forced_summary.json']
seats={'seat1','seat2','seat4','seat5'}
hand='2878365559'
readings={}
for line in (run/names[0]).open(encoding='utf-8'):
    r=json.loads(line)
    if r.get('hand_id')==hand and r.get('stage')=='STACK_OCR' and r.get('seat') in seats and 258<=r['source_frame']<=286:
        readings[r['seat'],r['source_frame']]=r
source=VisibilitySource()
templates=[]
for line in (run/names[1]).open(encoding='utf-8'):
    r=json.loads(line)
    key=(r['seat'],r['source_frame'])
    if r['table_id']==5 and r['field']=='stack' and r['source_frame']==258 and r['seat'] in {'seat1','seat2','seat4'}:
        rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(r['png_base64']),np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['pixel_sha256']
        x,y,X,Y=r['roi'];a,b,A,B=source.rois['slot_05']['seats'][r['seat']]['stack']
        gray=cv2.cvtColor(rgb[b-y:B-y,a-x:A-x],cv2.COLOR_RGB2GRAY)[2:-2,6:-6]
        templates.append(dict(label={'seat1':'SB','seat2':'BB','seat4':'ANTE'}[r['seat']],
            source_table=5,source_frame=258,source_seat=r['seat'],source_rgb_sha256=r['pixel_sha256'],
            source_roi=r['roi'],projected_roi=[a,b,A,B],shape=list(gray.shape),
            pixels_base64=base64.b64encode(gray.tobytes()).decode('ascii'),
            pixel_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),
            label_authority='ROOT_VISUAL_REVIEW_T05_FORCED_OVERLAY_SHEET_20261007'))
    if r['table_id']!=5 or r['field']!='stack' or key not in readings:continue
    rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(r['png_base64']),np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['pixel_sha256']
    table=np.zeros((404,847,3),np.uint8)
    x,y,X,Y=r['roi'];table[y:Y,x:X]=rgb
    readings[key]['visibility_source']=source.capture(table,5,r['seat'],r['roi'])
    assert source.verify(readings[key]) is not None
facts=[json.loads(l) for l in (run/names[2]).read_text('utf-8').splitlines()]
events=[json.loads(l) for l in (run/names[3]).read_text('utf-8').splitlines()]
snapshots=json.loads((run/names[4]).read_text('utf-8'))['forced_snapshots']
selected=[]
for seat in sorted(seats):
    fact=copy.deepcopy(next(r for r in facts if r['hand_id']==hand and r['seat']==seat))
    if fact.get('frames'):
        fact['evidence']=[readings[seat,f] for f in fact['frames']]
    selected.append(dict(seat=seat,readings=[readings[seat,f] for f in range(258,287)],
                         fact=reconcile_stack_facts(fact,events,forced_snapshots=snapshots)))
payload=dict(source_run=str(run.relative_to(ROOT)),source_hashes={n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in names},
             visual_review='T05: forced overlays 238..282, blank283, initial numbers284..286; original source retained',
             contexts=selected,canonical_events=[r for r in events if r['hand_id']==hand])
target=ROOT/'tests/fixtures/first_visible_stack_smoke19.json'
target.write_text(json.dumps(payload,ensure_ascii=False,indent=2),'utf-8')
print(target, target.stat().st_size)
model=ROOT/'models/forced_stack_overlays.json'
if not model.exists():
    model.write_text(json.dumps(dict(schema='forced-stack-overlay-native-patterns-v1',
        source_capture_sha256=payload['source_hashes']['start_stack_capture.jsonl'],
        purpose='LABEL_VISIBILITY_ONLY; NOT_STACK_NUMBERS_OR_FORCED_AMOUNTS',templates=templates),indent=2),'utf-8')
print(model,model.stat().st_size)
