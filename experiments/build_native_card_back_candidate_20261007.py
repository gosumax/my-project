"""Audit-only source freeze for visibly complete card backs missing from the old model."""
import base64
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'runs/full_hh_production_20261006/occupancy_source_review/joint_native'
sources=json.loads((folder/'source_receipts.json').read_text('utf-8'))
slots=json.loads((ROOT/'configs/layout_847x404.json').read_text('utf-8'))['slots']
rows=[]
for table,seat in ((7,'seat2'),(8,'seat7')):
    source=next(r for r in sources['records'] if r['table']==table)
    path=ROOT/source['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==source['png_sha256']
    bgr=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    assert hashlib.sha256(rgb.tobytes()).hexdigest()==source['table_rgb_sha256']
    roi=slots[f'slot_{table:02d}']['seats'][seat]['shown_cards'];x,y,X,Y=roi
    crop=np.ascontiguousarray(bgr[y:Y,x:X])
    rows.append(dict(template_id=f'native_T{table:02d}_{seat}',label='TWO_VISIBLE_CARD_BACKS',
        table_id=table,seat=seat,roi=roi,source=source,video_sha256=sources['video_sha256'],
        shape=list(crop.shape),pixel_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
        pixels_base64=base64.b64encode(crop).decode()))
    ok,png=cv2.imencode('.png',crop);assert ok
    (folder/f'T{table:02d}_{seat}_card_roi.png').write_bytes(png.tobytes())
model=dict(schema='source-bound-table-card-back-supplement-v1',threshold=.98,
    authority='POSITIVE_FIXED_ROI_CARDS_ONLY_WHEN_FROZEN_TEMPLATE_ABSENT',templates=rows)
target=folder/'native_card_back_candidate.json'
target.write_text(json.dumps(model,indent=2)+'\n','utf-8')
print(json.dumps(dict(path=str(target),templates=[{k:r[k] for k in ('template_id','shape','roi','pixel_sha256')} for r in rows])))
