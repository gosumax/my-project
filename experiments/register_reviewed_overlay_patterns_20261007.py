"""Register visually reviewed native word variants, retaining exact provenance."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
audit=ROOT/'runs/full_hh_production_20261006/stack_overlay_timing'
# Root reviewed the complete 72-cell sheet. Empty, all-in and two obstructed
# panels are excluded; these labels have no player or amount authority.
labels={
 1:['BB','ANTE','ANTE','ANTE','ANTE','ANTE','ANTE','SB'],
 2:['SB','BB','ANTE','ANTE','ANTE','ANTE',None,'ANTE'],
 3:['ANTE',None,'BB',None,'ANTE','ANTE','ANTE','ANTE'],
 4:[None,'ANTE','ANTE','ANTE','ANTE','SB',None,'BB'],
 5:['SB','BB','ANTE','ANTE','ANTE','ANTE','ANTE','ANTE'],
 6:['ANTE',None,'ANTE','ANTE',None,'ANTE','SB',None],
 7:['ANTE','ANTE','SB','BB',None,None,'ANTE','ANTE'],
 8:['ANTE','SB',None,'BB',None,'ANTE',None,'ANTE'],
 9:['ANTE','ANTE','ANTE','SB','BB','ANTE','ANTE','ANTE']}
path=ROOT/'models/forced_stack_overlays.json'
model=json.loads(path.read_text('utf-8'))
model['templates']=[r for r in model['templates'] if r.get('label_authority')!='ROOT_72_CELL_OVERLAY_REVIEW_20261007']
for r in json.loads((audit/'forced_overlay_variants_candidates.json').read_text('utf-8')):
    label=labels[r['table']][int(r['seat'][4:])-1]
    if label is None:continue
    model['templates'].append(dict(label=label,source_table=r['table'],source_seat=r['seat'],source_frame=r['frame'],
        source_hand_id=r['hand_id'],source_run=r['source_run'],source_rgb_sha256=r['source_rgb_sha256'],
        source_roi=r['source_roi'],projected_roi=r['projected_roi'],shape=r['shape'],pixels_base64=r['pixels_base64'],
        pixel_sha256=r['pixel_sha256'],label_authority='ROOT_72_CELL_OVERLAY_REVIEW_20261007'))
model['review_sheet_sha256']=hashlib.sha256((audit/'forced_overlay_variants_review.png').read_bytes()).hexdigest()
path.write_text(json.dumps(model,indent=2),'utf-8')
print('reviewed_native_templates',len(model['templates']))
