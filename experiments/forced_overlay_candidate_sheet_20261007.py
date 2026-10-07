"""Review-only native overlay variants; no production authority is assigned."""
import base64,hashlib,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'runs/full_hh_production_20261006/stack_overlay_timing'
measurements=json.loads((out/'measurements.json').read_text('utf-8'))
first={}
for r in measurements:
    if r['table'] not in first or r['header_anchor']<first[r['table']]['header_anchor']:first[r['table']]=r
slots=json.loads((ROOT/'configs/stack_roi_refined.json').read_text('utf-8'))['slots']
samples={}
paths=[ROOT/'runs/full_hh_production_20261006/short_v1/start_stack_capture.jsonl',ROOT/'runs/smoke19_full_metrics_20261004/start_stack_capture.jsonl']
for path in paths:
    for line in path.open(encoding='utf-8'):
        r=json.loads(line);table=r['table_id'];seat=r['seat']
        if r['field']!='stack' or not first[table]['header_anchor']<=r['source_frame']<=first[table]['header_anchor']+10 or (table,seat) in samples:continue
        rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(r['png_base64']),np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['pixel_sha256']
        x,y,X,Y=r['roi'];a,b,A,B=slots[f'slot_{table:02d}']['seats'][seat]['stack']
        patch=cv2.cvtColor(rgb[b-y:B-y,a-x:A-x],cv2.COLOR_RGB2GRAY)[2:-2,6:-6]
        samples[table,seat]=dict(table=table,seat=seat,frame=r['source_frame'],hand_id=first[table]['hand'],source_run=str(path.relative_to(ROOT)),
            source_rgb_sha256=r['pixel_sha256'],source_roi=r['roi'],projected_roi=[a,b,A,B],shape=list(patch.shape),
            pixels_base64=base64.b64encode(patch.tobytes()).decode('ascii'),pixel_sha256=hashlib.sha256(patch.tobytes()).hexdigest(),
            source_rgb_png_base64=r['png_base64'])
sheet=Image.new('RGB',(1440,810),'white');draw=ImageDraw.Draw(sheet)
for (table,seat),r in samples.items():
    x=(int(seat[4:])-1)*180;y=(table-1)*90
    draw.text((x+2,y+2),f'T{table:02} {seat} f{r["frame"]}',fill='black')
    rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(r['source_rgb_png_base64']),np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    sheet.paste(Image.fromarray(rgb).resize((164,36)),(x+2,y+24))
    patch=np.frombuffer(base64.b64decode(r['pixels_base64']),np.uint8).reshape(r['shape'])
    sheet.paste(Image.fromarray(patch).resize((140,20)).convert('RGB'),(x+2,y+63))
sheet.save(out/'forced_overlay_variants_review.png')
(out/'forced_overlay_variants_candidates.json').write_text(json.dumps(list(samples.values()),indent=2),'utf-8')
print('review_only_samples',len(samples))
