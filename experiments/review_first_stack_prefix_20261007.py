"""Native saved source contact sheet, no OCR or production mutations."""
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'runs/full_hh_production_20261006/short_v3/start_stack_readings.jsonl'
out=ROOT/'runs/full_hh_production_20261006/stack_prefix_review'
out.mkdir(exist_ok=True)
groups={(5,'2878365932','seat1'):(1938,1968),(9,'2878365800','seat2'):(1350,1380)}
rows={k:[] for k in groups}
for line in source.open(encoding='utf-8'):
    r=json.loads(line);key=(r['table_id'],r['hand_id'],r['seat'])
    if key in groups and groups[key][0]<=r['source_frame']<=groups[key][1]:rows[key].append(r)
for key,records in rows.items():
    canvas=Image.new('RGB',(600,58*((len(records)+1)//2)),(250,250,250));draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(records):
        p=cv2.imdecode(np.frombuffer(base64.b64decode(r['png_base64']),np.uint8),cv2.IMREAD_GRAYSCALE)
        x=(i%2)*300;y=(i//2)*58
        draw.text((x,y),f"f{r['source_frame']} {r['raw_text']} {r['confidence']}",fill=(0,0,0))
        canvas.paste(Image.fromarray(p).resize((p.shape[1]*2,p.shape[0]*2)),(x,y+16))
    path=out/f'T{key[0]:02d}_{key[1]}_{key[2]}.png';canvas.save(path)
    print(str(path),len(records))
