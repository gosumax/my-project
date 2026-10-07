"""Freeze manually reviewed full-button sources for a diagnostic candidate."""
import base64
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/full_hh_production_20261006/occupancy_source_review'
slots=json.loads((ROOT/'configs/layout_847x404.json').read_text('utf-8'))['slots']
reviewed={(6,466):['seat5'],(7,151):['seat5','seat6'],(8,131):['seat3','seat5']}
templates=[];sources=[]
sheet=Image.new('RGB',(760,len(reviewed)*8*105),(240,240,240));draw=ImageDraw.Draw(sheet)
index=0
for (table,frame),empty in reviewed.items():
    path=OUT/f'T{table:02d}_frame{frame}_table.png'
    rgb=np.asarray(Image.open(path).convert('RGB'))
    for seat,b in slots[f'slot_{table:02d}']['seats'].items():
        x=b['nick'][0];Y=b['stack'][3]
        roi=[max(0,x-40),max(0,Y-42),min(rgb.shape[1],x+88),min(rgb.shape[0],Y+6)]
        x,y,X,Y=roi;gray=cv2.cvtColor(rgb[y:Y,x:X],cv2.COLOR_RGB2GRAY)
        row=dict(table=table,frame=frame,seat=seat,roi=roi,shape=list(gray.shape),
            label='EMPTY' if seat in empty else 'NOT_EMPTY',
            pixel_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),
            pixels_base64=base64.b64encode(gray.tobytes()).decode('ascii'),
            source_table_png_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        sources.append(row)
        if seat in empty:
            templates.append(dict(row,template_id=f'T{table:02d}_F{frame}_{seat}',
                label_authority='ROOT_VISUAL_FULL_TABLE_AND_BUTTON_REVIEW_20261007'))
        draw.text((5,index*105+4),f'T{table:02d} {seat} F{frame}: {row["label"]} / roi{roi}',fill='black')
        image=Image.fromarray(gray).convert('RGB');image=image.resize((gray.shape[1]*2,gray.shape[0]*2),Image.Resampling.NEAREST)
        sheet.paste(image,(440,index*105));index+=1
payload=dict(schema='native-full-empty-seat-button-v1',full_ncc_min=.98,text_ncc_min=.98,
    text_box=[20,12,108,36],translation_radius=2,
    geometry_rule='[nick.x0-40,stack.y1-42,nick.x0+88,stack.y1+6] clipped to table',
    authority='CURRENT_EMPTY_BUTTON_ONLY; NO_ACTOR_EPOCH_OR_START_STACK_AUTHORITY',templates=templates)
(OUT/'empty_button_model_candidate.json').write_text(json.dumps(payload,indent=2),'utf-8')
(OUT/'empty_button_source_fixture.json').write_text(json.dumps(sources,indent=2),'utf-8')
sheet.save(OUT/'empty_button_candidate_sheet.png')
print('Templates',len(templates),'native labeled examples',len(sources))
