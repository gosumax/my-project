"""Read-only source sheets for the four missing short_v3 occupancy contexts."""
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'runs/full_hh_production_20261006/short_v3/seat_forced_inputs.jsonl'
output=ROOT/'runs/full_hh_production_20261006/occupancy_source_review'
output.mkdir(exist_ok=True)
contexts={(6,'2878365610'),(6,'2878365755'),(7,'2878365529'),(8,'2878365523')}
rows=[]
for line in source.open(encoding='utf-8'):
    r=json.loads(line)
    if r.get('kind')=='NICK' and (r['table_id'],r['hand_id']) in contexts:rows.append(r)
receipts=[]
for table,hand in sorted(contexts):
    canvas=Image.new('RGB',(1100,8*150),(242,242,242))
    draw=ImageDraw.Draw(canvas)
    for i in range(8):
        seat=f'seat{i+1}'
        candidates=sorted((r for r in rows if r['table_id']==table and r['hand_id']==hand and r['seat']==seat),
                          key=lambda r:r['source_frame'])
        if not candidates:continue
        first=candidates[0]
        best=max(candidates,key=lambda r:(r['ocr_response'].get('score') or 0,-r['source_frame']))
        for col,r in enumerate([first,best]):
            raw=r['ocr_response']
            x,y=col*550,i*150
            draw.text((x+5,y+5),f'T{table:02d} H{hand} {seat} frame {r["source_frame"]}',fill='black')
            draw.text((x+5,y+25),f'{"first" if col==0 else "max-confidence"}: {raw.get("raw_text")} / {raw.get("score",0):.4f}',fill='black')
            crop=Image.open(io.BytesIO(base64.b64decode(r['png_base64']))).convert('RGB')
            canvas.paste(crop.resize((crop.width*4,crop.height*4),Image.Resampling.NEAREST),(x+5,y+52))
            receipts.append(r)
    canvas.save(output/f'T{table:02d}_{hand}.png')
(output/'source_receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),'utf-8')
print('Wrote four diagnostic sheets outside production output;',len(rows),'source NICK records retained in original log')
