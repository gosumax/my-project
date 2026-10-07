"""Native context for geometry review; no classifiers and no production writes."""
import hashlib
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/full_hh_production_20261006/occupancy_source_review'
layout={r['table']:r for r in json.loads((ROOT/'configs/nine_table_layout.json').read_text('utf-8'))['tables']}
slots=json.loads((ROOT/'configs/layout_847x404.json').read_text('utf-8'))['slots']
todo={131:[8],151:[7],466:[6]}
receipts=[]
with av.open(str(ROOT/'video/smoke_19.mp4')) as container:
    for index,frame in enumerate(container.decode(video=0)):
        if index>max(todo):break
        if index not in todo:continue
        rgb=frame.to_ndarray(format='rgb24')
        for table in todo[index]:
            geometry=layout[table]
            image=rgb[geometry['y']:geometry['y']+geometry['height'],geometry['x']:geometry['x']+geometry['width']]
            native=Image.fromarray(image)
            native.save(OUT/f'T{table:02d}_frame{index}_table.png')
            sheet=Image.new('RGB',(800,8*150),(242,242,242));draw=ImageDraw.Draw(sheet)
            for i,(seat,boxes) in enumerate(slots[f'slot_{table:02d}']['seats'].items()):
                x,y,X,Y=boxes['nick'];a,b,A,B=boxes['stack']
                roi=[max(0,x-18),max(0,min(y,b)-12),min(image.shape[1],max(X,A)+18),min(image.shape[0],max(Y,B)+15)]
                x,y,X,Y=roi;crop=native.crop(roi)
                draw.text((5,i*150+5),f'T{table:02d} {seat} frame{index} roi{roi}',fill='black')
                sheet.paste(crop.resize((crop.width*3,crop.height*3),Image.Resampling.NEAREST),(320,i*150))
                receipts.append(dict(table=table,seat=seat,frame=index,pts=frame.pts,
                    time_base=[frame.time_base.numerator,frame.time_base.denominator],roi=roi,
                    rgb_sha256=hashlib.sha256(image[y:Y,x:X].tobytes()).hexdigest()))
            sheet.save(OUT/f'T{table:02d}_frame{index}_wide_regions.png')
(OUT/'wide_region_receipts.json').write_text(json.dumps(receipts,indent=2),'utf-8')
print('Native context frames',len(todo),'wide regions',len(receipts))
