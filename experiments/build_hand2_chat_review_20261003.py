"""Read saved evidence only; no pipeline imports or video decoding."""
import base64, collections, hashlib, html, io, json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'runs/first_stack_clean_window_20261003/acceptance_ON_math'
OUT = ROOT / 'runs/deal_complete_fix_chat_review_20261003/chat_review'
OUT.mkdir(parents=True, exist_ok=True)
HAND = '2876719777'
records = []

def read(name):
    with (SRC/name).open(encoding='utf-8') as f:
        for line, s in enumerate(f, 1):
            yield line, json.loads(s)

ocr = {x['row_key']: x for _, x in read('raw_ocr_results.jsonl')}
def add(name, line, x, section, image, text):
    frame = x.get('frame_id', x.get('source_frame'))
    raw = base64.b64decode(x.get('row_pixels_base64','')) if 'row_pixels_base64' in x else None
    expected = x.get('row_pixel_sha256',x.get('pixel_sha256'))
    actual = hashlib.sha256(raw if raw is not None else image.tobytes()).hexdigest()
    if actual != expected:
        raise ValueError((name,line,'pixel hash mismatch',actual,expected))
    b = io.BytesIO(); image.save(b, format='PNG')
    records.append({'section': section, 'frame': frame, 'source_file': str(SRC/name),
                    'source_line': line, 'record': {k:v for k,v in x.items() if 'base64' not in k},
                    'ocr_text': text, 'hash_verified':True, 'image':image,
                    'data_uri':'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()})

for line,x in read('seat_forced_inputs.jsonl'):
    if x.get('table_id')==3 and x.get('hand_id')==HAND and x.get('kind')=='CHAT':
        add('seat_forced_inputs.jsonl',line,x,'snapshot2530',Image.open(io.BytesIO(base64.b64decode(x['png_base64']))),x['ocr_response']['raw_text'])
for line,x in read('physical_rows.jsonl'):
    if x.get('table_id')==3 and 2450<=x['frame_id']<3837:
        shape=x['pixel_shape']; im=Image.frombytes('L',(shape[1],shape[0]),base64.b64decode(x['row_pixels_base64']))
        add('physical_rows.jsonl',line,x,'physical',im,ocr.get(x['row_key'],{}).get('raw_ocr_response',{}).get('raw_text','OCR unavailable'))
headers={}
for line,x in read('hand_header_ocr.jsonl'):
    if x.get('table_id')==3 and 2516<=x.get('source_frame',0)<=3837:
        key=(x['source_frame'],x['source_span_index'])
        if 'png_base64' in x: headers[key]=(line,x)
        elif x.get('ocr_response') and key in headers:
            ln,a=headers[key]; a['ocr_response']=x['ocr_response'];a['confirmation_source_line']=line
for line,x in headers.values():
    add('hand_header_ocr.jsonl',line,x,'headers',Image.open(io.BytesIO(base64.b64decode(x['png_base64']))),x.get('ocr_response',{}).get('raw_text','UNKNOWN'))
for line,x in read('start_stack_capture.jsonl'):
    if x.get('table_id')==3 and 2538<=x.get('source_frame',0)<=2542 and x.get('field')=='stack':
        add('start_stack_capture.jsonl',line,x,'provisional_stack',Image.open(io.BytesIO(base64.b64decode(x['png_base64']))),x['seat']+' · PROVISIONAL only; no numeric OCR added')

snapshot=[r for r in records if r['section']=='snapshot2530']
physical=[r for r in records if r['section']=='physical']
def sheet(rs, path):
    width=1000; height=sum(max(44,r['image'].height*2+8) for r in rs)+40
    im=Image.new('RGB',(width,height),'#eeeeee');draw=ImageDraw.Draw(im);y=20
    for r in rs:
        crop=r['image'].resize((r['image'].width*2,r['image'].height*2),Image.Resampling.NEAREST)
        im.paste(crop,(10,y));draw.text((480,y),f"frame {r['frame']} | {r['ocr_text']}",fill='black')
        y+=max(44,crop.height+8)
    im.save(OUT/path)
sheet(snapshot,'snapshot_2530_raw_rows.png')
sheet([r for r in physical if r['frame']>=2516],'hand_available_physical_rows.png')
sheet([r for r in records if r['section']=='provisional_stack' and r['frame'] in (2540,2541)],'provisional_stack_2540_2541.png')
clean=[{k:v for k,v in r.items() if k not in ('image','data_uri')} for r in records]
(OUT/'source_inventory.json').write_text(json.dumps(clean,ensure_ascii=False,indent=2),encoding='utf-8')
counts=collections.Counter(r['section'] for r in records)
summary={'table':3,'hand_id':HAND,'confirmed_header_source_frame':2516,'next_header_frame':3837,
         'hand_source_interval':[2516,3836],'deal_complete_frame':2530,'blocker_start':2540,
         'missing_live_numeric_ocr_frames':[2540,2541],'provisional_saved_stack_pixels_present':[2540,2541], 'counts':dict(counts),'longchat_or_stitched_in_run':False,
         'last_physical_before_blocker':[r for r in clean if r['section']=='physical' and r['frame']<2540],
         'caveat':'Temporal interval and visible snapshot are not certified hand row assignment. Snapshot rows0-1 predate header. Next-header crop3837 is context only.'}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
parts=['<!doctype html><meta charset="utf-8"><title>Table3 hand2876719777 source CHAT</title><style>body{font:16px Arial;background:#eee;margin:24px}article{background:white;padding:12px;margin:8px 0}img{image-rendering:pixelated;min-width:452px}pre{white-space:pre-wrap;overflow-wrap:anywhere}small{overflow-wrap:anywhere}nav{position:sticky;top:0;background:#eee;padding:12px}</style>',
       '<h1>Table 3 · Hand #2876719777 · Saved CHAT evidence</h1>',
       '<p>Hand header frame2516; DEAL_COMPLETE2530; next different hand header3837. The live numeric stack lane lacks frames2540–2541, but all8 seat stack PNGs at each frame survive in the separate PROVISIONAL capture lane. The prior claim that no pixels/hashes exist was incorrect. CHAT header crop2540 also survives. This page uses saved pixels only; no video reread, new OCR or inferred replacement.</p>',
       '<p>Images are raw physical rows or independently saved CHAT snapshot rows. Contact sheets concatenate evidence for review; they are not an original LongChat or reconstructed transcript. Source observation time does not prove action onset. Snapshot first two rows precede the hand header.</p>',
       '<nav><a href="#snapshot2530">Visible CHAT at2530</a> · <a href="#physical">Physical rows through3836</a> · <a href="#headers">Header at2540 and continuity</a> · <a href="#provisional_stack">Actual stack crops2538–2542</a> · <a href="source_inventory.json">Provenance JSON</a></nav>']
for section,label in [('snapshot2530','All18 visible CHAT rows at DEAL_COMPLETE2530'),('physical','Physical rows2459–3836 (2459 is previous-hand context)'),('headers','Saved header crops, including2540 and next hand3837'),('provisional_stack','Saved PROVISIONAL stack ROI pixels2538–2542; no OCR interpretation')]:
    parts.append(f'<h2 id="{section}">{label}</h2>')
    for r in records:
        if r['section']!=section:continue
        parts.append(f'<article><strong>frame {r["frame"]} · {html.escape(r["ocr_text"])}</strong><br><img src="{r["data_uri"]}"><details><summary>Exact provenance / verified pixel hash</summary><pre>{html.escape(json.dumps({k:v for k,v in r.items() if k not in ("image","data_uri")},ensure_ascii=False,indent=2))}</pre></details></article>')
(OUT/'index.html').write_text('\n'.join(parts),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,default=str))
