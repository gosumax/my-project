"""Build bounded representative native-pixel controls and cause inventories."""
import argparse
import base64
from collections import Counter
import html
import json
from pathlib import Path
import zlib
import cv2
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from parser_core.chat_rows import physical_spans, exclude_first_visible_span


def build(out,tolerant=False):
    prefix='row_reference_' if tolerant else ''
    selected=[]; counts=Counter(); actual=Counter(); pair_lookup={}
    for line in (out/f'{prefix}transitions.jsonl').open(encoding='utf-8'):
        r=json.loads(line)
        if not r['evaluation']: continue
        pair_lookup[(r['table'],r['after_frame'])]=r
        ref=r['reference']; alg=r['algorithms']['NCC_MASK']['algorithm']; kinds=[]
        if ref:
            if ref['count_known']:
                n=len(ref['new_indices'])
                if n==1:kinds.append('ONE_NEW_VISIBLE_ROW')
                if 2<=n<=3:kinds.append('TWO_OR_THREE_NEW_VISIBLE_ROWS')
                if n>=4:kinds.append('FAST_SCROLL_VISIBLE_SUFFIX')
                if ref['shift']==0 and n>0:kinds.append('APPEND_WITHOUT_SCROLL')
            if ref['shift']>0:kinds.append('TOP_ROWS_EXIT_VIEWPORT')
            if not ref['count_known'] and ref['shift']==0:kinds.append('INSIDE_ROW_PIXEL_CHANGE_CANDIDATE')
            if abs(ref['shift']/r['row_step_px']-round(ref['shift']/r['row_step_px']))>.1:kinds.append('NONINTEGER_SHIFT_RATIO')
        else:kinds.append('UNKNOWN_REFERENCE')
        if alg['margin']<.02:kinds.append('NCC_AMBIGUOUS')
        if not alg['accepted']:kinds.append('NCC_ABSTAIN')
        for k in kinds:
            actual[k]+=1
            if counts[k]<3: selected.append((k,r)); counts[k]+=1
    wanted={(r['table'],r[f'{s}_frame']) for _,r in selected for s in ['before','after']}
    states={}; previous={}
    for line in (out/'states.jsonl').open(encoding='utf-8'):
        r=json.loads(line); key=(r['table'],r['frame'])
        gray=np.frombuffer(zlib.decompress(base64.b64decode(r['zlib_gray'])),np.uint8).reshape(r['shape']).copy()
        spans=exclude_first_visible_span(physical_spans(gray))[0]
        if key in wanted: states[key]=gray
        pair=pair_lookup.get(key)
        if pair and pair['reference'] and pair['reference']['count_known'] and r['table'] in previous:
            old_frame,old,os=previous[r['table']]
            repeated=[]
            for j in pair['reference']['new_indices']:
                a,b=spans[j]
                if any(np.array_equal(gray[a:b,:-12],old[c:d,:-12]) for c,d in os): repeated.append(j)
            if repeated:
                kind='NEW_SUFFIX_REPEATS_OLD_NATIVE_ROW_PIXELS'; actual[kind]+=1
                if counts[kind]<3:
                    pair={**pair,'repeated_pixel_new_indices':repeated}; selected.append((kind,pair)); counts[kind]+=1
                    states[(r['table'],old_frame)]=old; states[key]=gray
        previous[r['table']]=(r['frame'],gray,spans)
    gallery=out/'controls'; gallery.mkdir(exist_ok=True); items=[]
    for kind,r in selected:
        key=f"{kind}_T{r['table']}_{r['before_frame']}_{r['after_frame']}"
        for side in ['before','after']:
            cv2.imwrite(str(gallery/f'{key}_{side}.png'),states[(r['table'],r[f'{side}_frame'])])
        (gallery/f'{key}.json').write_text(json.dumps(r,indent=2),'utf-8')
        items.append(f'<h2>{key}</h2><img src="controls/{key}_before.png"><img src="controls/{key}_after.png"><pre>{html.escape(json.dumps(r,indent=2))}</pre>')
    (out/f'{prefix}controls.html').write_text('<meta charset="utf-8"><style>body{background:#222;color:#eee;font-family:monospace}img{width:456px;image-rendering:pixelated}pre{white-space:pre-wrap}</style><h1>Real Smoke17 controls; CANDIDATE labels require visual interpretation</h1>'+''.join(items),'utf-8')
    result={'observed_pair_categories':dict(actual),'saved_controls':dict(counts),'not_independently_established':['CROP_JITTER (configured ROI boxes fixed)','ANIMATION vs source raster/compression changes','ANTI_ALIASING cause vs generic TEXT_PIXEL_CHANGE','all repeated-text births (no OCR identity permitted)']}
    (out/f'{prefix}controls_inventory.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--tolerant',action='store_true'); a=p.parse_args(); build(a.output,a.tolerant)
