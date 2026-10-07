"""Bounded offline Tiny input feasibility; no live START_STACK proof."""
import base64
import collections
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import av
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parser_core.chat_rows import physical_spans,padded_row_span
from parser_core.layout import load_fixed_layout
from parser_core.ram_first_stage import png_compatible_gray
from parser_core.tiny_chat_client import TinyChatClient

OUT=ROOT/'runs/seat_forced_binding_20261003/input_probe'
THRESHOLD=.95
HANDS={884:'2876719109',2530:'2876719777'}
NICKS={911:'2876719109',912:'2876719109',913:'2876719109',2564:'2876719777',2565:'2876719777',2566:'2876719777'}
FORCED=re.compile(r'^Dealer:\s+(.+?)\s+posts\s+(Ante|Small Blind|Big Blind)(?:\s+(\d+(?:[ \u00a0]\d{3})*))?\s*$',re.I)
NUMBER=re.compile(r'^\d+(?:[ \u00a0]\d{3})*$')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def extract():
    layout=load_fixed_layout(ROOT/'configs/nine_table_layout.json',ROOT/'configs/layout_847x404.json',ROOT/'configs/chat_log_bottom_rois.json')
    slot=next(s for s in layout.slots if s.slot_id==3)
    profile=json.loads((ROOT/'configs/layout_847x404.json').read_text('utf-8'))['slots']['slot_03']
    jobs=[]
    def job(gray,frame,pts,base,kind,hand,**extra):
        ok,packed=cv2.imencode('.png',gray)
        if not ok:raise ValueError('RAM_ENCODING_FAILED')
        item_id=f'{kind}:{frame}:{extra.get("seat",extra.get("row_index"))}'
        jobs.append(dict(item_id=item_id,table=3,hand_id=hand,frame=frame,source_pts=pts,time_base=base,
            field=kind,pixel_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),crop_shape=list(gray.shape),
            png_base64=base64.b64encode(packed).decode('ascii'),**extra))
    for first,last in ((884,913),(2530,2566)):
        with av.open(str(ROOT/'video/smoke_17.mp4')) as video:
            stream=video.streams.video[0]
            video.seek(first*512,stream=stream,backward=True,any_frame=False)
            for fr in video.decode(video=0):
                if fr.pts is None or fr.pts%512:continue
                frame=fr.pts//512
                if frame>last:break
                if frame not in HANDS and frame not in NICKS:continue
                rgb=fr.to_ndarray(format='rgb24');base=[fr.time_base.numerator,fr.time_base.denominator]
                if frame in HANDS:
                    x,y,X,Y=slot.chat_box;chat=png_compatible_gray(np.ascontiguousarray(rgb[y:Y,x:X]));spans=physical_spans(chat)
                    for index in range(len(spans)):
                        a,b=padded_row_span(spans,index,len(chat))
                        job(np.ascontiguousarray(chat[a:b]),frame,fr.pts,base,'CHAT',HANDS[frame],
                            row_index=index,row_bounds=[a,b],source_roi=[x,y+a,X,y+b])
                if frame in NICKS:
                    x,y,X,Y=slot.table_box;table=rgb[y:Y,x:X]
                    for seat,boxes in profile['seats'].items():
                        a,b,A,B=boxes['nick'];crop=png_compatible_gray(np.ascontiguousarray(table[b:B,a:A]))
                        job(crop,frame,fr.pts,base,'NICK',NICKS[frame],seat=seat,source_roi=[x+a,y+b,x+A,y+B],roi=boxes['nick'])
    return jobs


def analyze(jobs):
    chat=collections.defaultdict(list);nick=collections.defaultdict(list)
    for r in jobs:
        (chat if r['field']=='CHAT' else nick)[r['hand_id']].append(r)
    forced=[]
    for hand,rr in chat.items():
        rr.sort(key=lambda r:r['row_index'])
        for i,r in enumerate(rr):
            raw=r['response'];text=(raw.get('raw_text') or '').strip();m=FORCED.fullmatch(text)
            if not m:continue
            amount=m.group(3);amount_row=None
            if amount is None and i+1<len(rr):
                nxt=rr[i+1];nr=nxt['response'];nt=(nr.get('raw_text') or '').strip()
                if NUMBER.fullmatch(nt) and nxt['row_index']==r['row_index']+1:
                    amount=nt;amount_row=nxt
            confidence_ok=raw.get('status')=='OK' and raw.get('score',0)>=THRESHOLD
            if amount_row is not None:
                confidence_ok &= amount_row['response'].get('status')=='OK' and amount_row['response'].get('score',0)>=THRESHOLD
            forced.append(dict(hand_id=hand,actor=m.group(1),kind=m.group(2),amount=int(amount.replace(' ','').replace('\u00a0','')) if amount else None,
                source_frame=r['frame'],row_index=r['row_index'],row_bounds=r['row_bounds'],row_sha256=r['pixel_sha256'],
                raw_text=text,confidence=raw.get('score'),amount_wrap_item=amount_row['item_id'] if amount_row else None,
                amount_wrap_confidence=amount_row['response'].get('score') if amount_row else None,
                strict_confidence_PASS=bool(confidence_ok and amount is not None)))
    groups=[]
    for hand,rr in nick.items():
        actors={r['actor'] for r in forced if r['hand_id']==hand}
        for seat in sorted({r['seat'] for r in rr}):
            reads=sorted([r for r in rr if r['seat']==seat],key=lambda r:r['frame'])
            values=[(r['response'].get('raw_text') or '').strip() for r in reads]
            confidence=[r['response'].get('score') for r in reads]
            strict=len(reads)==3 and len(set(values))==1 and bool(values[0]) and all(r['response'].get('status')=='OK' and (r['response'].get('score') or 0)>=THRESHOLD for r in reads)
            matched=values[0] if strict and values[0] in actors else None
            groups.append(dict(hand_id=hand,seat=seat,frames=[r['frame'] for r in reads],raw_nicks=values,confidences=confidence,
                strict_nickname_triple_PASS=strict,exact_forced_actor_match=matched,
                exact_match_is_unique_among_seats=False))
    for g in groups:
        actor=g['exact_forced_actor_match']
        g['exact_match_is_unique_among_seats']=bool(actor and sum(h['hand_id']==g['hand_id'] and h['exact_forced_actor_match']==actor for h in groups)==1)
    return forced,groups


def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'configs/ocr_tiny_local.json').read_text('utf-8'))
    input_names=['configs/ocr_tiny_local.json','configs/layout_847x404.json','configs/nine_table_layout.json','configs/chat_log_bottom_rois.json','parser_core/tiny_chat_client.py','scripts/tiny_chat_worker.py']
    hashes={name:sha(ROOT/name) for name in input_names}
    jobs=extract();extraction=time.perf_counter()-started
    client=TinyChatClient(Path(config['python']),Path(config['model_dir']),config['model_name'],config['weights_sha256'],config['device'],OUT/'tiny_probe.stderr.log')
    batches=[]
    try:
        for offset in range(0,len(jobs),8):
            batch=jobs[offset:offset+8];before=time.perf_counter()
            response=client.read_batch([dict(item_id=r['item_id'],png_base64=r['png_base64'],source_sha256=r['pixel_sha256']) for r in batch],8)
            results={r['item_id']:r for r in response['results']}
            for r in batch:r['response']=results[r['item_id']]
            batches.append(dict(offset=offset,count=len(batch),wall_seconds=time.perf_counter()-before,response=response))
    finally:client.close()
    with (OUT/'raw_probe_jobs.jsonl').open('w',encoding='utf-8') as log:
        for r in jobs:log.write(json.dumps(r,ensure_ascii=False)+'\n')
    forced,nicks=analyze(jobs)
    summary=dict(scope='OFFLINE_INPUT_FEASIBILITY_ONLY_NOT_LIVE_PASS',table=3,threshold=THRESHOLD,job_count=len(jobs),
        nickname_jobs=sum(r['field']=='NICK' for r in jobs),chat_jobs=sum(r['field']=='CHAT' for r in jobs),
        strict_nickname_triples=sum(r['strict_nickname_triple_PASS'] for r in nicks),
        exact_unique_actor_matches=sum(r['exact_match_is_unique_among_seats'] for r in nicks),
        strict_forced_rows=sum(r['strict_confidence_PASS'] for r in forced),forced_rows=len(forced),
        forced=forced,nickname_groups=nicks,worker_ready=client.ready,batches=batches,
        sit_event_grammar='NOT_MEASURED_NO_SIT_EVENT_IN_BOUNDED_INPUT',
        source_order_note='CHAT forced snapshot frames884/2530 precede nickname frames911-913/2564-2566. No future money events used; action onset remains UNKNOWN.',
        extraction_seconds=extraction,wall_seconds=time.perf_counter()-started,
        frozen_input_hashes=hashes,input_hashes_unchanged=all(sha(ROOT/n)==h for n,h in hashes.items()),production_edited=False)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    lines=['# Offline seat/forced-input feasibility', '', 'No live epoch or START_STACK PASS is claimed. Exact OCR spelling matching only; no nickname repair.', '',
        '|hand|seat|frames|raw nicknames|confidence|strict triple|exact unique CHAT actor|',
        '|---|---|---|---|---|---|---|']
    for g in nicks:lines.append('|'+ '|'.join(str(g[k]) for k in ['hand_id','seat','frames','raw_nicks','confidences','strict_nickname_triple_PASS','exact_match_is_unique_among_seats'])+'|')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n','utf-8')
    print(json.dumps({k:summary[k] for k in ['scope','job_count','nickname_jobs','chat_jobs','strict_nickname_triples','exact_unique_actor_matches','strict_forced_rows','forced_rows','extraction_seconds','wall_seconds','input_hashes_unchanged']},ensure_ascii=False))


if __name__=='__main__':main()
