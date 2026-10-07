"""Bounded OFFLINE_REPROCESS of 24 saved sources; never live acceptance."""
import base64
import hashlib
import json
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.dealing_start_stack import DealingStartStackSidecar, CleanStackWindow, NUMERIC
from parser_core.start_stack_live import StartStackLiveConnection, first_visible_stack_triple
from parser_core.ram_first_stage import png_compatible_gray
from parser_core.parallel_ocr import ParallelRowOCR
from parser_core.tiny_chat_client import TinyChatClient

BASE = ROOT / 'runs/first_stack_clean_window_20261003/acceptance_ON_math'
OUT = ROOT / 'runs/deal_complete_fix_chat_review_20261003/first_numeric_reprocess'
def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)
def sha(p): return hashlib.sha256(p).hexdigest()

def main():
    started=time.perf_counter(); OUT.mkdir(exist_ok=True)
    flags=json.loads((ROOT/'configs/dealing_start_stack.json').read_text('utf-8'))
    config=json.loads((ROOT/'configs/ocr_tiny_local.json').read_text('utf-8'))
    profile=json.loads((ROOT/flags['roi_profile']).read_text('utf-8'))
    manifest=json.loads((ROOT/'production_manifest.json').read_text('utf-8'))
    frozen={n:sha((ROOT/n).read_bytes()) for n in manifest['hashes']}
    captures=[r for r in rows(BASE/'start_stack_capture.jsonl') if r.get('table_id')==3 and r.get('field')=='stack' and r.get('source_frame') in (2564,2565,2566)]
    assert len(captures)==24
    from PIL import Image, ImageDraw
    selected=[r for r in captures if r['seat']=='seat4']
    im=Image.new('RGB',(1000,350),'#eeeeee'); d=ImageDraw.Draw(im)
    for idx,r in enumerate(selected):
        crop=Image.open(__import__('io').BytesIO(base64.b64decode(r['png_base64']))); crop.load()
        im.paste(crop.resize((crop.width*8,crop.height*8),Image.Resampling.NEAREST),(220,15+idx*110))
        d.text((10,30+idx*110),str(r['source_frame'])+' RGB exact crop',fill='black')
    im.save(OUT/'seat4_2564_2566_rgb.png')
    deal=next(r for r in rows(BASE/'deal_complete_receipts.jsonl') if r.get('table_id')==3 and r.get('hand_id')=='2876719777')
    headers=[r for r in rows(BASE/'hand_state_receipts.jsonl') if r.get('table_id')==3 and r.get('hand_id')=='2876719777' and r.get('status')=='HAND_HEADER_CONFIRMED']
    confirmed=next(r for r in headers if r['source_frame']==2540)
    h=dict(min(headers,key=lambda r:r['source_frame']),time_base=[1,15360])
    key=(3,deal['session_id'],deal['hand_id']); w=CleanStackWindow(flags['stack_confidence']); w.confirm_deal(deal)
    side=DealingStartStackSidecar.__new__(DealingStartStackSidecar)
    side.lock=threading.RLock(); side.current={3:h}; side.active_sessions={3:key[1]}; side.windows={key:w}
    side.flags=flags; side.profile=profile; side.outstanding=defaultdict(set); side.header_fences={}; side.hand_state_gate=None
    side.histories=defaultdict(list,{key[:2]:headers}); side.logs={}
    side._log=lambda name,r: None
    conn=StartStackLiveConnection.__new__(StartStackLiveConnection)
    conn.sidecar=side; conn.lock=threading.RLock(); conn.counts=Counter(); conn.states={}; conn.jobs={}
    conn.log=(OUT/'registered_callback_receipts.jsonl').open('w',encoding='utf-8')
    state=dict(card_run=0,last=None,deal_frame=2530,crops={},submitted=Counter(),numeric_runs={},numeric_triples={},stop=set(),source_readings={},canonical_attempts={})
    conn.states[key]=state
    existing=[r for r in rows(BASE/'start_stack_readings.jsonl') if (r.get('table_id'),r.get('session_id'),r.get('hand_id'))==key]
    for r in existing:
        if 2531<=r['source_frame']<=2563:
            state['source_readings'].setdefault(r['seat'],{})[r['source_frame']]=r
    gap=ROOT/'runs/deal_complete_fix_chat_review_20261003/provisional_reprocess/registered_callback_receipts.jsonl'
    for r in rows(gap):
        assert r['source_frame'] in (2540,2541)
        state['source_readings'].setdefault(r['seat'],{})[r['source_frame']]=r
    # Preserve actual earlier receipts; never fill absent observations with visual labels.
    # The diagnostic first-triple check remains fail closed if earlier numeric-lane coverage is absent.
    provenance=[]; batches=[]
    for frame in (2564,2565,2566):
        group=[r for r in captures if r['source_frame']==frame]; evidence={}
        for r in group:
            assert r['session_id']==key[1] and r['source_pts']==frame*512 and r['time_base']==[1,15360]
            assert r['roi']==profile['slots']['slot_03']['seats'][r['seat']]['stack']
            bgr=cv2.imdecode(np.frombuffer(base64.b64decode(r['png_base64']),np.uint8),cv2.IMREAD_COLOR)
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            assert sha(rgb.tobytes())==r['pixel_sha256']
            gray=png_compatible_gray(np.ascontiguousarray(rgb)); evidence[r['seat']]=gray
            provenance.append(dict(r,hand_id=key[2],rgb_hash_verified=True,gray_sha256=sha(gray.tobytes()),same_hand_confirmation=confirmed,scope='OFFLINE_REPROCESS'))
        batches.append(conn._observe_frame(3,key[1],frame,frame*512,(1,15360),None,None,stack_evidence=evidence))
    assert sum(map(len,batches))==24
    (OUT/'source_provenance.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in provenance),'utf-8')
    def factory(worker):
        return TinyChatClient(Path(config['python']),Path(config['model_dir']),config['model_name'],config['weights_sha256'],config['device'],OUT/'tiny.stderr.log')
    ocr=ParallelRowOCR(factory,workers=1,queue_size=32,batch_size=8,result_observer=conn.completed)
    for batch in batches: ocr.submit_batch(batch)
    ocr.finish(); conn.log.close()
    assert not ocr.errors, ocr.errors
    processed=list(rows(OUT/'registered_callback_receipts.jsonl'))
    assert len(processed)==24
    table=[dict(seat=r['seat'],frame=r['source_frame'],raw_text=r['raw_text'],confidence=r['confidence'],ocr_status=r['ocr_status'],qualified_numeric=r['ocr_status']=='OK' and bool(NUMERIC.fullmatch((r['raw_text'] or '').strip())) and r['confidence']>=flags['stack_confidence'],pixel_sha256=r['pixel_sha256']) for r in processed]
    attempts={s:first_visible_stack_triple(rr,flags['stack_confidence'],final=True,start_frame=2531) for s,rr in state['source_readings'].items()}
    old=json.loads((ROOT/'runs/seat_forced_binding_20261003/acceptance_ON/seat_forced_summary.json').read_text('utf-8'))
    prior={r['seat']:dict(frames=r['frames'],displayed_stack=r['displayed_stack']) for r in old['results'] if r['hand_id']==key[2]}
    report=dict(scope='OFFLINE_REPROCESS_NOT_LIVE_ACCEPTANCE',readings=table,attempts=attempts,prior_diagnostic=prior,registered_hashes_unchanged=all(sha((ROOT/n).read_bytes())==v for n,v in frozen.items()),errors=ocr.errors,wall_seconds=time.perf_counter()-started)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2),'utf-8')
    print(json.dumps(dict(scope=report['scope'],readings=table,attempts={s:{k:v for k,v in a.items() if k!='evidence'} if a else None for s,a in attempts.items()},registered_hashes_unchanged=report['registered_hashes_unchanged'],wall_seconds=report['wall_seconds'])))
if __name__=='__main__': main()
