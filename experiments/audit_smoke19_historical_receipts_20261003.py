"""Per-context/frame raw evidence proof; no OCR or production mutation."""
import base64,hashlib,json,re,sys,threading
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from parser_core.seat_forced_binding import SeatForcedBinding,LABEL_OVERLAYS
from parser_core.seat_epoch_state import SeatEpochMachine
RUN=ROOT/'runs/smoke19_full_start_stack_validation_20261003/acceptance_ON';OUT=RUN.parent/'validation'
def rows(n):return [json.loads(l) for l in (RUN/(n+'.jsonl')).read_text('utf-8').splitlines() if l.strip()]
def main():
    seat=json.loads((RUN/'seat_forced_summary.json').read_text('utf-8'));inputs=rows('seat_forced_inputs');deals={ (r['table_id'],r['session_id'],r['hand_id']):r for r in rows('deal_complete_receipts') if r.get('evidence_status')=='PROVEN'};profile=json.loads((ROOT/'configs/layout_847x404.json').read_text('utf-8'))
    labels=defaultdict(list);runs={};nick=defaultdict(list);chat=defaultdict(list)
    for r in inputs:
        k=(r['table_id'],r['session_id'],r['hand_id']);frame=r['source_frame'];response=r['ocr_response'];text=(response.get('raw_text') or '').strip();score=response.get('score') or 0
        if r['kind']=='CHAT':chat[(*k,frame,r['source_ref'])].append(r);continue
        bk=(*k,r['seat']);nick[(*bk,frame)].append(r)
        if response.get('status')!='OK' or score<.95 or not text:runs.pop(bk,None);continue
        old=runs.get(bk);frames=old['frames']+[frame] if old and old['text']==text and old['frames'][-1]==frame-1 else [frame]
        runs[bk]=dict(text=text,frames=frames[-3:])
        if len(frames)>=3 and text.upper() not in LABEL_OVERLAYS and re.search(r'[A-Za-zА-Яа-я]',text) and not text.endswith('...'):labels[bk].append(dict(actor=text,frames=frames[-3:],source_ref=r['pixel_sha256']))
    machine=SeatEpochMachine()
    for t in rows('dealing_cards_triggers'):machine.hand(t['table_id'],t['session_id'],t['hand_id'],t['trigger_source_frame'])
    for e in seat['state_receipts']:machine.observe(e['table'],e['session'],e['seat'],e['kind'],e['frame'],actor=e.get('actor'),hand_id=e.get('hand_id'),evidence_status=e['evidence_status'],source_ref=e.get('source_ref'),source_index=e['source_index'])
    machine.build();b=SeatForcedBinding.__new__(SeatForcedBinding);b.machine=machine;b.nick_labels=labels;b.sidecar=SimpleNamespace(windows={k:SimpleNamespace(deal=d) for k,d in deals.items()},flags={'capture_frames':64})
    proofs=[]
    for saved in seat['historical_binding_receipts']:
        k=(saved['table'],saved['session'],saved['hand_id']);actor=saved['actor'];posts=[p for p in seat['bound_posts'] if (p['table'],p['session'],p['hand_id'],p['actor'])==(*k,actor)]
        replay,block=b.historical_forced_binding(k,actor,posts,seat['unresolved_state_events']);raw_checks=[]
        for label in saved['label_evidence']:
            frames=label['frames'];consecutive=frames==list(range(frames[0],frames[0]+3))
            for frame in frames:
                candidates=nick[(*k,saved['seat'],frame)];checks=[]
                for r in candidates:
                    pixels=cv2.imdecode(np.frombuffer(base64.b64decode(r['png_base64']),np.uint8),cv2.IMREAD_GRAYSCALE)
                    checks.append(dict(hash=hashlib.sha256(pixels.tobytes()).hexdigest()==r['pixel_sha256'],ROI=r['roi']==profile['slots'][f"slot_{k[0]:02d}"]['seats'][saved['seat']]['nick'],PTS=r['source_pts']==frame*512 and r['time_base']==[1,15360],exact_actor=r['ocr_response']['raw_text'].strip()==actor,confidence=r['ocr_response']['score']>=.95,status=r['ocr_response']['status']=='OK'))
                raw_checks.append(dict(frame=frame,source_count=len(candidates),consecutive=consecutive,checks=checks,valid=len(candidates)==1 and consecutive and all(all(c.values()) for c in checks)))
        post_checks=[]
        for p in posts:
            for ref in p['source_refs']:
                matches=chat[(*k,p['source_frame'],ref)]
                valid=len(matches)==1
                if valid:
                    r=matches[0];pixels=cv2.imdecode(np.frombuffer(base64.b64decode(r['png_base64']),np.uint8),cv2.IMREAD_GRAYSCALE);valid=hashlib.sha256(pixels.tobytes()).hexdigest()==ref and r['source_pts']==p['source_pts'] and r['time_base']==p['time_base'] and r['ocr_response']['status']=='OK' and r['ocr_response']['score']>=.95
                post_checks.append(dict(role=p['role'],frame=p['source_frame'],source_ref=ref,source_count=len(matches),valid=valid))
        proofs.append(dict(table=k[0],session=k[1],hand=k[2],actor=actor,seat=saved['seat'],anchor=saved['binding_anchor_frame'],registered_replay_exact=replay==saved,blocker=block,nickname_raw_checks=raw_checks,post_raw_checks=post_checks,all_checks=replay==saved and all(r['valid'] for r in raw_checks+post_checks)))
    report=dict(receipts=len(proofs),passed=sum(r['all_checks'] for r in proofs),failed=sum(not r['all_checks'] for r in proofs),scope='RAW_CONTEXT_FRAME_LOOKUP_NO_PIXEL_HASH_COLLAPSE',proofs=proofs)
    (OUT/'historical_raw_receipt_audit.json').write_text(json.dumps(report,indent=2),'utf-8');print(json.dumps({k:v for k,v in report.items() if k!='proofs'}))
if __name__=='__main__':main()
