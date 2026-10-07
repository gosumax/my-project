"""Offline original-source review crops, outside immutable acceptance output."""
import hashlib,json,sys,time
from collections import defaultdict
from pathlib import Path
import av,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from parser_core.layout import load_fixed_layout
OUT=ROOT/'runs/smoke19_full_start_stack_validation_20261003';RUN=OUT/'acceptance_ON';REV=OUT/'review'
def rows(name):return [json.loads(s) for s in (RUN/name).read_text('utf-8').splitlines() if s.strip()]
def main():
    start=time.perf_counter();wanted=defaultdict(list)
    layout=load_fixed_layout(ROOT/'configs/nine_table_layout.json',ROOT/'configs/layout_847x404.json',ROOT/'configs/chat_log_bottom_rois.json')
    slots={s.slot_id:s for s in layout.slots}
    for t in rows('dealing_cards_triggers.jsonl'):wanted[t['trigger_source_frame']].append(('CHAT',t))
    for d in rows('deal_complete_receipts.jsonl'):
        if d.get('evidence_status')=='PROVEN':
            for f in d['card_stable_frames']:wanted[f].append(('TABLE_NCC',d))
    proofs=[];cards=[];dest=REV/'assets/source';dest.mkdir(parents=True,exist_ok=True)
    with av.open(str(ROOT/'video/smoke_19.mp4')) as container:
        for frame_id,frame in enumerate(container.decode(video=0)):
            if frame_id in wanted:
                rgb=frame.to_ndarray(format='rgb24')
                for kind,r in wanted[frame_id]:
                    slot=slots[r['table_id']];x,y,X,Y=slot.chat_box if kind=='CHAT' else slot.table_box
                    pixels=np.ascontiguousarray(rgb[y:Y,x:X]);digest=hashlib.sha256(pixels.tobytes()).hexdigest()
                    name=f"t{r['table_id']:02d}_{r['hand_id']}_{kind}_f{frame_id:05d}.png"
                    Image.fromarray(pixels).save(dest/name)
                    expected=r.get('table_pixel_sha256') if kind=='TABLE_NCC' and frame_id==r['source_frame'] else None
                    proof=dict(kind=kind,table=r['table_id'],session=r['session_id'],hand=r['hand_id'],frame=frame_id,
                        pts=frame.pts,time_base=str(frame.time_base),ROI=[x,y,X,Y],rgb_sha256=digest,
                        expected_confirmed_table_hash=expected,hash_match=digest==expected if expected else 'NO_SAVED_FULL_CROP_HASH',
                        file='assets/source/'+name)
                    proofs.append(proof);cards.append(f"<section><h3>T{r['table_id']} hand{r['hand_id']} {kind} frame{frame_id} PTS{frame.pts}</h3><img src='{proof['file']}'><p>{digest}</p></section>")
            if frame_id>=max(wanted):break
    (REV/'source_frame_provenance.json').write_text(json.dumps(proofs,indent=2),'utf-8')
    (REV/'source_frames.html').write_text('<!doctype html><meta charset=utf-8><title>Smoke19 original sources</title><p>Offline redecoded source crops; no OCR or production changes. Only final TABLE confirmation has a saved full RGB hash; preceding frames are linked to original video/frame/PTS and NCC receipts.</p>'+''.join(cards),'utf-8')
    summary=dict(crops=len(proofs),chat=sum(p['kind']=='CHAT' for p in proofs),table_ncc=sum(p['kind']=='TABLE_NCC' for p in proofs),
        confirmed_hashes=sum(p['expected_confirmed_table_hash'] is not None for p in proofs),hash_conflicts=sum(p['hash_match'] is False for p in proofs),seconds=time.perf_counter()-start)
    (REV/'source_extraction_summary.json').write_text(json.dumps(summary,indent=2),'utf-8');print(json.dumps(summary))
if __name__=='__main__':main()
