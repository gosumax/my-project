"""Bounded direct-source TABLE proof extraction outside official run outputs."""
import hashlib
import json
from pathlib import Path

import av
import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/start_stack_live_path_20261003/connection_independent_table_review'
RUN = ROOT/'runs/start_stack_live_path_20261003/connection_acceptance_ON'


def main():
    OUT.mkdir(exist_ok=True)
    summary=json.loads((RUN/'start_stack_live_summary.json').read_text('utf-8'))
    hands={(h['table_id'],h['hand_id']):h for h in summary['hands'] if h.get('deal_complete_frame') is not None}
    layout=json.loads((ROOT/'configs/nine_table_layout.json').read_text('utf-8'))
    slots={int(r['table']):r for r in layout['tables']}
    proofs=[]
    for line in (RUN/'start_stack_live_receipts.jsonl').open(encoding='utf-8'):
        r=json.loads(line)
        if r['stage']=='DEAL_COMPLETE' and (r['table_id'],r['hand_id']) in hands and r['source_frame']==hands[r['table_id'],r['hand_id']]['deal_complete_frame']:
            proofs.append(r)
    output=[]
    for proof in proofs:
        table, hand=proof['table_id'],proof['hand_id']
        wanted=set(proof['card_stable_frames']); frames=[]
        with av.open(str(ROOT/'video/smoke_17.mp4')) as video:
            stream=video.streams.video[0]
            # Known source PTS/timebase from the actual receipt, no timer opener.
            step=512
            video.seek(min(wanted)*step,stream=stream,backward=True,any_frame=False)
            for fr in video.decode(video=0):
                if fr.pts is None or fr.pts % step:
                    continue
                frame=fr.pts//step
                if frame>max(wanted):
                    break
                if frame not in wanted:
                    continue
                rgb=fr.to_ndarray(format='rgb24');s=slots[table]
                crop=rgb[s['y']:s['y']+s['height'],s['x']:s['x']+s['width']]
                path=OUT/f'T{table:02d}_{hand}_{frame}_table.png'
                ok,packed=cv2.imencode('.png',cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
                if not ok:
                    raise RuntimeError('INDEPENDENT_REVIEW_PNG_ENCODING_FAILED')
                path.write_bytes(packed.tobytes())
                frames.append({'frame':frame,'source_pts':fr.pts,'table_pixel_sha256':hashlib.sha256(crop.tobytes()).hexdigest(),
                    'image':str(path.relative_to(ROOT)),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        exact=next((r['table_pixel_sha256']==proof['table_pixel_sha256'] for r in frames if r['frame']==proof['source_frame']),False)
        output.append({'table':table,'hand_id':hand,'live_deal_frame':proof['source_frame'],
            'source_frames_complete':{r['frame'] for r in frames}==wanted,'live_confirmation_pixel_hash_exact':exact,
            'frames':frames,'visual_semantic_review':'PENDING_MANUAL_INSPECTION'})
    (OUT/'all_proof_source_images.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({'proof_groups':len(output),'source_images':sum(len(r['frames']) for r in output),
        'all_frame_sets_exact':all(r['source_frames_complete'] for r in output),
        'all_live_confirmation_hashes_exact':all(r['live_confirmation_pixel_hash_exact'] for r in output)}))


if __name__=='__main__':
    main()
