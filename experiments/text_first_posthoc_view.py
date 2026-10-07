"""RAM-only future validation sheets; never used for causal identity decisions."""
import base64,io,json,hashlib,sys
from bisect import bisect_right
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.capture_temporal_evidence import read
from experiments.run_cross_track_linker import Frames
from parser_core.layout import load_fixed_layout
from parser_core.chat_rows import physical_spans,padded_row_span
OUT=ROOT/'runs/text_first_precision_review_smoke17_20261001'

def main():
    code=sys.argv[1];offset=int(sys.argv[2]) if len(sys.argv)>2 else 0
    frozen={r['pair_id']:r for r in read(ROOT/'runs/cross_track_linker_smoke17_20261001/relationships.jsonl')}
    e=next(r for r in read(OUT/'sealed_evidence.jsonl') if r['case_id']==code);r=frozen[e['pair_id']]
    old,new=r['old'],r['new'];t=r['decision_frame']
    ret=min(a for a,b in r['old_terminal_metadata']['visible_spans'] if a>t)
    states=[s for s in read(ROOT/'runs/temporal_context_smoke17_20261001/chat_states.jsonl') if s['session_id']==old['session_id']]
    frames=sorted(s['source_frame'] for s in states if t<=s['source_frame']<=ret)
    chosen=frames[offset:offset+4];assert chosen
    lookup={s['source_frame']:s['track_ids'] for s in states}
    times=read(ROOT/'runs/temporal_context_smoke17_20261001/frame_times.jsonl')
    src=json.loads((ROOT/'runs/production_smoke17_20261001_013909/run_metadata.json').read_text('utf-8'))
    layout=load_fixed_layout(ROOT/'configs/nine_table_layout.json',ROOT/'configs/layout_847x404.json',ROOT/'configs/chat_log_bottom_rois.json')
    reader=Frames(Path(src['video']),times,layout);panels=[];proof=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    try:
        for fid in chosen:
            gray=reader.get(fid,old['table_id']);digest=hashlib.sha256(gray.tobytes()).hexdigest()
            w,h=gray.shape[1]*2,gray.shape[0]*2;im=Image.new('RGB',(w+16,h+100),(24,24,24));draw=ImageDraw.Draw(im)
            draw.text((8,3),f'{code} POST_HOC_CHECK {old["table_id"]} frame={fid}',font=font,fill='white')
            draw.text((8,23),f'PTS={times[fid]["timestamp_ms"]:.3f}ms old={old["track_id"].split(":")[-1]} new={new["track_id"].split(":")[-1]}',font=font,fill='white')
            im.paste(Image.fromarray(gray).convert('RGB').resize((w,h),Image.Resampling.NEAREST),(8,75))
            spans=physical_spans(gray);ids=lookup[fid];assert len(ids)==len(spans)-1
            found=[]
            for track_id in (old['track_id'],new['track_id']):
                if track_id in ids:
                    idx=ids.index(track_id)+1;a,b=padded_row_span(spans,idx,gray.shape[0]);draw.rectangle((8,75+a*2,w+7,75+b*2-1),outline='yellow',width=1)
                    found.append(dict(track_id=track_id,row_index=idx,y1=a,y2=b))
            draw.text((8,45),' / '.join(f'{x["track_id"].split(":")[-1]} row={x["row_index"]} y={x["y1"]}..{x["y2"]}' for x in found),font=font,fill='yellow')
            assert digest==hashlib.sha256(gray.tobytes()).hexdigest()
            proof.append(dict(frame=fid,source_roi_sha256=digest,target_positions=found,source_pixels_unchanged=True,phase='POST_HOC_CHECK'));panels.append(im);del gray
    finally:end=reader.close()
    w=max(i.width for i in panels);h=max(i.height for i in panels);sheet=Image.new('RGB',(w*2,h*((len(panels)+1)//2)),(36,36,36))
    for i,im in enumerate(panels):sheet.paste(im,((i%2)*w,(i//2)*h))
    buf=io.BytesIO();sheet.save(buf,format='PNG');data=buf.getvalue()
    receipt=dict(case_id=code,all_changed_frames=frames,offset=offset,shown_frames=chosen,source_proofs=proof,
        image_writes=0,roi_images_at_end=end,display_sha256=hashlib.sha256(data).hexdigest(),phase='POST_HOC_CHECK',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'posthoc_render_receipts').mkdir(exist_ok=True)
    (OUT/'posthoc_render_receipts'/f'{code}_{offset}.json').write_text(json.dumps(receipt),encoding='utf-8')
    print(json.dumps(dict(image_base64=base64.b64encode(data).decode(),receipt=receipt)))

if __name__=='__main__':main()
