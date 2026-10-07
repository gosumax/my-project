"""Frozen sample and RAM-only source viewport display for blind visual review."""
from __future__ import annotations
import argparse, base64, hashlib, io, json, textwrap
from collections import Counter
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.capture_temporal_evidence import read,sha,write
from experiments.run_cross_track_linker import Frames
from experiments.run_ram_crop_metadata import image_inventory
from parser_core.layout import load_fixed_layout
from parser_core.chat_rows import physical_spans,padded_row_span
OUT=ROOT/'runs/text_first_precision_review_smoke17_20261001'
FROZEN=ROOT/'runs/cross_track_linker_smoke17_20261001/relationships.jsonl'
TEXT=ROOT/'runs/text_first_identity_smoke17_20261001/relationships.jsonl'
CONTENT=ROOT/'runs/content_identity_smoke17_20261001/pairs.jsonl'

def paths():
    manifest=json.loads((ROOT/'production_manifest.json').read_text('utf-8'))
    source=json.loads((ROOT/'runs/production_smoke17_20261001_013909/run_metadata.json').read_text('utf-8'))
    return manifest,[ROOT/p for p in manifest['hashes']]+[ROOT/'production_manifest.json',FROZEN,TEXT,CONTENT,
        ROOT/'experiments/cross_track_linker.py',ROOT/'experiments/run_cross_track_linker.py',
        ROOT/'experiments/text_first_identity_audit.py',ROOT/'experiments/row_content_evidence.py',
        ROOT/'experiments/run_content_identity_audit.py',Path(__file__),
        ROOT/'runs/temporal_context_smoke17_20261001/frame_times.jsonl',Path(source['video'])]

def prepare():
    OUT.mkdir(exist_ok=False)
    manifest,inputs=paths();hashes={str(p):sha(p) for p in inputs}
    assert all(hashes[str(ROOT/p)]==h for p,h in manifest['hashes'].items())
    inventory=image_inventory()
    (OUT/'inventory_before.json').write_text(json.dumps(inventory),encoding='utf-8')
    frozen={r['pair_id']:r for r in read(FROZEN)}
    content={r['pair_id']:r for r in read(CONTENT)}
    decisions=read(TEXT);candidates=[r for r in decisions if r['simulation_verdict']=='TEXT_FIRST_SAME_CANDIDATE']
    ambiguous=[r for r in decisions if r['simulation_verdict']=='AMBIGUOUS']
    assert len(candidates)==280 and len(ambiguous)==310
    selected={r['pair_id']:r for r in candidates};fn_ids=set();fn_tags={}
    def take(label,group,n):
        group=sorted(group,key=lambda r:(len(r['blocking_reasons']),sha_key(r['pair_id'])))
        for r in group[:n]:
            fn_ids.add(r['pair_id']);fn_tags.setdefault(r['pair_id'],[]).append(label);selected[r['pair_id']]=r
    take('FN_GAP2',[r for r in ambiguous if frozen[r['pair_id']]['features']['delta_frame']==2],7)
    take('FN_DUPLICATE',[r for r in ambiguous if r['text_class']!='TEXT_UNIQUE'],15)
    take('FN_NO_MAPPING',[r for r in ambiguous if r['mapping']=='NO_TEXT_GEOMETRY_MAPPING'],10)
    take('FN_WEAK_CONTENT',[r for r in ambiguous if 'SUPPORTING_CONTENT' in r['blocking_reasons']],15)
    take('FN_NEAR_ADMISSION',[r for r in ambiguous if len(r['blocking_reasons'])==1],15)
    take('FN_NORMAL',[r for r in ambiguous if not r['repeat_danger_flags']],15)
    if len(fn_ids)<50:take('FN_FILL',[r for r in ambiguous if r['pair_id'] not in fn_ids],50-len(fn_ids))
    # All candidates reviewed, including all weak/max-y/repeated-template tails.
    weak=sorted(candidates,key=lambda r:(-max(
        content[r['pair_id']]['target']['best_alignment']['foreground_union']['mae']/content[r['pair_id']]['control_max_foreground_mae'] if content[r['pair_id']]['control_max_foreground_mae'] else 0,
        content[r['pair_id']]['support_xor_ratio']/content[r['pair_id']]['control_max_support_xor_ratio'] if content[r['pair_id']]['control_max_support_xor_ratio'] else 0),sha_key(r['pair_id'])))[:28]
    weak_ids={r['pair_id'] for r in weak}
    yres=lambda r:next(g['center_residual'] for g in content[r['pair_id']]['all_new_row_geometry'] if g['new_id']==frozen[r['pair_id']]['new']['track_id'])
    max_y=max(map(yres,candidates))
    sealed=[];public=[]
    for i,r in enumerate(sorted(selected.values(),key=lambda r:sha_key(r['pair_id']))):
        f=frozen[r['pair_id']];c=content[r['pair_id']];o,n=f['old'],f['new'];cid=f'V{i+1:03}'
        categories=list(r['repeat_danger_flags']) or ['NORMAL']
        if r['pair_id'] in weak_ids:categories.append('WEAK_CONTENT')
        if yres(r)==max_y:categories.append('MAX_Y_RESIDUAL')
        if f['features']['delta_frame']==2:categories.append('GAP2')
        if r['POST_HOC_CHECK']['old_reappears_after_decision']:categories.append('POST_HOC_RISK')
        categories+=fn_tags.get(r['pair_id'],[])
        ev=dict(case_id=cid,pair_id=r['pair_id'],categories=categories,original_simulation_verdict=r['simulation_verdict'],
            old=o,new=n,decision_frame=f['decision_frame'],delta_frames=f['features']['delta_frame'],delta_ms=f['features']['delta_ms'],
            normalized_text=o['normalized_text'],text_uniqueness={k:r[k] for k in ('text_class','old_viewport_text_count','new_viewport_text_count','history_other_same_text_ids','unknown_viewport_text_ids')},
            expected_y=c['old_target_tight_center']-f['motion']['shift_px'],actual_y=c['old_target_tight_center']-f['motion']['shift_px']+0,
            y_residual=yres(r),viewport_shift=f['motion']['shift_px'],same_id_coherent_neighbors=r['same_id_coherent_neighbors'],
            neighbor_details=c['coherent_neighbors'],target_foreground_mae=c['target']['best_alignment']['foreground_union']['mae'],
            neighbor_foreground_mae_envelope=c['control_max_foreground_mae'],target_support_xor=c['support_xor_ratio'],neighbor_support_xor_envelope=c['control_max_support_xor_ratio'],
            registered_row_distance=r['supporting_registered_row_distance'],content_distance_alignment=c['target'],
            boundary_frames=f['features']['hand_boundary_frames'],simultaneous_visibility=f['features']['causal_visibility_overlap'],
            local_loss_bridge=f['motion']['local_loss_bridge'],POST_HOC_CHECK=r['POST_HOC_CHECK'])
        ev['actual_y']=next(g['tight_center'] for g in c['all_new_row_geometry'] if g['new_id']==n['track_id'])
        sealed.append(ev)
        public.append(dict(case_id=cid,old=o,new=n,decision_frame=f['decision_frame'],before_roi_hash=f['motion']['before_hash'],after_roi_hash=f['motion']['after_hash']))
    write(OUT/'sealed_evidence.jsonl',sealed);write(OUT/'blind_cases.jsonl',public)
    ids=[r['case_id'] for r in public];shared=ids[:20]
    (OUT/'review_assignments.json').write_text(json.dumps(dict(A=ids[::2]+[v for v in shared if v not in ids[::2]],B=ids[1::2]+[v for v in shared if v not in ids[1::2]],shared=shared),indent=2),encoding='utf-8')
    summary=dict(sample_total=len(sealed),prospective_candidates=280,ambiguous_sample=len(fn_ids),
        candidate_category_counts=dict(Counter(cat for e in sealed if e['original_simulation_verdict']=='TEXT_FIRST_SAME_CANDIDATE' for cat in e['categories'])),
        ambiguous_category_counts=dict(Counter(cat for e in sealed if e['original_simulation_verdict']=='AMBIGUOUS' for cat in e['categories'])),
        max_candidate_y_residual=max_y,weak_content_definition='28 closest to local observed control envelope boundary (top decile), all reviewed plus all remaining candidates',
        selection='All280 candidates + stratified >=50 AMBIGUOUS; deterministic SHA256 ordering hides admission class',
        hashes_before=hashes,image_inventory_count=len(inventory),review_labels=('VISUALLY_SAME','VISUALLY_DIFFERENT','VISUALLY_UNCERTAIN'),
        blindness='Images expose only IDs/table/frame/time/target/raw pixels/text; classification VISUAL_REVIEW_PENDING, never TEXT-FIRST verdict or scores. No terminal evidence until posthoc phase.')
    (OUT/'sample_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in summary.items() if k!='hashes_before'}))

def sha_key(s):return hashlib.sha256(s.encode()).hexdigest()

def render(codes,scale=2,posthoc=False):
    cases={r['case_id']:r for r in read(OUT/'blind_cases.jsonl')}
    times=read(ROOT/'runs/temporal_context_smoke17_20261001/frame_times.jsonl')
    source=json.loads((ROOT/'runs/production_smoke17_20261001_013909/run_metadata.json').read_text('utf-8'))
    layout=load_fixed_layout(ROOT/'configs/nine_table_layout.json',ROOT/'configs/layout_847x404.json',ROOT/'configs/chat_log_bottom_rois.json')
    reader=Frames(Path(source['video']),times,layout);panels=[];proofs=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
    try:
        for code in codes:
            c=cases[code];o,n=c['old'],c['new'];frames=[o['last_frame'],n['first_frame']]
            if frames[1]-frames[0]>1:frames.insert(1,frames[1]-1)
            if posthoc:
                sealed=next(r for r in read(OUT/'sealed_evidence.jsonl') if r['case_id']==code)
                full=next(r for r in read(FROZEN) if r['pair_id']==sealed['pair_id'])['old_terminal_metadata']['visible_spans']
                future=[a for a,b in full if a>c['decision_frame']]
                if future:frames.append(min(future))
            width=reader.get(frames[0],o['table_id']).shape[1];height=reader.get(frames[0],o['table_id']).shape[0]
            panel=Image.new('RGB',(len(frames)*(width*scale+16)+8,height*scale+190),(24,24,24));draw=ImageDraw.Draw(panel)
            heading=f"{code} | {o['table_id']} {o['session_id']} | {'POST_HOC_ONLY' if posthoc else 'VISUAL_REVIEW_PENDING'}"
            draw.text((8,4),heading,font=font,fill='white')
            draw.text((8,23),f"OLD {o['track_id']}  NEW {n['track_id']}",font=font,fill='white')
            for j,fid in enumerate(frames):
                assert posthoc or fid<=c['decision_frame']
                gray=reader.get(fid,o['table_id']);original=hashlib.sha256(gray.tobytes()).hexdigest()
                if fid==o['last_frame']:assert original==c['before_roi_hash']
                if fid==n['first_frame']:assert original==c['after_roi_hash']
                rgb=Image.fromarray(gray).convert('RGB').resize((width*scale,height*scale),Image.Resampling.NEAREST)
                x=8+j*(width*scale+16);y=100;panel.paste(rgb,(x,y))
                title='BEFORE' if fid==o['last_frame'] else 'AFTER' if fid==n['first_frame'] else 'POST_HOC' if fid>c['decision_frame'] else 'CAUSAL_PREVIOUS'
                draw.text((x,47),f'{title} frame={fid} PTS={times[fid]["timestamp_ms"]:.3f}ms',font=font,fill='white')
                target=o if fid==o['last_frame'] else n if fid==n['first_frame'] else None
                if target:
                    spans=physical_spans(gray);a,b=padded_row_span(spans,target['row_index'],gray.shape[0]);crop=gray[a:b]
                    assert hashlib.sha256(crop.tobytes()).hexdigest()==target['pixel_hash']
                    draw.text((x,67),f'row={target["row_index"]} y={a}..{b} ID={target["track_id"].split(":")[-1]}',font=font,fill='yellow')
                    draw.rectangle((x,y+a*scale,x+width*scale-1,y+b*scale-1),outline='yellow',width=1)
                    zoom=Image.fromarray(crop).convert('RGB').resize((width*scale,crop.shape[0]*3),Image.Resampling.NEAREST)
                    panel.paste(zoom,(x,y+height*scale+7))
                    del crop
                assert hashlib.sha256(gray.tobytes()).hexdigest()==original
                proofs.append(dict(case_id=code,frame=fid,source_roi_sha256=original,source_pixels_unchanged=True,phase='POST_HOC_CHECK' if fid>c['decision_frame'] else 'CAUSAL',timestamp_ms=times[fid]['timestamp_ms']))
                del gray,rgb
            text=' / '.join(textwrap.wrap(o['normalized_text'],90))
            draw.text((8,height*scale+168),text,font=font,fill='white')
            panels.append(panel)
    finally:
        end=reader.close()
    cols=2 if len(panels)>1 else 1;rows=(len(panels)+cols-1)//cols
    w=max(p.width for p in panels);h=max(p.height for p in panels)
    sheet=Image.new('RGB',(w*cols,h*rows),(36,36,36))
    for i,p in enumerate(panels):sheet.paste(p,(i%cols*w,i//cols*h))
    buffer=io.BytesIO();sheet.save(buffer,format='PNG');payload=buffer.getvalue()
    proof=dict(cases=codes,posthoc=posthoc,source_proofs=proofs,display_sha256=hashlib.sha256(payload).hexdigest(),
        image_writes=0,roi_images_at_end=end,decoded_seek_frames=reader.decoded)
    key=sha_key(','.join(codes)+str(posthoc))[:16]
    (OUT/'render_receipts').mkdir(exist_ok=True)
    (OUT/'render_receipts'/f'{key}.json').write_text(json.dumps(proof),encoding='utf-8')
    print(json.dumps(dict(image_base64=base64.b64encode(payload).decode(),mime_type='image/png',receipt=proof)))

def audit():
    sample=json.loads((OUT/'sample_summary.json').read_text('utf-8'));_,inputs=paths()
    hashes={str(p):sha(p) for p in inputs};assert hashes==sample['hashes_before']
    inventory=image_inventory();before=json.loads((OUT/'inventory_before.json').read_text('utf-8'));assert {k:list(v) for k,v in inventory.items()}==before
    (OUT/'safety_audit.json').write_text(json.dumps(dict(hashes_unchanged=True,image_inventory_count=len(inventory),image_writes=0,source_data_changed=False,production_code_changed=False,hashes_after=hashes),indent=2),encoding='utf-8')
    print(json.dumps(dict(hashes_unchanged=True,image_writes=0,image_inventory_count=len(inventory))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','render','audit'));p.add_argument('--cases',nargs='+');p.add_argument('--scale',type=int,default=2);p.add_argument('--posthoc',action='store_true');a=p.parse_args()
    prepare() if a.mode=='prepare' else render(a.cases,a.scale,a.posthoc) if a.mode=='render' else audit()
