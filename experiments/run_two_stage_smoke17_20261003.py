"""Bounded direct-decode held-out measurement; never launches production."""
import argparse, base64, collections, hashlib, json, re, sys, time
from pathlib import Path
import av
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from parser_core.layout import load_fixed_layout
from parser_core.chat_rows import physical_spans,padded_row_span
from parser_core.ram_first_stage import png_compatible_gray

SOURCE=ROOT/'runs/production_poker_smoke17_full_candidate_v2_20261002'
EXPECTED_VIDEO_SHA='27c0ba10544e0f6025e2d563baf7693d36c185cdbbfeb4517d3486116e9f8baf'
HEADER=re.compile(r'^(?:Hand\s*)?#\s*(\d{8,})\s*$')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):
    with path.open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]

def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def prepare_labels(first,last):
    physical={x['row_key']:x for x in read(SOURCE/'physical_rows.jsonl') if first<=x['frame_id']<=last}
    byframe=collections.defaultdict(list)
    for o in read(SOURCE/'raw_ocr_results.jsonl'):
        if not first<=o['frame_id']<=last:continue
        p=physical[o['row_key']];assert p['row_pixel_sha256']==o['row_pixel_sha256']
        byframe[(p['frame_id'],p['table_id'])].append((p,o['raw_ocr_response'].get('raw_text','')))
    return byframe

def run(args):
    from dealing_two_stage_ncc_20261003 import TwoStageModel,CausalRowFrontier,OncePerHandLatch,HandStateTrigger,bind_current
    start=time.perf_counter();args.output.mkdir(parents=True,exist_ok=True)
    layout=load_fixed_layout(ROOT/'configs/nine_table_layout.json',ROOT/'configs/layout_847x404.json',ROOT/'configs/chat_log_bottom_rois.json')
    # Freeze window and configuration before looking at held-out labels.
    config=args.config
    engine=TwoStageModel(config);frontiers={s.slot_id:CausalRowFrontier() for s in layout.slots};latch=OncePerHandLatch(3);hand_state=HandStateTrigger(1)
    manifest={'scope':'NEW_HELDOUT_SMOKE17_DIRECT_VIDEO','frames_inclusive':[args.first_frame,args.last_frame],
              'thresholds_frozen_before_labels':True,'video':str(args.video),'video_sha256':digest(args.video),
              'source_labels':'Raw OCR labels inspected only for hand headers before classification; Dealing labels only after classification.',
              'hand_state_mode':args.hand_state,'production_changed':False,'core_sha256':digest(ROOT/'experiments/dealing_two_stage_ncc_20261003.py'),
              'runner_sha256':digest(Path(__file__)),'frozen_config_sha256':digest(config)}
    assert manifest['video_sha256']==EXPECTED_VIDEO_SHA
    dump(args.output/'frozen_manifest.json',manifest)
    labels=prepare_labels(args.first_frame,args.last_frame);contexts={};verified=[];truth=[];triggers=[]
    counts=collections.Counter();lastheader={};headerfirst={};headerindex={}
    def line(f,x):f.write(json.dumps(x,ensure_ascii=False)+'\n')
    with (args.output/'row_evaluations.jsonl').open('w',encoding='utf-8') as rowsout,(args.output/'frame_receipts.jsonl').open('w',encoding='utf-8') as frameout,av.open(str(args.video)) as source:
        stream=source.streams.video[0];stream.thread_type='FRAME';stream.codec_context.thread_count=3
        for frame_id,frame in enumerate(source.decode(stream)):
            if frame_id>args.last_frame:break
            counts['source_decoded_frames']+=1
            if frame_id<args.first_frame:continue
            rgb=frame.to_ndarray(format='rgb24');counts['decoded_frames']+=1
            for slot in layout.slots:
                x1,y1,x2,y2=slot.chat_box;gray=png_compatible_gray(rgb[y1:y2,x1:x2]);spans=physical_spans(gray)
                candidates=[]
                for i in range(len(spans)):
                    a,b=padded_row_span(spans,i,gray.shape[0]);pixels=np.ascontiguousarray(gray[a:b])
                    candidates.append({'gray':pixels,'row_index':i,'row_bounds':[a,b],'pixel_hash':hashlib.sha256(pixels.tobytes()).hexdigest()})
                # Independent header evidence is accepted only when its exact
                # saved raw pixels occur at the original frame and source y.
                frame_labels=[]
                for physical,text in labels.get((frame_id,slot.slot_id),[]):
                    a=physical['y1'];h,w=physical['pixel_shape'];actual=np.ascontiguousarray(gray[a:a+h,:w]);sha=hashlib.sha256(actual.tobytes()).hexdigest()
                    ok=sha==physical['row_pixel_sha256'];counts['source_pixel_pass' if ok else 'source_pixel_fail']+=1
                    record={'row_key':physical['row_key'],'frame':frame_id,'table':slot.slot_id,'expected_sha256':physical['row_pixel_sha256'],'actual_sha256':sha,'pass':ok,'raw_ocr_text':text,'row_bounds':[a,a+h]}
                    verified.append(record)
                    if not ok:continue
                    frame_labels.append(record);m=HEADER.fullmatch(text.strip())
                    if m:record['hand_header_id']=m.group(1)
                    if m and frame_id>args.first_frame:
                        hand=m.group(1)
                        if slot.slot_id not in lastheader or int(hand)>int(lastheader[slot.slot_id]):
                            contexts[slot.slot_id]=hand;lastheader[slot.slot_id]=hand;headerfirst[(slot.slot_id,hand)]=frame_id;headerindex[(slot.slot_id,hand)]=physical['source_span_index']
                            hand_state.confirm_hand(slot.slot_id,hand,frame_id,sha)
                hand=contexts.get(slot.slot_id)
                observed=frontiers[slot.slot_id].observe(frame_id,candidates);evaluations=[];eligible=[]
                for row in observed:
                    e=engine.classify(row['gray']);current,reason=bind_current(row,hand,headerfirst.get((slot.slot_id,hand)),headerindex.get((slot.slot_id,hand)))
                    e.update({k:v for k,v in row.items() if k!='gray'})
                    e.update(table=slot.slot_id,frame=frame_id,hand_id=hand,current_hand_bound=current,current_hand_reason=reason)
                    matching_header=next((r for r in frame_labels if r.get('hand_header_id') and r['expected_sha256']==row['pixel_hash']),None)
                    if matching_header:e['hand_header_id']=matching_header['hand_header_id']
                    if e['status']=='cards' and current:eligible.append(e)
                    counts['gate_classified' if e['gate_pass'] else 'gate_excluded']+=1;counts['unknown_rows']+=e['status']=='UNKNOWN';evaluations.append(e)
                if args.hand_state:
                    event,latch_reason,eligible_indices=hand_state.observe(slot.slot_id,frame_id,evaluations)
                    for e in evaluations:
                        e['current_hand_bound']=e['row_index'] in eligible_indices
                        e['current_hand_reason']='CO_VISIBLE_CURRENT_HAND_HEADER_ORDER' if e['current_hand_bound'] else latch_reason
                        e['physical_row_freshness_required']=False
                else:event,latch_reason=latch.observe(slot.slot_id,hand,frame_id,eligible)
                if event:triggers.append(event)
                counts['physical_rows']+=len(evaluations)
                for e in evaluations:
                    e.update(source_pts=frame.pts,time_base_num=frame.time_base.numerator,time_base_den=frame.time_base.denominator,source_chat_box=list(slot.chat_box),latch_reason=latch_reason)
                    line(rowsout,e)
                # Dealing OCR never enters engine; label association follows it.
                for record in frame_labels:
                    if 'Dealing' in record['raw_ocr_text']:
                        record['context_hand_id']=hand;record['header_first_frame']=headerfirst.get((slot.slot_id,hand));truth.append(record)
                line(frameout,{'table':slot.slot_id,'frame':frame_id,'hand_id':hand,'chat_pixel_sha256':hashlib.sha256(gray.tobytes()).hexdigest(),'row_count':len(candidates),'source_pts':frame.pts})
            if frame_id and frame_id%300==0:print(json.dumps({'progress_frame':frame_id,'elapsed_seconds':round(time.perf_counter()-start,3)}),flush=True)
    for name,data in [('source_pixel_controls',verified),('independent_dealing_truth',truth),('triggers',triggers)]:
        with (args.output/(name+'.jsonl')).open('w',encoding='utf-8') as f:
            for x in data:line(f,x)
    dump(args.output/'summary.json',{'counts':dict(counts),'trigger_count':len(triggers),'wall_seconds':time.perf_counter()-start,'metrics_status':'AWAITING_INDEPENDENT_TRIGGER_LABEL_REVIEW'})
    return json.loads((args.output/'summary.json').read_text(encoding='utf-8'))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--first-frame',type=int,default=0);p.add_argument('--last-frame',type=int,default=1800);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'runs/start_stack_two_stage_ncc_20261003/frozen_config.json')
    p.add_argument('--hand-state',action='store_true');p.add_argument('--video',type=Path,default=ROOT/'video/smoke_17.mp4');a=p.parse_args()
    print(json.dumps(run(a),ensure_ascii=False,indent=2))
