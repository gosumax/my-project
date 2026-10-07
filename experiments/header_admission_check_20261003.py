"""Bounded source-only gate replay; labels enter only posthoc adjudication."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import av
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parser_core.hand_header_admission import HandHeaderAdmission,HeaderPrefixGate
from parser_core.dealing_start_stack import DealingStartStackSidecar,HEADER
from parser_core.layout import load_fixed_layout
from parser_core.ram_first_stage import png_compatible_gray
from parser_core.parallel_ocr import ParallelRowOCR,row_key
from parser_core.tiny_chat_client import TinyChatClient

def analyze_on(output):
    timing={'elapsed_ms','hybrid_elapsed_seconds','logic_ms','pixel_elapsed_seconds','vl_elapsed_seconds'}
    def clean(obj):
        if isinstance(obj,dict):return {k:clean(v) for k,v in obj.items() if k not in timing}
        if isinstance(obj,list):return [clean(v) for v in obj]
        return obj
    def jsonl(path):return [json.loads(line) for line in path.read_text('utf-8').splitlines() if line.strip()]
    lane_log=jsonl(output/'hand_header_ocr.jsonl')
    admitted={(r['table_id'],r['source_frame'],r['source_span_index'],r['pixel_sha256']):r for r in lane_log if r['status']=='ADMITTED'}
    callbacks={(r['table_id'],r['source_frame'],r['source_span_index'],r['pixel_sha256']):r for r in lane_log if 'ocr_response' in r}
    source_labels=json.loads((output.parent/'source_replay_v1/source_76_adjudication.json').read_text('utf-8'))
    joined=[]
    for label in source_labels:
        key=(label['table_id'],label['source_frame'],label['source_span_index'],label['pixel_sha256']);response=callbacks.get(key,{})
        match=HEADER.fullmatch((response.get('ocr_response',{}).get('raw_text') or '').strip())
        joined.append(dict(label_table=label['table_id'],label_hand=label['hand_id'],source_frame=label['source_frame'],source_pts=admitted.get(key,{}).get('source_pts'),pixel_sha256=label['pixel_sha256'],admitted=key in admitted,ocr_attempted=key in callbacks,ocr_correct=bool(match and match.group(1)==label['hand_id']),provider_status=response.get('status'),current_hand_correct=response.get('current_hand_id')==label['hand_id'],raw_response=response.get('ocr_response')))
    unknown_jobs=[r for r in callbacks.values() if not (r['ocr_response'].get('status')=='OK' and HEADER.fullmatch((r['ocr_response'].get('raw_text') or '').strip()))]
    chain={'SOURCE':76,'ADMITTED':sum(r['admitted'] for r in joined),'OCR_ATTEMPTED':sum(r['ocr_attempted'] for r in joined),'OCR_CORRECT':sum(r['ocr_correct'] for r in joined),'HAND_HEADER_CONFIRMED':sum(r['provider_status']=='HAND_HEADER_CONFIRMED' for r in joined),'current_hand_id':sum(r['provider_status']=='HAND_HEADER_CONFIRMED' and r['current_hand_correct'] for r in joined),'source76_provider_status_counts':{status:sum(r['provider_status']==status for r in joined) for status in {r['provider_status'] for r in joined}},'all_ordinary_or_unknown_jobs':unknown_jobs,'all_lane_admitted':len(admitted),'all_lane_completed':len(callbacks),'rows':joined}
    (output.parent/'ON_header_chain_adjudication.json').write_text(json.dumps(chain,ensure_ascii=False,indent=2),'utf-8')
    groups=json.loads((ROOT/'runs/start_stack_production_connection_20261003/historical_controls/known_new_hand_groups.json').read_text('utf-8'))['groups']
    triggers=jsonl(output/'dealing_cards_triggers.jsonl')
    headers=jsonl(output/'hand_state_receipts.jsonl')
    by_table={}
    for g in groups:by_table.setdefault(g['table_id'],[]).append(g)
    for items in by_table.values():items.sort(key=lambda g:g['header_frame'])
    adjud=[]
    for group in groups:
        end=min([g['header_frame'] for g in by_table[group['table_id']] if g['header_frame']>group['header_frame']]+[5401])
        bound=[h for h in headers if h.get('table_id')==group['table_id'] and h.get('hand_id')==group['hand_id'] and h['status']=='HAND_HEADER_CONFIRMED']
        events=[t for t in triggers if t['table_id']==group['table_id'] and t['hand_id']==group['hand_id']]
        valid=[t for t in events if group['header_frame']<=t['trigger_source_frame']<end]
        adjud.append(dict(group,hand_state_confirmed=bool(bound),first_header_receipt=min(bound,key=lambda h:h['source_frame']) if bound else None,events=events,valid_events=valid,first_failed_stage=None if valid else 'HAND_CONTEXT_COMPLETENESS_OR_CANDIDATE_BINDING_UNKNOWN' if bound else 'HAND_HEADER_CONFIRMED',end_frame=end))
    known={(g['table_id'],g['hand_id']) for g in groups}
    unknown=[];false=[]
    for event in triggers:
        table=event['table_id'];frame=event['trigger_source_frame']
        expected=[g for g in by_table.get(table,[]) if g['header_frame']<=frame]
        if not expected:unknown.append(event)
        elif event['hand_id']!=expected[-1]['hand_id']:false.append(event)
    duplicates=len(triggers)-len({(t['table_id'],t['session_id'],t['hand_id']) for t in triggers})
    ncc_t09=[]
    with (output/'dealing_ncc_receipts.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            if r['table_id']==9 and r['source_frame']==4360:ncc_t09.append(r)
    special=next(g for g in adjud if g['table_id']==9 and g['hand_id']=='2876720526')
    result={'HITS':sum(bool(g['valid_events']) for g in adjud),'MISSES':sum(not g['valid_events'] for g in adjud),'FALSE_POSITIVES':len(false),'DUPLICATE_TRIGGERS':duplicates,'UNKNOWN_issued_initial_partial_events':len(unknown),'UNKNOWN_missed_groups':sum(not g['valid_events'] for g in adjud),'issued_triggers':len(triggers),'known_hands':len(groups),'hand_state_confirmed_groups':sum(g['hand_state_confirmed'] for g in adjud),'false_positive_events':false,'unknown_initial_events':unknown,'groups':adjud,'T09_2876720526':dict(special,ncc_frame4360=ncc_t09)}
    (output.parent/'ON_trigger_adjudication.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    comparisons=[]
    for baseline_name in ['acceptance_OFF2','acceptance_ON2']:
        baseline=ROOT/'runs/start_stack_production_connection_20261003'/baseline_name
        checks=[]
        for name in ['physical_rows.jsonl','raw_ocr_results.jsonl','ordered_timeline.jsonl','removed_duplicates.jsonl','visual_evidence.jsonl','hand_analysis.jsonl','context_decisions.jsonl','validated_hands.jsonl']:
            old=baseline/name;new=output/name;a=jsonl(old);b=jsonl(new)
            checks.append(dict(file=name,baseline_rows=len(a),new_rows=len(b),baseline_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),new_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),exact_bytes=old.read_bytes()==new.read_bytes(),semantic_excluding_only_named_timing_fields=clean(a)==clean(b)))
        comparisons.append(dict(baseline=baseline_name,checks=checks,verdict='PASS' if all(c['semantic_excluding_only_named_timing_fields'] for c in checks) else 'FAIL'))
    (output.parent/'ON_authoritative_parity.json').write_text(json.dumps(dict(excluded_timing_fields=sorted(timing),comparisons=comparisons),indent=2),'utf-8')
    frozen=json.loads((output.parent/'candidate/snapshot_hashes.json').read_text('utf-8'))
    changes=[f for f,h in frozen.items() if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h]
    (output.parent/'candidate/acceptance_after_hashcheck.json').write_text(json.dumps(dict(files=len(frozen),changed_files=changes),indent=2),'utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['groups','T09_2876720526','false_positive_events','unknown_initial_events']},ensure_ascii=False))
    print(json.dumps({'PARITY':{r['baseline']:r['verdict'] for r in comparisons},'CANDIDATE_CHANGED':changes}))
    print(json.dumps({'ON_HEADER_CHAIN':{k:v for k,v in chain.items() if k not in ['rows','all_ordinary_or_unknown_jobs']},'ORDINARY_UNKNOWN_JOBS':len(unknown_jobs)}))
    return 0 if not changes and all(r['verdict']=='PASS' for r in comparisons) else 1

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--analyze-on',action='store_true')
    args=parser.parse_args();output=ROOT/args.output
    if args.analyze_on:return analyze_on(output)
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    control=json.loads((ROOT/'runs/start_stack_production_connection_20261003/historical_controls/header_admission_OFF2.json').read_text('utf-8'))
    labels=[dict(row,hand_id=g['hand_id']) for g in control['groups'] for row in g['header_rows']]
    label_positions={}
    for row in labels:label_positions.setdefault((row['frame_id'],row['table_id']),[]).append(row)
    flags=json.loads((ROOT/'configs/dealing_start_stack.json').read_text('utf-8'))
    provider=DealingStartStackSidecar(ROOT,output,flags)
    config=json.loads((ROOT/'configs/ocr_tiny_local.json').read_text('utf-8'))
    def factory(worker):
        return TinyChatClient(Path(config['python']),Path(config['model_dir']),config['model_name'],config['weights_sha256'],config['device'],output/f'header_worker_{worker}.stderr.log')
    ocr=ParallelRowOCR(factory,workers=2,batch_size=8,result_observer=lambda row,response:lane.ocr_complete(row,response))
    lane=HandHeaderAdmission(ROOT,output,provider,ocr)
    layout=load_fixed_layout(ROOT/'configs/nine_table_layout.json',ROOT/'configs/layout_847x404.json',ROOT/'configs/chat_log_bottom_rois.json')
    frames=0
    # Labels do not choose frames, tables, spans, gate calls, jobs or OCR inputs.
    with av.open(str(ROOT/'video/smoke_17.mp4')) as container:
        stream=container.streams.video[0];stream.thread_type='FRAME';stream.codec_context.thread_count=3
        for frame_index,frame in enumerate(container.decode(stream)):
            if frame_index>5400:break
            rgb=frame.to_ndarray(format='rgb24');base=frame.time_base or stream.time_base
            for slot in layout.slots:
                x1,y1,x2,y2=slot.chat_box;gray=png_compatible_gray(rgb[y1:y2,x1:x2])
                session=f'HEADER_REPLAY_T{slot.slot_id:02d}'
                lane.observe_frame(slot.slot_id,session,frame_index,frame.pts,[base.numerator,base.denominator],None,gray)
                for label in label_positions.get((frame_index,slot.slot_id),[]):
                    pixels=np.ascontiguousarray(gray[label['row_y1']:label['row_y1']+label['pixel_shape'][0]])
                    label['verified_source_sha256']=hashlib.sha256(pixels.tobytes()).hexdigest()
                    label['content_key']=HeaderPrefixGate.content_key(pixels)
            frames+=1
            if frames%900==0:print(json.dumps({'progress_frames':frames,'admitted':lane.counts['ADMITTED'],'seconds':time.perf_counter()-started}),flush=True)
    capture_seconds=time.perf_counter()-started
    ocr.finish();lane.finish();provider.finish()
    admissions={}
    for row in lane.rows:
        admissions.setdefault((row.table_id,HeaderPrefixGate.content_key(row.pixels)),row)
    adjudication=[]
    for label in labels:
        row=admissions.get((label['table_id'],label.get('content_key')))
        response=ocr.results.get(row_key(row),{}) if row is not None else {}
        match=HEADER.fullmatch((response.get('raw_text') or '').strip())
        confirmed=response.get('status')=='OK' and match is not None and match.group(1)==label['hand_id']
        history=provider.histories[(label['table_id'],f"HEADER_REPLAY_T{label['table_id']:02d}")]
        bound=any(h['hand_id']==label['hand_id'] for h in history)
        adjudication.append(dict(table_id=label['table_id'],hand_id=label['hand_id'],label_frame=label['frame_id'],label_sha256=label['row_pixel_sha256'],source_hash_verified=label.get('verified_source_sha256')==label['row_pixel_sha256'],admitted=row is not None,source_frame=row.frame_id if row is not None else None,source_pts=row.source_pts if row is not None else None,source_span_index=row.source_span_index if row is not None else None,pixel_sha256=row.pixel_sha256 if row is not None else None,attempted=row is not None and row_key(row) in ocr.results,ocr_response=response,hand_header_confirmed=confirmed,current_hand_id_bound=confirmed and bound))
    raw=[dict(table_id=row.table_id,session_id=row.session_id,source_frame=row.frame_id,source_pts=row.source_pts,time_base=[row.time_base_num,row.time_base_den],source_span_index=row.source_span_index,pixel_sha256=row.pixel_sha256,response=ocr.results.get(row_key(row))) for row in lane.rows]
    (output/'raw_header_responses.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2),'utf-8')
    (output/'source_76_adjudication.json').write_text(json.dumps(adjudication,ensure_ascii=False,indent=2),'utf-8')
    unknown=[r for r in raw if not (r['response'].get('status')=='OK' and HEADER.fullmatch((r['response'].get('raw_text') or '').strip()))]
    summary={'command':' '.join(sys.argv),'frames':frames,'source_headers':len(labels),'source_hash_verified':sum(a['source_hash_verified'] for a in adjudication),'ADMITTED':sum(a['admitted'] for a in adjudication),'OCR_ATTEMPTED':sum(a['attempted'] for a in adjudication),'HAND_HEADER_CONFIRMED':sum(a['hand_header_confirmed'] for a in adjudication),'current_hand_id':sum(a['current_hand_id_bound'] for a in adjudication),'known_hands':len(control['groups']),'known_hands_bound':len({(a['table_id'],a['hand_id']) for a in adjudication if a['current_hand_id_bound']}),'all_lane_counts':dict(lane.counts),'provider_counts':dict(provider.counts),'all_jobs':len(raw),'ordinary_chat_or_unknown_attempts':unknown,'ocr_errors':ocr.errors,'capture_seconds':capture_seconds,'total_seconds':time.perf_counter()-started,'metrics_not_collected':['DEALING_CARDS','DEAL_COMPLETE','START_STACK','official_ON_acceptance']}
    summary['verdict']='PASS' if all(summary[k]==76 for k in ['source_headers','source_hash_verified','ADMITTED','OCR_ATTEMPTED','HAND_HEADER_CONFIRMED','current_hand_id']) and not unknown and not ocr.errors else 'FAIL'
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 0 if summary['verdict']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
