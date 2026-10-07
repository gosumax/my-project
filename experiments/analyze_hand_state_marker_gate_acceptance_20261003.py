"""Post-run source joins, initial-partial accounting and frozen output parity."""
import json,re,hashlib,time,sys
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/hand_state_marker_gate_20261003';RUN=OUT/'acceptance_ON'
sys.path.insert(0,str(ROOT))
BASE=ROOT/'runs/header_admission_fix_20261003/acceptance_ON'
def read(p):return json.loads(p.read_text('utf-8'))
def rows(p):
    with p.open(encoding='utf-8') as f:
        for line in f:yield json.loads(line)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
HEADER=re.compile(r'^(?:Hand\s*)?#\s*(\d{8,})\s*$',re.I)
TIMING={'elapsed_ms','hybrid_elapsed_seconds','logic_ms','pixel_elapsed_seconds','vl_elapsed_seconds'}
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in TIMING}
    if isinstance(x,list):return [clean(v) for v in x]
    return x
def audit_initial_events(events,labels):
    import av,numpy as np
    from parser_core.ram_first_stage import png_compatible_gray
    from parser_core.layout import load_fixed_layout
    from parser_core.chat_rows import physical_spans,padded_row_span
    from parser_core.hand_header_admission import HeaderPrefixGate
    layout=load_fixed_layout(ROOT/'configs/nine_table_layout.json',ROOT/'configs/layout_847x404.json',ROOT/'configs/chat_log_bottom_rois.json')
    source_labels={(r['table_id'],r['row_pixel_sha256']):r for r in labels if r['frame_id']==0}
    wanted={0}|{e['trigger_source_frame'] for e in events};decoded={};prefix=HeaderPrefixGate(ROOT)
    def content(gray):
        mask=(gray[:,:-12]>70).astype(np.uint8);yy,xx=np.nonzero(mask)
        if not len(xx):return None
        body=np.ascontiguousarray(mask[yy.min():yy.max()+1,xx.min():xx.max()+1]);return (body.shape,hashlib.sha256(body.tobytes()).hexdigest())
    with av.open(str(ROOT/'video/smoke_17.mp4')) as video:
        for fid,fr in enumerate(video.decode(video=0)):
            if fid>max(wanted):break
            if fid not in wanted:continue
            rgb=fr.to_ndarray(format='rgb24')
            for slot in layout.slots:
                if not any(e['table_id']==slot.slot_id for e in events):continue
                x1,y1,x2,y2=slot.chat_box;gray=png_compatible_gray(rgb[y1:y2,x1:x2]);spans=physical_spans(gray);rr=[]
                for i in range(len(spans)):
                    a,b=padded_row_span(spans,i,len(gray));crop=np.ascontiguousarray(gray[a:b]);dig=hashlib.sha256(crop.tobytes()).hexdigest()
                    rr.append({'row_index':i,'row_bounds':[a,b],'pixel_sha256':dig,'content_key':content(crop),'prefix_candidate':prefix.classify(crop)['candidate']})
                decoded[fid,slot.slot_id]=rr
    out=[]
    for e in events:
        t=e['table_id'];f=e['trigger_source_frame'];refs={r['content_key']:source_labels[t,r['pixel_sha256']] for r in decoded[0,t] if (t,r['pixel_sha256']) in source_labels}
        rr=decoded[f,t];candidate=next((r for r in rr if r['pixel_sha256']==e['pixel_hash']),None)
        preceding=[r for r in rr if candidate and r['row_bounds'][1]<=candidate['row_bounds'][0] and r['prefix_candidate']]
        closest=max(preceding,key=lambda r:r['row_bounds'][0]) if preceding else None
        cref=refs.get(candidate['content_key']) if candidate else None;href=refs.get(closest['content_key']) if closest else None
        match=HEADER.fullmatch(href.get('raw_text','').strip()) if href else None
        proven=bool(candidate and cref and 'Dealing cards' in cref['raw_text'] and match and match.group(1)==e['hand_id'])
        manual=None
        manual_path=OUT/'initial_partial_t07_manual_source_review.json'
        if not proven and manual_path.exists():
            review=read(manual_path)
            if t==review['table_id'] and f==review['source_frame'] and e['hand_id']==review['hand_id'] and candidate and closest:
                if candidate['pixel_sha256']==review['candidate_pixel_sha256'] and closest['pixel_sha256']==review['header_pixel_sha256'] and sha(OUT/review['chat_image'])==review['chat_image_sha256']:
                    manual=review;proven=review['result']=='PROVEN_INITIAL_PARTIAL_BINDING'
        out.append({'event':e,'result':'PROVEN_INITIAL_PARTIAL_BINDING' if proven else 'UNRESOLVED_INITIAL_BINDING',
          'candidate_source_pixel_exact':bool(candidate),'candidate_independent_label':cref,'closest_preceding_header_independent_label':href,
          'physical_candidate':candidate,'physical_closest_header':closest,'manual_source_review':manual,
          'comparison':'EXACT_SOURCE_PIXELS + translation-invariant binary foreground identity to independently OCR-labelled source frame0 rows; explicit manual source review receipt if pixel identity differs'})
    return out
def main():
    started=time.perf_counter()
    groups=read(ROOT/'runs/start_stack_production_connection_20261003/historical_controls/known_new_hand_groups.json')['groups']
    expected={(g['table_id'],g['hand_id']):g for g in groups}
    audit=read(ROOT/'runs/hand_context_completeness_audit_20261003/all_38_hands.json')
    expected_cards={(g['table'],g['hand_id']):g['first_cards_receipt'] for g in audit['hands']}
    labels=list(rows(ROOT/'runs/start_stack_production_connection_20261003/historical_controls/independent_labels_0_5400.jsonl'))
    source_headers=[];initial_ids=set()
    for label in labels:
        match=HEADER.fullmatch(label.get('raw_text','').strip())
        if not match:continue
        key=(label['table_id'],match.group(1))
        if label['frame_id']==0:initial_ids.add(key)
        if key in expected and label['frame_id']==expected[key]['header_frame']:source_headers.append(dict(label,hand_id=match.group(1)))
    header_jobs=list(rows(RUN/'hand_header_ocr.jsonl'))
    admitted={(r['table_id'],r['source_frame'],r['pixel_sha256']):r for r in header_jobs if r['status']=='ADMITTED'}
    completed={(r['table_id'],r['source_frame'],r['pixel_sha256']):r for r in header_jobs if 'ocr_response' in r}
    chain=[]
    for source in source_headers:
        key=(source['table_id'],source['frame_id'],source['row_pixel_sha256']);a=admitted.get(key);c=completed.get(key)
        chain.append({'table':key[0],'hand_id':source['hand_id'],'source_frame':key[1],'pixel_sha256':key[2],
          'ADMITTED':a is not None,'OCR_ATTEMPTED':c is not None,'HAND_HEADER_CONFIRMED':bool(c and c['status']=='HAND_HEADER_CONFIRMED'),
          'current_hand_id_correct':bool(c and c.get('current_hand_id')==source['hand_id'])})
    triggers=list(rows(RUN/'dealing_cards_triggers.jsonl'));buckets=defaultdict(list)
    for event in triggers:buckets[event['table_id'],event['hand_id']].append(event)
    hands=[];hits=0;binding_conflicts=0
    for key,g in expected.items():
        events=buckets[key];hits+=bool(events);expected_c=expected_cards[key]
        bound=bool(events and events[0]['trigger_source_frame']==expected_c['source_frame'] and events[0]['pixel_hash']==expected_c['pixel_hash'])
        if events and not bound:binding_conflicts+=1
        hands.append({'table':key[0],'hand_id':key[1],'result':'HIT' if events else 'MISS','expected_first_cards_frame':expected_c['source_frame'],
          'source_binding_exact':bound,'events':events})
    initial=[e for e in triggers if (e['table_id'],e['hand_id']) not in expected and (e['table_id'],e['hand_id']) in initial_ids]
    initial_audit=audit_initial_events(initial,labels)
    false_positive=[e for e in triggers if (e['table_id'],e['hand_id']) not in expected and (e['table_id'],e['hand_id']) not in initial_ids]
    duplicate=sum(max(0,len(ev)-1) for ev in buckets.values())
    t09=next(h for h in hands if h['table']==9 and h['hand_id']=='2876720526')
    for e in t09['events']:
        confirmed=e.get('confirmation_monotonic_ns');observed=e.get('candidate_observation_monotonic_ns')
        e['confirmed_before_or_at_candidate_observation']=confirmed<=observed if confirmed is not None and observed is not None else 'NOT_RECORDED'
        e['confirmation_vs_observation_ns']=confirmed-observed if confirmed is not None and observed is not None else 'NOT_RECORDED'
    parity=[]
    streams=['physical_rows','raw_ocr_results','ordered_timeline','context_decisions','removed_duplicates','validated_hands','visual_evidence','hand_analysis']
    for name in streams:
        old=BASE/(name+'.jsonl');new=RUN/(name+'.jsonl');exact=sha(old)==sha(new)
        semantic=exact;diff=None;nold=nnew=0
        if not exact or name in {'visual_evidence','hand_analysis'}:
            aa=list(rows(old));bb=list(rows(new));nold=len(aa);nnew=len(bb)
            semantic=len(aa)==len(bb) and all(clean(a)==clean(b) for a,b in zip(aa,bb))
            if not semantic:diff=next((i for i,(a,b) in enumerate(zip(aa,bb)) if clean(a)!=clean(b)),min(len(aa),len(bb)))
        else:nold=nnew=sum(1 for _ in old.open(encoding='utf-8'))
        parity.append({'stream':name,'exact_bytes':exact,'semantic_equal_timing_only_exclusions':semantic,'baseline_rows':nold,'new_rows':nnew,'first_difference_index':diff,'old_sha256':sha(old),'new_sha256':sha(new)})
    summary=read(RUN/'dealing_start_stack_summary.json');final=read(RUN/'final_summary.json')
    result={'scope':'OFFICIAL_LIVE_ON_ACCEPTANCE_NO_PROMOTION','HITS':hits,'MISSES':38-hits,'FALSE_POSITIVES':len(false_positive),
      'DUPLICATE_TRIGGERS':duplicate,'UNKNOWN':38-hits,'INITIAL_PARTIAL_TRIGGERS':len(initial),'source_binding_conflicts':binding_conflicts,
      'hands':hands,'initial_partial_events':initial,'initial_partial_binding_audit':initial_audit,
      'UNRESOLVED_INITIAL_BINDINGS':sum(r['result']!='PROVEN_INITIAL_PARTIAL_BINDING' for r in initial_audit),
      'false_positive_events':false_positive,'T09':t09,
      'header_chain_counts':{'SOURCE_HEADERS':len(chain),'ADMITTED':sum(r['ADMITTED'] for r in chain),'OCR_ATTEMPTED':sum(r['OCR_ATTEMPTED'] for r in chain),
        'HAND_HEADER_CONFIRMED':sum(r['HAND_HEADER_CONFIRMED'] for r in chain),'CURRENT_HAND_ID_CORRECT':sum(r['current_hand_id_correct'] for r in chain)},
      'header_chain':chain,'parity':parity,'parity_PASS':all(r['semantic_equal_timing_only_exclusions'] for r in parity),
      'timing_exclusions':sorted(TIMING),'summary':summary,'final_summary':final,
      'NCC_receipts_exact_bytes':sha(BASE/'dealing_ncc_receipts.jsonl')==sha(RUN/'dealing_ncc_receipts.jsonl'),
      'default_flags':read(ROOT/'configs/dealing_start_stack.json'),
      'png_files':len(list(RUN.rglob('*.png'))),'analysis_seconds':time.perf_counter()-started}
    (OUT/'acceptance_adjudication.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({k:result[k] for k in ['HITS','MISSES','FALSE_POSITIVES','DUPLICATE_TRIGGERS','UNKNOWN','INITIAL_PARTIAL_TRIGGERS','source_binding_conflicts','header_chain_counts','parity_PASS','NCC_receipts_exact_bytes','analysis_seconds']},ensure_ascii=False))
if __name__=='__main__':main()
