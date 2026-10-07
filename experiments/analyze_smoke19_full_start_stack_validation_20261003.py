"""Read-only full-run source-contract validation; no independent ground truth."""
import csv,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from parser_core.start_stack_live import first_visible_stack_triple
from parser_core.poker_math_engine import calculate_start_stack
from parser_core.seat_forced_binding import parse_forced_snapshot,POST
from parser_core.seat_epoch_state import SeatEpochMachine
OUT=ROOT/'runs/smoke19_full_start_stack_validation_20261003';RUN=OUT/'acceptance_ON';VAL=OUT/'validation'
def read(p):return json.loads(p.read_text('utf-8'))
def rows(p):
    with p.open(encoding='utf-8') as f:
        for number,line in enumerate(f,1):
            if line.strip():yield number,json.loads(line)
def key(r):return r['table_id'],r['session_id'],r['hand_id']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvwrite(name,data):
    if not data:return
    with (VAL/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def main():
    VAL.mkdir(exist_ok=True);exit=read(OUT/'exit_receipt.json');final=read(RUN/'final_summary.json');live=read(RUN/'start_stack_live_summary.json');seat=read(RUN/'seat_forced_summary.json');facts=[r for _,r in rows(RUN/'start_stack_facts.jsonl')];math=[r for _,r in rows(RUN/'start_stack_math.jsonl')]
    frozen=read(OUT/'acceptance_candidate/frozen_hashes.json');old=read(ROOT/'runs/historical_forced_binding_integration_20261003/acceptance_candidate/frozen_hashes.json');flags=read(ROOT/'configs/dealing_start_stack.json')
    triggers=[r for _,r in rows(RUN/'dealing_cards_triggers.jsonl')];deals=[r for _,r in rows(RUN/'deal_complete_receipts.jsonl') if r.get('evidence_status')=='PROVEN'];headers=[r for _,r in rows(RUN/'hand_state_receipts.jsonl') if r.get('status')=='HAND_HEADER_CONFIRMED'];trigger_map={key(r):r for r in triggers};header_map=defaultdict(list)
    for r in headers:header_map[key(r)].append(r)
    table_ncc={};stack_sources=defaultdict(dict)
    for line,r in rows(RUN/'start_stack_live_receipts.jsonl'):
        if r.get('stage')=='TABLE_CARD_NCC':table_ncc[(*key(r),r['source_frame'])]=r
    for line,r in rows(RUN/'start_stack_readings.jsonl'):
        stack_sources[(*key(r),r['seat'])][r['source_frame']]=r
    hand_checks=[]
    for d in deals:
        k=key(d);frame=d['source_frame'];ncc=[table_ncc.get((*k,f)) for f in range(frame-2,frame+1)];t=trigger_map.get(k);h=header_map[k]
        hand_checks.append(dict(table=k[0],session=k[1],hand=k[2],deal_frame=frame,card_frames=json.dumps(d['card_stable_frames']),header_confirmed=any(r['source_frame']<=frame for r in h),trigger_present=t is not None,three_table_ncc_sources=all(r and r['status']=='CANDIDATE' and all(v is not None and v>=.95 for v in r['scores'].values()) for r in ncc),occupied_seats=len(d['occupied_seats']),table_pixel_sha256=d['table_pixel_sha256']))
    seat_checks=[];attempt_mismatches=[];substitutions=[]
    for hand in live['hands']:
        k=key(hand);deal=hand['deal_complete_frame']
        if deal is None:continue
        for s,a in hand['canonical_attempts'].items():
            rr=stack_sources[(*k,s)];replay=first_visible_stack_triple(rr,.95,final=True,start_frame=deal+1)
            fields=['first_numeric_visible_frame','frames','ocr_values','displayed_stack','status','blocker'];equal=replay is not None and all(a.get(f)==replay.get(f) for f in fields)
            if not equal:attempt_mismatches.append(dict(table=k[0],hand=k[2],seat=s,saved={f:a.get(f) for f in fields},replay={f:replay.get(f) for f in fields} if replay else None))
            first=a.get('first_numeric_visible_frame');end=max(a['frames']) if a['frames'] else deal
            gaps=[f for f in range(deal+1,end+1) if f not in rr]
            seat_checks.append(dict(table=k[0],hand=k[2],seat=s,deal=deal,first_numeric=first,canonical_frames=json.dumps(a['frames']),values=json.dumps(a['ocr_values']),displayed_stack=a['displayed_stack'],status=a['status'],blocker=a['blocker'],source_replay_equal=equal,gaps=json.dumps(gaps),source_readings=len(rr)))
    snapshot_rows=defaultdict(list);provenance={}
    for line,r in rows(RUN/'seat_forced_inputs.jsonl'):
        provenance[r['source_ref']]=dict(file='seat_forced_inputs.jsonl',line=line,frame=r['source_frame'],pts=r['source_pts'],time_base=r['time_base'],kind=r['kind'],seat=r.get('seat'),roi=r.get('roi'),pixel_sha256=r['pixel_sha256'],ocr_response=r['ocr_response'])
        if r['kind']=='CHAT':snapshot_rows[key(r)].append(dict(index=r['index'],text=(r['ocr_response'].get('raw_text') or '').strip(),score=r['ocr_response'].get('score') or 0,source_ref=r['source_ref'],source_frame=r['source_frame']))
    machine=SeatEpochMachine()
    for t in triggers:machine.hand(*key(t),t['trigger_source_frame'])
    for e in seat['state_receipts']:machine.observe(e['table'],e['session'],e['seat'],e['kind'],e['frame'],actor=e.get('actor'),hand_id=e.get('hand_id'),evidence_status=e['evidence_status'],source_ref=e.get('source_ref'),source_index=e['source_index'])
    machine.build();snapshots={};post_checks=[];snapshot_checks=[]
    for d in deals:
        k=key(d);snap,block=parse_forced_snapshot(snapshot_rows[k],k[2],len(d['occupied_seats']))
        if snap:snapshots[k]=snap
        snapshot_checks.append(dict(table=k[0],session=k[1],hand=k[2],deal_frame=d['source_frame'],physical_rows=len(snapshot_rows[k]),raw_post_grammar_candidates=sum(bool(POST.fullmatch(r['text'])) for r in snapshot_rows[k]),raw_posts_word_candidates=sum('posts' in r['text'].lower() for r in snapshot_rows[k]),confidence_qualified_post_rows=sum(bool(POST.fullmatch(r['text'])) and r['score']>=.95 for r in snapshot_rows[k]),complete_snapshot=snap is not None,qualified_complete_snapshot_posts=len(snap['posts']) if snap else 0,blocker=block))
    receipts=seat.get('historical_binding_receipts',[])
    for p in seat['bound_posts']:
        k=(p['table'],p['session'],p['hand_id']);r=next((r for r in receipts if (r['table'],r['session'],r['hand_id'],r['actor'])==(*k,p['actor'])),None);anchor=r['binding_anchor_frame'] if r else None
        evidence=[]
        for label in (r['label_evidence'] if r else []):evidence.extend(label['frames'])
        epoch_ok=bool(r) and all(machine.epoch_at(*k[:2],p['seat'],f)==p['epoch'] for f in [p['source_frame'],anchor]+evidence)
        post_checks.append(dict(table=k[0],hand=k[2],seat=p['seat'],actor=p['actor'],role=p['role'],amount=p['amount'],forced_frame=p['source_frame'],anchor=anchor,label_confirmation=r['label_confirmation_frame'] if r else None,epoch=p['epoch'],same_epoch=epoch_ok,participant=machine.membership(*k[:2],p['seat'],k[2],p['epoch'])['participant'],provenance_complete=all(ref in provenance for ref in p['source_refs']),late_conflicts=json.dumps(r['later_literal_conflicts'] if r else [])))
    unbound=[]
    for k,snap in snapshots.items():
        for p in snap['posts']:
            if not any((b['table'],b['session'],b['hand_id'],b['actor'],b['role'],b['amount'])==(*k,p['actor'],p['role'],p['amount']) for b in seat['bound_posts']):unbound.append(dict(table=k[0],hand=k[2],actor=p['actor'],role=p['role'],amount=p['amount'],source_refs=json.dumps(p['source_refs'])))
    math_equal=len(facts)==len(math) and all(calculate_start_stack(f)==m for f,m in zip(facts,math));math_checks=[dict(table=m['table'],hand=m['hand_id'],seat=m['seat'],displayed=m.get('displayed_stack'),corrections=json.dumps(m.get('forced_bets')),START_STACK=m.get('START_STACK'),status=m['status'],blocker=m.get('blocker')) for m in math]
    duplicates=len(triggers)-len({key(r) for r in triggers});guards=dict(exit0=exit['exit_code']==0,fullframes=final['frames']==exit['input']['actual_decoded_frames']==10100,frozen_current=all(sha(ROOT/n)==h for n,h in frozen.items()),Smoke17_frozen_same=frozen==old,flags_false=not flags['enable_dealing_cards_hand_state_trigger'] and not flags['enable_start_stack_capture'],errors0=not final['errors'] and final['retry_count']==0,PNG0=final['png_files']==0 and final['png_bytes']==0,hand_deal_contract=all(all(r[k] for k in ['header_confirmed','trigger_present','three_table_ncc_sources']) for r in hand_checks),first_attempt_replay=not attempt_mismatches,historical_epoch_contract=all(r['same_epoch'] and r['participant'] and r['provenance_complete'] for r in post_checks),math_replay_exact=math_equal,no_duplicate_trigger=duplicates==0,vision_no_arithmetic=all('START_STACK' not in f and 'start_stack' not in f for f in facts))
    report=dict(proposed_verdict='FAIL' if not all(guards.values()) else 'PARTIAL',reason='Independent hand/card/stack ground truth and precision/recall NOT_MEASURED; explicit UNKNOWN retained',guards=guards,counts=dict(triggers=len(triggers),duplicate_triggers=duplicates,deals=len(deals),occupied_seat_hand_slots=sum(len(d['occupied_seats']) for d in deals),canonical_attempts=len(seat_checks),numeric_triples=seat['numeric_triples'],facts=len(facts),math_statuses=dict(Counter(r['status'] for r in math)),math_blockers=dict(Counter(str(r.get('blocker')) for r in math)),bound_posts=len(post_checks),unbound_parsed_posts=len(unbound),historical_receipts=len(receipts),late_conflict_receipts=sum(bool(r['later_literal_conflicts']) for r in receipts),unresolved_state_events=len(seat['unresolved_state_events']),forced_snapshot_blockers=seat['forced_snapshot_blockers']),independent_metrics=dict(marker_misses='NOT_MEASURED',marker_false_positives='NOT_MEASURED',stack_accuracy='NOT_MEASURED',certified_hh=0),runtime=exit['pipeline_metrics'],wrapper_seconds=exit['wrapper_wall_seconds'],regression_classes=dict(first_source_gap_attempts=[r for r in seat_checks if r['blocker']=='FIRST_NUMERIC_SOURCE_COVERAGE_UNKNOWN'],numeric_instability=[r for r in seat_checks if r['blocker']=='FIRST_NUMERIC_TRIPLE_NOT_STABLE'],unresolved_state=seat['unresolved_state_events'],historical_blockers=seat.get('historical_binding_blockers',[])),attempt_mismatches=attempt_mismatches)
    report['counts'].update(distinct_confirmed_header_hands=len(header_map),distinct_trigger_hands=len(trigger_map),distinct_deal_hands=len({key(r) for r in deals}),complete_forced_snapshots=len(snapshots),qualified_complete_snapshot_posts=sum(r['qualified_complete_snapshot_posts'] for r in snapshot_checks),raw_posts_word_candidates=sum(r['raw_posts_word_candidates'] for r in snapshot_checks),raw_post_grammar_candidates=sum(r['raw_post_grammar_candidates'] for r in snapshot_checks),qualified_post_rows=sum(r['confidence_qualified_post_rows'] for r in snapshot_checks),snapshot_physical_rows=sum(r['physical_rows'] for r in snapshot_checks))
    report['post_count_scope']='Only saved physical forced snapshot OCR lanes at 18 positive TABLE deals. Raw rows include incomplete/low-confidence/unparseable candidates. Unbound parsed posts means 36 posts from 15 complete snapshots; this is not a video-wide missing-post metric.'
    report['header_hands_without_positive_TABLE']=[dict(table=k[0],session=k[1],hand=k[2],header_frames=[r['source_frame'] for r in v],blocker='TABLE_DEAL_COMPLETE_NOT_PROVEN') for k,v in header_map.items() if k not in {key(d) for d in deals}]
    snapshot_lookup={(r['table'],r['session'],r['hand']):r for r in snapshot_checks}
    report['unknown_fact_receipts']=[dict(table=m['table'],session=m['session'],hand=m['hand_id'],seat=m['seat'],blocker=m['blocker'],displayed_stack=m.get('displayed_stack'),frames=m.get('frames'),first_numeric_visible_frame=m.get('first_numeric_visible_frame'),underlying_snapshot=snapshot_lookup.get((m['table'],m['session'],m['hand_id']))) for m in math if m['status']=='UNKNOWN']
    csvwrite('forced_snapshot_coverage.csv',snapshot_checks)
    for name,data in [('hands_deals.csv',hand_checks),('seat_source_windows.csv',seat_checks),('forced_bound.csv',post_checks),('forced_unbound.csv',unbound),('math_breakdown.csv',math_checks)]:csvwrite(name,data)
    (VAL/'source_provenance.json').write_text(json.dumps(provenance,indent=2),'utf-8');(VAL/'validation_summary.json').write_text(json.dumps(report,indent=2),'utf-8');print(json.dumps({k:report[k] for k in ['proposed_verdict','guards','counts','independent_metrics','runtime','wrapper_seconds']},ensure_ascii=False))
if __name__=='__main__':main()
