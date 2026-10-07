"""Post-run numerical/parity receipt; no recognition or production edits."""
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parser_core.seat_forced_binding import parse_forced_snapshot
OUT=ROOT/'runs/seat_forced_binding_20261003'
RUN=OUT/'acceptance_ON'
BASE=ROOT/'runs/start_stack_live_path_20261003/connection_acceptance_ON'
TIMING={'elapsed_ms','hybrid_elapsed_seconds','logic_ms','pixel_elapsed_seconds','vl_elapsed_seconds'}


def read(path):return json.loads(path.read_text('utf-8'))


def rows(path):
    if not path.exists():return
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k not in TIMING}
    if isinstance(value,list):return [clean(v) for v in value]
    return value


def main():
    started=time.perf_counter()
    known=read(ROOT/'runs/hand_state_marker_gate_20261003/acceptance_adjudication.json')
    expected={(h['table'],h['hand_id']):h for h in known['hands']}
    initial={(r['table_id'],r['hand_id'],r['trigger_source_frame'],r['pixel_hash']) for r in known['initial_partial_events']}
    triggers=list(rows(RUN/'dealing_cards_triggers.jsonl'));buckets=collections.defaultdict(list)
    for r in triggers:buckets[r['table_id'],r['hand_id']].append(r)
    unexpected=[r for r in triggers if (r['table_id'],r['hand_id']) not in expected]
    false=[r for r in unexpected if (r['table_id'],r['hand_id'],r['trigger_source_frame'],r['pixel_hash']) not in initial]
    marker_conflicts=[r for k,rr in buckets.items() if k in expected for r in rr
        if (r['trigger_source_frame'],r['pixel_hash'])!=(expected[k]['events'][0]['trigger_source_frame'],expected[k]['events'][0]['pixel_hash'])]
    markers=dict(HITS=sum(bool(buckets[k]) for k in expected),MISSES=sum(not buckets[k] for k in expected),
        FALSE_POSITIVES=len(false),DUPLICATES=sum(max(0,len(rr)-1) for rr in buckets.values()),
        SOURCE_BINDING_CONFLICTS=len(marker_conflicts),INITIAL_PARTIALS=len(unexpected)-len(false))
    parity=[]
    for name in ['physical_rows','raw_ocr_results','ordered_timeline','context_decisions','removed_duplicates','validated_hands','visual_evidence','hand_analysis']:
        old,new=BASE/(name+'.jsonl'),RUN/(name+'.jsonl')
        if not new.exists():parity.append(dict(stream=name,status='NOT_MEASURED_MISSING_OUTPUT'));continue
        aa,bb=list(rows(old)),list(rows(new));equal=len(aa)==len(bb) and all(clean(a)==clean(b) for a,b in zip(aa,bb))
        parity.append(dict(stream=name,exact_bytes=sha(old)==sha(new),semantic_equal_timing_only_exclusions=equal,
            baseline_rows=len(aa),new_rows=len(bb),old_sha256=sha(old),new_sha256=sha(new)))
    live=read(RUN/'start_stack_live_summary.json') if (RUN/'start_stack_live_summary.json').exists() else None
    binding=read(RUN/'seat_forced_summary.json') if (RUN/'seat_forced_summary.json').exists() else None
    inputs=list(rows(RUN/'seat_forced_inputs.jsonl'))
    input_kinds=collections.Counter(r['kind'] for r in inputs)
    snapshots=collections.defaultdict(list)
    for r in inputs:
        if r['kind']=='CHAT':
            response=r['ocr_response']
            snapshots[r['table_id'],r['session_id'],r['hand_id']].append(dict(index=r['index'],
                text=response.get('raw_text') or '',score=(response.get('score') or 0) if response.get('status')=='OK' else 0,
                source_ref=r['pixel_sha256'],source_frame=r['source_frame']))
    snapshot_audit=[]
    for key,rr in snapshots.items():
        snap,block=parse_forced_snapshot(rr,key[2],8,.95)
        snapshot_audit.append(dict(table=key[0],session=key[1],hand_id=key[2],post_grammar_count=len(snap['posts']) if snap else 0,
            amount_wraps=sum(len(p['source_refs'])==2 for p in snap['posts']) if snap else 0,
            state_events=len(snap['state_events']) if snap else 0,blocker=block))
    old_live=read(BASE/'start_stack_live_summary.json')
    new_numeric={(h['table_id'],h['hand_id'],seat):r for h in (live or {}).get('hands',[]) for seat,r in h['numeric_triples'].items()}
    old_numeric={(h['table_id'],h['hand_id'],seat):r for h in old_live['hands'] for seat,r in h['numeric_triples'].items()}
    new_deals={(h['table_id'],h['hand_id']):h['deal_complete_frame'] for h in (live or {}).get('hands',[]) if h['deal_complete_frame'] is not None}
    old_deals={(h['table_id'],h['hand_id']):h['deal_complete_frame'] for h in old_live['hands'] if h['deal_complete_frame'] is not None}
    final=read(RUN/'final_summary.json') if (RUN/'final_summary.json').exists() else None
    old_final=read(BASE/'final_summary.json')
    result=dict(scope='OFFICIAL_5401_ON_SEAT_FORCED_ACCEPTANCE_NO_PROMOTION',markers=markers,parity=parity,
        parity_PASS=all(r.get('semantic_equal_timing_only_exclusions') for r in parity),timing_exclusions=sorted(TIMING),
        NCC_receipts_exact=sha(BASE/'dealing_ncc_receipts.jsonl')==sha(RUN/'dealing_ncc_receipts.jsonl') if (RUN/'dealing_ncc_receipts.jsonl').exists() else None,
        DEAL_COMPLETE=len(new_deals),DEAL_COMPLETE_frames_exact_regression=new_deals==old_deals,
        stack_OCR_attempted=(live or {}).get('counts',{}).get('STACK_OCR_ATTEMPTED'),
        stack_OCR_completed=(live or {}).get('counts',{}).get('STACK_OCR_COMPLETED'),RAW_NUMERIC_TRIPLES=len(new_numeric),
        RAW_NUMERIC_TRIPLES_exact_source_regression=new_numeric==old_numeric,
        original_stack_readings_exact_bytes=sha(BASE/'start_stack_readings.jsonl')==sha(RUN/'start_stack_readings.jsonl') if (RUN/'start_stack_readings.jsonl').exists() else None,
        seat_forced_summary=binding,start_stack_live_summary=live,
        CONFIGURED_SEAT_GROUPS=304,NUMERIC_TRIPLE_GROUPS=len(new_numeric),
        NUMERIC_START_STACK_UNKNOWN=(binding or {}).get('start_stack_unknown'),
        seat_input_kind_counts=dict(input_kinds),forced_snapshot_audit=snapshot_audit,
        observed_complete_forced_posts=sum(r['post_grammar_count'] for r in snapshot_audit),
        resolved_activity_or_empty_events=sum(r.get('status')=='APPLIED' and r.get('kind')!='OCCUPIED' for r in (binding or {}).get('state_receipts',[])),
        unresolved_state_events=len((binding or {}).get('unresolved_state_events',[])),
        epoch_denominator_note='Epochs initialized only by positive TABLE masks; activity stays UNKNOWN without observed state proof. Unobserved36 hands are not extrapolated ACTIVE/EMPTY.',
        binding_result_blockers=dict(collections.Counter(r.get('blocker') for r in (binding or {}).get('results',[]))),
        forced_actor_seat_groups=len({(p['table'],p['session'],p['hand_id'],p['seat']) for p in (binding or {}).get('bound_posts',[])}),
        runtime=dict(prior_wall_seconds=old_final['wall_seconds'],new_wall_seconds=final['wall_seconds'] if final else None,
            prior_capture_seconds=old_final['first_stage_wall_seconds'],new_capture_seconds=final['first_stage_wall_seconds'] if final else None,
            prior_peak_pipeline_rss_mb=old_final['peak_pipeline_rss_mb'],new_peak_pipeline_rss_mb=final['peak_pipeline_rss_mb'] if final else None,
            new_rtf=final['rtf'] if final else None),
        guard=read(RUN/'production_guard.json') if (RUN/'production_guard.json').exists() else None,
        errors=read(RUN/'errors_retries.json') if (RUN/'errors_retries.json').exists() else None,
        png_files=len(list(RUN.rglob('*.png'))),default_flags=read(ROOT/'configs/dealing_start_stack.json'),
        analysis_seconds=time.perf_counter()-started)
    (OUT/'acceptance_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({k:result[k] for k in ['markers','parity_PASS','NCC_receipts_exact','DEAL_COMPLETE','DEAL_COMPLETE_frames_exact_regression','stack_OCR_attempted','stack_OCR_completed','RAW_NUMERIC_TRIPLES','RAW_NUMERIC_TRIPLES_exact_source_regression','forced_actor_seat_groups','binding_result_blockers','analysis_seconds']}))


if __name__=='__main__':main()
