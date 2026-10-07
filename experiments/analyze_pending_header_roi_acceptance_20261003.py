"""Compare actual prior live 13/8 groups and protected outputs; read-only."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/deal_complete_fix_chat_review_20261003'
RUN = OUT / 'acceptance_ON'
BASE = ROOT / 'runs/seat_forced_binding_20261003/acceptance_ON'
TIMING = {'elapsed_ms','hybrid_elapsed_seconds','logic_ms','pixel_elapsed_seconds','vl_elapsed_seconds'}


def read(path):
    return json.loads(path.read_text('utf-8'))


def rows(path):
    return [json.loads(l) for l in path.read_text('utf-8').splitlines() if l.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k:clean(v) for k,v in value.items() if k not in TIMING}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def key(r):
    return r['table'], r['hand_id'], r['seat']


def main():
    old = read(BASE / 'seat_forced_summary.json')
    new = read(RUN / 'seat_forced_summary.json')
    new.update(read(RUN / 'start_stack_math_summary.json'))
    live = read(RUN / 'start_stack_live_summary.json')
    oldlive = read(BASE / 'start_stack_live_summary.json')
    current = {key(r):r for r in new['results']}
    mapped = []
    for prior in old['results']:
        r = current.get(key(prior), {})
        mapped.append(dict(table=prior['table'],hand_id=prior['hand_id'],seat=prior['seat'],
            occupancy_epoch=r.get('occupancy_epoch'),deal_complete_frame=r.get('deal_complete_frame'),
            first_numeric_visible_frame=r.get('first_numeric_visible_frame'),canonical_frames=r.get('frames'),
            ocr_values=r.get('ocr_values'),displayed_stack=r.get('displayed_stack'),
            bound_ANTE=r.get('forced_bets',{}).get('ANTE'),bound_SB=r.get('forced_bets',{}).get('SB'),
            bound_BB=r.get('forced_bets',{}).get('BB'),START_STACK=r.get('START_STACK'),
            status=r.get('status','UNKNOWN'),blocker=r.get('blocker','MISSING_NEW_GROUP'),
            old_frames=prior['frames'],old_displayed_stack=prior['displayed_stack'],
            same_source_snapshot=r.get('frames')==prior['frames'] and r.get('displayed_stack')==prior['displayed_stack'],
            previous_arithmetic_candidate='START_STACK_CANDIDATE' in prior))
    parity = []
    for name in ['physical_rows','raw_ocr_results','ordered_timeline','context_decisions','removed_duplicates','validated_hands','visual_evidence','hand_analysis']:
        a,b=rows(BASE/(name+'.jsonl')),rows(RUN/(name+'.jsonl'))
        parity.append(dict(stream=name,old_rows=len(a),new_rows=len(b),semantic_equal_timing_only_exclusions=clean(a)==clean(b),exact_bytes=sha(BASE/(name+'.jsonl'))==sha(RUN/(name+'.jsonl'))))
    def deals(v):
        return {(h['table_id'],h['hand_id']):h['deal_complete_frame'] for h in v['hands'] if h['deal_complete_frame'] is not None}
    adjudication=read(ROOT/'runs/hand_state_marker_gate_20261003/acceptance_adjudication.json')
    expected={(h['table'],h['hand_id']):h for h in adjudication['hands']}
    initial={(r['table_id'],r['hand_id'],r['trigger_source_frame'],r['pixel_hash']) for r in adjudication['initial_partial_events']}
    triggers=rows(RUN/'dealing_cards_triggers.jsonl');buckets=collections.defaultdict(list)
    for r in triggers:buckets[r['table_id'],r['hand_id']].append(r)
    unexpected=[r for r in triggers if (r['table_id'],r['hand_id']) not in expected]
    markers=dict(HITS=sum(bool(buckets[k]) for k in expected),MISSES=sum(not buckets[k] for k in expected),
        FALSE_POSITIVES=sum((r['table_id'],r['hand_id'],r['trigger_source_frame'],r['pixel_hash']) not in initial for r in unexpected),
        DUPLICATES=sum(max(0,len(v)-1) for v in buckets.values()),
        SOURCE_BINDING_CONFLICTS=sum((r['trigger_source_frame'],r['pixel_hash'])!=(expected[k]['events'][0]['trigger_source_frame'],expected[k]['events'][0]['pixel_hash']) for k,v in buckets.items() if k in expected for r in v))
    before=read(OUT/'before/production_manifest.json');now=read(ROOT/'production_manifest.json')
    changed=[n for n,h in before['hashes'].items() if sha(ROOT/n)!=h]
    allowed={'PRODUCTION.md','parser_core/start_stack_live.py','tests/test_start_stack_live.py'}
    result=dict(markers=markers,DEAL_COMPLETE_frames_exact=deals(oldlive)==deals(live),
        DEAL_COMPLETE_frames=[dict(table=k[0],hand_id=k[1],frame=v) for k,v in deals(live).items()],
        NCC_receipts_exact=sha(BASE/'dealing_ncc_receipts.jsonl')==sha(RUN/'dealing_ncc_receipts.jsonl'),
        protected_registered_hashes_unchanged=set(changed)<=allowed,changed_registered_files=changed,
        default_flags=read(ROOT/'configs/dealing_start_stack.json'),registered_hashes_match_manifest=all(sha(ROOT/n)==h for n,h in now['hashes'].items()),
        parity=parity,parity_PASS=all(r['semantic_equal_timing_only_exclusions'] for r in parity),
        old_bound_posts=len(old['bound_posts']),new_bound_posts=len(new['bound_posts']),
        bound_posts_exact=clean(old['bound_posts'])==clean(new['bound_posts']),
        occupancy_receipts_exact=old['state_receipts']==new['state_receipts'],
        original_numeric_groups=len(mapped),original_source_snapshots_same=sum(r['same_source_snapshot'] for r in mapped),
        previous_arithmetic_candidates=sum(r['previous_arithmetic_candidate'] for r in mapped),
        previous_arithmetic_now_PASS=sum(r['previous_arithmetic_candidate'] and r['status']=='PASS' for r in mapped),
        numeric_triples=new['numeric_triples'],START_STACK_PASS=new['start_stack_pass'],START_STACK_UNKNOWN=new['start_stack_unknown'],
        original_13_PASS=sum(r['status']=='PASS' for r in mapped),original_13_UNKNOWN=sum(r['status']!='PASS' for r in mapped),
        blockers=dict(collections.Counter(r['blocker'] for r in new['results'])),original_13=mapped,
        guard=read(RUN/'production_guard.json'),errors=read(RUN/'errors_retries.json'),
        final=read(RUN/'final_summary.json'),exit_receipt=read(OUT/'acceptance_ON_exit_receipt.json'))
    (OUT/'acceptance_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    header=['table','hand_id','seat','occupancy_epoch','deal_complete_frame','first_numeric_visible_frame','canonical_frames','ocr_values','displayed_stack','bound_ANTE','bound_SB','bound_BB','START_STACK','status','blocker']
    lines=['# Original 13 live numeric triples','','| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']
    for r in mapped:
        lines.append('| '+' | '.join(str(r.get(k)) for k in header)+' |')
    (OUT/'13_NUMERIC_TRIPLES.md').write_text('\n'.join(lines)+'\n','utf-8')
    print(json.dumps({k:result[k] for k in ['markers','DEAL_COMPLETE_frames_exact','NCC_receipts_exact','parity_PASS','bound_posts_exact','occupancy_receipts_exact','original_source_snapshots_same','previous_arithmetic_now_PASS','START_STACK_PASS','START_STACK_UNKNOWN','blockers']},ensure_ascii=False))


if __name__ == '__main__':
    main()
