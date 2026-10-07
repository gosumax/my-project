"""Saved post-pixel Smoke19 quality replay; launched only by run_production.py."""
from __future__ import annotations
import copy
import hashlib
import json
import shutil
import threading
from collections import Counter
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import base64
import cv2
import numpy as np

from parser_core.normalize_chat import normalize
from parser_core.message_assembly import RowFact, group_rows
from parser_core.poker_pipeline import analyze_timeline
from parser_core.hand_lifecycle import post_excel_contexts, write_lifecycle, context_key, table_key
from parser_core.poker_math_engine import write_start_stack_math
from parser_core.poker_reconstruction_layer import write_reconstruction
from parser_core.seat_forced_binding import SeatForcedBinding
from parser_core.seat_epoch_state import SeatEpochMachine
from parser_core.parallel_ocr import row_key
from parser_core.canonical_events import canonical_join


def read(path):
    return [json.loads(line) for line in path.read_text('utf-8').splitlines() if line.strip()]


def write(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), 'utf-8')


def replay_nickname_binding(source, output):
    """Run the production admission/binder over immutable saved Nick responses."""
    original=json.loads((source/'seat_forced_summary.json').read_text('utf-8'))
    deals={}
    for r in read(source/'start_stack_live_receipts.jsonl'):
        if r.get('status')=='DEAL_COMPLETE':deals[(r['table_id'],r['session_id'],r['hand_id'])]=r
    binder=SeatForcedBinding.__new__(SeatForcedBinding)
    flags=json.loads((Path(__file__).resolve().parents[1]/'configs/dealing_start_stack.json').read_text('utf-8'))
    binder.sidecar=SimpleNamespace(lock=threading.RLock(),flags=flags,
        windows={k:SimpleNamespace(deal=r) for k,r in deals.items()})
    binder.lock=threading.RLock();binder.jobs={};binder.counts=Counter();binder.prefix_candidates=Counter()
    binder.nick_runs={};binder.nick_labels=defaultdict(list);binder.machine=SeatEpochMachine()
    binder.forced={};binder.snapshots=defaultdict(dict);binder.blockers={};binder.state_events=[]
    inputs=read(source/'seat_forced_inputs.jsonl')
    binder.snapshot_sizes=Counter((r['table_id'],r['session_id'],r['hand_id']) for r in inputs if r['kind']=='CHAT')
    binder.snapshot_occupied={k:len(r['occupied_seats']) for k,r in deals.items()}
    with (output/'nickname_admission_replay.jsonl').open('w',encoding='utf-8') as log:
        binder.log=log
        for line,receipt in enumerate(inputs,1):
            if receipt['kind'] not in {'CHAT','NICK'}:continue
            key=(receipt['table_id'],receipt['session_id'],receipt['hand_id'])
            pixels=cv2.imdecode(np.frombuffer(base64.b64decode(receipt['png_base64']),np.uint8),cv2.IMREAD_UNCHANGED)
            assert pixels is not None and hashlib.sha256(pixels.tobytes()).hexdigest()==receipt['pixel_sha256']
            index=int(receipt['ocr_response']['item_id'].split(':')[-1])
            row=binder.row(key[0],key[1],receipt['source_frame'],receipt['source_pts'],receipt['time_base'],
                           index,receipt['roi'][1] if receipt['kind']=='NICK' else receipt['row_bounds'][0],pixels,
                           'NICK_ONLY' if receipt['kind']=='NICK' else 'FORCED_SNAPSHOT')
            meta={k:v for k,v in receipt.items() if k not in {'png_base64','ocr_response','pixel_sha256'}}
            meta.update(saved_input_file='seat_forced_inputs.jsonl',saved_input_line=line)
            binder.jobs[row_key(row)]=(key,meta)
            binder.completed(row,copy.deepcopy(receipt['ocr_response']))
    old_snapshots={(s['table'],s['session'],s['hand_id']):s for s in original['forced_snapshots']}
    post_signature=lambda p:(p['actor'],p['role'],p['amount'],tuple(p['source_refs']),p['source_frame'])
    assert set(binder.forced)==set(old_snapshots)
    assert all(Counter(post_signature(p) for p in binder.forced[k]['posts'])==
               Counter(post_signature(p) for p in old_snapshots[k]['posts']) for k in binder.forced)
    # Reuse exactly the exported occupancy/state receipts, not new inferred
    # occupancy from nickname responses. The production epoch algorithm runs.
    machine=SeatEpochMachine()
    for t in read(source/'dealing_cards_triggers.jsonl'):
        machine.hand(t['table_id'],t['session_id'],t['hand_id'],t['source_frame'])
    for e in original['state_receipts']:
        machine.observe(e['table'],e['session'],e['seat'],e['kind'],e['frame'],actor=e.get('actor'),
            hand_id=e.get('hand_id'),evidence_status=e['evidence_status'],source_ref=e['source_ref'],source_index=e['source_index'])
    machine.build();binder.machine=machine
    assert machine.audit==original['state_receipts']
    bound=[];proofs=[];blockers=[]
    for key,snapshot in binder.forced.items():
        if key not in deals:continue
        for actor in snapshot['actors']:
            posts=[p for p in snapshot['posts'] if p['actor']==actor]
            receipt,reason=binder.historical_forced_binding(key,actor,posts,original['unresolved_state_events'])
            if receipt:
                proofs.append(receipt)
                bound.extend(dict(p,table=key[0],session=key[1],hand_id=key[2],seat=receipt['seat'],epoch=receipt['epoch']) for p in posts)
            else:blockers.append(dict(table=key[0],session=key[1],hand_id=key[2],actor=actor,blocker=reason))
    summary=copy.deepcopy(original)
    summary.update(bound_posts=bound,historical_binding_receipts=proofs,historical_binding_blockers=blockers,
                   forced_bets_bound=len(bound),saved_binding_replay=True,
                   original_summary_sha256=hashlib.sha256((source/'seat_forced_summary.json').read_bytes()).hexdigest())
    (output/'seat_forced_summary_source.json').write_bytes((source/'seat_forced_summary.json').read_bytes())
    (output/'seat_forced_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    signature=lambda p:(p['table'],p['session'],p['hand_id'],p['actor'],p['seat'],p['epoch'],p['role'],p['amount'],tuple(p['source_refs']))
    assert {signature(p) for p in original['bound_posts']}.issubset({signature(p) for p in bound})
    return dict(bound_before=len(original['bound_posts']),bound_after=len(bound),occupancy_receipt_parity=True,
                saved_callback_order_preserved=True,physical_forced_snapshot_parity=True)


def metrics(records, reconstruction, timeline, math, snapshots):
    complete = [r for r in records if r.get('lifecycle', {}).get('status') == 'COMPLETE']
    observed={(table_key(r['table_id']),r['session_id'],r['hand_id'],r['player'],
               r['canonical_provenance']['post']['role'],r['amount'],tuple(r['canonical_provenance']['post']['source_refs']))
              for r in timeline if r.get('canonical_event_id') and not r.get('canonical_duplicate_of') and not r.get('canonical_conflict')}
    full_prefix={(s['table'],s['session'],s['hand_id']) for s in snapshots if s.get('forced_snapshot_complete') and
                 all((s['table'],s['session'],s['hand_id'],p['actor'],p['role'],p['amount'],tuple(p['source_refs'])) in observed for p in s['posts'])}
    return dict(COMPLETE=len(complete), validation=dict(Counter(r['validation']['status'] for r in complete)),
        reconstruction=dict(Counter(r['status'] for r in reconstruction)),
        VALID=sum(r['fully_validated'] for r in complete), CERTIFIED=sum(r['certified_hh'] for r in complete),
        errors=dict(Counter(i['code'] for r in complete for i in r['validation']['issues'])),
        raw_UNKNOWN=sum(r['event_type']=='UNKNOWN_EVENT' for r in timeline),
        semantic_UNKNOWN=sum(m['action']=='UNKNOWN_EVENT' for r in complete for m in r['observed_hand']['messages']),
        forced=dict(Counter(r.get('action_type') for r in timeline if r.get('canonical_event_id'))),
        complete_forced_prefix=sum(context_key(r['observed_hand']) in full_prefix for r in complete),
        both_blinds_observed=sum({'SMALL_BLIND','BIG_BLIND'}.issubset({m['action'] for m in r['observed_hand']['messages']}) for r in complete),
        stack_math_statuses=dict(Counter(r['status'] for r in math)),
        stack_snapshot_candidates=sum(r.get('snapshot_arithmetic_status')=='PASS' or r['status']=='PASS' for r in math),
        stack_certified=sum(bool(r.get('start_stack_certified')) for r in math),
        stack_candidates_joined=sum(len(r.get('joined_stack_snapshot_candidates',{})) for r in reconstruction),
        stack_joined_into_solver=sum(len(r.get('joined_start_stacks',{})) for r in reconstruction))


def main(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError('Replay output must be separate from immutable source run')
    output.mkdir(parents=True, exist_ok=False)
    names=['ordered_timeline.jsonl','raw_ocr_results.jsonl','physical_rows.jsonl',
        'complete_hand_analysis.jsonl','logical_hand_analysis.jsonl','poker_reconstruction.jsonl',
        'start_stack_math.jsonl','start_stack_facts.jsonl','seat_forced_summary.json',
        'canonical_forced_events.jsonl','canonical_join_summary.json','hand_header_ocr.jsonl','hand_state_receipts.jsonl']
    names+=['seat_forced_inputs.jsonl','start_stack_live_receipts.jsonl','dealing_cards_triggers.jsonl']
    hashes={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in names}
    for name in ['start_stack_facts.jsonl','seat_forced_summary.json','canonical_forced_events.jsonl',
                 'canonical_join_summary.json','hand_header_ocr.jsonl','hand_state_receipts.jsonl']:
        shutil.copy2(source/name,output/name)
    binding_replay=replay_nickname_binding(source,output)
    original=read(source/'ordered_timeline.jsonl')
    # Rejoin derived canonical events from the same immutable ordinary rows.
    # Original canonical rows/report remain available in the source run.
    ordinary=[r for r in original if not r.get('canonical_provenance') and r.get('source_kind')!='SIDECAR_HEADER_RECEIPT']
    timeline=copy.deepcopy(ordinary)
    raw=read(source/'raw_ocr_results.jsonl');ocr={r['row_id']:r['raw_ocr_response']['status'] for r in raw}
    for row in timeline:
        if row.get('canonical_stream_version') or row['event_type']=='BOARD':
            continue
        c=normalize([row['raw_ocr_text']]);row['event_type']=c.event_type
        row['notes_error']=c.reason or ''
        if row['status']=='UNKNOWN' and c.event_type!='UNKNOWN_EVENT':row['status']='RAW_UNREVIEWED'
    for table in sorted({r['table_id'] for r in timeline}):
        rr=[r for r in timeline if r['table_id']==table and not r.get('canonical_stream_version')]
        facts=[RowFact(r['stable_row_id'],r['session_id'],str(r['epoch']),r['frame_id'],r['row_index'],
                       r['raw_ocr_text'],None,None,r['origin']) for r in rr]
        by_id={r['stable_row_id']:r for r in rr}
        for i,m in enumerate(group_rows(facts),1):
            candidate=normalize([r.text for r in m.rows])
            for fact in m.rows:
                by_id[fact.row_id].update(message_id=f'{table}:M{i:06d}',message_event_type=candidate.event_type,
                                         message_status=m.completeness,message_reason=m.reason or '')
    identity_fields=['stable_row_id','table_id','session_id','frame_id','row_index','raw_ocr_text','row_pixel_sha256',
                     'source_pts','time_base','video_time','epoch','canonical_event_id']
    assert [[r.get(k) for k in identity_fields] for r in ordinary]==[[r.get(k) for k in identity_fields] for r in timeline]
    timeline,_=canonical_join(output,timeline)
    old_event_ids={r['canonical_event_id'] for r in original if r.get('canonical_event_id')}
    new_event_ids={r['canonical_event_id'] for r in timeline if r.get('canonical_event_id')}
    assert old_event_ids.issubset(new_event_ids)
    records, summary=analyze_timeline(timeline,ocr_status=ocr)
    logical, complete, eligible, lifecycle=post_excel_contexts(output,records)
    write_lifecycle(output,logical,lifecycle)
    write(output/'ordered_timeline.jsonl',timeline);write(output/'hand_analysis.jsonl',records)
    write(output/'logical_hand_analysis.jsonl',logical);write(output/'complete_hand_analysis.jsonl',complete)
    (output/'hand_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    write_start_stack_math(output,eligible_contexts=eligible)
    reconstruction=write_reconstruction(output,complete)
    snapshots=json.loads((source/'seat_forced_summary.json').read_text('utf-8'))['forced_snapshots']
    before=metrics(read(source/'logical_hand_analysis.jsonl'),read(source/'poker_reconstruction.jsonl'),original,read(source/'start_stack_math.jsonl'),snapshots)
    after=metrics(logical,read(output/'poker_reconstruction.jsonl'),timeline,read(output/'start_stack_math.jsonl'),snapshots)
    report=dict(scope='SAVED_POST_PIXEL_REPLAY_NO_DECODE_NO_OCR_NO_NEW_CAPTURE',source=str(source),
        before=before,after=after,ordinary_source_rows_identity_order_raw_hash_parity=True,
        original_canonical_event_ids_retained=True,new_canonical_event_ids=sorted(new_event_ids-old_event_ids),
        source_input_hashes=hashes,reconstruction_summary=reconstruction,
        binding_replay=binding_replay,
        performance_code_changed=False,frozen_vendor_changed=False,verdict='QUALITY_PARTIAL')
    assert hashes=={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in names}
    (output/'quality_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(dict(before=before,after=after),ensure_ascii=False,indent=2))
    return report
