"""Saved-evidence baseline plus isolated historical binding experiment."""
import base64, copy, hashlib, json, sys, threading, time
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from parser_core.seat_forced_binding import SeatForcedBinding
from parser_core.seat_epoch_state import SeatEpochMachine
from parser_core.dealing_start_stack import CleanStackWindow
from parser_core.parallel_ocr import row_key
from parser_core.poker_math_engine import calculate_start_stack
BASE=ROOT/'runs/deal_complete_fix_chat_review_20261003/acceptance_ON'
OUT=ROOT/'runs/historical_forced_binding_20261003'
def read(p):return json.loads(p.read_text('utf-8'))
def rows(p):return [json.loads(l) for l in p.read_text('utf-8').splitlines() if l.strip()]

def historical(b,key,actor,unresolved):
    w=b.sidecar.windows[key];source=w.deal['source_frame'];limit=source+b.sidecar.flags['capture_frames']
    matches=[(r['frames'][-1],seat,r) for seat in w.deal['occupied_seats'] for r in b.nick_labels[(*key,seat)] if r['actor']==actor and r['frames'][-1]<=limit]
    if not matches:return None,'NO_LITERAL_LABEL'
    earliest=min(f for f,s,r in matches);anchor=max(source,earliest);seats={s for f,s,r in matches if f==earliest}
    if len(seats)!=1:return None,'ACTOR_SEAT_CONFLICT'
    seat=next(iter(seats));epoch=b.machine.epoch_at(*key[:2],seat,source)
    if not epoch or b.machine.epoch_at(*key[:2],seat,anchor)!=epoch:return None,'EPOCH_CONFLICT'
    first_labels=[r for f,s,r in matches if f==earliest and s==seat]
    if any(b.machine.epoch_at(*key[:2],seat,f)!=epoch for r in first_labels for f in r['frames']):return None,'LABEL_TRIPLE_EPOCH_CONFLICT'
    actor_seats={s for s in w.deal['occupied_seats'] for r in b.nick_labels[(*key,s)] if r['actor']==actor and r['frames'][-1]<=anchor}
    reverse={r['actor'] for r in b.nick_labels[(*key,seat)] if r['frames'][-1]<=anchor}
    if actor_seats!={seat} or reverse!={actor}:return None,'LITERAL_CONFLICT_BEFORE_ANCHOR'
    if not b.machine.membership(*key[:2],seat,key[2],epoch)['participant']:return None,'PARTICIPANT_UNKNOWN'
    if any(e['table']==key[0] and e['session']==key[1] and e['frame']<=anchor for e in unresolved):return None,'UNRESOLVED_LINEAGE'
    if any(e['table']==key[0] and e['session']==key[1] and e['frame']<=anchor and e['evidence_status']!='PROVEN' for e in b.machine.events):return None,'UNKNOWN_EPOCH_EVENT'
    later=[dict(seat=seat,**r) for r in b.nick_labels[(*key,seat)] if r['frames'][-1]>anchor and r['actor']!=actor]
    return dict(seat=seat,epoch=epoch,anchor=anchor,later_literal_conflicts=later),None

def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=True)
    manifest=read(ROOT/'production_manifest.json');frozen={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in manifest['hashes']}
    original=read(BASE/'seat_forced_summary.json');live=read(BASE/'start_stack_live_summary.json');deals=[r for r in rows(BASE/'deal_complete_receipts.jsonl') if r.get('evidence_status')=='PROVEN'];inputs=rows(BASE/'seat_forced_inputs.jsonl')
    windows={};states={};triggers=rows(BASE/'dealing_cards_triggers.jsonl')
    for d in deals:
        key=(d['table_id'],d['session_id'],d['hand_id']);w=CleanStackWindow();w.confirm_deal(d);windows[key]=w
    for h in live['hands']:
        key=(h['table_id'],h['session_id'],h['hand_id'])
        if key in windows:states[key]=dict(numeric_triples=h['numeric_triples'],canonical_attempts=h['canonical_attempts'],source_readings=defaultdict(dict))
    for r in rows(BASE/'start_stack_readings.jsonl'):
        key=(r['table_id'],r['session_id'],r['hand_id'])
        if key in states:states[key]['source_readings'][r['seat']][r['source_frame']]=r
    side=SimpleNamespace(lock=threading.RLock(),windows=windows,triggers=triggers,flags=read(ROOT/'configs/dealing_start_stack.json'),logs={'start_stack_readings':open(OUT/'start_stack_readings.jsonl','w',encoding='utf-8')})
    side.logs['start_stack_readings'].write((BASE/'start_stack_readings.jsonl').read_text('utf-8'))
    b=SeatForcedBinding.__new__(SeatForcedBinding);b.output=OUT;b.sidecar=side;b.stack_live=SimpleNamespace(states=states);b.lock=threading.RLock();b.machine=SeatEpochMachine();b.counts=Counter();b.jobs={};b.snapshots=defaultdict(dict);b.snapshot_sizes={};b.snapshot_occupied={};b.forced={};b.blockers={};b.nick_runs={};b.nick_labels=defaultdict(list);b.state_events=[];b.closed=False;b.log=open(OUT/'seat_forced_inputs_replay.jsonl','w',encoding='utf-8');b.ocr=SimpleNamespace(finish=lambda:None,errors=[])
    for e in original['state_receipts']:
        if e['kind']=='OCCUPIED':b.machine.observe(e['table'],e['session'],e['seat'],e['kind'],e['frame'],hand_id=e['hand_id'],source_ref=e['source_ref'],source_index=e['source_index'])
    for key,w in windows.items():
        b.snapshot_sizes[key]=sum(r['kind']=='CHAT' and (r['table_id'],r['session_id'],r['hand_id'])==key for r in inputs);b.snapshot_occupied[key]=len(w.deal['occupied_seats'])
    for r in inputs:
        key=(r['table_id'],r['session_id'],r['hand_id']);pixels=cv2.imdecode(np.frombuffer(base64.b64decode(r['png_base64']),np.uint8),cv2.IMREAD_GRAYSCALE)
        assert hashlib.sha256(pixels.tobytes()).hexdigest()==r['pixel_sha256']
        index=r['index']+1000 if r['kind']=='CHAT' else 100+int(r['seat'][4:])-1
        row=b.row(r['table_id'],r['session_id'],r['source_frame'],r['source_pts'],r['time_base'],index,0,pixels,r['kind'])
        meta={k:v for k,v in r.items() if k not in ['png_base64','ocr_response','pixel_sha256']};b.jobs[row_key(row)]=(key,meta);b.completed(row,r['ocr_response'])
    b.state_events.extend({k:v for k,v in e.items() if k!='blocker'} for e in original['unresolved_state_events'])
    b.finish();side.logs['start_stack_readings'].close();baseline=read(OUT/'seat_forced_summary.json')
    assert baseline['bound_posts']==original['bound_posts'],'BASELINE_BOUND_PARITY_FAIL'
    assert baseline['results']==original['results'],'BASELINE_FACT_PARITY_FAIL'
    bound=[];bindings=[]
    for key,snap in b.forced.items():
        for actor in snap['actors']:
            match,block=historical(b,key,actor,baseline['unresolved_state_events']);bindings.append(dict(hand=key[2],actor=actor,match=match,blocker=block))
            if match:bound.extend(dict(p,table=key[0],session=key[1],hand_id=key[2],seat=match['seat'],epoch=match['epoch']) for p in snap['posts'] if p['actor']==actor)
    target=next(k for k in windows if k[2]=='2876719777');saved=copy.deepcopy(b.nick_labels);cases=[]
    for name in ['same_anchor_reverse','before_anchor_reverse','same_anchor_actor_other_seat','replacement','triple_crosses_replacement','unknown_event','unresolved_lineage','pre_snapshot_label_sitout']:
        c=copy.copy(b);c.nick_labels=copy.deepcopy(saved);c.machine=copy.deepcopy(b.machine);unresolved=[]
        if name=='same_anchor_reverse':c.nick_labels[(*target,'seat3')].append(dict(actor='other',frames=[2564,2565,2566],source_ref='synthetic'))
        if name=='before_anchor_reverse':c.nick_labels[(*target,'seat3')].append(dict(actor='other',frames=[2540,2541,2542],source_ref='synthetic'))
        if name=='same_anchor_actor_other_seat':c.nick_labels[(*target,'seat2')].append(dict(actor='bankiir',frames=[2564,2565,2566],source_ref='synthetic'))
        if name in ('replacement','unknown_event'):c.machine.observe(*target[:2],'seat3','REPLACE' if name=='replacement' else 'OCCUPIED',2550,hand_id=target[2],evidence_status='PROVEN' if name=='replacement' else 'UNKNOWN');c.machine.build()
        if name=='triple_crosses_replacement':c.machine.observe(*target[:2],'seat3','REPLACE',2565,hand_id=target[2]);c.machine.build()
        if name=='unresolved_lineage':unresolved=[dict(table=target[0],session=target[1],frame=2550,actor='unknown')]
        if name=='pre_snapshot_label_sitout':
            c.machine=SeatEpochMachine();c.machine.hand(*target,2516);c.machine.observe(*target[:2],'seat3','SIT_OUT',2500,actor='bankiir');c.machine.observe(*target[:2],'seat3','OCCUPIED',2530,hand_id=target[2]);c.machine.build();c.nick_labels[(*target,'seat3')]=[dict(actor='bankiir',frames=[2501,2502,2503],source_ref='synthetic')]
        match,block=historical(c,target,'bankiir',unresolved);cases.append(dict(case=name,rejected=match is None,expected_reject=name!='pre_snapshot_label_sitout',test_pass=(match is None)==(name!='pre_snapshot_label_sitout'),blocker=block))
    provenance=[r for r in inputs if r.get('seat')=='seat3' and r['hand_id']==target[2] and r['source_frame'] in [2530,2564,2565,2566,2580,2581,2582]]
    refs={ref for p in b.forced[target]['posts'] if p['actor']=='bankiir' for ref in p['source_refs']}
    provenance.extend(r for r in inputs if r['kind']=='CHAT' and r['source_ref'] in refs)
    (OUT/'seat3_source_provenance.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in provenance),'utf-8')
    math=[calculate_start_stack(r) for r in baseline['results']]
    report=dict(scope='OFFLINE_HISTORICAL_RULE_EXPERIMENT_NO_PROMOTION',baseline_bound_posts=len(baseline['bound_posts']),baseline_facts_exact=True,candidate_bound_posts=len(bound),previous15_retained=all(r in bound for r in original['bound_posts']),candidate_posts=bound,bindings=bindings,synthetic=cases,math_statuses=dict(Counter(r['status'] for r in math)),math_facts_unchanged=True,registered_hashes_unchanged=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in frozen.items()),wall_seconds=time.perf_counter()-start)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2),'utf-8');print(json.dumps({k:v for k,v in report.items() if k not in ['candidate_posts','bindings']}))
if __name__=='__main__':main()
