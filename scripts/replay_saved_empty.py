"""Official bounded saved native EMPTY lifecycle proof; never fresh acquisition."""
import argparse,ast,base64,copy,hashlib,json,sys,time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
from parser_core.dealing_start_stack import DealingStartStackSidecar
import parser_core.start_stack_live as live
from parser_core.empty_fast_capture import install_empty_capture

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(c,why):
    if not c:raise ValueError(why)
def key(s):return (int(s['table_id']),s['session_id'],str(s['hand_id']))
def load(p):return json.loads(Path(p).read_text('utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def diff(a,b,p='$'):
    if type(a)!=type(b):return p+':TYPE'
    if isinstance(a,dict):
        if set(a)!=set(b):return p+':KEYS'
        for k in a:
            d=diff(a[k],b[k],p+'.'+k)
            if d:return d
    elif isinstance(a,list):
        if len(a)!=len(b):return p+':LENGTH'
        for i,(x,y) in enumerate(zip(a,b)):
            d=diff(x,y,p+f'[{i}]')
            if d:return d
    elif a!=b:return p+':VALUE'
    return None
class NoAcquisitionQueue:
    def __init__(self,*a,**kw):self.errors=[];self.calls=0
    def submit_batch(self,rows):self.calls+=1;raise ValueError('NATIVE_LIFECYCLE_NEW_OCR_SUBMISSION')
    def finish(self):pass

def jvalue(value):
    if isinstance(value,np.ndarray):
        return dict(dtype=value.dtype.str,shape=list(value.shape),sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                    pixels_base64=base64.b64encode(value.tobytes()).decode('ascii'))
    if isinstance(value,dict):return {str(k):jvalue(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,deque)):return [jvalue(v) for v in value]
    if isinstance(value,set):return sorted(jvalue(v) for v in value)
    return value

def snapshot(sidecar,connection):
    def clocks(rows):
        return {k:{f:v for f,v in row.items() if f!='confirmation_monotonic_ns'} for k,row in rows.items()}
    state={name:getattr(sidecar,name) for name in ('header_fences','active_sessions','outstanding',
        'source_barriers','actions','frames','capturing','sealed','context_proofs','previous_cards_count',
        'capture_episodes','counts','fired','flags')}
    state.update(current=clocks(sidecar.current),
        histories={k:[{f:v for f,v in row.items() if f!='confirmation_monotonic_ns'} for row in rows]
                   for k,rows in sidecar.histories.items()},
        triggers=[{f:v for f,v in row.items() if f!='confirmation_monotonic_ns'} for row in sidecar.triggers],
        windows={k:vars(w) for k,w in sidecar.windows.items()},
        hand_state_gate=vars(sidecar.hand_state_gate) if sidecar.hand_state_gate is not None else None)
    state['connection']={name:getattr(connection,name) for name in ('states','pending_tables','pending_stacks',
        'jobs','counts','closed','empty_button_sources','visibility_sources','pending_buffer_frames')}
    return jvalue(state)

def producer_equivalence(old_path,new_path):
    """Every original producer body is unchanged after removing only hook wiring."""
    old=ast.parse(Path(old_path).read_text('utf-8-sig'))
    new=ast.parse(Path(new_path).read_text('utf-8-sig'))
    cls=next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name=='StartStackLiveConnection')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    require(ast.unparse(init.body[-2])=='self.empty_fast_hook = None','PRODUCER_INIT_HOOK_SCOPE')
    require(ast.unparse(init.body[-1])=="""if self.empty_button_model is not None:
    from parser_core.empty_fast_capture import install_empty_capture
    self.empty_fast_hook = install_empty_capture(self)""",'PRODUCER_INSTALL_HOOK_SCOPE')
    del init.body[-2:]
    observe=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='observe_frame')
    require(ast.unparse(observe.body[0].body[0])=="""if getattr(self, 'empty_fast_hook', None) is not None:
    self.empty_fast_hook.sync(table, session, frame)""",'PRODUCER_SYNC_HOOK_SCOPE')
    del observe.body[0].body[0]
    finish=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='finish')
    require(ast.unparse(finish)=="""def finish(self):
    try:
        self._finish()
    finally:
        if getattr(self, 'empty_fast_hook', None) is not None:
            self.empty_fast_hook.close()""",'PRODUCER_FINISH_HOOK_SCOPE')
    cls.body.remove(finish)
    next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_finish').name='finish'
    require(ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False),
            'PRODUCER_UNRELATED_BODY_CHANGE')

def main(args=None):
    if args is None:
        p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--capture',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    start=time.perf_counter();out=Path(args.output).resolve();require(not out.exists(),'FRESH_OUTPUT_REQUIRED');out.mkdir(parents=True)
    planpath=Path(args.plan).resolve();cap=Path(args.capture).resolve();plan=load(planpath);receipt=load(cap/'capture_receipt.json')
    # Original capture pins define immutable acquisition inputs. Four routing
    # controls are separately owned by the current promotion, not pixel inputs.
    replaced={str(ROOT/x) for x in ['run_production.py','PRODUCTION.md','production_manifest.json','parser_core/start_stack_live.py']}
    pins={p:h for p,h in receipt['input_sha256'].items() if p not in replaced}
    for rel in ['parser_core/start_stack_live.py','parser_core/empty_fast_capture.py','parser_core/dealing_start_stack.py','parser_core/empty_seat_evidence.py','off_empty_dispatch.py','off_empty_bound.py','off_empty_pixels.py','models/empty_seat_buttons.json','models/start_stack_table_cards.json','models/table_card_back_supplement.json','configs/dealing_start_stack.json','configs/ocr_tiny_local.json','configs/empty_fast_path.json']:
        path=str(ROOT/rel);digest=sha(path)
        require(path not in pins or pins[path]==digest,'HISTORICAL_INPUT_CHANGED:'+path)
        pins[path]=digest
    for rel in ('off_empty_dispatch.py','off_empty_bound.py','off_empty_pixels.py'):
        golden=ROOT/'provenance/golden-empty-fast-path-20261012/files'/rel
        require(sha(golden)==pins[str(ROOT/rel)],'GOLDEN_MATH_CHANGED:'+rel)
        pins[str(golden)]=sha(golden)
    pins.update({str(planpath):receipt['plan_sha256'],str(cap/'pixels.npz'):receipt['pixels_sha256'],str(cap/'observations.jsonl'):receipt['observations_sha256'],str(cap/'capture_receipt.json'):sha(cap/'capture_receipt.json'),str(Path(__file__).resolve()):sha(__file__)})
    old_producer=ROOT/'runs/empty_fast_promotion_20261012/baseline/rollback/parser_core/start_stack_live.py'
    pins[str(old_producer)]='be78fc14054723eb3f23280836c55d32d4dc0ca9486cd8b53a31b998fe6c672a'
    for p,h in pins.items():require(sha(p)==h,'SOURCE_PIN_CHANGED:'+p)
    require(receipt['verdict']=='PASS' and receipt['covered_hands']==57,'SOURCE_CAPTURE_SCOPE')
    producer_equivalence(old_producer,ROOT/'parser_core/start_stack_live.py')
    records=[json.loads(l) for l in (cap/'observations.jsonl').open(encoding='utf-8') if l.strip()]
    requests={(key(r['source']),r['source']['source_frame']):r for r in plan['requests']}
    initial=[r for r in records if 'FIRST_THREE_RETAINED_TABLE_CHECKS' in r['reasons']]
    joint=[r for r in records if 'RETAINED_JOINT_PIXEL_ANCHOR' in r['reasons']]
    require(len(initial)==171 and len({key(r['source']) for r in initial})==57,'INITIAL_57_SCOPE')
    require(len(joint)==48 and len({key(r['source']) for r in joint})==16,'JOINT_16_SCOPE')
    representative=[r for r in initial if key(r['source'])==key(initial[0]['source'])]
    positive=[r for r in joint if key(r['source'])==key(joint[0]['source'])]
    scenarios=[('all_selected_capture',records),('positive_immediate',positive),('positive_pending_after_two',positive),('positive_pending_EOF',positive),('initial_immediate',initial),('joint_immediate',joint),('pending_callback_after_two',representative),('pending_callback_at_EOF',representative),('pending_unresolved_EOF',representative),('same_hand_newer_header',representative),('failed_callback',representative),('unknown_fence',representative),('closed_window',representative),('session_switch',representative),('hand_switch',representative),('bounded_expiration',[representative[0]]+[representative[1]]*49)]
    original_queue=live.ParallelRowOCR;live.ParallelRowOCR=NoAcquisitionQueue;allmetrics={};results=[]
    try:
        with np.load(cap/'pixels.npz',allow_pickle=False) as pixels:
            for scenario,selected in scenarios:
                arms={};metrics={}
                for mode in ['OLD','NEW']:
                    require(time.perf_counter()-start<240,'LIFECYCLE_240S_BOUND')
                    folder=out/scenario/mode;folder.mkdir(parents=True)
                    from parser_core.hand_state_marker_gate import HandStateMarkerGate
                    sidecar=DealingStartStackSidecar(ROOT,folder,load(ROOT/'configs/dealing_start_stack.json'),hand_state_gate=HandStateMarkerGate())
                    conn=live.StartStackLiveConnection(ROOT,folder,sidecar,load(ROOT/'configs/ocr_tiny_local.json'),first_visible_contract=True,empty_card_contract=True)
                    default_hook=conn.empty_fast_hook
                    if mode=='OLD' and default_hook is not None:default_hook.close()
                    hook=(default_hook or install_empty_capture(conn,enabled=True)) if mode=='NEW' else None
                    conn.empty_fast_hook=hook
                    try:
                        events=[];raw=[];seeded=set();pendingrow=None;observe_wall=0.0
                        def checkpoint(event,source=None):
                            events.append({'event':event,'source':source,'state':snapshot(sidecar,conn)})
                            raw.append({'event':event,'source':source,'state':snapshot(sidecar,conn),'raw_current':jvalue(sidecar.current),'raw_histories':jvalue(sidecar.histories),'raw_triggers':jvalue(sidecar.triggers)})
                        def header_row(trigger,frame=None,pending_lane=False):
                            f=trigger['source_frame'] if frame is None else frame
                            return SimpleNamespace(table_id=trigger['table_id'],session_id=trigger['session_id'],frame_id=f,source_pts=trigger['source_pts'] if frame is None else f*512,source_span_index=trigger['source_position'][1]+(1000000 if pending_lane else 0),time_base_num=trigger['time_base'][0],time_base_den=trigger['time_base'][1],pixel_sha256=trigger['header_pixel_sha256'])
                        def callback(row,hand):
                            sidecar.ocr_complete(row,{'status':'OK','raw_text':'Hand #'+hand});checkpoint('SAVED_HAND_HEADER_CALLBACK')
                        for index,r in enumerate(selected):
                            source=r['source'];k=key(source);request=requests[(k,source['source_frame'])]
                            if k not in seeded:
                                require(len(request['retained_trigger_candidates'])==1,'TRIGGER_NOT_UNIQUE');trigger=request['retained_trigger_candidates'][0]
                                sidecar.active_sessions[k[0]]=k[1]
                                callback(header_row(trigger),k[2])
                                sidecar._commit_marker(sidecar.current[k[0]],dict(source_frame=trigger['trigger_source_frame'],source_pts=trigger['trigger_source_pts'],observed_current_hand_id=trigger.get('observed_current_hand_id')),dict(row_bounds=trigger['row_bounds'],pixel_hash=trigger['pixel_hash']))
                                checkpoint('EXACT_RETAINED_MARKER_SEED',source);seeded.add(k)
                                if scenario.startswith('pending') or scenario.startswith('positive_pending') or scenario in ('same_hand_newer_header','failed_callback','bounded_expiration'):
                                    pendingrow=header_row(trigger,pending_lane=True);sidecar.note_ocr_submission(pendingrow);checkpoint('PENDING_HEADER_SUBMISSION_CONTRACT_FIXTURE',source)
                            if scenario=='unknown_fence' and index==1:
                                sidecar.header_fences[k[:2]]['unknown']=True;checkpoint('UNKNOWN_FENCE_CONTRACT_FIXTURE',source)
                            if scenario=='closed_window' and index==1:
                                sidecar.windows[k].closed=source['source_frame'];checkpoint('CLOSED_WINDOW_CONTRACT_FIXTURE',source)
                            if scenario=='session_switch' and index==1:
                                sidecar.active_sessions[k[0]]=k[1]+'_fixture';checkpoint('SESSION_SWITCH_CONTRACT_FIXTURE',source)
                            if scenario=='hand_switch' and index==1:
                                sidecar.header(k[0],k[1],str(int(k[2])+1),source['source_frame'],source['source_pts'],0,source['time_base'],trigger['header_pixel_sha256']);checkpoint('HAND_SWITCH_CONTRACT_FIXTURE_NO_AUTHORITY',source)
                            if index==2 and scenario in ('pending_callback_after_two','positive_pending_after_two'):callback(pendingrow,k[2])
                            if index==2 and scenario=='same_hand_newer_header':
                                sidecar.outstanding[k[:2]].discard((pendingrow.frame_id,pendingrow.source_span_index))
                                callback(header_row(trigger,source['source_frame'],True),k[2])
                                checkpoint('NEWER_FLOOR_CALLBACK_CONTRACT_FIXTURE_NO_FRESH_SOURCE_AUTHORITY',source)
                            if index==2 and scenario=='failed_callback':
                                sidecar.hand_state_gate.header_unknown(pendingrow);sidecar.ocr_complete(pendingrow,{'status':'ERROR','raw_text':None});checkpoint('UNKNOWN_CALLBACK_NO_AUTHORITY',source)
                            tick=time.perf_counter()
                            if scenario=='all_selected_capture':
                                with sidecar.lock,conn.lock:
                                    observed=conn.empty_button_model.capture(pixels[r['table_array']],source,sidecar.profile['slots'][f'slot_{k[0]:02d}']['seats'])
                                    conn.empty_button_sources[(k,source['source_frame'])]=observed
                                    conn.empty_button_sources={sk:v for sk,v in conn.empty_button_sources.items()
                                        if sk[0][0]!=k[0] or sk[0]==k and sk[1]>source['source_frame']-conn.pending_buffer_frames}
                            else:conn.observe_frame(k[0],k[1],source['source_frame'],source['source_pts'],source['time_base'],pixels[r['table_array']],None)
                            observe_wall+=time.perf_counter()-tick
                            checkpoint('OBSERVE_SELECTED_NATIVE_FRAME',source)
                        if scenario in ('pending_callback_at_EOF','positive_pending_EOF'):callback(pendingrow,key(selected[-1]['source'])[2])
                        checkpoint('BEFORE_DRAIN');dispatch=dict(hook.dispatcher.counts) if hook else {};route_counts=dict(hook.counts) if hook else {}
                        conn.finish();checkpoint('AFTER_DRAIN')
                        require(conn.ocr.calls==0 and not conn.jobs,'NO_NEW_OCR_OR_JOBS')
                        if hook:require(hook.closed and not hook.dispatcher.entries and not hook.contexts and getattr(hook.local,'capture',None) is None,'FAST_CACHE_RELEASE_AFTER_DRAIN')
                        for stream in sidecar.logs.values():stream.close()
                        with (folder/'state_snapshots.jsonl').open('w',encoding='utf-8') as f:
                            for e in events:f.write(json.dumps(e,ensure_ascii=False)+'\n')
                        with (folder/'raw_state_snapshots.jsonl').open('w',encoding='utf-8') as f:
                            for e in raw:f.write(json.dumps(e,ensure_ascii=False)+'\n')
                        native=[json.loads(l) for l in (folder/'start_stack_live_receipts.jsonl').open(encoding='utf-8') if l.strip()]
                        arms[mode]={'states':events,'receipts':native}
                        metrics[mode]={'seeded_hands':len(seeded),'selected_frame_calls':len(selected),'checkpoints':len(events),'confirmed_deals':sum(s['deal_frame'] is not None for s in conn.states.values()),'counts':dict(conn.counts),'dispatcher_counts':dispatch,'routing_counts':route_counts,'capture_observe_wall_seconds':observe_wall,'timing_scope':'EMPTY_CAPTURE_ONLY' if scenario=='all_selected_capture' else 'STARTSTACK_OBSERVE_INCLUDING_OLD_VERIFY','jobs_after_drain':len(conn.jobs),'pending_tables_after_drain':len(conn.pending_tables),'pending_stacks_after_drain':len(conn.pending_stacks),'new_ocr_calls':conn.ocr.calls,'hook_closed':hook.closed if hook else None}
                    except BaseException:
                        save(folder/'failure_raw_snapshots_before_cleanup.json',raw)
                        save(folder/'failure_snapshots_before_cleanup.json',events)
                        raise
                    finally:
                        if not conn.closed:
                            try:conn.finish()
                            finally:
                                if hook is not None:hook.close()
                        for stream in sidecar.logs.values():
                            if not stream.closed:stream.close()
                mismatch=diff(arms['OLD'],arms['NEW']);results.append({'scenario':scenario,'PASS':mismatch is None,'first_difference':mismatch});allmetrics[scenario]=metrics
                if mismatch:save(out/(scenario+'_FULL_OLD_NEW_MISMATCH.json'),arms);raise ValueError('LIFECYCLE_PARITY:'+scenario+':'+mismatch)
        for name in ('positive_immediate','positive_pending_after_two','positive_pending_EOF'):
            require(allmetrics[name]['OLD']['confirmed_deals']==allmetrics[name]['NEW']['confirmed_deals']==1,'POSITIVE_PENDING_DEAL_BRANCH:'+name)
        for p,h in pins.items():require(sha(p)==h,'SOURCE_CHANGED_DURING_REPLAY:'+p)
        require(allmetrics['joint_immediate']['OLD']['confirmed_deals']==allmetrics['joint_immediate']['NEW']['confirmed_deals']==16,'JOINT_DEAL_BRANCH')
        require(allmetrics['bounded_expiration']['OLD']['counts'].get('TABLE_PENDING_BUFFER_EXPIRED',0)>0,'EXPIRATION_BRANCH_NOT_EXERCISED')
        fast_batches=sum(m['NEW']['routing_counts'].get('fast_capture_batches',0) for m in allmetrics.values());require(fast_batches>0,'FAST_BRANCH_NOT_EXERCISED')
        result={'verdict':'PASS','scope':'OFF_SOURCE_SEEDED_NATIVE_LIFECYCLE_AND_EXPLICIT_NON_AUTHORITY_CALLBACK_CONTRACT_FIXTURES','decision_mismatches':0,'state_mismatches':0,'source_receipt_mismatches':0,'native_initial_hands':57,'native_joint_hands':16,'fast_capture_batches':fast_batches,'callback_schedules':len(scenarios),'producer_body_equivalence':'PASS','scenario_results':results,'metrics':allmetrics,'source_input_sha256':pins,'raw_evidence_retained':True,'state_exclusions':['regenerated confirmation_monotonic_ns inside sidecar current/histories/triggers only; values retained in raw_state_snapshots.jsonl'],'complete_callback_tape':False,'full_video_acquisition_parity':False,'synthetic_contract_cases':['pending_lane_index','same_hand_newer_header_clock_without_new_pixels','unknown_fence','closed_window','session_switch','hand_switch','duplicate_saved_frame_buffer_expiration'],'synthetic_numeric_or_WHO_authority':False,'new_ocr_calls':0,'new_solver_calls':0,'new_video_decode_calls':0,'wall_seconds':time.perf_counter()-start}
        save(out/'lifecycle_receipt.json',result);print(json.dumps({k:result[k] for k in ['verdict','native_initial_hands','native_joint_hands','fast_capture_batches','callback_schedules','wall_seconds']}));return result
    except BaseException as e:
        save(out/'lifecycle_failure.json',{'verdict':'FAIL','reason':str(e),'scenario_results':results,'metrics':allmetrics,'wall_seconds':time.perf_counter()-start});raise
    finally:live.ParallelRowOCR=original_queue

if __name__=='__main__':main()


