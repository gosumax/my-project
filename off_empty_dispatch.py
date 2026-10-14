"""Capture-scoped pure EMPTY dispatch. No consumer state or receipt is cached."""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib, json, marshal, os, threading, time
from types import MappingProxyType
import numpy as np
import off_empty_bound as bound_module
import off_empty_pixels as pixel_module
from off_empty_bound import BoundedSadDispatcher
from off_empty_pixels import aligned_pairs
from parser_core.empty_seat_evidence import EmptySeatButtonModel, button_roi

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(x):return hashlib.sha256(x).hexdigest()
def immutable(a):return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)
ORIGINAL_CLASSIFY=EmptySeatButtonModel.classify
ORIGINAL_CAPTURE=EmptySeatButtonModel.capture
BOUND_DECIDE=BoundedSadDispatcher.decide
BOUND_PREPARE=BoundedSadDispatcher._prepare
ALIGNED_PAIRS=aligned_pairs
IMPLEMENTATION_CODES=(ORIGINAL_CLASSIFY.__code__,BOUND_DECIDE.__code__,BOUND_PREPARE.__code__,ALIGNED_PAIRS.__code__)
IMPLEMENTATION=digest(b''.join(marshal.dumps(code) for code in IMPLEMENTATION_CODES))

@dataclass(frozen=True)
class Entry:
    binding:tuple
    frame:int
    pixels:np.ndarray
    pixel_digest:str
    result_bytes:bytes
    run_length:int

class Dispatcher:
    def __init__(self,mode,original=ORIGINAL_CLASSIFY):
        if mode not in ('OLD','NEW'):raise ValueError('EMPTY_MODE')
        self.mode=mode;self.original=original;self.entries={};self.contracts={};self.model_refs={}
        self.counts=Counter();self.cost_ns=Counter();self.history={}
        self.bound=BoundedSadDispatcher(self.old)
        self.bound_contract={}
        self.model_hash_bindings={}

    def contract(self,model):
        # Detect in-memory mutation, not merely a stale declared model hash.
        cfg=canonical(model.model);config_hash=digest(cfg)
        templates=tuple((name,a.shape,a.dtype.str,digest(a.tobytes())) for name,a in model.templates)
        current=(model.sha256,config_hash,templates)
        prior_hash_binding=self.model_hash_bindings.get(model.sha256)
        if prior_hash_binding is not None and prior_hash_binding!=(config_hash,templates):raise ValueError('EMPTY_MODEL_HASH_CONFLICT')
        old=self.contracts.get(id(model))
        if old is not None and old!=current:raise ValueError('EMPTY_MODEL_CONFIG_MUTATED')
        if old is None:
            # Reproduce the loaded native model instead of trusting caller metadata.
            declared=model.model['templates']
            if len(declared)!=len(model.templates):raise ValueError('EMPTY_TEMPLATES')
            for row,(name,a) in zip(declared,model.templates):
                if row['template_id']!=name or row['pixel_sha256']!=digest(a.tobytes()) or row['shape']!=list(a.shape):
                    raise ValueError('EMPTY_TEMPLATE_MUTATED')
            self.contracts[id(model)]=current
            self.model_refs[id(model)]=model
            self.model_hash_bindings[model.sha256]=(config_hash,templates)
        if (BoundedSadDispatcher.decide is not BOUND_DECIDE or BoundedSadDispatcher._prepare is not BOUND_PREPARE
            or bound_module.ROUNDOFF_SLACK!=1e-9 or bound_module.MAX_COMPONENTS!=100000
            or bound_module.BoundedSadDispatcher is not BoundedSadDispatcher
            or getattr(self.bound.decide,'__func__',None) is not BOUND_DECIDE):raise ValueError('EMPTY_BOUND_IMPLEMENTATION_MUTATED')
        if (self.original is not ORIGINAL_CLASSIFY or pixel_module.aligned_pairs is not ALIGNED_PAIRS or
            any(f.__code__ is not code for f,code in zip((ORIGINAL_CLASSIFY,BOUND_DECIDE,BOUND_PREPARE,ALIGNED_PAIRS),IMPLEMENTATION_CODES))):
            raise ValueError('EMPTY_IMPLEMENTATION_MUTATED')
        return (id(model),model.sha256,config_hash,templates,IMPLEMENTATION)

    def old(self,model,gray):
        start=time.perf_counter_ns();cpu=time.thread_time_ns()
        try:
            result=self.original(model,gray)
            self.counts['heavy_old_calls']+=1;self.counts['old_'+result['status']]+=1
            return result
        finally:
            self.cost_ns['old_wall']+=time.perf_counter_ns()-start
            self.cost_ns['old_thread_cpu']+=time.thread_time_ns()-cpu

    def prepare_bound(self,model):
        if model.sha256 not in self.bound.prepared:
            groups=self.bound._prepare(model)
            frozen=MappingProxyType({shape:tuple((dx,dy,None if u is None else immutable(u),n) for dx,dy,u,n in rows) for shape,rows in groups.items()})
            self.bound.prepared[model.sha256]=frozen
            self.bound_contract[model.sha256]=frozen
        if self.bound.prepared[model.sha256] is not self.bound_contract[model.sha256]:raise ValueError('EMPTY_PREPARED_BOUND_MUTATED')

    def decide(self,model,gray,source):
        start=time.perf_counter_ns();cpu=time.thread_time_ns()
        try:return self._decide(model,gray,source)
        finally:
            self.cost_ns['dispatch_wall']+=time.perf_counter_ns()-start
            self.cost_ns['dispatch_thread_cpu']+=time.thread_time_ns()-cpu

    def _decide(self,model,gray,source):
        self.counts['observations']+=1
        contract=self.contract(model)
        slot=(source['table_id'],source['seat'])
        context=(source['session_id'],source['hand_id'],source.get('window_context'))
        binding=(slot,context,tuple(source['geometry']),gray.shape,gray.dtype.str,contract)
        frame=source['source_frame'];previous=self.entries.get(slot)
        pxhash=digest(gray.tobytes()) # Census only; never equality authority.
        hist=self.history.setdefault(binding,[])
        if len(hist)>=2 and hist[-2]==pxhash and hist[-1]!=pxhash:self.counts['A_B_A']+=1
        hist.append(pxhash)
        if previous is not None:
            if previous.binding!=binding or frame<previous.frame:
                self.counts['cache_invalidations']+=1
                self.counts['context_changes']+=int(previous.binding[1]!=context)
                self.counts['geometry_changes']+=int(previous.binding[2]!=binding[2])
                self.counts['shape_dtype_changes']+=int(previous.binding[3:5]!=binding[3:5])
                self.counts['model_config_changes']+=int(previous.binding[5]!=contract)
                self.counts['frame_order_changes']+=int(frame<previous.frame)
            else:
                if previous.pixels.flags.writeable or digest(previous.pixels.tobytes())!=previous.pixel_digest:
                    self.entries.pop(slot);self.counts['cache_invalidations']+=1
                    raise ValueError('EMPTY_CACHED_PIXELS_MUTATED')
                tick=time.perf_counter_ns();equal=np.array_equal(previous.pixels,gray)
                self.cost_ns['exact_equality_wall']+=time.perf_counter_ns()-tick
                if equal:
                    length=previous.run_length+1
                    self.counts['max_repeated_roi_sequence']=max(self.counts['max_repeated_roi_sequence'],length)
                    if self.mode=='NEW':
                        self.counts['exact_hits']+=1
                        self.entries[slot]=Entry(binding,frame,previous.pixels,pxhash,previous.result_bytes,length)
                        result=json.loads(previous.result_bytes)
                        self.counts['result_'+result['status']]+=1
                        return result
                else:length=1
        if previous is None or previous.binding!=binding or frame<previous.frame:length=1
        self.counts['max_repeated_roi_sequence']=max(self.counts['max_repeated_roi_sequence'],length)
        if self.mode=='NEW':
            tick=time.perf_counter_ns();cpu=time.thread_time_ns()
            old_wall_before=self.cost_ns['old_wall'];old_cpu_before=self.cost_ns['old_thread_cpu']
            self.prepare_bound(model)
            old_before=self.counts['heavy_old_calls'];result=self.bound.decide(model,gray)
            elapsed=time.perf_counter_ns()-tick;elapsed_cpu=time.thread_time_ns()-cpu
            self.cost_ns['bound_inclusive_wall']+=elapsed
            self.cost_ns['bound_wall_excluding_old']+=elapsed-(self.cost_ns['old_wall']-old_wall_before)
            self.cost_ns['bound_thread_cpu_excluding_old']+=elapsed_cpu-(self.cost_ns['old_thread_cpu']-old_cpu_before)
            if self.counts['heavy_old_calls']==old_before:
                self.counts['bound_rejects']+=1;self.counts['bound_UNKNOWN']+=1
        else:result=self.old(model,gray)
        self.counts['result_'+result['status']]+=1
        # Mapping insertion order is part of the typed tape and JSON byte contract.
        result_bytes=json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
        self.entries[slot]=Entry(binding,frame,immutable(gray),pxhash,result_bytes,length)
        return deepcopy(result)

class Adapter:
    def __init__(self):
        self.mode=os.environ.get('OFF_EMPTY_MODE','OLD');self.dispatcher=Dispatcher(self.mode)
        self.local=threading.local();self.receipts=[];self.verify_counts=Counter();self.verify_cost=Counter()
        self.journal=None;self.window_refs={}
        adapter=self
        def capture(model,rgb,source,boxes):
            if getattr(adapter.local,'capture',None) is not None:raise ValueError('EMPTY_NESTED_CAPTURE')
            adapter.local.capture=(source,iter(boxes),rgb.shape,boxes)
            try:result=ORIGINAL_CAPTURE(model,rgb,source,boxes)
            finally:adapter.local.capture=None
            # Retain each independently constructed native receipt, including its PNG.
            adapter.receipts.append(deepcopy(result))
            return result
        def classify(model,gray):
            context=getattr(adapter.local,'capture',None)
            if context is None:
                tick=time.perf_counter_ns();cpu=time.thread_time_ns()
                try:
                    result=ORIGINAL_CLASSIFY(model,gray);adapter.verify_counts[result['status']]+=1
                    return result
                finally:
                    adapter.verify_cost['old_wall']+=time.perf_counter_ns()-tick
                    adapter.verify_cost['old_thread_cpu']+=time.thread_time_ns()-cpu
            source,seats,shape,boxes=context;seat=next(seats)
            geometry=button_roi(boxes[seat],shape[1],shape[0])
            window_context=source.get('window_context')
            if adapter.journal is not None:
                provider=adapter.journal.objects[0]
                key=(source['table_id'],source['session_id'],source['hand_id'])
                window=provider.windows.get(key)
                header=provider.current.get(source['table_id'])
                if window is None or header is None:raise ValueError('EMPTY_CAPTURE_WINDOW_UNKNOWN')
                adapter.window_refs[id(window)]=window
                window_context=(key,id(window),header['source_frame'],provider.flags['capture_frames'])
            return adapter.dispatcher.decide(model,gray,dict(source,seat=seat,geometry=geometry,window_context=window_context))
        EmptySeatButtonModel.capture=capture;EmptySeatButtonModel.classify=classify

    def bind(self,journal):self.journal=journal
    def finish(self,body_ns):
        j=self.journal;t=j.tape
        events=t.events;callbacks=[e for e in events if e['kind']=='enter' and e['name'].startswith('callback.')]
        checkpoints=[e for e in events if e['kind']=='checkpoint']
        from off_native_journal import canonical as typed_canonical
        receipt={'mode':self.mode,'counts':dict(self.dispatcher.counts),'cost_ns':dict(self.dispatcher.cost_ns),
            'verify_old_counts':dict(self.verify_counts),'verify_old_cost_ns':dict(self.verify_cost),
            'consumer_replay_body_wall_s':body_ns/1e9,'verification_scope':'ALL_PROVIDER_CHECKPOINTS_FINAL_TYPED_STATE_18_BYTE_FILES',
            'events_consumed':t.position,'callback_count':len(callbacks),'callback_tape_sha256':digest(typed_canonical(callbacks)),
            'provider_checkpoints_checked':len(checkpoints),'provider_checkpoint_tape_sha256':digest(typed_canonical(checkpoints)),
            'final_state_sha256':digest(typed_canonical(j.document['final_state'])),
            'source_receipts_sha256':digest(canonical(self.receipts)),
            'capture_batches':len(self.receipts),'unconsumed_tape':len(events)-t.position,
            'pending_stack_jobs':len(j.objects[2].jobs),'pending_forced_jobs':len(j.objects[3].jobs),
            'provider_outstanding':sum(map(len,j.objects[0].outstanding.values())),'all_consumers_finished':j.closed,
            'new_video':False,'new_OCR':False,'status':'PASS'}
        (j.output/'empty_metrics.json').write_bytes(canonical(receipt))
        (j.output/'empty_source_receipts.json').write_bytes(canonical(self.receipts))
        EmptySeatButtonModel.capture=ORIGINAL_CAPTURE;EmptySeatButtonModel.classify=ORIGINAL_CLASSIFY
