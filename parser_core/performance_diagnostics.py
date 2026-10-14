"""Opt-in measurements only. No admission, ordering, queue or OCR decisions.

Thread CPU measures the caller; interval process CPU includes concurrent native
threads. Inclusive intervals overlap and are never summed as critical-path wall.
"""
from __future__ import annotations
import functools
import gc
import importlib
import json
import math
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from array import array
from contextlib import contextmanager
from pathlib import Path
from .bounded_storage import BoundedStats, DiskRecords, DiskIntervals

ACTIVE = None


class MeasuredLock:
    def __init__(self,lock,diagnostic,name):self.lock=lock;self.diagnostic=diagnostic;self.name=name
    def acquire(self,*args,**kwargs):
        tick=self.diagnostic.tick()
        try:return self.lock.acquire(*args,**kwargs)
        finally:self.diagnostic.record(self.name+'/lock_acquire_wait',tick,kind='wait')
    def release(self):return self.lock.release()
    def __enter__(self):return self.acquire()
    def __exit__(self,*args):self.release()
    def __getattr__(self,key):return getattr(self.lock,key)


def distribution(values):
    if isinstance(values,BoundedStats):return values.distribution()
    values=sorted(values)
    if not values:return dict(calls=0,total=0,mean=None,median=None,p95=None,p99=None,max=None)
    return dict(calls=len(values),total=sum(values),mean=statistics.fmean(values),median=statistics.median(values),
                p95=values[max(0,math.ceil(len(values)*.95)-1)],p99=values[max(0,math.ceil(len(values)*.99)-1)],max=values[-1])


def union_seconds(intervals, lo=None, hi=None):
    pairs=sorted((max(a,lo) if lo is not None else a,min(b,hi) if hi is not None else b) for a,b in intervals)
    total=0;end=float('-inf')
    for a,b in pairs:
        if b>a and b>end:total+=b-max(a,end);end=b
    return total


class Diagnostics:
    def __init__(self, output):
        self.output=Path(output);self.start=time.perf_counter();self.cpu_start=time.process_time()
        self.main_thread=threading.get_ident();self.local=threading.local();self.lock=threading.RLock()
        self.metrics=defaultdict(lambda:dict(wall=BoundedStats(),thread_cpu=BoundedStats(),process_cpu=BoundedStats(),objects=0,kind='compute'))
        self.intervals=DiskIntervals(self.output/'performance_intervals.sqlite');self.counters=Counter();self.queues={};self.queue_depth=defaultdict(BoundedStats)
        self.job_submit={};self.jobs=DiskRecords(self.output/'performance_jobs.jsonl');self.samples=DiskRecords(self.output/'performance_samples_spool.jsonl');self.patches=[];self.owners=[]
        self.ocr_compute=DiskRecords(self.output/'performance_ocr_compute.jsonl');self.latencies=defaultdict(BoundedStats)
        self.running_jobs={}
        self.stop=threading.Event();self.capture_bounds=None
        self.active_workers=Counter();self.max_active=Counter();self.last_progress=0
        from .performance_work import WorkCounters
        self.work=WorkCounters(self)

    def tick(self):return time.perf_counter(),time.thread_time(),time.process_time()

    def record(self,name,tick,objects=1,kind='compute',table=None):
        end=time.perf_counter();tcpu=time.thread_time()-tick[1];pcpu=time.process_time()-tick[2]
        table=table if table is not None else getattr(self.local,'table',None)
        names=[name]+([f'T{int(table):02d}/{name}'] if isinstance(table,int) else [])
        with self.lock:
            for key in names:
                m=self.metrics[key];m['wall'].append(end-tick[0]);m['thread_cpu'].append(tcpu);m['process_cpu'].append(pcpu);m['objects']+=objects;m['kind']=kind
            if threading.get_ident()==self.main_thread:
                self.intervals.append((tick[0],end,name,kind))

    @contextmanager
    def span(self,name,objects=1,kind='compute'):
        tick=self.tick()
        try:yield
        finally:self.record(name,tick,objects,kind)

    def count(self,name,value=1):
        with self.lock:
            self.counters[name]+=value
            table=getattr(self.local,'table',None)
            if isinstance(table,int) and name.startswith(('tracker_','array_equal_','NCC_')):
                self.counters[f'T{table:02d}/{name}']+=value

    def retained_diagnostic_bytes(self):
        # Explicit shallow + known numeric/interval/job storage estimate. RSS
        # cannot subtract this exactly: allocators and process sharing differ.
        total=sys.getsizeof(self.metrics)+sys.getsizeof(self.intervals.pending)
        stats=[v for m in self.metrics.values() for v in (m['wall'],m['thread_cpu'],m['process_cpu'])]+list(self.queue_depth.values())+list(self.latencies.values())
        total+=sum(sys.getsizeof(v)+sys.getsizeof(v.values)+len(v.values)*24 for v in stats)
        total+=sum(sys.getsizeof(i)+sum(sys.getsizeof(v) for v in i) for i in self.intervals.pending)
        total+=sum(sys.getsizeof(j)+sum(sys.getsizeof(v) for v in j.values()) for j in self.running_jobs.values())
        return total

    def wrap(self,owner,attr,name,kind='compute',table_arg=None,after=None):
        original=owner[attr] if isinstance(owner,dict) else getattr(owner,attr)
        @functools.wraps(original)
        def measured(*args,**kwargs):
            old_table=getattr(self.local,'table',None)
            if table_arg is not None and len(args)>table_arg:self.local.table=args[table_arg]
            elif args and hasattr(args[0],'table_id'):self.local.table=args[0].table_id
            tick=self.tick()
            try:
                value=original(*args,**kwargs)
                if after is not None:after(args,kwargs,value)
                return value
            finally:
                self.record(name,tick,kind=kind);self.local.table=old_table
        if isinstance(owner,dict):owner[attr]=measured
        else:setattr(owner,attr,measured)
        self.patches.append((owner,attr,original))

    def install(self,runner_globals=None):
        # Patch only explicit methods/functions. OFF never invokes install().
        specs={
            'parser_core.chat_tracker': [('AdjacentChatTracker.process','chat_tracker_total','compute'),
                ('physical_spans','chat_tracker_row_detection','compute'),('symmetry','chat_tracker_identity_matching','compute'),
                ('coordinate_map','chat_tracker_identity_matching','compute'),('zone_scores','chat_tracker_ncc','compute')],
            'parser_core.hand_header_admission':[('HandHeaderAdmission.observe_frame','HandHeaderAdmission','container'),
                ('HandHeaderAdmission.ocr_complete','header_callback_processing','compute'),('HandHeaderAdmission.finish','header_final_flush','container')],
            'parser_core.dealing_start_stack':[('DealingStartStackSidecar.observe_frame','DealingStartStackSidecar','container'),
                ('DealingStartStackSidecar.header','header_admission','compute'),('DealingStartStackSidecar.ocr_complete','ordinary_sidecar_callback','compute'),
                ('DealingStartStackSidecar.finish','dealing_final_flush','container')],
            'parser_core.start_stack_live':[('StartStackLiveConnection.observe_frame','StartStackLiveConnection.observe_frame','container'),
                ('StartStackLiveConnection._replay_pending','replay_traversal','container'),
                ('StartStackLiveConnection._observe_frame','replay_admission_gates_and_stack_capture','container'),
                ('StartStackLiveConnection._buffer_metrics','pending_buffer_metrics','compute'),
                ('StartStackLiveConnection.completed','stack_callback_processing','compute'),('StartStackLiveConnection.finish','stack_final_flush','container')],
            'parser_core.seat_forced_binding':[('SeatForcedBinding.observe_frame','SeatForcedBinding.observe_frame','container'),
                ('SeatForcedBinding.completed','forced_callback_processing','compute'),('SeatForcedBinding.finish','forced_final_flush','container')],
            'parser_core.tiny_chat_client':[('TinyChatClient._start','OCR_process_startup','container'),
                ('TinyChatClient._next_line','OCR_response_wait','wait'),('TinyChatClient.close','OCR_shutdown_wait','wait')],
            'parser_core.visual_tile_queue':[('VisualTileQueue.finish','card_final_drain','wait')],
            'parser_core.card_hybrid':[('CardHybrid.read_bytes','card_worker','compute')],
            'parser_core.temporal_overlap':[('TemporalObservations.reduce','temporal_dedup','container')],
            'parser_core.poker_pipeline':[('analyze_timeline','poker_analysis','compute')],
            'parser_core.poker_reconstruction_layer':[('write_reconstruction','reconstruction','compute')],
        }
        for module,entries in specs.items():
            mod=importlib.import_module(module)
            for path,name,kind in entries:
                owner=mod;parts=path.split('.')
                for part in parts[:-1]:owner=getattr(owner,part)
                if not hasattr(owner,parts[-1]):continue
                self.wrap(owner,parts[-1],name,kind,table_arg=1 if parts[-1] in {'observe_frame','_observe_frame','_replay_pending','header'} else None,
                          after=self.tracker_result if name=='chat_tracker_total' else
                          (lambda a,k,r:self.count('tracker_NCC_shift_comparisons',len(r[0]))) if path=='zone_scores' else None)
        # Bindings imported by the orchestration module predate method patches.
        import sys
        runner=runner_globals or sys.modules.get('scripts.run_smoke14_ram')
        if runner:
            for name in ('analyze_timeline','write_reconstruction'):
                self.wrap(runner,name,name,'compute')
        import numpy as np
        self.wrap(np,'array_equal','array_equal',after=self.comparison_result)
        import cv2
        self.wrap(cv2,'matchTemplate','NCC_matchTemplate',after=lambda a,k,r:self.count('NCC_comparisons'))
        for module,cls in [('parser_core.dealing_start_stack','DealingStartStackSidecar'),
                           ('parser_core.start_stack_live','StartStackLiveConnection'),
                           ('parser_core.seat_forced_binding','SeatForcedBinding')]:
            self.wrap(getattr(importlib.import_module(module),cls),'__init__',cls+'/init','container',
                      after=lambda a,k,r:self.measure_owner_locks(a[0]))

        from .performance_work_targets import install
        install(self.work,runner)

    def measure_owner_locks(self,owner):
        for attr in ('lock','_lock'):
            lock=getattr(owner,attr,None)
            if lock is not None and not isinstance(lock,MeasuredLock):setattr(owner,attr,MeasuredLock(lock,self,type(owner).__name__))

    def tracker_result(self,args,kwargs,result):
        self.count('tracker_calls');self.count('tracker_rows',len(result['new_observed_spans']))
        self.count('tracker_cartesian_candidate_upper_bound',len(result['old_ids'])*len(result['new_ids']))
        old={v for v in result['old_ids'] if v is not None};new={v for v in result['new_ids'] if v is not None}
        self.count('tracker_NEW',len(new-old));self.count('tracker_SAME',len(new&old))
        self.count('tracker_LOST',len(old-new))
        self.count('tracker_OCR_admitted',len(result['OCR_queue_row_ids']))
        self.count('tracker_confirmed_removed_top',len(result['removed_top_ids']))
        self.count('tracker_REAPPEARED_UNMEASURED',0)
        self.count('tracker_'+result['classification'])

    def comparison_result(self,args,kwargs,result):
        self.count('array_equal_changed' if not result else 'array_equal_no_change')
        self.count('array_equal_operand_bytes',sum(getattr(a,'nbytes',0) for a in args[:2]))
        self.count('array_equal_operand_elements',sum(getattr(a,'size',0) for a in args[:2]))

    def register_queue(self,owner,lane):
        self.queues[id(owner.jobs)]=(lane,owner.jobs,owner)
        self.owners.append(owner)
        self.measure_owner_locks(owner)

    def queue_sample(self,owner):
        lane=self.queues.get(id(owner.jobs),('other',))[0];depth=owner.jobs.qsize()
        with self.lock:self.queue_depth[lane].append(depth)
        self.count(lane+'/queue_empty_observations',depth==0)
        self.count(lane+'/queue_limit_observations',depth==owner.jobs.maxsize)

    def submitted(self,owner,rows):
        now=time.perf_counter();lane=self.queues[id(owner.jobs)][0]
        with self.lock:
            for row in rows:self.job_submit[(id(owner),row.table_id,row.frame_id,row.source_span_index)]=(now,lane)
        self.count(lane+'/jobs_submitted',len(rows))
        for row in rows:
            self.count(f'T{row.table_id:02d}/{lane}/jobs_submitted')
            pixel_hash=getattr(row,'pixel_sha256',None)
            shape=getattr(getattr(row,'pixels',None),'shape',None)
            self.work.note('ocr/'+lane+'/submitted_crop', category='model/ocr_input',
                key=dict(pixel_sha256=pixel_hash,shape=list(shape) if shape is not None else None)
                    if pixel_hash is not None else None, crops=1)

    def batch_started(self,owner,batch):
        now=time.perf_counter();lane=self.queues[id(owner.jobs)][0]
        with self.lock:
            self.active_workers[lane]+=1;self.max_active[lane]=max(self.max_active[lane],self.active_workers[lane])
            for row in batch:
                submitted,_=self.job_submit.pop((id(owner),row.table_id,row.frame_id,row.source_span_index),(now,lane))
                job=dict(lane=lane,table=row.table_id,frame=row.frame_id,span=row.source_span_index,
                    submit=submitted-self.start,start=now-self.start,queue_wait_seconds=now-submitted,batch_size=len(batch))
                self.running_jobs[(id(owner),row.table_id,row.frame_id,row.source_span_index)]=job
        self.queue_sample(owner)
        return now

    def batch_done(self,owner,batch,started):
        lane=self.queues[id(owner.jobs)][0]
        with self.lock:
            self.active_workers[lane]-=1
            for row in batch:
                job=self.running_jobs.pop((id(owner),row.table_id,row.frame_id,row.source_span_index),None)
                if job:
                    job['callback_end']=time.perf_counter()-self.start
                    self.jobs.append(job)
        self.count(lane+'/jobs_completed',len(batch))

    def compute_received(self,owner,batch,detail):
        with self.lock:
            for row in batch:
                job=self.running_jobs.get((id(owner),row.table_id,row.frame_id,row.source_span_index))
                if job:
                    job['response_received']=time.perf_counter()-self.start
                    if detail:
                        job.update(actual_compute_wall=detail.get('inference_wall_seconds'),
                            actual_compute_cpu=detail.get('inference_process_cpu_seconds'),
                            compute_end_monotonic_seconds=detail['compute_end_monotonic_seconds'],
                            compute_end_to_response_seconds=time.perf_counter()-detail['compute_end_monotonic_seconds'])

    def callback_boundary(self,owner,row,boundary):
        """Availability at callback entry/return, not a proven accepted hand event."""
        now=time.perf_counter();lane=self.queues[id(owner.jobs)][0]
        with self.lock:
            job=self.running_jobs.get((id(owner),row.table_id,row.frame_id,row.source_span_index))
            if not job:return
            job['callback_'+boundary]=now-self.start
            end=job.get('compute_end_monotonic_seconds')
            if end is not None:
                latency=now-end
                job['compute_end_to_callback_'+boundary+'_seconds']=latency
                self.latencies[lane+'/compute_end_to_callback_'+boundary].append(latency)

    def sample(self,state):
        import psutil
        process=psutil.Process();members=[process]+process.children(recursive=True)
        data=dict(elapsed=time.perf_counter()-self.start,phase=state.get('phase'),gc_count=gc.get_count(),gc_stats=gc.get_stats(),
                  system=psutil.virtual_memory()._asdict(),per_core_cpu_percent=psutil.cpu_percent(interval=None,percpu=True),processes=[])
        data['diagnostic_retained_bytes_estimate']=self.retained_diagnostic_bytes()
        data['work_events_recorded']=self.work.events.count
        data['work_calls_aggregated']=self.work.aggregated_calls
        for p in members:
            try:
                ct=p.cpu_times();io=p.io_counters()
                data['processes'].append(dict(pid=p.pid,created=p.create_time(),rss=p.memory_info().rss,cpu_seconds=ct.user+ct.system,
                    read_bytes=io.read_bytes,write_bytes=io.write_bytes))
            except psutil.Error:pass
        data['queue_depth']={lane:q.qsize() for lane,q,_ in list(self.queues.values())}
        data['queue_retained']=[]
        for lane,q,_ in list(self.queues.values()):
            items=list(q.queue)
            rows=[r for batch in items for r in (batch if isinstance(batch,list) else [batch]) if hasattr(r,'pixels')]
            data['queue_retained'].append(dict(lane=lane,objects=len(rows),bytes=sum(r.pixels.nbytes for r in rows),
                note='Approximate shallow queued ROI sum; shared objects may be counted more than once'))
        data['active_workers']=dict(self.active_workers)
        data['processed_ocr_jobs']={k:v for k,v in self.counters.items() if k.endswith('/jobs_completed')}
        data['retained_buffers']=[]
        live=state.get('start_stack_live')
        if live:
            # Snapshot references only; never take pipeline locks in sampler.
            try:
                for key,pending in list(live.pending_tables.items()):
                    arrays=[p for source,e in list(pending['sources']) for kind in ('cards','stacks') for p in list(e[kind].values())]
                    data['retained_buffers'].append(dict(key=list(key),kind='table',objects=len(arrays),bytes=sum(p.nbytes for p in arrays)))
                for key,pending in list(live.pending_stacks.items()):
                    arrays=[p for source,e in list(pending) for p in list(e.values())]
                    data['retained_buffers'].append(dict(key=list(key),kind='stack',objects=len(arrays),bytes=sum(p.nbytes for p in arrays)))
            except (RuntimeError,KeyError):data['buffer_sample_status']='CONCURRENT_MUTATION_UNAVAILABLE'
        if len(self.samples):
            old=self.samples[-1];dt=data['elapsed']-old['elapsed']
            previous={(p['pid'],p['created']):p['cpu_seconds'] for p in old['processes']}
            data['process_cpu_percent_sum']=100*sum(max(0,p['cpu_seconds']-previous.get((p['pid'],p['created']),p['cpu_seconds'])) for p in data['processes'])/dt if dt else None
        self.samples.append(data)
        with (self.output/'performance_resource_samples.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(data)+'\n')
        for lane,q,owner in list(self.queues.values()):self.queue_sample(owner)
        if data['elapsed']-self.last_progress>=30 or not self.last_progress:
            stage=state.get('stage');frame=stage.seen_frames if stage else 0;duration=stage.last_video_time if stage else None
            progress=dict(data,frame=frame,fps=frame/data['elapsed'] if data['elapsed'] else None,
                          rtf_so_far=data['elapsed']/duration if duration else None)
            with (self.output/'performance_progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(progress)+'\n')
            self.last_progress=data['elapsed']

    def start_sampler(self,state):
        def run():
            while not self.stop.is_set():
                try:self.sample(state)
                except Exception as exc:self.count('sampler_errors');self.last_sample_error=repr(exc)
                self.stop.wait(5)
        self.sampler=threading.Thread(target=run,daemon=True);self.sampler.start()

    def finish(self,summary):
        try:
            return self._finish(summary)
        finally:
            self.stop.set()
            for owner,attr,original in reversed(self.patches):
                if isinstance(owner,dict):owner[attr]=original
                else:setattr(owner,attr,original)
            self.patches.clear()
            global ACTIVE
            ACTIVE=None
            try:self.work.events.close()
            except Exception:pass

    def _finish(self,summary):
        self.stop.set();self.sampler.join(timeout=6)
        wall=time.perf_counter()-self.start;cpu=time.process_time()-self.cpu_start
        stages=[]
        for name,m in self.metrics.items():
            d=distribution(m['wall']);tcpu=m['thread_cpu'].total;pcpu=m['process_cpu'].total
            stages.append(dict(stage=name,wall=d,thread_cpu_seconds=tcpu,process_cpu_interval_seconds=pcpu,
                thread_cpu=distribution(m['thread_cpu']),process_cpu_interval=distribution(m['process_cpu']),
                attributed_cpu_seconds=tcpu,cpu_scope='CALLER_THREAD_INCLUSIVE',
                effective_caller_thread_cores=tcpu/d['total'] if d['total'] else None,
                effective_process_cores_in_interval=pcpu/d['total'] if d['total'] else None,
                objects=m['objects'],throughput=m['objects']/d['total'] if d['total'] else None,kind=m['kind'],
                note='Inclusive overlapping windows; process CPU also includes concurrent work'))
        for lane in {v['lane'] for v in self.ocr_compute}:
            values=BoundedStats();cpus=BoundedStats();objects=0
            for v in self.ocr_compute:
                if v['lane']==lane:
                    values.append(v['inference_wall_seconds']);cpus.append(v['inference_process_cpu_seconds']);objects+=v['batch_size']
            d=distribution(values);actual_cpu=cpus.total
            stages.append(dict(stage=lane+'/OCR_actual_inference',wall=d,thread_cpu_seconds=None,
                process_cpu_interval=distribution(cpus),
                process_cpu_interval_seconds=actual_cpu,attributed_cpu_seconds=actual_cpu,cpu_scope='OCR_CHILD_PROCESS_INFERENCE',
                effective_caller_thread_cores=None,effective_process_cores_in_interval=actual_cpu/d['total'] if d['total'] else None,
                objects=objects,throughput=objects/d['total'] if d['total'] else None,
                kind='child_compute',note='Actual child inference CPU/wall; different workers overlap in wall'))
        known_compute=known_wait=unaccounted=None
        if self.capture_bounds:
            lo,hi=self.capture_bounds
            known_wait=self.intervals.union(lo,hi,kinds=('wait',))
            accounted=self.intervals.union(lo,hi,kinds=('wait','compute'))
            known_compute=accounted-known_wait;unaccounted=max(0,hi-lo-accounted)
        rss=BoundedStats();rss_first=rss_last=None;monotonic=True;largest_drop=0;phases={}
        cpu_history={}
        for s in self.samples:
            value=sum(p['rss'] for p in s['processes']);rss.append(value)
            if rss_first is None:rss_first=value
            if rss_last is not None:
                monotonic=monotonic and value>=rss_last;largest_drop=max(largest_drop,rss_last-value)
            rss_last=value
            phase=phases.setdefault(s['phase'],dict(rss_min=value,rss_max=value))
            phase['rss_min']=min(phase['rss_min'],value);phase['rss_max']=max(phase['rss_max'],value)
            for p in s['processes']:
                key=(p['pid'],p['created']);old=cpu_history.get(key)
                cpu_history[key]=(old[0] if old else p['cpu_seconds'],p['cpu_seconds'])
        pipeline_cpu=sum(b-a for a,b in cpu_history.values())
        lo,hi=self.capture_bounds or (self.start,self.start)
        observer_cost=self.intervals.union(lo,hi,names=('HandHeaderAdmission','DealingStartStackSidecar','StartStackLiveConnection.observe_frame','SeatForcedBinding.observe_frame'))
        ocr_backpressure=self.intervals.union(lo,hi,suffix='/queue_put_wait')
        main_path=[]
        for name in self.intervals.names():
            main_path.append(dict(stage=name,wall_union_seconds=self.intervals.union(lo,hi,names=(name,),exclude_container=True)))
        main_path.sort(key=lambda v:v['wall_union_seconds'],reverse=True)
        queues={}
        for lane,values in self.queue_depth.items():
            waits=BoundedStats();thresholds={str(t):0 for t in (.1,.5,1,5)}
            for job in self.jobs:
                if job['lane']==lane:
                    v=job['queue_wait_seconds'];waits.append(v)
                    for t in (.1,.5,1,5):thresholds[str(t)]+=int(v>t)
            queues[lane]=dict(depth_observation_distribution=distribution(values),current=values[-1] if values else None,
                queue_wait_seconds=distribution(waits),wait_threshold_counts=thresholds,
                max_active_workers=self.max_active[lane],units='batches; depth samples include operations and 5s snapshots; not time-weighted')
            submitted=self.counters[lane+'/jobs_submitted'];completed=self.counters[lane+'/jobs_completed']
            actual=sum(v.get('inference_wall_seconds',0) for v in self.ocr_compute if v['lane']==lane)
            queues[lane].update(producer_jobs_per_run_second=submitted/wall,consumer_jobs_per_run_second=completed/wall,
                actual_OCR_compute_worker_seconds=actual,
                inference_concurrency=self.inference_concurrency(lane),
                consumer_compute_jobs_per_worker_second=completed/actual if actual else None,
                producer_consumer_conclusion='Use sampled backlog growth and service/arrival windows; equal EOF totals alone do not prove spare capacity')
        report=dict(FULL_WALL=summary.get('wall_seconds',wall),RTF=summary.get('rtf'),PEAK_RSS=rss.maximum,
            AVG_CPU=100*pipeline_cpu/wall if wall else None,EFFECTIVE_CORES=pipeline_cpu/wall if wall else None,
            parent_cpu_seconds=cpu,pipeline_cpu_seconds_sampled= pipeline_cpu,
            stages=stages,queues=queues,counters=dict(self.counters),
            ocr_compute=dict(file='performance_ocr_compute.jsonl',records=len(self.ocr_compute)),callback_latencies={k:distribution(v) for k,v in self.latencies.items()},
            TOP15_WALL=sorted(stages,key=lambda x:x['wall']['total'],reverse=True)[:15],
            TOP15_CPU=sorted(stages,key=lambda x:x['attributed_cpu_seconds'],reverse=True)[:15],
            TOP15_WAIT=sorted((s for s in stages if s['kind']=='wait'),key=lambda x:x['wall']['total'],reverse=True)[:15],
            critical_path=dict(first_stage_seconds=(self.capture_bounds[1]-self.capture_bounds[0]) if self.capture_bounds else None,
                KNOWN_COMPUTE=known_compute,KNOWN_WAIT=known_wait,UNACCOUNTED_FIRST_STAGE_TIME=unaccounted,
                main_thread_stage_unions=main_path,
                compute_scope='Known measured call spans minus explicit waits; not proven CPU execution. Stage unions can overlap.',
                rule='Union of main-thread intervals inside capture; wait union wins overlaps. Containers excluded. Generator yield suspension included in parent capture.'),
            limitations=['Quantiles use bounded reservoir and are approximate after 1024 measurements; counts/sums/min/max and critical-path interval unions remain exact.',
                'Detailed intervals/jobs/inference/samples are losslessly spooled to disk; memory does not retain full event history.',
                'Interval process CPU cannot uniquely attribute concurrent native threads.',
                'Sampled child CPU can miss startup and final tails of short-lived children.',
                'REAPPEARED is not an established tracker state; no inferred count.',
                'Tracker NEW/SAME/LOST count visible object-ID set differences/intersections, including uncertain links; not certified identity transitions.',
                'CPU-bound/waiting and GIL bottleneck cannot be inferred from wall minus thread CPU.',
                'Buffer samples are approximate shallow snapshots; instrumentation itself adds overhead.',
                'Decode next() measures frame delivery, not isolated native decoder/prefetch CPU; ndarray conversion includes copying.',
                'Callback return is an availability boundary; accepted/deferred admission may occur later.',
                'I/O byte deltas are parent-process counters; file/PNG counts are surviving-file census, not create-event counts.',
                'Queue depth statistics are observation-weighted; worker lifetime activity differs from inference concurrency.'],
            io=dict(read_bytes=summary.get('disk_read_bytes'),write_bytes=summary.get('disk_write_bytes'),files_created=summary.get('files_created'),
                png_count=summary.get('png_files'),png_bytes=summary.get('png_bytes')))
        report['memory']=dict(system_used_peak_sampled=max((s['system']['used'] for s in self.samples),default=None),
            diagnostic_retained_bytes_estimate=self.retained_diagnostic_bytes(),
            diagnostic_peak_retained_sampled=max((s.get('diagnostic_retained_bytes_estimate',0) for s in self.samples),default=0),
            note='Bounded reservoirs, in-flight jobs and interval batch only; spool/SQLite sorting on disk. Other parser buffers are outside this memory bound.',
            rss_first=rss_first,rss_last=rss_last,
            sampled_monotonic_growth=monotonic if len(rss)>1 else None,
            largest_sampled_drop=largest_drop if len(rss)>1 else None,phases=phases)
        report['scenarios']=dict(current_architecture='Measured baseline; no optimization applied',
            dedicated_START_HAND_worker=dict(synchronous_observer_wall_union=observer_cost,
                theoretical_max_saving_seconds=observer_cost,bound='Upper bound only; shared receipt gates/order and added queue costs can eliminate savings'),
            per_table_ordered_workers=dict(table_work='See T01..T09 stage metrics; overlap bound requires disjoint main-thread intervals',
                serial='decode, frame conversion, output chronology, shared hand/session gates'),
            more_OCR_workers=dict(theoretical_capture_max_saving_seconds=ocr_backpressure,
                bound='Only explicit main-thread queue backpressure; response/final drains need separate union. No speedup asserted'))
        wall_rank=[s for s in report['TOP15_WALL'] if not s['stage'].startswith('T')]
        cpu_rank=[s for s in report['TOP15_CPU'] if not s['stage'].startswith('T')]
        report['verdict_answers']=dict(
            largest_wall=main_path[0]['stage'] if main_path else 'UNMEASURED',
            largest_inclusive_wall=wall_rank[0]['stage'] if wall_rank else 'UNMEASURED',
            largest_measured_CPU=cpu_rank[0]['stage'] if cpu_rank else 'UNMEASURED',
            CPU_rank_scope=cpu_rank[0]['cpu_scope'] if cpu_rank else 'UNMEASURED',
            waits=[s['stage'] for s in report['TOP15_WAIT'][:5]],
            more_core_blocker='Synchronous main-thread observer/tracker intervals plus ordered shared state; native and child CPU are separate, GIL cause unproven',
            next_experiment=('Isolated ordered START/HAND-worker replay with exact source/availability parity' if observer_cost>ocr_backpressure
                             else 'Bounded OCR-worker-count A/B with identical batches/source/order'),
            scope='Measured run only; residual critical-path time may require narrower timers before an optimization decision')
        try:
            report['work_counts']=self.work.finish()
        except Exception as exc:
            report['work_counts']=dict(status='COUNTER_AGGREGATION_FAILED',error=repr(exc))
            try:self.work.events.close()
            except Exception:pass
        (self.output/'performance_diagnostics.json').write_text(json.dumps(report,indent=2),'utf-8')
        for records in (self.jobs,self.samples,self.ocr_compute):
            if isinstance(records,DiskRecords):records.close()
        self.intervals.close()
        self.write_report(report)
    def write_report(self,r):
        lines=['# Opt-in performance evidence',f"FULL WALL: {r['FULL_WALL']} s",f"RTF: {r['RTF']}",f"PEAK RSS sampled: {r['PEAK_RSS']} bytes",f"AVG CPU sampled: {r['AVG_CPU']}%",f"EFFECTIVE CORES sampled: {r['EFFECTIVE_CORES']}",
               '','Inclusive stage sums overlap. Critical path uses interval unions on the main capture thread.',
               '',json.dumps(r['critical_path'],indent=2)]
        for ranking in ('TOP15_WALL','TOP15_CPU','TOP15_WAIT'):
            lines += ['',ranking+' (inclusive; overlaps are not additive)',
                      '| Stage | Calls | Wall s | CPU s | CPU scope |', '|---|---:|---:|---:|---|']
            for stage in r[ranking]:
                lines.append(f"| {stage['stage']} | {stage['wall']['calls']} | {stage['wall']['total']:.6f} | {stage['attributed_cpu_seconds']:.6f} | {stage['cpu_scope']} |")
        lines += ['', '| Bottleneck | Evidence | Current cost | Theoretical max saving | Safe optimization direction | Risk |',
                  '|---|---|---|---|---|---|']
        for stage in r['TOP15_WALL']:
            if stage['stage'].startswith('T') or stage['kind']=='container':continue
            lines.append(f"| {stage['stage']} | measured calls={stage['wall']['calls']}; CPU={stage['attributed_cpu_seconds']:.3f}s ({stage['cpu_scope']}) | inclusive {stage['wall']['total']:.3f}s | at most its critical-path union; overlapping wall is not additive | bounded isolated replay with source parity | output ordering, lock ownership, instrumentation overhead |")
        lines += ['', 'Scenarios (analytical only):',
            '- Current architecture: measured critical path, queues and per-table stages are the baseline.',
            '- 1 decoder + dedicated START/HAND worker: only measured synchronous observer compute is a candidate for overlap; decode/conversion and ordered hand gates remain serial. No saving claimed without callback/availability parity.',
            '- 1 decoder + per-table ordered workers: table work may overlap where receipts and shared state permit; decoder/conversion remain serial. Native-thread oversubscription and memory retention need an isolated experiment.',
            '- More OCR workers: useful only with sustained queue wait/backpressure and idle CPU capacity. Header/stack single-worker ordering and synchronous callbacks remain serial.',
            '', 'Verdict evidence: largest wall/CPU/wait stages are ranked in performance_diagnostics.json. More-core blocker and next experiment remain conditional on queue/CPU/critical-path coverage; no GIL or speedup claim follows from these timers alone.',
            '', json.dumps(r['verdict_answers'],indent=2),
            '', 'Limitations:', *['- '+x for x in r['limitations']]]
        work=r.get('work_counts',{})
        lines += ['', 'Work counters (inclusive; repeated keys do not prove removable work):',
                  'Full results: performance_work_counts.json; source events: performance_work_events.jsonl.']
        if work.get('status') == 'COUNTER_AGGREGATION_FAILED':
            lines.append('Counter aggregation failed: '+str(work.get('error')))
        else:
            lines += ['','| Operation | Calls | Unique keys | Repeated keys | Unknown keys | Wall s | Thread CPU s |',
                      '|---|---:|---:|---:|---:|---:|---:|']
            for op in sorted(work.get('operations',[]),key=lambda v:v['wall_seconds'],reverse=True)[:30]:
                lines.append(f"| {op['operation']} | {op['calls']} | {op['unique_request_keys']} | {op['repeated_request_keys']} | {op['unknown_identity_calls']} | {op['wall_seconds']:.6f} | {op['thread_cpu_seconds']:.6f} |")
            lines += ['', 'Actual cache counters: '+json.dumps(work.get('actual_cache_counters',{})),
                      'Counter self-cost: '+json.dumps(work.get('instrumentation',{})),
                      'Measurement errors: '+json.dumps(work.get('measurement_errors',{}))]
        (self.output/'PERFORMANCE_REPORT.md').write_text('\n'.join(lines),'utf-8')

    def inference_concurrency(self,lane):
        import sqlite3
        db=sqlite3.connect(self.output/'performance_concurrency.sqlite')
        db.execute('pragma temp_store=FILE')
        db.execute('create table if not exists events(point real,delta integer)')
        db.execute('delete from events')
        events=[]
        for item in self.ocr_compute:
            if item['lane']!=lane:continue
            a=item.get('compute_start_monotonic_seconds');b=item.get('compute_end_monotonic_seconds')
            if a is not None and b is not None and b>=a:events.extend(((a,1),(b,-1)))
            if len(events)>=2048:
                db.executemany('insert into events values(?,?)',events);events.clear()
        db.executemany('insert into events values(?,?)',events);db.commit()
        active=maximum=0;occupied=weighted=0;previous=None
        for point,delta in db.execute('select point,delta from events order by point,delta'):
            if previous is not None:
                dt=point-previous;weighted+=active*dt
                if active:occupied+=dt
            active+=delta;maximum=max(maximum,active);previous=point
        db.close()
        return dict(max=maximum,mean_while_any_inference_active=weighted/occupied if occupied else None,
                    inference_wall_union_seconds=occupied,scope='Actual child inference monotonic intervals; available replies only')


def enable(output,runner_globals=None):
    global ACTIVE
    ACTIVE=Diagnostics(output);ACTIVE.install(runner_globals);return ACTIVE
