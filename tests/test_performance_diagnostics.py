import json
from types import SimpleNamespace
import numpy as np
from parser_core import performance_diagnostics as diagnostics
from parser_core.parallel_ocr import ParallelRowOCR
from parser_core.ram_first_stage import RamRow

def test_critical_path_union_excludes_double_counting():
    assert diagnostics.union_seconds([(0,4),(1,2),(3,5)],0,6)==5
    assert diagnostics.union_seconds([(0,4),(1,2)],2,3)==1

class FakeClient:
    def read_batch(self,items,batch_size):
        return dict(results=[dict(item_id=i['item_id'],status='OK',raw_text='Dealer: a checks',score=.99) for i in items])
    def close(self):pass

def row():
    return RamRow(10,1,1,'s',2,3,'NEW_CANDIDATE',1,np.zeros((12,20),dtype=np.uint8),'a'*64,'id')

def test_disabled_diagnostics_never_requested_or_recorded(tmp_path):
    assert diagnostics.ACTIVE is None
    ocr=ParallelRowOCR(lambda _:FakeClient(),workers=1)
    assert ocr.diagnostics is None
    ocr.submit(row());ocr.finish()
    assert not list(tmp_path.iterdir()) and ocr.results['1:10:2']['raw_text']=='Dealer: a checks'

def test_opt_in_queue_instrumentation_preserves_results(tmp_path):
    profile=diagnostics.Diagnostics(tmp_path);diagnostics.ACTIVE=profile
    profile.start_sampler({'phase':'test'})
    try:
        callbacks=[]
        ocr=ParallelRowOCR(lambda _:FakeClient(),workers=1,result_observer=lambda r,v:callbacks.append(v),diagnostic_lane='header')
        ocr.submit(row());ocr.finish()
        assert not ocr.errors and callbacks==[ocr.results['1:10:2']]
        profile.finish(dict(wall_seconds=1,rtf=1))
        report=json.loads((tmp_path/'performance_diagnostics.json').read_text())
        assert report['counters']['header/jobs_completed']==1
        assert report['queues']['header']['max_active_workers']==1
        assert report['queues']['header']['queue_wait_seconds']['calls']==1
        assert diagnostics.ACTIVE is None
    finally:
        diagnostics.ACTIVE=None;profile.stop.set()

def test_enabled_wrapper_preserves_return_exception_and_restores(tmp_path):
    owner=SimpleNamespace(f=lambda x:x+1);original=owner.f
    profile=diagnostics.Diagnostics(tmp_path);profile.wrap(owner,'f','pure')
    assert owner.f(4)==5 and profile.metrics['pure']['wall']
    profile.start_sampler({'phase':'test'});profile.finish({})
    assert owner.f is original

def test_actual_child_CPU_dominates_ranking_and_diagnostic_memory_exposed(tmp_path):
    profile=diagnostics.Diagnostics(tmp_path);profile.start_sampler({'phase':'test'})
    profile.ocr_compute.append(dict(lane='stack',worker=0,batch_size=8,
        inference_wall_seconds=10,inference_process_cpu_seconds=50))
    profile.finish({})
    report=json.loads((tmp_path/'performance_diagnostics.json').read_text())
    assert report['TOP15_CPU'][0]['stage']=='stack/OCR_actual_inference'
    assert report['TOP15_CPU'][0]['cpu_scope']=='OCR_CHILD_PROCESS_INFERENCE'
    assert report['memory']['diagnostic_retained_bytes_estimate']>0

def test_inference_concurrency_uses_actual_intervals_not_worker_lifetime(tmp_path):
    profile=diagnostics.Diagnostics(tmp_path)
    profile.ocr_compute=[dict(lane='header',compute_start_monotonic_seconds=a,compute_end_monotonic_seconds=b)
                         for a,b in ((1,3),(2,4),(4,5))]
    measured=profile.inference_concurrency('header')
    assert measured['max']==2
    assert measured['inference_wall_union_seconds']==4
    assert measured['mean_while_any_inference_active']==1.25

def test_tracker_actual_candidates_and_results_are_unchanged_when_enabled(tmp_path):
    from parser_core.chat_tracker import coordinate_map, zone_scores
    profile=diagnostics.Diagnostics(tmp_path)
    old=[(10,20),(30,40),(50,60)];new=[(10,20),(30,40)]
    expected=coordinate_map(old,new)
    try:
        diagnostics.ACTIVE=profile
        assert coordinate_map(old,new)==expected
        assert profile.counters['tracker_identity_candidates']==5
    finally:diagnostics.ACTIVE=None
