"""One native call per proof. Original set traversal supplied as a schedule.

All geometry, candidates, row argmax, sorting, competitors and gates run in C++.
PhysicalChatAdmission, row IDs, receipts and consumers remain original Python.
"""
import ctypes as C,json
import numpy as np
import native_bridge
from native_bridge import P,exp
class Span(C.Structure):_fields_=[('a',C.c_int64),('b',C.c_int64)]
class Trace(C.Structure):_fields_=[('a',exp.bridge.View),('b',exp.bridge.View),('score',C.c_double)]
class Result(C.Structure):
    _fields_=[('json',C.c_void_p),('bytes',C.c_uint64),('trace',C.POINTER(Trace)),('requests',C.c_uint64),('misses',C.c_uint64),('supports',C.c_uint64),('inner_batches',C.c_uint64),('code',C.c_int)]

def module(threads=1,trace=False):
    lib=native_bridge.native(threads,'wholeproof.dll')
    f=lib.ncc_proof_gil
    f.argtypes=[exp.bridge.View,exp.bridge.View,C.POINTER(Span),C.c_uint64,C.POINTER(Span),C.c_uint64,C.POINTER(C.c_int64),C.c_uint64,C.c_double,C.c_double,C.c_double,C.c_int]
    f.restype=C.POINTER(Result)
    m=exp.study.plain_module('OLD');m.all_scores=[];m.proof_counts=[];m.runtime_lib=lib
    def view(a):return exp.bridge.View(a.ctypes.data,*a.shape,*a.strides)
    def key(v):return int(C.cast(v.data,C.c_void_p).value),(v.rows,v.cols),(v.row_stride,v.col_stride),np.dtype(np.uint8).str
    def proof(old,new,old_spans,new_spans):
        if old.shape!=new.shape or not old_spans or not new_spans:
            return dict(status='UNKNOWN',reason='NO_RETAINED_ROW_SUPPORT',candidates=[])
        if old.dtype!=np.uint8 or new.dtype!=np.uint8 or old.ndim!=2 or new.ndim!=2:raise TypeError('Isolated fullproof requires 2D uint8 authoritative pixels')
        for a,b in [*old_spans,*new_spans]:
            if not 0<=a<=b<=len(old):raise ValueError('Unsupported malformed span')
        os=(Span*len(old_spans))(*(Span(*s) for s in old_spans));ns=(Span*len(new_spans))(*(Span(*s) for s in new_spans))
        shifts={0}
        for a,b in old_spans:
            for c,d in new_spans[:2]:
                delta=round((a+b-c-d)/2);shifts.update(range(max(0,delta-1),min(len(new),delta+2)))
        traversal=(C.c_int64*len(shifts))(*shifts)
        pointer=f(view(old),view(new),os,len(os),ns,len(ns),traversal,len(traversal),m.GEOMETRY_TOLERANCE_PX,m.NCC_SCORE_MIN,m.NCC_MARGIN_MIN,int(trace))
        if not pointer:raise RuntimeError('Native fullproof missing GIL binding')
        r=pointer.contents
        if r.code:raise RuntimeError('Native fullproof error '+str(r.code))
        result=json.loads(C.string_at(r.json,r.bytes))
        if trace:
            for i in range(r.requests):t=r.trace[i];m.all_scores.append((key(t.a),key(t.b),t.score))
            m.proof_counts.append(dict(requests=r.requests,misses=r.misses,supports=r.supports,inner_batches=r.inner_batches))
        return result
    m.physical_tail_proof=proof
    return m
