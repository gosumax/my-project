"""Sealed PROOF4 ABI binding; no experimental imports or native rebuild.

Candidate only. All native configuration/calls/result copies share one lock.
The DLL owns a non-reentrant process-global pool and thread-local results.
"""
import ctypes as C
import hashlib
import json
import platform
import threading
from pathlib import Path

import numpy as np
from parser_core import physical_chat_admission as old

DLL_SHA256='c6ba8e3ebace49deb97cabb7d462341ca6efabbd76d0379a5d92885899ec061b'
OLD_PROOF=old.physical_tail_proof
_LOCK=threading.Lock()
_RUNTIME=None
_IDENTITY=None


class View(C.Structure):
    _fields_=[('data',C.c_void_p),('rows',C.c_int64),('cols',C.c_int64),
              ('row_stride',C.c_int64),('col_stride',C.c_int64)]


class Span(C.Structure):
    _fields_=[('a',C.c_int64),('b',C.c_int64)]


class Trace(C.Structure):
    _fields_=[('a',View),('b',View),('score',C.c_double)]


class Result(C.Structure):
    _fields_=[('json',C.c_void_p),('bytes',C.c_uint64),('trace',C.POINTER(Trace)),
              ('requests',C.c_uint64),('misses',C.c_uint64),('supports',C.c_uint64),
              ('inner_batches',C.c_uint64),('code',C.c_int)]


def _runtime_unlocked(path,threads):
    """Caller must hold the sole process lock, including initialization."""
    global _RUNTIME,_IDENTITY
    identity=(str(Path(path).resolve()),threads)
    if _RUNTIME is not None:
        if _IDENTITY!=identity:raise RuntimeError('PROOF4_PROCESS_RECONFIGURATION_FORBIDDEN')
        return _RUNTIME
    if platform.system()!='Windows' or platform.python_implementation()!='CPython' or C.sizeof(C.c_void_p)!=8:
        raise RuntimeError('PROOF4_REQUIRES_WINDOWS_X64_CPYTHON')
    core=getattr(np,'_core',None) or np.core
    if not core._multiarray_umath.__cpu_features__.get('AVX2',False):
        raise RuntimeError('PROOF4_AVX2_OS_STATE_UNAVAILABLE')
    if threads not in (1,4):raise ValueError('PROOF4_THREADS_MUST_BE_4_OR_ISOLATED_1_CONTROL')
    if (C.sizeof(View),C.sizeof(Span),C.sizeof(Trace),C.sizeof(Result))!=(40,16,88,64):
        raise RuntimeError('PROOF4_ABI_SIZE_CONFLICT')
    if (View.rows.offset,Span.b.offset,Trace.score.offset,Result.code.offset)!=(8,8,80,56):
        raise RuntimeError('PROOF4_ABI_OFFSET_CONFLICT')
    with Path(path).open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=DLL_SHA256:
            raise RuntimeError('PROOF4_DLL_SEAL_CONFLICT')
    lib=C.PyDLL(str(Path(path).resolve()))
    lib.ncc_bind_gil.argtypes=[C.c_void_p,C.c_void_p];lib.ncc_bind_gil.restype=None
    lib.ncc_config_threads.argtypes=[C.c_int];lib.ncc_config_threads.restype=C.c_int
    lib.ncc_proof_gil.argtypes=[View,View,C.POINTER(Span),C.c_uint64,C.POINTER(Span),C.c_uint64,
        C.POINTER(C.c_int64),C.c_uint64,C.c_double,C.c_double,C.c_double,C.c_int]
    lib.ncc_proof_gil.restype=C.POINTER(Result)
    lib.ncc_bind_gil(C.cast(C.pythonapi.PyEval_SaveThread,C.c_void_p),
                     C.cast(C.pythonapi.PyEval_RestoreThread,C.c_void_p))
    if lib.ncc_config_threads(threads):raise RuntimeError('PROOF4_POOL_CONFIGURATION_FAILED')
    _IDENTITY=identity;_RUNTIME=lib
    return lib


def supported(a,b,old_spans,new_spans):
    if not all(isinstance(x,np.ndarray) and x.dtype==np.uint8 and x.ndim==2 for x in (a,b)):
        return False
    if a.shape!=b.shape or not all(0<n<=4096 for n in a.shape):return False
    for spans in (old_spans,new_spans):
        if not isinstance(spans,(list,tuple)) or not 0<len(spans)<=a.shape[0]:return False
        previous=0
        for span in spans:
            if not isinstance(span,(list,tuple)) or len(span)!=2:return False
            lo,hi=span
            if not all(isinstance(v,(int,np.integer)) and not isinstance(v,(bool,np.bool_)) for v in span):return False
            if not previous<=lo<hi<=a.shape[0]:return False
            previous=hi
    return True


class Proof4Binding:
    def __init__(self,path,*,threads=4,trace=False):
        self.path=Path(path);self.threads=threads;self.trace=trace
        self.calls=0;self.fallbacks=0;self.last_trace=[];self.last_counts={}
        with _LOCK:_runtime_unlocked(self.path,threads)

    def __call__(self,a,b,old_spans,new_spans):
        try:
            frozen_old=tuple(tuple(span) for span in old_spans)
            frozen_new=tuple(tuple(span) for span in new_spans)
        except TypeError:
            self.fallbacks+=1
            return OLD_PROOF(a,b,old_spans,new_spans)
        if not supported(a,b,frozen_old,frozen_new):
            self.fallbacks+=1
            return OLD_PROOF(a,b,old_spans,new_spans)
        old_spans=tuple((int(x),int(y)) for x,y in frozen_old)
        new_spans=tuple((int(x),int(y)) for x,y in frozen_new)
        # Owned immutable-for-this-call buffers survive GIL release. No cross-call cache.
        before=np.array(a,dtype=np.uint8,order='C',copy=True)
        after=np.array(b,dtype=np.uint8,order='C',copy=True)
        if not supported(before,after,old_spans,new_spans):
            raise RuntimeError('PROOF4_INPUT_CHANGED_DURING_SNAPSHOT')
        os=(Span*len(old_spans))(*(Span(int(x),int(y)) for x,y in old_spans))
        ns=(Span*len(new_spans))(*(Span(int(x),int(y)) for x,y in new_spans))
        shifts={0}
        for x,y in old_spans:
            for c,d in new_spans[:2]:
                delta=round((x+y-c-d)/2)
                shifts.update(range(max(0,delta-1),min(len(after),delta+2)))
        traversal=(C.c_int64*len(shifts))(*shifts)
        def view(x):return View(x.ctypes.data,*x.shape,*x.strides)
        def traced_view(v):
            ptr=int(v.data)
            for name,x in (('old',before),('new',after)):
                offset=ptr-x.ctypes.data
                if 0<=offset<x.nbytes:
                    return (name,offset,(v.rows,v.cols),(v.row_stride,v.col_stride))
            raise RuntimeError('PROOF4_TRACE_POINTER_OUTSIDE_INPUT')
        with _LOCK:
            lib=_runtime_unlocked(self.path,self.threads)
            pointer=lib.ncc_proof_gil(view(before),view(after),os,len(os),ns,len(ns),traversal,len(traversal),
                old.GEOMETRY_TOLERANCE_PX,old.NCC_SCORE_MIN,old.NCC_MARGIN_MIN,int(self.trace))
            if not pointer:raise RuntimeError('PROOF4_NULL_RESULT')
            result=pointer.contents
            if result.code:raise RuntimeError('PROOF4_NATIVE_ERROR:'+str(result.code))
            if not result.json or not 0<result.bytes<=16*1024*1024:raise RuntimeError('PROOF4_RESULT_SIZE_CONFLICT')
            output=json.loads(C.string_at(result.json,result.bytes))
            self.last_trace=[]
            if self.trace:
                if result.requests>10_000_000:raise RuntimeError('PROOF4_TRACE_SIZE_CONFLICT')
                for i in range(result.requests):
                    item=result.trace[i]
                    self.last_trace.append((traced_view(item.a),traced_view(item.b),float(item.score)))
            self.last_counts={name:int(getattr(result,name)) for name in ('requests','misses','supports','inner_batches')}
            self.calls+=1
            return output
