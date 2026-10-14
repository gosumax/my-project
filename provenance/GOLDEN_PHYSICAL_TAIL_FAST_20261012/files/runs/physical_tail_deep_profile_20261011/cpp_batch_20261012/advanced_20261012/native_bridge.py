"""Explicit CPython GIL release inside native batch; immutable array pointers."""
import ctypes as C
import pathlib
import sys
P=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent))
import semantic_experiment as exp
import bulk_bridge

class Runtime(C.Structure):
    _fields_=[('calls',C.c_uint64),('native_wall_s',C.c_double),('gil_release_s',C.c_double),('gil_restore_s',C.c_double)]

def native(threads=1,dll='native_threads.dll'):
    # PyDLL holds GIL upon entry; native wrapper explicitly SaveThread/RestoreThread.
    lib=C.PyDLL(str(P/dll))
    lib.ncc_create.argtypes=[C.c_uint64];lib.ncc_create.restype=C.c_void_p
    lib.ncc_destroy.argtypes=[C.c_void_p]
    lib.ncc_batch.argtypes=[C.c_void_p,C.POINTER(exp.bridge.Task),C.c_uint64,C.POINTER(C.c_double)];lib.ncc_batch.restype=C.c_int
    lib.ncc_counters.argtypes=[C.c_void_p,C.POINTER(exp.bridge.Counts)]
    lib.ncc_config_threads.argtypes=[C.c_int]
    lib.ncc_config_threads.restype=C.c_int
    lib.ncc_test_fail_thread.argtypes=[C.c_int];lib.ncc_test_fail_thread.restype=None
    lib.ncc_bind_gil.argtypes=[C.c_void_p,C.c_void_p]
    lib.ncc_runtime_get.argtypes=[C.POINTER(Runtime)]
    lib.ncc_gil_pause.argtypes=[C.c_int]
    lib.ncc_runtime_reset.argtypes=[]
    for name in ('ncc_destroy','ncc_counters','ncc_bind_gil','ncc_runtime_get','ncc_runtime_reset','ncc_gil_pause'):
        getattr(lib,name).restype=None
    lib.ncc_bind_gil(C.cast(C.pythonapi.PyEval_SaveThread,C.c_void_p),C.cast(C.pythonapi.PyEval_RestoreThread,C.c_void_p))
    if lib.ncc_config_threads(threads):raise RuntimeError('Native pool configuration failed')
    return lib

def module(threads=1,trace=False):
    lib=native(threads);m=bulk_bridge.cpp_module(lib,trace=trace);m.runtime_lib=lib
    return m

def metrics(lib):
    r=Runtime();lib.ncc_runtime_get(C.byref(r));return {name:getattr(r,name) for name,_ in r._fields_}
