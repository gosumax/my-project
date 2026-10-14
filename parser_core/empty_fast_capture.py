"""Owner-bound capture routing for the exact frozen EMPTY dispatcher.

Classification math lives in the byte-identical off_empty_* runtime bundle.
This module owns eligibility, lifetime and the unmodified native fallback.
"""
from __future__ import annotations
import json
import threading
from collections import Counter
from fractions import Fraction
from types import MethodType
import numpy as np
from .empty_seat_evidence import EmptySeatButtonModel,button_roi


def install_empty_capture(connection,*,enabled=None):
    """OFF leaves original model methods untouched; override is for proof/tests."""
    if enabled is None:
        config=json.loads((connection.root/'configs/empty_fast_path.json').read_text('utf-8'))
        if config.get('schema')!='poker.empty-capture-fast-path.v1' or type(config.get('enabled')) is not bool:
            raise ValueError('EMPTY_FAST_CONFIGURATION_UNKNOWN')
        enabled=config['enabled']
    if type(enabled) is not bool:raise ValueError('EMPTY_FAST_ENABLED_NOT_BOOLEAN')
    if not enabled or connection.empty_button_model is None:return None
    return EmptyCaptureHook(connection)


class EmptyCaptureHook:
    def __init__(self,connection):
        # Lazy import: disabled routing neither constructs nor checks fast state.
        from off_empty_dispatch import Dispatcher,ORIGINAL_CLASSIFY,ORIGINAL_CAPTURE
        self.owner=connection;self.model=connection.empty_button_model
        if type(self.model) is not EmptySeatButtonModel:
            raise ValueError('EMPTY_FAST_MODEL_IMPLEMENTATION_UNKNOWN')
        if ('capture' in vars(self.model) or 'classify' in vars(self.model) or
            EmptySeatButtonModel.capture is not ORIGINAL_CAPTURE or EmptySeatButtonModel.classify is not ORIGINAL_CLASSIFY):
            raise ValueError('EMPTY_FAST_NATIVE_METHOD_CONFLICT')
        self.original_classify=ORIGINAL_CLASSIFY;self.original_capture=ORIGINAL_CAPTURE
        self.dispatcher=Dispatcher('NEW');self.local=threading.local()
        self.contexts={};self.counts=Counter();self.closed=False
        hook=self
        def capture(model,rgb,source,boxes):return hook.capture(model,rgb,source,boxes)
        def classify(model,gray):return hook.classify(model,gray)
        self.capture_method=MethodType(capture,self.model);self.classify_method=MethodType(classify,self.model)
        self.model.capture=self.capture_method;self.model.classify=self.classify_method

    def _invalidate(self,table=None):
        if type(table) is not int:table=None
        if table is None:
            self.contexts.clear();self.dispatcher.entries.clear();self.dispatcher.history.clear()
        else:
            self.contexts.pop(table,None)
            for slot in list(self.dispatcher.entries):
                if slot[0]==table:self.dispatcher.entries.pop(slot)
            active={entry.binding for entry in self.dispatcher.entries.values()}
            self.dispatcher.history={b:values[-2:] for b,values in self.dispatcher.history.items() if b in active}
        self.counts['invalidations']+=1

    def _context(self,table,session):
        """Read actual owner state. Missing/malformed fast metadata is ineligible."""
        sidecar=self.owner.sidecar
        try:
            if self.closed or self.owner.closed or self.owner.empty_button_model is not self.model:return None
            if type(table) is not int or not 1<=table<=9 or type(session) is not str or not session:return None
            if not sidecar.lock._is_owned() or not self.owner.lock._is_owned():return None
            header=sidecar.current.get(table)
            if not isinstance(header,dict) or header['session_id']!=session:return None
            if sidecar.active_sessions.get(table)!=session:return None
            hand=header['hand_id']
            if type(hand) is not str or not hand.isdigit():return None
            key=(table,session,hand);window=sidecar.windows.get(key)
            if window is None or window.deal is not None or window.closed is not None:return None
            if self.owner.states.get(key,{}).get('deal_frame') is not None:return None
            fence=sidecar.header_fences.get((table,session),{})
            if not isinstance(fence,dict) or fence.get('unknown',False) or fence.get('hand_id')!=hand:return None
            frame=header['source_frame'];position=header['source_position'];pts=header['source_pts'];base=header['time_base']
            pixel=header.get('header_pixel_sha256')
            budget=sidecar.flags['capture_frames']
            if (type(frame) is not int or frame<0 or type(pts) is not int or pts<0 or
                type(budget) is not int or budget<1 or not isinstance(position,(list,tuple)) or
                len(position)!=2 or any(type(v) is not int for v in position) or position[0]!=frame or
                not isinstance(base,(list,tuple)) or len(base)!=2 or any(type(v) is not int or v<=0 for v in base)):
                return None
            if (fence.get('source_position')!=list(position) or type(pixel) is not str or len(pixel)!=64 or
                any(c not in '0123456789abcdefABCDEF' for c in pixel)):return None
            # Header/window objects remain strongly held while a cache may use
            # their nonce, preventing id reuse for an ABA window replacement.
            signature=(key,id(window),frame,tuple(position),pts,tuple(base),pixel,budget)
            return signature,window,header
        except (KeyError,TypeError,AttributeError,ValueError,IndexError):return None

    def sync(self,table,session,frame):
        """Retire context before the existing observe/replay path, not authority."""
        context=self._context(table,session)
        if type(table) is not int:self._invalidate();return
        old=self.contexts.get(table)
        if context is None or old is not None and old['signature']!=context[0]:self._invalidate(table)
        if old is not None and (type(frame) is not int or frame<old['last_frame']):self._invalidate(table)

    def _eligible(self,model,rgb,source,boxes):
        try:
            if model is not self.model or not isinstance(source,dict):return None
            table=source['table_id'];session=source['session_id'];context=self._context(table,session)
            if context is None:return None
            signature,window,header=context;key=signature[0]
            if source['hand_id']!=key[2]:return None
            frame=source['source_frame'];pts=source['source_pts'];base=source['time_base']
            if (type(frame) is not int or type(pts) is not int or pts<0 or
                not isinstance(base,(list,tuple)) or len(base)!=2 or any(type(v) is not int or v<=0 for v in base)):
                return None
            if not header['source_frame']<=frame<=header['source_frame']+signature[-1]:return None
            stamp=Fraction(pts*base[0],base[1]);floor=Fraction(header['source_pts']*header['time_base'][0],header['time_base'][1])
            if stamp<floor:return None
            if (not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.shape!=(404,847,3)):
                return None
            native=self.model.slots[f'slot_{table:02d}']['seats']
            if boxes!=native or not isinstance(boxes,dict):return None
            sidecar=self.owner.sidecar;source_key=(table,session)
            if any(f<=frame and index>=1000000 for f,index in sidecar.outstanding.get(source_key,())):return None
            gate=sidecar.hand_state_gate
            if gate is not None and any(tuple(header['source_position'])<tuple(p) and p[0]<=frame
                for p in gate.failed_headers.get(source_key,())):return None
            old=self.contexts.get(table)
            if old is not None and (old['signature']!=signature or frame<old['last_frame'] or stamp<old['last_stamp']):
                self._invalidate(table)
                if frame<old['last_frame'] or stamp<old['last_stamp']:return None
            return table,signature,window,header,frame,stamp
        except (KeyError,TypeError,AttributeError,ValueError,IndexError,ZeroDivisionError):return None

    def capture(self,model,rgb,source,boxes):
        if not self.closed and (self.model.capture is not self.capture_method or self.model.classify is not self.classify_method):
            raise ValueError('EMPTY_FAST_OWNER_METHOD_MUTATED')
        if getattr(self.local,'capture',None) is not None:raise ValueError('EMPTY_FAST_NESTED_CAPTURE')
        eligible=self._eligible(model,rgb,source,boxes)
        if eligible is None:
            self._invalidate(source.get('table_id') if isinstance(source,dict) else None)
            self.counts['old_capture_batches']+=1
            return self.original_capture(model,rgb,source,boxes)
        table,signature,window,header,frame,stamp=eligible
        self.contexts[table]=dict(signature=signature,window=window,header=header,last_frame=frame,last_stamp=stamp)
        self.local.capture=(dict(source),iter(boxes),rgb.shape,boxes,signature)
        self.counts['fast_capture_batches']+=1
        try:return self.original_capture(model,rgb,source,boxes)
        finally:
            self.local.capture=None
            # Golden's A/B/A census uses only the preceding two digests. Bound
            # diagnostic retention without changing _decide or its arithmetic.
            active={entry.binding for entry in self.dispatcher.entries.values()}
            self.dispatcher.history={b:values[-2:] for b,values in self.dispatcher.history.items() if b in active}

    def classify(self,model,gray):
        context=getattr(self.local,'capture',None)
        if context is None:
            self.counts['old_non_capture_calls']+=1
            return self.original_classify(model,gray)
        source,seats,shape,boxes,signature=context;seat=next(seats)
        return self.dispatcher.decide(model,gray,dict(source,seat=seat,
            geometry=button_roi(boxes[seat],shape[1],shape[0]),window_context=signature))

    def close(self):
        if self.closed:return
        if self.model.capture is not self.capture_method or self.model.classify is not self.classify_method:
            raise ValueError('EMPTY_FAST_OWNER_METHOD_MUTATED')
        del self.model.capture;del self.model.classify
        self._invalidate();self.dispatcher.contracts.clear();self.dispatcher.model_refs.clear()
        self.dispatcher.model_hash_bindings.clear();self.dispatcher.bound.prepared.clear();self.dispatcher.bound_contract.clear()
        self.closed=True
