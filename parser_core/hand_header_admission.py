"""Source-bound header-only OCR lane, independent of CHAT birth admission.

Pixel prefixes are candidates, never hand IDs. Only the existing strict OCR
header grammar can feed current_hand_id. Ordinary CHAT output is untouched.
"""
from __future__ import annotations
import base64
import hashlib
import json
from collections import Counter, OrderedDict
from pathlib import Path
import threading
import cv2
import numpy as np
from parser_core.chat_rows import physical_spans, padded_row_span
from parser_core.dealing_start_stack import HEADER
from parser_core.ram_first_stage import RamRow


class HeaderPrefixGate:
    def __init__(self, root):
        root=Path(root)
        self.config=json.loads((root/'models/hand_header_prefix/model.json').read_text('utf-8'))
        self.templates={};self.cache=OrderedDict()
        for name,meta in self.config['templates'].items():
            data=(root/meta['path']).read_bytes()
            if hashlib.sha256(data).hexdigest()!=meta['sha256']:
                raise ValueError('HEADER_TEMPLATE_HASH_MISMATCH')
            self.templates[name]=cv2.imdecode(np.frombuffer(data,np.uint8),0)

    def classify(self, pixels):
        key=(pixels.shape,hashlib.sha256(pixels[:,:-12].tobytes()).hexdigest())
        if key in self.cache:
            self.cache.move_to_end(key);return dict(self.cache[key])
        result={'candidate':False,'prefix':None,'scores':{}}
        if pixels.ndim==2 and pixels.shape[0]==16 and pixels.shape[1]>=120:
            for name,template in self.templates.items():
                # Fixed left origin; a hash inside a nickname/body is excluded.
                left,right=self.config['search_x'][name]
                score=float(cv2.minMaxLoc(cv2.matchTemplate(pixels[:,left:right],template,cv2.TM_CCOEFF_NORMED))[1])
                result['scores'][name]=score
            best=max(result['scores'],key=result['scores'].get)
            if result['scores'][best]>=self.config['threshold']:
                result.update(candidate=True,prefix=best)
        self.cache[key]=dict(result)
        if len(self.cache)>4096:self.cache.popitem(last=False)
        return result

    @staticmethod
    def content_key(pixels):
        # Ignore scrollbar and padding/vertical translation only. Digit pixels
        # remain in this identity; no nickname/hand-number labels participate.
        mask=(pixels[:,:-12]>70).astype(np.uint8)
        yy,xx=np.nonzero(mask)
        if not len(xx):return None
        body=np.ascontiguousarray(mask[yy.min():yy.max()+1,xx.min():xx.max()+1])
        return (body.shape,hashlib.sha256(body.tobytes()).hexdigest())


class HandHeaderAdmission:
    def __init__(self, root, output, provider, ocr):
        self.gate=HeaderPrefixGate(root);self.provider=provider;self.ocr=ocr
        self.lock=threading.RLock();self.counts=Counter();self.seen={};self.rows=[]
        self._seen_counts={};self._seen_count_source=self.seen;self._seen_count_size=0
        self.closed=False
        self.stream=(Path(output)/'hand_header_ocr.jsonl').open('w',encoding='utf-8')

    def _log(self, value):
        self.stream.write(json.dumps(value,ensure_ascii=False)+'\n')

    def _session_seen_count(self, table, session):
        # Preserve the original query on first use and after fixture replacement.
        # Ordinary ownership only appends; other sessions' counts stay unchanged.
        if (self.seen is not self._seen_count_source or
                len(self.seen) != self._seen_count_size):
            self._seen_counts={};self._seen_count_source=self.seen
            self._seen_count_size=len(self.seen)
        pair=(table,session)
        if pair not in self._seen_counts:
            self._seen_counts[pair]=sum(k[0]==table and k[1]==session for k in self.seen)
        return self._seen_counts[pair]

    def observe_frame(self, table, session, frame, pts, time_base, table_rgb, chat_gray):
        # Install replacement-session context before a fast worker can reply.
        # Do not release candidates here: this frame's header jobs are not yet
        # registered, and the normal observer still owns source-frame sealing.
        with self.provider.lock:
            previous=self.provider.active_sessions.get(table)
            if previous is not None and previous!=session:
                self.provider.current.pop(table,None)
                self.provider.counts['SESSION_RESET']+=1
            self.provider.active_sessions[table]=session
        spans=physical_spans(chat_gray);batch=[]
        for index in range(len(spans)):
            a,b=padded_row_span(spans,index,len(chat_gray))
            pixels=np.ascontiguousarray(chat_gray[a:b]);gate=self.gate.classify(pixels)
            self.counts['ROWS_CHECKED']+=1
            if not gate['candidate']:continue
            self.counts['HEADER_CANDIDATE_OBSERVATIONS']+=1
            content=self.gate.content_key(pixels);key=(table,session,content)
            with self.lock:
                if key in self.seen:
                    self.counts['REPEATED_CANDIDATE_SUPPRESSED']+=1;continue
                if self._session_seen_count(table,session)>=4096:
                    raise RuntimeError('HEADER_SESSION_CAPACITY_UNKNOWN')
                self.seen[key]=(frame,index)
                self._seen_counts[(table,session)]+=1;self._seen_count_size=len(self.seen)
                digest=hashlib.sha256(pixels.tobytes()).hexdigest()
                row=RamRow(frame,None,table,session,index,a,'HEADER_ONLY_CANDIDATE',0,
                           pixels.copy(),digest,f'{session}:HEADER:{frame}:{index}',pts,*time_base)
                self.rows.append(row);self.counts['ADMITTED']+=1
                ok,packed=cv2.imencode('.png',pixels)
                if not ok:raise ValueError('HEADER_RAM_ENCODING_FAILED')
                self._log(dict(table_id=table,session_id=session,source_frame=frame,source_pts=pts,
                               time_base=time_base,source_span_index=index,row_bounds=[a,b],
                               pixel_sha256=digest,gate=gate,status='ADMITTED',
                               png_base64=base64.b64encode(packed).decode('ascii')))
                # Job-qualified token prevents the ordinary OCR callback for
                # this same physical row from clearing header-lane coverage.
                with self.provider.lock:
                    self.provider.outstanding[(table,session)].add((frame,1000000+index))
                batch.append(row)
        # Dispatch immediately, never wait for an eight-row tail to bind a hand.
        for offset in range(0,len(batch),8):
            self.counts['OCR_ATTEMPTED']+=len(batch[offset:offset+8])
            self.ocr.submit_batch(batch[offset:offset+8])

    def ocr_complete(self,row,response):
        with self.lock, self.provider.lock:
            match=HEADER.fullmatch((response.get('raw_text') or '').strip())
            confirmed=response.get('status')=='OK' and match is not None
            trace=getattr(self.provider,'trace_order',lambda *a,**k:None)
            trace('HEADER_CALLBACK_ENTER',row.table_id,row.session_id,row.frame_id,
                lane='HEADER_ONLY',source_pts=row.source_pts,source_span_index=row.source_span_index,
                ocr_status=response.get('status'),raw_text=response.get('raw_text'))
            if confirmed:
                status=self.provider.header(row.table_id,row.session_id,match.group(1),
                    row.frame_id,row.source_pts,row.source_span_index,
                    [row.time_base_num,row.time_base_den],row.pixel_sha256)
            else:
                status='HEADER_OCR_UNKNOWN'
                self.provider.source_barriers[(row.table_id,row.session_id)].add((row.frame_id,row.source_span_index))
                self.provider.counts['OCR_HAND_CONTEXT_BARRIER']+=1
                if self.provider.hand_state_gate is not None:self.provider.hand_state_gate.header_unknown(row)
            self.provider.outstanding[(row.table_id,row.session_id)].discard((row.frame_id,1000000+row.source_span_index))
            self.provider._release(row.table_id,row.session_id)
            trace('HEADER_CALLBACK_RETURN',row.table_id,row.session_id,row.frame_id,
                lane='HEADER_ONLY',status=status)
            self.counts[status]+=1
            self._log(dict(table_id=row.table_id,session_id=row.session_id,
                           source_frame=row.frame_id,source_pts=row.source_pts,
                           time_base=[row.time_base_num,row.time_base_den],
                           source_span_index=row.source_span_index,pixel_sha256=row.pixel_sha256,
                           status=status,ocr_response=response,
                           current_hand_id=self.provider.current.get(row.table_id,{}).get('hand_id')))

    def finish(self):
        with self.lock:
            if self.closed:return
            self._log({'status':'SUMMARY','counts':dict(self.counts)})
            self.stream.close();self.closed=True
