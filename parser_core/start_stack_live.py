"""Live wiring of frozen Smoke14 card NCC and existing Tiny/window APIs.

All configured positive seats or an independent complete occupancy mask are
required. Epoch and forced-bet proof are not invented.
"""
import base64
import hashlib
import json
import threading
from collections import Counter
from fractions import Fraction
from pathlib import Path
from parser_core import performance_diagnostics as diagnostics

import cv2
import numpy as np

from parser_core.dealing_start_stack import NUMERIC
from parser_core.parallel_ocr import ParallelRowOCR, row_key
from parser_core.ram_first_stage import RamRow, png_compatible_gray
from parser_core.tiny_chat_client import TinyChatClient


def first_visible_stack_triple(readings, confidence, final=False, start_frame=None):
    """First numeric source fixes the only attempt; pending earlier jobs fence it."""
    ordered = sorted(readings.items())
    first = None
    expected = start_frame
    for frame, r in ordered:
        if expected is not None and frame != expected:
            return dict(first_numeric_visible_frame=None,frames=[expected,expected+1,expected+2],
                ocr_values=[],displayed_stack=None,status='UNKNOWN',blocker='FIRST_NUMERIC_SOURCE_COVERAGE_UNKNOWN',evidence=[])
        expected = frame + 1
        if r is None:
            return None
        text = (r.get('raw_text') or '').strip()
        if (r.get('ocr_status') == 'OK' and NUMERIC.fullmatch(text)
                and r.get('confidence') is not None and r['confidence'] >= confidence):
            first = frame
            break
    if first is None:
        return None
    frames = list(range(first, first + 3))
    if any(f in readings and readings[f] is None for f in frames):
        return None
    if not final and any(f not in readings and f > max(readings) for f in frames):
        return None
    evidence = [readings.get(f) or {} for f in frames]
    values = [(r.get('raw_text') or '').strip().replace(' ', '').replace('\u00a0', '') for r in evidence]
    valid = all(r.get('ocr_status') == 'OK' and NUMERIC.fullmatch((r.get('raw_text') or '').strip())
                and r.get('confidence') is not None and r['confidence'] >= confidence for r in evidence)
    stable = valid and len(set(values)) == 1
    return dict(first_numeric_visible_frame=first, frames=frames, ocr_values=values,
                displayed_stack=values[0] if stable else None,
                clean_window_rule='FIRST_VISIBLE_STABLE_STACK_TRIPLE_AFTER_DEAL_COMPLETE',
                status='PASS' if stable else 'UNKNOWN',
                blocker=None if stable else ('FIRST_NUMERIC_TRIPLE_SOURCE_GAP' if any(f not in readings for f in frames) else 'FIRST_NUMERIC_TRIPLE_NOT_STABLE'), evidence=evidence)


class StartStackLiveConnection:
    def __init__(self, root, output, sidecar, ocr_config):
        self.root = Path(root)
        self.sidecar = sidecar
        self.output = Path(output)
        self.lock = threading.RLock()
        self.counts = Counter()
        self.states = {}
        self.occupancy_provider = None
        self.pending_tables = {}
        self.pending_stacks = {}
        # Saved Smoke17/19: max consecutive pending chain 34 sources, not
        # merely max individual OCR latency (24). 48 includes 14-source margin.
        # Nine-slot conservative table+stack raw ROI bound: 31,173,120 bytes.
        self.pending_buffer_frames = 48
        self.jobs = {}
        self.closed = False
        model = json.loads((self.root / 'models/start_stack_table_cards.json').read_text('utf-8'))
        self.threshold = model['threshold']
        self.consecutive = model['consecutive_frames']
        self.templates = {}
        for r in model['templates']:
            p = np.frombuffer(base64.b64decode(r['pixels_base64']), np.uint8).reshape(r['template_shape'])
            if hashlib.sha256(p.tobytes()).hexdigest() != r['template_bgr_sha256']:
                raise ValueError('TABLE_CARD_TEMPLATE_HASH_CONFLICT')
            self.templates[(r['slot'], r['seat'])] = p
        self.log = (self.output / 'start_stack_live_receipts.jsonl').open('w', encoding='utf-8')
        def factory(worker):
            return TinyChatClient(Path(ocr_config['python']), Path(ocr_config['model_dir']),
                ocr_config['model_name'], ocr_config['weights_sha256'], ocr_config['device'],
                self.output / f'tiny_start_stack_{worker}.stderr.log')
        self.ocr = ParallelRowOCR(factory, workers=1, queue_size=32, batch_size=8,
                                  result_observer=self.completed)

    def write(self, receipt):
        self.log.write(json.dumps(receipt, ensure_ascii=False) + '\n')

    def _trace(self,event,table,session,frame,**details):
        trace=getattr(self.sidecar,'trace_order',None)
        if trace is not None:trace(event,table,session,frame,**details)

    def observe_frame(self, table, session, frame, pts, time_base, rgb, chat):
        # Bounded pending-header buffers retain only fixed TABLE-card or
        # stack ROIs. OCR waits for identity proof; source time stays immutable.
        with self.sidecar.lock, self.lock:
            batch = self._replay_pending(table, session, frame)
            batch.extend(self._observe_frame(table, session, frame, pts, time_base, rgb, chat) or [])
        if batch:
            # Never block on a full queue while holding callback/provider locks.
            for start in range(0, len(batch), 8):
                self.ocr.submit_batch(batch[start:start+8])

    def _replay_pending(self, table, session, frame):
        h = self.sidecar.current.get(table)
        key = (table, session, h['hand_id']) if h and h['session_id'] == session else None
        for buffers in (self.pending_tables, self.pending_stacks):
            for old in list(buffers):
                if ((old[:2] == (table, session) and old != key)
                        or (old[0] == table and old[1] != session)):
                    buffers.pop(old)
                    self.counts['PENDING_CONTEXT_DISCARDED'] += 1
        batch = []
        pending = self.pending_tables.get(key)
        unresolved = any(f <= frame and i >= 1000000
                         for f,i in self.sidecar.outstanding[(table, session)])
        if pending is not None or key in self.pending_stacks:
            self._trace('REPLAY_CHECK',table,session,frame,unresolved=unresolved,
                table_sources=[s[0][2] for s in pending['sources']] if pending else [],
                stack_sources=[s[0][2] for s in self.pending_stacks.get(key,[])])
        if pending is not None and not unresolved:
            self.pending_tables.pop(key)
            state = self.states[key]
            state['card_run'], state['last'] = pending['checkpoint']
            for source, evidence in pending['sources']:
                meta=dict(buffer_kind='TABLE',replay_capture_frame=frame)
                self._trace('REPLAY_SOURCE_ENTER',*source[:3],source_pts=source[3],**meta)
                if state['deal_frame'] is None:
                    self._observe_frame(*source, table_evidence=evidence,_trace_replay=meta,_retained_source=(key,source,evidence))
                else:
                    batch.extend(self._observe_frame(*source, stack_evidence=evidence['stacks'],_trace_replay=meta,_retained_source=(key,source,evidence['stacks'])) or [])
            self.counts['TABLE_PENDING_REPLAYED'] += len(pending['sources'])
        pending_stack = self.pending_stacks.get(key)
        if pending_stack is not None and not unresolved:
            self.pending_stacks.pop(key)
            for source, evidence in pending_stack:
                meta=dict(buffer_kind='STACK',replay_capture_frame=frame)
                self._trace('REPLAY_SOURCE_ENTER',*source[:3],source_pts=source[3],**meta)
                rows=self._observe_frame(*source, stack_evidence=evidence,_trace_replay=meta,_retained_source=(key,source,evidence)) or []
                batch.extend(rows)
                self._trace('REPLAY_SOURCE_RETURN',*source[:3],ocr_jobs=len(rows),**meta)
            self.counts['STACK_PENDING_REPLAYED'] += len(pending_stack)
        return batch

    def _observe_frame(self, table, session, frame, pts, time_base, rgb, chat, table_evidence=None, stack_evidence=None,_trace_replay=None,_retained_source=None):
        # Lock order matches all existing header and ordinary OCR callbacks.
        with self.sidecar.lock, self.lock:
            h = self.sidecar.current.get(table)
            if not h or h['session_id'] != session:
                return
            key = (table, session, h['hand_id'])
            if key not in self.sidecar.windows:
                return
            w = self.sidecar.windows[key]
            state = self.states.setdefault(key, {'card_run': 0, 'last': None,
                'deal_frame': None, 'crops': {}, 'submitted': Counter(),
                'numeric_runs': {}, 'numeric_triples': {}, 'stop': set(), 'source_readings': {}, 'canonical_attempts': {}})
            source_key = (table, session)
            fence = self.sidecar.header_fences.get(source_key, {})
            # Only _replay_pending supplies this exact retained source/evidence.
            # Latest same-ID OCR remains current; historical replay uses the
            # immutable confirmed anchor, never a general stale-frame bypass.
            replay_anchor=None
            if (_retained_source is not None and _retained_source[0]==key
                    and _retained_source[1][:5]==(table,session,frame,pts,tuple(time_base))
                    and rgb is None and chat is None
                    and _retained_source[2] is (table_evidence if table_evidence is not None else stack_evidence)
                    and self.sidecar.active_sessions.get(table)==session
                    and fence.get('hand_id')==key[2] and not fence.get('unknown',False)):
                anchors=[a for a in self.sidecar.histories[source_key]
                    if a['hand_id']==key[2] and a['session_id']==session]
                if anchors:
                    candidate=anchors[0]
                    conflicting=any(a['hand_id']!=key[2] and candidate['source_frame']<=a['source_frame']<=frame
                        for a in self.sidecar.histories[source_key])
                    if not conflicting and candidate.get('source_pts') is not None and candidate.get('time_base') and pts is not None:
                        replay_anchor=candidate
            floor=replay_anchor or h
            gate = self.sidecar.hand_state_gate
            pending_header = any(f <= frame and index >= 1000000
                for f,index in self.sidecar.outstanding[source_key])
            unknown_header = gate is not None and any(
                tuple(floor['source_position']) < tuple(p) and p[0] <= frame
                for p in gate.failed_headers[source_key])
            source_conflict = (fence.get('unknown', False) or frame < floor['source_frame']
                or (_retained_source is not None and replay_anchor is None))
            if floor.get('source_pts') is not None and floor.get('time_base') and pts is not None:
                source_conflict |= Fraction(pts * time_base[0], time_base[1]) < Fraction(floor['source_pts'] * floor['time_base'][0], floor['time_base'][1])
            if _trace_replay is not None:
                reasons=[]
                if pending_header:reasons.append('PENDING_HEADER')
                if unknown_header:reasons.append('UNKNOWN_HEADER')
                if fence.get('unknown',False):reasons.append('HEADER_FENCE_UNKNOWN')
                if _retained_source is not None and replay_anchor is None:reasons.append('RETAINED_CONTEXT_UNPROVEN')
                if frame<floor['source_frame']:reasons.append('SOURCE_FRAME_BEFORE_REPLAY_ANCHOR' if replay_anchor else 'SOURCE_FRAME_BEFORE_CURRENT')
                if floor.get('source_pts') is not None and floor.get('time_base') and pts is not None and Fraction(pts*time_base[0],time_base[1])<Fraction(floor['source_pts']*floor['time_base'][0],floor['time_base'][1]):
                    reasons.append('SOURCE_PTS_BEFORE_REPLAY_ANCHOR' if replay_anchor else 'SOURCE_PTS_BEFORE_CURRENT')
                if state['deal_frame'] is None and w.closed is not None and frame>=w.closed:reasons.append('WINDOW_CLOSED_BEFORE_DEAL')
                self._trace('REPLAY_GATES',table,session,frame,source_pts=pts,time_base=time_base,
                    reasons=reasons,pending_header=pending_header,unknown_header=unknown_header,
                    source_conflict=source_conflict,deal_frame=state['deal_frame'],window_closed=w.closed,
                    header_fence=dict(fence),retained_same_hand_replay=replay_anchor is not None,
                    admission_anchor=dict(floor),**_trace_replay)
            if pending_header or unknown_header or source_conflict or (state['deal_frame'] is None and w.closed is not None and frame >= w.closed):
                if (pending_header and state['deal_frame'] is None and not unknown_header
                        and not source_conflict and (w.closed is None or frame < w.closed)):
                    diagnostic_tick=diagnostics.ACTIVE.tick() if diagnostics.ACTIVE else None
                    boxes = self.sidecar.profile['slots'][f'slot_{table:02d}']['seats']
                    pending = self.pending_tables.setdefault(key, dict(
                        checkpoint=(state['card_run'], state['last']), sources=[]))
                    evidence = dict(table_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), cards={}, stacks={})
                    for seat,b in boxes.items():
                        x,y,X,Y=b['shown_cards']
                        evidence['cards'][seat]=np.ascontiguousarray(rgb[y:Y,x:X]).copy()
                        x,y,X,Y=b['stack']
                        evidence['stacks'][seat]=png_compatible_gray(np.ascontiguousarray(rgb[y:Y,x:X])).copy()
                    pending['sources'].append(((table, session, frame, pts, tuple(time_base), None, None), evidence))
                    if len(pending['sources']) > self.pending_buffer_frames:
                        pending['sources'].pop(0)
                        pending['checkpoint'] = (0, None)
                        self.counts['TABLE_PENDING_BUFFER_EXPIRED'] += 1
                    self._buffer_metrics()
                    self.write(dict(table_id=table, session_id=session, hand_id=h['hand_id'],
                        source_frame=frame, source_pts=pts, time_base=time_base,
                        stage='TABLE_PENDING_HEADER', status='WITHHELD',
                        table_pixel_sha256=evidence['table_pixel_sha256']))
                    if diagnostics.ACTIVE:diagnostics.ACTIVE.record('pending_buffer_write',diagnostic_tick)
                elif (pending_header and state['deal_frame'] is not None and not unknown_header
                        and not source_conflict and frame <= state['deal_frame']+self.sidecar.flags['capture_frames']):
                    diagnostic_tick=diagnostics.ACTIVE.tick() if diagnostics.ACTIVE else None
                    boxes = self.sidecar.profile['slots'][f'slot_{table:02d}']['seats']
                    crops = {}
                    for seat,b in boxes.items():
                        if seat in state['stop']:continue
                        x,y,X,Y=b['stack']
                        crops[seat]=png_compatible_gray(np.ascontiguousarray(rgb[y:Y,x:X])).copy()
                    pending = self.pending_stacks.setdefault(key, [])
                    pending.append(((table, session, frame, pts, tuple(time_base), None, None), crops))
                    if len(pending)>self.pending_buffer_frames:
                        pending.pop(0)
                        self.counts['STACK_PENDING_BUFFER_EXPIRED'] += 1
                    self._buffer_metrics()
                    self.counts['STACK_PENDING_RAM_FRAMES'] += 1
                    self.write(dict(table_id=table, session_id=session, hand_id=h['hand_id'],
                        source_frame=frame, source_pts=pts, time_base=time_base,
                        stage='STACK_PENDING_HEADER', status='WITHHELD',
                        crop_hashes={s:hashlib.sha256(p.tobytes()).hexdigest() for s,p in crops.items()}))
                    if diagnostics.ACTIVE:diagnostics.ACTIVE.record('pending_buffer_write',diagnostic_tick)
                else:
                    if _trace_replay is not None:
                        self._trace('REPLAY_REJECT',table,session,frame,reasons=reasons,**_trace_replay)
                    self.pending_tables.pop(key, None)
                    self.pending_stacks.pop(key, None)
                state['card_run'] = 0
                state['last'] = None
                self.counts['HAND_OR_WINDOW_SOURCE_UNKNOWN'] += 1
                return
            source = dict(table_id=table, session_id=session, hand_id=h['hand_id'],
                          source_frame=frame, source_pts=pts, time_base=time_base)
            boxes = self.sidecar.profile['slots'][f'slot_{table:02d}']['seats']
            if state['deal_frame'] is None:
                scores = {}
                for seat, b in boxes.items():
                    x,y,X,Y = b['shown_cards']
                    card_rgb = table_evidence['cards'][seat] if table_evidence is not None else np.ascontiguousarray(rgb[y:Y,x:X])
                    crop = cv2.cvtColor(card_rgb, cv2.COLOR_RGB2BGR)
                    template = self.templates.get((table, seat))
                    scores[seat] = (float(cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)[0,0])
                                    if template is not None else None)
                provider=getattr(self,'occupancy_provider',None)
                occupancy=provider(key,frame,list(boxes)) if provider is not None and table_evidence is None else None
                occupied=occupancy['occupied_seats'] if occupancy is not None else list(boxes)
                support=(tuple((s,occupancy['seats'][s]['state'],occupancy['seats'][s]['actor'],
                                occupancy['seats'][s]['current_pixel_sha256']) for s in boxes)
                         if occupancy is not None else ('ALL_CONFIGURED_SEATS',))
                all_present = bool(occupied) and all(scores[s] is not None and scores[s] >= self.threshold for s in occupied)
                same_support=state.get('card_support',support)==support
                state['card_run'] = ((state['card_run'] + 1) if state['last'] == frame - 1 and same_support else 1) if all_present else 0
                state['card_support']=support
                state['last'] = frame
                self.counts['TABLE_FRAMES_CHECKED'] += 1
                self.write(dict(source, stage='TABLE_CARD_NCC', scores=scores,
                    consecutive_matches=state['card_run'], status='CANDIDATE' if all_present else 'UNKNOWN',
                    occupancy_mask_source='EXACT_LABEL_CONTINUITY' if occupancy is not None else 'ALL_CONFIGURED_SEATS',
                    occupied_seats=occupied,
                    blocker=None if all_present else 'COMPLETE_OCCUPANCY_MASK_UNPROVEN'))
                if state['card_run'] < self.consecutive:
                    return
                receipt = dict(source, evidence_status='PROVEN',
                    table_pixel_sha256=table_evidence['table_pixel_sha256'] if table_evidence is not None else hashlib.sha256(rgb.tobytes()).hexdigest(),
                    occupied_seats=occupied, card_stable_frames=list(range(frame-2,frame+1)),
                    proof=('COMPLETE_INDEPENDENT_OCCUPANCY_AND_POSITIVE_FROZEN_CARD_NCC' if occupancy is not None
                           else 'ALL_CONFIGURED_SEATS_POSITIVE_FROZEN_CARD_NCC'))
                if occupancy is not None:receipt['occupancy_receipt']=occupancy
                card_pixels = {}
                for seat,b in boxes.items():
                    x,y,X,Y = b['shown_cards']
                    crop = table_evidence['cards'][seat] if table_evidence is not None else np.ascontiguousarray(rgb[y:Y,x:X])
                    ok,packed = cv2.imencode('.png',cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
                    if not ok:raise ValueError('TABLE_CARD_RAM_ENCODING_FAILED')
                    card_pixels[seat] = {'roi':b['shown_cards'],
                        'pixel_sha256':hashlib.sha256(crop.tobytes()).hexdigest(),
                        'png_base64':base64.b64encode(packed).decode('ascii')}
                receipt['card_pixels'] = card_pixels
                status = self.sidecar.confirm_deal_complete(key, receipt)
                self.write(dict(receipt, stage='DEAL_COMPLETE', status=status))
                if status != 'DEAL_COMPLETE':
                    return
                state['deal_frame'] = frame
                self.counts['DEAL_COMPLETE'] += 1
                if occupancy is not None:self.counts['INDEPENDENT_OCCUPANCY_DEAL_COMPLETE'] += 1
                # Numeric capture begins strictly after confirmed TABLE completion.
                return
            # Evidence budget is not a window opener or a stack proof.
            if frame > state['deal_frame'] + self.sidecar.flags['capture_frames']:
                self.counts['STACK_CAPTURE_BUDGET_EXHAUSTED_FRAMES'] += 1
                return
            batch = []
            for i, (seat,b) in enumerate(boxes.items()):
                if seat not in w.deal['occupied_seats']:continue
                if seat in state['stop']:
                    continue
                x,y,X,Y = b['stack']
                pixels = stack_evidence.get(seat) if stack_evidence is not None else png_compatible_gray(np.ascontiguousarray(rgb[y:Y,x:X])).copy()
                if pixels is None:continue
                sha = hashlib.sha256(pixels.tobytes()).hexdigest()
                # Submit actual consecutive sources until the first attempt resolves.
                state['source_readings'].setdefault(seat, {})[frame] = None
                row = RamRow(frame, None, table, session, i, y, 'STACK_ONLY', 0,
                    pixels, sha, f'{session}:STACK:{h["hand_id"]}:{seat}:{frame}', pts, *time_base)
                self.jobs[row_key(row)] = (key, seat, dict(source, seat=seat, pixel_sha256=sha,
                    roi=b['stack'], occupancy_epoch=None, occupancy_proven=False,
                    forced_bets=None, forced_bets_proven=False))
                batch.append(row)
            if batch:
                # Existing bounded OCR queue. It is a separate lane and cannot
                # alter ordinary CHAT admission, header jobs or marker state.
                self.counts['STACK_OCR_ATTEMPTED'] += len(batch)
            return batch

    def completed(self, row, response):
        with self.sidecar.lock, self.lock:
            job = self.jobs.pop(row_key(row), None)
            if job is None:
                self.counts['DUPLICATE_STACK_CALLBACK'] += 1
                return
            key, seat, receipt = job
            ok,packed = cv2.imencode('.png', row.pixels)
            if not ok:raise ValueError('STACK_RAM_ENCODING_FAILED')
            receipt['png_base64'] = base64.b64encode(packed).decode('ascii')
            receipt['ocr_response'] = dict(response)
            receipt.update(raw_text=response.get('raw_text'), confidence=response.get('score'),
                           ocr_status=response.get('status'))
            later_headers = [h for h in self.sidecar.histories[key[:2]]
                if h['hand_id'] != key[2] and h['source_frame'] <= row.frame_id
                and h['source_frame'] >= self.sidecar.windows[key].deal['source_frame']]
            if self.sidecar.active_sessions.get(row.table_id) != row.session_id or later_headers:
                self.write(dict(receipt, stage='STACK_OCR', status='SOURCE_CONTEXT_UNKNOWN', result=None))
                self.counts['STACK_SOURCE_CONTEXT_UNKNOWN'] += 1
                return
            result, status = self.sidecar.stack_reading(key, receipt)
            receipt.update(stage='STACK_OCR', status=status, result=result,
                blocker='OCCUPANCY_EPOCH_AND_FORCED_BET_BINDING_UNPROVEN')
            self.write(receipt)
            self.counts['STACK_OCR_COMPLETED'] += 1
            state = self.states[key]
            state['source_readings'][seat][row.frame_id] = dict(receipt)
            attempt = first_visible_stack_triple(state['source_readings'][seat], self.sidecar.flags['stack_confidence'], start_frame=state['deal_frame']+1)
            if attempt is not None:
                state['canonical_attempts'][seat] = attempt
                state['stop'].add(seat)
                if attempt['status'] == 'PASS':
                    state['numeric_triples'][seat] = attempt

    def _buffer_metrics(self):
        total = sum(p.nbytes for pending in self.pending_tables.values()
                    for source,evidence in pending['sources']
                    for kind in ('cards', 'stacks') for p in evidence[kind].values())
        total += sum(p.nbytes for pending in self.pending_stacks.values()
                     for source,evidence in pending for p in evidence.values())
        self.counts['PEAK_BUFFERED_ROI_BYTES'] = max(self.counts['PEAK_BUFFERED_ROI_BYTES'], total)
        lengths = [len(p['sources']) for p in self.pending_tables.values()] + [len(p) for p in self.pending_stacks.values()]
        self.counts['PEAK_BUFFERED_SOURCES_PER_BUFFER'] = max(self.counts['PEAK_BUFFERED_SOURCES_PER_BUFFER'], max(lengths, default=0))

    def finish(self):
        if self.closed:
            return
        with self.sidecar.lock, self.lock:
            batch = []
            for table,session in {k[:2] for k in list(self.pending_tables)+list(self.pending_stacks)}:
                frames = [source[2] for key,pending in self.pending_tables.items() if key[:2]==(table,session) for source,evidence in pending['sources']]
                frames += [source[2] for key,pending in self.pending_stacks.items() if key[:2]==(table,session) for source,evidence in pending]
                batch.extend(self._replay_pending(table,session,max(frames,default=-1)))
            self.pending_tables.clear()
            self.pending_stacks.clear()
        for start in range(0,len(batch),8):
            self.ocr.submit_batch(batch[start:start+8])
        self.ocr.finish()
        with self.sidecar.lock, self.lock:
            for key,state in self.states.items():
                w = self.sidecar.windows[key]
                anchors = [h for h in self.sidecar.histories[key[:2]] if h['hand_id'] == key[2]]
                start = min((h['source_frame'] for h in anchors), default=-1)
                next_headers = [h['source_frame'] for h in self.sidecar.histories[key[:2]]
                    if h['hand_id'] != key[2] and h['source_frame'] > start]
                boundary = min(next_headers, default=None)
                for seat, readings in state['source_readings'].items():
                    attempt = first_visible_stack_triple(readings, self.sidecar.flags['stack_confidence'], final=True, start_frame=state['deal_frame']+1) if state['deal_frame'] is not None else None
                    if attempt is not None:
                        state['canonical_attempts'][seat] = attempt
                if boundary is not None:
                    for attempt in state['canonical_attempts'].values():
                        if attempt['frames'][-1] >= boundary or (state['deal_frame'] is not None and state['deal_frame'] >= boundary):
                            attempt.update(status='UNKNOWN', blocker='LATE_HEADER_SOURCE_CONTRADICTION')
                    state['numeric_triples'] = {s:r for s,r in state['numeric_triples'].items() if r['frames'][-1] < boundary}
                    if state['deal_frame'] is not None and state['deal_frame'] >= boundary:
                        self.write(dict(table_id=key[0],session_id=key[1],hand_id=key[2],
                            stage='DEAL_COMPLETE_FINAL_CONTEXT', status='UNKNOWN',
                            blocker='LATE_HEADER_SOURCE_CONTRADICTION', next_header_frame=boundary))
                        state['deal_frame'] = None
                        w.deal = None
                        w.results.clear()
            summary = {'counts':dict(self.counts), 'pending_buffer_frame_cap':self.pending_buffer_frames, 'hands':[
                dict(table_id=k[0],session_id=k[1],hand_id=k[2],deal_complete_frame=s['deal_frame'],
                     numeric_triples=s['numeric_triples'],canonical_attempts=s['canonical_attempts'],first_stack_changing_observation_frame=self.sidecar.windows[k].closed,
                     start_stack_pass=len(self.sidecar.windows[k].results)) for k,s in self.states.items()],
                'errors':self.ocr.errors,'authority':'LIVE_DIAGNOSTIC; forced bets/epoch UNPROVEN',
                'start_stack_pass':sum(len(w.results) for w in self.sidecar.windows.values())}
            (self.output/'start_stack_live_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
            self.log.close()
            self.closed = True
