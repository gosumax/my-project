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


def delay_source_guard(projection, state):
    """Existing positive visibility predicates; no OCR or numeric claim.

    Unexplained pixels restore the ordinary OCR lane. In particular a digit
    gate rejection is never sufficient to withhold a recognizer request.
    """
    from parser_core.start_stack_visibility import (ForcedOverlayPatterns,
        OVERLAY_NCC_MIN, BLANK_GRAY_RANGE_MAX)
    patch = projection['gray'][2:-2,6:-6].astype(np.float64)
    if not patch.size:return None
    gray_range = float(np.ptp(patch))
    if gray_range <= BLANK_GRAY_RANGE_MAX:
        state['blank_seen'] = True
        return dict(phase='BLANK_WITHIN_CODEC_NOISE',gray_range=gray_range)
    if state.get('blank_seen'):return None
    references = state.setdefault('references',[])
    identity = None
    if len(references) < 3:
        if 'patterns' not in state:state['patterns'] = ForcedOverlayPatterns()
        identity = state['patterns'].identify(patch)
        if identity is None:return None
        identities = state.setdefault('identities',[])
        if identities and identity['label'] != identities[0]['label']:return None
        identities.append(identity)
        references.append(patch)
    ref = np.mean(references,axis=0)
    a=ref-ref.mean();b=patch-patch.mean()
    denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    ncc=float((a*b).sum()/denominator) if denominator else None
    gain=float((a*b).sum()/(a*a).sum()) if (a*a).sum() else None
    if not (ncc is not None and ncc>=OVERLAY_NCC_MIN and gain is not None and gain>0):
        return None
    result = dict(phase='POSITIVE_SAME_FORCED_OVERLAY_OR_FADE',gray_range=gray_range,
        ncc_to_forced_overlay=ncc,brightness_gain=gain)
    if identity is not None:result['forced_overlay_identity'] = identity
    return result


def first_visible_stack_triple(readings, confidence, final=False, start_frame=None, *, allow_scaled=False,numeric_policy=None,
                               direct_ocr_delay_frames=0, single_frame=False):
    """First numeric source fixes the only attempt; pending earlier jobs fence it."""
    from parser_core.first_stack_ocr import valid_score,direct_numeric_policy
    if not valid_score(confidence):return None
    if numeric_policy is not None and (numeric_policy!=direct_numeric_policy() or
            confidence!=numeric_policy['min_confidence'] or not allow_scaled):return None
    ordered = sorted(readings.items())
    first = None
    expected = start_frame
    skipped_context = None
    skipped_time = None
    skipped_verifier = None
    skipped_guard = {}
    for frame, r in ordered:
        if single_frame and start_frame is not None and frame<start_frame:continue
        if expected is not None and frame != expected and not single_frame:
            return dict(first_numeric_visible_frame=None,frames=[expected,expected+1,expected+2],
                ocr_values=[],displayed_stack=None,status='UNKNOWN',blocker='FIRST_NUMERIC_SOURCE_COVERAGE_UNKNOWN',evidence=[])
        expected = frame + 1
        if r is None:
            return None
        if r.get('source_state') == 'OCR_SKIPPED_BY_DELAY':
            # A real source, without a recognizer result. This advances only
            # acquisition coverage; it makes no numeric/non-numeric claim.
            # The unchanged final visibility proof still checks every pixel.
            valid = (direct_ocr_delay_frames == 20 and start_frame is not None
                and start_frame <= frame < start_frame + direct_ocr_delay_frames
                and r.get('source_frame') == frame and r.get('source_only') is True
                and r.get('ocr_submitted') is False
                and r.get('direct_ocr_delay_frames') == direct_ocr_delay_frames
                and r.get('deal_complete_frame') == start_frame - 1
                and not any(k in r for k in ('ocr_response','raw_text','confidence','ocr_status')))
            context = tuple(r.get(k) for k in ('table_id','session_id','hand_id','seat'))
            valid = valid and all(v is not None for v in context)
            valid = valid and (skipped_context is None or context == skipped_context)
            try:
                stamp = Fraction(r['source_pts'] * r['time_base'][0], r['time_base'][1])
                valid = valid and (skipped_time is None or stamp > skipped_time)
            except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
                valid = False
            if valid:
                from parser_core.start_stack_visibility import VisibilitySource
                if skipped_verifier is None:skipped_verifier = VisibilitySource()
                projection = skipped_verifier.verify(r)
                proof = delay_source_guard(projection,skipped_guard) if projection is not None else None
                valid = proof is not None and ('delay_source_proof' not in r or r['delay_source_proof'] == proof)
            if not valid:
                return dict(first_numeric_visible_frame=None,frames=[frame] if single_frame else [frame,frame+1,frame+2],
                    ocr_values=[],displayed_stack=None,status='UNKNOWN',
                    blocker='DELAY_SOURCE_PROOF_INVALID',evidence=[])
            skipped_context, skipped_time = context, stamp
            continue
        text = (r.get('raw_text') or '').strip()
        if allow_scaled:
            from parser_core.first_stack_ocr import effective_numeric
            number=effective_numeric(r,confidence,allow_scaled=True,numeric_policy=numeric_policy)
        else:number=text.replace(' ','').replace('\u00a0','') if (r.get('ocr_status')=='OK' and NUMERIC.fullmatch(text) and valid_score(r.get('confidence')) and r['confidence']>=confidence) else None
        if number is not None:
            first = frame
            break
    if first is None:
        return None
    frames = [first] if single_frame else list(range(first, first + 3))
    if any(f in readings and readings[f] is None for f in frames):
        return None
    if not final and any(f not in readings and f > max(readings) for f in frames):
        return None
    evidence = [readings.get(f) or {} for f in frames]
    values = [(r.get('raw_text') or '').strip().replace(' ', '').replace('\u00a0', '') for r in evidence]
    valid = all(r.get('ocr_status') == 'OK' and NUMERIC.fullmatch((r.get('raw_text') or '').strip())
                and valid_score(r.get('confidence')) and r['confidence'] >= confidence for r in evidence)
    if allow_scaled:
        values=[effective_numeric(r,confidence,allow_scaled=True,numeric_policy=numeric_policy) for r in evidence]
        valid=all(v is not None for v in values)
    stable = valid if single_frame else valid and len(set(values)) == 1
    from .start_stack_visibility import FIRST_QUALIFIED_RULE,FIRST_QUALIFIED_CONTRACT
    result=dict(first_numeric_visible_frame=first, frames=frames, ocr_values=values,
                displayed_stack=values[0] if stable else None,
                clean_window_rule=FIRST_QUALIFIED_RULE if single_frame else 'FIRST_VISIBLE_STABLE_STACK_TRIPLE_AFTER_DEAL_COMPLETE',
                status='PASS' if stable else 'UNKNOWN',
                blocker=None if stable else ('FIRST_NUMERIC_TRIPLE_SOURCE_GAP' if any(f not in readings for f in frames) else 'FIRST_NUMERIC_TRIPLE_NOT_STABLE'), evidence=evidence)
    if single_frame:result['stack_witness_policy']=FIRST_QUALIFIED_CONTRACT
    return result


class StartStackLiveConnection:
    def __init__(self, root, output, sidecar, ocr_config, *, first_visible_contract=False,
                 empty_card_contract=False, direct_ocr_delay_frames=0):
        self.root = Path(root)
        self.sidecar = sidecar
        self.output = Path(output)
        self.lock = threading.RLock()
        self.counts = Counter()
        self.states = {}
        self.occupancy_provider = None
        self.occupancy_labels_provider = None
        self.empty_button_model = None
        self.card_back_supplement = None
        self.empty_button_sources = {}
        # Capture returns fresh rows; production consumers only read them.
        # These subtotals serve diagnostics, never source admission or decisions.
        self._empty_source_sizes = {}
        self._empty_source_counts = Counter()
        self._empty_source_encoded = 0
        self._empty_source_metrics_sources = self.empty_button_sources
        self._empty_source_metrics_valid = True
        if empty_card_contract:
            from parser_core.empty_seat_evidence import EmptySeatButtonModel
            from parser_core.table_card_back_evidence import CardBackSupplement
            self.empty_button_model=EmptySeatButtonModel(self.root)
            self.card_back_supplement=CardBackSupplement(self.root)
        self.visibility_source = None
        self.visibility_sources = {}
        self.numeric_policy = None
        if first_visible_contract:
            from parser_core.start_stack_visibility import VisibilitySource
            from parser_core.first_stack_ocr import direct_numeric_policy
            self.visibility_source = VisibilitySource()
            self.numeric_policy = direct_numeric_policy()
        self.numeric_confidence = (self.numeric_policy['min_confidence'] if self.numeric_policy is not None
                                   else self.sidecar.flags['stack_confidence'])
        if (type(direct_ocr_delay_frames) is not int or direct_ocr_delay_frames not in (0,20)
                or (direct_ocr_delay_frames and self.visibility_source is None)):
            raise ValueError('DIRECT_OCR_DELAY_POLICY_INVALID')
        self.direct_ocr_delay_frames = direct_ocr_delay_frames
        self.pending_tables = {}
        self.pending_stacks = {}
        self._pending_roi_metrics = {}
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
            client=TinyChatClient(Path(ocr_config['python']), Path(ocr_config['model_dir']),
                ocr_config['model_name'], ocr_config['weights_sha256'], ocr_config['device'],
                self.output / f'tiny_start_stack_{worker}.stderr.log')
            if first_visible_contract:
                from parser_core.first_stack_ocr import FirstStackOCR
                return FirstStackOCR(client,model_name=ocr_config['model_name'],weights_sha256=ocr_config['weights_sha256'],
                    confidence=self.sidecar.flags['stack_confidence'],worker=worker)
            return client
        self.ocr = ParallelRowOCR(factory, workers=1, queue_size=32, batch_size=8,
                                  result_observer=self.completed)
        self.empty_fast_hook = None
        if self.empty_button_model is not None:
            from parser_core.empty_fast_capture import install_empty_capture
            self.empty_fast_hook = install_empty_capture(self)

    def write(self, receipt):
        self.log.write(json.dumps(receipt, ensure_ascii=False) + '\n')

    def _trace(self,event,table,session,frame,**details):
        trace=getattr(self.sidecar,'trace_order',None)
        if trace is not None:trace(event,table,session,frame,**details)

    def observe_frame(self, table, session, frame, pts, time_base, rgb, chat):
        # Bounded pending-header buffers retain only fixed TABLE-card or
        # stack ROIs. OCR waits for identity proof; source time stays immutable.
        with self.sidecar.lock, self.lock:
            if getattr(self,'empty_fast_hook',None) is not None:
                self.empty_fast_hook.sync(table, session, frame)
            if getattr(self,'empty_button_model',None) is not None:
                metrics_current = self.empty_button_sources is getattr(self, '_empty_source_metrics_sources', None)
                captured_key = None
                h=self.sidecar.current.get(table)
                key=(table,session,h['hand_id']) if h and h['session_id']==session else None
                state=self.states.get(key,{})
                if (key in self.sidecar.windows and state.get('deal_frame') is None and
                        h['source_frame']<=frame<=h['source_frame']+self.sidecar.flags['capture_frames']):
                    boxes=self.sidecar.profile['slots'][f'slot_{table:02d}']['seats']
                    self.empty_button_sources[(key,frame)]=self.empty_button_model.capture(rgb,
                        dict(table_id=table,session_id=session,hand_id=key[2],source_frame=frame,
                             source_pts=pts,time_base=list(time_base)),boxes)
                    captured_key = (key, frame)
                self.empty_button_sources={k:v for k,v in self.empty_button_sources.items()
                    if k[0][0]!=table or k[0]==key and k[1]>frame-self.pending_buffer_frames}
                self._empty_source_peaks(table, captured_key, metrics_current)
            if self.visibility_source is not None:
                h = self.sidecar.current.get(table)
                key = (table,session,h['hand_id']) if h and h['session_id']==session else None
                state=self.states.get(key,{})
                capture_start=state.get('deal_frame') or (h or {}).get('source_frame',-1)
                if key in self.sidecar.windows and frame<=capture_start+self.sidecar.flags['capture_frames']:
                    boxes=self.sidecar.profile['slots'][f'slot_{table:02d}']['seats']
                    for seat,b in boxes.items():
                        if seat in state.get('stop',set()):continue
                        self.visibility_sources[(key,frame,seat)]=self.visibility_source.capture(rgb,table,seat,b['stack'])
                self.visibility_sources={k:v for k,v in self.visibility_sources.items()
                    if k[0][0]!=table or k[0]==key and k[1]>=frame-self.pending_buffer_frames}
            batch = self._replay_pending(table, session, frame)
            batch.extend(self._observe_frame(table, session, frame, pts, time_base, rgb, chat) or [])
        if batch:
            # Never block on a full queue while holding callback/provider locks.
            for start in range(0, len(batch), 8):
                self.ocr.submit_batch(batch[start:start+8])

    def _empty_source_peaks(self, table, captured_key, metrics_current):
        """Sample the same post-prune peaks, using only changed row subtotals.

        An external fixture map replacement or a previous size-evaluation error
        rebuilds from the retained map. Frame peak still precedes encoded-size
        errors, and a capture discarded by pruning is never evaluated.
        """
        sources = self.empty_button_sources
        valid = metrics_current and getattr(self, '_empty_source_metrics_valid', False)
        self._empty_source_metrics_sources = sources
        self._empty_source_metrics_valid = False
        if valid:
            sizes = self._empty_source_sizes
            counts = self._empty_source_counts
            removed = [k for k in sizes if k not in sources]
            for k in removed:
                counts[k[0][0]] -= 1
            if captured_key in sources and captured_key not in sizes:
                counts[captured_key[0][0]] += 1
            self.counts['PEAK_EMPTY_SOURCE_FRAMES_PER_TABLE'] = max(
                self.counts['PEAK_EMPTY_SOURCE_FRAMES_PER_TABLE'], counts[table])
            encoded = self._empty_source_encoded
            for k in removed:
                encoded -= sizes.pop(k)
            if captured_key in sources:
                subtotal = sum(len(r.get('png_base64', '')) for r in sources[captured_key].values())
                encoded += subtotal - sizes.get(captured_key, 0)
                sizes[captured_key] = subtotal
        else:
            self.counts['PEAK_EMPTY_SOURCE_FRAMES_PER_TABLE'] = max(
                self.counts['PEAK_EMPTY_SOURCE_FRAMES_PER_TABLE'],
                sum(k[0][0] == table for k in sources))
            sizes = {}
            counts = Counter()
            encoded = 0
            for k, rows in sources.items():
                subtotal = sum(len(r.get('png_base64', '')) for r in rows.values())
                sizes[k] = subtotal
                counts[k[0][0]] += 1
                encoded += subtotal
            self._empty_source_sizes = sizes
            self._empty_source_counts = counts
        self._empty_source_encoded = encoded
        self.counts['PEAK_EMPTY_SOURCE_ENCODED_BYTES'] = max(
            self.counts['PEAK_EMPTY_SOURCE_ENCODED_BYTES'], encoded)
        self._empty_source_metrics_valid = True

    def _replay_pending(self, table, session, frame):
        h = self.sidecar.current.get(table)
        key = (table, session, h['hand_id']) if h and h['session_id'] == session else None
        for buffers in (self.pending_tables, self.pending_stacks):
            for old in list(buffers):
                if ((old[:2] == (table, session) and old != key)
                        or (old[0] == table and old[1] != session)):
                    buffers.pop(old)
                    self._forget_pending_metric('TABLE' if buffers is self.pending_tables else 'STACK', old)
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
            self._forget_pending_metric('TABLE', key)
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
            self._forget_pending_metric('STACK', key)
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
                'numeric_runs': {}, 'numeric_triples': {}, 'stop': set(), 'source_readings': {}, 'canonical_attempts': {},
                'source_states': {},'delayed_rows': {},'delay_guards': {},'delay_disabled': set()})
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
                    if getattr(self,'empty_button_model',None) is not None:
                        evidence['empty_button_observations']=self.empty_button_sources.get((key,frame))
                        provider=getattr(self,'occupancy_labels_provider',None)
                        evidence['occupancy_labels']=(provider(key,frame,list(boxes)) if provider else {})
                    for seat,b in boxes.items():
                        x,y,X,Y=b['shown_cards']
                        evidence['cards'][seat]=np.ascontiguousarray(rgb[y:Y,x:X]).copy()
                        x,y,X,Y=b['stack']
                        evidence['stacks'][seat]=png_compatible_gray(np.ascontiguousarray(rgb[y:Y,x:X])).copy()
                    added = ((table, session, frame, pts, tuple(time_base), None, None), evidence)
                    pending['sources'].append(added)
                    removed = None
                    if len(pending['sources']) > self.pending_buffer_frames:
                        removed = pending['sources'].pop(0)
                        pending['checkpoint'] = (0, None)
                        self.counts['TABLE_PENDING_BUFFER_EXPIRED'] += 1
                    self._buffer_metrics(('TABLE', key, added, removed))
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
                    added = ((table, session, frame, pts, tuple(time_base), None, None), crops)
                    pending.append(added)
                    removed = None
                    if len(pending)>self.pending_buffer_frames:
                        removed = pending.pop(0)
                        self.counts['STACK_PENDING_BUFFER_EXPIRED'] += 1
                    self._buffer_metrics(('STACK', key, added, removed))
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
                    self._forget_pending_metric('TABLE', key)
                    self.pending_stacks.pop(key, None)
                    self._forget_pending_metric('STACK', key)
                state['card_run'] = 0
                state['last'] = None
                self.counts['HAND_OR_WINDOW_SOURCE_UNKNOWN'] += 1
                return
            source = dict(table_id=table, session_id=session, hand_id=h['hand_id'],
                          source_frame=frame, source_pts=pts, time_base=time_base)
            boxes = self.sidecar.profile['slots'][f'slot_{table:02d}']['seats']
            if state['deal_frame'] is None:
                scores = {}
                supplemental_cards = {}
                for seat, b in boxes.items():
                    x,y,X,Y = b['shown_cards']
                    card_rgb = table_evidence['cards'][seat] if table_evidence is not None else np.ascontiguousarray(rgb[y:Y,x:X])
                    crop = cv2.cvtColor(card_rgb, cv2.COLOR_RGB2BGR)
                    template = self.templates.get((table, seat))
                    scores[seat] = (float(cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)[0,0])
                                    if template is not None else None)
                    supplement=getattr(self,'card_back_supplement',None)
                    if template is None and supplement is not None:
                        matched=supplement.match(table,seat,b['shown_cards'],crop)
                        if matched is not None:
                            scores[seat]=matched['score']
                            supplemental_cards[seat]=matched
                provider=getattr(self,'occupancy_provider',None)
                occupancy=provider(key,frame,list(boxes)) if provider is not None and table_evidence is None else None
                native_partition=None
                empty_model=getattr(self,'empty_button_model',None)
                if empty_model is not None:
                    from parser_core.empty_seat_evidence import frame_partition,RULE
                    empty_source=dict(source,time_base=list(time_base))
                    observed=(table_evidence.get('empty_button_observations') if table_evidence is not None
                              else self.empty_button_sources.get((key,frame)))
                    label_provider=getattr(self,'occupancy_labels_provider',None)
                    labels=(table_evidence.get('occupancy_labels') if table_evidence is not None else
                            label_provider(key,frame,list(boxes)) if label_provider is not None else {})
                    invalid=any(r.get('status')=='EMPTY' and not empty_model.verify(r,empty_source)
                                for r in (observed or {}).values())
                    invalid |= labels is None
                    partition=frame_partition(empty_source,list(boxes),scores,self.threshold,observed,labels or {})
                    if invalid or partition['status']=='CONFLICT':
                        state.update(card_run=0,last=frame,card_support=('EMPTY_BUTTON_CONFLICT',),joint_seat_frames=[])
                        self.counts['TABLE_FRAMES_CHECKED']+=1
                        self.write(dict(source,stage='TABLE_CARD_NCC',scores=scores,consecutive_matches=0,
                            status='UNKNOWN',occupancy_mask_source=RULE,occupied_seats=[],
                            blocker='EMPTY_BUTTON_SOURCE_PROOF_INVALID' if invalid else partition['reason']))
                        return
                    if partition['status']=='PROVEN_FRAME':native_partition=partition
                occupied=occupancy['occupied_seats'] if occupancy is not None else list(boxes)
                support=(tuple((s,occupancy['seats'][s]['state'],occupancy['seats'][s]['actor'],
                                occupancy['seats'][s]['current_pixel_sha256']) for s in boxes)
                         if occupancy is not None else ('ALL_CONFIGURED_SEATS',))
                if native_partition is not None:
                    occupied=native_partition['occupied_seats']
                    support=(RULE,tuple(native_partition['empty_seats']),tuple(occupied))
                all_present = bool(occupied) and all(scores[s] is not None and scores[s] >= self.threshold for s in occupied)
                joint_required=native_partition is not None or bool(supplemental_cards)
                same_support=state.get('card_support',support)==support
                state['card_run'] = ((state['card_run'] + 1) if state['last'] == frame - 1 and same_support else 1) if all_present else 0
                if joint_required and all_present:
                    joint=dict(empty_source,empty_seats=[s for s in boxes if s not in occupied],occupied_seats=occupied,
                               empty_evidence=native_partition['empty_evidence'] if native_partition else {},cards={})
                    if occupancy is not None:joint['occupancy_receipt']=occupancy
                    for seat in occupied:
                        x,y,X,Y=boxes[seat]['shown_cards']
                        pixels=table_evidence['cards'][seat] if table_evidence is not None else np.ascontiguousarray(rgb[y:Y,x:X])
                        ok,packed=cv2.imencode('.png',cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR))
                        if not ok:raise ValueError('JOINT_CARD_RAM_ENCODING_FAILED')
                        joint['cards'][seat]=dict(roi=boxes[seat]['shown_cards'],score=scores[seat],
                            pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                            png_base64=base64.b64encode(packed).decode('ascii'))
                        if seat in supplemental_cards:joint['cards'][seat]['supplemental_match']=supplemental_cards[seat]
                    prior=state.get('joint_seat_frames',[])
                    continuous=(bool(prior) and state['last']==frame-1 and same_support and
                        prior[-1]['source_frame']==frame-1 and prior[-1]['time_base']==list(time_base) and
                        type(pts) is int and type(prior[-1]['source_pts']) is int and prior[-1]['source_pts']<pts)
                    state['joint_seat_frames']=(prior+[joint])[-3:] if continuous else [joint]
                    state['card_run']=len(state['joint_seat_frames']) if type(pts) is int else 0
                else:
                    state.pop('joint_seat_frames',None)
                    if joint_required:state['card_run']=0
                state['card_support']=support
                state['last'] = frame
                self.counts['TABLE_FRAMES_CHECKED'] += 1
                self.write(dict(source, stage='TABLE_CARD_NCC', scores=scores,
                    consecutive_matches=state['card_run'], status='CANDIDATE' if all_present else 'UNKNOWN',
                    occupancy_mask_source=RULE if native_partition is not None else 'EXACT_LABEL_CONTINUITY' if occupancy is not None else 'ALL_CONFIGURED_SEATS',
                    occupied_seats=occupied,
                    blocker=None if all_present else 'COMPLETE_OCCUPANCY_MASK_UNPROVEN'))
                if state['card_run'] < self.consecutive:
                    return
                if native_partition is not None:
                    latest=state['joint_seat_frames'][-1]
                    occupancy=dict(empty_source,occupied_seats=occupied,empty_seats=native_partition['empty_seats'],
                        evidence_status='PROVEN',proof=RULE,authority='CURRENT_EMPTY_OR_DEALT_CARDS_ONLY; NO_ACTOR_AUTHORITY',
                        seats={s:dict(state='OCCUPIED' if s in occupied else 'EMPTY',actor=None,
                            current_source_frame=frame,current_pixel_sha256=(latest['cards'][s] if s in occupied
                            else latest['empty_evidence'][s])['pixel_sha256']) for s in boxes})
                receipt = dict(source, evidence_status='PROVEN',
                    table_pixel_sha256=table_evidence['table_pixel_sha256'] if table_evidence is not None else hashlib.sha256(rgb.tobytes()).hexdigest(),
                    occupied_seats=occupied, card_stable_frames=list(range(frame-2,frame+1)),
                    proof=(RULE if native_partition is not None else 'COMPLETE_MIXED_CARD_EVIDENCE_V1' if supplemental_cards
                           else 'COMPLETE_INDEPENDENT_OCCUPANCY_AND_POSITIVE_FROZEN_CARD_NCC' if occupancy is not None
                           else 'ALL_CONFIGURED_SEATS_POSITIVE_FROZEN_CARD_NCC'))
                if occupancy is not None:receipt['occupancy_receipt']=occupancy
                if supplemental_cards:receipt['supplemental_card_matches']=supplemental_cards
                if joint_required:receipt['joint_seat_frames']=state['joint_seat_frames']
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
                receipt = dict(source, seat=seat, pixel_sha256=sha,
                    roi=b['stack'], occupancy_epoch=None, occupancy_proven=False,
                    forced_bets=None, forced_bets_proven=False)
                projection=self.visibility_sources.get((key,frame,seat))
                if projection is not None:receipt['visibility_source']=projection
                source_states=state.setdefault('source_states',{}).setdefault(seat,{})
                self.counts['STACK_SOURCES_OBSERVED'] += 1
                row = RamRow(frame, None, table, session, i, y, 'STACK_ONLY', 0,
                    pixels, sha, f'{session}:STACK:{h["hand_id"]}:{seat}:{frame}', pts, *time_base)
                if (seat not in state.setdefault('delay_disabled',set()) and
                        state['deal_frame'] < frame <= state['deal_frame'] + self.direct_ocr_delay_frames):
                    ok,packed = cv2.imencode('.png', pixels)
                    if not ok:raise ValueError('STACK_RAM_ENCODING_FAILED')
                    receipt.update(png_base64=base64.b64encode(packed).decode('ascii'),
                        source_state='OCR_SKIPPED_BY_DELAY',source_only=True,ocr_submitted=False,
                        shape=list(pixels.shape),
                        direct_ocr_delay_frames=self.direct_ocr_delay_frames,
                        deal_complete_frame=state['deal_frame'])
                    from parser_core.start_stack_visibility import VisibilitySource
                    verified = VisibilitySource().verify(receipt)
                    guard = state.setdefault('delay_guards',{}).setdefault(seat,{})
                    proof = delay_source_guard(verified,guard) if verified is not None else None
                    if proof is not None:
                        receipt['delay_source_proof'] = proof
                        state['source_readings'].setdefault(seat,{})[frame] = receipt
                        source_states[frame] = 'OCR_SKIPPED_BY_DELAY'
                        state.setdefault('delayed_rows',{}).setdefault(seat,[]).append((row,receipt))
                        self.write(dict(receipt,stage='STACK_SOURCE',status='OCR_SKIPPED_BY_DELAY'))
                        self.counts['STACK_OCR_SKIPPED_BY_DELAY'] += 1
                        continue
                    # Real early numbers/ALL-IN/unexplained pixels must remain
                    # available to SIMPLE and forced binding. Restore every
                    # withheld source in original order, outside queue locks.
                    state['delay_disabled'].add(seat)
                    previous = state.setdefault('delayed_rows',{}).pop(seat,[])
                    for old_row,old_receipt in previous:
                        old_receipt = self._ordinary_delay_receipt(old_receipt)
                        old_receipt['delay_fallback_reason'] = 'PREFIX_PIXELS_NOT_POSITIVELY_EXPLAINED'
                        state['source_readings'][seat][old_row.frame_id] = None
                        source_states[old_row.frame_id] = 'OCR_SUBMITTED_PENDING'
                        self.jobs[row_key(old_row)] = (key,seat,old_receipt)
                        batch.append(old_row)
                        self.write(dict(old_receipt,stage='STACK_SOURCE_STATE',
                            source_state='OCR_SUBMITTED_PENDING',ocr_submitted=True,
                            reason='DELAY_PREFIX_FALLBACK'))
                    self.counts['STACK_DELAY_BACKFILLED'] += len(previous)
                    self.counts['STACK_DELAY_FALLBACK_SEATS'] += 1
                    self.write(dict(source,seat=seat,stage='STACK_DELAY_FALLBACK',
                        status='ORDINARY_OCR_REQUIRED',backfilled_frames=[r.frame_id for r,_ in previous],
                        reason='PREFIX_PIXELS_NOT_POSITIVELY_EXPLAINED'))
                    receipt = self._ordinary_delay_receipt(receipt)
                if frame > state['deal_frame'] + self.direct_ocr_delay_frames:
                    state.setdefault('delayed_rows',{}).pop(seat,None)
                    state.setdefault('delay_guards',{}).pop(seat,None)
                # Submit actual consecutive sources until the first attempt resolves.
                state['source_readings'].setdefault(seat, {})[frame] = None
                source_states[frame] = 'OCR_SUBMITTED_PENDING'
                self.jobs[row_key(row)] = (key, seat, receipt)
                batch.append(row)
            if batch:
                # Existing bounded OCR queue. It is a separate lane and cannot
                # alter ordinary CHAT admission, header jobs or marker state.
                self.counts['STACK_OCR_ATTEMPTED'] += len(batch)
            return batch

    @staticmethod
    def _ordinary_delay_receipt(receipt):
        receipt = dict(receipt)
        for name in ('source_state','source_only','ocr_submitted','direct_ocr_delay_frames',
                     'deal_complete_frame','delay_source_proof'):
            receipt.pop(name,None)
        return receipt

    def completed(self, row, response):
        with self.sidecar.lock, self.lock:
            job = self.jobs.pop(row_key(row), None)
            if job is None:
                self.counts['DUPLICATE_STACK_CALLBACK'] += 1
                return
            key, seat, receipt = job
            self.states[key].setdefault('source_states',{}).setdefault(seat,{})[row.frame_id] = 'OCR_COMPLETED'
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
            attempt = first_visible_stack_triple(state['source_readings'][seat], self.numeric_confidence, start_frame=state['deal_frame']+1,
                allow_scaled=self.visibility_source is not None,numeric_policy=self.numeric_policy,
                direct_ocr_delay_frames=self.direct_ocr_delay_frames,single_frame=self.visibility_source is not None)
            if attempt is not None:
                state['canonical_attempts'][seat] = attempt
                state['stop'].add(seat)
                if attempt['status'] == 'PASS':
                    state['numeric_triples'][seat] = attempt

    def _forget_pending_metric(self, kind, key):
        getattr(self, '_pending_roi_metrics', {}).pop((kind, key), None)

    @staticmethod
    def _pending_source_bytes(kind, entry):
        source, evidence = entry
        if kind == 'TABLE':
            return sum(p.nbytes for field in ('cards', 'stacks') for p in evidence[field].values())
        return sum(p.nbytes for p in evidence.values())

    def _buffer_metrics(self, change=None):
        # Sources contain owned ROI copies. Subtotals never control source life.
        # No-argument/manual calls retain the original full traversal behavior.
        records = getattr(self, '_pending_roi_metrics', None)
        if records is None:
            records = self._pending_roi_metrics = {}
        total = 0
        lengths = []
        active = set()
        try:
            for kind, buffers in (('TABLE', self.pending_tables), ('STACK', self.pending_stacks)):
                for key, pending in buffers.items():
                    sources = pending['sources'] if kind == 'TABLE' else pending
                    pair = (kind, key)
                    active.add(pair)
                    record = records.get(pair)
                    valid = change is not None and record is not None and record['sources'] is sources
                    changed = change is not None and pair == change[:2]
                    if valid:
                        entries = record['entries']
                        if changed:
                            added, removed = change[2:]
                            if removed is added:
                                valid = not entries and not sources
                                next_entries = []
                                subtotal = 0
                            else:
                                valid = (len(sources) == len(entries) + 1 - int(removed is not None)
                                    and (removed is None or bool(entries) and entries[0][0] is removed))
                                if valid:
                                    start = int(removed is not None)
                                    valid = (bool(sources) and sources[-1] is added and
                                        sources[0] is (entries[start][0] if len(entries) > start else added))
                                if valid:
                                    # Evaluate the changed source at its original
                                    # TABLE/STACK position, after preceding buffers.
                                    size = self._pending_source_bytes(kind, added)
                                    next_entries = entries[start:]
                                    next_entries.append((added, size))
                                    subtotal = record['total'] - (entries[0][1] if removed is not None else 0) + size
                        else:
                            valid = (len(sources) == len(entries) and (not sources or
                                sources[0] is entries[0][0] and sources[-1] is entries[-1][0]))
                            if valid:
                                next_entries = entries
                                subtotal = record['total']
                    if not valid:
                        next_entries = [(entry, self._pending_source_bytes(kind, entry)) for entry in sources]
                        subtotal = sum(size for entry, size in next_entries)
                    if not valid or changed:
                        records[pair] = dict(sources=sources, entries=next_entries, total=subtotal)
                    total += subtotal
                    lengths.append(len(sources))
            for pair in list(records):
                if pair not in active:
                    del records[pair]
        except Exception:
            records.clear()
            raise
        self.counts['PEAK_BUFFERED_ROI_BYTES'] = max(self.counts['PEAK_BUFFERED_ROI_BYTES'], total)
        self.counts['PEAK_BUFFERED_SOURCES_PER_BUFFER'] = max(self.counts['PEAK_BUFFERED_SOURCES_PER_BUFFER'], max(lengths, default=0))

    def finish(self):
        try:
            self._finish()
        finally:
            if getattr(self,'empty_fast_hook',None) is not None:
                self.empty_fast_hook.close()

    def _finish(self):
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
            getattr(self, '_pending_roi_metrics', {}).clear()
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
                    attempt = first_visible_stack_triple(readings, self.numeric_confidence, final=True, start_frame=state['deal_frame']+1,
                        allow_scaled=self.visibility_source is not None,numeric_policy=self.numeric_policy,
                        direct_ocr_delay_frames=self.direct_ocr_delay_frames,
                        single_frame=self.visibility_source is not None) if state['deal_frame'] is not None else None
                    if attempt is not None:
                        if self.visibility_source is not None:
                            from parser_core.start_stack_visibility import first_visible_proof
                            attempt['first_visible_proof']=first_visible_proof(readings,attempt['frames'],
                                start_frame=state['deal_frame']+1,confidence=self.numeric_confidence,numeric_policy=self.numeric_policy,
                                qualified_single=True,clean_window_end_frame=w.closed)
                            if attempt['first_visible_proof']['status']!='PROVEN':
                                attempt.update(status='UNKNOWN',blocker=attempt['first_visible_proof']['reason'])
                        state['canonical_attempts'][seat] = attempt
                        if attempt['status']=='PASS':state['numeric_triples'][seat]=attempt
                        else:state['numeric_triples'].pop(seat,None)
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
                     start_stack_pass=len(self.sidecar.windows[k].results),
                     source_states=s.get('source_states',{})) for k,s in self.states.items()],
                'errors':self.ocr.errors,'authority':'LIVE_DIAGNOSTIC; forced bets/epoch UNPROVEN',
                'direct_ocr_delay_frames':self.direct_ocr_delay_frames,
                'pending_jobs':len(self.jobs),
                'submitted_pending_sources':sum(v=='OCR_SUBMITTED_PENDING'
                    for s in self.states.values() for frames in s.get('source_states',{}).values() for v in frames.values()),
                'start_stack_pass':sum(len(w.results) for w in self.sidecar.windows.values())}
            (self.output/'start_stack_live_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
            self.log.close()
            self.closed = True
