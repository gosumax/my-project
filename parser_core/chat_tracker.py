"""Promoted adjacent-frame CHAT tracker; frozen v4 admission semantics.

Motion uses the previous observation, never SAFE_ID_STATE. Geometry KEEP
preserves candidate object IDs and their uncertainty, not certified lineage.
There is no OCR input, history resurrection, or alternative motion method.
"""
from __future__ import annotations

from collections import Counter
import time
import numpy as np

from parser_core.chat_rows import physical_spans
from parser_core import performance_diagnostics as diagnostics

ROW_STEP_PX = 20.17647058823529
NCC_SCORE_MIN = .90
NCC_MARGIN_MIN = .02
MIN_ZONE_OVERLAP_PX = 48
GEOMETRY_TOLERANCE_PX = 2.
CONFIRMED_LINK_STATES = frozenset(('INITIAL_VISIBLE', 'GLOBAL_NEW_BOTTOM', 'GLOBAL_CONFIRMED_CARRY'))
CHAT_TRACKER_VERSION = 'adjacent-safe-v4'


def zone_scores(old, new, y1, y2):
    a = old[:, :-12:3].astype(np.float64)
    b = new[:, :-12:3].astype(np.float64)
    h, w = a.shape
    size = 1 << (2*h-1).bit_length()
    masked = np.zeros_like(b)
    masked[y1:y2] = b[y1:y2]
    dots = np.fft.irfft(np.sum(np.fft.rfft(a, n=size, axis=0)*np.conj(
        np.fft.rfft(masked, n=size, axis=0)), axis=1), n=size)
    shifts = np.arange(max(0, h-y1-MIN_ZONE_OVERLAP_PX+1))
    end = np.minimum(y2, h-shifts)
    support = end-y1
    valid = support >= MIN_ZONE_OVERLAP_PX
    shifts, end, support = shifts[valid], end[valid], support[valid]
    def prefix(x):
        return np.r_[0., np.cumsum(x.sum(axis=1))]
    pa, pb, paa, pbb = prefix(a), prefix(b), prefix(a*a), prefix(b*b)
    sa = pa[shifts+end]-pa[shifts+y1]
    sb = pb[end]-pb[y1]
    n = support*w
    va = paa[shifts+end]-paa[shifts+y1]-sa*sa/n
    vb = pbb[end]-pbb[y1]-sb*sb/n
    den = np.sqrt(np.maximum(va, 0)*np.maximum(vb, 0))
    scores = np.divide(dots[shifts]-sa*sb/n, den,
                       out=np.full(len(shifts), -1.), where=den > 0)
    return shifts, scores, support


def measure_zone(old, new, y1, y2):
    shifts, scores, support = zone_scores(old, new, y1, y2)
    if not len(shifts):
        return {'accepted': False, 'shift_px': None, 'score': None, 'reason': 'NO_OVERLAP'}
    order = np.argsort(scores)[::-1]
    index = int(order[0])
    shift, score = int(shifts[index]), float(scores[index])
    second = max((float(scores[i]) for i in range(len(scores))
                  if abs(int(shifts[i])-shift) > 2), default=-1.)
    return {'shift_px': shift, 'score': score, 'margin': score-second,
            'support_px': int(support[index]), 'accepted': score >= NCC_SCORE_MIN and score-second >= NCC_MARGIN_MIN,
            'y1': y1, 'y2': y2,
            'top_shifts': [[int(shifts[i]), float(scores[i])] for i in order[:3]]}


def global_scroll(old, new):
    height = old.shape[0]
    cuts = [0, height//3, 2*height//3, height]
    zones = {name: measure_zone(old, new, cuts[i], cuts[i+1])
             for i, name in enumerate(('top', 'middle', 'bottom'))}
    accepted = [(name, z['shift_px']) for name, z in zones.items() if z['accepted']]
    result = {'classification': 'UNRESOLVED', 'shift_px': None,
              'reason': 'INSUFFICIENT_CONFIDENT_ZONES', 'support_zones': [name for name, _ in accepted]}
    if len(accepted) >= 2:
        shifts = [shift for _, shift in accepted]
        if max(shifts)-min(shifts) > GEOMETRY_TOLERANCE_PX:
            result.update(classification='LOCAL_CHANGE', reason='CONFIDENT_ZONE_SHIFT_CONFLICT')
        else:
            shift = int(round(float(np.median(shifts))))
            if shift > 2:
                result.update(classification='GLOBAL_SCROLL', shift_px=shift, reason='NONZERO_ZONE_CONSENSUS')
            else:
                result.update(classification='LOCAL_CHANGE', reason='ZERO_ZONE_CONSENSUS_CHANGED_CROP')
    result.update(zones=zones, tolerance_px=GEOMETRY_TOLERANCE_PX)
    return result


def complete_spans(all_spans):
    edge = all_spans[:1] if all_spans and all_spans[0][0] <= 1 else []
    return all_spans[len(edge):], edge


def symmetry(old_spans, new_spans, shift):
    n = round(shift/ROW_STEP_PX)
    result = {'N': n, 'old_count': len(old_spans), 'new_count': len(new_spans), 'passed': False}
    if n < 1 or n >= len(old_spans) or n > len(new_spans):
        result['reason'] = 'N_OUTSIDE_PROVEN_OVERLAP'
        return result
    old_centers = np.array([(a+b)/2-shift for a, b in old_spans])
    new_centers = np.array([(a+b)/2 for a, b in new_spans])
    nout = int(np.sum(old_centers < new_centers[0]-GEOMETRY_TOLERANCE_PX))
    nin = int(np.sum(new_centers > old_centers[-1]+GEOMETRY_TOLERANCE_PX))
    result.update(N_OUT=nout, N_IN=nin)
    if len(old_spans) != len(new_spans):
        result['reason'] = 'VISIBLE_WINDOW_COUNT_CHANGED'
        return result
    if nout != n or nin != n:
        result['reason'] = 'N_OUT_N_IN_DISAGREE_WITH_SHIFT'
        return result
    residuals = (old_centers[n:]-new_centers[:-n]).tolist()
    result['retained_center_residuals_px'] = residuals
    if max(map(abs, residuals), default=0) > GEOMETRY_TOLERANCE_PX:
        result['reason'] = 'RETAINED_ORDER_OR_COORDINATES_UNRESOLVED'
        return result
    result.update(passed=True, reason='N_OUT_EQUALS_N_IN_WITH_ORDERED_COORDINATE_CARRY')
    return result


def coordinate_map(old_spans, new_spans):
    mapping, used = {}, set()
    for j, new in enumerate(new_spans):
        distances = [(abs((old[0]+old[1]-new[0]-new[1])/2), i)
                     for i, old in enumerate(old_spans) if i not in used]
        if diagnostics.ACTIVE is not None:
            diagnostics.ACTIVE.count('tracker_identity_candidates',len(distances))
        if distances:
            delta, i = min(distances)
            if delta <= GEOMETRY_TOLERANCE_PX:
                mapping[j] = i
                used.add(i)
    return mapping


class AdjacentChatTracker:
    def __init__(self, table_id: int, session_id: str):
        self.table_id = table_id
        self.session_id = session_id
        self.previous = None
        self.safe_id_state = None
        self.journal = {}
        self.counter = 0
        self.metrics = Counter()

    def _new_id(self, frame, span, initial=False):
        self.counter += 1
        rid = f'{self.session_id}:R{self.counter:06d}'
        self.journal[rid] = {'table': self.table_id, 'birth_frame': frame,
                             'initial': initial, 'initial_span': span}
        return rid

    def process(self, gray: np.ndarray, frame_id: int):
        if gray.dtype != np.uint8 or gray.ndim != 2 or not gray.size:
            raise ValueError('Expected nonempty native uint8 grayscale CHAT')
        all_spans = physical_spans(gray)
        spans, edge = complete_spans(all_spans)
        previous, safe_before = self.previous, self.safe_id_state
        diagnostic_state=diagnostics.ACTIVE.tick() if diagnostics.ACTIVE is not None else None
        current_ids = [None]*len(spans)
        model_spans = [list(span) for span in spans]
        states = ['UNKNOWN']*len(spans)
        guard, carried, removed, admitted = None, [], [], []
        if previous is None:
            current_ids = [self._new_id(frame_id, span, True) for span in spans]
            states = ['INITIAL_VISIBLE']*len(spans)
            self.safe_id_state = {'frame': frame_id, 'row_ids': current_ids,
                                  'model_spans': spans, 'link_states': states}
            model_spans = spans
            gate = {'classification': 'INITIAL_VISIBLE', 'shift_px': None, 'reason': 'INITIAL_SEED_NO_OCR', 'zones': {}}
        else:
            if diagnostic_state is not None:
                diagnostics.ACTIVE.record('chat_tracker_state_update',diagnostic_state,table=self.table_id)
                diagnostic_state=None
            if frame_id <= previous['frame'] or gray.shape != previous['gray'].shape:
                raise ValueError('CHAT frame order/ROI shape changed within one session')
            tick = time.perf_counter()
            gate = global_scroll(previous['gray'], gray)
            self.metrics['zonal_ncc_seconds'] += time.perf_counter()-tick
        if diagnostic_state is not None:
            diagnostics.ACTIVE.record('chat_tracker_state_update',diagnostic_state,table=self.table_id)
        candidate = gate['classification']
        if candidate == 'GLOBAL_SCROLL':
            guard = symmetry(previous['observed_spans'], spans, gate['shift_px'])
            if not guard['passed']:
                gate.update(classification='UNRESOLVED', reason='SYMMETRY:'+guard['reason'])
            else:
                diagnostic_state=diagnostics.ACTIVE.tick() if diagnostics.ACTIVE is not None else None
                n, shift = guard['N'], gate['shift_px']
                admitted = [self._new_id(frame_id, span) for span in spans[-n:]]
                current_ids = previous['row_ids'][n:]+admitted
                model_spans = [[a-shift, b-shift] for a, b in previous['model_spans'][n:]]+[list(s) for s in spans[-n:]]
                states[-n:] = ['GLOBAL_NEW_BOTTOM']*n
                for j in range(len(spans)-n):
                    i = j+n
                    parent = previous['link_states'][i]
                    states[j] = 'GLOBAL_CONFIRMED_CARRY' if parent in CONFIRMED_LINK_STATES and current_ids[j] is not None else 'GLOBAL_PARENT_UNRESOLVED'
                    carried.append({'old_index': i, 'new_index': j, 'row_id': current_ids[j],
                                    'parent_link_state': parent, 'old_span': previous['model_spans'][i],
                                    'translated_span': model_spans[j], 'observed_span': spans[j], 'OCR_submitted': False})
                removed = previous['row_ids'][:n]
                self.safe_id_state = {'frame': frame_id, 'row_ids': current_ids,
                                      'model_spans': model_spans, 'link_states': states}
                if diagnostic_state is not None:
                    diagnostics.ACTIVE.record('chat_tracker_state_update',diagnostic_state,table=self.table_id)
        if gate['classification'] == 'LOCAL_CHANGE':
            old_index = {tuple(s): i for i, s in enumerate(previous['observed_spans'])}
            for j, i in coordinate_map(previous['observed_spans'], spans).items():
                diagnostic_state=diagnostics.ACTIVE.tick() if diagnostics.ACTIVE is not None else None
                current_ids[j] = previous['row_ids'][i]
                model_spans[j] = previous['model_spans'][i]
                if current_ids[j] is not None:
                    a, b = spans[j]
                    exact_i = old_index.get((a, b))
                    exact = (exact_i is not None and previous['row_ids'][exact_i] == current_ids[j]
                             and np.array_equal(previous['gray'][a:b, :-12], gray[a:b, :-12]))
                    states[j] = previous['link_states'][i] if exact else 'LOCAL_LINK_UNRESOLVED'
                if diagnostic_state is not None:
                    diagnostics.ACTIVE.record('chat_tracker_state_update',diagnostic_state,table=self.table_id)
        diagnostic_state=diagnostics.ACTIVE.tick() if diagnostics.ACTIVE is not None else None
        known = [rid for rid in current_ids if rid is not None]
        if len(known) != len(set(known)):
            raise ValueError('Duplicate CHAT object ID in visible window')
        safe_unchanged = self.safe_id_state == safe_before
        if previous is not None and gate['classification'] != 'GLOBAL_SCROLL' and not safe_unchanged:
            raise ValueError('Non-GLOBAL transition corrupted SAFE_ID_STATE')
        self.previous = {'frame': frame_id, 'gray': gray.copy(), 'observed_spans': spans,
                         'model_spans': model_spans, 'row_ids': current_ids, 'edge_spans': edge, 'link_states': states}
        self.metrics[gate['classification']+'_COUNT'] += 1
        self.metrics['OCR_NEW_BOTTOM_ROWS'] += len(admitted)
        if diagnostic_state is not None:
            diagnostics.ACTIVE.record('chat_tracker_state_update',diagnostic_state,table=self.table_id)
        return {'table': self.table_id, 'frame_from': previous['frame'] if previous else None,
                'frame_to': frame_id, 'candidate_classification': candidate, 'classification': gate['classification'],
                'reason': gate['reason'], 'SHIFT_PX': gate['shift_px'], 'ROW_STEP_PX': ROW_STEP_PX,
                'symmetry': guard, 'zones': gate['zones'], 'all_spans': all_spans,
                'old_ids': previous['row_ids'] if previous else [], 'new_ids': current_ids,
                'old_observed_spans': previous['observed_spans'] if previous else [], 'new_observed_spans': spans,
                'old_model_spans': previous['model_spans'] if previous else [], 'new_model_spans': model_spans,
                'old_edge_spans': previous['edge_spans'] if previous else [], 'new_edge_spans': edge,
                'old_link_states': previous['link_states'] if previous else [], 'new_link_states': states,
                'removed_top_ids': removed, 'carried_rows': carried, 'OCR_queue_row_ids': admitted,
                'safe_frame_from': safe_before['frame'] if safe_before else None,
                'SAFE_ID_STATE_UNCHANGED': safe_unchanged, 'PREVIOUS_FRAME_UPDATED': True,
                'checkpoint_committed': bool(admitted), 'visible_unknown_links': sum(rid is None for rid in current_ids)}
