"""Supplemental physical row births; never changes legacy tracker identity.

A single unique, pixel-supported suffix/prefix translation can establish the
newly exposed bottom region without establishing GLOBAL_SCROLL identity. Only
complete rows wholly in that region are submitted. Ambiguous transitions keep
an explicit rejection receipt. No OCR text or event type is an input.
"""
from __future__ import annotations

import hashlib
import numpy as np

from parser_core.chat_tracker import GEOMETRY_TOLERANCE_PX, NCC_SCORE_MIN, NCC_MARGIN_MIN
from parser_core.performance_work import note


def pixel_hash(gray):
    return hashlib.sha256(np.ascontiguousarray(gray).tobytes()).hexdigest()


def correlation(a, b):
    if a.shape != b.shape or not a.size:
        return -1.0
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    den = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / den) if den else -1.0


class _CorrelationMemo:
    """Exact paired views, alive and immutable during one tail proof only.

    The proof owns old/new arrays throughout the call and correlation only
    mutates its float64 copies. Shape, stride and dtype are part of identity;
    pointers are never reused as cross-call or cross-frame cache keys.
    """
    def __init__(self):
        self.values = {}
        self.hits = 0

    def __call__(self, a, b):
        key = (int(a.__array_interface__['data'][0]), a.shape, a.strides, a.dtype.str,
               int(b.__array_interface__['data'][0]), b.shape, b.strides, b.dtype.str)
        if key in self.values:
            self.hits += 1
            return self.values[key]
        value = correlation(a, b)
        self.values[key] = value
        return value

    def finish(self, result):
        note('physical_chat_admission.correlation_cache', category='capture/cache',
             hits=self.hits, misses=len(self.values), entries=len(self.values))
        return result


def physical_tail_proof(old, new, old_spans, new_spans):
    """Prove a unique retained suffix and complete newly exposed bottom rows.

    NCC constants retain their existing values; this does not call or mutate
    the legacy GLOBAL gate. At least two complete retained lines are required.
    Geometric alternatives within the existing two-pixel tolerance represent
    the same translation, not competing identities.
    """
    if old.shape != new.shape or not old_spans or not new_spans:
        return dict(status='UNKNOWN', reason='NO_RETAINED_ROW_SUPPORT', candidates=[])
    correlate = _CorrelationMemo()
    height = len(new)
    width = new.shape[1] - 12 if new.shape[1] >= 50 else new.shape[1]
    shifts = {0}
    for a, b in old_spans:
        for c, d in new_spans[:2]:
            delta = round((a + b - c - d) / 2)
            shifts.update(range(max(0, delta - 1), min(height, delta + 2)))
    candidates = []
    for shift in sorted(shifts):
        if 0 < shift <= GEOMETRY_TOLERANCE_PX:
            continue
        # The bottom blank margin is not a physical row. A new row can begin
        # there while only its lower pixels enter from outside the old viewport.
        # Bound births by translated last *content*, not the ROI rectangle.
        boundary = old_spans[-1][1] - shift + GEOMETRY_TOLERANCE_PX
        entering = [j for j, (a, b) in enumerate(new_spans)
                    if a >= boundary and b < height]
        if not entering:
            continue
        retained = []
        used = set()
        for j, (a, b) in enumerate(new_spans):
            if j >= entering[0]:
                break
            possible = [i for i, (c, d) in enumerate(old_spans)
                        if i not in used and abs((a + b - c - d) / 2 + shift) <= GEOMETRY_TOLERANCE_PX]
            if len(possible) != 1:
                retained = []
                break
            i = possible[0]
            used.add(i)
            retained.append((j, i))
        if len(retained) < 2:
            continue
        # The retained complete rows must be precisely the old suffix and new
        # prefix, in order. No missing middle row may be silently bridged.
        if [i for _, i in retained] != list(range(retained[0][1], len(old_spans))):
            continue
        if [j for j, _ in retained] != list(range(entering[0])):
            continue
        support_end = min(height - shift, new_spans[entering[0]][0])
        if shift == 0:
            # Append without scrolling: require the entering region was blank.
            if any(np.count_nonzero(old[a:b, :width] > 70) >= 5 for a, b in (new_spans[j] for j in entering)):
                continue
        score = correlate(old[shift:shift + support_end, :width], new[:support_end, :width])
        rows = []
        for j, i in retained:
            a, b = new_spans[j]
            lo, hi = max(0, a - 2), min(support_end, b + 2)
            alignments = [(delta, correlate(old[lo + shift + delta:hi + shift + delta, :width],new[lo:hi,:width]))
                          for delta in range(-int(GEOMETRY_TOLERANCE_PX),int(GEOMETRY_TOLERANCE_PX)+1)
                          if lo+shift+delta>=0 and hi+shift+delta<=height]
            delta, row_score = max(alignments,key=lambda pair:pair[1])
            rows.append(dict(old_index=i, new_index=j, old_span=list(old_spans[i]),
                             new_span=list(new_spans[j]),
                             row_alignment_residual_px=delta,score=row_score))
        candidates.append(dict(shift_px=shift, score=score, boundary=boundary, support_end=support_end,
                               new_indices=entering, retained_rows=rows))
    candidates.sort(key=lambda c: (-c['score'], c['shift_px']))
    if not candidates:
        return correlate.finish(dict(status='UNKNOWN', reason='NO_ORDERED_RETAINED_SUFFIX_AND_NEW_TAIL', candidates=[]))
    best = candidates[0]
    # Compare exactly the same retained pixels against *all* translations,
    # including no-birth / zero-shift explanations. Repeated old rows plus a
    # local edit must never manufacture a scroll and a second OCR submission.
    support = best['support_end']
    alternatives = [dict(shift_px=shift,
        score=correlate(old[shift:shift + support, :width], new[:support, :width]))
        for shift in shifts if shift + support <= height and
        abs(shift - best['shift_px']) > GEOMETRY_TOLERANCE_PX]
    second = max((c['score'] for c in alternatives), default=-1.)
    best['competing_retained_translations'] = sorted(alternatives, key=lambda c:-c['score'])[:3]
    best['margin'] = best['score'] - second
    # New-row exposure needs two independent retained pixel anchors and the
    # whole-overlap proof. An intervening weak/locally changed row never gains
    # identity authority and must not erase separately proved new bottom rows.
    best['confident_retained_rows'] = sum(r['score'] >= NCC_SCORE_MIN for r in best['retained_rows'])
    for row in best['retained_rows']:
        row['identity_status'] = 'UNKNOWN'
        row['pixel_anchor'] = row['score'] >= NCC_SCORE_MIN
    if best['score'] < NCC_SCORE_MIN or best['confident_retained_rows'] < 2:
        return correlate.finish(dict(status='UNKNOWN', reason='RETAINED_PIXELS_UNPROVEN', candidates=candidates))
    if best['margin'] < NCC_MARGIN_MIN:
        return correlate.finish(dict(status='UNKNOWN', reason='RETAINED_TRANSLATION_AMBIGUOUS', candidates=candidates))
    return correlate.finish(dict(status='PROVEN_PHYSICAL_NEW', reason='UNIQUE_PIXEL_SUFFIX_PREFIX_WITH_NEW_BOTTOM_REGION',
                candidates=candidates, **best))


class PhysicalChatAdmission:
    """Independent ID namespace; legacy R counters and state stay untouched."""
    def __init__(self, table_id, session_id):
        self.table_id = table_id
        self.session_id = session_id
        self.previous = None
        self.counter = 0

    def process(self, gray, transition):
        previous = self.previous
        spans = transition['new_observed_spans']
        self.previous = dict(gray=gray.copy(), spans=spans, frame=transition['frame_to'])
        if previous is None or transition['classification'] not in {'LOCAL_CHANGE', 'UNRESOLVED'}:
            return None
        if gray.shape != previous['gray'].shape or transition['frame_to'] <= previous['frame']:
            raise ValueError('Supplemental CHAT observation order/shape mismatch')
        proof = physical_tail_proof(previous['gray'], gray, previous['spans'], spans)
        admitted = []
        if proof['status'] == 'PROVEN_PHYSICAL_NEW':
            for index in proof['new_indices']:
                self.counter += 1
                admitted.append(dict(index=index, span=list(spans[index]),
                    row_id=f'{self.session_id}:F{self.counter:06d}',
                    row_pixel_sha256=pixel_hash(gray[slice(*spans[index])]),
                    OCR_admission=True, identity_status='UNKNOWN',
                    admission_reason=proof['reason']))
        return dict(schema='physical-chat-admission-v1', table=self.table_id, session_id=self.session_id,
            frame_before=previous['frame'], frame_after=transition['frame_to'],
            old_classification=transition['classification'], old_reason=transition['reason'],
            old_spans=previous['spans'], new_spans=spans,
            source_pixel_hash_before=pixel_hash(previous['gray']), source_pixel_hash_after=pixel_hash(gray),
            geometry=dict(shape=list(gray.shape), top_edge_spans=transition['new_edge_spans']),
            identity_status='UNKNOWN', old_identity_reused=False,
            proof=proof, admissions=admitted)
