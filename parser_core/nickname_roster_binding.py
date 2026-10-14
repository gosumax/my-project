"""Reproducible literal-prefix matching against a complete physical roster.

This module is opt-in. It never edits raw nicknames or invents an alias. A full
actor name comes from CHAT; each seat needs its own literal visual constraint.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

RULE = 'UNIQUE_LITERAL_PREFIX_COMPLETE_ROSTER_V1'
OVERLAY = re.compile(r'(?:(?:ANTE|SB|BB|FOLD|CHECK|CALL|BET|RAISE)(?:\s+\d[\d\s]*)?'
                     r'|SIT[\s-]*OUT|ALL[\s-]*IN|OPEN SEAT)[.!●]?', re.I)


def literal_constraint(text):
    text = text.strip()
    if not text or OVERLAY.fullmatch(text) or any(c in text for c in ('?', '\ufffd')):
        return None
    suffix = '...' if text.endswith('...') else '…' if text.endswith('…') else None
    prefix = text[:-len(suffix)] if suffix else text
    if not prefix or '...' in prefix or '…' in prefix or not re.search(r'\w', prefix):
        return None
    return dict(kind='PREFIX' if suffix else 'EXACT', text=prefix)


def matches(actor, constraint):
    return actor.startswith(constraint['text']) if constraint['kind'] == 'PREFIX' else actor == constraint['text']


def _unique_matching(edges):
    solutions = []
    seats = sorted(edges, key=lambda s: (len(edges[s]), s))
    def visit(i, assigned, used):
        if len(solutions) > 1:
            return
        if i == len(seats):
            solutions.append(dict(assigned))
            return
        seat = seats[i]
        for actor in sorted(edges[seat] - used):
            assigned[seat] = actor
            visit(i+1, assigned, used | {actor})
            assigned.pop(seat)
    visit(0, {}, set())
    return solutions


@lru_cache(maxsize=1)
def _seat_rois():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/layout_847x404.json').read_text('utf-8'))['slots']


def _verify_gray_source(row):
    pixels = cv2.imdecode(np.frombuffer(base64.b64decode(row['png_base64'], validate=True), np.uint8), cv2.IMREAD_GRAYSCALE)
    if pixels is None:
        raise ValueError('SOURCE_PIXELS_UNKNOWN')
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    if digest != row.get('pixel_sha256') or digest != row.get('source_ref'):
        raise ValueError('SOURCE_PIXEL_HASH_CONFLICT')
    return pixels


def _legacy_nickname_triples(rows, key, source, end, confidence, deal, forced_source):
    by_seat = defaultdict(dict)
    frame_pts = {source:deal['source_pts']}
    frame_pts[forced_source[0]]=forced_source[1]
    for row in rows:
        if row.get('kind') != 'NICK' or row.get('pre_deal'):
            continue
        if (row.get('table_id'), row.get('session_id'), row.get('hand_id')) != key:
            raise ValueError('NICKNAME_SOURCE_CONTEXT_CONFLICT')
        frame = row.get('source_frame')
        if not isinstance(frame, int) or isinstance(frame, bool):
            raise ValueError('NICKNAME_SOURCE_FRAME_UNKNOWN')
        if not source < frame <= end:
            continue
        seat = row.get('seat')
        if (row.get('time_base') != deal['time_base'] or type(row.get('source_pts')) is not int
                or row['source_pts'] <= deal['source_pts']):
            raise ValueError('NICKNAME_TIME_PRECEDES_FORCED_SOURCE')
        if frame in frame_pts and frame_pts[frame] != row['source_pts']:
            raise ValueError('NICKNAME_SOURCE_FRAME_PTS_CONFLICT')
        frame_pts[frame] = row['source_pts']
        prior = by_seat[seat].get(frame)
        if prior and prior != row:
            raise ValueError('NICKNAME_DUPLICATE_SOURCE_CONFLICT')
        by_seat[seat][frame] = row
    timestamps = [frame_pts[f] for f in sorted(frame_pts)]
    if timestamps != sorted(set(timestamps)):
        raise ValueError('NICKNAME_SOURCE_ORDER_CONFLICT')
    triples = defaultdict(list)
    for seat, readings in by_seat.items():
        for frame in sorted(readings):
            triple = [readings.get(f) for f in range(frame-2, frame+1)]
            if any(r is None for r in triple):
                continue
            responses = [r.get('ocr_response', {}) for r in triple]
            if any(r.get('status') != 'OK' or type(r.get('score')) not in (int, float)
                   or not math.isfinite(r['score']) or r['score'] < confidence for r in responses):
                continue
            texts = [(r.get('raw_text') or '').strip() for r in responses]
            if len(set(texts)) != 1:
                continue
            constraint = literal_constraint(texts[0])
            if constraint is None:
                continue
            rois = {tuple(r.get('roi', [])) for r in triple}
            if len(rois) != 1 or len(next(iter(rois))) != 4:
                raise ValueError('NICKNAME_ROI_CONFLICT')
            if list(next(iter(rois))) != _seat_rois()[f'slot_{key[0]:02d}']['seats'][seat]['nick']:
                raise ValueError('NICKNAME_FIXED_SEAT_ROI_CONFLICT')
            times = []
            for r in triple:
                pixels = _verify_gray_source(r)
                x, y, X, Y = r['roi']
                if pixels is None or pixels.shape != (Y-y, X-x):
                    raise ValueError('NICKNAME_PIXEL_SHAPE_CONFLICT')
                num, den = r['time_base']
                if num <= 0 or den <= 0:
                    raise ValueError('NICKNAME_TIME_BASE_UNKNOWN')
                times.append(r['source_pts'] * num / den)
            if times != sorted(set(times)):
                raise ValueError('NICKNAME_SOURCE_TIME_CONFLICT')
            triples[seat].append(dict(constraint=constraint, frames=list(range(frame-2, frame+1)),
                                      source_rows=copy.deepcopy(triple)))
    return triples


from .nickname_policy import (FIRST_NICK_POLICY, FIRST_NICK_POLICY_V1,
    FIRST_NICK_POLICIES, nickname_witness_floor)


def nickname_epoch_matches(machine, key, seat, frame, deal_frame, epoch):
    """An early name may precede DEAL occupancy, never a physical replacement."""
    if frame >= deal_frame:
        return machine.epoch_at(*key[:2], seat, frame) == epoch
    if any((e['table'], e['session'], e['seat']) == (*key[:2], seat)
           and frame <= e['frame'] <= deal_frame
           and (e['kind'] in {'EMPTY', 'REPLACE'} or e['status'] != 'APPLIED')
           for e in machine.audit):
        return False
    earlier = machine.epoch_at(*key[:2], seat, frame)
    return earlier in (None, epoch) and machine.epoch_at(*key[:2], seat, deal_frame) == epoch


def nickname_hand_bounds(inputs):
    """Current confirmed hand, ending before the next header or observed EOF."""
    key=tuple(inputs['context'])
    triggers=[t for t in inputs['hand_triggers'] if
        (t['table_id'],t['session_id'])==key[:2]]
    own=[t['source_frame'] for t in triggers if t['hand_id']==key[2]]
    if not own:raise ValueError('NICK_HAND_START_UNKNOWN')
    start=min(own)
    following=[t['source_frame'] for t in triggers if t['hand_id']!=key[2]
               and t['source_frame']>start]
    observed=inputs.get('observed_capture_end_frame')
    if observed is None:
        observed=max([inputs['deal']['source_frame']]+[
            r['source_frame'] for r in inputs['nick_rows'] if
            (r.get('table_id'),r.get('session_id'),r.get('hand_id'))==key])
    if type(observed) is not int or observed<start:raise ValueError('NICK_HAND_END_UNKNOWN')
    return start,min(observed,min(following)-1) if following else observed


def _nickname_triples(rows, key, source, end, confidence, deal, forced_source):
    """Historical function name; new witnesses contain one real source frame."""
    by_seat=defaultdict(dict)
    frame_pts={deal['source_frame']:deal['source_pts'],forced_source[0]:forced_source[1]}
    for row in rows:
        if row.get('kind')!='NICK':continue
        if (row.get('table_id'),row.get('session_id'),row.get('hand_id'))!=key:
            raise ValueError('NICKNAME_SOURCE_CONTEXT_CONFLICT')
        frame=row.get('source_frame')
        if type(frame) is not int:raise ValueError('NICKNAME_SOURCE_FRAME_UNKNOWN')
        if not source<=frame<=end:continue
        pts=row.get('source_pts')
        if (row.get('time_base')!=deal['time_base'] or type(pts) is not int or
                (frame<deal['source_frame'] and pts>=deal['source_pts']) or
                (frame==deal['source_frame'] and pts!=deal['source_pts']) or
                (frame>deal['source_frame'] and pts<=deal['source_pts'])):
            raise ValueError('NICKNAME_SOURCE_TIME_CONFLICT')
        if frame in frame_pts and frame_pts[frame]!=pts:raise ValueError('NICKNAME_SOURCE_FRAME_PTS_CONFLICT')
        frame_pts[frame]=pts
        prior=by_seat[row['seat']].get(frame)
        if prior is not None and prior!=row:raise ValueError('NICKNAME_DUPLICATE_SOURCE_CONFLICT')
        by_seat[row['seat']][frame]=row
    times=[frame_pts[f] for f in sorted(frame_pts)]
    if times!=sorted(set(times)):raise ValueError('NICKNAME_SOURCE_ORDER_CONFLICT')
    from .chat_nick_actor import literal_constraint as current_constraint, prove_nick_ui_overlay
    result=defaultdict(list)
    for seat,readings in by_seat.items():
        for frame,row in sorted(readings.items()):
            response=row.get('ocr_response',{});score=response.get('score')
            if (response.get('status')!='OK' or type(score) not in (int,float) or
                    not math.isfinite(score) or not confidence<=score<=1):continue
            constraint=current_constraint((response.get('raw_text') or '').strip())
            if constraint is None or prove_nick_ui_overlay(row) is not None:continue
            roi=_seat_rois()[f'slot_{key[0]:02d}']['seats'][seat]['nick']
            if list(row.get('roi',[]))!=list(roi):raise ValueError('NICKNAME_FIXED_SEAT_ROI_CONFLICT')
            pixels=_verify_gray_source(row)
            if pixels.shape!=(roi[3]-roi[1],roi[2]-roi[0]):raise ValueError('NICKNAME_PIXEL_SHAPE_CONFLICT')
            result[seat].append(dict(constraint=constraint,frames=[frame],
                source_rows=[copy.deepcopy(row)]))
            break
    return result


def prove_roster_binding(inputs, *, _legacy=False, _witness_policy=FIRST_NICK_POLICY):
    """Find the first complete literal constraint graph; never retry conflicts."""
    def unknown(reason):
        return dict(status='UNKNOWN', reason=reason, binding_rule=RULE)
    try:
        key = tuple(inputs['context'])
        snapshot, deal = inputs['snapshot'], inputs['deal']
        if len(key) != 3 or (deal.get('table_id'), deal.get('session_id'), deal.get('hand_id')) != key:
            return unknown('ROSTER_CONTEXT_CONFLICT')
        if not snapshot.get('forced_snapshot_complete') or deal.get('evidence_status') != 'PROVEN':
            return unknown('COMPLETE_ROSTER_OR_OCCUPANCY_UNKNOWN')
        seats = deal.get('occupied_seats', [])
        actors = snapshot.get('actors', [])
        if (len(seats) < 2 or len(seats) != len(set(seats)) or len(actors) != len(set(actors))
                or len(actors) != len(seats) or any(not isinstance(a, str) or not a for a in actors)):
            return unknown('ROSTER_PARTICIPANT_CARDINALITY_CONFLICT')
        posts = snapshot.get('posts', [])
        if Counter(p['actor'] for p in posts if p['role'] == 'ANTE') != Counter(actors):
            return unknown('COMPLETE_DISTINCT_ANTE_ROSTER_UNKNOWN')
        sources = {p.get('source_frame') for p in posts}
        if len(sources) != 1 or None in sources:
            return unknown('ROSTER_FORCED_SOURCE_UNKNOWN')
        source = next(iter(sources))
        deal_frame=deal.get('source_frame')
        if (not isinstance(deal_frame,int) or not deal_frame <= source <= deal_frame+inputs['capture_frames']
                or any(p.get('phase') != 'BEFORE_DEAL' for p in posts)):
            return unknown('ROSTER_FORCED_DEAL_ORDER_UNKNOWN')
        # Replay the complete forced prefix from its raw OCR source rows.
        from parser_core.seat_forced_binding import parse_forced_snapshot
        prefix = inputs.get('forced_rows', [])
        for row in prefix:
            pixels = _verify_gray_source(row)
            response = row.get('ocr_response', {})
            if (type(row.get('score')) not in (int,float) or not math.isfinite(row['score']) or
                    not 0 <= row['score'] <= 1 or row.get('text') != (response.get('raw_text') or '').strip() or
                    row.get('score') != (response.get('score') if response.get('status') == 'OK' else 0) or
                    pixels.shape[0] != row['row_bounds'][1]-row['row_bounds'][0]):
                return unknown('ROSTER_FORCED_OCR_SOURCE_CONFLICT')
        reproduced, reason = parse_forced_snapshot(prefix, key[2], len(seats), inputs.get('confidence', .95))
        fields = ('actor', 'role', 'amount', 'source_refs', 'source_index', 'phase')
        if (reproduced is None or reproduced['actors'] != actors or
                [{k:p.get(k) for k in fields} for p in reproduced['posts']] !=
                [{k:p.get(k) for k in fields} for p in posts]):
            return unknown(reason or 'ROSTER_PREFIX_NOT_REPRODUCIBLE')
        lo, hi = reproduced['header_index'], reproduced['deal_index']
        if sorted(r['index'] for r in prefix if lo <= r['index'] <= hi) != list(range(lo, hi+1)):
            return unknown('ROSTER_FORCED_PREFIX_SOURCE_GAP')
        if any((r.get('table_id'),r.get('session_id'),r.get('hand_id'),r.get('source_frame')) != (*key,source)
               for r in prefix):
            return unknown('ROSTER_PREFIX_SOURCE_CONTEXT_CONFLICT')
        prefix_times={(r.get('source_pts'),tuple(r.get('time_base',[]))) for r in prefix}
        if len(prefix_times)!=1:
            return unknown('ROSTER_PREFIX_SOURCE_TIME_CONFLICT')
        prefix_pts,prefix_base=next(iter(prefix_times))
        if (type(prefix_pts) is not int or list(prefix_base)!=deal.get('time_base') or
                (source==deal_frame and prefix_pts!=deal.get('source_pts')) or
                (source>deal_frame and prefix_pts<=deal.get('source_pts')) or
                any(p.get('source_pts')!=prefix_pts or p.get('time_base')!=list(prefix_base) for p in posts)):
            return unknown('ROSTER_PREFIX_SOURCE_TIME_CONFLICT')
        confidence = inputs.get('confidence', .95)
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or confidence < .95:
            return unknown('ROSTER_CONFIDENCE_GATE_UNKNOWN')
        start,end=(deal_frame,deal_frame+inputs['capture_frames']) if _legacy else nickname_hand_bounds(inputs)
        witness=_legacy_nickname_triples if _legacy else _nickname_triples
        nick_confidence=confidence if _legacy or _witness_policy==FIRST_NICK_POLICY_V1 else nickname_witness_floor(_witness_policy)
        triples = witness(inputs['nick_rows'], key, start, end, nick_confidence, deal,(source,prefix_pts))
        if any(not triples.get(seat) for seat in seats):
            return unknown('ROSTER_SEAT_LITERAL_TRIPLE_UNKNOWN')
        anchor = max(source,max(min(t['frames'][-1] for t in triples[seat]) for seat in seats))
        labels = {seat:[t for t in triples[seat] if t['frames'][-1] <= anchor] for seat in seats}
        events = [e for e in inputs['state_receipts'] if (e.get('table'),e.get('session')) == key[:2]
                  and deal_frame <= e.get('frame', -1) <= anchor]
        if any(e.get('status') != 'APPLIED' or e.get('kind') != 'OCCUPIED'
               or e.get('hand_id') != key[2] for e in events):
            return unknown('ROSTER_SEAT_STATE_CHANGED_OR_UNKNOWN')
        from parser_core.seat_epoch_state import SeatEpochMachine
        machine = SeatEpochMachine()
        for trigger in inputs['hand_triggers']:
            machine.hand(trigger['table_id'],trigger['session_id'],trigger['hand_id'],trigger['source_frame'])
        if key not in machine.hands or machine.hands[key] > deal_frame:
            return unknown('ROSTER_HAND_START_UNKNOWN')
        for event in inputs['state_receipts']:
            if event['frame'] > anchor:
                continue
            machine.observe(event['table'],event['session'],event['seat'],event['kind'],event['frame'],
                **{k:event[k] for k in ('actor','hand_id','evidence_status','source_ref','source_index') if k in event})
        machine.build()
        if any((e['table'],e['session']) == key[:2] and e['status'] != 'APPLIED' for e in machine.audit):
            return unknown('ROSTER_OCCUPANCY_EVIDENCE_UNKNOWN')
        epochs = {}
        for seat in seats:
            evidence = [e for e in events if e.get('seat') == seat]
            first = [e for e in evidence if e.get('frame') == deal_frame and
                     e.get('source_ref') == deal.get('table_pixel_sha256')]
            found = {e.get('epoch') for e in evidence}
            if not first or len(found) != 1 or None in found:
                return unknown('ROSTER_PARTICIPANT_EPOCH_UNKNOWN')
            epochs[seat] = next(iter(found))
            epoch = epochs[seat]
            if (not machine.membership(*key[:2],seat,key[2],epoch)['participant'] or
                    machine.epoch_at(*key[:2],seat,source) != epoch or
                    any((machine.epoch_at(*key[:2],seat,f) != epoch if _legacy else
                         not nickname_epoch_matches(machine,key,seat,f,deal_frame,epoch))
                        for t in labels[seat] for f in t['frames'])):
                return unknown('ROSTER_PARTICIPANT_MEMBERSHIP_UNPROVEN')
        if any((e.get('table'),e.get('session')) == key[:2] and e.get('frame', -1) <= anchor
               for e in inputs['unresolved_events']):
            return unknown('ROSTER_UNRESOLVED_SEAT_EVENT')
        edges = {seat:{a for a in actors if all(matches(a,t['constraint']) for t in labels[seat])} for seat in seats}
        solutions = _unique_matching(edges)
        if len(solutions) != 1:
            return unknown('ROSTER_MATCHING_CONFLICT' if not solutions else 'ROSTER_MATCHING_AMBIGUOUS')
        matching = solutions[0]
        for seat, actor in matching.items():
            if not all(matches(actor,t['constraint']) for t in labels[seat]):
                return unknown('ROSTER_SELECTED_ACTOR_CONTRADICTS_LABEL')
        retained = copy.deepcopy(inputs)
        retained['nick_rows'] = [r for r in retained['nick_rows'] if r.get('source_frame',end+1) <= anchor]
        retained['state_receipts'] = [e for e in retained['state_receipts'] if e.get('frame',end+1) <= anchor]
        retained['unresolved_events'] = [e for e in retained['unresolved_events'] if e.get('frame',end+1) <= anchor]
        result=dict(status='PROVEN', reason=None, binding_rule=RULE, context=list(key),
                    forced_source_frame=source, deal_source_frame=deal_frame,
                    forced_source_pts=prefix_pts,binding_anchor_frame=anchor,
                    matching=matching, edges={s:sorted(a) for s,a in edges.items()},
                    epochs=epochs, label_evidence=labels, source_inputs=retained)
        if not _legacy:result['nickname_witness_policy']=_witness_policy
        return result
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, cv2.error):
        return unknown('ROSTER_SOURCE_RECEIPT_INVALID')


def binding_receipt(proof, actor, posts):
    if proof.get('status') != 'PROVEN':
        return None
    seat = next((s for s,a in proof['matching'].items() if a == actor), None)
    if seat is None:
        return None
    table, session, hand = proof['context']
    return dict(table=table, session=session, hand_id=hand, actor=actor, seat=seat,
        epoch=proof['epochs'][seat], binding_rule=RULE,
        forced_source_frame=proof['forced_source_frame'],
        label_confirmation_frame=proof['binding_anchor_frame'],
        binding_anchor_frame=proof['binding_anchor_frame'],
        label_evidence=copy.deepcopy(proof['label_evidence'][seat]),
        forced_source_refs=[p['source_refs'] for p in posts], roster_proof=proof)


def verify_binding_receipt(receipt, snapshot):
    try:
        proof = receipt['roster_proof']
        policy=proof.get('nickname_witness_policy')
        if proof.get('status') != 'PROVEN' or proof != prove_roster_binding(proof['source_inputs'],
                _legacy=policy not in FIRST_NICK_POLICIES,
                _witness_policy=policy if policy in FIRST_NICK_POLICIES else FIRST_NICK_POLICY):
            return False
        fields = ('posts', 'actors', 'forced_snapshot_complete')
        if any(snapshot.get(k) != proof['source_inputs']['snapshot'].get(k) for k in fields):
            return False
        expected = binding_receipt(proof, receipt['actor'], [p for p in snapshot['posts'] if p['actor'] == receipt['actor']])
        return expected == receipt
    except (KeyError, TypeError, ValueError):
        return False
