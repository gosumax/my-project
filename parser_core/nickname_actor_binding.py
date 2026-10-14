"""Positive literal-prefix binding for one actor in a complete observed roster.

Unknown labels on unrelated seats never supply a complement or an alias.
Every accepted actor has its own source-bound, uniquely matching prefix.
"""
import copy
import math
from collections import Counter
from pathlib import Path

import cv2

from .nickname_roster_binding import (_nickname_triples, _legacy_nickname_triples,
    _verify_gray_source, matches, nickname_hand_bounds, nickname_epoch_matches, FIRST_NICK_POLICY,
    FIRST_NICK_POLICY_V1, FIRST_NICK_POLICIES, nickname_witness_floor)
from .seat_epoch_state import SeatEpochMachine

RULE = 'UNIQUE_LITERAL_PREFIX_ACTOR_IN_COMPLETE_ROSTER_V1'


def _deal_empty_refs(deal):
    """Reproduce positive empty seats; no absence-of-cards inference."""
    from .empty_seat_evidence import EmptySeatButtonModel
    joint = deal.get('joint_seat_frames', [])
    if deal.get('proof') != 'COMPLETE_EMPTY_BUTTON_AND_FROZEN_CARDS_V1' or len(joint) != 3:
        return {}
    model = EmptySeatButtonModel(Path(__file__).resolve().parents[1])
    configured = set(model.slots[f"slot_{deal['table_id']:02d}"]['seats'])
    occupied = deal['occupied_seats']
    expected_frames = list(range(deal['source_frame']-2, deal['source_frame']+1))
    if [r['source_frame'] for r in joint] != expected_frames:
        return {}
    prior_pts = None
    empties = None
    for row in joint:
        empty = row['empty_seats']
        if (any(row[k] != deal[k] for k in ('table_id', 'session_id', 'hand_id', 'time_base')) or
                row['occupied_seats'] != occupied or len(empty) != len(set(empty)) or
                set(empty) & set(occupied) or set(empty) | set(occupied) != configured or
                (empties is not None and empty != empties) or type(row['source_pts']) is not int or
                (prior_pts is not None and row['source_pts'] <= prior_pts) or
                any(row['empty_evidence'][s].get('seat') != s or
                    not model.verify(row['empty_evidence'][s], row) for s in empty)):
            return {}
        empties, prior_pts = empty, row['source_pts']
    if prior_pts != deal['source_pts']:
        return {}
    return {s:joint[-1]['empty_evidence'][s]['pixel_sha256'] for s in empties}


def prove_actor_binding(inputs, actor, *, _legacy=False, _witness_policy=FIRST_NICK_POLICY):
    def unknown(reason):
        return dict(status='UNKNOWN', reason=reason, binding_rule=RULE, actor=actor)
    try:
        key = tuple(inputs['context'])
        snapshot, deal = inputs['snapshot'], inputs['deal']
        if (len(key) != 3 or not deal or
                (snapshot['table'], snapshot['session'], snapshot['hand_id']) != key or
                (deal['table_id'], deal['session_id'], deal['hand_id']) != key):
            return unknown('ACTOR_CONTEXT_CONFLICT')
        actors, seats = snapshot['actors'], deal['occupied_seats']
        if (not snapshot.get('forced_snapshot_complete') or deal.get('evidence_status') != 'PROVEN' or
                len(seats) < 2 or len(set(seats)) != len(seats) or len(set(actors)) != len(actors) or
                len(actors) != len(seats) or actor not in actors or
                any(not isinstance(a, str) or not a for a in actors)):
            return unknown('COMPLETE_DISTINCT_ROSTER_OR_OCCUPANCY_UNKNOWN')
        posts = snapshot['posts']
        if Counter(p['actor'] for p in posts if p['role'] == 'ANTE') != Counter(actors):
            return unknown('COMPLETE_DISTINCT_ANTE_ROSTER_UNKNOWN')
        confidence, window = inputs['confidence'], inputs['capture_frames']
        if (type(confidence) not in (int, float) or not math.isfinite(confidence) or
                not .95 <= confidence <= 1 or type(window) is not int or window <= 0):
            return unknown('ACTOR_SOURCE_WINDOW_OR_CONFIDENCE_UNKNOWN')
        prefix = inputs['forced_rows']
        for row in prefix:
            pixels = _verify_gray_source(row)
            response = row['ocr_response']
            if (type(row.get('score')) not in (int, float) or not math.isfinite(row['score']) or
                    not 0 <= row['score'] <= 1 or
                    row['text'] != (response.get('raw_text') or '').strip() or
                    row['score'] != (response.get('score') if response.get('status') == 'OK' else 0) or
                    pixels.shape[0] != row['row_bounds'][1]-row['row_bounds'][0]):
                return unknown('ACTOR_FORCED_OCR_SOURCE_CONFLICT')
        from .seat_forced_binding import parse_forced_snapshot
        reproduced, reason = parse_forced_snapshot(prefix, key[2], len(seats), confidence)
        fields = ('actor', 'role', 'amount', 'source_refs', 'source_index', 'phase')
        if (not reproduced or reproduced['actors'] != actors or
                [{k:p.get(k) for k in fields} for p in reproduced['posts']] !=
                [{k:p.get(k) for k in fields} for p in posts]):
            return unknown(reason or 'ACTOR_PREFIX_NOT_REPRODUCIBLE')
        lo, hi = reproduced['header_index'], reproduced['deal_index']
        if sorted(r['index'] for r in prefix if lo <= r['index'] <= hi) != list(range(lo, hi+1)):
            return unknown('ACTOR_FORCED_PREFIX_SOURCE_GAP')
        sources = {(r['source_frame'], r['source_pts'], tuple(r['time_base'])) for r in prefix}
        if len(sources) != 1 or any((r['table_id'], r['session_id'], r['hand_id']) != key for r in prefix):
            return unknown('ACTOR_PREFIX_SOURCE_CONTEXT_CONFLICT')
        source, pts, base = next(iter(sources))
        deal_frame, deal_pts = deal['source_frame'], deal['source_pts']
        if (any(type(v) is not int for v in (source, pts, deal_frame, deal_pts)) or
                len(base) != 2 or any(type(v) is not int or v <= 0 for v in base) or
                not deal_frame <= source <= deal_frame+window or list(base) != deal['time_base'] or
                (source == deal_frame and pts != deal_pts) or (source > deal_frame and pts <= deal_pts) or
                any(p['phase'] != 'BEFORE_DEAL' or
                    (p['source_frame'], p['source_pts'], tuple(p['time_base'])) != (source, pts, base) for p in posts)):
            return unknown('ACTOR_PREFIX_SOURCE_TIME_CONFLICT')
        for row in inputs['nick_rows']:
            response = row.get('ocr_response', {})
            score = response.get('score')
            if response.get('status') == 'OK' and (type(score) not in (int, float) or
                    not math.isfinite(score) or not 0 <= score <= 1):
                return unknown('ACTOR_NICK_CONFIDENCE_INVALID')
        start,end=(deal_frame,deal_frame+window) if _legacy else nickname_hand_bounds(inputs)
        nickname_witness=_legacy_nickname_triples if _legacy else _nickname_triples
        nick_confidence=confidence if _legacy or _witness_policy==FIRST_NICK_POLICY_V1 else nickname_witness_floor(_witness_policy)
        triples = nickname_witness(inputs['nick_rows'], key, start, end,
                                   nick_confidence, deal, (source, pts))
        positives = [(t['frames'][-1], seat) for seat in seats for t in triples.get(seat, [])
                     if t['constraint']['kind'] == 'PREFIX' and
                     [a for a in actors if matches(a, t['constraint'])] == [actor]]
        if not positives:
            return unknown('UNIQUE_POSITIVE_ACTOR_PREFIX_UNKNOWN')
        earliest = min(f for f, _ in positives)
        targets = {s for f, s in positives if f == earliest}
        if len(targets) != 1:
            return unknown('ACTOR_PREFIX_SEAT_CONFLICT')
        seat = next(iter(targets))
        anchor = max(source, earliest)
        labels = {s:[t for t in ts if t['frames'][-1] <= anchor] for s, ts in triples.items()}
        if (any(not matches(actor, t['constraint']) for t in labels[seat]) or
                any(matches(actor, t['constraint']) for s, ts in labels.items() if s != seat for t in ts)):
            return unknown('ACTOR_EARLIEST_PREFIX_CONTRADICTION')
        triggers = [t for t in inputs['hand_triggers']
                    if (t['table_id'], t['session_id'], t['hand_id']) == key]
        if len({t['source_frame'] for t in triggers}) != 1 or triggers[0]['source_frame'] > deal_frame:
            return unknown('ACTOR_HAND_START_UNKNOWN')
        events = [e for e in inputs['state_receipts'] if e['frame'] <= anchor]
        active = [e for e in events if (e['table'], e['session']) == key[:2] and e['frame'] >= deal_frame]
        empty_refs = _deal_empty_refs(deal) if any(e['kind'] == 'EMPTY' for e in active) else {}
        def positive_partition_empty(e):
            return (e['kind'] == 'EMPTY' and e['seat'] != seat and e['seat'] not in seats and
                    e['frame'] == deal_frame and e['evidence_status'] == 'PROVEN' and
                    e['seat'] in empty_refs and e['source_ref'] == empty_refs[e['seat']])
        if any(e['status'] != 'APPLIED' or e['hand_id'] != key[2] or
               (e['kind'] != 'OCCUPIED' and not positive_partition_empty(e)) for e in active):
            return unknown('ACTOR_SEAT_STATE_CHANGED_OR_UNKNOWN')
        machine = SeatEpochMachine()
        for trigger in inputs['hand_triggers']:
            machine.hand(trigger['table_id'], trigger['session_id'], trigger['hand_id'], trigger['source_frame'])
        for event in events:
            machine.observe(event['table'], event['session'], event['seat'], event['kind'], event['frame'],
                **{k:event[k] for k in ('actor', 'hand_id', 'evidence_status', 'source_ref', 'source_index')})
        machine.build()
        if machine.audit != events or any((e['table'], e['session']) == key[:2] and e['status'] != 'APPLIED' for e in events):
            return unknown('ACTOR_OCCUPANCY_AUDIT_UNKNOWN')
        witness = [e for e in active if e['seat'] == seat and e['frame'] == deal_frame and
                   e['source_ref'] == deal['table_pixel_sha256'] and e['evidence_status'] == 'PROVEN']
        epochs = {e.get('epoch') for e in witness}
        if len(epochs) != 1 or None in epochs:
            return unknown('ACTOR_DEAL_MEMBERSHIP_WITNESS_UNKNOWN')
        epoch = next(iter(epochs))
        if (not machine.membership(*key[:2], seat, key[2], epoch)['participant'] or
                any(machine.epoch_at(*key[:2], seat, f) != epoch for f in
                    [source, anchor]) or
                any((machine.epoch_at(*key[:2],seat,f) != epoch if _legacy else
                     not nickname_epoch_matches(machine,key,seat,f,deal_frame,epoch))
                    for t in labels[seat] for f in t['frames'])):
            return unknown('ACTOR_PARTICIPANT_EPOCH_UNKNOWN')
        if any((e['table'], e['session']) == key[:2] and e['frame'] <= anchor for e in inputs['unresolved_events']):
            return unknown('ACTOR_UNRESOLVED_SEAT_EVENT')
        retained = copy.deepcopy(inputs)
        retained['nick_rows'] = [r for r in retained['nick_rows'] if r['source_frame'] <= anchor]
        retained['state_receipts'] = events
        retained['unresolved_events'] = [e for e in retained['unresolved_events'] if e['frame'] <= anchor]
        result=dict(status='PROVEN', reason=None, binding_rule=RULE, actor=actor, context=list(key),
                    seat=seat, epoch=epoch, forced_source_frame=source, forced_source_pts=pts,
                    deal_source_frame=deal_frame, binding_anchor_frame=anchor,
                    label_evidence=labels[seat], source_inputs=retained)
        if not _legacy:result['nickname_witness_policy']=_witness_policy
        return result
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, cv2.error):
        return unknown('ACTOR_SOURCE_RECEIPT_INVALID')


def binding_receipt(proof, actor, posts):
    if proof.get('status') != 'PROVEN' or actor != proof.get('actor'):
        return None
    table, session, hand = proof['context']
    return dict(table=table, session=session, hand_id=hand, actor=actor, seat=proof['seat'],
                epoch=proof['epoch'], binding_rule=RULE, forced_source_frame=proof['forced_source_frame'],
                label_confirmation_frame=proof['binding_anchor_frame'], binding_anchor_frame=proof['binding_anchor_frame'],
                label_evidence=copy.deepcopy(proof['label_evidence']),
                forced_source_refs=[p['source_refs'] for p in posts], actor_prefix_proof=proof)


def verify_binding_receipt(receipt, snapshot):
    try:
        proof = receipt['actor_prefix_proof']
        policy=proof.get('nickname_witness_policy')
        if proof.get('status') != 'PROVEN' or proof != prove_actor_binding(proof['source_inputs'], receipt['actor'],
                _legacy=policy not in FIRST_NICK_POLICIES,
                _witness_policy=policy if policy in FIRST_NICK_POLICIES else FIRST_NICK_POLICY):
            return False
        if any(snapshot.get(k) != proof['source_inputs']['snapshot'].get(k) for k in
               ('table', 'session', 'hand_id', 'posts', 'actors', 'forced_snapshot_complete')):
            return False
        return receipt == binding_receipt(proof, receipt['actor'],
                                         [p for p in snapshot['posts'] if p['actor'] == receipt['actor']])
    except (KeyError, TypeError, ValueError):
        return False
