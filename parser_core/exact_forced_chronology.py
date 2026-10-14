"""Reproduce an exact seat label and forced-post chronology from saved sources.

This supplies an additional proof route, not a fallback confidence threshold.
Every first-visible, source, membership and exact-label requirement is retained.
"""
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from .nickname_roster_binding import (_nickname_triples, _legacy_nickname_triples,
    _verify_gray_source, matches, nickname_hand_bounds, nickname_epoch_matches, FIRST_NICK_POLICY,
    FIRST_NICK_POLICIES, nickname_witness_floor)
from .seat_epoch_state import SeatEpochMachine
from .seat_forced_binding import parse_forced_snapshot
from .start_stack_visibility import verify_first_visible_fact

RULE = 'EXACT_LABEL_COMPLETE_ACTOR_POSTS_BEFORE_FIRST_NUMBER_V1'
PREFIX_RULE = 'UNIQUE_ACTOR_PREFIX_COMPLETE_POSTS_BEFORE_FIRST_NUMBER_V1'


def load_sources(output):
    root = Path(output)
    names = ('seat_forced_inputs.jsonl', 'seat_forced_summary.json',
             'dealing_cards_triggers.jsonl', 'start_stack_live_receipts.jsonl')
    if not all((root / name).is_file() for name in names):
        return {}
    def rows(name):
        with (root / name).open(encoding='utf-8') as stream:
            return [json.loads(line) for line in stream if line.strip()]
    summary = json.loads((root / names[1]).read_text('utf-8'))
    inputs = defaultdict(lambda: dict(nick_rows=[], forced_rows=[]))
    for row in rows(names[0]):
        key = (row['table_id'], row['session_id'], row['hand_id'])
        if row['kind'] == 'NICK':
            inputs[key]['nick_rows'].append(row)
        elif row['kind'] == 'CHAT':
            response = row['ocr_response']
            inputs[key]['forced_rows'].append(dict(row, text=(response.get('raw_text') or '').strip(),
                score=response.get('score') if response.get('status') == 'OK' else 0))
    deals = {}
    for row in rows(names[3]):
        if row.get('status') != 'DEAL_COMPLETE':
            continue
        key = (row['table_id'], row['session_id'], row['hand_id'])
        # Repeated identical receipts are harmless; conflicting receipts are not
        # resolved by file order.
        if key not in deals:
            deals[key] = row
        elif deals[key] != row:
            deals[key] = None
    triggers = rows(names[2])
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    for key, item in inputs.items():
        item.update(context=list(key), deal=deals.get(key),
                    hand_triggers=triggers, state_receipts=summary['state_receipts'],
                    unresolved_events=summary['unresolved_state_events'], source_hashes=hashes)
        ends = [r['source_frame'] for r in summary.get('observed_hand_ends', [])
                if (r['table'], r['session'], r['hand_id']) == key]
        if len(ends) == 1:
            item['observed_capture_end_frame'] = ends[0]
    return dict(inputs)


def prove(fact, snapshot, matched, inputs):
    """Return a self-contained proof or None; unknown sources grant no authority."""
    try:
        key = (fact['table'], fact['session'], fact['hand_id'])
        if not inputs or inputs.get('context') != list(key) or not verify_first_visible_fact(fact):
            return None
        if (not snapshot.get('forced_snapshot_complete') or not matched or
                (snapshot['table'], snapshot['session'], snapshot['hand_id']) != key):
            return None
        for event in matched:
            post = event['canonical_provenance']['post']
            if ((event['table_id'], event['session_id'], event['hand_id']) !=
                    (f'T{key[0]:02d}', key[1], key[2]) or
                    (post['table'], post['session'], post['hand_id']) != key or
                    post['epoch'] != fact['occupancy_epoch'] or post['seat'] != fact['seat']):
                return None
        receipts = [e['canonical_provenance']['binding_receipt'] for e in matched]
        receipt = receipts[0]
        if any(r != receipt for r in receipts):
            return None
        prefix_binding = receipt.get('binding_rule') == 'UNIQUE_LITERAL_PREFIX_ACTOR_IN_COMPLETE_ROSTER_V1'
        if prefix_binding:
            from .nickname_actor_binding import verify_binding_receipt
            if not verify_binding_receipt(receipt, snapshot):
                return None
        elif receipt.get('binding_rule') != 'FIRST_EXACT_LABEL_HISTORICAL_EPOCH':
            return None
        actor, seat, epoch = receipt['actor'], receipt['seat'], receipt['epoch']
        if (seat != fact['seat'] or epoch != fact['occupancy_epoch'] or
                (receipt['table'], receipt['session'], receipt['hand_id']) != key or
                {e['player'] for e in matched} != {actor} or
                any(e.get('canonical_conflict') or e.get('canonical_duplicate_of') or
                    e.get('seat') != seat for e in matched)):
            return None
        own = [p for p in snapshot['posts'] if p['actor'] == actor]
        signature = lambda posts: Counter((p['role'], p['amount'], tuple(p['source_refs'])) for p in posts)
        if not own or signature(own) != signature(e['canonical_provenance']['post'] for e in matched):
            return None
        deal = inputs['deal']
        if (not deal or deal.get('evidence_status') != 'PROVEN' or
                (deal['table_id'], deal['session_id'], deal['hand_id']) != key or
                seat not in deal['occupied_seats']):
            return None
        prefix = inputs['forced_rows']
        for row in prefix:
            pixels = _verify_gray_source(row)
            response = row['ocr_response']
            if (type(row.get('score')) not in (int, float) or
                    not math.isfinite(row['score']) or not 0 <= row['score'] <= 1 or
                    row['text'] != (response.get('raw_text') or '').strip() or
                    row['score'] != (response.get('score') if response.get('status') == 'OK' else 0) or
                    pixels.shape[0] != row['row_bounds'][1]-row['row_bounds'][0]):
                return None
        reproduced, _ = parse_forced_snapshot(prefix, key[2], len(deal['occupied_seats']), .95)
        fields = ('actor', 'role', 'amount', 'source_refs', 'source_index', 'phase')
        if (not reproduced or reproduced['actors'] != snapshot['actors'] or
                [{k:p.get(k) for k in fields} for p in reproduced['posts']] !=
                [{k:p.get(k) for k in fields} for p in snapshot['posts']]):
            return None
        lo, hi = reproduced['header_index'], reproduced['deal_index']
        if sorted(r['index'] for r in prefix if lo <= r['index'] <= hi) != list(range(lo, hi+1)):
            return None
        sources = {(r['source_frame'], r['source_pts'], tuple(r['time_base'])) for r in prefix}
        if len(sources) != 1 or any((r['table_id'], r['session_id'], r['hand_id']) != key for r in prefix):
            return None
        source, pts, base = next(iter(sources))
        numeric = fact['evidence'][0]
        if (source != receipt['forced_source_frame'] or source < deal['source_frame'] or
                source > numeric['source_frame'] or list(base) != numeric['time_base'] or
                Fraction(pts*base[0], base[1]) > Fraction(numeric['source_pts']*base[0], base[1]) or
                (source == numeric['source_frame']) != (pts == numeric['source_pts']) or
                any(p.get('phase') != 'BEFORE_DEAL' or
                    (p.get('source_frame'),p.get('source_pts'),tuple(p.get('time_base',[]))) !=
                    (source,pts,base) for p in snapshot['posts'])):
            return None
        anchor = max(fact['frames'][-1], receipt['label_confirmation_frame'], source)
        policies={p for p in (receipt.get('nickname_witness_policy'),
            receipt.get('roster_proof',{}).get('nickname_witness_policy'),
            receipt.get('actor_prefix_proof',{}).get('nickname_witness_policy')) if p in FIRST_NICK_POLICIES}
        if len(policies)>1:return None
        policy=next(iter(policies),None)
        first_nick=policy is not None
        if first_nick:
            start, end = nickname_hand_bounds(inputs)
            if not start <= receipt['binding_anchor_frame'] <= anchor <= end:
                return None
        elif anchor != receipt['binding_anchor_frame']:
            return None  # No retrospective extension beyond the original binding proof.
        occupancy = [e for e in inputs['state_receipts']
                     if (e['table'], e['session'], e['hand_id'], e['seat']) == (*key, seat)
                     and e['kind'] == 'OCCUPIED' and e['evidence_status'] == 'PROVEN'
                     and e['status'] == 'APPLIED' and e['epoch'] == epoch
                     and e['frame'] == deal['source_frame'] <= anchor
                     and e['source_ref'] == deal['table_pixel_sha256']]
        if not occupancy:
            return None  # A later same-hand membership cannot backfill this deal.
        for row in inputs['nick_rows']:
            response = row.get('ocr_response', {})
            score = response.get('score')
            if response.get('status') == 'OK' and (type(score) not in (int, float) or
                    not math.isfinite(score) or not 0 <= score <= 1):
                return None
        witness=_nickname_triples if first_nick else _legacy_nickname_triples
        start=nickname_hand_bounds(inputs)[0] if first_nick else deal['source_frame']
        triples = witness(inputs['nick_rows'], key, start, anchor, nickname_witness_floor(policy), deal, (source, pts))
        labels = receipt['label_evidence']
        if not labels:
            return None
        if prefix_binding:
            if any(label not in triples.get(seat, []) for label in labels):
                return None
        else:
            if any(label.get('actor') != actor for label in labels):
                return None
            for label in labels:
                found = [t for t in triples.get(seat, []) if t['frames'] == label['frames'] and
                         t['constraint'] == dict(kind='EXACT', text=actor) and
                         t['source_rows'][-1]['source_ref'] == label['source_ref']]
                if len(found) != 1:
                    return None
            exact_seats = {s for s, ts in triples.items() for t in ts
                           if t['constraint'] == dict(kind='EXACT', text=actor)}
            if exact_seats != {seat} or any(not matches(actor,t['constraint']) for t in triples.get(seat, [])):
                return None
        machine = SeatEpochMachine()
        for trigger in inputs['hand_triggers']:
            machine.hand(trigger['table_id'], trigger['session_id'], trigger['hand_id'], trigger['source_frame'])
        for event in inputs['state_receipts']:
            machine.observe(event['table'], event['session'], event['seat'], event['kind'], event['frame'],
                **{k:event[k] for k in ('actor','hand_id','evidence_status','source_ref','source_index')})
        machine.build()
        if machine.audit != inputs['state_receipts'] or not machine.membership(*key[:2],seat,key[2],epoch)['participant']:
            return None
        frames = [source,anchor,*fact['frames'],*(f for label in labels for f in label['frames'])]
        if any((not nickname_epoch_matches(machine,key,seat,f,deal['source_frame'],epoch)
                if first_nick else machine.epoch_at(*key[:2],seat,f) != epoch) for f in frames):
            return None
        if any((e['table'],e['session']) == key[:2] and e['frame'] <= anchor and
               e['status'] != 'APPLIED' for e in machine.audit):
            return None
        if any((e['table'],e['session']) == key[:2] and e['frame'] <= anchor for e in inputs['unresolved_events']):
            return None
        retained = copy.deepcopy(inputs)
        retained['nick_rows'] = [r for r in retained['nick_rows'] if r['source_frame'] <= anchor]
        return dict(rule=PREFIX_RULE if prefix_binding else RULE, actor=actor, seat=seat, epoch=epoch,
                    forced_source_frame=source, forced_source_pts=pts, forced_time_base=list(base),
                    first_numeric_source_frame=numeric['source_frame'],
                    first_numeric_source_pts=numeric['source_pts'], first_numeric_time_base=list(base),
                    binding_anchor_frame=anchor, binding_receipt=copy.deepcopy(receipt),
                    exact_binding_inputs=retained, forced_snapshot=copy.deepcopy(snapshot),
                    matched_events=copy.deepcopy(matched),
                    first_visible_contract=fact['first_visible_proof']['contract'])
    except (KeyError,TypeError,ValueError,IndexError,ZeroDivisionError):
        return None


def verify(fact):
    """Reproduce this additional route at consumption, including its own debits."""
    try:
        proof = fact['forced_chronology_proof']
        if proof.get('rule') not in {RULE, PREFIX_RULE}:
            return False
        matched = proof['matched_events']
        regenerated = prove(fact['vision_fact'], proof['forced_snapshot'], matched,
                            proof['exact_binding_inputs'])
        bets = {e['canonical_provenance']['post']['role']:
                e['canonical_provenance']['post']['amount'] for e in matched}
        return (regenerated is not None and regenerated == proof and
                fact['forced_bets'] == bets and fact['actor'] == proof['actor'] and
                fact['canonical_forced_event_ids'] == [e['canonical_event_id'] for e in matched] and
                all(fact[k] == fact['vision_fact'][k] for k in
                    ('table', 'session', 'hand_id', 'seat', 'occupancy_epoch', 'frames',
                     'evidence', 'displayed_stack', 'first_visible_proof')))
    except (KeyError, TypeError, ValueError):
        return False
