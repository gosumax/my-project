"""Read-only displayed-stack evidence. Physical seats do not identify actors."""
import copy
import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from .poker_reconstruction.models import RuleError
from .performance_work import note as work_note, scope as work_scope


class StackEvidenceProvider:
    def __init__(self, path, *, video_sha256):
        self.path = Path(path).resolve()
        with sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            self.metadata = {r['key']: json.loads(r['value']) for r in db.execute('select * from metadata')}
            if self.metadata.get('schema') != 'stack-evidence-timeline-v1':
                raise ValueError('Unsupported STACK evidence schema')
            if not video_sha256 or self.metadata.get('video_sha256') != video_sha256:
                raise ValueError('STACK source video identity mismatch')
            if self.metadata.get('fps') != 30.0:
                raise ValueError('STACK frame/time alignment unsupported')
            for row in db.execute('select width,height,raw_gray,exact_hash from visual_states'):
                with work_scope('stack/sqlite_pixel_validation',category='proof/pixels',
                        key=(str(self.path),row['exact_hash']),source_rows=1,input_bytes=len(row['raw_gray'])):
                    if len(row['raw_gray']) != row['width'] * row['height'] or hashlib.sha256(row['raw_gray']).hexdigest() != row['exact_hash']:
                        raise ValueError('STACK raw grayscale provenance mismatch')
            self.points = [dict(r) for r in db.execute('select t.*,s.source_state_id,s.ocr_raw,s.digit_gate_status,s.exact_hash '
                'from numeric_timeline t join visual_states s using(state_id) order by observation_id')]
            seen = {}
            ranges = self.metadata.get('coverage_ranges', [])
            for point in self.points:
                if point['status'] not in {'CONFIRMED_NUMERIC', 'LOW_CONFIDENCE_NUMERIC', 'UNKNOWN', 'AMBIGUOUS'}:
                    raise ValueError('STACK unsupported numeric status')
                segment = point['coverage_segment']
                if not 1 <= segment <= len(ranges) or not ranges[segment-1][0] <= point['frame_index'] <= ranges[segment-1][1]:
                    raise ValueError('STACK observation outside recorded coverage')
                if abs(point['timestamp'] - point['frame_index'] / 30.) > .000001:
                    raise ValueError('STACK timestamp/frame mismatch')
                for link in ('previous_observation_id', 'previous_known_observation_id'):
                    previous = seen.get(point[link]) if point[link] is not None else None
                    if point[link] is not None and (previous is None or any(
                            previous[k] != point[k] for k in ('table_id', 'seat_id', 'coverage_segment'))):
                        raise ValueError('STACK observation link crosses identity/coverage')
                previous = seen.get(point['previous_known_observation_id'])
                if point['observed_delta'] is not None and (previous is None or point['stack_value'] is None
                        or int(point['stack_value']) - int(previous['stack_value']) != point['observed_delta']):
                    raise ValueError('STACK delta arithmetic mismatch')
                seen[point['observation_id']] = point
        with work_scope('stack/sqlite_file_hash',category='proof/source_seal',key=str(self.path),hash_calls=1):
            self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def observations(self, table_id, seat_id, first_frame, last_frame):
        work_note("stack/observation_scan",category="lookup/scan",
            key=(str(self.path),table_id,seat_id,first_frame,last_frame),candidates_scanned=len(self.points))
        return [copy.deepcopy(p) for p in self.points if p['table_id'] == table_id
                and p['seat_id'] == seat_id and first_frame <= p['frame_index'] <= last_frame]

    def match_snapshot(self, fact):
        """Match distinct ROI representations by their common original RGB source."""
        from .start_stack_visibility import VisibilitySource
        verifier=VisibilitySource()
        sources={}
        for reading in fact.get('evidence',[]):
            checked=verifier.verify(reading)
            if checked is not None and (reading.get('table_id'),reading.get('session_id'),
                    reading.get('hand_id'),reading.get('seat')) == (
                    fact.get('table'),fact.get('session'),fact.get('hand_id'),fact.get('seat')):
                sources[reading['source_frame'],checked['timeline_pixel_sha256']]=(reading['source_pts'],reading['time_base'])
        points=[p for p in self.points if (p['table_id'],p.get('session_id'),p['seat_id']) ==
                (fact.get('table'),fact.get('session'),fact.get('seat')) and
                (p['frame_index'],p['exact_hash']) in sources and
                (p.get('source_pts'),json.loads(p.get('source_time_base') or 'null')) ==
                sources[p['frame_index'],p['exact_hash']]]
        if not points:
            return dict(status='UNKNOWN',reason='SNAPSHOT_SQLITE_SOURCE_IDENTITY_UNPROVEN',points=[])
        if any(p.get('stack_value') is not None and p['status']=='CONFIRMED_NUMERIC'
               and p['stack_value']!=fact.get('displayed_stack') for p in points):
            return dict(status='CONFLICT',reason='INDEPENDENT_STACK_OCR_CONFLICT',points=points)
        return dict(status='PROVEN',reason='SAME_RGB_SOURCE_PROJECTED_TO_BOTH_ROIS',points=points)

    def write_audit(self, output, records, *, start_stack=False, reconstruction_results=()):
        """Current production has no continuous occupancy/action-onset binding.

        Retain this missing proof explicitly instead of manufacturing an actor
        from a physical seat or treating a CHAT observation as action onset.
        """
        output = Path(output)
        simple=getattr(self,'simple_sources',None)
        if simple is not None:
            simple.write_audit(output)
            receipts=[p for r in records for p in simple.player_rows(r['observed_hand'],segment_id=r['segment_id'])]
            (output/'stack_start_stack.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'
                for r in receipts),encoding='utf-8')
            seats=[p for r in records for p in simple.seat_rows(r['observed_hand'])]
            summary=dict(enabled=True,policy=simple.__class__.__name__,start_stack_enabled=True,
                eligible_players=len(receipts),recovered_proven=sum(r['status']=='START_STACK_PROVEN' for r in receipts),
                start_stack_unknown=sum(r['status']=='START_STACK_UNKNOWN' for r in receipts),
                DISPLAYED_STACK_known=sum(r['DISPLAYED_STACK'] is not None for r in seats),
                fixed_seats=len(seats),verdict='SAVED_POLICY_CANDIDATE_NOT_FULL_HH_ACCEPTANCE')
            (output/'stack_evidence_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
            return summary
        receipts = []
        reconstruction={r['segment_id']:r for r in reconstruction_results}
        linked_observations=set();snapshot_candidates=0
        for record in records:
            hand = record['observed_hand']
            sitout_source=getattr(self,'sitout_sources',None)
            sitout_joined=reconstruction.get(record['segment_id'],{}).get('sitout_join_receipts',[])
            sitout_verified=[]
            if start_stack and sitout_joined and sitout_source is not None:
                from .sitout_hand_evidence import join_receipt as sitout_receipt
                sitout_verified=[sitout_receipt(p) for p in sitout_source.for_hand(hand)]
            carry_source=getattr(self,'carry_sources',None)
            carry_joined=reconstruction.get(record['segment_id'],{}).get('adjacent_stack_carry_join_receipts',[])
            carry_verified=[]
            if start_stack and carry_joined and carry_source is not None:
                from .adjacent_stack_carry import join_receipts as carry_receipts
                carry_verified=[r for p in carry_source.for_hand(hand) for r in carry_receipts(p)]
            constant_source=getattr(self,'constant_sources',None)
            constant_joined=reconstruction.get(record['segment_id'],{}).get('constant_stack_balance_join_receipts',[])
            constant_verified=[]
            if start_stack and constant_joined and constant_source is not None:
                from .constant_stack_balance import join_receipt as constant_receipt
                constant_verified=[constant_receipt(p) for p in constant_source.for_hand(hand)]
            direct_source=getattr(self,'direct_sources',None)
            direct_joined=reconstruction.get(record['segment_id'],{}).get('direct_start_stack_join_receipts',[])
            direct_verified=[]
            if start_stack and direct_joined and direct_source is not None:
                from .direct_start_stack import join_receipt as direct_receipt
                direct_verified=[direct_receipt(p) for p in direct_source.for_hand(hand)]
            rows = [r for m in hand['messages'] for r in m['source_rows']]
            players = sorted({m.get('player') or m.get('actor') for m in hand['messages']
                              if m.get('player') or m.get('actor')})
            # Parsed events are the existing engine's actor authority.
            from .poker_reconstruction.timeline import parse_rows
            players = sorted({e.player for e in parse_rows(rows) if e.player})
            for player in players:
                receipt=dict(hand_id=hand.get('hand_id'), segment_id=record['segment_id'],
                    table_id=hand['table_id'], session_id=hand['session_id'], player=player,
                    seat_id=None, observed_stack=None, forced_amount_added_back=None,
                    voluntary_amount_added_back=None, calculated_start_stack=None,
                    status='START_STACK_UNKNOWN', reason='SEAT_PLAYER_EPOCH_AND_ACTION_INTERVAL_UNPROVEN',
                    source_observation_id=None, source_state_id=None)
                statuses=[r for r in sitout_joined if r.get('actor')==player and r in sitout_verified]
                if len(statuses)==1:
                    s=statuses[0]
                    receipt.update(seat_id=s['seat'],occupancy_epoch=None,player_status='SIT_OUT',
                        stack_requirement='NOT_REQUIRED_SIT_OUT',reason='SIT_OUT_NUMERIC_STACK_NOT_REQUIRED',
                        sitout_binding_receipt=s,numeric_authority=None)
                    receipts.append(receipt);continue
                candidates=[r for r in reconstruction.get(record['segment_id'],{}).get('start_stack_join_receipts',[])
                            if r['actor']==player]
                conflicts=[r for r in reconstruction.get(record['segment_id'],{}).get('start_stack_join_rejections',[])
                           if r.get('actor')==player and r.get('reason') in
                           {'INDEPENDENT_STACK_OCR_CONFLICT','INDEPENDENT_START_STACK_PROOF_CONFLICT'}]
                if conflicts:
                    receipt.update(status='START_STACK_CONFLICT',reason=conflicts[0]['reason'],
                                   seat_id=conflicts[0]['seat'],conflict_receipts=copy.deepcopy(conflicts))
                if len(candidates)==1 and not conflicts:
                    joined=candidates[0];fact=joined['math_fact']
                    context=(int(str(hand['table_id']).removeprefix('T')),hand['session_id'],hand['hand_id'])
                    if (fact.get('table'),fact.get('session'),fact.get('hand_id'))==context:
                        evidence={(r.get('source_frame'),r.get('pixel_sha256')) for r in fact.get('evidence',[])}
                        points=[p for p in self.points if p['table_id']==context[0] and p['seat_id']==joined['seat']
                                and p.get('exact_hash') and (p['frame_index'],p['exact_hash']) in evidence
                                and p.get('stack_value')==fact.get('displayed_stack')]
                        snapshot=self.match_snapshot(fact)
                        if snapshot['points']:points=snapshot['points']
                        receipt.update(seat_id=joined['seat'],occupancy_epoch=joined['occupancy_epoch'],
                            snapshot_candidate_start_stack=joined['START_STACK_CANDIDATE'],
                            canonical_binding_and_math_receipt=copy.deepcopy(joined),
                            matched_stack_observation_ids=[p['observation_id'] for p in points],
                            reason=joined['blocker'] if points else 'SNAPSHOT_SQLITE_SOURCE_IDENTITY_UNPROVEN')
                        snapshot_candidates+=1
                        if points:
                            point=min(points,key=lambda p:p['frame_index'])
                            receipt.update(observed_stack=point['stack_value'],
                                source_observation_id=point['observation_id'],source_state_id=point['source_state_id'])
                            linked_observations.update(p['observation_id'] for p in points)
                        if start_stack and joined.get('START_STACK') is not None:
                            from .poker_math_engine import calculate_start_stack
                            checked=calculate_start_stack(dict(fact,status='PROVEN_FOR_MATH'),first_visible_contract=True)
                            if snapshot['status']=='CONFLICT':
                                receipt.update(status='START_STACK_CONFLICT',reason=snapshot['reason'])
                            elif (snapshot['status']=='PROVEN' and checked.get('status')=='PASS'
                                  and checked.get('START_STACK')==joined['START_STACK']):
                                receipt.update(status='START_STACK_PROVEN',reason='FIRST_VISIBLE_POST_FORCED_SOURCE_AND_CANONICAL_ACTOR',
                                    observed_stack=fact['displayed_stack'],calculated_start_stack=int(joined['START_STACK']),
                                    forced_amount_added_back=sum(int(v) for v in fact['forced_bets'].values()),
                                    voluntary_amount_added_back=0,numeric_authority='DIRECT_LIVE_THREE_SOURCE_OCR',
                                    sqlite_numeric_observations=copy.deepcopy(points),
                                    binding_receipt=dict(player=player,seat_id=joined['seat'],table_id=hand['table_id'],
                                        session_id=hand['session_id'],hand_id=hand['hand_id'],
                                        occupancy_epoch=joined['occupancy_epoch'],canonical_forced_event_ids=fact['canonical_forced_event_ids']))
                joint=[r for r in reconstruction.get(record['segment_id'],{}).get('joint_start_stack_join_receipts',[])
                       if r.get('actor')==player]
                provider=getattr(self,'joint_sources',None)
                if start_stack and len(joint)==1 and provider is not None and not conflicts:
                    from .joint_start_stack import join_receipt
                    proven=[p for p in provider.for_hand(hand) if p['actor']==player and join_receipt(p)==joint[0]]
                    if len(proven)==1:
                        p=proven[0];j=joint[0]
                        receipt.update(status='START_STACK_PROVEN',reason=j['rule'],seat_id=p['seat'],
                            observed_stack=str(p['displayed_stack']),calculated_start_stack=p['calculated_start_stack'],
                            forced_amount_added_back=p['forced_amount_added_back'],voluntary_amount_added_back=0,
                            source_observation_id=j['source_observation_ids'][0],source_state_id=j['source_state_ids'][0],
                            numeric_authority='SAVED_SQL_OCR_EXACT_NATIVE_RGB_PROJECTION',joint_binding_receipt=j)
                        linked_observations.update(j['source_observation_ids'])
                carried=[r for r in carry_joined if r.get('actor')==player and r in carry_verified]
                if start_stack and len(carried)==1 and not conflicts and receipt['status']!='START_STACK_CONFLICT':
                    c=carried[0]
                    if receipt['status']=='START_STACK_PROVEN':
                        if (receipt['seat_id'],receipt['calculated_start_stack'])!=(c['seat'],c['calculated_start_stack']):
                            receipt.update(status='START_STACK_CONFLICT',reason='INDEPENDENT_START_STACK_PROOF_CONFLICT')
                    else:
                        receipt.update(status='START_STACK_PROVEN',reason=c['rule'],seat_id=c['seat'],
                            calculated_start_stack=c['calculated_start_stack'],observed_stack=None,
                            forced_amount_added_back=None,voluntary_amount_added_back=None,
                            source_observation_id=None,source_state_id=None,occupancy_epoch=None,
                            numeric_authority='PROVEN_PREVIOUS_END_BALANCE',carry_binding_receipt=c)
                constants=[r for r in constant_joined if r.get('actor')==player and r in constant_verified]
                if start_stack and len(constants)==1 and not conflicts and receipt['status']!='START_STACK_CONFLICT':
                    c=constants[0]
                    if receipt['status']=='START_STACK_PROVEN':
                        if (receipt['seat_id'],receipt['calculated_start_stack'])!=(c['seat'],c['calculated_start_stack']):
                            receipt.update(status='START_STACK_CONFLICT',reason='INDEPENDENT_START_STACK_PROOF_CONFLICT')
                    else:
                        receipt.update(status='START_STACK_PROVEN',reason=c['rule'],seat_id=c['seat'],
                            calculated_start_stack=c['calculated_start_stack'],observed_stack=str(c['displayed_stack']),
                            forced_amount_added_back=c['forced_amount_added_back'],voluntary_amount_added_back=0,
                            source_observation_id=c['source_observation_ids'][0],source_state_id=c['source_state_ids'][0],
                            occupancy_epoch=None,numeric_authority='SOURCE_REPLAYED_CONSTANT_POST_FORCED_BALANCE',
                            constant_binding_receipt=c)
                        linked_observations.update(c['source_observation_ids'])
                direct=[r for r in direct_joined if r.get('actor')==player and r in direct_verified]
                if start_stack and len(direct)==1 and not conflicts and receipt['status']!='START_STACK_CONFLICT':
                    d=direct[0]
                    if receipt['status']=='START_STACK_PROVEN':
                        if (receipt['seat_id'],receipt['calculated_start_stack'])!=(d['seat'],d['calculated_start_stack']):
                            receipt.update(status='START_STACK_CONFLICT',reason='INDEPENDENT_START_STACK_PROOF_CONFLICT')
                    else:
                        receipt.update(status='START_STACK_PROVEN',reason=d['rule'],seat_id=d['seat'],
                            calculated_start_stack=d['calculated_start_stack'],observed_stack=str(d['displayed_stack']),
                            forced_amount_added_back=d['forced_amount_added_back'],voluntary_amount_added_back=0,
                            source_observation_id=None,source_state_id=None,occupancy_epoch=None,
                            numeric_authority='SOURCE_REPLAYED_DIRECT_FIRST_NUMBER',direct_binding_receipt=d)
                # A joined numeric snapshot is not proof of the action interval.
                # calculated_start_stack stays null until those gates are met.
                receipts.append(receipt)
        summary = dict(enabled=True, sqlite_sha256=self.sha256, metadata=self.metadata,
            observations=len(self.points), numeric_observations=sum(p['stack_value'] is not None for p in self.points),
            unknown_observations=sum(p['stack_value'] is None for p in self.points),
            aggregate_intervals=sum(p['delta_kind'] == 'AGGREGATE_CONSTRAINT' for p in self.points),
            actions_with_recovered_amount=0, stack_supported_actions=0, stack_conflicts=0,
            actor_bound_observations=len(linked_observations), unbound_observations=len(self.points)-len(linked_observations),
            snapshot_candidates_joined=snapshot_candidates,
            start_stack_enabled=start_stack, eligible_players=len(receipts),
            recovered_proven=sum(r['status']=='START_STACK_PROVEN' for r in receipts),
            recovered_supported=0, start_stack_unknown=sum(r['status']=='START_STACK_UNKNOWN' for r in receipts) if start_stack else 0,
            start_stack_conflicts=sum(r['status']=='START_STACK_CONFLICT' for r in receipts), accuracy=None,
            verdict='STACK_EVIDENCE_INTEGRATION_PARTIAL',
            start_stack_verdict='START_STACK_FROM_TIMELINE_NOT_PROVEN',
            existing_output_changed=False)
        (output / 'stack_evidence_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), 'utf-8')
        with (output / 'stack_observations.jsonl').open('w', encoding='utf-8') as stream:
            for point in self.points:
                stream.write(json.dumps(point, ensure_ascii=False) + '\n')
        if start_stack:
            with (output / 'stack_start_stack.jsonl').open('w', encoding='utf-8') as stream:
                for receipt in receipts:
                    stream.write(json.dumps(receipt, ensure_ascii=False) + '\n')
        return summary


def evaluate_action(state, event, previous, current, *, binding):
    """Check a typed action with existing poker rules; never guess its kind.

    Binding is reserved for an independent, source-bound interval provider.
    Current SQLite does not supply it, so the production adapter cannot call
    this function with a proved binding.
    """
    required = ('source_identity', 'hand_membership', 'occupancy_epoch', 'single_action_interval', 'visibility_complete')
    if any(not binding.get(k) for k in required):
        return dict(status='UNKNOWN', reason='ACTION_BINDING_UNPROVEN')
    if previous['table_id'] != current['table_id'] or previous['seat_id'] != current['seat_id']:
        return dict(status='UNKNOWN', reason='ROUTE_MISMATCH')
    if current['frame_index'] <= previous['frame_index'] or not (
            previous['frame_index'] < binding.get('action_frame', -1) <= current['frame_index']):
        return dict(status='UNKNOWN', reason='FRAME_ALIGNMENT_UNPROVEN')
    if current.get('unknown_between'):
        return dict(status='UNKNOWN', reason='AGGREGATE_ONLY', observed_delta=current.get('observed_delta'))
    if any(p.get('stack_value') is None for p in (previous, current)):
        return dict(status='UNKNOWN', reason='NUMERIC_UNKNOWN')
    if any(p.get('status') not in {'CONFIRMED_NUMERIC', 'LOW_CONFIDENCE_NUMERIC'} for p in (previous, current)):
        return dict(status='UNKNOWN', reason='NUMERIC_AMBIGUOUS')
    if previous.get('coverage_segment') != current.get('coverage_segment'):
        return dict(status='UNKNOWN', reason='COVERAGE_GAP')
    paid = int(previous['stack_value']) - int(current['stack_value'])
    if paid <= 0 or event.kind not in {'CALL', 'BET', 'RAISE'}:
        return dict(status='UNKNOWN', reason='NO_TYPED_DEBIT')
    if event.player not in state.players or binding.get('player') != event.player:
        return dict(status='UNKNOWN', reason='ACTOR_MISMATCH')
    old = state.players[event.player].street_commit
    amount = old + paid if event.semantics == 'TO' else paid
    if event.semantics == 'UNKNOWN':
        return dict(status='UNKNOWN', reason='AMOUNT_SEMANTICS_UNPROVEN')
    if event.amount is not None and event.amount != amount:
        return dict(status='STACK_CONFLICT', reason='CHAT_STACK_AMOUNT_MISMATCH', paid_delta=paid)
    candidate = replace(event, amount=amount)
    try:
        copy.deepcopy(state).apply(candidate)
    except RuleError as exc:
        return dict(status='STACK_CONFLICT', reason=str(exc), paid_delta=paid)
    confirmed = all(p.get('status') == 'CONFIRMED_NUMERIC' for p in (previous, current))
    return dict(status='CONFIRMED' if confirmed else 'SUPPORTED', paid_delta=paid,
                candidate_amount=amount, new_street_contribution=old + paid,
                event_kind=event.kind, authoritative=False, candidate_only=True, proof_origin='CALLER_SUPPLIED')


def evaluate_start_stack(point, *, proof):
    """Separate arithmetic candidate from complete hand accounting authority."""
    result = dict(status='START_STACK_UNKNOWN', calculated_start_stack=None,
                  authoritative=False, candidate_only=True, proof_origin='CALLER_SUPPLIED',
                  observed_stack=point.get('stack_value'), forced_amount_added_back=None,
                  voluntary_amount_added_back=None, source_observation_id=point.get('observation_id'))
    required = ('source_identity', 'boundary', 'occupancy_epoch', 'hand_membership',
                'ordered_complete_debits', 'visibility_complete')
    if any(not proof.get(k) for k in required):
        return dict(result, reason='START_STACK_PROOF_INCOMPLETE')
    if point.get('unknown_between') or proof.get('unknown_before_observation'):
        return dict(result, reason='UNKNOWN_INTERVAL')
    start = proof.get('hand_start_frame')
    if start is None or point['frame_index'] < start or proof.get('video_starts_mid_hand'):
        return dict(result, reason='HAND_BOUNDARY_UNPROVEN')
    if point.get('stack_value') is None:
        return dict(result, reason='NUMERIC_UNKNOWN')
    debits = proof.get('debits', [])
    if any(type(d.get('amount')) is not int or d['amount'] < 0 or
           not d.get('source_refs') or not start <= d.get('frame', -1) <= point['frame_index']
           for d in debits):
        return dict(result, reason='DEBIT_AMOUNT_OR_ORDER_UNPROVEN')
    forced = sum(d['amount'] for d in debits if d['kind'] in {'ANTE', 'SB', 'BB'})
    voluntary = sum(d['amount'] for d in debits if d['kind'] in {'CALL', 'BET', 'RAISE'})
    if any(d['kind'] not in {'ANTE', 'SB', 'BB', 'CALL', 'BET', 'RAISE'} for d in debits):
        return dict(result, reason='UNSUPPORTED_PRE_OBSERVATION_EVENT')
    calculated = int(point['stack_value']) + forced + voluntary
    result.update(calculated_start_stack=calculated, forced_amount_added_back=forced,
                  voluntary_amount_added_back=voluntary)
    accounting = proof.get('accounting')
    if not accounting or not accounting.get('complete'):
        return dict(result, status='START_STACK_SUPPORTED', reason='FULL_HAND_ACCOUNTING_UNPROVEN')
    if calculated != accounting['final_stack'] + accounting['debits'] - accounting['returns'] - accounting['wins']:
        return dict(result, status='START_STACK_CONFLICT', reason='HAND_ACCOUNTING_MISMATCH')
    if point.get('status') != 'CONFIRMED_NUMERIC':
        return dict(result, status='START_STACK_SUPPORTED', reason='LOW_CONFIDENCE_NUMERIC')
    return dict(result, status='START_STACK_PROVEN', reason='ORDERED_DEBITS_AND_HAND_ACCOUNTING_MATCH')
