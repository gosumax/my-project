"""Additive, fail-closed adapter for the NOT_PROVEN reconstruction engine.

Production assembly owns boundaries; the engine never edits the timeline or
the existing hand verdict. Recovery proposals retain their complete evidence.
"""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from dataclasses import replace

from parser_core.poker_reconstruction.assembly import candidate_assemblies
from parser_core.poker_reconstruction.reconstruct import reconstruct_hand
from parser_core.poker_reconstruction.timeline import parse_rows
from parser_core.hand_lifecycle import classify_lifecycle
from parser_core.hand_lifecycle import table_key
from parser_core.poker_math_engine import calculate_start_stack

CONFIG = Path(__file__).resolve().parents[1] / 'configs/poker_reconstruction.json'


def reconstruction_config():
    config = json.loads(CONFIG.read_text('utf-8'))['poker_reconstruction']
    for key in ('enabled', 'allow_logic_math_output'):
        if type(config[key]) is not bool:
            raise ValueError(f'poker_reconstruction.{key} must be boolean')
    if type(config['max_nodes']) is not int or config['max_nodes'] <= 0:
        raise ValueError('poker_reconstruction.max_nodes must be positive integer')
    return config


def proven_start_stacks(hand, math_results, *, first_visible_contract=False, stack_provider=None):
    """Join exact actors; only reproduced first-visible facts grant authority."""
    context = (table_key(hand['table_id']), hand['session_id'], hand['hand_id'])
    events, conflicting_ids = {}, set()
    for message in hand['messages']:
        for row in message['source_rows']:
            if not row.get('canonical_event_id') or row.get('canonical_conflict') or row.get('canonical_duplicate_of'):
                continue
            event_id=row['canonical_event_id']
            prior=events.get(event_id)
            fields=('table_id','session_id','hand_id','seat','player','amount','action_type')
            if prior and any(prior.get(k)!=row.get(k) for k in fields):conflicting_ids.add(event_id)
            events[event_id]=row
    for event_id in conflicting_ids:events.pop(event_id,None)
    stacks, receipts, rejected = {}, [], []
    conflicted = set()
    for fact in math_results:
        if (fact.get('table'), fact.get('session'), fact.get('hand_id')) != context:
            continue
        reason = None
        ids = fact.get('canonical_forced_event_ids') or []
        matched = [events[event_id] for event_id in ids if event_id in events]
        actors = {r.get('player') for r in matched}
        if fact.get('status') != 'PASS' and fact.get('snapshot_arithmetic_status') != 'PASS':
            reason = fact.get('blocker') or 'START_STACK_MATH_UNKNOWN'
        elif not ids or len(matched) != len(ids) or len(actors) != 1 or None in actors:
            reason = 'STACK_CANONICAL_ACTOR_JOIN_UNPROVEN'
        elif any((table_key(r['table_id']), r['session_id'], r.get('hand_id')) != context or
                 r.get('seat') != fact.get('seat') or
                 r['canonical_provenance']['post'].get('epoch') != fact.get('occupancy_epoch') or
                 r['canonical_provenance'].get('binding_receipt',{}).get('seat') != fact.get('seat') or
                 r['canonical_provenance'].get('binding_receipt',{}).get('epoch') != fact.get('occupancy_epoch') or
                 r['canonical_provenance'].get('binding_receipt',{}).get('actor') != r.get('player') for r in matched):
            reason = 'STACK_CONTEXT_SEAT_EPOCH_CONFLICT'
        elif {r['canonical_provenance']['post']['role']: r['amount'] for r in matched} != fact.get('forced_bets'):
            reason = 'STACK_CANONICAL_FORCED_CONFLICT'
        else:
            checked = calculate_start_stack(dict(fact, status='PROVEN_FOR_MATH'),first_visible_contract=first_visible_contract)
            if checked.get('snapshot_arithmetic_status') != 'PASS' or checked.get('START_STACK_CANDIDATE') != (
                    fact.get('START_STACK_CANDIDATE') or fact.get('START_STACK')):
                reason = 'STACK_MATH_RECEIPT_NOT_REPRODUCIBLE'
        if reason:
            rejected.append(dict(seat=fact.get('seat'), reason=reason))
            continue
        if first_visible_contract and stack_provider is not None:
            snapshot=stack_provider.match_snapshot(fact)
            if snapshot['status']=='CONFLICT':
                rejected.append(dict(seat=fact.get('seat'),actor=next(iter(actors)),reason=snapshot['reason']))
                continue
        actor = next(iter(actors))
        value = fact.get('START_STACK_CANDIDATE') or fact.get('START_STACK')
        if not str(value).isdigit():
            rejected.append(dict(seat=fact.get('seat'), reason='STACK_INTEGER_AMOUNT_UNPROVEN'))
            continue
        prior = next((r for r in receipts if r['actor'] == actor), None)
        if prior and (prior['seat'], prior['occupancy_epoch'], prior['START_STACK_CANDIDATE']) != (
                fact['seat'], fact['occupancy_epoch'], value):
            conflicted.add(actor)
            rejected.append(dict(seat=fact['seat'], reason='STACK_ACTOR_FACT_CONFLICT'))
            continue
        proven=first_visible_contract and checked.get('status')=='PASS' and checked.get('START_STACK')==value
        blocker=None if proven else checked.get('blocker') or 'CLEAN_WINDOW_TIMING_UNPROVEN'
        receipts.append(dict(actor=actor, seat=fact['seat'], occupancy_epoch=fact['occupancy_epoch'],
                             START_STACK_CANDIDATE=value, START_STACK=value if proven else None,
                             blocker=blocker, math_fact=copy.deepcopy(fact)))
        if proven:stacks[actor]=int(value)
        else:rejected.append(dict(seat=fact['seat'], reason=blocker))
    for actor in conflicted:
        stacks.pop(actor, None)
    receipts = [r for r in receipts if r['actor'] not in conflicted]
    return stacks, receipts, rejected


def reconstruct_production_hands(records, *, config=None, start_stack_results=(), stack_provider=None):
    config = reconstruction_config() if config is None else config
    results = []
    if not config['enabled']:
        return results, {'enabled': False, 'hands_seen': 0, 'statuses': {},
                         'engine_verdict': 'NOT_PROVEN'}
    records = copy.deepcopy(records)
    if any('lifecycle' not in r for r in records):
        classify_lifecycle(records)
    for record in records:
        hand = copy.deepcopy(record['observed_hand'])
        rows = copy.deepcopy([row for message in hand['messages'] for row in message['source_rows']])
        for index, row in enumerate(rows, 1):
            row['source_index'] = index
        lifecycle = record['lifecycle']['status']
        if lifecycle != 'COMPLETE':
            status = 'TRUNCATED_END' if lifecycle == 'END_PARTIAL' else 'TRUNCATED_START'
            results.append(dict(table=hand['table_id'], hand_id=hand['hand_id'],
                segment_id=record['segment_id'], session_id=hand['session_id'],
                status=status, reconstruction_status=status, lifecycle_status=lifecycle,
                decision_reason=record['lifecycle']['reason'],
                reconstruction_reason=record['lifecycle']['reason'],
                first_source_frame=record['lifecycle']['first_source_frame'], last_source_frame=record['lifecycle']['last_source_frame'],
                evidence_refs=record['lifecycle']['evidence_refs'],
                raw_events=copy.deepcopy([r for m in hand['messages'] for r in m['source_rows']]),
                assembled_events=copy.deepcopy(hand['messages']),
                unresolved_rows=copy.deepcopy([r for m in hand['messages'] for r in m['source_rows']]),
                output_eligible=False, output_blockers=[lifecycle], output_disposition='PARTIAL_DIAGNOSTIC',
                engine_verdict='NOT_PROVEN', solver_invoked=False))
            continue
        events = parse_rows(rows)
        assemblies, exhausted, graph = candidate_assemblies(rows)
        strict=config.get('strict_source_uniqueness',False)
        if strict:
            def all_call_interpretations(event):
                return replace(event,semantics='UNKNOWN') if event.kind=='CALL' and event.semantics=='DELTA' else event
            events=[all_call_interpretations(event) for event in events]
            assemblies=[[all_call_interpretations(event) for event in assembly] for assembly in assemblies]
        stacks, stack_receipts, stack_rejected = proven_start_stacks(hand, start_stack_results,
            first_visible_contract=config.get('first_visible_stack_contract',False),stack_provider=stack_provider)
        # Unknown facts never enter the solver. Actor identity comes only from
        # the same source-bound canonical forced events already in this hand.
        result = reconstruct_hand(events, table=hand['table_id'],
            hand_id=hand['hand_id'] if hand['start_observed'] else None,
            at_eof=not hand['next_boundary_observed'],
            max_nodes=config['max_nodes'], assemblies=assemblies,
            assembly_exhausted=exhausted, start_stacks=stacks,strict_source_uniqueness=strict)
        result.update(segment_id=record['segment_id'], session_id=hand['session_id'],
                      candidate_grouping_graph=graph, lifecycle_status=lifecycle, solver_invoked=True,
                      joined_start_stacks=stacks, start_stack_join_receipts=stack_receipts,
                      joined_stack_snapshot_candidates={r['actor']:r['START_STACK_CANDIDATE'] for r in stack_receipts},
                      start_stack_join_rejections=stack_rejected)
        result['engine_raw_events'] = result['raw_events']
        # Exact unmodified production rows and assembled messages are retained
        # independently from the solver's event interpretation.
        result['raw_events'] = copy.deepcopy([row for m in hand['messages'] for row in m['source_rows']])
        result['assembled_events'] = copy.deepcopy(hand['messages'])
        result['dropped_stale'] = copy.deepcopy(result['dropped_events'])
        result['player_contributions'] = {
            name: player['gross_commit'] for name, player in result['players'].items()}
        result['returns'] = copy.deepcopy(result['uncalled_returns'])
        result['reconstruction_status'] = result['status']
        result['reconstruction_reason'] = result['decision_reason']
        reasons = []
        if result['status'] not in {'EXACT', 'AUTO_RECOVERED', 'RECOVERED_BY_LOGIC_MATH'}:
            reasons.append('NO_UNIQUE_COMPLETE_EXACT_SOLUTION')
        if result['dropped_events']:
            reasons.append('SOURCE_ROW_REMOVAL_NOT_AUTHORIZED')
        if result['inferred_events']:
            reasons.append('INVENTED_ACTION_NOT_AUTHORIZED')
        if result.get('chosen_identity_hypothesis', {}).get('aliases'):
            reasons.append('IDENTITY_ALIAS_NOT_AUTHORIZED')
        kept_ids = Counter(row['stable_row_id'] for event in result['kept_events']
                           for row in event['source'].get('rows', []))
        if kept_ids != Counter(row['stable_row_id'] for row in rows):
            reasons.append('KEPT_SOURCE_COVERAGE_NOT_EXACT')
        if any(len(event['source'].get('rows', [])) > 1 and
               event['source'].get('physical_adjacency') is False
               for event in result['kept_events']):
            reasons.append('UNPROVEN_CROSS_FRAME_ASSEMBLY')
        if result['status'] == 'RECOVERED_BY_LOGIC_MATH' and not config['allow_logic_math_output']:
            reasons.append('LOGIC_MATH_OUTPUT_DISABLED')
        # Existing strict pixel/OCR/boundary/identity evidence gates remain the
        # authority. A balanced pot cannot bypass them.
        if strict and config['allow_logic_math_output']:
            from parser_core.reconstruction_validation import validate_reconstruction
            result['derived_final_validation']=validate_reconstruction(hand,result)
        if not record['fully_validated'] and not result.get('derived_final_validation',{}).get('accepted'):
            reasons.append('EXISTING_PRODUCTION_VALIDATION_UNRESOLVED')
        if result.get('pot_delta') != 0 or not result.get('payout_eligible'):
            reasons.append('EXACT_ELIGIBLE_ACCOUNTING_NOT_PROVEN')
        if result['inferred_amounts'] and result['status'] != 'RECOVERED_BY_LOGIC_MATH':
            reasons.append('INFERENCE_IN_EXACT_RESULT')
        result['output_eligible'] = not reasons
        result['output_blockers'] = reasons
        result['output_disposition'] = 'ADDITIONAL_VALIDATION' if not reasons else 'UNRESOLVED_CANDIDATE'
        if reasons:
            result['unresolved_rows'] = copy.deepcopy(result['raw_events'])
        else:
            result['unresolved_rows'] = []
        result['engine_verdict'] = 'NOT_PROVEN'
        results.append(result)
    expected = Counter(row['stable_row_id'] for r in records
        for m in r['observed_hand']['messages'] for row in m['source_rows'])
    actual = Counter(row['stable_row_id'] for r in results for row in r['raw_events'])
    if expected != actual:
        raise ValueError('Reconstruction adapter lost or invented production source rows')
    return results, {'enabled': True, 'allow_logic_math_output': config['allow_logic_math_output'],
        'strict_source_uniqueness':config.get('strict_source_uniqueness',False),
        'engine_verdict': 'NOT_PROVEN', 'hands_seen': len(results),
        'statuses': dict(Counter(r['reconstruction_status'] for r in results)),
        'output_eligible': sum(r['output_eligible'] for r in results),
        'logic_math_candidates': sum(r['status'] == 'RECOVERED_BY_LOGIC_MATH' for r in results),
        'source_rows': sum(actual.values()), 'old_output_changed': False}


def write_reconstruction(output, records, *, config=None, stack_provider=None, stack_start_stack=False):
    output = Path(output)
    math_path = output / 'start_stack_math.jsonl'
    math_results = [json.loads(line) for line in math_path.read_text('utf-8').splitlines()
                    if line.strip()] if math_path.exists() else []
    results, summary = reconstruct_production_hands(records, config=config, start_stack_results=math_results,stack_provider=stack_provider)
    summary['start_stacks_joined'] = sum(len(r.get('joined_start_stacks', {})) for r in results)
    summary['stack_snapshot_candidates_joined'] = sum(len(r.get('joined_stack_snapshot_candidates', {})) for r in results)
    for filename, selected in (
            ('poker_reconstruction.jsonl', [r for r in results if r['lifecycle_status'] == 'COMPLETE']),
            ('reconstructed_candidates.jsonl', [r for r in results if r['lifecycle_status'] == 'COMPLETE' and not r['output_eligible']]),
            ('reconstruction_validated_hands.jsonl', [r for r in results if r['output_eligible']])):
        with (output / filename).open('w', encoding='utf-8') as stream:
            for result in selected:
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
    (output / 'reconstruction_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')
    if stack_provider is not None:
        options={'reconstruction_results':results} if summary.get('strict_source_uniqueness') else {}
        stack_provider.write_audit(output, records, start_stack=stack_start_stack,**options)
    return summary
