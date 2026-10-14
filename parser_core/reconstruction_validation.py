"""Revalidate a uniquely proved economic history without changing observations."""
from collections import Counter
import copy

from parser_core.poker_hand_assembly import make_message
from parser_core.poker_logic_engine import validate_hand
from parser_core.poker_diagnostics import diagnose_hand
from parser_core.poker_numeric_evidence import numeric_alternatives
from parser_core.poker_reconstruction.timeline import parse_text as parse_economic_text
from parser_core.card_set_evidence import apply_to_hand as apply_card_set_to_hand
from parser_core.poker_amount_semantics import with_call_context


def validate_call_fallback(observed, result, *, hidden_card_source=None, no_sb_source=None):
    """Validate the existing candidate, or raw observations with joined facts.

    CALL context is mathematical authority only, never a reconstruction trace.
    """
    derived = result.get('derived_final_validation', {})
    if derived.get('candidate_hand') is not None:
        return dict(validation=derived['validation'], diagnostics=derived['diagnostics'],
                    diagnostics_ledger_view='SOURCE_PROVEN_DERIVED')
    checked = with_call_context(observed, result.get('simple_start_stack_join_receipts', []))
    return dict(validation=validate_hand(checked, hidden_card_source=hidden_card_source,no_sb_source=no_sb_source),
                diagnostics=diagnose_hand(checked, hidden_card_source=hidden_card_source,no_sb_source=no_sb_source),
                diagnostics_ledger_view='OBSERVED')


def validate_reconstruction(observed, result, *, nickname_canonicalization=None, hidden_card_source=None,no_sb_source=None):
    def reject(reason):
        return dict(accepted=False, reason=reason, candidate_hand=None)
    if (not result.get('strict_source_uniqueness') or result.get('exhausted') or
            result.get('solution_count') != 1 or result.get('status') not in
            {'EXACT','AUTO_RECOVERED','RECOVERED_BY_LOGIC_MATH'}):
        return reject('EXHAUSTIVE_UNIQUE_SOURCE_SOLUTION_NOT_PROVEN')
    if (result.get('dropped_events') or result.get('inferred_events') or
            result.get('chosen_identity_hypothesis',{}).get('aliases')):
        return reject('SOURCE_OR_IDENTITY_TRANSFORMATION_NOT_AUTHORIZED')
    originals=[r for m in observed['messages'] for r in m['source_rows']]
    source_id=lambda r:r.get('stable_row_id',r.get('row_id'))
    by_id={source_id(r):r for r in originals}
    events=result.get('kept_events',[]);trace=result.get('diagnostic_trace',[])
    ids=[source_id(r) for e in events for r in e['source'].get('rows',[])]
    if (None in by_id or len(by_id)!=len(originals) or Counter(ids)!=Counter(by_id.keys())
            or len(events)!=len(trace)):
        return reject('RECONSTRUCTION_SOURCE_COVERAGE_NOT_EXACT')
    candidate=with_call_context(observed, result.get('simple_start_stack_join_receipts',
        observed.get('call_start_stack_receipts', [])));messages=[];proofs=[]
    prior={};prior_street=None
    kinds=dict(SB='SMALL_BLIND',BB='BIG_BLIND',RETURN='UNCALLED_RETURN',START_HAND='HAND_HEADER')
    for event,step in zip(events,trace):
        event_rows=event['source']['rows']
        rows=[copy.deepcopy(by_id[source_id(r)]) for r in event_rows]
        if any({k:v for k,v in r.items() if k!='source_index'} !=
               {k:v for k,v in original.items() if k!='source_index'} for r,original in zip(event_rows,rows)):
            return reject('RECONSTRUCTION_SOURCE_FIELDS_CHANGED')
        if len(rows)>1 and not all(a.get('frame_id') is not None and a['frame_id']==b.get('frame_id')
                and a.get('epoch')==b.get('epoch') and a.get('table_id')==b.get('table_id')
                and a.get('session_id')==b.get('session_id') and isinstance(a.get('row_index'),int)
                and b.get('row_index')==a['row_index']+1 for a,b in zip(rows,rows[1:])):
            return reject('PHYSICAL_CONTINUATION_NOT_PROVEN')
        message=make_message(rows,require_pixel_cards=True,pixel_sources=by_id)
        actor=message.get('player')
        if actor in (nickname_canonicalization or {}):
            message['original_player']=actor
            message['player']=nickname_canonicalization[actor]
            message['field_sources']['player']='SAVED_OCCUPANT_NICKNAME_SELECTION'
        expected=event['street'] if event['kind']=='STREET' else kinds.get(event['kind'],event['kind'])
        compatible=(message['action']==expected or event['kind']=='START_HAND' and message['action'] in {'HAND_ID','NEW_HAND'}
                    or event['kind']=='INFO' and message['action'] in {'TABLE_INFO','NO_SHOW','NEW_HAND','BLIND_LEVEL'})
        if not compatible or (event.get('player') and event['player']!=message.get('player')):
            return reject('OBSERVED_ACTION_OR_ACTOR_NOT_REPRODUCIBLE')
        if step.get('decision')!='KEEP' or step['candidate_event']['kind']!=event['kind']:
            return reject('RECONSTRUCTION_TRACE_ACTION_MISMATCH')
        before=prior.get(event.get('player'),0) if prior_street==step['street'] else 0
        after=step['street_contributions'].get(event.get('player'),0)
        if event['kind'] in {'ANTE','SB','BB','CALL','BET','RAISE','RETURN','WIN'}:
            numeric_proof = event['source'].get('numeric_amount_proof')
            if numeric_proof:
                original_event = parse_economic_text(event['source']['raw_text'],
                    {k:v for k,v in event['source'].items() if k!='numeric_amount_proof'})
                alternatives = numeric_alternatives(original_event)
                if not any(e.amount==event['amount'] and e.source['numeric_amount_proof']==numeric_proof
                           for e in alternatives):
                    return reject('NUMERIC_SOURCE_CANDIDATE_NOT_REPRODUCIBLE')
            # The raw amount remains in immutable source rows and observed_amount.
            amount=(after-before if event['kind'] in {'CALL','BET'} else after if event['kind']=='RAISE' else event['amount'])
            if type(amount) is not int or amount<=0:
                return reject('EXACT_POSITIVE_AMOUNT_NOT_PROVEN')
            if (message['amount'] is not None and event['amount'] is not None and
                    str(message['amount'])!=str(event['amount'])):
                return reject('OBSERVED_AMOUNT_CHANGED')
            message.update(observed_amount=message['amount'],amount=str(amount),
                           all_in=event['all_in'],amount_interpretation='DERIVED_EXACT_ECONOMIC_EFFECT')
            message['field_sources']['amount']='UNIQUE_SOURCE_POKER_MATH'
            proof=dict(source_row_ids=list(message['source_row_ids']),kind=event['kind'],
                       actor=event.get('player'),observed_amount=message['observed_amount'],
                       derived_amount=amount,chosen_interpretation=step['candidate_event']['semantics'],
                       prior_street_contribution=before,resulting_street_contribution=after,
                       street=step['street'],rule='EXHAUSTIVE_UNIQUE_ORDERED_ECONOMIC_TRACE')
            if event['kind'] == 'CALL':
                # This candidate amount is already the proved paid delta. Keep
                # the original DELTA/TO choice separately in chosen_interpretation.
                proof['call_contract'] = 'ADDED'
            message['amount_proof']=proof;proofs.append(proof)
            if numeric_proof:
                message.update(raw_amount_text=numeric_proof['raw_amount_text'],
                               parsed_amount=numeric_proof['parsed_amount'],resolved_amount=amount,
                               resolution_reason='EXHAUSTIVE_UNIQUE_ORDERED_ECONOMIC_TRACE',
                               amount_source=numeric_proof['source'])
                proof['numeric_source_proof']=numeric_proof
        messages.append(message)
        prior=step['street_contributions'];prior_street=step['street']
    candidate.update(messages=messages,semantics=dict(call='ADDED',bet='ADDED',raise_to='TOTAL'),
                     reconstruction_proofs=proofs)
    # Preserve every original source row, including raw OCR and card evidence.
    if [r for m in candidate['messages'] for r in m['source_rows']] != originals:
        return reject('OBSERVED_SOURCE_ORDER_CHANGED')
    apply_card_set_to_hand(candidate)
    validation=validate_hand(candidate,hidden_card_source=hidden_card_source,no_sb_source=no_sb_source)
    diagnostics=diagnose_hand(candidate,hidden_card_source=hidden_card_source,no_sb_source=no_sb_source)
    accepted=(validation['status']=='FULLY_VALID_HAND' and diagnostics['final_status']=='FULLY_VALID'
              and validation['complete_candidate'] and not validation['certification_blockers'])
    return dict(accepted=bool(accepted),reason=None if accepted else 'DERIVED_FINAL_VALIDATION_UNRESOLVED',
                validation=validation,diagnostics=diagnostics,candidate_hand=candidate,
                observed_hand_changed=False,proof_scope='SOURCE_PRESERVING_ACTIONS; SEATS_AND_START_STACK_STILL_REQUIRED')
