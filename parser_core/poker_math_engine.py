"""Experimental Decimal contribution ledger; no visual pot authority."""
from decimal import Decimal
from time import perf_counter
from parser_core.poker_amount_semantics import resolve_call, call_contract, call_start_stack

VERSION = 'experimental-math-2-observed-and-prefix-ledgers'


def validate_math(ledger, payouts, returns):
    start = perf_counter()
    players = {}
    for name in sorted(set(ledger) | set(payouts) | set(returns)):
        entries = ledger.get(name, {})
        gross = sum(entries.values(), Decimal(0))
        returned = returns.get(name, Decimal(0))
        players[name] = {**{k: str(entries.get(k, Decimal(0))) for k in
                           ['ante', 'small_blind', 'big_blind', 'preflop_contribution',
                            'flop_contribution', 'turn_contribution', 'river_contribution']},
                         'gross_contribution': str(gross), 'uncalled_return': str(returned),
                         'net_contribution': str(gross-returned), 'payout': str(payouts.get(name, Decimal(0)))}
    net = sum((Decimal(p['net_contribution']) for p in players.values()), Decimal(0))
    paid = sum(payouts.values(), Decimal(0))
    delta = net-paid
    errors = []
    if delta != 0:
        errors.append('MATH_DELTA')
    if any(Decimal(p['net_contribution']) < 0 for p in players.values()):
        errors.append('RETURN_EXCEEDS_CONTRIBUTION')
    return {'pass': not errors, 'players': players, 'net_total': str(net), 'payout_total': str(paid),
            'delta': str(delta), 'errors': errors, 'elapsed_ms': (perf_counter()-start)*1000}


def audit_observed_ledger(hand):
    """Bookkeeping over every observed message, independent of betting legality.

    A running net-minus-paid before the payout is NOT a conservation failure.
    Missing fields and impossible deltas are exposed, never filled with facts.
    """
    from collections import defaultdict
    from decimal import InvalidOperation
    from parser_core.poker_hand_assembly import MONEY_ACTIONS
    ledger = defaultdict(lambda: defaultdict(Decimal))
    payouts, returns = defaultdict(Decimal), defaultdict(Decimal)
    street_contributions = defaultdict(Decimal)
    street, bet = 'PREFLOP', Decimal(0)
    steps, missing, suspicious = [], [], []
    call_ledger_known = True
    excluded = set(hand.get('excluded_message_indices', []))

    def snapshot():
        gross = sum((sum(v.values(), Decimal(0)) for v in ledger.values()), Decimal(0))
        back = sum(returns.values(), Decimal(0))
        paid = sum(payouts.values(), Decimal(0))
        return {'street': street, 'current_bet': str(bet),
                'total_contributions': str(gross), 'total_returns': str(back),
                'total_payouts': str(paid), 'current_delta': str(gross-back-paid),
                'contributions_by_player': {n: str(sum(v.values(), Decimal(0))) for n, v in ledger.items()},
                'contributions_by_street': {s: str(sum(v.get(s.lower()+'_contribution', Decimal(0)) for v in ledger.values()))
                                             for s in ['PREFLOP', 'FLOP', 'TURN', 'RIVER']},
                'forced_contributions': {s: str(sum(v.get(s, Decimal(0)) for v in ledger.values()))
                                        for s in ['ante', 'small_blind', 'big_blind']},
                'returns_by_player': {n: str(v) for n, v in returns.items()},
                'payouts_by_player': {n: str(v) for n, v in payouts.items()},
                'street_contributions_by_player': {n: str(v) for n, v in street_contributions.items()}}

    for index, msg in enumerate(hand['messages']):
        before = snapshot()
        action, player = msg['action'], msg.get('player')
        effect = {'contribution': '0', 'return': '0', 'payout': '0', 'status': 'NO_MONEY_EFFECT'}
        own = street_contributions.get(player, Decimal(0))
        required = bet-own if player else None
        if index in excluded:
            effect['status'] = 'EXCLUDED_CANDIDATE_MESSAGE'
        elif action in {'FLOP', 'TURN', 'RIVER'}:
            street, bet = action, Decimal(0)
            street_contributions = defaultdict(Decimal, {n: Decimal(0) for n in street_contributions})
        elif action in MONEY_ACTIONS:
            try:
                amount = Decimal(str(msg.get('amount')))
                if not amount.is_finite() or amount <= 0 or not player:
                    raise InvalidOperation
            except (InvalidOperation, ValueError):
                entry = {'message_index': index, 'source_row_ids': msg['source_row_ids'],
                         'action': action, 'player': player, 'street': street,
                         'RULE_TRIGGER': 'MISSING_AMOUNT' if player else 'PLAYER_IDENTITY_CONFLICT',
                         'PROVEN_CAUSE': 'UNKNOWN'}
                missing.append(entry)
                suspicious.append(entry)
                effect['status'] = 'UNKNOWN_MONEY_EFFECT'
                call_ledger_known = False
            else:
                delta = Decimal(0)
                if action in {'ANTE', 'SMALL_BLIND', 'BIG_BLIND'}:
                    delta = amount
                    ledger[player][action.lower()] += delta
                    if action != 'ANTE':
                        street_contributions[player] += delta
                        bet = max(bet, street_contributions[player])
                elif action == 'CALL':
                    semantics = call_contract(hand, msg)
                    proof = resolve_call(amount, own, bet, all_in=msg.get('all_in', False), contract=semantics,
                        start_stack=call_start_stack(hand, player) if call_ledger_known else None,
                        total_contribution=sum(ledger[player].values(), Decimal(0)), returned=returns[player])
                    if proof['status'] != 'RESOLVED':
                        call_ledger_known = False
                    effect['amount_resolution'] = proof
                    delta = (Decimal(proof['delta']) if proof['status'] == 'RESOLVED' else
                             amount-own if semantics == 'TOTAL' else amount)
                    if proof['status'] != 'RESOLVED':
                        suspicious.append({'message_index': index,
                            'RULE_TRIGGER': 'AMOUNT_SEMANTICS_UNPROVEN' if proof['status']=='AMBIGUOUS' else 'INVALID_CALL',
                            'PROVEN_CAUSE': 'UNKNOWN'})
                    street_contributions[player] += delta
                elif action == 'BET':
                    delta = amount
                    street_contributions[player] += delta
                    bet = street_contributions[player]
                elif action == 'RAISE':
                    delta = amount-own
                    street_contributions[player], bet = amount, amount
                    if delta <= 0:
                        call_ledger_known = False
                        suspicious.append({'message_index': index, 'RULE_TRIGGER': 'NON_POSITIVE_RAISE_DELTA', 'PROVEN_CAUSE': 'UNKNOWN'})
                elif action == 'UNCALLED_RETURN':
                    if amount > sum(ledger[player].values(), Decimal(0))-returns[player] or amount > own:
                        call_ledger_known = False
                    returns[player] += amount
                    street_contributions[player] -= amount
                    bet = max(street_contributions.values(), default=Decimal(0))
                    effect['return'] = str(amount)
                elif action == 'WIN':
                    payouts[player] += amount
                    effect['payout'] = str(amount)
                if action in {'CALL', 'BET', 'RAISE'}:
                    ledger[player][street.lower()+'_contribution'] += delta
                effect.update(contribution=str(delta), status='OBSERVED_BOOKKEEPING_ONLY')
        after = snapshot()
        if action == 'UNKNOWN_EVENT' and index not in excluded:
            call_ledger_known = False
        resolved = effect.get('amount_resolution', {})
        steps.append({'message_index': index, 'before': before, 'after': after, 'math_effect': effect,
                      'amount_to_call_before': str(required) if required is not None else None,
                      'hand_id': hand.get('hand_id'), 'street': before['street'],
                      'event_index': index, 'player': player, 'action': action,
                      'raw_amount': msg.get('raw_amount_text', msg.get('observed_amount', msg.get('amount'))),
                      'raw_source_event': msg.get('raw_text'),
                      'source_row_ids': list(msg['source_row_ids']),
                      'resolved_semantics': resolved.get('resolved_semantics', msg.get('amount_role')),
                      'delta': str(Decimal(effect['contribution'])-Decimal(effect['return'])),
                      'street_contribution_before': str(own),
                      'street_contribution_after': after['street_contributions_by_player'].get(player, '0'),
                      'ante_contribution': str(ledger.get(player, {}).get('ante', Decimal(0))),
                      'total_hand_contribution': str(sum(ledger.get(player, {}).values(), Decimal(0))-returns.get(player, Decimal(0))),
                      'current_highest_bet': before['current_bet'],
                      'stack_before': None, 'stack_after': None,
                      'pot_before': str(Decimal(before['total_contributions'])-Decimal(before['total_returns'])),
                      'pot_after': str(Decimal(after['total_contributions'])-Decimal(after['total_returns'])),
                      'inference_used': bool(msg.get('amount_proof')) or
                          resolved.get('source')=='POKER_MATH_INFERENCE' and resolved.get('status')=='RESOLVED',
                      'inference_reason': msg.get('resolution_reason') or
                          msg.get('amount_proof', {}).get('rule') or resolved.get('reason')})
    result = validate_math(ledger, payouts, returns)
    result.update(total_contributions=snapshot()['total_contributions'], total_returns=snapshot()['total_returns'],
                  contributions_by_street=snapshot()['contributions_by_street'],
                  scope='FULL_OBSERVED_LEDGER_NOT_PROOF_OF_LEGAL_ACTIONS', missing_fields=missing,
                  suspicious_rows=suspicious, steps=steps,
                  conservation_test_applicable=bool(hand.get('end_observed')))
    return result


def calculate_start_stack(facts, *, first_visible_contract=False):
    """Reverse proven debits from an immutable vision snapshot; never repair it."""
    from decimal import InvalidOperation
    result = dict(facts)
    result.update(START_STACK=None, status='UNKNOWN', blocker=facts.get('blocker'))
    if facts.get('displayed_stack_status') != 'VALID_FIXED_SEAT_ROI':
        result['blocker'] = result['blocker'] or 'DISPLAYED_STACK_UNKNOWN'
        return result
    if facts.get('status') != 'PROVEN_FOR_MATH':
        result['blocker'] = result['blocker'] or 'FORCED_CORRECTIONS_UNKNOWN'
        return result
    chronology_rule=facts.get('forced_chronology_proof', {}).get('rule')
    if (facts.get('forced_truth') in {'CANONICAL_COMPLETE_ACTOR_POSTS_EXACT_LABEL',
                                    'CANONICAL_COMPLETE_ACTOR_POSTS_ACTOR_PREFIX'} or
            facts.get('correction_proof',{}).get('rule') in {'COMPLETE_PHYSICAL_ACTOR_POST_SET_EXACT_LABEL',
                                                         'COMPLETE_PHYSICAL_ACTOR_POST_SET_ACTOR_PREFIX'} or
            chronology_rule in {'EXACT_LABEL_COMPLETE_ACTOR_POSTS_BEFORE_FIRST_NUMBER_V1',
                                'UNIQUE_ACTOR_PREFIX_COMPLETE_POSTS_BEFORE_FIRST_NUMBER_V1'}):
        from parser_core.exact_forced_chronology import verify
        if not verify(facts):
            result['blocker'] = 'EXACT_FORCED_CHRONOLOGY_NOT_REPRODUCIBLE'
            return result
    elif (first_visible_contract and facts.get('forced_truth')=='CANONICAL_COMPLETE_ACTOR_POSTS' and
            chronology_rule!='COMPLETE_FORCED_PREFIX_OBSERVED_BY_FIRST_POSTFORCED_NUMBER_V1' and
            not (facts.get('correction_proof',{}).get('posts') and
                 isinstance(facts.get('deal_complete_frame'),int) and
                 all(type(p.get('source_frame')) is int and p['source_frame']<=facts['deal_complete_frame']
                     for p in facts['correction_proof']['posts']))):
        result['blocker'] = 'FORCED_CHRONOLOGY_NOT_REPRODUCIBLE'
        return result
    if not facts.get('occupancy_epoch') or not facts.get('participant_proven'):
        result['blocker'] = 'OCCUPANCY_MEMBERSHIP_UNKNOWN'
        return result
    frames = facts.get('frames', [])
    from parser_core.start_stack_visibility import valid_numeric_witness
    if (not valid_numeric_witness(facts)
            or facts.get('deal_complete_frame') is None or frames[0] <= facts['deal_complete_frame']):
        result['blocker'] = 'FIRST_VISIBLE_STACK_TRIPLE_UNPROVEN'
        return result
    bets = facts.get('forced_bets')
    if not facts.get('forced_bets_debited_before_snapshot_proven'):
        result['blocker'] = 'FORCED_DEBIT_AT_SNAPSHOT_UNPROVEN'
        return result
    if not isinstance(bets, dict) or not bets or any(k not in {'ANTE','SB','BB'} for k in bets):
        result['blocker'] = 'FORCED_BET_UNKNOWN'
        return result
    try:
        displayed = Decimal(facts['displayed_stack'])
        amounts = [Decimal(str(v)) for v in bets.values() if not isinstance(v, bool)]
        if (len(amounts) != len(bets) or not displayed.is_finite() or displayed < 0
                or any(not v.is_finite() or v < 0 for v in amounts)):
            raise ValueError('Invalid observed amount')
    except (InvalidOperation, ValueError, TypeError, KeyError):
        result['blocker'] = 'OBSERVED_AMOUNT_INVALID'
        return result
    # A CHAT prefix and a stable numeric triple prove the arithmetic snapshot
    # contract, not absence of voluntary debits before numeric visibility.
    # No independent action-state interval provider is connected in this path.
    result.update(START_STACK_CANDIDATE=str(displayed+sum(amounts, Decimal(0))),
                  snapshot_arithmetic_status='PASS', START_STACK=None,
                  status='UNKNOWN', blocker='CLEAN_WINDOW_TIMING_UNPROVEN',
                  math_rule='DISPLAYED_PLUS_PROVEN_PRE_SNAPSHOT_FORCED_DEBITS',
                  start_stack_certified=False)
    if first_visible_contract:
        from parser_core.start_stack_visibility import CONTRACT, verify_first_visible_fact
        if verify_first_visible_fact(facts):
            result.update(START_STACK=result['START_STACK_CANDIDATE'],status='PASS',blocker=None,
                          initial_stack_contract=facts['first_visible_proof']['contract'],start_stack_certified=True)
        else:
            result['blocker']='FIRST_VISIBLE_POST_FORCED_SOURCE_UNPROVEN'
    return result


def write_start_stack_math(output, *, eligible_contexts=None, first_visible_contract=False):
    """Separate math outputs; input facts and existing hand analysis stay intact."""
    import json
    from pathlib import Path
    output = Path(output)
    facts = [json.loads(l) for l in (output/'start_stack_facts.jsonl').read_text('utf-8').splitlines() if l.strip()]
    if (output/'canonical_forced_events.jsonl').exists():
        from parser_core.canonical_events import reconcile_stack_facts, load_jsonl
        events=load_jsonl(output/'canonical_forced_events.jsonl')
        summary_path=output/'seat_forced_summary.json'
        snapshots=json.loads(summary_path.read_text('utf-8')).get('forced_snapshots',[]) if summary_path.exists() else []
        options={'first_visible_contract':True} if first_visible_contract else {}
        exact_sources={}
        if first_visible_contract:
            from parser_core.exact_forced_chronology import load_sources
            exact_sources=load_sources(output)
        facts=[reconcile_stack_facts(fact,events,forced_snapshots=snapshots,
            exact_binding_sources=exact_sources.get((fact['table'],fact['session'],fact['hand_id'])),
            **options) for fact in facts]
    selected = facts if eligible_contexts is None else [r for r in facts
        if (r['table'],r['session'],r['hand_id']) in eligible_contexts]
    excluded = [] if eligible_contexts is None else [r for r in facts
        if (r['table'],r['session'],r['hand_id']) not in eligible_contexts]
    options={'first_visible_contract':True} if first_visible_contract else {}
    results = [calculate_start_stack(r,**options) for r in selected]
    summary = dict(start_stack_pass=sum(r['status']=='PASS' for r in results),
                   start_stack_unknown=sum(r['status']!='PASS' for r in results),
                   arithmetic_candidates=sum('forced_bets' in r for r in selected), results=results,
                   snapshot_arithmetic_pass=sum(r.get('snapshot_arithmetic_status')=='PASS' for r in results),
                   input_fact_count=len(facts), eligible_fact_count=len(selected), excluded_partial_or_unproven_facts=len(excluded),
                   authority='MATH_FROM_VISION_FACTS; NO_HAND_HISTORY_PROMOTION')
    with (output/'start_stack_math.jsonl').open('w', encoding='utf-8') as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False)+'\n')
    if eligible_contexts is not None:
        with (output/'start_stack_math_excluded_diagnostics.jsonl').open('w',encoding='utf-8') as stream:
            for fact in excluded:
                stream.write(json.dumps(dict(status='NOT_EVALUATED_PARTIAL_OR_UNPROVEN',vision_fact=fact),ensure_ascii=False)+'\n')
    (output/'start_stack_math_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    return summary
