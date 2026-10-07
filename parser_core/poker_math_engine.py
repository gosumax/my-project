"""Experimental Decimal contribution ledger; no visual pot authority."""
from decimal import Decimal
from time import perf_counter

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
            else:
                delta = Decimal(0)
                if action in {'ANTE', 'SMALL_BLIND', 'BIG_BLIND'}:
                    delta = amount
                    ledger[player][action.lower()] += delta
                    if action != 'ANTE':
                        street_contributions[player] += delta
                        bet = max(bet, street_contributions[player])
                elif action == 'CALL':
                    semantics = hand.get('semantics', {}).get('call')
                    delta = amount-own if semantics == 'TOTAL' else amount
                    if semantics not in {'ADDED', 'TOTAL'}:
                        suspicious.append({'message_index': index, 'RULE_TRIGGER': 'AMOUNT_SEMANTICS_UNPROVEN', 'PROVEN_CAUSE': 'UNKNOWN'})
                    street_contributions[player] += delta
                elif action == 'BET':
                    delta = amount
                    street_contributions[player] += delta
                    bet = street_contributions[player]
                elif action == 'RAISE':
                    delta = amount-own
                    street_contributions[player], bet = amount, amount
                    if delta <= 0:
                        suspicious.append({'message_index': index, 'RULE_TRIGGER': 'NON_POSITIVE_RAISE_DELTA', 'PROVEN_CAUSE': 'UNKNOWN'})
                elif action == 'UNCALLED_RETURN':
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
        steps.append({'message_index': index, 'before': before, 'after': snapshot(), 'math_effect': effect,
                      'amount_to_call_before': str(required) if required is not None else None})
    result = validate_math(ledger, payouts, returns)
    result.update(total_contributions=snapshot()['total_contributions'], total_returns=snapshot()['total_returns'],
                  contributions_by_street=snapshot()['contributions_by_street'],
                  scope='FULL_OBSERVED_LEDGER_NOT_PROOF_OF_LEGAL_ACTIONS', missing_fields=missing,
                  suspicious_rows=suspicious, steps=steps,
                  conservation_test_applicable=bool(hand.get('end_observed')))
    return result


def calculate_start_stack(facts):
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
    if not facts.get('occupancy_epoch') or not facts.get('participant_proven'):
        result['blocker'] = 'OCCUPANCY_MEMBERSHIP_UNKNOWN'
        return result
    frames = facts.get('frames', [])
    if (len(frames) != 3 or frames != list(range(frames[0], frames[0]+3))
            or facts.get('clean_window_rule') != 'FIRST_VISIBLE_STABLE_STACK_TRIPLE_AFTER_DEAL_COMPLETE'
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
    return result


def write_start_stack_math(output, *, eligible_contexts=None):
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
        facts=[reconcile_stack_facts(fact,events,forced_snapshots=snapshots) for fact in facts]
    selected = facts if eligible_contexts is None else [r for r in facts
        if (r['table'],r['session'],r['hand_id']) in eligible_contexts]
    excluded = [] if eligible_contexts is None else [r for r in facts
        if (r['table'],r['session'],r['hand_id']) not in eligible_contexts]
    results = [calculate_start_stack(r) for r in selected]
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
