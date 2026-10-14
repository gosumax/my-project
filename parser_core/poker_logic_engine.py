"""Experimental fail-closed poker state validation and resolver entry point."""
from __future__ import annotations

import copy
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from time import perf_counter
from parser_core.poker_hand_assembly import BETTING, MONEY_ACTIONS
from parser_core.poker_math_engine import validate_math
from parser_core.poker_amount_semantics import resolve_call, call_contract, call_start_stack
from parser_core.poker_card_engine import best_rank, CLASSES

VERSION = 'experimental-logic-4-first-break-diagnostics'
STREETS = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']
INFORMATIONAL = {'DEAL', 'NO_SHOW', 'TABLE_INFO', 'BLIND_LEVEL'}
PAYOUT_ISSUES = {'OLD_WIN', 'INELIGIBLE_PAYOUT', 'MISSING_UNCALLED_RETURN',
                 'UNCALLED_RETURN_NOT_VERIFIABLE', 'SIDE_POT_ELIGIBILITY_CONFLICT',
                 'SIDE_POT_NOT_VERIFIABLE', 'ODD_CHIP_ALLOCATION_UNPROVEN'}


def issue_domain(code):
    if code.startswith('CARD_'):
        return 'CARDS'
    if code == 'ROW_IDENTITY_UNRESOLVED':
        return 'PROVENANCE'
    if code in {'INCOMPLETE_START', 'INCOMPLETE_END', 'HAND_BOUNDARY_CONFLICT'}:
        return 'COMPLETENESS'
    if code in PAYOUT_ISSUES:
        return 'PAYOUT'
    if code == 'MATH_DELTA':
        return 'MATH'
    return 'ACTION'


class _FirstLogicBreak(Exception):
    def __init__(self, issue):
        self.issue = issue


def validate_hand(hand, *, diagnostic_stop=False, hidden_card_source=None, no_sb_source=None):
    began = perf_counter()
    issues = []
    diagnostic_trace = []
    first_break = None
    amount_proofs = []
    call_ledger_known = True
    players = {}
    ledger = defaultdict(lambda: defaultdict(Decimal))
    payouts, returns = defaultdict(Decimal), defaultdict(Decimal)
    street, bet, board = 'PREFLOP', Decimal(0), []
    ended, betting_started = False, False
    seen_forced, seen_returns, headers = set(), set(), set()
    seen_start = False
    seen_deal = False
    hole = {}
    from .legitimate_no_sb import prove_no_sb
    no_sb_authority = prove_no_sb(hand,no_sb_source)
    excluded = set(hand.get('excluded_message_indices', []))
    # An informational observation becomes critical when another layer relies on
    # it to infer a money/action field. Its own action type is insufficient.
    inference_evidence = set(hand.get('blind_level', {}).get('source_row_ids', []))
    for inference in hand.get('inferences', []):
        inference_evidence.update(inference.get('affected_rows', []))
        inference_evidence.update(inference.get('decision', {}).get('evidence_rows', []))

    def error(code, index, detail='', *, domain=None, severity='BLOCKER', evidence_type='CONTRADICTION'):
        nonlocal call_ledger_known
        if severity == 'BLOCKER' and (domain or issue_domain(code)) != 'CARDS':
            call_ledger_known = False
        issues.append({'code': code, 'message_index': index, 'detail': detail,
                       'source_row_ids': hand['messages'][index]['source_row_ids'] if index >= 0 else [],
                       'domain': domain or issue_domain(code), 'severity': severity, 'evidence_type': evidence_type})
        issue = issues[-1]
        if diagnostic_stop and index >= 0 and severity == 'BLOCKER' and (
                issue['domain'] == 'ACTION' or
                issue['domain'] == 'COMPLETENESS' or
                code in {'OLD_WIN', 'INELIGIBLE_PAYOUT'}):
            raise _FirstLogicBreak(issue)

    def public_state(name=None):
        def serial(value):
            if isinstance(value, Decimal):
                return str(value)
            if isinstance(value, dict):
                return {k: serial(v) for k, v in value.items()}
            return value
        p = players.get(name)
        return {'street': street, 'current_bet': str(bet),
                'active_players': [n for n, p in players.items() if p['state'] == 'ACTIVE'],
                'folded_players': [n for n, p in players.items() if p['state'] == 'FOLDED'],
                'allin_players': [n for n, p in players.items() if p['state'] == 'ALL_IN'],
                'players': serial(copy.deepcopy(players)), 'player': name,
                'player_state': p['state'] if p else 'NOT_YET_OBSERVED',
                'player_street_contribution': str(p['street_contribution']) if p else '0',
                'amount_to_call': str(bet-(p['street_contribution'] if p else Decimal(0))) if name else None,
                'ended': ended, 'board': list(board), 'showdown': copy.deepcopy(hole),
                'prefix_math': {**{k: v for k, v in validate_math(ledger, payouts, returns).items()
                                   if k not in {'elapsed_ms', 'pass', 'errors'}},
                                'scope': 'PREFIX_SNAPSHOT', 'conservation_test_applicable': False},
                'total_contributions': str(sum((sum(v.values(), Decimal(0)) for v in ledger.values()), Decimal(0))),
                'total_returns': str(sum(returns.values(), Decimal(0))),
                'total_payouts': str(sum(payouts.values(), Decimal(0)))}

    def round_closed():
        active = [p for p in players.values() if p['state'] == 'ACTIVE']
        live = [p for p in players.values() if p['state'] != 'FOLDED']
        return len(live) == 1 or (len(active) <= 1 and all(p['street_contribution'] == bet for p in active)) or all(p['street_contribution'] == bet and p['acted'] for p in active)

    if not hand.get('start_observed'):
        error('INCOMPLETE_START', -1, evidence_type='MISSING_EVIDENCE')
    if not hand.get('end_observed'):
        error('INCOMPLETE_END', -1, evidence_type='MISSING_EVIDENCE')
    for conflict in hand.get('boundary_conflicts', []):
        error('HAND_BOUNDARY_CONFLICT', -1, conflict)
    for index, msg in enumerate(hand['messages']):
        before = public_state(msg.get('player')) if diagnostic_stop else None
        checkpoint = copy.deepcopy((players, ledger, payouts, returns, street, bet, board,
                                    ended, betting_started, seen_forced, seen_returns, headers, seen_start, hole, seen_deal)) if diagnostic_stop else None
        issue_start = len(issues)
        try:
            if index in excluded:
                continue
            action, name = msg['action'], msg.get('player')
            if any(r.get('ocr_status') == 'ERROR' or r.get('status') == 'ERROR'
                   for r in msg.get('source_rows', [])):
                error('OCR_EVIDENCE_ERROR', index, 'Failed OCR cannot support validated poker facts',
                      domain='PROVENANCE', evidence_type='MISSING_EVIDENCE')
            if any(r.get('origin') == 'UNRESOLVED_IDENTITY' for r in msg.get('source_rows', [])):
                informational = action in INFORMATIONAL and not (set(msg['source_row_ids']) & inference_evidence)
                error('ROW_IDENTITY_UNRESOLVED', index,
                      'IDENTITY_NONCRITICAL: informational event has no state/amount/card/boundary effect' if informational
                      else 'IDENTITY_CRITICAL: uncertain observation may affect order, action multiplicity, money, cards or boundary',
                      severity='WARNING' if informational else 'BLOCKER', evidence_type='MISSING_EVIDENCE')
            if msg.get('card_binding_conflict'):
                error('CARD_EVIDENCE_MISSING', index, 'Projected cards lack matching physical row evidence', evidence_type='MISSING_EVIDENCE')
            if msg.get('card_provenance', {}).get('hand_id') not in {None, hand.get('hand_id')}:
                error('CARD_LOGIC_CONFLICT', index, 'Projected cards bound to another hand')
            if action in {'HAND_HEADER', 'HAND_ID'}:
                number = msg.get('hand_number')
                if number in headers:
                    # Repeating the same explicit identifier does not constitute a poker
                    # action. Keep the row and report a redundant boundary observation.
                    error('CHAT_OVERLAP_DUPLICATE', index, 'Repeated same explicit hand marker',
                          domain='ASSEMBLY', severity='WARNING', evidence_type='REDUNDANT_METADATA')
                if headers and number not in headers:
                    error('HAND_BOUNDARY_CONFLICT', index)
                headers.add(number)
                continue
            if action == 'NEW_HAND':
                if seen_start or betting_started or seen_forced:
                    error('CHAT_OVERLAP_DUPLICATE', index, 'Repeated Starting hand: boundary meaning unproven',
                          domain='COMPLETENESS', evidence_type='AMBIGUOUS_EVIDENCE')
                seen_start = True
                continue
            if action in {'DEAL', 'NO_SHOW', 'TABLE_INFO', 'BLIND_LEVEL'}:
                if action == 'DEAL' and diagnostic_stop:
                    if seen_deal:
                        error('CHAT_OVERLAP_DUPLICATE', index, 'Repeated Dealing cards observation; boundary meaning unproven')
                    seen_deal = True
                continue
            if action == 'UNKNOWN_EVENT':
                call_ledger_known = False
                error('MISSING_ACTION', index, evidence_type='MISSING_EVIDENCE')
                continue
            if action in STREETS[1:]:
                if ended:
                    error('BROKEN_STREET_ORDER', index, 'Street after payout')
                expected = STREETS.index(street)+1
                if STREETS.index(action) != expected:
                    error('CHAT_OVERLAP_DUPLICATE' if action == street else 'BROKEN_STREET_ORDER', index)
                if not round_closed():
                    error('BETTING_ROUND_INCOMPLETE', index)
                cards = msg.get('cards', [])
                count = STREETS.index(action)+2
                if len(cards) < count:
                    error('CARD_EVIDENCE_MISSING', index, 'Incomplete board observation', evidence_type='MISSING_EVIDENCE')
                if len(cards) > count or len(set(cards)) != len(cards) or (cards and board and cards[:len(board)] != board) or any(set(cards) & set(v) for v in hole.values()):
                    error('CARD_LOGIC_CONFLICT', index, 'Duplicate or inconsistent observed board')
                board, street, bet = cards, action, Decimal(0)
                for p in players.values():
                    p['street_contribution'], p['acted'] = Decimal(0), False
                continue
            if action == 'SHOWDOWN':
                if not round_closed():
                    error('BETTING_ROUND_INCOMPLETE', index)
                continue
            if action == 'HAND_END':
                if not payouts or not round_closed():
                    error('INCOMPLETE_END', index, 'End marker lacks legal completion/payout')
                ended = True
                continue
            if not name or '?' in name or '\ufffd' in name:
                call_ledger_known = False
                error('PLAYER_IDENTITY_CONFLICT', index)
                continue
            if name not in players and any(old.rstrip('.') == name.rstrip('.') and old != name for old in players):
                error('PLAYER_IDENTITY_CONFLICT', index, 'Punctuation alias needs explicit identity evidence')
            # Punctuation may be part of a real nickname: never silently merge it.
            p = players.setdefault(name, {'state': 'ACTIVE', 'street_contribution': Decimal(0),
                                         'total_contribution': Decimal(0), 'last_action': None,
                                         'eligible_to_act': True, 'acted': False})
            if action == 'SHOW':
                if p['state'] == 'FOLDED':
                    error('CARD_LOGIC_CONFLICT', index, 'Folded player showdown')
                cards = msg.get('cards', [])
                if len(cards) < 2:
                    error('CARD_EVIDENCE_MISSING', index, 'Incomplete hole-card observation', evidence_type='MISSING_EVIDENCE')
                if len(cards) > 2 or len(set(cards)) != len(cards) or set(cards) & set(board):
                    error('CARD_LOGIC_CONFLICT', index, 'Invalid hole cards')
                if name in hole and hole[name] != cards:
                    error('CARD_LOGIC_CONFLICT', index, 'Inconsistent repeated showdown')
                if any(set(cards) & set(old) for other, old in hole.items() if other != name):
                    error('CARD_LOGIC_CONFLICT', index, 'Shared physical card')
                hole[name] = cards
                continue
            if action in BETTING:
                if diagnostic_stop and ('SMALL_BLIND' not in seen_forced and no_sb_authority is None or 'BIG_BLIND' not in seen_forced):
                    error('MISSING_BLINDS_BEFORE_ACTION', index, 'Required Small Blind and Big Blind before voluntary action', evidence_type='MISSING_EVIDENCE')
                betting_started = True
                if ended:
                    error('ACTION_AFTER_HAND_END', index)
                if p['state'] == 'FOLDED':
                    error('ACTION_AFTER_FOLD', index)
                if p['state'] == 'ALL_IN':
                    error('ACTION_AFTER_ALLIN', index)
            amount = None
            if action in {'ANTE', 'SMALL_BLIND', 'BIG_BLIND'}:
                key = (name, action) if action == 'ANTE' else action
                if key in seen_forced:
                    error('CHAT_OVERLAP_DUPLICATE', index, 'Repeated forced contribution')
                if betting_started or street != 'PREFLOP' or ended:
                    error('STALE_BLIND_OR_ANTE', index)
                seen_forced.add(key)
            if action in MONEY_ACTIONS:
                try:
                    amount = Decimal(str(msg.get('amount')))
                    if not amount.is_finite() or amount <= 0:
                        raise InvalidOperation
                except (InvalidOperation, ValueError):
                    call_ledger_known = False
                    error('MISSING_AMOUNT', index, evidence_type='MISSING_EVIDENCE')
                    continue
            delta = Decimal(0)
            if action in {'ANTE', 'SMALL_BLIND', 'BIG_BLIND'}:
                delta = amount
                ledger[name][action.lower()] += delta
                if action != 'ANTE':
                    p['street_contribution'] += delta
                    bet = max(bet, p['street_contribution'])
            elif action == 'FOLD':
                p['state'] = 'FOLDED'
            elif action == 'CHECK':
                if bet != p['street_contribution']:
                    error('CHECK_FACING_BET', index)
            elif action == 'CALL':
                required = bet-p['street_contribution']
                if required <= 0:
                    error('CALL_WITH_NOTHING_TO_CALL', index)
                semantics = call_contract(hand, msg)
                proof = resolve_call(amount, p['street_contribution'], bet,
                    all_in=msg.get('all_in', False), contract=semantics,
                    start_stack=call_start_stack(hand, name) if call_ledger_known else None,
                    total_contribution=p['total_contribution'], returned=returns[name])
                if proof['status'] != 'RESOLVED':
                    call_ledger_known = False
                amount_proofs.append(dict(proof, message_index=index,
                                         source_row_ids=list(msg['source_row_ids'])))
                if proof['status'] == 'AMBIGUOUS':
                    error('AMOUNT_SEMANTICS_UNPROVEN', index, 'Multiple legal CALL effects',
                          evidence_type='AMBIGUOUS_EVIDENCE')
                elif proof['status'] == 'CONTRADICTION':
                    error('INVALID_CALL', index, proof['reason'])
                delta = (Decimal(proof['delta']) if proof['status'] == 'RESOLVED' else
                         amount-p['street_contribution'] if semantics == 'TOTAL' else amount)
                if delta <= 0 or delta > required or (delta < required and not msg.get('all_in')):
                    error('INVALID_CALL', index)
                p['street_contribution'] += delta
            elif action == 'BET':
                if bet != 0:
                    error('BET_INTO_EXISTING_BET', index)
                # A literal BET into an unopened round has a single paid effect.
                # Existing-bet contradictions remain an independent blocker.
                delta = amount
                p['street_contribution'] += delta
                bet = p['street_contribution']
            elif action == 'RAISE':
                if bet <= 0 or amount <= bet:
                    error('INVALID_RAISE', index)
                delta = amount-p['street_contribution']
                if delta <= 0:
                    error('INVALID_RAISE', index, 'Non-positive delta')
                p['street_contribution'], bet = amount, amount
            elif action == 'UNCALLED_RETURN':
                key = (name, street, str(amount))
                if key in seen_returns:
                    error('CHAT_OVERLAP_DUPLICATE', index, 'Repeated return')
                seen_returns.add(key)
                if amount > p['street_contribution'] or amount > p['total_contribution']-returns[name]:
                    error('RETURN_EXCEEDS_CONTRIBUTION', index)
                other = max((v['street_contribution'] for n, v in players.items() if n != name), default=Decimal(0))
                if amount != p['street_contribution']-other:
                    error('INVALID_UNCALLED_RETURN', index)
                p['street_contribution'] -= amount
                bet = max((v['street_contribution'] for v in players.values()), default=Decimal(0))
                returns[name] += amount
            elif action == 'WIN':
                if p['state'] == 'FOLDED':
                    error('INELIGIBLE_PAYOUT', index)
                live = [v for v in players.values() if v['state'] != 'FOLDED']
                if len(live) > 1 and (not round_closed() or (street != 'RIVER' and not all(v['state'] == 'ALL_IN' for v in live))):
                    error('OLD_WIN', index, 'Payout before legal completion')
                if len(live) > 1 and len(board) != 5:
                    error('CARD_EVIDENCE_MISSING', index, 'Contested payout without full board', evidence_type='MISSING_EVIDENCE')
                payouts[name] += amount
                ended = True
            else:
                error('MISSING_ACTION', index, action)
            if action in {'CALL', 'BET', 'RAISE'}:
                ledger[name][street.lower()+'_contribution'] += delta
                if action in {'BET', 'RAISE'}:
                    for other in players.values():
                        other['acted'] = False
            p['total_contribution'] += delta
            p['last_action'] = action
            if action in BETTING:
                p['acted'] = True
            if msg.get('all_in') and action in MONEY_ACTIONS:
                p['state'] = 'ALL_IN'
            p['eligible_to_act'] = p['state'] == 'ACTIVE' and not ended
        except _FirstLogicBreak as stop:
            first_break = stop.issue
            (players, ledger, payouts, returns, street, bet, board, ended, betting_started,
             seen_forced, seen_returns, headers, seen_start, hole, seen_deal) = checkpoint
            break
        finally:
            if diagnostic_stop:
                diagnostic_trace.append({'message_index': index,
                    'status': 'FIRST_BREAK' if first_break else 'IGNORED_DUPLICATE' if index in excluded else 'OK',
                    'state_before': before, 'state_after': public_state(msg.get('player')),
                    'issues': copy.deepcopy(issues[issue_start:]), 'is_first_break': bool(first_break)})
    if diagnostic_stop:
        for index in range(len(diagnostic_trace), len(hand['messages'])):
            diagnostic_trace.append({'message_index': index, 'status': 'AFTER_BREAK',
                'rule_result': 'NOT_EVALUATED_AFTER_FIRST_BREAK', 'state_before': None,
                'state_after': None, 'issues': [], 'is_first_break': False})
        # Aggregate completeness is a virtual end-of-input check, never an
        # accusation against an earlier observed legal message.
        if first_break is None:
            eof_code = ('INCOMPLETE_START' if not hand.get('start_observed') or
                        'SMALL_BLIND' not in seen_forced and no_sb_authority is None or 'BIG_BLIND' not in seen_forced else
                        'HAND_BOUNDARY_CONFLICT' if hand.get('boundary_conflicts') else
                        'INCOMPLETE_END' if not hand.get('end_observed') or not payouts else
                        'BETTING_ROUND_INCOMPLETE' if not round_closed() else None)
            if eof_code:
                first_break = {'code': eof_code, 'message_index': -1, 'source_row_ids': [],
                               'domain': issue_domain(eof_code), 'severity': 'DIAGNOSTIC',
                               'detail': 'Virtual end-of-segment: completion not established',
                               'evidence_type': 'MISSING_EVIDENCE'}
        return {'first_issue': first_break, 'trace': diagnostic_trace,
                'prefix_state': public_state(), 'issues_evaluated': issues,
                'logic_ms': (perf_counter()-began)*1000}
    if 'SMALL_BLIND' not in seen_forced and no_sb_authority is None or 'BIG_BLIND' not in seen_forced:
        error('INCOMPLETE_START', -1, 'Missing forced blind sequence', evidence_type='MISSING_EVIDENCE')
    if not payouts:
        error('INCOMPLETE_END', -1, 'No observed payout', evidence_type='MISSING_EVIDENCE')
    if not round_closed():
        error('BETTING_ROUND_INCOMPLETE', -1)
    # A player cannot win more than the pots for which their net contribution qualifies.
    nets = {n: p['total_contribution']-returns[n] for n, p in players.items()}
    ledger_known = not any(i['severity'] == 'BLOCKER' and
                           (i['domain'] in {'ACTION', 'PROVENANCE', 'COMPLETENESS'} or i['code'] in {'INVALID_UNCALLED_RETURN', 'RETURN_EXCEEDS_CONTRIBUTION'})
                           for i in issues)
    positive_nets = sorted((v for v in nets.values() if v > 0), reverse=True)
    if positive_nets and (len(positive_nets) == 1 or positive_nets[0] > positive_nets[1]):
        error('MISSING_UNCALLED_RETURN' if ledger_known else 'UNCALLED_RETURN_NOT_VERIFIABLE', -1,
              'Unmatched contribution layer cannot be a pot payout' if ledger_known else 'Ledger completeness unproven; unmatched layer is diagnostic only',
              evidence_type='CONTRADICTION' if ledger_known else 'MISSING_EVIDENCE')
    for n, paid in payouts.items():
        ceiling = sum((min(max(Decimal(0), value), max(Decimal(0), nets[n])) for value in nets.values()), Decimal(0))
        if paid > ceiling:
            error('SIDE_POT_ELIGIBILITY_CONFLICT' if ledger_known else 'SIDE_POT_NOT_VERIFIABLE', -1,
                  n if ledger_known else 'Incomplete ledger cannot prove payout eligibility: '+n,
                  evidence_type='CONTRADICTION' if ledger_known else 'MISSING_EVIDENCE')
    for n, cards in hole.items():
        if set(cards) & set(board):
            error('CARD_LOGIC_CONFLICT', -1, 'Hole card equals final board')
    live = [n for n, p in players.items() if p['state'] != 'FOLDED']
    ranks = {}
    hidden_authority = None
    if len(live) > 1 and payouts:
        missing = {n for n in live if len(hole.get(n, [])) != 2}
        if missing and hidden_card_source is not None:
            from parser_core.hidden_loser_evidence import HiddenLoserSource
            if isinstance(hidden_card_source, HiddenLoserSource):
                hidden_authority = hidden_card_source.authorize(hand, players=players,
                    nets=nets,payouts=payouts,board=board,hole=hole,
                    ledger_known=ledger_known,math=validate_math(ledger,payouts,returns))
        exempt = set(hidden_authority['hidden_losers']) if hidden_authority else set()
        if len(board) != 5 or missing-exempt:
            error('CARD_EVIDENCE_MISSING', -1, 'Contested payout rank comparison lacks full board/live hole cards', evidence_type='MISSING_EVIDENCE')
            error('SIDE_POT_NOT_VERIFIABLE', -1, 'Winner ranks cannot be independently verified', evidence_type='MISSING_EVIDENCE')
        else:
            try:
                # Explicit native NO_SHOW exempts only independent comparison.
                # Unknown cards/ranks are never inserted into either mapping.
                ranks = {n: best_rank(board+hole[n]) for n in live if n not in exempt}
            except ValueError:
                error('CARD_LOGIC_CONFLICT', -1, 'Invalid physical cards')
    description_board = []
    for index, msg in enumerate(hand['messages']):
        if index in excluded:
            continue
        if msg['action'] in {'FLOP', 'TURN', 'RIVER'}:
            description_board = list(msg.get('cards') or [])
        expected_class = msg.get('showdown_class')
        n = msg.get('player')
        if expected_class and msg['action'] in {'SHOW', 'WIN'}:
            # A player may reveal while all-in before the runout completes.
            # Their literal SHOW describes the board at that source event;
            # final payout descriptions still require the complete final board.
            class_board = description_board if msg['action'] == 'SHOW' else board
            enough_board = (3 <= len(class_board) <= 5 if msg['action'] == 'SHOW'
                            else len(class_board) == 5)
            if enough_board and len(hole.get(n, [])) == 2:
                try:
                    actual_class = CLASSES[best_rank(class_board+hole[n])[0]]
                    if expected_class != actual_class:
                        error('CARD_LOGIC_CONFLICT', index, 'Showdown description contradicts physical cards')
                except ValueError:
                    error('CARD_LOGIC_CONFLICT', index, 'Invalid class evidence')
            else:
                error('CARD_EVIDENCE_MISSING', index, 'Missing cards to verify described class', evidence_type='MISSING_EVIDENCE')
    if ranks and not ledger_known:
        error('SIDE_POT_NOT_VERIFIABLE', -1, 'Ranks are available but contribution ledger is incomplete', evidence_type='MISSING_EVIDENCE')
    if ranks and ledger_known:
        expected_payouts = defaultdict(Decimal)
        levels = sorted({v for v in nets.values() if v > 0})
        previous = Decimal(0)
        for level in levels:
            contributors = [n for n, value in nets.items() if value >= level]
            eligible = [n for n in contributors if n in ranks]
            pot = (level-previous)*len(contributors)
            previous = level
            if not eligible:
                error('SIDE_POT_ELIGIBILITY_CONFLICT', -1, 'Pot with no eligible player')
                continue
            best = max(ranks[n] for n in eligible)
            winners = [n for n in eligible if ranks[n] == best]
            share = pot/len(winners)
            if share != share.to_integral_value():
                error('ODD_CHIP_ALLOCATION_UNPROVEN', -1)
            for n in winners:
                expected_payouts[n] += share
        if dict(expected_payouts) != dict(payouts):
            error('CARD_LOGIC_CONFLICT', -1, 'Payouts contradict deterministic side-pot winners')
    logic_ms = (perf_counter()-began)*1000
    math = validate_math(ledger, payouts, returns)
    card_issues = [i for i in issues if i['domain'] == 'CARDS' and i['severity'] == 'BLOCKER']
    card_status = ('CONFLICT' if any(i['evidence_type'] == 'CONTRADICTION' for i in card_issues)
                   else 'NOT_VERIFIABLE' if card_issues else 'PASS')
    cards_pass = card_status == 'PASS'
    for code in math['errors']:
        error(code, -1)
    complete = hand.get('start_observed') and hand.get('end_observed') and not any(i['code'] in {'INCOMPLETE_START', 'INCOMPLETE_END', 'HAND_BOUNDARY_CONFLICT'} for i in issues)
    blockers = [i for i in issues if i['severity'] == 'BLOCKER']
    action_issues = [i for i in blockers if i['domain'] == 'ACTION' and i['code'] not in math['errors']]
    context_issues = [i for i in blockers if i['domain'] == 'COMPLETENESS']
    action_status = ('FAIL' if action_issues else 'NOT_VERIFIABLE' if context_issues else 'PASS')
    logic_pass = action_status == 'PASS'
    payout_issues = [i for i in blockers if i['domain'] == 'PAYOUT']
    side_pot_status = ('CONFLICT' if any(i['evidence_type'] == 'CONTRADICTION' for i in payout_issues)
                       else 'NOT_VERIFIABLE' if payout_issues else 'PASS')
    identity_critical = any(i['domain'] == 'PROVENANCE' and i['severity'] == 'BLOCKER' for i in issues)
    status = ('FULLY_VALID_HAND' if not blockers else 'INCOMPLETE_START' if any(i['code'] == 'INCOMPLETE_START' for i in issues)
              else 'INCOMPLETE_END' if any(i['code'] == 'INCOMPLETE_END' for i in issues)
              else 'CARD_CONFLICT' if card_status == 'CONFLICT' else 'LOGIC_FAIL' if action_status == 'FAIL'
              else 'MATH_FAIL' if not math['pass'] else 'UNRESOLVED_HAND')
    serial_players = {n: {**{k: str(v) if isinstance(v, Decimal) else v for k, v in p.items()},
                           'folded': p['state'] == 'FOLDED', 'all_in': p['state'] == 'ALL_IN',
                           'net_contribution': str(nets[n])} for n, p in players.items()}
    return {'status': status, 'complete_candidate': bool(complete), 'logic_pass': logic_pass,
            'amount_resolution_proofs': amount_proofs,
            'action_logic_pass': logic_pass, 'action_logic_status': action_status,
            'card_status': card_status, 'side_pot_status': side_pot_status,
            'identity_status': 'IDENTITY_CRITICAL' if identity_critical else 'IDENTITY_NONCRITICAL' if any(i['domain'] == 'PROVENANCE' for i in issues) else 'PASS',
            'certification_blockers': blockers,
            'cards_pass': cards_pass, 'math': math, 'issues': issues, 'players': serial_players,
            'street': street, 'board': board, 'showdown': hole, 'logic_ms': logic_ms,
            'state': 'HAND_END' if ended else 'SHOWDOWN' if hole and street == 'RIVER' else street,
            'accepted_inference': False,
            **({'hidden_loser_authority':hidden_authority} if hidden_authority else {}),
            **({'no_sb_authority':no_sb_authority} if no_sb_authority else {})}


def resolve_hand(hand, resolver):
    """Jev only sees failed complete hands; all candidates undergo same validator."""
    initial = validate_hand(hand)
    if initial['status'] == 'FULLY_VALID_HAND' or not initial['complete_candidate']:
        return {'initial': initial, 'final': initial, 'resolution': None}
    resolution = resolver.resolve(hand, initial)
    final = initial
    if resolution.get('candidate_hand') is not None:
        candidate = resolution.pop('candidate_hand')
        resolution['candidate_view'] = candidate
        checked = validate_hand(candidate)
        resolution['revalidation'] = checked
        if checked['status'] == 'FULLY_VALID_HAND':
            checked['status'] = 'JEV_ASSISTED_CERTIFIED'
            checked['accepted_inference'] = True
            final = checked
            resolution['accepted'] = True
    if final is initial:
        final = dict(initial, status='UNRESOLVED_HAND')
        resolution['accepted'] = False
    return {'initial': initial, 'final': final, 'resolution': resolution}
