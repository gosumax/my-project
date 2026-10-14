"""Resolve only a unique economic effect of a literal observed amount.

The observed amount and source rows are never changed. Unknown stacks do not
prevent checking an ordinary call; an ambiguous short all-in stays unknown.
"""
import copy
from decimal import Decimal, InvalidOperation
from parser_core.poker_reconstruction.models import Player


def with_call_context(hand, receipts):
    """Carry already verified provider receipts; never acquire or alter facts."""
    result = copy.deepcopy(hand)
    result['call_start_stack_receipts'] = copy.deepcopy(list(receipts))
    return result


def call_start_stack(hand, actor):
    from parser_core.hand_lifecycle import table_key
    values = set()
    seats = set()
    for receipt in hand.get('call_start_stack_receipts', []):
        if (receipt.get('status') != 'PROVEN' or receipt.get('actor') != actor or
                receipt.get('seat') is None or hand.get('table_id') is None or
                table_key(receipt.get('table_id')) != table_key(hand['table_id']) or
                any(hand.get(k) is None or receipt.get(k) != hand[k]
                    for k in ('session_id', 'hand_id'))):
            continue
        seats.add(receipt['seat'])
        identity = receipt.get('identity_proof') or {}
        if identity and (identity.get('seat') != receipt['seat'] or
                identity.get('canonical_actor', identity.get('actor')) != actor):
            return None
        value = receipt.get('calculated_start_stack')
        if isinstance(value, bool):
            return None
        try:
            value = Decimal(str(value))
            if not value.is_finite() or value < 0:
                return None
        except (InvalidOperation, ValueError):
            return None
        values.add(value)
    return next(iter(values)) if len(values) == 1 and len(seats) == 1 else None


def call_contract(hand, message):
    """A trace contract describes the amount representation actually passed."""
    proof = message.get('amount_proof') or {}
    if 'call_contract' in proof:
        if (proof.get('rule') == 'EXHAUSTIVE_UNIQUE_ORDERED_ECONOMIC_TRACE' and
                proof.get('kind') == 'CALL' and proof.get('actor') == message.get('player') and
                proof.get('source_row_ids') == message.get('source_row_ids') and
                str(proof.get('derived_amount')) == str(message.get('amount')) and
                proof['call_contract'] in {'ADDED', 'TOTAL'}):
            return proof['call_contract']
        return 'INVALID_CONTRACT'
    return hand.get('semantics', {}).get('call', 'UNPROVEN')


def resolve_call(amount, own, current_bet, *, all_in=False, contract=None,
                 start_stack=None, total_contribution=None, returned=None):
    owed = current_bet - own
    options = {'ADDED': amount, 'TOTAL': amount - own}
    if contract in options:
        options = {contract: options[contract]}
    elif contract == 'INVALID_CONTRACT':
        options = {}
    legal = {meaning: delta for meaning, delta in options.items()
             if delta > 0 and owed > 0 and
             (delta == owed or all_in and delta < owed)}
    # Reuse State's exact net-stack accounting, with the already observed ledger.
    # A proven contract has one option only; an invalid contract is not re-guessed.
    remaining = None
    if all(v is not None for v in (start_stack, total_contribution, returned)):
        try:
            context = [Decimal(str(v)) for v in (start_stack, total_contribution, returned)]
            if (not any(isinstance(v, bool) for v in (start_stack, total_contribution, returned)) and
                    all(v.is_finite() and v >= 0 for v in context) and context[2] <= context[1]):
                remaining = Player(start_stack=context[0], total_commit=context[1], returned=context[2]).remaining_stack
        except (InvalidOperation, ValueError):
            pass
    if remaining is not None:
        legal = {meaning: delta for meaning, delta in legal.items()
                 if delta <= remaining and (not all_in or delta == remaining)}
    effects = set(legal.values())
    status = 'RESOLVED' if len(effects) == 1 else 'AMBIGUOUS' if effects else 'CONTRADICTION'
    delta = next(iter(effects)) if len(effects) == 1 else None
    result = dict(status=status, raw_amount=str(amount),
                resolved_semantics=next(iter(legal)) if len(legal) == 1 else
                    'EQUIVALENT_ADDED_TOTAL' if len(effects) == 1 else None,
                delta=str(delta) if delta is not None else None,
                street_contribution_before=str(own), current_highest_bet=str(current_bet),
                amount_to_call=str(owed),
                alternatives={k: str(v) for k, v in legal.items()},
                reason='UNIQUE_LEGAL_CALL_EFFECT' if len(effects) == 1 else
                    'MULTIPLE_LEGAL_CALL_EFFECTS' if effects else 'NO_LEGAL_CALL_EFFECT',
                source='OBSERVED_CONTRACT' if contract in {'ADDED', 'TOTAL'} else 'POKER_MATH_INFERENCE')
    if remaining is not None:
        result['stack_context'] = dict(start_stack=str(context[0]), total_contribution=str(context[1]),
            returned=str(context[2]), remaining_before=str(remaining), rule='EXACT_NET_STACK_EXHAUSTION')
    return result
