from copy import deepcopy
from dataclasses import replace
from .accounting import audit
from .models import Player, RuleError
from .state_machine import State
from .stale_filter import drop_candidate


def clone_state(state):
    # Trace snapshots are append-only plain values. Sharing their prior entries
    # avoids recursively copying the whole diagnostic history at each branch.
    candidate = deepcopy(state, {id(state.trace): state.trace})
    candidate.trace = list(state.trace)
    return candidate


def amount_variants(event, state, future):
    if event.amount is not None or event.kind not in {"BB", "SB", "CALL", "RETURN"}:
        return [(event, None)]
    if event.player not in state.players:
        return [(event, None)]
    p = state.players[event.player]
    if event.kind == 'CALL' and event.semantics == 'UNKNOWN' and not event.all_in:
        # Keep the intended interpretation paired with its amount. Crossing
        # these amounts with both meanings is not exhaustive valid inference.
        return [(variant, dict(inference, interpretation=semantics))
                for semantics in ('DELTA','TO')
                for variant,inference in amount_variants(replace(event,semantics=semantics),state,future)
                if inference is not None]
    amounts = set()
    reason = None
    if event.kind == "CALL" and not event.all_in:
        amounts.add(state.current_bet if event.semantics == "TO" else state.current_bet - p.street_commit)
        reason = "OBSERVED_STANDARD_CALL_REQUIRES_EXACT_TO_CALL"
    elif event.kind == "RETURN" and state.closed():
        amounts.add(p.street_commit - max((q.street_commit for n, q in state.players.items() if n != event.player), default=0))
        reason = "OBSERVED_RETURN_REQUIRES_EXACT_UNMATCHED_EXCESS"
    elif event.kind in {"SB", "BB"}:
        # An observed later call delta and observed raise-to constrain the
        # missing earlier blind. Enumerate candidates, never use a blind ratio.
        targets = []
        for later in future:
            if later.kind == "STREET":
                break
            if later.kind == "RAISE" and later.semantics == "TO" and later.amount is not None:
                targets.append(later.amount)
            if later.player == event.player and later.kind == "CALL" and later.amount is not None:
                if later.semantics == "DELTA":
                    amounts.update(t - later.amount for t in targets if t > later.amount)
                break
            if later.player == event.player and later.kind in {"BET", "RAISE", "FOLD"}:
                break
        reason = "OBSERVED_RAISE_TO_MINUS_OBSERVED_FIRST_CALL_DELTA_CONSTRAINS_BLIND"
    if not amounts:
        return [(event, None)]
    return [(replace(event, amount=amount), {"source": event.source, "amount": amount, "reason": reason})
            for amount in sorted(amounts) if amount > 0]


def solve(events, start_stacks=None, previous_hand=(), max_nodes=20000, *, strict_source_uniqueness=False):
    names = {e.player for e in events if e.player and e.kind not in {"WIN", "SHOW", "RETURN", "INFO", "UNKNOWN"}}
    initial = State(players={name: Player(start_stack=(start_stacks or {}).get(name)) for name in sorted(names)})
    branches = [(initial, [], [])]
    failures, visited, exhausted = [], 0, False
    first_blocker = None
    for index, event in enumerate(events):
        next_branches = []
        reason = None if strict_source_uniqueness else drop_candidate(event, events[:index], previous_hand, events[index + 1:])
        for state, kept, dropped in branches:
            variants = amount_variants(event, state, events[index + 1:])
            choices = [(variant, inference, interpretation) for variant, inference in variants
                       for interpretation in (("DELTA", "TO") if variant.semantics == "UNKNOWN" and variant.amount is not None else (None,))]
            for variant, inference, choice in choices:
                visited += 1
                if visited > max_nodes:
                    exhausted = True
                    break
                candidate = clone_state(state)
                try:
                    candidate.apply(variant, choice)
                    if inference:
                        candidate.inferences.append(inference)
                    candidate.trace.append({"source_indices": [r.get("source_index") for r in variant.source.get("rows", [])],
                        "candidate_event": {"kind": variant.kind, "player": variant.player, "amount": variant.amount, "semantics": choice or variant.semantics},
                        "decision": "KEEP", "street": candidate.street,
                        "street_contributions": {n: p.street_commit for n, p in candidate.players.items()},
                        **audit(candidate)})
                    next_branches.append((candidate, kept + [variant], dropped))
                except RuleError as error:
                    if len(failures) < 50:
                        failures.append({"source": event.source, "event": event.evidence(), "rule": str(error)})
            if reason:
                drop_state = clone_state(state)
                drop_state.trace.append({"source_indices": [r.get("source_index") for r in event.source.get("rows", [])],
                                         "candidate_event": {"kind": event.kind, "player": event.player, "amount": event.amount, "semantics": event.semantics},
                                         "decision": "DROP_CANDIDATE", "reason": reason, "street": state.street,
                                         "street_contributions": {n: p.street_commit for n, p in state.players.items()}, **audit(state)})
                next_branches.append((drop_state, kept, dropped + [{"event": event.evidence(), "reason": reason}]))
            if exhausted:
                break
        if exhausted:
            break
        if not next_branches:
            # Keep last viable prefix accounting as diagnostic only.
            first_blocker = {"stage": "POKER_SOLVER", "event": event.evidence(),
                             "source_indices": [r.get("source_index") for r in event.source.get("rows", [])],
                             "reason": "No viable continuation from any retained prefix under supported interpretations"}
            break
        branches = next_branches
    else:
        index = len(events)
    valid = []
    diagnostics = []
    for state, kept, dropped in branches:
        ledger = audit(state)
        diagnostics.append(ledger)
        if index == len(events) and not exhausted and state.blind_kinds == {"SB", "BB"} and state.seen_payout and state.closed() and ledger["calculated_pot"] > 0 and ledger["pot_delta"] == 0 and ledger["payout_eligible"]:
            # Unreturned unmatched excess is never a pot layer that can be won.
            if any(len(layer["contributors"]) == 1 for layer in ledger["pot_layers"]):
                continue
            valid.append((state, kept, dropped, ledger))
    if valid and not strict_source_uniqueness:
        best = min((len(drop), len(state.inferences)) for state, _, drop, _ in valid)
        valid = [v for v in valid if (len(v[2]), len(v[0].inferences)) == best]
        # Source-row alternatives remain distinct: same text is not identity.
    if first_blocker is None and not valid:
        first_blocker = {"stage": "CHIP_ACCOUNTING_OR_COMPLETION", "reason": "No exact eligible payout with complete supported start and terminal state"}
    return {"solutions": valid, "failures": failures, "nodes": visited,
            "exhausted": exhausted, "processed_event_count": index,
            "candidate_ledgers": diagnostics[:5], "candidate_traces": [state.trace for state, _, _ in branches[:5]],
            "first_engine_blocker": first_blocker}
