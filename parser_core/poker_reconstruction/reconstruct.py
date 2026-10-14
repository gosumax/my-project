from collections import defaultdict
from dataclasses import replace
from difflib import SequenceMatcher
from itertools import combinations
import json
from .identity import identity_hypotheses
from .solver import solve


def economic_source_signature(solution):
    """Equal HH effects require every ordered source group and ledger step.

    DELTA/TO can name the identical payment when the prior contribution is zero.
    Final pot equality alone never collapses two possible action histories.
    """
    state,kept,dropped,ledger,hypothesis,assembly=solution
    if dropped or hypothesis['aliases']:
        raise ValueError('Strict uniqueness cannot collapse dropped sources or aliases')
    steps=[];prior={};prior_street=None
    for event,step in zip(kept,state.trace):
        if step['decision']!='KEEP':raise ValueError('Strict source trace contains a drop')
        before=prior.get(event.player,0) if prior_street==step['street'] else 0
        after=step['street_contributions'].get(event.player,0)
        amount=(after-before if event.kind in {'CALL','BET'} else after if event.kind=='RAISE' else event.amount)
        core={k:v for k,v in step.items() if k not in {'candidate_event','source_indices'}}
        core.update(source_group=[(r.get('stable_row_id',r.get('row_id')),r.get('row_pixel_sha256'))
                                  for r in event.source.get('rows',[])],
                    event=dict(kind=event.kind,player=event.player,amount=amount,
                               all_in=event.all_in,street=event.street))
        steps.append(core)
        prior=step['street_contributions'];prior_street=step['street']
    if len(kept)!=len(state.trace):raise ValueError('Strict source trace length mismatch')
    return json.dumps(steps,sort_keys=True,ensure_ascii=False)


def reconstruct_hand(events, *, table="UNKNOWN", hand_id=None, at_eof=False,
                     start_stacks=None, previous_hand=(), max_nodes=20000,
                     assemblies=None, assembly_exhausted=False, alias_candidates=None,
                     strict_source_uniqueness=False, require_small_blind=True):
    hypotheses, identity_exhausted, rejected = identity_hypotheses(alias_candidates)
    solutions, outcome, spent = [], None, 0
    for assembly in assemblies or [events]:
        for hypothesis in hypotheses:
            def canonical(name):
                mapping = hypothesis["aliases"]
                while name in mapping and mapping[name] != name:
                    name = mapping[name]
                return name
            interpreted = [replace(e, player=canonical(e.player)) for e in assembly]
            mapped_stacks = {}
            stack_conflict = False
            for name, value in (start_stacks or {}).items():
                key = canonical(name)
                if key in mapped_stacks and mapped_stacks[key] != value:
                    stack_conflict = True
                mapped_stacks[key] = value
            if stack_conflict:
                continue
            current = solve(interpreted, mapped_stacks, previous_hand, max(1, max_nodes - spent),
                            strict_source_uniqueness=strict_source_uniqueness,
                            require_small_blind=require_small_blind)
            spent += current["nodes"]
            for state, kept, dropped, ledger in current.pop("solutions"):
                if hypothesis["aliases"]:
                    state.inferences.append({"reason": "IDENTITY_EVIDENCE_ALIAS_HYPOTHESIS", "hypothesis": hypothesis})
                solutions.append((state, kept, dropped, ledger, hypothesis, assembly))
            if outcome is None or current["processed_event_count"] > outcome["processed_event_count"]:
                outcome = current
            if current["exhausted"] or spent >= max_nodes:
                assembly_exhausted = True
                break
        if assembly_exhausted:
            break
    if outcome is None:
        outcome = {"exhausted": False, "nodes": 0, "processed_event_count": 0,
                   "failures": [{"rule": "CONFLICTING_EXACT_STACK_IDENTITY_EVIDENCE"}],
                   "candidate_ledgers": [], "candidate_traces": []}
    outcome["exhausted"] = assembly_exhausted or identity_exhausted
    outcome["nodes"] = spent
    raw_solution_count=len(solutions)
    equivalent_interpretations=[]
    if solutions and strict_source_uniqueness:
        groups={}
        for solution in solutions:
            if solution[2] or solution[4]['aliases']:
                continue
            signature=economic_source_signature(solution)
            groups.setdefault(signature,[]).append(solution)
        solutions=[group[0] for group in groups.values()]
        equivalent_interpretations=[dict(count=len(group),
            interpretations=[[s['candidate_event'] for s in solution[0].trace] for solution in group])
            for group in groups.values() if len(group)>1]
    elif solutions:
        def rank(solution):
            # Default distinct spelling is not proof of A!=B. Adding an alias
            # hypothesis must not lose a tie merely for having that hypothesis.
            substantive = sum(i["reason"] != "IDENTITY_EVIDENCE_ALIAS_HYPOTHESIS" for i in solution[0].inferences)
            return len(solution[2]), substantive
        best = min(rank(s) for s in solutions)
        solutions = [s for s in solutions if rank(s) == best]
    if hand_id is None:
        status, reason = "TRUNCATED_START", "No confirmed numbered start boundary in source"
    elif outcome["exhausted"]:
        status, reason = "AMBIGUOUS", "Solver budget exhausted; uniqueness was not proven"
    elif len(solutions) > 1:
        status, reason = "AMBIGUOUS", "Several equally ranked legal exact-accounting source sequences"
    elif len(solutions) == 1:
        status = "RECOVERED_BY_LOGIC_MATH" if solutions[0][0].inferences else "AUTO_RECOVERED"
        reason = "Unique maximal retained source sequence with legal state and exact eligible payouts"
    elif at_eof and (not any(e.kind == "WIN" for e in events) or
                    max((i for i, e in enumerate(events) if e.kind == "STREET"), default=-1) >
                    max((i for i, e in enumerate(events) if e.kind == "WIN"), default=-1)):
        status, reason = "TRUNCATED_END", "Final source segment lacks payout/result; future actions unknown"
    elif any(e.kind == "UNKNOWN" or (e.amount is None and e.kind in {"ANTE", "SB", "BB", "CALL", "BET", "RAISE", "WIN", "RETURN"}) for e in events):
        status, reason = "AMBIGUOUS", "Unparsed source or missing observed amount prevents exhaustive legal interpretation"
    else:
        status, reason = "INVALID", "No legal exact-accounting sequence under supported evidence rules"
    result = {"table": table, "hand_id": hand_id, "status": status, "decision_reason": reason,
              "raw_events": [e.evidence() for e in events], "raw_event_count": len(events),
              "kept_events": [], "dropped_events": [], "inferred_events": [], "inferred_amounts": [],
              "calculated_pot": None, "observed_payout": None, "pot_delta": None,
              "players": {}, "uncalled_returns": {}, "solution_count": len(solutions), **outcome}
    result.update(strict_source_uniqueness=strict_source_uniqueness,
                  required_blinds=['SB', 'BB'] if require_small_blind else ['BB'],
                  raw_solution_count=raw_solution_count,equivalent_interpretations=equivalent_interpretations)
    result["identity_hypotheses"] = hypotheses
    result["identity_candidates_rejected"] = rejected
    names = sorted({e.player for e in events if e.player})
    result["unresolved_nickname_relationships"] = [{"a": a, "b": b, "status": "UNRESOLVED",
        "reason": "OCR_SIMILARITY_ONLY_NO_INDEPENDENT_SEAT_PLAYER_CONTINUITY",
        "similarity": SequenceMatcher(None, a, b).ratio()} for a, b in combinations(names, 2)
        if SequenceMatcher(None, a, b).ratio() >= 0.8 and not alias_candidates]
    result["candidate_assembly_count"] = len(assemblies or [events])
    result["alternative_solutions"] = [{"grouping": [[r["source_index"] for r in e.source.get("rows", [])] for e in assembly],
                                         "identity_hypothesis": hypothesis, "ledger": ledger,
                                         "trace": state.trace, "dropped_events": dropped,
                                         "inferences": state.inferences}
                                        for state, kept, dropped, ledger, hypothesis, assembly in solutions]
    if outcome["exhausted"]:
        result["first_engine_blocker"] = {"stage": "CANDIDATE_ASSEMBLY_IDENTITY_OR_SOLVER_BUDGET", "reason": "Search not exhaustive"}
    elif len(solutions) > 1:
        result["first_engine_blocker"] = {"stage": "UNIQUENESS", "reason": "Multiple equally ranked source/identity interpretations"}
    if len(solutions) == 1 and hand_id is not None and not outcome["exhausted"]:
        state, kept, dropped, ledger, hypothesis, assembly = solutions[0]
        result.update(ledger)
        result.update(kept_events=[e.evidence() for e in kept], dropped_events=dropped,
                      inferred_amounts=state.inferences, clean_events=[e.evidence() for e in kept])
        result["uncalled_returns"] = {n: p.returned for n, p in state.players.items() if p.returned}
        result["diagnostic_trace"] = state.trace
        result["chosen_identity_hypothesis"] = hypothesis
        result["chosen_grouping"] = [[r["source_index"] for r in e.source.get("rows", [])] for e in assembly]
    result["kept_event_count"] = len(result["kept_events"])
    result["dropped_event_count"] = len(result["dropped_events"])
    result["inferred_event_count"] = len(result["inferred_events"])
    return result


def reconstruct_rows(rows, *, max_nodes=20000, start_stacks=None, identity_evidence=None):
    from .timeline import parse_rows
    from .assembly import candidate_assemblies
    groups = defaultdict(list)
    for row in rows:
        groups[(str(row.get("table_id", row.get("table", "UNKNOWN"))),
                str(row.get("session_id", "UNKNOWN")))].append(row)
    results = []
    for (table, session), source in groups.items():
        segments, current, number = [], [], None
        for event in parse_rows(source):
            observed_number = event.source.get("parsed_hand_id")
            if event.kind == "START_HAND" and observed_number and observed_number != number:
                if current:
                    segments.append((number, current))
                current, number = [], observed_number
            current.append(event)
        if current:
            segments.append((number, current))
        previous = []
        for index, (number, events) in enumerate(segments):
            raw_rows = [r for e in events for r in e.source["rows"]]
            assemblies, exhausted, graph = candidate_assemblies(raw_rows)
            result = reconstruct_hand(events, table=table, hand_id=number,
                                      at_eof=index == len(segments) - 1,
                                      previous_hand=previous, max_nodes=max_nodes,
                                      start_stacks=(start_stacks or {}).get(f"{table}:{number}"),
                                      assemblies=assemblies, assembly_exhausted=exhausted,
                                      alias_candidates=(identity_evidence or {}).get(f"{table}:{number}"))
            result["session_id"] = session
            result["candidate_grouping_graph"] = graph
            results.append(result)
            previous = events
    return results
