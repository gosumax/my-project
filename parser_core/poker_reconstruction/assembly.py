"""Candidate event graph over raw rows; grouping is decided by poker constraints."""
from dataclasses import replace
import re
from .models import Event
from .timeline import parse_text, raw_text, NUMBER, verified_board_payload
from parser_core.poker_hand_assembly import split_small_blind, observed_wrapped_continuation
from parser_core.poker_numeric_evidence import numeric_alternatives, TOKEN
from parser_core.win_tail_evidence import verified_win_tail, current_win_context
from parser_core.card_set_evidence import validated_rows, verified_group
from parser_core.table_service_events import describe_service
from parser_core.service_text_normalizer import (normalize_service, physical_service_group,
    literal_win_pot_continuation)


def candidate_assemblies(rows, max_candidates=2048):
    # A retained card proof cannot authorize a different current source hand.
    # Withhold only derived authority in local copies; keep all raw observations.
    rows=validated_rows(rows)
    for i,row in enumerate(rows):
        if row.get('win_best_five_proof') and not verified_win_tail(
                rows[i:i+3],context=current_win_context(rows,i)):
            rows[i]=dict(row,win_best_five_conflict='CURRENT_SOURCE_WINDOW_CONFLICT')
            rows[i].pop('win_best_five_proof',None)
    # Physical adjacency is evidence, not truth. Across-frame neighboring rows
    # are hypotheses within a presegmented numbered hand, validated downstream.
    # A fully native-pixel-bound board/SHOW continuation is one source object.
    # Do not invent a competing singleton INFO interpretation of those pixels.
    proven_groups={}
    for i,row in enumerate(rows):
        if literal_win_pot_continuation(rows[i:i+2]):
            proven_groups[i]=i+2
        for end in range(i + 1, min(i + 4, len(rows))):
            group = rows[i:end + 1]
            if not observed_wrapped_continuation(group[:-1], group[-1]):
                break
            notice = describe_service(' '.join(raw_text(r) for r in group))
            if (notice and notice['kind'] in {'BOUNTY', 'RESEATING'}
                    and not notice.get('continuation_pending')):
                proven_groups[i] = end + 1
        proof=row.get('unique_card_set_proof')
        if proof:
            width=len(proof.get('target_refs',[]))
            if verified_group(proof,rows[i:i+width]):proven_groups[i]=i+width
        if row.get('win_best_five_proof') and i+2<len(rows) and verified_win_tail(rows[i:i+3]):
            proven_groups[i]=i+3
        if (i+1<len(rows) and physical_service_group(rows[i:i+2])
                and observed_wrapped_continuation([row],rows[i+1])):
            head=parse_text(raw_text(row),{})
            joined=parse_text(raw_text(row)+' '+raw_text(rows[i+1]),{'rows': rows[i:i+2]})
            service=describe_service(raw_text(row)+' '+raw_text(rows[i+1]))
            if joined.kind=='INFO' and not (service and service.get('continuation_pending')):
                # The existing literal grammar and exact physical adjacency
                # prove one wrapped server notice. It is not two competing
                # poker histories. Retain both source rows in that object.
                proven_groups[i]=i+2
        if not row.get('card_row_id'):continue
        for end in range(i+1,min(i+3,len(rows))):
            if observed_wrapped_continuation(rows[i:end],rows[end]):
                if rows[end].get('stable_row_id',rows[end].get('row_id'))==row['card_row_id']:
                    proven_groups[i]=end+1
    graph = {}
    for i, row in enumerate(rows):
        choices = []
        for width in range(1, min(4, len(rows) - i) + 1):
            if i in proven_groups and i+width!=proven_groups[i]:continue
            group = rows[i:i + width]
            if width > 1:
                continuation = raw_text(group[-1]).strip()
                if physical_service_group(group):
                    continuation = normalize_service(continuation,
                        previous_clause=' '.join(raw_text(r) for r in group[:-1])).normalized_text
                if re.match(r"^(?:\[\d+:\d+\]|Dealer\s*[:;]|Дилер:|Hand #|#\d+)", continuation, re.I):
                    break
                previous = group[-2]
                if any(group[-1].get(k) != previous.get(k) for k in ("table_id", "session_id", "epoch")):
                    break
                first = parse_text(raw_text(row), {})
                money_tail = re.fullmatch(NUMBER + r"(?:\s+(?:back|and is All[ -]?in|is All[ -]?in))?[.!●]?|(?:and is All[ -]?in|is All[ -]?in|All[ -]?in|back)[.!●]?", continuation, re.I)
                payout_tail = re.fullmatch(r"(?:from )?(?:the )?(?:main pot|side pot(?: #\d+)?)(?: with .*)?[.!●]?", continuation, re.I)
                rank_tail = re.fullmatch(r"(?:High card|Pair|Two pairs|Three of a kind|Straight|Flush|Full house|Four of a kind|Straight flush)[ ,].+", continuation, re.I)
                malformed_money_tail = first.kind in {'RAISE','CALL','BET'} and first.amount is None and re.fullmatch(TOKEN, continuation)
                if not (malformed_money_tail or observed_wrapped_continuation(group[:-1],group[-1]) or
                        (width == 2 and split_small_blind(row, group[-1])) or
                        (first.kind in {"ANTE", "SB", "BB", "CALL", "BET", "RAISE", "RETURN", "WIN"} and money_tail)
                        or (first.kind == "WIN" and payout_tail) or (first.kind == "SHOW" and rank_tail)):
                    break
            source = {"rows": group, "source_index": row["source_index"],
                      "frame": row.get("frame_id"), "row_id": row.get("stable_row_id", row.get("row_id")),
                      "raw_text": " ".join(raw_text(r) for r in group),
                      "grouping": "SINGLE_ROW" if width == 1 else "CANDIDATE_NEIGHBOR_CONTINUATION",
                      "physical_adjacency": all(a.get("frame_id") == b.get("frame_id") and
                          isinstance(a.get("row_index"), int) and b.get("row_index") == a["row_index"] + 1
                          for a, b in zip(group, group[1:]))}
            event = parse_text(source["raw_text"], source)
            numeric = numeric_alternatives(event)
            if numeric:
                choices.extend((i + width, candidate) for candidate in numeric)
                continue
            if width > 1 and event.amount is None and event.kind in {"ANTE", "SB", "BB", "CALL", "BET", "RAISE", "RETURN", "WIN"}:
                continue
            # Singleton orphan monetary/all-in tokens cannot be poker actions.
            # Their grouped candidates still retain every raw row. Other UNKNOWN
            # rows remain barriers, never silently classified as financial INFO.
            if width == 1 and event.kind == "UNKNOWN" and re.fullmatch(NUMBER + r"(?:\s+(?:back|and is All[ -]?in|is All[ -]?in))?[.!●]?|(?:and is All[ -]?in|is All[ -]?in|All[ -]?in|back)[.!●]?", raw_text(row).strip(), re.I):
                continue
            if width == 1 and event.kind == "UNKNOWN" and i:
                prior = parse_text(raw_text(rows[i - 1]), {})
                if prior.kind == "STREET" and verified_board_payload(rows[i - 1], row):
                    event = replace(event, kind="INFO", source=dict(source, interpretation="OPAQUE_BOARD_PAYLOAD_NO_CARD_TRUTH"))
            choices.append((i + width, event))
        graph[i] = choices
    # An orphan token with no possible parent grouping must remain an UNKNOWN
    # barrier. It must not force every other correctly grouped event to be
    # reparsed as single raw rows through the diagnostic fallback.
    for i, choices in graph.items():
        if not choices and not any(start < i < end for start, edges in graph.items() for end, _ in edges):
            row = rows[i]
            source = {"rows": [row], "source_index": row["source_index"], "frame": row.get("frame_id"),
                      "raw_text": raw_text(row), "row_id": row.get("row_id"), "grouping": "UNRESOLVED_ORPHAN_TOKEN"}
            choices.append((i + 1, Event("UNKNOWN", source=source)))
    paths, exhausted = [], False
    stack = [(0, [])]
    while stack:
        position, events = stack.pop()
        if position == len(rows):
            paths.append(events)
            if len(paths) >= max_candidates and stack:
                exhausted = True
                break
            continue
        for end, event in reversed(graph[position]):
            stack.append((end, events + [event]))
    if not paths:
        # Diagnostic path preserves every row when no supported full assembly
        # exists. The solver's UNKNOWN barrier prevents accepting it.
        paths = [[parse_text(raw_text(r), {"rows": [r], "source_index": r["source_index"],
                  "frame": r.get("frame_id"), "raw_text": raw_text(r), "row_id": r.get("row_id")}) for r in rows]]
    return paths, exhausted, {str(i): [{"next_row_offset": end, "event": e.evidence()} for end, e in choices] for i, choices in graph.items()}
