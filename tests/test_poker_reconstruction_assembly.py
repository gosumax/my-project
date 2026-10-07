from parser_core.poker_reconstruction.assembly import candidate_assemblies
from parser_core.poker_reconstruction.models import Event
from parser_core.poker_reconstruction.reconstruct import reconstruct_hand


def rows(texts):
    return [{"source_index": i + 1, "frame_id": i + 1, "table_id": "T01", "session_id": "s",
             "epoch": 1, "row_index": 1, "raw_ocr_text": text} for i, text in enumerate(texts)]


def test_three_row_raise_allin_keeps_raw_evidence():
    source = rows(["Dealer: bob raises", "1 326 400", "All-in"])
    candidates, exhausted, graph = candidate_assemblies(source)
    assert not exhausted
    assert len(candidates) == 1
    event = candidates[0][0]
    assert event.kind == "RAISE" and event.amount == 1326400 and event.all_in
    assert event.semantics == "UNKNOWN"
    assert event.source["rows"] == source
    assert len(graph["0"]) == 3  # All grouping hypotheses before poker pruning.


def test_neighbor_number_not_forced_to_belong_to_check():
    candidates, _, _ = candidate_assemblies(rows(["Dealer: bob checks", "100"]))
    assert any(e.kind == "UNKNOWN" for e in candidates[0])


def test_no_grouping_across_epoch():
    source = rows(["Dealer: bob bets", "100"])
    source[1]["epoch"] = 2
    candidates, _, _ = candidate_assemblies(source)
    assert not any(e.kind == "BET" and e.amount == 100 for path in candidates for e in path)


def alias_events():
    return [Event("SB", "a", 5), Event("BB", "bob", 10), Event("DEAL"),
            Event("FOLD", "a"), Event("RETURN", "boob", 5), Event("WIN", "bob", 10)]


def test_alias_needs_independent_evidence():
    h = reconstruct_hand(alias_events(), hand_id="fixture", alias_candidates=[{"a": "bob", "b": "boob"}])
    assert h["status"] not in {"AUTO_RECOVERED", "RECOVERED_BY_LOGIC_MATH"}
    assert h["identity_candidates_rejected"]


def test_alias_evidence_plus_unique_state():
    evidence = {"a": "bob", "b": "boob", "seat_a": 2, "seat_b": 2,
                "continuous_identity": True, "same_table_hand_context": True,
                "identity_evidence_refs": ["observed_seat_track_receipt"]}
    h = reconstruct_hand(alias_events(), hand_id="fixture", alias_candidates=[evidence])
    assert h["status"] == "RECOVERED_BY_LOGIC_MATH"
    assert h["chosen_identity_hypothesis"]["decisions"][0]["hypothesis"] == "A==B"
    assert len(h["identity_hypotheses"]) == 2


def test_unknown_call_to_uses_target_not_delta():
    events = [Event("SB", "a", 5), Event("BB", "b", 10), Event("DEAL"),
              Event("RAISE", "a", 20, "TO"), Event("CALL", "b", None, "TO")]
    for street in ("FLOP", "TURN", "RIVER"):
        events.extend([Event("STREET", street=street), Event("CHECK", "a"), Event("CHECK", "b")])
    events.append(Event("WIN", "a", 40))
    h = reconstruct_hand(events, hand_id="fixture")
    assert h["status"] == "RECOVERED_BY_LOGIC_MATH"
    assert h["inferred_amounts"][0]["amount"] == 20


def test_malformed_action_after_street_is_not_card_info():
    source = rows(["Dealer: Dealing Flop", "a raisxs 100"])
    source[0].update(frame_id=1, row_index=1)
    source[1].update(frame_id=1, row_index=2)
    candidates, _, _ = candidate_assemblies(source)
    assert candidates[0][-1].kind == "UNKNOWN"


def test_null_row_coordinates_are_preserved():
    source = rows(["Dealer: Dealing Flop", "unknown"])
    source[0]["row_index"] = None
    source[1]["row_index"] = None
    candidates, _, _ = candidate_assemblies(source)
    assert candidates[0][-1].kind == "UNKNOWN"


def test_unattachable_orphan_preserves_other_grouping():
    source = rows(["Dealer: a posts Ante", "10", "Dealer: b checks", "100"])
    candidates, _, _ = candidate_assemblies(source)
    assert candidates[0][0].amount == 10
    assert candidates[0][-1].kind == "UNKNOWN"


def test_alias_hypotheses_both_valid_are_ambiguous():
    evidence = {"a": "bob", "b": "boob", "seat_a": 2, "seat_b": 2,
                "continuous_identity": True, "same_table_hand_context": True,
                "identity_evidence_refs": ["observed_seat_track_receipt"]}
    events = [Event("SB", "a", 5), Event("BB", "bob", 10), Event("DEAL"),
              Event("FOLD", "a"), Event("RETURN", "bob", 5), Event("WIN", "bob", 10),
              Event("INFO", "boob")]
    hand = reconstruct_hand(events, hand_id="fixture", alias_candidates=[evidence])
    assert hand["status"] == "AMBIGUOUS"
    assert hand["solution_count"] == 2
