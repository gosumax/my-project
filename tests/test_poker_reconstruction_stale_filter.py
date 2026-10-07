from parser_core.poker_reconstruction.models import Event
from parser_core.poker_reconstruction.reconstruct import reconstruct_hand


def simple():
    return [Event("ANTE", "a", 10), Event("ANTE", "b", 10), Event("SB", "a", 40),
            Event("BB", "b", 80), Event("DEAL"), Event("FOLD", "a"),
            Event("RETURN", "b", 40), Event("WIN", "b", 100)]


def test_winner_by_folds():
    h = reconstruct_hand(simple(), hand_id="fixture")
    assert h["status"] == "AUTO_RECOVERED"
    assert h["calculated_pot"] == 100


def test_duplicate_fold_and_late_ante():
    events = simple()
    events.insert(6, Event("FOLD", "a"))
    events.insert(5, Event("ANTE", "a", 10))
    h = reconstruct_hand(events, hand_id="fixture")
    assert h["status"] == "AMBIGUOUS"  # Either source FOLD can be real.


def test_duplicate_win_source_ambiguity():
    events = simple() + [Event("WIN", "b", 100)]
    h = reconstruct_hand(events, hand_id="fixture")
    assert h["status"] == "AMBIGUOUS"
    assert not h["dropped_events"]


def test_duplicate_return_uniquely_impossible():
    events = simple()
    events.insert(7, Event("RETURN", "b", 40))
    h = reconstruct_hand(events, hand_id="fixture")
    assert h["status"] == "AMBIGUOUS"


def test_missing_payout_end_and_start():
    assert reconstruct_hand(simple()[:-1], hand_id="fixture", at_eof=True)["status"] == "TRUNCATED_END"
    assert reconstruct_hand(simple()[4:], hand_id=None)["status"] == "TRUNCATED_START"


def test_unknown_amount_not_guessed():
    events = simple()
    events[3] = Event("BB", "b", None)
    assert reconstruct_hand(events, hand_id="fixture")["status"] == "AMBIGUOUS"


def test_unknown_raise_semantics():
    events = [Event("SB", "a", 5), Event("BB", "b", 10), Event("DEAL"),
              Event("RAISE", "a", 20, "UNKNOWN"), Event("FOLD", "b"),
              Event("RETURN", "a", 10), Event("WIN", "a", 20)]
    h = reconstruct_hand(events, hand_id="fixture")
    assert h["status"] == "RECOVERED_BY_LOGIC_MATH"
    assert h["inferred_amounts"][0]["interpretation"] == "TO"


def test_budget_is_ambiguous():
    assert reconstruct_hand(simple(), hand_id="fixture", max_nodes=1)["status"] == "AMBIGUOUS"


def test_show_does_not_replace_missing_streets():
    events = [Event("SB", "a", 5), Event("BB", "b", 10), Event("DEAL"),
              Event("CALL", "a", 5), Event("CHECK", "b"), Event("SHOW", "a"), Event("WIN", "a", 20)]
    assert reconstruct_hand(events, hand_id="fixture")["status"] == "INVALID"


def test_pot_equality_does_not_replace_missing_blinds():
    events = [Event("ANTE", "a", 10), Event("ANTE", "b", 10), Event("FOLD", "a"), Event("WIN", "b", 20)]
    assert reconstruct_hand(events, hand_id="fixture")["status"] == "INVALID"


def test_blind_amount_from_observed_call_delta():
    events = [Event("SB", "a", 5), Event("BB", "b", None), Event("DEAL"),
              Event("RAISE", "a", 20, "TO"), Event("CALL", "b", 10)]
    for street in ("FLOP", "TURN", "RIVER"):
        events.extend([Event("STREET", street=street), Event("CHECK", "a"), Event("CHECK", "b")])
    events.append(Event("WIN", "a", 40))
    hand = reconstruct_hand(events, hand_id="fixture")
    assert hand["status"] == "RECOVERED_BY_LOGIC_MATH"
    assert hand["inferred_amounts"][0]["amount"] == 10


def test_identical_bets_on_different_streets_are_kept():
    events = [Event("SB", "a", 5), Event("BB", "b", 10), Event("DEAL"),
              Event("CALL", "a", 5), Event("CHECK", "b")]
    for street in ("FLOP", "TURN"):
        events.extend([Event("STREET", street=street), Event("BET", "a", 10), Event("CALL", "b", 10)])
    events.extend([Event("STREET", street="RIVER"), Event("CHECK", "a"), Event("CHECK", "b"), Event("WIN", "a", 60)])
    hand = reconstruct_hand(events, hand_id="fixture")
    assert hand["status"] == "AUTO_RECOVERED"
    assert not hand["dropped_events"]


def test_old_bet_after_turn_uniquely_removed():
    events = [Event("SB", "a", 5), Event("BB", "b", 10), Event("DEAL"),
              Event("CALL", "a", 5), Event("CHECK", "b"), Event("STREET", street="FLOP"),
              Event("BET", "a", 10), Event("CALL", "b", 10), Event("STREET", street="TURN"),
              Event("BET", "a", 10), Event("CHECK", "a"), Event("CHECK", "b"),
              Event("STREET", street="RIVER"), Event("CHECK", "a"), Event("CHECK", "b"), Event("WIN", "a", 40)]
    hand = reconstruct_hand(events, hand_id="fixture")
    assert hand["status"] == "AUTO_RECOVERED"
    assert len(hand["dropped_events"]) == 1
