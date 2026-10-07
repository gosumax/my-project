import pytest
from parser_core.poker_reconstruction.models import Event, Player, RuleError
from parser_core.poker_reconstruction.state_machine import State


def state():
    return State(players={name: Player() for name in ("a", "b", "c")})


def test_ante_blinds_call_raise_to():
    s = state()
    for n in s.players:
        s.apply(Event("ANTE", n, 10))
    s.apply(Event("SB", "a", 40))
    s.apply(Event("BB", "b", 80))
    s.apply(Event("DEAL"))
    s.apply(Event("RAISE", "c", 160, "TO"))
    s.apply(Event("CALL", "b", 160, "TO"))
    assert s.players["b"].total_commit == 170
    s.apply(Event("RAISE", "a", 320, "TO"))
    assert s.players["a"].total_commit == 330
    assert s.current_bet == 320


@pytest.mark.parametrize("kind", ["FOLD", "CHECK", "CALL", "BET", "RAISE"])
def test_fold_terminal(kind):
    s = state()
    s.apply(Event("FOLD", "a"))
    with pytest.raises(RuleError, match="TERMINAL"):
        s.apply(Event(kind, "a", 10))


def test_check_facing_bet():
    s = state()
    s.apply(Event("BET", "a", 10))
    with pytest.raises(RuleError, match="CHECK_FACING"):
        s.apply(Event("CHECK", "b"))


def test_transition_pending_response():
    s = state()
    s.apply(Event("BET", "a", 10))
    with pytest.raises(RuleError, match="PENDING"):
        s.apply(Event("STREET", street="FLOP"))


def test_street_reset():
    s = state()
    for street in ("FLOP", "TURN", "RIVER"):
        for n in s.players:
            s.apply(Event("CHECK", n))
        s.apply(Event("STREET", street=street))
        assert not any(p.street_commit for p in s.players.values())


def test_short_all_in_raise_and_call():
    s = state()
    s.apply(Event("BET", "a", 100))
    s.apply(Event("RAISE", "b", 140, "TO", True))
    s.apply(Event("CALL", "c", 80, all_in=True))
    with pytest.raises(RuleError, match="REOPEN"):
        s.apply(Event("RAISE", "a", 240, "TO"))
    s.apply(Event("CALL", "a", 40))
    assert s.closed()


def test_late_ante():
    s = state()
    s.apply(Event("DEAL"))
    with pytest.raises(RuleError, match="ANTE"):
        s.apply(Event("ANTE", "a", 10))


def test_exact_stack_all_in():
    s = State(players={"a": Player(start_stack=100), "b": Player()})
    s.apply(Event("BET", "a", None, all_in=True))
    assert s.players["a"].total_commit == 100
    assert s.inferences[0]["reason"] == "EXACT_STACK_ALL_IN"


def test_exceeds_stack():
    s = State(players={"a": Player(start_stack=10)})
    with pytest.raises(RuleError, match="EXCEEDS_STACK"):
        s.apply(Event("BET", "a", 11))


def test_unknown_show_actor_is_diagnostic():
    s = state()
    for n in s.players:
        s.apply(Event("CHECK", n))
    with pytest.raises(RuleError, match="UNKNOWN_SHOW_PLAYER"):
        s.apply(Event("SHOW", "ghost"))


def test_duplicate_check_requires_new_response():
    s = state()
    s.apply(Event("CHECK", "a"))
    with pytest.raises(RuleError, match="PENDING_TURN"):
        s.apply(Event("CHECK", "a"))


@pytest.mark.parametrize("kind", ["ANTE", "SB", "BB"])
def test_forced_allin_must_exhaust_exact_stack(kind):
    s = State(players={"a": Player(start_stack=100)})
    with pytest.raises(RuleError, match="EXHAUST"):
        s.apply(Event(kind, "a", 5, all_in=True))


def test_ante_allin_terminal():
    s = State(players={"a": Player(start_stack=10)})
    s.apply(Event("ANTE", "a", 10, all_in=True))
    assert s.players["a"].status == "ALL_IN"


def test_blinds_have_different_actors():
    s = state()
    s.apply(Event("SB", "a", 5))
    with pytest.raises(RuleError, match="BOTH_BLINDS"):
        s.apply(Event("BB", "a", 10))
