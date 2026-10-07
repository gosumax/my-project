import pytest
from parser_core.poker_reconstruction.models import Event, Player, RuleError
from parser_core.poker_reconstruction.state_machine import State
from parser_core.poker_reconstruction.accounting import audit, pot_layers, payout_feasible


@pytest.mark.parametrize("called", [0, 40])
def test_uncalled_return(called):
    s = State(players={"a": Player(), "b": Player()})
    s.apply(Event("BET", "a", 100))
    if called:
        s.apply(Event("CALL", "b", called, all_in=True))
    else:
        s.apply(Event("FOLD", "b"))
    s.apply(Event("RETURN", "a", 100 - called))
    assert audit(s)["calculated_pot"] == called * 2
    with pytest.raises(RuleError, match="RETURN"):
        s.apply(Event("RETURN", "a", 100 - called))


def test_side_pots_folded_eligibility():
    players = {"a": Player(total_commit=100, status="ALL_IN"),
               "b": Player(total_commit=200), "c": Player(total_commit=200, status="FOLDED")}
    layers = pot_layers(players)
    assert [p["amount"] for p in layers] == [300, 200]
    assert payout_feasible(layers, {"a": 300, "b": 200})
    assert not payout_feasible(layers, {"a": 500})
    assert not payout_feasible(layers, {"c": 500})


def test_split_payout():
    layers = pot_layers({"a": Player(total_commit=100), "b": Player(total_commit=100)})
    assert payout_feasible(layers, {"a": 100, "b": 100})
