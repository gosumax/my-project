from dataclasses import dataclass, field
from .models import Player, RuleError

STREETS = ["PREFLOP", "FLOP", "TURN", "RIVER", "SHOWDOWN"]


@dataclass
class State:
    players: dict = field(default_factory=dict)
    street: str = "PREFLOP"
    phase: str = "HAND_START"
    current_bet: int = 0
    min_raise: int = 0
    big_blind: int = 0
    pending: set = field(default_factory=set)
    acted_at: dict = field(default_factory=dict)
    ante_players: set = field(default_factory=set)
    blind_kinds: set = field(default_factory=set)
    blind_actors: set = field(default_factory=set)
    trace: list = field(default_factory=list)
    inferences: list = field(default_factory=list)
    seen_payout: bool = False

    def active(self):
        return {name for name, p in self.players.items() if p.status == "ACTIVE"}

    def contenders(self):
        return {name for name, p in self.players.items() if p.status != "FOLDED"}

    def closed(self):
        if len(self.contenders()) == 1:
            return True
        return not (self.pending & self.active()) and all(
            p.street_commit == self.current_bet for p in self.players.values() if p.status == "ACTIVE")

    def invest(self, name, delta, street=True):
        if type(delta) is not int or delta < 0:
            raise RuleError("INVALID_INTEGER_DELTA")
        p = self.players[name]
        if p.start_stack is not None and p.total_commit - p.returned + delta > p.start_stack:
            raise RuleError("CONTRIBUTION_EXCEEDS_STACK")
        p.total_commit += delta
        if street:
            p.street_commit += delta

    def apply(self, event, interpretation=None):
        k, name, amount = event.kind, event.player, event.amount
        if k in {"UNKNOWN", "UNPARSED_ACTION"}:
            raise RuleError("UNPARSED_MATERIAL_EVENT")
        if k in {"START_HAND", "INFO"}:
            return
        if k == "DEAL":
            if self.phase not in {"HAND_START", "ANTES", "BLINDS"}:
                raise RuleError("REPEATED_DEAL")
            self.phase = "PREFLOP"
            self.pending = self.active()
            return
        if k == "STREET":
            if event.street not in STREETS or STREETS.index(event.street) != STREETS.index(self.street) + 1:
                raise RuleError("STREET_ORDER")
            if not self.closed():
                raise RuleError("STREET_WITH_PENDING_RESPONSE")
            if self.seen_payout:
                raise RuleError("STREET_AFTER_PAYOUT")
            self.street = event.street
            self.phase = event.street
            self.current_bet = 0
            self.min_raise = self.big_blind
            self.acted_at = {}
            for p in self.players.values():
                p.street_commit = 0
            self.pending = self.active() if len(self.contenders()) > 1 else set()
            # With only one live player facing all-ins, no check is required.
            if len(self.active()) <= 1:
                self.pending = set()
            return
        if k == "SHOW":
            if not self.closed():
                raise RuleError("SHOW_WITH_PENDING_RESPONSE")
            if name not in self.players:
                raise RuleError("UNKNOWN_SHOW_PLAYER")
            if self.players[name].status == "FOLDED":
                raise RuleError("SHOW_AFTER_FOLD")
            # Showing cards can happen before an all-in runout. It cannot
            # manufacture the missing streets or make an early payout legal.
            if self.street == "RIVER":
                self.phase = "SHOWDOWN"
            return
        if name not in self.players:
            raise RuleError("UNKNOWN_PLAYER")
        p = self.players[name]
        if k == "WIN":
            if not self.closed() or (len(self.contenders()) > 1 and self.street not in {"RIVER", "SHOWDOWN"}):
                raise RuleError("PAYOUT_BEFORE_TERMINAL_STATE")
            if p.status == "FOLDED" or type(amount) is not int or amount <= 0:
                raise RuleError("INVALID_PAYOUT")
            p.payout += amount
            self.seen_payout = True
            self.phase = "HAND_END"
            return
        if k == "RETURN":
            if type(amount) is not int or amount <= 0 or not self.closed() or self.seen_payout:
                raise RuleError("INVALID_RETURN_PHASE")
            other = max((q.street_commit for actor, q in self.players.items() if actor != name), default=0)
            if amount != p.street_commit - other or amount > p.total_commit - p.returned:
                raise RuleError("RETURN_NOT_UNMATCHED_EXCESS")
            p.returned += amount
            p.street_commit -= amount
            self.current_bet = max(q.street_commit for q in self.players.values())
            return
        if p.status != "ACTIVE" or self.seen_payout:
            raise RuleError("ACTION_AFTER_TERMINAL_PLAYER_OR_PAYOUT")
        if k == "ANTE":
            if self.phase not in {"HAND_START", "ANTES"} or name in self.ante_players:
                raise RuleError("LATE_OR_REPEATED_ANTE")
            if type(amount) is not int or amount <= 0:
                raise RuleError("INVALID_ANTE")
            self.invest(name, amount, False)
            self.ante_players.add(name)
            self.phase = "ANTES"
            if event.all_in:
                if p.start_stack is not None and p.total_commit != p.start_stack:
                    raise RuleError("ALL_IN_DOES_NOT_EXHAUST_EXACT_STACK")
                p.status = "ALL_IN"
            return
        if k in {"SB", "BB"}:
            if self.phase not in {"HAND_START", "ANTES", "BLINDS"} or k in self.blind_kinds:
                raise RuleError("LATE_OR_REPEATED_BLIND")
            if name in self.blind_actors:
                raise RuleError("SAME_PLAYER_POSTS_BOTH_BLINDS")
            if type(amount) is not int or amount <= 0:
                raise RuleError("INVALID_BLIND")
            self.invest(name, amount)
            self.blind_kinds.add(k)
            self.blind_actors.add(name)
            self.current_bet = max(self.current_bet, p.street_commit)
            self.min_raise = self.current_bet
            if k == "BB":
                self.big_blind = amount
            self.phase = "BLINDS"
            if event.all_in:
                if p.start_stack is not None and p.total_commit != p.start_stack:
                    raise RuleError("ALL_IN_DOES_NOT_EXHAUST_EXACT_STACK")
                p.status = "ALL_IN"
            return
        if self.phase in {"HAND_START", "ANTES", "BLINDS"}:
            self.phase = "PREFLOP"
            self.pending = self.active()
        if name not in self.pending:
            raise RuleError("ACTION_WITHOUT_PENDING_TURN")
        if k == "FOLD":
            p.status = "FOLDED"
            self.pending.discard(name)
            return
        owed = self.current_bet - p.street_commit
        if k == "CHECK":
            if owed:
                raise RuleError("CHECK_FACING_BET")
        elif k in {"CALL", "BET", "RAISE", "ALL_IN"}:
            semantics = interpretation or event.semantics
            if amount is None:
                if not event.all_in or p.start_stack is None:
                    raise RuleError("UNKNOWN_ALL_IN_AMOUNT")
                delta = p.start_stack - (p.total_commit - p.returned)
                self.inferences.append({"source": event.source, "amount": delta, "reason": "EXACT_STACK_ALL_IN"})
            else:
                delta = amount - p.street_commit if semantics == "TO" else amount
            target = p.street_commit + delta
            if k == "CALL" or (k == "ALL_IN" and target <= self.current_bet):
                if delta <= 0 or (delta != owed and not (event.all_in and delta < owed)):
                    raise RuleError("CALL_AMOUNT_NOT_TO_CALL")
            else:
                if k == "BET" and self.current_bet != 0:
                    raise RuleError("BET_FACING_EXISTING_BET")
                if target <= self.current_bet:
                    raise RuleError("RAISE_NOT_ABOVE_CURRENT_BET")
                increase = target - self.current_bet
                if increase < self.min_raise and not event.all_in:
                    raise RuleError("BELOW_MINIMUM_RAISE")
                if name in self.acted_at and self.current_bet - self.acted_at[name] < self.min_raise:
                    raise RuleError("SHORT_RAISE_DID_NOT_REOPEN")
                if increase >= self.min_raise:
                    self.min_raise = increase
                self.current_bet = target
                self.pending = self.active() - {name}
            self.invest(name, delta)
            if event.all_in:
                if p.start_stack is not None and p.total_commit - p.returned != p.start_stack:
                    raise RuleError("ALL_IN_DOES_NOT_EXHAUST_EXACT_STACK")
                p.status = "ALL_IN"
            if event.semantics == "UNKNOWN":
                self.inferences.append({"source": event.source, "interpretation": semantics,
                                        "delta": delta, "reason": "UNIQUE_LEGAL_EXACT_ACCOUNTING_INTERPRETATION"})
        else:
            raise RuleError("UNSUPPORTED_EVENT")
        self.pending.discard(name)
        self.acted_at[name] = self.current_bet
