from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    kind: str
    player: str | None = None
    amount: int | None = None
    semantics: str = "DELTA"
    all_in: bool = False
    street: str | None = None
    source: dict[str, Any] = field(default_factory=dict)

    def signature(self):
        return self.kind, self.player, self.amount, self.semantics, self.all_in, self.street

    def evidence(self):
        return asdict(self)


@dataclass
class Player:
    status: str = "ACTIVE"
    street_commit: int = 0
    total_commit: int = 0
    returned: int = 0
    payout: int = 0
    start_stack: int | None = None


class RuleError(ValueError):
    pass
