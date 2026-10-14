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

    @property
    def remaining_stack(self):
        """Exact saved initial stack less net paid chips; unknown stays unknown."""
        if self.start_stack is None:
            return None
        return self.start_stack - (self.total_commit - self.returned)


class RuleError(ValueError):
    pass
