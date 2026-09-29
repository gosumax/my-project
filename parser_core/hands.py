"""Partition ordered candidate events into hand windows without certification."""

from __future__ import annotations

from dataclasses import dataclass, field


REDUCER_VERSION = "partial_reducer_v1"


@dataclass(frozen=True)
class OrderedEvent:
    event_id: str
    revision: int
    table_session_id: str
    order_key: str
    event_type: str
    hand_number_candidate: str | None


@dataclass
class HandWindow:
    events: list[OrderedEvent] = field(default_factory=list)
    start_boundary: str = "OPEN_START"
    end_boundary: str = "OPEN_END"

    @property
    def hand_numbers(self) -> list[str]:
        return sorted({event.hand_number_candidate for event in self.events
                       if event.hand_number_candidate})


def partition(events: list[OrderedEvent]) -> list[HandWindow]:
    windows: list[HandWindow] = []
    current: HandWindow | None = None
    for event in sorted(events, key=lambda item: item.order_key):
        if event.event_type == "HAND_HEADER":
            if current is not None and current.events:
                current.end_boundary = "NEXT_HAND_HEADER"
                windows.append(current)
            current = HandWindow(start_boundary="HAND_HEADER")
        elif event.event_type == "NEW_HAND":
            if current is None:
                current = HandWindow(start_boundary="NEW_HAND")
            elif any(old.event_type not in {"HAND_HEADER", "HAND_ID"}
                     for old in current.events):
                current.end_boundary = "NEXT_NEW_HAND"
                windows.append(current)
                current = HandWindow(start_boundary="NEW_HAND")
        elif current is None:
            current = HandWindow()
        current.events.append(event)
    if current is not None and current.events:
        windows.append(current)
    return windows
