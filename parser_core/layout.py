"""Validated config-driven fixed 3x3 layout for bounded nine-table pilots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Slot:
    slot_id: int
    table_box: tuple[int, int, int, int]
    chat_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class FixedLayout:
    layout_id: str
    source_width: int
    source_height: int
    slots: tuple[Slot, ...]
    calibration_schema: str


def load_fixed_layout(layout_path: Path, profile_path: Path,
                      calibration_path: Path) -> FixedLayout:
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if layout.get("schema") != "pokerdom.nine_table_layout.v1":
        raise ValueError("Unexpected layout schema")
    if profile.get("schema") != "pokerdom.layout_profile_847x404.slot_routed.v1":
        raise ValueError("Unexpected slot profile schema")
    if calibration.get("schema") != "pokerdom.chat-log-bottom-rois.v1":
        raise ValueError("Unexpected chat calibration schema")
    source_width, source_height = map(int, layout["source_resolution_wh"])
    if len(layout["tables"]) != 9 or len(profile["slots"]) != 9:
        raise ValueError("Exactly nine configured slots are required")
    if set(calibration.get("slots", {})) != set(profile["slots"]):
        raise ValueError("Chat calibration slots do not match profile")
    slots = []
    for row in layout["tables"]:
        slot_id = int(row["table"])
        x, y = int(row["x"]), int(row["y"])
        width, height = int(row["width"]), int(row["height"])
        table_box = (x, y, x + width, y + height)
        slot_key = f"slot_{slot_id:02d}"
        chat_local = tuple(profile["slots"][slot_key]["chat"]["chat_log"])
        bottom = calibration["slots"][slot_key]
        if type(bottom) is not int or not chat_local[3] < bottom <= height:
            raise ValueError(f"Invalid chat calibration for slot {slot_id}")
        chat_local = (*chat_local[:3], bottom)
        if any(type(coordinate) is not int for coordinate in chat_local):
            raise ValueError(f"Invalid chat coordinates for slot {slot_id}")
        if not (0 <= x < x + width <= source_width and
                0 <= y < y + height <= source_height and
                0 <= chat_local[0] < chat_local[2] <= width and
                0 <= chat_local[1] < chat_local[3] <= height):
            raise ValueError(f"Invalid geometry for slot {slot_id}")
        chat_box = (x + chat_local[0], y + chat_local[1],
                    x + chat_local[2], y + chat_local[3])
        slots.append(Slot(slot_id, table_box, chat_box))
    slots.sort(key=lambda slot: slot.slot_id)
    if [slot.slot_id for slot in slots] != list(range(1, 10)):
        raise ValueError("Slot IDs must be 1..9")
    for index, first in enumerate(slots):
        for second in slots[index + 1:]:
            if (max(first.table_box[0], second.table_box[0]) <
                min(first.table_box[2], second.table_box[2]) and
                max(first.table_box[1], second.table_box[1]) <
                min(first.table_box[3], second.table_box[3])):
                raise ValueError(f"Overlapping slots {first.slot_id}, {second.slot_id}")
    return FixedLayout(layout["layout_id"], source_width, source_height,
                       tuple(slots), calibration["schema"])
