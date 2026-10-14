"""Saved-data candidate: first parseable fixed-seat number, without admission gates."""
import re

NUMBER = re.compile(r"\d+(?:[ \u00a0]\d{3})*")
POLICY = "FIRST_PARSEABLE_NUMERIC_FIXED_STACK_ROI_AFTER_HAND_START_V7"


def parse_numeric(raw):
    """Reuse the stack integer text grammar, without score/status qualification."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not NUMBER.fullmatch(text):
        return None
    return int(text.replace(" ", "").replace("\u00a0", ""))


def first_numeric(observations, *, table_id, session_id, seat, hand_start, hand_end):
    """ROI route identifies seat; frame bounds identify the current hand.

    Same-frame readings use saved source order as a deterministic tie break.
    Conflicting readings are diagnostics and never reject the first number.
    """
    for row in sorted(observations, key=lambda r: (r["frame"], r.get("source_order", 0))):
        if (row["table_id"], row["session_id"], row["seat"]) != (table_id, session_id, seat):
            continue
        if not hand_start <= row["frame"] <= hand_end:
            continue
        value = parse_numeric(row.get("raw_text"))
        if value is not None:
            return dict(row, displayed_stack=value)
    return None


def add_forced(displayed, forced):
    """Unknown forced corrections do not alter the independent displayed fact."""
    if displayed is None or any(forced.get(role) is None for role in ("ANTE", "SB", "BB")):
        return None
    return displayed + sum(forced[role] for role in ("ANTE", "SB", "BB"))
