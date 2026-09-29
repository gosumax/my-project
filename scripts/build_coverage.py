"""Turn existing semantic catalogs into an explicit, unverified coverage matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAT = Path(r"D:\Проэкты\V2.0 PD\outputs\chat_only_inventory_smoke11_20260927")
ENTITIES = Path(r"D:\Проэкты\V2.0 PD\outputs\chat_entities_smoke11_20260927")
BOUNDARY = {"HAND_HEADER", "HAND_ID", "NEW_HAND", "DEAL", "FLOP", "TURN", "RIVER"}
ACTION = {"ANTE", "SMALL_BLIND", "BIG_BLIND", "FOLD", "CHECK", "CALL", "BET", "RAISE", "RETURN", "WIN", "SHOW", "NO_SHOW"}
FRAGMENT = {"ALL_IN_FRAGMENT", "RETURN_FRAGMENT", "POT_FRAGMENT", "COMBINATION_FRAGMENT", "TEXT_FRAGMENT", "UNREADABLE", "UNKNOWN_EVENT"}
ENTITY_VISUAL = {
    "PLAYER_NICK": "NICK", "ACTION_PHRASE": "TECH", "MONEY_AMOUNT": "VALUE",
    "HAND_NUMBER": "HAND_NUMBER", "BLIND_LEVEL": "VALUE+TECH", "STREET": "TECH",
    "ALL_IN_MARKER": "TECH", "POT_REFERENCE": "TECH", "COMBINATION": "TECH",
    "CARD_TEXT_CANDIDATE": "CARD_TEXT", "TIME_DURATION": "VALUE+TECH",
    "SYSTEM_MESSAGE": "SYSTEM",
}
MONEY_ROLES = [
    "ANTE", "CALL", "RAISE_TO", "BET", "RETURNED_UNCALLED_FRAGMENT", "WIN_PAYOUT",
    "POT_SHARE_FRAGMENT", "BIG_BLIND", "SMALL_BLIND", "ALL_IN_FRAGMENT",
    "BOUNTY_RUB", "LEVEL_ANTE", "LEVEL_BIG_BLIND", "LEVEL_SMALL_BLIND",
]


def main() -> None:
    events = json.loads((CHAT / "summary.json").read_text(encoding="utf-8"))["counts_by_type"]
    entities = json.loads((ENTITIES / "summary.json").read_text(encoding="utf-8"))["counts_by_type"]
    records = []
    for semantic_type, count in sorted(events.items()):
        group = "boundary" if semantic_type in BOUNDARY else (
            "action" if semantic_type in ACTION else (
                "fragment" if semantic_type in FRAGMENT else "service"))
        rule = {
            "boundary": "Link same-hand headers; require ordered table context",
            "action": "Link actor, amount role and complete message before hand reducer",
            "fragment": "Join fragments into a complete message; never post alone",
            "service": "Preserve service event; no automatic pot change",
        }[group]
        records.append(("message", semantic_type, count, str(CHAT / "chat_event_candidates.csv"),
                        "CATALOG_ONLY", "SYSTEM" if group in {"boundary", "service"} else "MIXED",
                        "full-message OCR candidate; field reader undecided", rule, "NOT_TESTED"))
    for semantic_type, count in sorted(entities.items()):
        records.append(("entity", semantic_type, count, str(ENTITIES / "chat_entity_mentions.csv"),
                        "CATALOG_ONLY", ENTITY_VISUAL[semantic_type],
                        "field OCR candidate; engine undecided",
                        "Attach span to message; validate semantic role and provenance", "NOT_TESTED"))
    for role in MONEY_ROLES:
        records.append(("amount_role", role, "", str(ENTITIES / "money_amounts.csv"),
                        "CATALOG_ONLY", "VALUE", "strict decimal parse after OCR",
                        "Use role and unit; FRAGMENT needs assembled message; LEVEL is not a contribution",
                        "NOT_TESTED"))
    target = ROOT / "docs" / "coverage.csv"
    target.parent.mkdir(exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("kind", "semantic_type", "catalog_count", "source_examples", "review_status",
                         "proposed_visual_class", "proposed_reader", "assembly_rule", "independent_test"))
        writer.writerows(records)
    print(f"Wrote {len(records)} coverage rows to {target}")


if __name__ == "__main__":
    main()
