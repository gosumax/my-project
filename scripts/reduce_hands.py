"""Create PARTIAL hand revisions from event candidates; never certify them."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.hands import REDUCER_VERSION, OrderedEvent, partition
from parser_core.journal import Journal, stable_id
from parser_core.message_assembly import ASSEMBLER_VERSION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    journal = Journal(run / "journal.sqlite3")
    events = journal.db.execute("""SELECT e.event_id,e.revision,e.table_session_id,
        e.order_key,e.event_type,e.evidence_json,s.state
        FROM event_revisions e JOIN messages m ON m.message_id=e.message_id
        JOIN table_sessions t ON t.table_session_id=e.table_session_id
        JOIN sources s ON s.source_id=t.source_id
        WHERE m.assembler_version=? AND e.revision=(SELECT MAX(e2.revision) FROM event_revisions e2
                          WHERE e2.event_id=e.event_id)
        ORDER BY e.table_session_id,e.order_key""", (ASSEMBLER_VERSION,)).fetchall()
    by_session = defaultdict(list)
    source_states = {}
    for event_id, revision, session_id, order_key, event_type, evidence_json, source_state in events:
        evidence = json.loads(evidence_json)
        by_session[session_id].append(OrderedEvent(event_id, revision, session_id,
            order_key, event_type, evidence.get("hand_number_candidate")))
        source_states[session_id] = source_state
    report_hands = []
    journal.begin_batch()
    try:
        for session_id, ordered in by_session.items():
            for window in partition(ordered):
                anchor = next((event.event_id for event in window.events
                               if event.event_type in {"HAND_HEADER", "NEW_HAND"}),
                              window.events[0].event_id)
                hand_id = stable_id("hand", session_id, anchor)
                reasons = ["EVENTS_UNFINALIZED"]
                if window.start_boundary == "OPEN_START":
                    reasons.append("HAND_START_UNOBSERVED")
                if window.end_boundary == "OPEN_END":
                    reasons.append("HAND_END_UNOBSERVED")
                    if source_states[session_id] == "PARTIAL":
                        reasons.append("SOURCE_PARTIAL")
                if len(window.hand_numbers) > 1:
                    reasons.append("HAND_NUMBER_CONFLICT")
                versions = [{"event_id": event.event_id, "revision": event.revision}
                            for event in window.events]
                state = {"reducer_version": REDUCER_VERSION,
                         "start_boundary": window.start_boundary,
                         "end_boundary": window.end_boundary,
                         "hand_number_candidates": window.hand_numbers,
                         "source_state": source_states[session_id],
                         "actions": "NOT_ASSEMBLED",
                         "board": "UNKNOWN", "starting_stacks": "UNKNOWN"}
                validation = {"money": "NOT_RUN", "cards": "NOT_RUN",
                              "sequence": "NOT_RUN", "reason": "EVENTS_UNFINALIZED"}
                record = {"hand_id": hand_id, "table_session_id": session_id,
                          "status": "PARTIAL", "reasons_json": json.dumps(reasons),
                          "event_versions_json": json.dumps(versions),
                          "state_json": json.dumps(state, sort_keys=True),
                          "validation_json": json.dumps(validation, sort_keys=True)}
                _, revision = journal.append_hand(record)
                report_hands.append({"hand_id": hand_id, "revision": revision,
                                     "status": "PARTIAL", "reasons": reasons,
                                     "hand_number_candidates": window.hand_numbers,
                                     "event_versions": versions})
        journal.end_batch()
    except Exception:
        if journal._batched:
            journal.end_batch(commit=False)
        raise
    finally:
        totals = journal.counts()
        journal.close()
    report = {"reducer_version": REDUCER_VERSION, "hand_windows": report_hands,
              "journal_counts": totals, "verified_hands": 0, "hh_exports": 0}
    target = run / "partial_hands.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(report_hands)} PARTIAL hand windows to {target}")


if __name__ == "__main__":
    main()
