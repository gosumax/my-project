"""Store non-final event candidates from versioned messages and raw OCR."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.journal import Journal, stable_id
from parser_core.message_assembly import ASSEMBLER_VERSION
from parser_core.normalize_chat import NORMALIZER_VERSION, normalize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    journal = Journal(run / "journal.sqlite3")
    query = """SELECT m.message_id,m.table_session_id,m.completeness,m.reason,
        mf.ordinal,p.physical_row_id,r.recognition_id,r.raw_text,
        o.frame_id,f.source_pts,o.bbox_json,r.preprocessing_json
        FROM messages m JOIN message_fragments mf ON mf.message_id=m.message_id
        JOIN physical_rows p ON p.physical_row_id=mf.physical_row_id
        JOIN row_observations ro ON ro.physical_row_id=p.physical_row_id
        JOIN observations o ON o.observation_id=ro.observation_id
        JOIN frames f ON f.source_id=o.source_id AND f.frame_id=o.frame_id
        LEFT JOIN recognitions r ON r.recognition_id=mf.recognition_id
        WHERE m.assembler_version=?
          AND ro.ordinal=(SELECT MIN(ro2.ordinal) FROM row_observations ro2
                          WHERE ro2.physical_row_id=p.physical_row_id)
        ORDER BY m.message_id,mf.ordinal"""
    by_message = defaultdict(list)
    for row in journal.db.execute(query, (ASSEMBLER_VERSION,)):
        by_message[row[0]].append(row)
    report_events = []
    counts: Counter[str] = Counter()
    journal.begin_batch()
    try:
        for message_id, fragments in by_message.items():
            first = fragments[0]
            candidate = normalize([part[7] for part in fragments])
            reasons = ["OCR_UNREVIEWED"]
            if first[2] != "CANDIDATE":
                reasons.append("MESSAGE_PARTIAL")
            if "OCR_RETRY_UNVERIFIED" in (first[3] or "").split(";"):
                reasons.append("OCR_RETRY_UNVERIFIED")
            if candidate.reason:
                reasons.append(candidate.reason)
            if candidate.actor is not None:
                reasons.append("ACTOR_SEAT_UNVERIFIED")
            glyph_conflict = any(json.loads(part[11] or "{}").get(
                "known_glyph_conflict", False) for part in fragments)
            if glyph_conflict:
                reasons.append("ACTOR_OCR_CONFLICT")
            status = ("CANDIDATE" if first[2] == "CANDIDATE" and
                      candidate.event_type != "UNKNOWN_EVENT" and
                      candidate.reason is None and not glyph_conflict else "UNRESOLVED")
            event_id = stable_id("event", message_id, "primary")
            bbox = json.loads(first[10])
            evidence = {"row_ids": [part[5] for part in fragments],
                        "recognition_ids": [part[6] for part in fragments],
                        "message_reason": first[3],
                        "hand_number_candidate": candidate.hand_number,
                        "source_pts_role": "OBSERVATION_ONLY",
                        "amount_unit_status": "UNKNOWN" if candidate.amount_decimal else None}
            record = {"event_id": event_id, "table_session_id": first[1],
                      "message_id": message_id, "observation_id": None,
                      "event_type": candidate.event_type, "actor": candidate.actor,
                      "seat_id": None, "amount_decimal": candidate.amount_decimal,
                      "amount_unit": "UNKNOWN" if candidate.amount_decimal else None,
                      "amount_role": candidate.amount_role, "cards_json": "[]",
                      "order_key": f"{first[8]:012d}:{bbox[1]:06d}:{message_id}",
                      "observed_source_pts": first[9], "action_time_status": "UNKNOWN",
                      "status": status, "reason": ";".join(reasons),
                      "normalizer_version": NORMALIZER_VERSION,
                      "evidence_json": json.dumps(evidence, ensure_ascii=False, sort_keys=True)}
            _, revision = journal.append_event(record)
            counts[status] += 1
            counts[candidate.event_type] += 1
            report_events.append({"event_id": event_id, "revision": revision,
                                  "message_id": message_id, "type": candidate.event_type,
                                  "order_key": record["order_key"],
                                  "observed_source_pts": first[9],
                                  "actor_raw": candidate.actor,
                                  "amount_decimal": candidate.amount_decimal,
                                  "amount_role": candidate.amount_role,
                                  "status": status, "reason": record["reason"]})
        journal.end_batch()
    except Exception:
        if journal._batched:
            journal.end_batch(commit=False)
        raise
    finally:
        totals = journal.counts()
        journal.close()
    report = {"normalizer_version": NORMALIZER_VERSION, "counts": dict(counts),
              "journal_counts": totals,
              "events": sorted(report_events, key=lambda event: event["order_key"]),
              "publication_status": "NONE"}
    target = run / "event_candidates.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(report_events)} event candidates to {target}")


if __name__ == "__main__":
    main()
