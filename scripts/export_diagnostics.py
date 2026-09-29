"""Export current evidence and OCR status without implying complete HH."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    db = sqlite3.connect(run / "journal.sqlite3")
    target = run / "diagnostics.csv"
    query = """SELECT o.source_id,o.table_session_id,o.frame_id,f.source_pts,
        f.time_base_num,f.time_base_den,o.observation_id,o.roi_type,o.bbox_json,
        o.crop_path,o.crop_sha256,o.status,o.reason,r.recognition_id,r.revision,
        r.raw_text,r.engine,r.model_version,r.raw_score,r.status,r.reason,
        rd.physical_row_id,rd.algorithm_version,rd.decision,rd.reason,
        rd.shift_px,rd.alignment_score,rd.zero_score
        FROM observations o JOIN frames f
          ON f.source_id=o.source_id AND f.frame_id=o.frame_id
        LEFT JOIN recognitions r ON r.observation_id=o.observation_id
          AND r.revision=(SELECT MAX(r2.revision) FROM recognitions r2
                          WHERE r2.observation_id=o.observation_id AND r2.engine=r.engine)
        LEFT JOIN row_decisions rd ON rd.observation_id=o.observation_id
        ORDER BY o.source_id,o.frame_id,o.roi_type"""
    try:
        with target.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("source_id", "table_session_id", "frame_id", "source_pts",
                             "time_base_num", "time_base_den", "observation_id", "roi_type",
                             "bbox_json", "crop_path", "crop_sha256", "observation_status",
                             "observation_reason", "recognition_id", "recognition_revision",
                             "raw_text", "engine", "model_version", "raw_score",
                             "recognition_status", "recognition_reason", "physical_row_id",
                             "row_tracker_version", "row_decision", "row_decision_reason",
                             "alignment_shift_px", "alignment_score", "zero_score"))
            count = 0
            for row in db.execute(query):
                writer.writerow(row)
                count += 1
        event_target = run / "events_diagnostics.csv"
        with event_target.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("source_id", "table_session_id", "event_id", "revision",
                             "message_id", "order_key", "observed_source_pts",
                             "action_time_status", "event_type", "actor_raw", "seat_id",
                             "amount_decimal", "amount_unit", "amount_role", "cards_json",
                             "status", "reason", "normalizer_version", "evidence_json"))
            event_count = 0
            for row in db.execute("""SELECT s.source_id,e.table_session_id,e.event_id,e.revision,
                e.message_id,e.order_key,e.observed_source_pts,e.action_time_status,
                e.event_type,e.actor,e.seat_id,e.amount_decimal,e.amount_unit,
                e.amount_role,e.cards_json,e.status,e.reason,e.normalizer_version,
                e.evidence_json FROM event_revisions e
                JOIN table_sessions t ON t.table_session_id=e.table_session_id
                JOIN sources s ON s.source_id=t.source_id
                WHERE e.revision=(SELECT MAX(e2.revision) FROM event_revisions e2
                                  WHERE e2.event_id=e.event_id)
                ORDER BY e.table_session_id,e.order_key"""):
                writer.writerow(row)
                event_count += 1
        hand_target = run / "hands_diagnostics.csv"
        with hand_target.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("hand_id", "revision", "table_session_id", "status",
                             "reasons_json", "event_versions_json", "state_json",
                             "validation_json"))
            hand_count = 0
            for row in db.execute("""SELECT h.* FROM hand_revisions h
                WHERE h.revision=(SELECT MAX(h2.revision) FROM hand_revisions h2
                                  WHERE h2.hand_id=h.hand_id)
                ORDER BY h.table_session_id,h.hand_id"""):
                writer.writerow(row)
                hand_count += 1
        fragment_target = run / "event_fragments_diagnostics.csv"
        with fragment_target.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("source_id", "table_session_id", "event_id", "revision",
                             "message_id", "assembler_version", "event_type",
                             "actor_raw", "seat_id", "amount_decimal", "amount_unit",
                             "amount_role", "event_status", "event_reason",
                             "message_completeness", "message_reason", "fragment_ordinal",
                             "physical_row_id", "frame_id", "source_pts",
                             "time_base_num", "time_base_den", "observation_id",
                             "crop_path", "crop_sha256", "recognition_id", "raw_text",
                             "engine", "model_version", "raw_score",
                             "recognition_status", "recognition_reason"))
            fragment_count = 0
            for row in db.execute("""SELECT s.source_id,e.table_session_id,e.event_id,
                e.revision,e.message_id,m.assembler_version,e.event_type,e.actor,
                e.seat_id,e.amount_decimal,e.amount_unit,e.amount_role,e.status,
                e.reason,m.completeness,m.reason,mf.ordinal,
                p.physical_row_id,o.frame_id,f.source_pts,f.time_base_num,
                f.time_base_den,o.observation_id,o.crop_path,o.crop_sha256,
                r.recognition_id,r.raw_text,r.engine,r.model_version,r.raw_score,
                r.status,r.reason
                FROM event_revisions e
                JOIN messages m ON m.message_id=e.message_id
                JOIN message_fragments mf ON mf.message_id=m.message_id
                JOIN physical_rows p ON p.physical_row_id=mf.physical_row_id
                JOIN row_observations ro ON ro.physical_row_id=p.physical_row_id
                JOIN observations o ON o.observation_id=ro.observation_id
                JOIN frames f ON f.source_id=o.source_id AND f.frame_id=o.frame_id
                JOIN table_sessions t ON t.table_session_id=e.table_session_id
                JOIN sources s ON s.source_id=t.source_id
                LEFT JOIN recognitions r ON r.recognition_id=mf.recognition_id
                WHERE e.revision=(SELECT MAX(e2.revision) FROM event_revisions e2
                                  WHERE e2.event_id=e.event_id)
                  AND ro.ordinal=(SELECT MIN(ro2.ordinal) FROM row_observations ro2
                                  WHERE ro2.physical_row_id=p.physical_row_id)
                ORDER BY e.table_session_id,e.order_key,mf.ordinal"""):
                writer.writerow(row)
                fragment_count += 1
    finally:
        db.close()
    print(f"Wrote {count} observations, {event_count} events, {hand_count} hands, "
          f"{fragment_count} event fragments to {run}")


if __name__ == "__main__":
    main()
