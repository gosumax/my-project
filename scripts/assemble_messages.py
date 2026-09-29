"""Build versioned candidate messages from tracked rows and raw OCR."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.chat_rows import ROW_TRACKER_VERSION
from parser_core.journal import Journal, stable_id
from parser_core.message_assembly import ASSEMBLER_VERSION, RowFact, group_rows


def right_margin(path: Path, digest: str) -> int | None:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Row crop hash mismatch: {path}")
    gray = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Cannot decode row crop: {path}")
    usable_width = gray.shape[1] - 18 if gray.shape[1] > 50 else gray.shape[1]
    xs = np.nonzero(gray[:, :usable_width] > 100)[1]
    return None if len(xs) == 0 else usable_width - 1 - int(xs.max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    journal = Journal(run / "journal.sqlite3")
    query = """SELECT p.physical_row_id,p.table_session_id,p.chat_epoch,p.state,
        o.frame_id,o.bbox_json,o.crop_path,o.crop_sha256,r.recognition_id,r.raw_text,
        r.preprocessing_json,r.engine
        FROM physical_rows p
        JOIN row_observations ro ON ro.physical_row_id=p.physical_row_id
        JOIN observations o ON o.observation_id=ro.observation_id
        LEFT JOIN recognitions r ON r.recognition_id=(
            SELECT t.recognition_id FROM recognitions t
            WHERE t.observation_id=o.observation_id AND t.engine='PP-OCRv6_tiny_rec'
            ORDER BY t.revision DESC LIMIT 1)
        WHERE p.chat_epoch LIKE ? AND ro.ordinal=(SELECT MIN(ro2.ordinal) FROM row_observations ro2
                          WHERE ro2.physical_row_id=p.physical_row_id)"""
    by_session: dict[str, list[RowFact]] = defaultdict(list)
    for row_id, session_id, epoch, origin_state, frame_id, bbox_json, relative, digest, rec_id, raw_text, preprocessing_json, engine in journal.db.execute(query, (ROW_TRACKER_VERSION + ":%",)):
        path = (run / relative).resolve(strict=True)
        if not path.is_relative_to(run):
            raise ValueError(f"Row crop outside run: {path}")
        y1 = json.loads(bbox_json)[1]
        by_session[session_id].append(RowFact(row_id, session_id, epoch, frame_id,
                                              y1, raw_text, rec_id,
                                              right_margin(path, digest), origin_state,
                                              False, False, False))
    report_messages = []
    status_counts: Counter[str] = Counter()
    journal.begin_batch()
    try:
        for session_id, rows in by_session.items():
            rows.sort(key=lambda row: (row.frame_id, row.y1, row.row_id))
            for candidate in group_rows(rows):
                message_id = stable_id("message", ASSEMBLER_VERSION,
                                       session_id, candidate.rows[0].row_id)
                journal.add_message(message_id, session_id, candidate.completeness,
                                    candidate.start_status, candidate.end_status,
                                    candidate.reason, ASSEMBLER_VERSION)
                for ordinal, row in enumerate(candidate.rows):
                    journal.add_message_fragment(message_id, ordinal,
                                                 row.row_id, row.recognition_id)
                status_counts[candidate.completeness] += 1
                report_messages.append({"message_id": message_id,
                                        "status": candidate.completeness,
                                        "reason": candidate.reason,
                                        "row_ids": [row.row_id for row in candidate.rows],
                                        "raw_text": [row.text for row in candidate.rows]})
        journal.end_batch()
    except Exception:
        if journal._batched:
            journal.end_batch(commit=False)
        raise
    finally:
        totals = journal.counts()
        journal.close()
    report = {"assembler_version": ASSEMBLER_VERSION, "counts": dict(status_counts),
              "journal_counts": totals, "messages": report_messages,
              "interpretation": "CANDIDATES_NOT_ACCEPTED_EVENTS"}
    target = run / "message_candidates.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(report_messages)} candidate messages to {target}")


if __name__ == "__main__":
    main()
