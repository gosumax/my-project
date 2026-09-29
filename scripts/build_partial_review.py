"""Render every PARTIAL hand window with raw OCR and linked row evidence."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    report = json.loads((run / "partial_hands.json").read_text(encoding="utf-8"))
    with sqlite3.connect(run / "journal.sqlite3") as db:
        lines = ["# Все окна PARTIAL", "",
                 "Только диагностические окна из текущего журнала. Raw OCR не является проверенной истиной.", "",
                 "| № | Стол | Кандидат номера | Событий | Причины |",
                 "|---:|---:|---|---:|---|"]
        details = []
        for index, window in enumerate(report["hand_windows"], 1):
            hand_id = window["hand_id"]
            row = db.execute("""SELECT h.table_session_id,s.geometry_json
                FROM hand_revisions h JOIN table_sessions s USING(table_session_id)
                WHERE h.hand_id=? AND h.revision=?""",
                (hand_id, window["revision"])).fetchone()
            if row is None:
                raise ValueError(f"Missing hand revision: {hand_id}")
            slot = json.loads(row[1])["slot_id"]
            number = ", ".join(window["hand_number_candidates"]) or "UNKNOWN"
            reasons = ", ".join(window["reasons"])
            lines.append(f"| {index} | {slot:02d} | {number} | "
                         f"{len(window['event_versions'])} | {reasons} |")
            details.extend(["", f"## {index}. Стол {slot:02d} — {hand_id}", "",
                            f"Номер руки: `{number}` (кандидат OCR). Статус: `PARTIAL`. "
                            f"Причины: `{reasons}`.", ""])
            for event_number, version in enumerate(window["event_versions"], 1):
                event = db.execute("""SELECT event_type,message_id,status,reason,
                    observed_source_pts,action_time_status FROM event_revisions
                    WHERE event_id=? AND revision=?""",
                    (version["event_id"], version["revision"])).fetchone()
                if event is None:
                    raise ValueError(f"Missing event revision: {version}")
                event_type, message_id, status, reason, pts, action_time = event
                details.append(f"{event_number}. `{event_type}` — `{status}`; "
                               f"PTS наблюдения `{pts}`; время действия `{action_time}`; "
                               f"причина `{reason}`.")
                fragments = db.execute("""SELECT r.raw_text,o.crop_path
                    FROM message_fragments mf
                    LEFT JOIN recognitions r ON r.recognition_id=mf.recognition_id
                    LEFT JOIN observations o ON o.observation_id=r.observation_id
                    WHERE mf.message_id=? ORDER BY mf.ordinal""", (message_id,)).fetchall()
                for raw, relative in fragments:
                    raw_one_line = (raw or "UNKNOWN").replace("\r", " ").replace("\n", " ")
                    evidence = f"[PNG](<{(run / relative).resolve().as_posix()}>)" if relative else "PNG: UNKNOWN"
                    details.append(f"   - raw OCR: `{raw_one_line}`; {evidence}")
            details.append("")
    target = run / "partial_windows_review.md"
    target.write_text("\n".join(lines + details) + "\n", encoding="utf-8")
    print(f"Wrote {len(report['hand_windows'])} PARTIAL windows to {target}")


if __name__ == "__main__":
    main()
