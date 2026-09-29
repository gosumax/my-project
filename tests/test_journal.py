from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from parser_core.journal import Journal, stable_id


class JournalTests(unittest.TestCase):
    def test_replay_is_idempotent_and_conflicting_evidence_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "test.sqlite3")
            source = stable_id("source", "same-bytes")
            session = stable_id("table", source, "layout", [0, 0, 100, 100], [1, 1, 20, 20])
            obs = stable_id("obs", source, 0, session, "chat", [1, 1, 20, 20])
            for _ in range(2):
                journal.add_source(source, "video.mkv", "same-bytes", 123)
                journal.add_session(session, source, "layout", {"chat_roi": [1, 1, 20, 20]})
                journal.add_frame(source, 0, 42, 1, 1000, 1920, 1080)
                journal.add_observation(obs, source, 0, session, "chat", [1, 1, 20, 20],
                                        "crops/chat.png", "crop-hash", "RAW")
            self.assertEqual(journal.counts()["observations"], 1)
            with self.assertRaisesRegex(ValueError, "Frame identity conflict"):
                journal.add_frame(source, 0, 43, 1, 1000, 1920, 1080)
            with self.assertRaisesRegex(ValueError, "Observation identity conflict"):
                journal.add_observation(obs, source, 0, session, "chat", [1, 1, 20, 20],
                                        "crops/chat.png", "different-hash", "RAW")
            first = journal.add_recognition(obs, "bet 35", "Tesseract", "5.5", {"psm": 6},
                                            "RAW_UNREVIEWED")
            self.assertEqual(first, journal.add_recognition(obs, "bet 35", "Tesseract",
                                                             "5.5", {"psm": 6}, "RAW_UNREVIEWED"))
            second = journal.add_recognition(obs, "bet 36", "Tesseract", "5.5", {"psm": 6},
                                             "RAW_UNREVIEWED")
            self.assertEqual((first[1], second[1]), (1, 2))
            journal.add_physical_row("row", session, "row_track_v2:epoch", 42,
                                     "INITIAL_VISIBLE", "FIRST_OBSERVED_FRAME")
            journal.link_row_observation("row", obs, 0)
            journal.add_row_decision("row", obs, "row_track_v2", "INITIAL_VISIBLE",
                                     "FIRST_OBSERVED_FRAME", None, None, None)
            for _ in range(2):
                journal.add_row_suppression(source, 0, session, (0, 5),
                                            "row_track_v9", "FIRST_VISIBLE_ROW_EXCLUDED")
            self.assertEqual(journal.counts()["row_suppressions"], 1)
            with self.assertRaisesRegex(ValueError, "Row suppression conflict"):
                journal.add_row_suppression(source, 0, session, (0, 5),
                                            "row_track_v9", "DIFFERENT_REASON")
            with self.assertRaisesRegex(ValueError, "Row decision conflict"):
                journal.add_row_decision("row", obs, "row_track_v2", "TRACKED",
                                         None, 0, 0.0, 0.0)
            journal.set_source_state(source, "EOF")
            journal.set_source_state(source, "PARTIAL")
            state = journal.db.execute("SELECT state FROM sources WHERE source_id=?", (source,)).fetchone()[0]
            self.assertEqual(state, "EOF")
            journal.close()

    def test_pts_requires_time_base_and_evidence_foreign_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "test.sqlite3")
            try:
                journal.add_source("s", "video.mkv", "hash", 1)
                with self.assertRaises(Exception):
                    journal.add_frame("s", 0, 42, None, None, 10, 10)
                with self.assertRaises(Exception):
                    journal.add_observation("o", "s", 0, "missing-table", "chat",
                                            [0, 0, 2, 2], "missing.png", "hash", "RAW")
            finally:
                journal.close()

    def test_batch_rollback_does_not_publish_partial_row(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "test.sqlite3")
            journal.add_source("s", "video.mkv", "hash", 1)
            journal.add_session("t", "s", "layout", {})
            journal.begin_batch()
            journal.add_physical_row("r", "t", "epoch", 10, "INITIAL_VISIBLE")
            journal.end_batch(commit=False)
            self.assertEqual(journal.counts()["physical_rows"], 0)
            journal.close()

    def test_interpretation_changes_append_revisions(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "test.sqlite3")
            journal.add_source("s", "video.mkv", "hash", 1)
            journal.add_session("t", "s", "layout", {})
            journal.add_frame("s", 0, 0, 1, 1000, 10, 10)
            journal.add_observation("o", "s", 0, "t", "CHAT_ROW",
                                    [0, 0, 10, 5], "row.png", "crop", "RAW")
            event = {"event_id": "e", "table_session_id": "t", "message_id": None,
                     "observation_id": "o", "event_type": "BET", "actor": "raw",
                     "seat_id": None, "amount_decimal": "35", "amount_unit": "UNKNOWN",
                     "amount_role": "BET", "cards_json": "[]", "order_key": "0000",
                     "observed_source_pts": 0, "action_time_status": "UNKNOWN",
                     "status": "UNRESOLVED", "reason": "OCR_UNREVIEWED",
                     "normalizer_version": "v1", "evidence_json": "{}"}
            self.assertEqual(journal.append_event(event), ("e", 1))
            self.assertEqual(journal.append_event(event), ("e", 1))
            changed = dict(event, event_type="RAISE", amount_role="RAISE_TO",
                           normalizer_version="v2")
            self.assertEqual(journal.append_event(changed), ("e", 2))
            hand = {"hand_id": "h", "table_session_id": "t", "status": "PARTIAL",
                    "reasons_json": '["EVENTS_UNFINALIZED"]',
                    "event_versions_json": '[{"event_id":"e","revision":2}]',
                    "state_json": "{}", "validation_json": "{}"}
            self.assertEqual(journal.append_hand(hand), ("h", 1))
            self.assertEqual(journal.append_hand(hand), ("h", 1))
            self.assertEqual(journal.counts()["event_revisions"], 2)
            journal.close()


if __name__ == "__main__":
    unittest.main()
