from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from parser_core.journal import Journal
from scripts.ocr_chat_pilot import ENGINE, ROW_OCR_RETRY_POLICY, existing_reading


class TinySinglePassTests(unittest.TestCase):
    def test_policy_disables_retries(self):
        self.assertEqual(ROW_OCR_RETRY_POLICY, "NONE_SINGLE_PASS")

    def test_same_tiny_input_is_reused_with_recorded_score(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "journal.sqlite3")
            try:
                journal.add_source("s", "video", "source", 1)
                journal.add_session("t", "s", "layout", {})
                journal.add_frame("s", 0, 0, 1, 1000, 20, 20)
                journal.add_observation("o", "s", 0, "t", "CHAT_ROW",
                                        [0, 0, 20, 16], "row.png", "hash", "RAW")
                preprocessing = {"policy": "single", "source_crop_sha256": "hash"}
                recognition = journal.add_recognition(
                    "o", "player1", ENGINE, "test",
                    {**preprocessing, "raw_score": 0.91}, "RAW_UNREVIEWED")
                self.assertEqual(existing_reading(
                    journal, "o", "test", preprocessing)[:2], recognition)
                self.assertEqual(journal.counts()["recognitions"], 1)
            finally:
                journal.close()


if __name__ == "__main__":
    unittest.main()
