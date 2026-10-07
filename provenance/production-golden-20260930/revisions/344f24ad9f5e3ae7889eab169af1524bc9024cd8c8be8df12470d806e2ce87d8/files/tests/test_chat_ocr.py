from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from parser_core.chat_ocr import (board_tiles_visible, dealer_nickname,
                                  latinize_mixed_nickname,
                                  nickname_route_reasons,
                                  prepare_row_png, substitute_nickname,
                                  uncertain_glyph_conflict)
from parser_core.chat_rows import ROW_TRACKER_VERSION
from parser_core.journal import Journal
from parser_core.message_assembly import ASSEMBLER_VERSION
from scripts import assemble_messages


class ChatOcrTests(unittest.TestCase):
    def test_trim_removes_only_right_border_and_preserves_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "row.png"
            image = Image.new("RGB", (228, 16), "black")
            for x in range(216, 228):
                for y in range(16):
                    image.putpixel((x, y), (255, 255, 255))
            image.save(path)
            original = path.read_bytes()
            cropped, digest = prepare_row_png(path, hashlib.sha256(original).hexdigest())
            with Image.open(io.BytesIO(cropped)) as result:
                self.assertEqual(result.size, (432, 32))
                self.assertEqual(result.getpixel((431, 0)), (0, 0, 0))
            self.assertEqual(hashlib.sha256(cropped).hexdigest(), digest)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(ValueError):
                prepare_row_png(path, "wrong")

    def test_route_uses_only_player_name_and_substitution_preserves_amount(self):
        self.assertEqual(nickname_route_reasons("Дилер: nasguilS сбрасывает"),
                         ("LATIN_I_OR_L_IN_NICKNAME",))
        self.assertEqual(nickname_route_reasons("Дилер: деп1из9 сбрасывает"),
                         ("CYRILLIC_IN_NICKNAME",))
        self.assertEqual(nickname_route_reasons("Дилер: Раздаются карты"), ())
        self.assertEqual(nickname_route_reasons("Дилер: egorkunov выигрывает"), ())
        self.assertEqual(nickname_route_reasons("Дилер: 77kayf?? сбрасывает"),
                         ("UNCERTAIN_GLYPH_IN_NICKNAME",))
        self.assertEqual(nickname_route_reasons("Дилер: papa_moget сбрасывает"), ())
        self.assertEqual(substitute_nickname(
            "Дилер: деп1из9 повышает до 2 000",
            "Дилер: gen1usd повышает до 2000"),
            "Дилер: gen1usd повышает до 2 000")
        self.assertIsNone(substitute_nickname(
            "Дилер: деп1из9 сбрасывает", "Дилер: gen1usd повышает до 2 000"))
        self.assertIsNone(substitute_nickname(
            "Дилер: деп1из9 сбрасывает", "Дилер: ген1usd сбрасывает"))
        self.assertEqual(substitute_nickname(
            "Дилер: аие163затaга сбрасывает",
            "Дилер: аue163samara сбрасывает"),
            "Дилер: aue163samara сбрасывает")
        self.assertIsNone(latinize_mixed_nickname("алексa"))
        self.assertIsNone(latinize_mixed_nickname("аue163samara-ж"))
        self.assertIsNone(substitute_nickname(
            "Дилер: деп1из9 не показывает", "Дилер: gen1usd показывает"))
        self.assertIsNone(substitute_nickname(
            "Дилер: 77kayf?? сбрасывает", "Дилер: 77kayf?7 сбрасывает"))
        self.assertTrue(uncertain_glyph_conflict(
            "Дилер: 77kayf?? сбрасывает", "Дилер: 77kavf77 сбрасывает"))
        self.assertFalse(uncertain_glyph_conflict(
            "Дилер: gambit?3 повышает", "Дилер: gambit73 повышает"))
        self.assertFalse(uncertain_glyph_conflict(
            "Дилер: livsill? делает чек", "Дилер: livsi117 делает чек"))

    def test_broken_actor_spacing_routes_and_replaces_whole_span(self):
        self.assertEqual(dealer_nickname("Дилер: On ingrinder сбрасывает")[0],
                         "On ingrinder")
        self.assertEqual(nickname_route_reasons("Дилер: On ingrinder сбрасывает"),
                         ("LATIN_I_OR_L_IN_NICKNAME",))
        self.assertEqual(substitute_nickname(
            "Дилер: On ingrinder сбрасывает",
            "Дилер: rOn1ngrinder сбрасывает"),
            "Дилер: rOn1ngrinder сбрасывает")
        self.assertEqual(substitute_nickname(
            "Дилер: Кош! ИК ставит большой",
            "Дилер: kourhik ставит большой"),
            "Дилер: kourhik ставит большой")

    def test_board_detector_requires_card_strip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "row.png"
            image = Image.new("RGB", (228, 16), "black")
            for x in range(20, 80):
                for y in range(3, 13):
                    image.putpixel((x, y), (255, 255, 255))
            image.save(path)
            self.assertTrue(board_tiles_visible(path, hashlib.sha256(path.read_bytes()).hexdigest()))
            image = Image.new("RGB", (228, 16), "black")
            image.save(path)
            self.assertFalse(board_tiles_visible(path, hashlib.sha256(path.read_bytes()).hexdigest()))

    def test_nine_sessions_select_only_tiny_evidence_and_keep_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            journal = Journal(run / "journal.sqlite3")
            try:
                journal.add_source("source", "video", "hash", 1)
                journal.add_frame("source", 0, 0, 1, 1000, 228, 16)
                for slot in range(1, 10):
                    session = f"slot{slot}"
                    journal.add_session(session, "source", "layout", {"slot": slot})
                    image = Image.new("RGB", (228, 16), "black")
                    relative = f"crop{slot}.png"
                    image.save(run / relative)
                    digest = hashlib.sha256((run / relative).read_bytes()).hexdigest()
                    observation = f"observation{slot}"
                    row = f"row{slot}"
                    journal.add_observation(observation, "source", 0, session,
                                            "CHAT_ROW", [0, 0, 228, 16], relative,
                                            digest, "RAW")
                    journal.add_physical_row(row, session,
                        f"{ROW_TRACKER_VERSION}:epoch_{slot}", 0, "TRACKED")
                    journal.link_row_observation(row, observation, 0)
                    journal.add_recognition(observation, f"tiny-row-{slot}",
                        "PP-OCRv6_tiny_rec", "test", {"raw_score": 0.9},
                        "RAW_UNREVIEWED")
                    journal.add_recognition(observation, f"old-tesseract-{slot}",
                        "Tesseract", "old", {}, "RAW_UNREVIEWED")
            finally:
                journal.close()
            with patch.object(sys, "argv", ["assemble_messages.py", str(run)]):
                assemble_messages.main()
            report = json.loads((run / "message_candidates.json").read_text("utf-8"))
            self.assertEqual(report["assembler_version"], ASSEMBLER_VERSION)
            self.assertEqual(len(report["messages"]), 9)
            by_row = {message["row_ids"][0]: message for message in report["messages"]}
            self.assertEqual(by_row["row1"]["raw_text"], ["tiny-row-1"])
            self.assertEqual(by_row["row2"]["raw_text"], ["tiny-row-2"])
            self.assertEqual(by_row["row3"]["raw_text"], ["tiny-row-3"])


if __name__ == "__main__":
    unittest.main()
