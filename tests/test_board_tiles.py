from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from parser_core.board_tiles import read_board_tiles


FIXTURES = Path(__file__).parent / "fixtures" / "board_tiles"


class BoardTilesTests(unittest.TestCase):
    def test_malformed_vl_suits_are_checked_against_tile_glyphs(self):
        examples = (
            ("bad_four.png", "3+9+3♥K♥", "3 ♦ 9 ♣ 3 ♥ K ♥"),
            ("bad_five.png", "5+10❤️Q❤️A+7❤️", "5 ♣ 10 ♥ Q ♥ A ♣ 7 ♥"),
            ("bad_four_five.png", "5÷10♥Q♥A♣", "5 ♣ 10 ♥ Q ♥ A ♣"),
            ("bad_three.png", "9 ♦ ♀ ♥ 9 ♣", "9 ♦ Q ♥ 9 ♣"),
        )
        for filename, vl_raw, expected in examples:
            with self.subTest(filename=filename):
                path = FIXTURES / filename
                result = read_board_tiles(
                    path, hashlib.sha256(path.read_bytes()).hexdigest(), vl_raw)
                self.assertIsNotNone(result)
                self.assertEqual(result.text, expected)

    def test_well_formed_rows_stay_identical_and_unreadable_rows_abstain(self):
        path = FIXTURES / "good_five.png"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        raw = "8 ♥ 9 ♣ 10 ♣ J ♠ Q ♣"
        self.assertEqual(read_board_tiles(path, digest, raw).text, raw)
        self.assertIsNone(read_board_tiles(path, digest, "8 ♥ 9 ♣"))
        with self.assertRaises(ValueError):
            read_board_tiles(path, "incorrect hash", raw)


if __name__ == "__main__":
    unittest.main()
