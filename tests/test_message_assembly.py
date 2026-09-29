from __future__ import annotations

import unittest

from parser_core.message_assembly import RowFact, group_rows, visible_start


class MessageAssemblyTests(unittest.TestCase):
    def test_wrap_and_new_hand_header_remain_distinct(self):
        def row(n, text, margin):
            return RowFact(str(n), "table", "epoch", 0, n * 30, text,
                           f"rec{n}", margin)
        rows = [row(0, "[1:44] Игрок выигрывает из главного", 5),
                row(1, "банка", 300),
                row(2, "Рука #123", 200),
                row(3, "[1:44] Начинается новая рука", 100),
                row(4, "#123", 200)]
        messages = group_rows(rows)
        self.assertEqual([[r.row_id for r in m.rows] for m in messages],
                         [["0", "1"], ["2"], ["3", "4"]])
        self.assertEqual(messages[0].completeness, "PARTIAL")
        self.assertEqual(messages[1].completeness, "CANDIDATE")

    def test_new_hand_number_wrap_is_contextual_and_stays_partial(self):
        def row(n, text, epoch="epoch", session="table"):
            return RowFact(str(n), session, epoch, 0, n * 20, text, f"r{n}", 200)
        for prefix in ["Дилер: Начинается новая рука", " дилер : НАЧИНАЕТСЯ новая РУКА. ",
                       "[1:44] Начинается новая рука"]:
            messages = group_rows([row(0,prefix), row(1," # 12345 "),
                                   row(2,"Дилер: player ставит анте 10")])
            self.assertEqual([[r.row_id for r in m.rows] for m in messages], [["0","1"],["2"]])
            self.assertEqual(messages[0].completeness, "PARTIAL")
            self.assertIn("CONTINUATION_WITHOUT_VISIBLE_START", messages[0].reason)
        for second in [row(1,"#123",epoch="other"), row(1,"#123",session="other"),
                       row(1,"Рука #123"), row(1,"#123 extra")]:
            messages = group_rows([row(0,"Дилер: Начинается новая рука"),second])
            # Malformed #123 extra retains existing standalone-start behavior.
            self.assertEqual(len(messages), 2)
        messages=group_rows([row(0,"Дилер: player сбрасывает"),row(1,"#123")])
        self.assertEqual(len(messages),2)
        self.assertTrue(visible_start("#123"))

    def test_missing_prefix_stays_unresolved(self):
        rows = [RowFact("a", "table", "epoch", 0, 0, "some text", "r", 250),
                RowFact("b", "table", "epoch", 0, 30, "other", "r2", 250)]
        messages = group_rows(rows)
        self.assertEqual(len(messages), 1)
        self.assertEqual([row.row_id for row in messages[0].rows], ["a", "b"])
        self.assertTrue(all(m.completeness == "PARTIAL" for m in messages))

    def test_unresolved_row_identity_blocks_candidate(self):
        rows = [RowFact("a", "table", "epoch", 0, 0, "[1:44] text", "r", 250,
                        "UNRESOLVED_IDENTITY"),
                RowFact("b", "table", "epoch", 1, 0, "[1:45] next", "r2", 250)]
        messages = group_rows(rows)
        self.assertEqual(messages[0].completeness, "PARTIAL")
        self.assertIn("ROW_IDENTITY_UNRESOLVED", messages[0].reason)

    def test_dealer_actions_start_separate_messages_but_amount_wraps(self):
        def row(n, text, margin):
            return RowFact(str(n), "table", "epoch", 1425, n * 16, text,
                           f"rec{n}", margin)

        rows = [row(0, "Дилер: Раздаются карты.", 5),
                row(1, "Дилер: first сбрасывает", 5),
                row(2, "Дилер: second сбрасывает", 5),
                row(3, "Дилер: third повышает до", 200),
                row(4, "6 000", 200),
                row(5, "Дилер: fourth сбрасывает", 200)]
        messages = group_rows(rows)

        self.assertEqual([[r.row_id for r in m.rows] for m in messages],
                         [["0"], ["1"], ["2"], ["3", "4"], ["5"]])
        self.assertTrue(all(m.start_status == "VISIBLE_START" for m in messages))
        self.assertTrue(messages[3].inferred_continuation)
        self.assertEqual(messages[3].completeness, "PARTIAL")
        self.assertTrue(visible_start("  Дилер : fifth проверяет"))
        self.assertFalse(visible_start("third повышает до Дилер: 6 000"))

    def test_continuation_does_not_cross_chat_epoch(self):
        rows = [RowFact("a", "table", "epoch_1", 0, 0,
                        "Дилер: player повышает до", "r1", 200),
                RowFact("b", "table", "epoch_2", 1, 0,
                        "6 000", "r2", 200)]
        messages = group_rows(rows)
        self.assertEqual([[row.row_id for row in message.rows] for message in messages],
                         [["a"], ["b"]])
        self.assertEqual(messages[0].end_status, "NEXT_ROW_UNRESOLVED")
        self.assertEqual(messages[1].start_status, "UNRESOLVED_START")

    def test_missing_ocr_row_remains_a_separate_unknown_message(self):
        rows = [RowFact("raise", "table", "epoch", 0, 0,
                        "Дилер: player повышает до", "r1", 10),
                RowFact("amount", "table", "epoch", 0, 16,
                        "300", "r2", 200),
                RowFact("unreadable", "table", "epoch", 0, 32,
                        "", "r3", 10),
                RowFact("whitespace", "table", "epoch", 0, 40,
                        "  ", "r5", 10),
                RowFact("fold", "table", "epoch", 0, 48,
                        "Дилер: other сбрасывает", "r4", 200)]
        messages = group_rows(rows)
        self.assertEqual([[row.row_id for row in message.rows] for message in messages],
                         [["raise", "amount"], ["unreadable"], ["whitespace"], ["fold"]])
        self.assertEqual(messages[1].completeness, "PARTIAL")
        self.assertIn("OCR_MISSING", messages[1].reason)
        self.assertIn("OCR_MISSING", messages[2].reason)

    def test_retry_reading_keeps_message_partial(self):
        rows = [RowFact("retry", "table", "epoch", 0, 0,
                        "Дилер: player сбрасывает", "retry_rec", 200,
                        ocr_retry=True),
                RowFact("next", "table", "epoch", 0, 16,
                        "Дилер: other проверяет", "primary_rec", 200)]
        messages = group_rows(rows)
        self.assertEqual(messages[0].completeness, "PARTIAL")
        self.assertIn("OCR_RETRY_UNVERIFIED", messages[0].reason)


if __name__ == "__main__":
    unittest.main()
