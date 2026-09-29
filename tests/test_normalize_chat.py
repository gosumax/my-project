from __future__ import annotations

import unittest

from parser_core.normalize_chat import normalize


class NormalizeChatTests(unittest.TestCase):
    def test_raise_to_keeps_total_role(self):
        result = normalize(["[1:44] @ buddist1 повышает до 600"])
        self.assertEqual((result.event_type, result.actor, result.amount_decimal,
                          result.amount_role), ("RAISE", "buddist1", "600", "RAISE_TO"))

    def test_wrapped_blind_and_unknown_amount(self):
        result = normalize(["[1:44] aleshka ставит малый |", "блайнд 150"])
        self.assertEqual((result.event_type, result.amount_decimal), ("SMALL_BLIND", "150"))
        ambiguous = normalize(["[1:44] player уравнивает ???"])
        self.assertEqual(ambiguous.reason, "AMOUNT_AMBIGUOUS_OR_MISSING")

    def test_hand_number_is_not_a_contribution(self):
        result = normalize(["Рука #2837663690"])
        self.assertEqual(result.event_type, "HAND_HEADER")
        self.assertIsNone(result.amount_decimal)

    def test_new_hand_wrap_preserves_number_without_monetary_value(self):
        for fragments in [["Дилер: Начинается новая рука", "#12345"],
                          ["[1:44] Начинается новая рука.", "# 12345 |"],
                          ["Дилер: Начинается новая рука #12345"]]:
            result = normalize(fragments)
            self.assertEqual((result.event_type, result.hand_number), ("NEW_HAND", "12345"))
            self.assertIsNone(result.actor)
            self.assertIsNone(result.amount_decimal)
            self.assertIsNone(result.amount_role)
        for tail in ["", "#12 #34", "#12 extra", "125"]:
            result=normalize(["Дилер: Начинается новая рука",tail])
            self.assertEqual(result.event_type,"NEW_HAND")
            self.assertIsNone(result.hand_number)
            self.assertIsNone(result.amount_decimal)
        self.assertEqual(normalize(["#12345"]).event_type,"HAND_ID")

    def test_unreadable_actor_is_not_candidate(self):
        result = normalize(["[1:44] @ ��� ставит анте 35 |"])
        self.assertEqual(result.reason, "ACTOR_OCR_MISSING")

    def test_uncertain_nickname_is_unresolved_until_ocr_resolves_it(self):
        uncertain = normalize(["Дилер: 77kayf?? сбрасывает"])
        self.assertEqual((uncertain.actor, uncertain.reason),
                         ("77kayf??", "ACTOR_OCR_AMBIGUOUS"))
        clean = normalize(["Дилер: 77kayf77 сбрасывает"])
        self.assertEqual((clean.actor, clean.reason), ("77kayf77", None))

    def test_blind_marker_with_ocr_punctuation_and_joined_amount(self):
        small = normalize(["Дилер: viarubina ставит малый", "‘блайнд1 500."])
        self.assertEqual((small.event_type, small.actor, small.amount_decimal,
                          small.amount_role),
                         ("SMALL_BLIND", "viarubina", "1500", "SMALL_BLIND"))
        big = normalize(["Дилер: player ставит большой", "'блайнд3 000."])
        self.assertEqual((big.event_type, big.amount_decimal, big.amount_role),
                         ("BIG_BLIND", "3000", "BIG_BLIND"))

    def test_unreadable_blind_term_does_not_fall_back_to_bet(self):
        unknown = normalize(["Дилер: player ставит малый", "??? 1 500"])
        self.assertEqual(unknown.event_type, "UNKNOWN_EVENT")
        self.assertEqual(unknown.reason, "BLIND_TYPE_UNREADABLE")
        bet = normalize(["Дилер: player ставит 1 500"])
        self.assertEqual((bet.event_type, bet.amount_decimal), ("BET", "1500"))

    def test_pokerdom_check_wording(self):
        check = normalize(["Дилер: tanivolk делает чек ||"])
        self.assertEqual((check.event_type, check.actor, check.amount_decimal),
                         ("CHECK", "tanivolk", None))

    def test_negated_show_does_not_publish_positive_show_or_extend_actor(self):
        for fragments in (["Дилер: player не показывает", "карты"],
                          ["Дилер: player не", "показывает карты"],
                          ["Дилер: player НЕ   ПОКАЗЫВАЕТ карты"]):
            with self.subTest(fragments=fragments):
                result = normalize(fragments)
                self.assertEqual((result.event_type, result.actor,
                                  result.amount_decimal, result.amount_role),
                                 ("NO_SHOW", "player", None, None))
                self.assertIsNone(result.reason)
        shown = normalize(["Дилер: player показывает карты"])
        self.assertEqual((shown.event_type, shown.actor), ("SHOW", "player"))


if __name__ == "__main__":
    unittest.main()
