from __future__ import annotations

import unittest

import numpy as np

from parser_core.chat_rows import (align_upward, exclude_first_visible_span,
                                   match_rows, padded_row_span, physical_spans)


class ChatRowTests(unittest.TestCase):
    def test_scrollbar_movement_does_not_create_a_second_row_identity(self):
        old = np.zeros((160, 100), dtype=np.uint8)
        for y in (30, 70):
            old[y:y+8, 10:70] = 190
        old[70:78, 88:] = 255
        new = np.zeros_like(old)
        new[:140, :88] = old[20:, :88]
        # Repeated text is still two distinct rows; a new copy is unmatched.
        new[140:148, 10:70] = 190
        original_old, original_new = old.copy(), new.copy()
        self.assertEqual(match_rows(old, physical_spans(old), new,
                                    physical_spans(new), 20), {0: 0, 1: 1})
        np.testing.assert_array_equal(old, original_old)
        np.testing.assert_array_equal(new, original_new)

    def test_excluding_scrollbar_does_not_match_changed_text(self):
        old = np.zeros((160, 100), dtype=np.uint8)
        old[70:78, 10:70] = 190
        old[70:78, 88:] = 255
        new = np.zeros_like(old)
        new[50:58, 35:85] = 190
        self.assertEqual(match_rows(old, physical_spans(old), new,
                                    physical_spans(new), 20), {})

    def test_bright_scrollbar_does_not_merge_distinct_text_lines(self):
        gray = np.zeros((80, 100), dtype=np.uint8)
        gray[10:18, 5:80] = 180
        gray[31:39, 5:80] = 180
        gray[55:63, 5:80] = 180
        gray[10:70, 92:100] = 210  # a right-edge scrollbar
        self.assertEqual(physical_spans(gray), [(10, 18), (31, 39), (55, 63)])
        self.assertEqual(padded_row_span(physical_spans(gray), 1, 80), (27, 43))

    def test_row_padding_stops_at_source_and_neighbour_boundaries(self):
        spans = [(0, 5), (18, 29), (38, 49)]
        self.assertEqual(padded_row_span(spans, 0, 60), (0, 16))
        middle = padded_row_span(spans, 1, 60)
        self.assertEqual(middle[1] - middle[0], 16)
        self.assertLessEqual(middle[1], spans[2][0] - 2)

    def test_first_visible_band_is_suppressed_even_when_not_at_pixel_edge(self):
        kept, suppressed = exclude_first_visible_span([(5, 17), (25, 37), (45, 57)])
        self.assertEqual(kept, [(25, 37), (45, 57)])
        self.assertEqual(suppressed, [(5, 17)])
        kept, suppressed = exclude_first_visible_span([(0, 7), (15, 27), (35, 47)])
        self.assertEqual(kept, [(15, 27), (35, 47)])
        self.assertEqual(suppressed, [(0, 7)])

    def test_scroll_keeps_repeated_rows_distinct(self):
        old = np.zeros((160, 100), dtype=np.uint8)
        old[30:38, 10:90] = 210
        old[70:78, 10:90] = 210  # identical text-like raster, different row
        new = np.zeros_like(old)
        new[:140] = old[20:]
        new[140:148, 10:90] = 210
        old_spans = physical_spans(old)
        new_spans = physical_spans(new)
        self.assertEqual((len(old_spans), len(new_spans)), (2, 3))
        alignment = align_upward(old, new, max_shift=50)
        self.assertTrue(alignment.proven)
        self.assertEqual(alignment.shift_px, 20)
        self.assertEqual(match_rows(old, old_spans, new, new_spans, 20), {0: 0, 1: 1})

    def test_unrelated_images_do_not_prove_alignment(self):
        rng = np.random.default_rng(1)
        old = rng.integers(0, 256, (160, 100), dtype=np.uint8)
        new = rng.integers(0, 256, (160, 100), dtype=np.uint8)
        self.assertFalse(align_upward(old, new, max_shift=50).proven)

    def test_large_scroll_can_prove_overlap(self):
        old = np.zeros((715, 100), dtype=np.uint8)
        for index, y in enumerate(range(250, 650, 40)):
            old[y:y+7, 10:30+index*3] = 180
        new = np.zeros_like(old)
        new[:286] = old[429:]
        for index, y in enumerate(range(320, 700, 40)):
            new[y:y+7, 10:50+index*2] = 180
        alignment = align_upward(old, new)
        matches = match_rows(old, physical_spans(old), new, physical_spans(new),
                             alignment.shift_px)
        self.assertTrue(alignment.proven)
        self.assertEqual(alignment.shift_px, 429)
        self.assertGreaterEqual(len(matches), 3)


if __name__ == "__main__":
    unittest.main()
