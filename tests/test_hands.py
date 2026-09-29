from __future__ import annotations

import unittest

from parser_core.hands import OrderedEvent, partition


class HandWindowTests(unittest.TestCase):
    def test_initial_win_and_header_new_hand_id(self):
        def event(n, kind, number=None):
            return OrderedEvent(str(n), 1, "table", f"{n:03d}", kind, number)
        windows = partition([event(0, "WIN"), event(1, "HAND_HEADER", "123"),
                             event(2, "NEW_HAND"), event(3, "HAND_ID", "123"),
                             event(4, "ANTE")])
        self.assertEqual([[e.event_type for e in window.events] for window in windows],
                         [["WIN"], ["HAND_HEADER", "NEW_HAND", "HAND_ID", "ANTE"]])
        self.assertEqual(windows[0].start_boundary, "OPEN_START")
        self.assertEqual(windows[1].hand_numbers, ["123"])


if __name__ == "__main__":
    unittest.main()
