from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from parser_core.layout import load_fixed_layout


class FixedLayoutTests(unittest.TestCase):
    def test_copied_layout_has_nine_distinct_in_bounds_slots(self):
        root = Path(__file__).resolve().parents[1]
        layout = load_fixed_layout(root / "configs" / "nine_table_layout.json",
                                   root / "configs" / "layout_847x404.json",
                                   root / "configs" / "chat_log_bottom_rois.json")
        self.assertEqual((layout.source_width, layout.source_height), (2560, 1440))
        self.assertEqual(len(layout.slots), 9)
        self.assertEqual(layout.slots[0].table_box, (1, 80, 848, 484))
        self.assertEqual(layout.slots[0].chat_box[3], 80 + 402)
        self.assertEqual(layout.slots[2].chat_box[3], 80 + 404)
        self.assertTrue(all(slot.chat_box[0] >= slot.table_box[0]
                            for slot in layout.slots))

    def test_rejects_chat_outside_its_configured_slot(self):
        root = Path(__file__).resolve().parents[1]
        profile = json.loads((root / "configs" / "layout_847x404.json").read_text(encoding="utf-8"))
        profile["slots"]["slot_01"]["chat"]["chat_log"] = [800, 25, 900, 388]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid_profile.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid geometry for slot 1"):
                load_fixed_layout(root / "configs" / "nine_table_layout.json", path,
                                  root / "configs" / "chat_log_bottom_rois.json")

    def test_rejects_calibration_outside_slot(self):
        root = Path(__file__).resolve().parents[1]
        calibration = json.loads((root / "configs" / "chat_log_bottom_rois.json").read_text(encoding="utf-8"))
        calibration["slots"]["slot_01"] = 405
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid_calibration.json"
            path.write_text(json.dumps(calibration), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid chat calibration for slot 1"):
                load_fixed_layout(root / "configs" / "nine_table_layout.json",
                                  root / "configs" / "layout_847x404.json", path)


if __name__ == "__main__":
    unittest.main()
