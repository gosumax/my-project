from collections import OrderedDict
from pathlib import Path

import numpy as np

from scripts.track_chat_rows import save_or_reuse_row


def test_only_identical_full_row_pixels_reuse_a_png(tmp_path: Path):
    cache = OrderedDict()
    original = np.zeros((32, 100), dtype=np.uint8)
    original[4:12, 10:80] = 180
    digest, relative, reused = save_or_reuse_row(
        original, (0, 16), Path("first.png"), tmp_path, cache)
    assert not reused
    duplicate = save_or_reuse_row(original.copy(), (0, 16), Path("second.png"), tmp_path, cache)
    assert duplicate == (digest, relative, True)
    assert not (tmp_path / "second.png").exists()
    changed = original.copy()
    changed[8, 99] = 255
    changed_hash, _, reused = save_or_reuse_row(
        changed, (0, 16), Path("changed.png"), tmp_path, cache)
    assert not reused
    assert changed_hash != digest
