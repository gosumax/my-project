import numpy as np
from experiments.temporal_overlap_dedup import View, anchored_duplicates
from experiments.temporal_overlap_native import native_signature, native_distance
import experiments.temporal_overlap_dedup as replay


def glyph(seed):
    rng = np.random.default_rng(seed)
    pixels = np.zeros((16, 228), np.uint8)
    pixels[4:12, 10:190] = rng.integers(0, 2, (8, 180), dtype=np.uint8) * 220
    return pixels


def test_native_keeps_fixed_geometry_and_input():
    pixels = glyph(1)
    original = pixels.copy()
    assert native_signature(pixels).shape == (16, 216)
    assert np.array_equal(pixels, original)


def test_one_pixel_vertical_phase_matches_without_stretch():
    a = glyph(2)
    b = np.zeros_like(a)
    b[1:] = a[:-1]
    assert native_distance(native_signature(a), native_signature(b)) == 0


def test_distinct_glyphs_reject_and_geometry_unknown():
    assert native_distance(native_signature(glyph(3)), native_signature(glyph(4))) > 9
    assert native_distance(np.zeros((15, 216), np.uint8), np.zeros((16, 216), np.uint8)) == float('inf')


def test_real_repeated_line_with_new_identity_is_kept(monkeypatch):
    monkeypatch.setattr(replay, 'canonical_row_distance', native_distance)
    def v(rid, seed, inherited=False, origin='UNRESOLVED_IDENTITY'):
        return View(rid, native_signature(glyph(seed)), str(seed), 0, 0, inherited, origin, 'same poker text')
    history = [v('a', 1), v('b', 2), v('c', 3)]
    assert anchored_duplicates(history, [v('new-a', 1), v('new-b', 2), v('new-c', 3)]) == []
    assert anchored_duplicates(history, [v('new', 1, origin='NEW_CANDIDATE'), v('b', 2, True), v('c', 3, True)]) == []
