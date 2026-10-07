import numpy as np
from experiments.temporal_overlap_dedup import View, anchored_duplicates


def v(rid, value, *, inherited=False, origin='UNRESOLVED_IDENTITY', text='action'):
    return View(rid, np.full((20, 192), value, np.uint8), f'hash{value}', 0, 0, inherited, origin, text)


def test_duplicate_prefix_with_two_inherited_anchors():
    old = [v('a', 10), v('b', 50), v('c', 90)]
    block = [v('dup', 10), v('b', 50, inherited=True), v('c', 90, inherited=True), v('new', 150, origin='NEW_CANDIDATE')]
    found = anchored_duplicates(old, block)
    assert [(a.row_id, b.row_id) for a, b, _ in found] == [('dup', 'a')]
    assert found[0][2]['exact_hash_match']


def test_same_text_and_pixels_without_two_anchors_is_kept():
    old = [v('a', 10), v('b', 50), v('c', 90)]
    assert anchored_duplicates(old, [v('dup', 10), v('b', 50, inherited=True)]) == []
    assert anchored_duplicates(old, [v('x', 10), v('y', 50), v('z', 90)]) == []


def test_conflicting_anchor_order_rejects():
    old = [v('a', 10), v('b', 50), v('c', 90)]
    assert anchored_duplicates(old, [v('dup', 10), v('c', 50, inherited=True), v('b', 90, inherited=True)]) == []


def test_text_or_pixel_conflict_keeps_row():
    old = [v('a', 10), v('b', 50), v('c', 90)]
    for candidate in (v('dup', 10, text='different'), v('dup', 40)):
        assert anchored_duplicates(old, [candidate, v('b', 50, inherited=True), v('c', 90, inherited=True)]) == []


def test_real_new_candidate_repeat_is_kept():
    old = [v('a', 10), v('b', 50), v('c', 90)]
    assert anchored_duplicates(old, [v('repeat', 10, origin='NEW_CANDIDATE'), v('b', 50, inherited=True), v('c', 90, inherited=True)]) == []


def test_invalid_geometry_cannot_match():
    old = [v('a', 10), v('b', 50), v('c', 90)]
    candidate = v('dup', 10)
    candidate.signature = np.empty((0, 0), np.uint8)
    assert anchored_duplicates(old, [candidate, v('b', 50, inherited=True), v('c', 90, inherited=True)]) == []
