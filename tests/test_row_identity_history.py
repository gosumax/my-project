import numpy as np

from parser_core.chat_rows import match_row_blocks


def row(value: int) -> np.ndarray:
    image = np.zeros((20, 192), dtype=np.uint8)
    image[:, 8:184] = value
    image[3:17, 20 + value % 30:40 + value % 30] = min(255, value + 30)
    return image


def flatten(blocks):
    out = {}
    for block in blocks:
        out.update(block.mapping)
    return out


def test_recovers_old_prefix_and_leaves_new_suffix_unmatched():
    old = [row(v) for v in (20, 40, 60, 80, 100)]
    current = [row(v) for v in (60, 80, 100, 120, 140)]
    matches = flatten(match_row_blocks(old, current))
    assert matches == {0: 2, 1: 3, 2: 4}
    assert 3 not in matches
    assert 4 not in matches


def test_single_repeated_poker_line_is_not_enough_to_suppress():
    old = [row(20), row(40), row(60)]
    current = [row(99), row(40), row(120)]
    assert match_row_blocks(old, current) == []


def test_two_row_match_requires_strong_visual_agreement():
    old = [row(20), row(40)]
    current = [old[0].copy(), old[1].copy()]
    blocks = match_row_blocks(old, current)
    assert flatten(blocks) == {0: 0, 1: 1}
