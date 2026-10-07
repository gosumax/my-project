import numpy as np
from experiments.row_content_evidence import compare_rows, represent_row


def row():
    a = np.zeros((14, 40), np.uint8)
    a[5:8, 8:13] = 180
    a[6, 16:20] = 150  # minus sign remains support
    return a


def test_shift_retains_all_foreground_and_source_bytes():
    a = row()
    b = np.zeros_like(a)
    b[2:, 1:] = a[:-2, :-1]
    originals = a.copy(), b.copy()
    result = compare_rows(a, b)
    assert result['best_alignment']['dx'] == -1
    assert result['best_alignment']['dy'] == -2
    assert result['best_alignment']['exact_support_equal_after_alignment']
    assert result['exact_content_native']
    assert np.array_equal(a, originals[0]) and np.array_equal(b, originals[1])


def test_background_border_difference_is_separate_from_content():
    a, b = row(), row()
    b[0, :20] = 20
    result = compare_rows(a, b)
    assert result['best_alignment']['foreground_union']['mae'] == 0
    assert result['before_alignment']['outside_content']['changed_pixels'] == 20
    b[5, 8] = 190
    assert compare_rows(a, b)['best_alignment']['foreground_union']['mae'] > 0


def test_cropped_glyph_cannot_become_equal_by_overlap():
    a = row()
    b = a[:, :10].copy()
    result = compare_rows(a, b, scrollbar_px=0)
    assert not result['best_alignment']['exact_support_equal_after_alignment']
    assert not result['exact_content_native']
    assert result['best_alignment']['foreground_union']['mae'] > 0
    assert result['before_alignment']['missing_foreground_after'] > 0


def test_empty_and_readonly_representation():
    a = np.zeros((3, 20), np.uint8)
    represented = represent_row(a)
    assert represented.metadata['empty']
    assert not represented.pixels.flags.writeable
    result = compare_rows(a, a)
    assert not result['exact_content_native']
    assert not result['best_alignment']['exact_support_equal_after_alignment']
    assert result['best_alignment']['foreground_union']['mae'] is None


def test_minus_sign_and_scrollbar_exclusion():
    a = row()
    a[:, -12:] = 255
    represented = represent_row(a)
    assert represented.metadata['support_pixels'] == 19
    assert represented.bbox == (8, 5, 20, 8)
    assert represented.metadata['scrollbar_excluded_px'] == 12


def test_dark_background_and_aligned_edge_regions():
    a, b = row(), row()
    b[0, :20] = 20
    measured = compare_rows(a, b)['before_alignment']
    assert measured['dark_background']['changed_pixels'] == 20
    assert measured['edge_diff']['changed_pixels'] == 20
    assert measured['foreground_union']['changed_pixels'] == 0
