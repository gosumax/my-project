"""Focused regression tests for conservative chat-row overlap identity."""

from __future__ import annotations

import cv2
import numpy as np

from parser_core.chat_rows import match_rows_overlap


def _rows(count: int = 8, height: int = 18, width: int = 320) -> list[np.ndarray]:
    output = []
    for index in range(count):
        image = np.zeros((height, width), np.uint8)
        cv2.putText(
            image,
            f"ROW_{index}_VALUE_{index * 137}",
            (4, 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            220,
            1,
            cv2.LINE_AA,
        )
        output.append(image)
    return output


def _stack(rows: list[np.ndarray], indices: list[int], gap: int = 5):
    height, width = rows[0].shape
    canvas = np.zeros((len(indices) * height + (len(indices) - 1) * gap, width), np.uint8)
    spans = []
    y = 0
    for index in indices:
        canvas[y:y + height] = rows[index]
        spans.append((y, y + height))
        y += height + gap
    return canvas, spans


def test_scroll_overlap_keeps_old_rows_and_leaves_new_tail_unmatched():
    rows = _rows()
    previous, old_spans = _stack(rows, [0, 1, 2, 3, 4, 5])
    current, new_spans = _stack(rows, [2, 3, 4, 5, 6, 7])
    assert match_rows_overlap(previous, old_spans, current, new_spans) == {
        0: 2,
        1: 3,
        2: 4,
        3: 5,
    }


def test_single_repeated_phrase_is_not_identity_proof():
    rows = _rows()
    previous, old_spans = _stack(rows, [0, 1, 2])
    current, new_spans = _stack(rows, [7, 1, 6])
    assert match_rows_overlap(previous, old_spans, current, new_spans) == {}


def test_same_sequence_maps_one_to_one():
    rows = _rows()
    previous, old_spans = _stack(rows, [0, 1, 2, 3])
    current, new_spans = _stack(rows, [0, 1, 2, 3])
    assert match_rows_overlap(previous, old_spans, current, new_spans) == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
    }
