"""Conservative chat-row geometry and one-to-one overlap matching.

The output is physical-row evidence, never an accepted poker action.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ROW_TRACKER_VERSION = "row_track_v11_history"


@dataclass(frozen=True)
class Alignment:
    shift_px: int
    score: float
    zero_score: float
    proven: bool


@dataclass(frozen=True)
class RowBlockMatch:
    """One conservative contiguous row-block match."""

    mapping: dict[int, int]
    length: int
    average_distance: float


def physical_spans(gray: np.ndarray) -> list[tuple[int, int]]:
    if gray.ndim != 2:
        raise ValueError("Expected grayscale chat crop")
    # A bright scrollbar may occupy the full height at the right edge and
    # connect otherwise separate text lines in the horizontal projection.
    # Exclude only that narrow edge for segmentation; saved row pixels retain
    # the original full width for audit and OCR.
    segmentation = gray[:, :-12] if gray.shape[1] >= 50 else gray
    active = np.count_nonzero(segmentation > 70, axis=1) >= 5
    spans: list[tuple[int, int]] = []
    start = last = None
    for y, used in enumerate(active):
        if not used:
            continue
        if start is None:
            start = last = y
        elif y - last <= 4:
            last = y
        else:
            if last - start + 1 >= 4:
                spans.append((start, last + 1))
            start = last = y
    if start is not None and last - start + 1 >= 4:
        spans.append((start, last + 1))
    return spans


def exclude_first_visible_span(
        spans: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Never emit a row ROI for the topmost visible chat text band."""
    return spans[1:], spans[:1]


def padded_row_span(spans: list[tuple[int, int]], index: int,
                    image_height: int, target_height: int = 16,
                    neighbour_guard: int = 2) -> tuple[int, int]:
    """Expand one detected text band while excluding neighbouring text bands.

    The tight spans remain the inputs for tracking; this expanded interval is
    only for the saved/OCR row crop. At a chat edge it may be shorter because
    pixels outside the source ROI do not exist.
    """
    if not (0 <= index < len(spans)) or target_height < 1 or neighbour_guard < 0:
        raise ValueError("Invalid row span request")
    top, bottom = spans[index]
    lower = spans[index - 1][1] + neighbour_guard if index else 0
    upper = spans[index + 1][0] - neighbour_guard if index + 1 < len(spans) else image_height
    if not (0 <= lower <= top < bottom <= upper <= image_height):
        raise ValueError("Overlapping or out-of-bounds text spans")
    if bottom - top >= target_height:
        return top, bottom
    start = max(lower, min(top, (top + bottom - target_height) // 2))
    end = min(upper, max(bottom, start + target_height))
    if end - start < target_height:
        start = max(lower, end - target_height)
    return start, end


def align_upward(previous: np.ndarray, current: np.ndarray,
                 max_shift: int = 600) -> Alignment:
    if previous.shape != current.shape or previous.ndim != 2:
        return Alignment(0, float("inf"), float("inf"), False)
    height = previous.shape[0]
    maximum = min(max_shift, height - 100)
    if maximum < 0:
        return Alignment(0, float("inf"), float("inf"), False)
    # Coarse search on half-resolution rasters, then refine at native y pixels.
    old_coarse = previous[::2, ::3]
    new_coarse = current[::2, ::3]
    coarse_height = old_coarse.shape[0]
    coarse_scores = [float(np.mean(cv2.absdiff(old_coarse[shift:],
                                               new_coarse[:coarse_height-shift])))
                     for shift in range(maximum // 2 + 1)]
    guess = int(np.argmin(coarse_scores)) * 2
    old = previous[:, ::3]
    new = current[:, ::3]
    refined = [(shift, float(np.mean(cv2.absdiff(old[shift:], new[:height-shift]))))
               for shift in range(max(0, guess - 3), min(maximum, guess + 3) + 1)]
    shift, score = min(refined, key=lambda item: item[1])
    zero = float(np.mean(cv2.absdiff(old, new)))
    if shift == 0:
        proven = score <= 6.0
    else:
        proven = score <= 6.0 and score <= zero * 0.65 and zero - score >= 0.8
    return Alignment(shift, score, zero, proven)


def canonical_row(gray: np.ndarray, span: tuple[int, int],
                  width: int = 192, height: int = 20) -> np.ndarray:
    """Small normalized row image used only for identity matching.

    The right scrollbar edge is masked, as in ``row_distance``.  Keeping this
    representation tiny makes short-history matching cheap enough to use only
    on difficult chat transitions.
    """
    row = gray[span[0]:span[1]].copy()
    if row.size == 0:
        return np.empty((0, 0), dtype=np.uint8)
    if row.shape[1] >= 50:
        row[:, -12:] = 0
    return cv2.resize(row, (width, height), interpolation=cv2.INTER_AREA)


def canonical_row_distance(old: np.ndarray, new: np.ndarray) -> float:
    if old.size == 0 or new.size == 0 or old.shape != new.shape:
        return float("inf")
    return float(np.mean(cv2.absdiff(old, new)))


def row_distance(previous: np.ndarray, old_span: tuple[int, int],
                 current: np.ndarray, new_span: tuple[int, int]) -> float:
    # The scrollbar is fixed to the viewport, not to a physical text row.
    # Mask the same edge as segmentation without changing the resize geometry;
    # full saved crops stay intact, and the source arrays are never modified.
    old = previous[old_span[0]:old_span[1]].copy()
    new = current[new_span[0]:new_span[1]].copy()
    if old.size == 0 or new.size == 0:
        return float("inf")
    if old.shape[1] >= 50:
        old[:, -12:] = 0
    if new.shape[1] >= 50:
        new[:, -12:] = 0
    old = cv2.resize(old, (256, 24), interpolation=cv2.INTER_AREA)
    new = cv2.resize(new, (256, 24), interpolation=cv2.INTER_AREA)
    return float(np.mean(cv2.absdiff(old, new)))


def match_rows(previous: np.ndarray, old_spans: list[tuple[int, int]],
               current: np.ndarray, new_spans: list[tuple[int, int]],
               shift: int, max_distance: float = 10.0) -> dict[int, int]:
    """Map each new row index to one old index using order, position, and pixels."""
    candidates = []
    for new_index, new_span in enumerate(new_spans):
        new_center = sum(new_span) / 2
        for old_index, old_span in enumerate(old_spans):
            old_center = sum(old_span) / 2 - shift
            delta = abs(new_center - old_center)
            if delta > 7:
                continue
            distance = row_distance(previous, old_span, current, new_span)
            if distance <= max_distance:
                candidates.append((distance + delta, new_index, old_index))
    matches: dict[int, int] = {}
    used_old: set[int] = set()
    for _, new_index, old_index in sorted(candidates):
        if new_index not in matches and old_index not in used_old:
            matches[new_index] = old_index
            used_old.add(old_index)
    # A crossed assignment cannot establish chronology, even when pixels look alike.
    ordered = sorted(matches.items())
    if [old for _, old in ordered] != sorted(old for _, old in ordered):
        return {}
    return matches


def match_row_blocks(old_rows: list[np.ndarray], current_rows: list[np.ndarray], *,
                     max_distance: float = 9.0, min_run: int = 2,
                     two_row_max_average: float = 4.5) -> list[RowBlockMatch]:
    """Find conservative contiguous row sequences shared by two chat states.

    A single visually identical row is intentionally *not* accepted because
    poker chat legitimately repeats lines such as ``Dealer: Dealing cards``.
    We only accept a contiguous run of at least two rows; two-row matches must
    be especially close, while runs of three or more may use ``max_distance``.

    The function is position-independent, so it can recover identity after a
    larger chat jump where ``align_upward`` cannot prove the viewport shift.
    """
    if min_run < 2:
        raise ValueError("min_run must be at least 2")
    if not old_rows or not current_rows:
        return []

    n_old = len(old_rows)
    n_new = len(current_rows)
    distances = np.full((n_new, n_old), np.inf, dtype=np.float32)
    similar = np.zeros((n_new, n_old), dtype=np.uint8)
    for new_index, new_row in enumerate(current_rows):
        for old_index, old_row in enumerate(old_rows):
            distance = canonical_row_distance(old_row, new_row)
            distances[new_index, old_index] = distance
            if distance <= max_distance:
                similar[new_index, old_index] = 1

    # Longest-common-substring DP over row order.  Only diagonal continuation
    # is allowed, so accepted blocks preserve chronology and adjacency.
    dp = np.zeros((n_new + 1, n_old + 1), dtype=np.int16)
    candidates: list[tuple[int, float, int, int]] = []
    for new_index in range(n_new):
        for old_index in range(n_old):
            if not similar[new_index, old_index]:
                continue
            run = int(dp[new_index, old_index]) + 1
            dp[new_index + 1, old_index + 1] = run
            next_is_similar = (new_index + 1 < n_new and old_index + 1 < n_old and
                               bool(similar[new_index + 1, old_index + 1]))
            if next_is_similar or run < min_run:
                continue
            new_start = new_index - run + 1
            old_start = old_index - run + 1
            diag = [float(distances[new_start + k, old_start + k]) for k in range(run)]
            average = float(sum(diag) / run)
            if run == 2 and average > two_row_max_average:
                continue
            candidates.append((run, average, new_start, old_start))

    # Prefer longer and cleaner blocks.  Keep matches one-to-one and do not
    # allow overlapping current/old rows inside this historical snapshot.
    used_new: set[int] = set()
    used_old: set[int] = set()
    output: list[RowBlockMatch] = []
    for run, average, new_start, old_start in sorted(
            candidates, key=lambda item: (-item[0], item[1], item[2], item[3])):
        new_indices = range(new_start, new_start + run)
        old_indices = range(old_start, old_start + run)
        if any(i in used_new for i in new_indices) or any(i in used_old for i in old_indices):
            continue
        mapping = {new_start + k: old_start + k for k in range(run)}
        output.append(RowBlockMatch(mapping, run, average))
        used_new.update(mapping)
        used_old.update(mapping.values())
    return output
