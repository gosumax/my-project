"""Bounded visual check for tiny fixed-layout board-card chat rows."""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


BOARD_TILES_VERSION = "board_tiles_fixed_chat_v1"
TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "models" / "board_glyphs_v1.json"
_RANK_TOKEN = re.compile(r"10|[2-9AJQK]")


@dataclass(frozen=True)
class BoardTileReading:
    text: str
    cards: tuple[str, ...]
    rank_source: str
    rank_distances: tuple[int | None, ...]
    suit_distances: tuple[int, ...]
    template_sha256: str


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = previous = None
    for value in values:
        x = int(value)
        if start is None:
            start = previous = x
        elif x > previous + 1:
            spans.append((start, previous))
            start = previous = x
        else:
            previous = x
    if start is not None:
        spans.append((start, previous))
    return spans


def card_spans(gray: np.ndarray, *, recover_split: bool = False) -> tuple[list[tuple[int, int]], list[dict]]:
    """Keep native row-3 boxes; join only ink-split card backgrounds.

    A second, uninterrupted background scanline must independently support the
    entire joined interval. Never widen an already admissible native span.
    """
    runs = _runs(np.flatnonzero(gray[3, :min(150, gray.shape[1])] > 150))
    spans, recovered = [], []
    index = 0
    while index < len(runs):
        start, end = runs[index]
        width = end-start+1
        if 18 <= width <= 22:
            spans.append((start, end))
        elif recover_split and width < 18 and index+1 < len(runs):
            right_start, right_end = runs[index+1]
            if (right_end-right_start+1 < 18 and 1 <= right_start-end-1 <= 2 and
                    18 <= right_end-start+1 <= 22 and
                    bool(np.all(gray[2, start:right_end+1] > 150))):
                spans.append((start, right_end))
                recovered.append({"box_inclusive": [start, right_end],
                                  "fragment_boxes_inclusive": [[start, end], [right_start, right_end]],
                                  "support_scanline_y": 2, "source": "NATIVE_BACKGROUND_BRIDGED_INK"})
                index += 1
        index += 1
    return spans, recovered


def secondary_card_spans(gray: np.ndarray) -> tuple[list[tuple[int, int]], list[dict]]:
    """Conservative native y2 fallback for a rejected y3 row geometry.

    Only complete fixed-render backgrounds with independent rank/suit glyphs
    qualify. Caller must retain any already-admissible primary geometry.
    """
    if gray.ndim != 2 or gray.shape[0] != 16:
        return [], []
    spans = [(start, end) for start, end in _runs(
        np.flatnonzero(gray[2, :min(150, gray.shape[1])] > 150))
        if 20 <= end-start+1 <= 21]
    if not (2 <= len(spans) <= 5 and all(
            2 <= spans[i+1][0]-spans[i][1]-1 <= 5 for i in range(len(spans)-1))):
        return [], []
    for start, end in spans:
        boxes = _glyph_boxes(gray, start, end)
        if boxes is None:
            return [], []
        (rx, ry, rw, rh), (sx, sy, sw, sh) = boxes
        if not (1 <= rw <= 9 and 5 <= rh <= 9 and rx+rw < sx and
                sw <= 9 and 2 <= sy and sy+sh <= 13 and sx+sw <= end+1):
            return [], []
    return spans, [{"source": "SECONDARY_NATIVE_BACKGROUND_Y2",
                    "support_scanline_y": 2,
                    "box_inclusive": [start, end]}
                   for start, end in spans]


def _glyph_components(gray: np.ndarray, start: int, end: int):
    ink = (gray[3:12, start + 1:end] < 190).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    components = sorted((index for index in range(1, count) if stats[index, 4] >= 5),
                        key=lambda index: stats[index, 0])
    if len(components) != 2:
        return None
    return labels, stats, components


def _glyph_boxes(gray: np.ndarray, start: int, end: int):
    data = _glyph_components(gray, start, end)
    if data is None:
        return None
    _, stats, components = data
    return [(start+1+int(stats[index, 0]), 3+int(stats[index, 1]),
             int(stats[index, 2]), int(stats[index, 3])) for index in components]


def _glyphs(gray: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray] | None:
    data = _glyph_components(gray, start, end)
    if data is None:
        return None
    labels, stats, components = data
    glyphs = []
    for index in components:
        x, y, width, height, _ = map(int, stats[index])
        mask = (labels[y:y + height, x:x + width] == index).astype(np.uint8)
        glyphs.append(cv2.resize(mask, (8, 8), interpolation=cv2.INTER_NEAREST))
    return glyphs[0], glyphs[1]


SPLIT_PARTITION_RULE = "SECONDARY_PARTITION_RANK_COMPONENT_BRIDGE_V1"


def prove_split_card_partition(gray: np.ndarray) -> dict | None:
    """Resolve a primary subset only from an independently complete partition.

    The missing y3 background must be split by the rank's own connected ink,
    not by an arbitrary dark gap. Existing primary boxes are never moved, and
    at least two unchanged cards supply independent width/spacing references.
    Card recognition and the event's expected card count remain separate gates.
    """
    if gray.ndim != 2 or gray.shape[0] != 16:
        return None
    primary, _ = card_spans(gray, recover_split=True)
    secondary, _ = secondary_card_spans(gray)
    if (len(primary) < 2 or len(secondary) <= len(primary)
            or any(span not in secondary for span in primary)):
        return None
    references = [end-start+1 for start, end in primary]
    runs = _runs(np.flatnonzero(gray[3, :min(150, gray.shape[1])] > 150))
    bridges = []
    for start, end in secondary:
        if (start, end) in primary:
            continue
        fragments = [(a, b) for a, b in runs if a <= end and b >= start]
        if (len(fragments) != 2 or fragments[0][0] != start or fragments[1][1] != end
                or any(b-a+1 >= 18 for a, b in fragments)
                or any(abs(end-start+1-width) > 1 for width in references)
                or not np.all(gray[2, start:end+1] > 150)):
            return None
        gap_start, gap_end = fragments[0][1]+1, fragments[1][0]-1
        components = _glyph_components(gray, start, end)
        boxes = _glyph_boxes(gray, start, end)
        if components is None or boxes is None:
            return None
        labels, _, indices = components
        rx, ry, rw, rh = boxes[0]
        if (not start < gap_start <= gap_end < end
                or not rx <= gap_start <= gap_end < rx+rw or ry != 3
                or not np.all(labels[0, gap_start-start-1:gap_end-start] == indices[0])):
            return None
        bridges.append(dict(source=SPLIT_PARTITION_RULE, box_inclusive=[start, end],
            fragment_boxes_inclusive=[list(span) for span in fragments],
            ink_gap_inclusive=[gap_start, gap_end], rank_glyph_box_xywh=list(boxes[0]),
            support_scanline_y=2))
    return dict(rule=SPLIT_PARTITION_RULE,
        model_gray_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),
        primary_spans_inclusive=[list(span) for span in primary],
        chosen_spans_inclusive=[list(span) for span in secondary],
        reference_card_widths=references,
        inter_card_gaps=[b[0]-a[1]-1 for a, b in zip(secondary, secondary[1:])],
        bridged_spans=bridges)


def verify_card_geometry_resolution(gray: np.ndarray, hybrid: dict) -> bool:
    """Replay the new partition receipt; retain the legacy no-recovery contract."""
    if gray.ndim != 2 or gray.shape[0] != 16 or gray.shape[1] <= 112:
        return False
    model = gray[:, :-12]
    proof = prove_split_card_partition(model)
    if (hybrid.get('row_geometry_source') == SPLIT_PARTITION_RULE
            or 'secondary_partition_proof' in hybrid):
        _, legacy_recovery = card_spans(model, recover_split=True)
        return bool(proof and hybrid.get('row_geometry_source') == SPLIT_PARTITION_RULE
            and hybrid.get('secondary_partition_proof') == proof
            and hybrid.get('detected_spans_inclusive') == proof['chosen_spans_inclusive']
            and hybrid.get('primary_incomplete_spans_inclusive') == proof['primary_spans_inclusive']
            and hybrid.get('recovered_split_spans') == legacy_recovery+proof['bridged_spans'])
    # Stripping the recovery envelope cannot turn this newly admitted partition
    # into a legacy receipt. Other legacy recovery types retain their old veto.
    return not hybrid.get('recovered_split_spans') and (proof is None or
        hybrid.get('detected_spans_inclusive') == proof['primary_spans_inclusive'])


def _scores(glyph: np.ndarray, templates: list[dict]) -> list[tuple[str, int]]:
    by_label: dict[str, int] = {}
    for template in templates:
        reference = np.fromiter((int(bit) for bit in template["bits"]),
                                dtype=np.uint8).reshape(8, 8)
        distance = int(np.count_nonzero(glyph != reference))
        label = template["label"]
        by_label[label] = min(by_label.get(label, 65), distance)
    return sorted(by_label.items(), key=lambda item: item[1])


def read_board_tiles(path: Path, source_sha256: str,
                     vl_text: str, template_path: Path = TEMPLATE_PATH) -> BoardTileReading | None:
    """Use VL ranks and tile glyphs; abstain if segmentation or suits are unclear."""
    return read_board_tiles_bytes(path.read_bytes(), source_sha256, vl_text, template_path)


def read_board_tiles_bytes(source: bytes, source_sha256: str,
                           vl_text: str, template_path: Path = TEMPLATE_PATH) -> BoardTileReading | None:
    if hashlib.sha256(source).hexdigest() != source_sha256:
        raise ValueError("Crop hash mismatch")
    with Image.open(io.BytesIO(source)) as opened:
        gray = np.asarray(opened.convert("L"))
    if gray.shape[0] != 16 or gray.shape[1] < 100:
        return None
    spans = [(start, end) for start, end in _runs(
        np.flatnonzero(gray[3, :min(150, gray.shape[1])] > 150))
        if 18 <= end - start + 1 <= 22]
    if not 2 <= len(spans) <= 5 or any(
            not 2 <= spans[index + 1][0] - spans[index][1] - 1 <= 5
            for index in range(len(spans) - 1)):
        return None
    glyphs = [_glyphs(gray, start, end) for start, end in spans]
    if any(pair is None for pair in glyphs):
        return None

    template_bytes = template_path.read_bytes()
    model = json.loads(template_bytes)
    if model.get("schema") != "pokerdom.board-glyphs.fixed-chat.v1":
        raise ValueError("Unsupported board glyph templates")
    rank_scores = [_scores(pair[0], model["ranks"]) for pair in glyphs]
    suit_scores = [_scores(pair[1], model["suits"]) for pair in glyphs]
    suits = []
    for scores in suit_scores:
        if len(scores) < 2 or scores[0][1] > 14 or scores[1][1] - scores[0][1] < 3:
            return None
        suits.append(scores[0][0])

    ranks = _RANK_TOKEN.findall(vl_text.upper())
    rank_source = "VL_TEXT"
    if len(ranks) == len(spans) - 1:
        missing = []
        for index, scores in enumerate(rank_scores):
            if len(scores) >= 2 and scores[0][1] <= 8 and scores[1][1] - scores[0][1] >= 4:
                candidate = ranks[:index] + [scores[0][0]] + ranks[index:]
                if len(candidate) == len(spans) and all(
                        not any(label == rank and distance > 20 for label, distance in rank_scores[j])
                        for j, rank in enumerate(candidate)):
                    missing.append(candidate)
        if len(missing) != 1:
            return None
        ranks = missing[0]
        rank_source = "VL_TEXT_WITH_ONE_VISUAL_RANK"
    if len(ranks) != len(spans):
        return None

    rank_distances = []
    for rank, scores in zip(ranks, rank_scores):
        distance = next((score for label, score in scores if label == rank), None)
        if distance is None:
            # A VL-only rank without a pixel template is not visual agreement.
            return None
        if distance > 20 or (scores[0][0] != rank and scores[0][1] <= 8 and
                             distance - scores[0][1] >= 8):
            return None
        rank_distances.append(distance)
    cards = tuple(rank + suit for rank, suit in zip(ranks, suits))
    if len(set(cards)) != len(cards):
        return None
    return BoardTileReading(" ".join(f"{rank} {suit}" for rank, suit in zip(ranks, suits)),
                            cards, rank_source, tuple(rank_distances),
                            tuple(scores[0][1] for scores in suit_scores),
                            hashlib.sha256(template_bytes).hexdigest())
