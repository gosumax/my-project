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


def _glyphs(gray: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray] | None:
    ink = (gray[3:12, start + 1:end] < 190).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    components = sorted((index for index in range(1, count) if stats[index, 4] >= 5),
                        key=lambda index: stats[index, 0])
    if len(components) != 2:
        return None
    glyphs = []
    for index in components:
        x, y, width, height, _ = map(int, stats[index])
        mask = (labels[y:y + height, x:x + width] == index).astype(np.uint8)
        glyphs.append(cv2.resize(mask, (8, 8), interpolation=cv2.INTER_NEAREST))
    return glyphs[0], glyphs[1]


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
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != source_sha256:
        raise ValueError(f"Crop hash mismatch: {path}")
    with Image.open(io.BytesIO(source)) as opened:
        gray = np.asarray(opened.convert("L"))
    if gray.shape[0] != 16 or gray.shape[1] < 100:
        return None
    spans = [(start, end) for start, end in _runs(
        np.flatnonzero(gray[3, :min(150, gray.shape[1])] > 150))
        if 18 <= end - start + 1 <= 22]
    if not 3 <= len(spans) <= 5 or any(
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
        if distance is not None:
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
