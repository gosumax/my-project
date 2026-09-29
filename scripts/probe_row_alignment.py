"""Print pairwise chat-alignment diagnostics for a saved capture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.chat_rows import align_upward, match_rows, physical_spans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("crop_directory", type=Path)
    parser.add_argument("frames", nargs="+", type=int)
    parser.add_argument("--max-shift", type=int, default=250)
    args = parser.parse_args()
    previous = None
    for frame_id in args.frames:
        path = args.crop_directory / f"{frame_id:09d}_chat.png"
        current = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if current is None:
            raise FileNotFoundError(path)
        spans = physical_spans(current)
        if previous is not None:
            old_id, old, old_spans = previous
            zero = match_rows(old, old_spans, current, spans, 0)
            alignment = align_upward(old, current, max_shift=args.max_shift)
            shifted = match_rows(old, old_spans, current, spans, alignment.shift_px)
            print(old_id, frame_id, "rows", len(old_spans), len(spans),
                  "zero_matches", len(zero), "shift_matches", len(shifted),
                  "shift", alignment.shift_px, "score", round(alignment.score, 2),
                  "zero_score", round(alignment.zero_score, 2), "proven", alignment.proven)
        previous = frame_id, current, spans


if __name__ == "__main__":
    main()
