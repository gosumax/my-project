"""Diagnostic metrics only: native row translation, no dedup decision."""
import argparse
import base64
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.temporal_overlap_dedup import read_jsonl, signature
from parser_core.chat_rows import canonical_row_distance, physical_spans


def native_metrics(a, b):
    if a.shape != b.shape:
        return {'shape_match': False}
    a, b = a[:, :-12], b[:, :-12]
    h, w = a.shape
    candidates = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            old = a[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)]
            new = b[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
            candidates.append({'mae': float(np.mean(cv2.absdiff(old, new))), 'dy': dy, 'dx': dx,
                               'compared_pixels': int(old.size),
                               'ncc': float(cv2.matchTemplate(old, new, cv2.TM_CCOEFF_NORMED)[0, 0])})
    return {'shape_match': True, 'zero_shift_mae': float(np.mean(cv2.absdiff(a, b))),
            'best_translation': min(candidates, key=lambda v: v['mae']),
            'scope': 'Illustrative bounded search; MAE is not calibrated against negatives and does not prove duplicate'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Fresh output required')
    rows = {r['row_id'].rsplit(':', 1)[-1]: r for r in read_jsonl(args.old / 'physical_rows.jsonl') if r['table_id'] == 1}
    pixels = lambda r: np.frombuffer(base64.b64decode(r['row_pixels_base64']), np.uint8).reshape(r['pixel_shape'])
    results = []
    for current, previous in [('R000019', 'R000002'), ('R000025', 'R000009'), ('R000026', 'R000022'), ('R000030', 'R000016')]:
        old, new = pixels(rows[previous]), pixels(rows[current])
        results.append({'current': current, 'previous': previous,
            'old_tight_spans': physical_spans(old), 'new_tight_spans': physical_spans(new),
            'canonical_mae': canonical_row_distance(signature(old), signature(new)), **native_metrics(old, new)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), 'utf-8')
    print(json.dumps(results))


if __name__ == '__main__':
    main()
