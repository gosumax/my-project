"""Bounded DEV metric probe, not a dedup implementation or acceptance test."""
import json
import base64
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.temporal_overlap_dedup import read_jsonl


def glyph_metrics(old, new):
    if old.shape != new.shape:
        return {'shape_match': False}
    h = old.shape[0]
    comparisons = []
    for dy in (-1, 0, 1):
        a = old[max(0, dy):min(h, h + dy)]
        b = new[max(0, -dy):min(h, h - dy)]
        dropped = ((old[:dy] > 70).any() or (new[h-dy:] > 70).any()) if dy > 0 else (
            ((old[h+dy:] > 70).any() or (new[:-dy] > 70).any()) if dy < 0 else False)
        if dropped:
            continue
        union = np.uint8((a > 70) | (b > 70))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(union, 8)
        locals_ = []
        for x, y, w, ht, area in stats[1:]:
            if area < 5:
                continue
            x0, x1, y0, y1 = max(0, x-1), min(a.shape[1], x+w+1), max(0, y-1), min(a.shape[0], y+ht+1)
            aa, bb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
            ncc = float(cv2.matchTemplate(aa, bb, cv2.TM_CCOEFF_NORMED)[0, 0]) if aa.std() and bb.std() else -1.0
            binary_a, binary_b = aa > 70, bb > 70
            dice = 2 * np.count_nonzero(binary_a & binary_b) / max(1, np.count_nonzero(binary_a) + np.count_nonzero(binary_b))
            locals_.append({'x': int(x), 'y': int(y), 'width': int(w), 'height': int(ht),
                            'area': int(area), 'ncc': ncc, 'dice': dice})
        comparisons.append({'dy': dy, 'mae': float(np.mean(cv2.absdiff(a, b))),
                            'min_glyph_ncc': min((r['ncc'] for r in locals_), default=-1),
                            'min_glyph_dice': min((r['dice'] for r in locals_), default=0),
                            'glyphs': locals_})
    return min(comparisons, key=lambda r: r['mae']) if comparisons else {'all_shifts_discard_ink': True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--new', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = {r['row_id']: r for r in read_jsonl(args.old / 'physical_rows.jsonl')}
    pix = lambda r: np.frombuffer(base64.b64decode(r['row_pixels_base64']), np.uint8).reshape(r['pixel_shape'])[:, :-12]
    out = []
    for r in read_jsonl(args.new / 'removed_duplicates.jsonl'):
        out.append({'table_id': r['table_id'], 'text': r['text'], 'current': r['old_row_id'],
                    'previous': r['matched_previous_row_id'],
                    **glyph_metrics(pix(rows[r['matched_previous_row_id']]), pix(rows[r['old_row_id']]))})
    a, b = np.zeros((16, 216), np.uint8), np.zeros((16, 216), np.uint8)
    cv2.putText(a, 'lifeforkaifff bets 44 055', (1, 11), cv2.FONT_HERSHEY_SIMPLEX, .3, 220, 1)
    cv2.putText(b, 'lifeforkaifff bets 44 056', (1, 11), cv2.FONT_HERSHEY_SIMPLEX, .3, 220, 1)
    out.append({'table_id': 'SYNTHETIC', 'text': '44 055 vs 44 056', **glyph_metrics(a, b)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps([{k: v for k, v in r.items() if k != 'glyphs'} for r in out], ensure_ascii=False))


if __name__ == '__main__':
    main()
