"""EXPERIMENT: change only V1 identity raster geometry; keep overlap policy.

Use padded source rows without independent tight-span vertical rescaling.
One-pixel translation checks are diagnostic identity comparisons, not OCR edits.
"""
from __future__ import annotations
from pathlib import Path
import sys
import json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import temporal_overlap_dedup as replay


def native_signature(pixels):
    if pixels.ndim != 2 or pixels.size == 0:
        return np.empty((0, 0), np.uint8)
    # The same existing scrollbar exclusion, without modifying source pixels.
    return pixels[:, :-12].copy() if pixels.shape[1] >= 50 else pixels.copy()


def native_distance(old, new):
    if old.size == 0 or new.size == 0 or old.shape != new.shape:
        return float('inf')
    h = old.shape[0]
    distances = []
    for dy in (-1, 0, 1):
        a = old[max(0, dy):min(h, h + dy)]
        b = new[max(0, -dy):min(h, h - dy)]
        if a.size:
            distances.append(float(np.mean(cv2.absdiff(a, b))))
    return min(distances, default=float('inf'))


def main():
    replay.signature = native_signature
    replay.canonical_row_distance = native_distance
    replay.main()
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    summary = json.loads((output / 'experiment_summary.json').read_text('utf-8'))
    summary.update(experiment='ANCHORED_TEMPORAL_OVERLAP_NATIVE_V2',
                   ab_changed_variable='identity raster geometry only; padded native pixels with +/-1 vertical translation',
                   distance_metric='native_overlap_mae',
                   experiment_code_sha256={str(p.relative_to(ROOT)): replay.sha(p) for p in (
                       Path(__file__).resolve(), ROOT / 'experiments/temporal_overlap_dedup.py')})
    (output / 'experiment_summary.json').write_text(json.dumps(summary, indent=2), 'utf-8')
    removed = replay.read_jsonl(output / 'removed_duplicates.jsonl')
    for row in removed:
        row['native_overlap_mae'] = row.pop('canonical_mae')
        row['distance_metric'] = 'native_overlap_mae'
        row['translation_search_y'] = [-1, 0, 1]
        for pair in row['overlap_pairs']:
            pair['native_overlap_mae'] = pair.pop('canonical_mae')
    replay.write_jsonl(output / 'removed_duplicates.jsonl', removed)
    print(json.dumps(summary))
    # This wrapper is an experiment, never a replacement of FirstStage or OCR.


if __name__ == '__main__':
    main()
