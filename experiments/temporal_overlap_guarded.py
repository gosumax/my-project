"""EXPERIMENT ONLY: ablate glyph/coverage guards, then confirmed-view gallery.

Same full golden capture and recognition evidence. No production code mutation.
The two arms differ only in allowing previously inherited appearance templates.
"""
from __future__ import annotations
import argparse
import base64
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import temporal_overlap_dedup as baseline
from experiments.temporal_overlap_native import native_signature, native_distance
from experiments.probe_glyph_overlap import glyph_metrics
from parser_core.ram_first_stage import FirstStage

GLYPH_NCC_MIN = 0.90  # DEV guard; no independent-video acceptance is claimed.
MAX_VIEWS = 256


@dataclass
class Observation:
    view: baseline.View
    frame_id: int
    pixels: np.ndarray


def guarded_metrics(old, new):
    metrics = glyph_metrics(old, new)
    if 'mae' not in metrics or not metrics['glyphs']:
        return None
    if metrics['mae'] > 9 or metrics['min_glyph_ncc'] < GLYPH_NCC_MIN:
        return None
    return metrics


class Reducer:
    def __init__(self, gallery=False):
        self.use_gallery = gallery
        self.histories = defaultdict(list)
        self.aliases = {}
        self.gallery = defaultdict(dict)
        self.first = {}
        self.removals = []
        self.seconds = 0.0
        self.block_count = 0

    def resolve(self, rid):
        seen = set()
        while rid in self.aliases:
            if rid in seen:
                raise ValueError('Identity alias cycle')
            seen.add(rid)
            rid = self.aliases[rid]
        return rid

    def process(self, table, frame, observations):
        start = time.perf_counter()
        block = []
        by_original = {}
        for obs in sorted(observations, key=lambda o: o.view.index):
            v = obs.view
            rid = self.resolve(v.row_id) if v.inherited else v.row_id
            copy = baseline.View(rid, v.signature, v.pixel_hash, v.y1, v.index,
                                 v.inherited, v.origin, v.text)
            block.append(copy)
            by_original[copy.row_id] = obs
        history = self.histories[table]
        proposals = baseline.anchored_duplicates(history[-256:], block)
        removed = set()
        for current, previous, evidence in proposals:
            # Unknown or conflicting OCR cannot be used to confirm a new alias.
            if not current.text.strip() or not previous.text.strip() or current.text != previous.text:
                continue
            templates = list(self.gallery[previous.row_id].values()) if self.use_gallery else [self.first[previous.row_id]]
            # Rank cheaply, then validate glyphs. Never use current/future views.
            templates = [obs for obs in templates if obs.frame_id < frame]
            templates.sort(key=lambda obs: native_distance(obs.view.signature, current.signature))
            selected, metrics = None, None
            for obs in templates:
                if native_distance(obs.view.signature, current.signature) > 9:
                    break
                candidate = guarded_metrics(obs.view.signature, current.signature)
                if candidate is not None:
                    selected, metrics = obs, candidate
                    break
            if selected is None:
                continue
            self.aliases[current.row_id] = previous.row_id
            removed.add(current.row_id)
            chosen = selected.view
            evidence = dict(evidence)
            evidence['proposal_native_mae'] = evidence.pop('canonical_mae')
            for pair in evidence['overlap_pairs']:
                pair['native_overlap_mae'] = pair.pop('canonical_mae')
            self.removals.append({'table_id': f'T{table:02d}', 'frame_id': frame,
                'text': current.text, 'y1': current.y1, 'source_span_index': current.index,
                'old_row_id': current.row_id, 'matched_previous_row_id': previous.row_id,
                **evidence, 'exact_hash_match': current.pixel_hash == chosen.pixel_hash,
                'distance_metric': 'native_overlap_mae_and_per_component_ncc',
                'native_overlap_mae': metrics['mae'], 'matched_shift_y': metrics['dy'],
                'min_glyph_ncc': metrics['min_glyph_ncc'], 'glyph_ncc_min': GLYPH_NCC_MIN,
                'glyphs': metrics['glyphs'], 'nonempty_exact_ocr_match': True,
                'active_edge_pixels_discarded': False,
                'current_pixel_sha256': current.pixel_hash, 'previous_pixel_sha256': chosen.pixel_hash,
                'matched_previous_observation': {'frame_id': selected.frame_id,
                    'stable_row_id': chosen.row_id, 'source_span_index': chosen.index,
                    'y1': chosen.y1, 'inherited': chosen.inherited,
                    'pixel_shape': list(selected.pixels.shape),
                    'row_pixels_base64': base64.b64encode(selected.pixels.tobytes()).decode('ascii')},
                'gallery_enabled': self.use_gallery})
        history.extend(v for v in block if not v.inherited and v.row_id not in removed)
        # Add observations only AFTER this block's decisions; never self-match.
        for original in observations:
            rid = self.resolve(original.view.row_id)
            if rid not in self.first:
                self.first[rid] = original
            if self.use_gallery:
                views = self.gallery[rid]
                views.setdefault(original.view.pixel_hash, original)
                if len(views) > MAX_VIEWS:
                    # First observation remains available separately.
                    del views[next(iter(views))]
                views.setdefault(self.first[rid].view.pixel_hash, self.first[rid])
        self.seconds += time.perf_counter() - start
        self.block_count += 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-frames', type=int, default=2040,
                   help='Verified full source count; no matching parameter changes')
    p.add_argument('--expected-source-sha256',
                   help='Explicit read-only source binding for another video')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Fresh experiment output required')
    manifest_bytes = (ROOT / 'production_manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    pinned = {name: baseline.sha(ROOT / name) for name in manifest['hashes']}
    if pinned != manifest['hashes']:
        raise ValueError('Golden integrity mismatch')
    meta = json.loads((args.old / 'run_metadata.json').read_text('utf-8'))
    summary = json.loads((args.old / 'final_summary.json').read_text('utf-8'))
    assert meta['production_manifest_sha256'] == hashlib.sha256(manifest_bytes).hexdigest()
    assert meta['production_modified_files'] == [] and not meta['vl_enabled'] and meta['cards_enabled']
    assert summary['frames'] == meta['frames_requested'] == args.expected_frames
    video = Path(meta['video'])
    expected_source = args.expected_source_sha256 or manifest['golden_benchmark']['source_sha256']
    assert baseline.sha(video) == meta['source_video_sha256'] == expected_source
    physical = baseline.read_jsonl(args.old / 'physical_rows.jsonl')
    timeline = baseline.read_jsonl(args.old / 'ordered_timeline.jsonl')
    by_key = {r['row_key']: r for r in physical}
    by_id = {r['stable_row_id']: r for r in timeline}
    reducers = {'NO_GALLERY': Reducer(), 'NEW': Reducer(gallery=True)}
    # Geometry is shared by both arms. Registered FirstStage functions stay intact.
    baseline.canonical_row_distance = native_distance
    pending, key = [], None
    emitted = set()

    def flush():
        if not pending:
            return
        for reducer in reducers.values():
            reducer.process(*key, pending)
        pending.clear()

    def record(table, frame, rid, index, y, origin, pixels, inherited):
        nonlocal key
        group = (table, frame)
        if key != group:
            flush()
            key = group
        raw = np.array(pixels, copy=True, order='C')
        v = baseline.View(rid, native_signature(raw), hashlib.sha256(raw.tobytes()).hexdigest(),
                          y, index, inherited, origin, by_id[rid]['raw_ocr_text'])
        pending.append(Observation(v, frame, raw))

    def observe(m, pixels):
        record(m['table_id'], m['frame_id'], m['stable_row_id'], m['source_span_index'],
               m['y1'], m['origin'], pixels, True)

    start = time.perf_counter()
    stage = FirstStage(video, meta['table_sessions'], summary['frames'], root=ROOT, identity_observer=observe)
    for row in stage.rows():
        row_key = f'{row.table_id}:{row.frame_id}:{row.source_span_index}'
        expected = by_key[row_key]
        assert row_key not in emitted
        assert (row.stable_row_id, row.pixel_sha256, row.y1, row.origin, row.epoch) == (
            expected['row_id'], expected['row_pixel_sha256'], expected['y1'], expected['origin'], expected['epoch'])
        assert row.pixels.tobytes() == base64.b64decode(expected['row_pixels_base64'])
        emitted.add(row_key)
        record(row.table_id, row.frame_id, row.stable_row_id, row.source_span_index,
               row.y1, row.origin, row.pixels, False)
    flush()
    wall = time.perf_counter() - start
    assert emitted == set(by_key) and stage.seen_frames == summary['frames']
    assert pinned == {name: baseline.sha(ROOT / name) for name in pinned}
    code_paths = ['experiments/temporal_overlap_guarded.py', 'experiments/temporal_overlap_native.py',
                  'experiments/temporal_overlap_dedup.py', 'experiments/probe_glyph_overlap.py']
    for arm, reducer in reducers.items():
        output = args.output / arm
        output.mkdir(parents=True)
        removed_ids = {r['old_row_id'] for r in reducer.removals}
        new = [r for r in timeline if r['stable_row_id'] not in removed_ids]
        for r in reducer.removals:
            t = by_id[r['old_row_id']]['video_time']
            r['timestamp_ms'] = t * 1000 if t is not None else None
        for name in ('physical_rows.jsonl', 'raw_ocr_results.jsonl', 'visual_evidence.jsonl'):
            shutil.copyfile(args.old / name, output / name)
        baseline.write_jsonl(output / 'ordered_timeline.jsonl', new)
        baseline.write_jsonl(output / 'removed_duplicates.jsonl', reducer.removals)
        baseline.write_jsonl(output / 'T01_before.jsonl', [r for r in timeline if r['table_id'] == 'T01'])
        baseline.write_jsonl(output / 'T01_after.jsonl', [r for r in new if r['table_id'] == 'T01'])
        result = {'experiment': 'GUARDED_NATIVE_GALLERY_V4' if reducer.use_gallery else 'GUARDED_NATIVE_V3',
            'old': str(args.old.resolve()), 'frames': stage.seen_frames, 'visible_blocks': reducer.block_count,
            'source_sha256': expected_source, 'expected_full_frames': args.expected_frames,
            'ab_changed_variable': 'gallery of confirmed previous identity appearances; all guards identical across these two arms',
            'old_physical_rows': len(physical), 'new_physical_rows': len(physical),
            'old_ocr_rows': len(physical), 'new_ocr_rows': len(physical),
            'old_timeline_rows': len(timeline), 'new_timeline_rows': len(new), 'removed_count': len(removed_ids),
            'old_wall_seconds': summary['wall_seconds'], 'replay_wall_seconds': wall,
            'dedup_seconds': reducer.seconds, 'dual_arm_replay': True,
            'performance_scope': 'One full capture replay shared by two arms; OCR/cards reused, not end-to-end speedup',
            'production_unchanged': True, 'physical_parity': True,
            'distance_metric': 'native_overlap_mae_and_per_component_ncc', 'glyph_ncc_min': GLYPH_NCC_MIN,
            'max_gallery_views_per_identity': MAX_VIEWS,
            'experiment_code_sha256': {name: baseline.sha(ROOT / name) for name in code_paths},
            'input_hashes': {name: baseline.sha(args.old / name) for name in ('physical_rows.jsonl', 'raw_ocr_results.jsonl', 'visual_evidence.jsonl', 'ordered_timeline.jsonl', 'run_metadata.json')},
            'raw_evidence_hash_parity': {name: baseline.sha(args.old / name) == baseline.sha(output / name)
                for name in ('physical_rows.jsonl', 'raw_ocr_results.jsonl', 'visual_evidence.jsonl')},
            'retained_timeline_rows_unchanged': True}
        (output / 'experiment_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
        print(json.dumps(result))


if __name__ == '__main__':
    main()
