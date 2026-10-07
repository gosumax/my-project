"""Full experimental capture replay of the manifest-pinned golden FirstStage.

Exactly the BROKEN experiment's fixed OLD recognition responses; no production
launcher substitution, synthetic OCR, recognition or source/config mutation.
"""
import argparse
import base64
from datetime import datetime, timezone, timedelta
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
from experiments.temporal_overlap_guarded import Observation
from experiments.temporal_overlap_boundary import BoundaryReducer
from experiments.temporal_overlap_native import native_signature, native_distance
from parser_core.ram_first_stage import FirstStage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--broken', type=Path, required=True)
    p.add_argument('--targeted', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    gate = json.loads((args.targeted / 'targeted_result.json').read_text('utf-8'))
    assert gate['pass'] and gate['recovered_physical_rows'] == 14 and gate['recovered_ante_actions'] == 12
    assert gate['previously_proven_positive_control_removed'] > 0
    args.output.mkdir(parents=True, exist_ok=False)
    mbytes = (ROOT / 'production_manifest.json').read_bytes()
    manifest = json.loads(mbytes)
    pinned = {name: baseline.sha(ROOT / name) for name in manifest['hashes']}
    assert pinned == manifest['hashes']
    meta = json.loads((args.old / 'run_metadata.json').read_text('utf-8'))
    old_summary = json.loads((args.old / 'final_summary.json').read_text('utf-8'))
    broken_summary = json.loads((args.broken / 'NEW/experiment_summary.json').read_text('utf-8'))
    assert meta['production_manifest_sha256'] == hashlib.sha256(mbytes).hexdigest()
    assert meta['production_modified_files'] == [] and meta['cards_enabled'] and not meta['vl_enabled']
    assert old_summary['frames'] == meta['frames_requested'] == broken_summary['frames'] == 28222
    video = Path(meta['video'])
    assert baseline.sha(video) == meta['source_video_sha256'] == broken_summary['source_sha256']
    paths = ['experiments/temporal_overlap_boundary.py', 'experiments/run_boundary_overlap.py',
             'experiments/temporal_overlap_guarded.py', 'experiments/temporal_overlap_dedup.py',
             'experiments/temporal_overlap_native.py', 'experiments/probe_glyph_overlap.py']
    exp_pinned = {name: baseline.sha(ROOT / name) for name in paths}
    started = datetime.now(timezone(timedelta(hours=3))).isoformat()
    protocol = {'command': sys.argv, 'started_at_moscow': started,
        'expected_frames': 28222, 'source_sha256': meta['source_video_sha256'],
        'production_manifest_sha256': hashlib.sha256(mbytes).hexdigest(),
        'experiment_code_sha256': exp_pinned, 'sessions': meta['table_sessions'],
        'recognition_scope': 'Same fixed real OLD OCR/cards as BROKEN; full source capture replay, not second live OCR benchmark',
        'original_worker_settings': {k: meta[k] for k in ('ocr_workers','visual_workers','card_workers')},
        'changed_variable': 'Explicit verified spatial hand/session context and conservative UNKNOWN barriers only'}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), 'utf-8')
    physical = baseline.read_jsonl(args.old / 'physical_rows.jsonl')
    timeline = baseline.read_jsonl(args.old / 'ordered_timeline.jsonl')
    by_key = {r['row_key']: r for r in physical}
    by_id = {r['stable_row_id']: r for r in timeline}
    reducer = BoundaryReducer(meta['table_sessions'])
    baseline.canonical_row_distance = native_distance
    pending, key = [], None
    emitted = []
    seen = set()

    def flush():
        if pending:
            reducer.process(*key, pending)
            pending.clear()

    def record(table, frame, rid, index, y, origin, pixels, inherited):
        nonlocal key
        group = (table, frame)
        if key != group:
            flush()
            key = group
        raw = np.array(pixels, copy=True, order='C')
        view = baseline.View(rid, native_signature(raw), hashlib.sha256(raw.tobytes()).hexdigest(),
            y, index, inherited, origin, by_id[rid]['raw_ocr_text'])
        pending.append(Observation(view, frame, raw))

    def observe(m, pixels):
        record(m['table_id'], m['frame_id'], m['stable_row_id'], m['source_span_index'],
               m['y1'], m['origin'], pixels, True)

    start = time.perf_counter()
    stage = FirstStage(video, meta['table_sessions'], old_summary['frames'], root=ROOT, identity_observer=observe)
    for row in stage.rows():
        row_key = f'{row.table_id}:{row.frame_id}:{row.source_span_index}'
        expected = by_key[row_key]
        assert row_key not in seen
        assert (row.stable_row_id, row.pixel_sha256, row.y1, row.origin, row.epoch) == (
            expected['row_id'], expected['row_pixel_sha256'], expected['y1'], expected['origin'], expected['epoch'])
        assert row.pixels.tobytes() == base64.b64decode(expected['row_pixels_base64'])
        seen.add(row_key)
        emitted.append(row_key)
        record(row.table_id, row.frame_id, row.stable_row_id, row.source_span_index,
            row.y1, row.origin, row.pixels, False)
        if stage.seen_frames and stage.seen_frames % 2000 == 0:
            print(json.dumps({'progress_frames': stage.seen_frames, 'emitted_rows': len(emitted),
                'elapsed_seconds': time.perf_counter()-start}), flush=True)
    flush()
    wall = time.perf_counter() - start
    assert seen == set(by_key) and stage.seen_frames == 28222
    assert pinned == {name: baseline.sha(ROOT / name) for name in pinned}
    assert exp_pinned == {name: baseline.sha(ROOT / name) for name in exp_pinned}
    removed_ids = {r['old_row_id'] for r in reducer.removals}
    new = [r for r in timeline if r['stable_row_id'] not in removed_ids]
    for r in reducer.removals:
        t = by_id[r['old_row_id']]['video_time']
        r['timestamp_ms'] = t * 1000 if t is not None else None
    for name in ('physical_rows.jsonl', 'raw_ocr_results.jsonl', 'visual_evidence.jsonl'):
        shutil.copyfile(args.old / name, args.output / name)
    baseline.write_jsonl(args.output / 'ordered_timeline.jsonl', new)
    baseline.write_jsonl(args.output / 'removed_duplicates.jsonl', reducer.removals)
    baseline.write_jsonl(args.output / 'context_decisions.jsonl', reducer.decisions)
    losses = json.loads((args.broken / 'independent_audit.json').read_text('utf-8'))['real_rows_lost']
    recovery = [{**{k:v for k,v in loss.items() if k in ('table_id','frame_id','text','old_row_id','current_hand','previous_hand')},
        'result': 'PRESERVED_REAL_ROW' if loss['old_row_id'] not in removed_ids else 'REAL_ROW_LOST'} for loss in losses]
    baseline.write_jsonl(args.output / 'recovered_real_rows.jsonl', recovery)
    prior_removed = {r['old_row_id'] for r in baseline.read_jsonl(args.broken / 'NEW/removed_duplicates.jsonl')}
    decisions = {r['row_id']: r for r in reducer.decisions}
    baseline.write_jsonl(args.output / 'previous_506_retained.jsonl', [
        {'row_id': rid, 'text': by_id[rid]['raw_ocr_text'], 'table_id': by_id[rid]['table_id'],
         'frame_id': by_id[rid]['frame_id'], 'decision': decisions[rid]} for rid in sorted(prior_removed-removed_ids)])
    summary = {'experiment': 'SPATIAL_HAND_CONTEXT_FENCE', 'old': str(args.old.resolve()),
        'broken': str(args.broken.resolve()), 'frames': stage.seen_frames,
        'physical_rows': len(physical), 'ocr_rows': len(physical), 'timeline_rows': len(new),
        'removed_count': len(removed_ids), 'previous_506_retained': len(prior_removed-removed_ids),
        'previous_removals_still_removed': len(prior_removed & removed_ids),
        'new_removal_ids': len(removed_ids-prior_removed), 'recovered_physical_rows': sum(r['result']=='PRESERVED_REAL_ROW' for r in recovery),
        'replay_wall_seconds': wall, 'dedup_seconds': reducer.seconds, 'visible_blocks': reducer.block_count,
        'started_at_moscow': started, 'finished_at_moscow': datetime.now(timezone(timedelta(hours=3))).isoformat(),
        'capture_emission_sequence_equal': emitted == [r['row_key'] for r in physical],
        'physical_pixel_identity_set_parity': True,
        'raw_evidence_sha_parity': {name: baseline.sha(args.old/name)==baseline.sha(args.output/name)
             for name in ('physical_rows.jsonl','raw_ocr_results.jsonl','visual_evidence.jsonl')},
        'recognition_scope': protocol['recognition_scope'], 'runtime_errors': [],
        'OCR_card_errors': old_summary['errors'], 'production_unchanged': True,
        'retained_timeline_records_order_equal': True, 'experiment_code_sha256': exp_pinned,
        'source_sha256': meta['source_video_sha256']}
    (args.output/'experiment_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
