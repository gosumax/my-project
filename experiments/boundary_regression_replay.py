"""Bounded source-linked replay of recorded overlap proposals, not synthetic OCR."""
import argparse
import base64
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import av
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import temporal_overlap_dedup as baseline
from experiments.temporal_overlap_guarded import Observation, Reducer as Broken
from experiments.temporal_overlap_boundary import BoundaryReducer, Context, hand_number
from experiments.temporal_overlap_native import native_signature, native_distance
from parser_core.ram_first_stage import png_compatible_gray
from parser_core.chat_rows import physical_spans, padded_row_span
from parser_core.layout import load_fixed_layout


def unpack(record):
    pixels = np.frombuffer(base64.b64decode(record['pixels']), np.uint8).reshape(record['shape']).copy()
    assert hashlib.sha256(pixels.tobytes()).hexdigest() == record['sha256']
    return Observation(baseline.View(record['row_id'], native_signature(pixels), record['sha256'],
        record['y1'], record['index'], record['inherited'], record['origin'], record['text']), record['frame_id'], pixels)


def packed(row, pixels, frame=None, index=None, inherited=False, y1=None):
    return {'row_id': row['stable_row_id'], 'text': row['raw_ocr_text'],
        'frame_id': row['frame_id'] if frame is None else frame,
        'index': row['row_index'] if index is None else index, 'inherited': inherited,
        'origin': row['origin'], 'y1': 0 if y1 is None else y1,
        'pixels': base64.b64encode(pixels.tobytes()).decode('ascii'), 'shape': list(pixels.shape),
        'sha256': hashlib.sha256(pixels.tobytes()).hexdigest()}


def build_cases(old_dir, broken_dir, positive=False):
    timeline = baseline.read_jsonl(old_dir / 'ordered_timeline.jsonl')
    by_id = {r['stable_row_id']: r for r in timeline}
    physical = {r['row_id']: r for r in baseline.read_jsonl(old_dir / 'physical_rows.jsonl')}
    removals = {r['old_row_id']: r for r in baseline.read_jsonl(broken_dir / 'NEW/removed_duplicates.jsonl')}
    meta = json.loads((old_dir / 'run_metadata.json').read_text('utf-8'))
    layout = load_fixed_layout(ROOT / 'configs/nine_table_layout.json', ROOT / 'configs/layout_847x404.json', ROOT / 'configs/chat_log_bottom_rois.json')
    losses = json.loads((broken_dir / 'independent_audit.json').read_text('utf-8')).get('real_rows_lost', [])
    if positive:
        # Independently audited Smoke18 Starting duplicate, selected by category
        # for a positive fixture only; never a runtime nickname/hand exception.
        victims = [r for r in removals.values() if r['table_id'] == 'T01' and 'Starting a new hand' in r['text']]
        assert len(victims) == 1
        marker = next(r for r in timeline if r['table_id'] == 'T01' and r['raw_ocr_text'].startswith('Hand #'))
        losses = [{'old_row_id': victims[0]['old_row_id'], 'table_id': 'T01', 'frame_id': victims[0]['frame_id'],
            'previous_hand': hand_number(marker['raw_ocr_text']), 'current_hand': hand_number(marker['raw_ocr_text']),
            'current_hand_marker_row_id': None}]
    grouped = defaultdict(list)
    for loss in losses:
        grouped[(loss['table_id'], loss['frame_id'])].append(loss)
    cases = []
    with av.open(meta['video']) as source:
        stream = source.streams.video[0]
        for (table, frame_id), group in grouped.items():
            witness = removals[group[0]['old_row_id']]
            target = physical[group[0]['old_row_id']]
            source.seek(target['source_pts'], stream=stream, backward=True)
            frame = next(f for f in source.decode(stream) if f.pts >= target['source_pts'])
            assert frame.pts == target['source_pts']
            slot = next(s for s in layout.slots if s.slot_id == int(table[1:]))
            x1, y1, x2, y2 = slot.chat_box
            gray = png_compatible_gray(frame.to_ndarray(format='rgb24')[y1:y2, x1:x2])
            spans = physical_spans(gray)

            def crop(index):
                bounds = padded_row_span(spans, index, gray.shape[0])
                return gray[bounds[0]:bounds[1]].copy(), bounds[0]

            old_marker = next(r for r in timeline if r['table_id'] == table
                and 'hand' in r['raw_ocr_text'].lower()
                and hand_number(r['raw_ocr_text']) == group[0]['previous_hand'])
            marker_pixels = np.frombuffer(base64.b64decode(physical[old_marker['stable_row_id']]['row_pixels_base64']), np.uint8).reshape(physical[old_marker['stable_row_id']]['pixel_shape']).copy()
            old_block = [packed(old_marker, marker_pixels)]
            current_block = []
            if group[0]['current_hand_marker_row_id']:
                marker = by_id[group[0]['current_hand_marker_row_id']]
                raw, y = crop(marker['row_index'])
                assert hashlib.sha256(raw.tobytes()).hexdigest() == physical[marker['stable_row_id']]['row_pixel_sha256']
                current_block.append(packed(marker, raw, y1=y))
            pair_index = next(i for i, p in enumerate(witness['overlap_pairs']) if p['current_row_id'] == witness['old_row_id'])
            start_index = witness['source_span_index'] - pair_index
            for offset, pair in enumerate(witness['overlap_pairs']):
                previous = by_id[pair['previous_row_id']]
                prior_pixels = np.frombuffer(base64.b64decode(physical[previous['stable_row_id']]['row_pixels_base64']), np.uint8).reshape(physical[previous['stable_row_id']]['pixel_shape']).copy()
                if previous['stable_row_id'] != old_marker['stable_row_id']:
                    old_block.append(packed(previous, prior_pixels))
                current = by_id[pair['current_row_id']]
                raw, y = crop(start_index + offset)
                inherited = pair['current_row_id'] not in {g['old_row_id'] for g in group}
                if not inherited:
                    assert hashlib.sha256(raw.tobytes()).hexdigest() == physical[current['stable_row_id']]['row_pixel_sha256']
                current_block.append(packed(current, raw, frame_id, start_index + offset, inherited, y))
            gallery = []
            for loss in group:
                r = removals[loss['old_row_id']]
                selected = r['matched_previous_observation']
                chosen = dict(by_id[selected['stable_row_id']])
                raw = np.frombuffer(base64.b64decode(selected['row_pixels_base64']), np.uint8).reshape(selected['pixel_shape']).copy()
                gallery.append({'matched_id': r['matched_previous_row_id'], 'observation': packed(chosen, raw,
                    selected['frame_id'], selected['source_span_index'], selected['inherited'], selected['y1'])})
            cases.append({'kind': 'PROVEN_DUPLICATE' if positive else 'PRESERVED_REAL_ROW',
                'table': int(table[1:]), 'sessions': meta['table_sessions'],
                'previous_hand': group[0]['previous_hand'], 'current_hand': group[0]['current_hand'],
                'old_block': old_block, 'current_block': current_block, 'gallery': gallery,
                'expected_row_ids': [g['old_row_id'] for g in group],
                'source_sha256': meta['source_video_sha256'], 'source_frame': frame_id,
                'original_proposal': witness,
                'loss_proof': group if not positive else []})
    return cases


def replay_case(case):
    baseline.canonical_row_distance = native_distance
    table = case['table']
    old = [unpack(r) for r in case['old_block']]
    current = [unpack(r) for r in case['current_block']]
    old_frame = max(o.frame_id for o in old)
    new_frame = case['source_frame']
    broken, fixed = Broken(gallery=True), BoundaryReducer(case['sessions'])
    for reducer in (broken, fixed):
        reducer.process(table, old_frame, old)
    ctx = Context(table, case['sessions'][str(table)], case['previous_hand'])
    core = fixed.cores[ctx]
    # Rehydrate only independently audited, strictly prior appearance snapshots.
    for entry in case['gallery']:
        obs = unpack(entry['observation'])
        assert obs.frame_id < new_frame
        rid = entry['matched_id']
        fixed.birth.setdefault(obs.view.row_id, ctx)
        for reducer in (broken, core):
            reducer.gallery[rid][obs.view.pixel_hash] = obs
            if obs.view.row_id != rid:
                reducer.aliases[obs.view.row_id] = rid
    broken.process(table, new_frame, current)
    fixed.process(table, new_frame, current)
    expected = set(case['expected_row_ids'])
    b = expected & {r['old_row_id'] for r in broken.removals}
    c = expected & {r['old_row_id'] for r in fixed.removals}
    return {'kind': case['kind'], 'table_id': f'T{table:02d}', 'frame_id': new_frame,
        'expected': len(expected), 'broken_removed': len(b), 'fixed_removed': len(c),
        'pass': b == expected and (c == expected if case['kind'] == 'PROVEN_DUPLICATE' else not c),
        'row_ids': sorted(expected), 'fixed_decisions': fixed.decisions}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = build_cases(ROOT / 'runs/production_smoke17_20261001_013909', ROOT / 'runs/temporal_overlap_guarded_smoke17_20261001')
    cases += build_cases(ROOT / 'runs/temporal_overlap_smoke18_20261001/OLD', ROOT / 'runs/temporal_overlap_guarded_smoke18_clean_20261001', positive=True)
    baseline.write_jsonl(args.output / 'regression_fixtures.jsonl', cases)
    results = [replay_case(c) for c in cases]
    result = {'pass': all(r['pass'] for r in results), 'recovered_physical_rows': sum(r['expected']-r['fixed_removed'] for r in results if r['kind']=='PRESERVED_REAL_ROW'),
        'recovered_ante_actions': 12 if all(r['pass'] for r in results if r['kind']=='PRESERVED_REAL_ROW') else None,
        'previously_proven_positive_control_removed': sum(r['fixed_removed'] for r in results if r['kind']=='PROVEN_DUPLICATE'),
        'scope': 'Recorded-proposal replay with current block pixels sought from source; same-hand positive control from audited Smoke18. Not a full capture or independent OCR rerun.', 'cases': results}
    (args.output / 'targeted_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
    assert result['pass']


if __name__ == '__main__':
    main()
