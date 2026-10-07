"""Causal RAM-only content measurements; no classifier or production mutations."""
from __future__ import annotations
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import threading
import time
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.row_content_evidence import compare_rows
from experiments.capture_temporal_evidence import read, sha, write
from experiments.run_cross_track_linker import Frames
from experiments.run_ram_crop_metadata import image_inventory
from experiments.cross_track_linker import digest
from parser_core.chat_rows import physical_spans, padded_row_span
from parser_core.layout import load_fixed_layout


def support_equal(score):
    return bool(score['foreground_union']['pixels'] and score['support_xor_pixels'] == 0
                and not score['missing_foreground_before'] and not score['missing_foreground_after'])


def compact(value):
    return {k: v for k, v in value.items() if k != 'alignment_candidates'}


def quantiles(values):
    values = [v for v in values if v is not None]
    return dict(zip(('min', 'p10', 'p25', 'median', 'p75', 'p90', 'max'),
                    np.quantile(values, (0, .1, .25, .5, .75, .9, 1)).tolist())) if values else None


def distributions(rows):
    result = {}
    for label, selected in (('all', rows), ('mapped_99', [r for r in rows if r['old_mapping_present']]),
                             ('unmapped_491', [r for r in rows if not r['old_mapping_present']])):
        result[label] = {'count': len(selected)}
        for alignment in ('before_alignment', 'best_alignment'):
            for region in ('foreground_union', 'content_region', 'outside_content', 'dark_background', 'edge_diff', 'full_crop'):
                result[label][alignment+'_'+region+'_mae'] = quantiles([r['target'][alignment][region]['mae'] for r in selected])
        result[label]['original_full_crop_mae'] = quantiles([r['target']['original_full_crop']['mae'] for r in selected])
        result[label]['support_xor_ratio'] = quantiles([r['support_xor_ratio'] for r in selected])
        result[label]['support_equal'] = sum(r['target_support_equal_after_alignment'] for r in selected)
        result[label]['exact_intensity_support'] = sum(r['target']['best_alignment']['exact_support_equal_after_alignment'] for r in selected)
        result[label]['exact_native_content'] = sum(r['target']['exact_content_native'] for r in selected)
        result[label]['content_candidate_count'] = dict(Counter(str(len(r['content_candidates'])) for r in selected))
        result[label]['envelope'] = dict(Counter(r['within_causal_neighbor_control_envelope'] for r in selected))
        result[label]['bbox_trim_different'] = sum(r['target']['before']['trims_left_top_right_bottom'] != r['target']['after']['trims_left_top_right_bottom'] for r in selected)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    temporal = ROOT/'runs/temporal_context_smoke17_20261001'
    baseline = ROOT/'runs/production_smoke17_20261001_013909'
    relation_path = ROOT/'runs/cross_track_linker_smoke17_20261001/relationships.jsonl'
    neighbors_path = ROOT/'runs/cross_track_condition_audit_smoke17_20261001/independent_neighbors.jsonl'
    manifest = json.loads((ROOT/'production_manifest.json').read_text('utf-8'))
    source = json.loads((baseline/'run_metadata.json').read_text('utf-8'))
    paths = [ROOT/p for p in manifest['hashes']] + [ROOT/'production_manifest.json', relation_path, neighbors_path,
        temporal/'chat_states.jsonl', temporal/'frame_times.jsonl', baseline/'run_metadata.json',
        ROOT/'experiments/cross_track_linker.py', ROOT/'experiments/run_cross_track_linker.py',
        ROOT/'experiments/row_content_evidence.py', Path(__file__), Path(source['video'])]
    print(json.dumps({'phase': 'HASH_IMAGE_PREFLIGHT'}), flush=True)
    hashes = {str(p): sha(p) for p in paths}
    assert all(hashes[str(ROOT/p)] == expected for p, expected in manifest['hashes'].items())
    assert hashes[str(Path(source['video']))] == source['source_video_sha256']
    inventory = image_inventory()
    records = read(relation_path)
    assert len(records) == 590
    independent = {r['pair_id']: r for r in read(neighbors_path)}
    states = defaultdict(dict)
    for r in read(temporal/'chat_states.jsonl'):
        states[r['session_id']][r['source_frame']] = r['track_ids']
    state_frames = {sid: sorted(values) for sid, values in states.items()}
    def state(sid, fid):
        frames = state_frames[sid]
        i = bisect_right(frames, fid)-1
        assert i >= 0
        return states[sid][frames[i]]
    times = read(temporal/'frame_times.jsonl')
    layout = load_fixed_layout(ROOT/'configs/nine_table_layout.json', ROOT/'configs/layout_847x404.json', ROOT/'configs/chat_log_bottom_rois.json')
    process = psutil.Process()
    peak = [process.memory_info().rss]
    rss_start = peak[0]
    stop = threading.Event()
    def sample():
        while not stop.wait(.05):
            peak[0] = max(peak[0], process.memory_info().rss)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    reader = Frames(Path(source['video']), times, layout)
    results = []
    decode_seconds = compare_seconds = 0.0
    last = time.perf_counter()
    try:
        for i, r in enumerate(records):
            old, new, motion = r['old'], r['new'], r['motion']
            assert new['first_frame'] == r['decision_frame']
            assert old['last_frame'] <= r['decision_frame']
            tick = time.perf_counter()
            before = reader.get(old['last_frame'], old['table_id'])
            after = reader.get(new['first_frame'], new['table_id'])
            decode_seconds += time.perf_counter()-tick
            tick = time.perf_counter()
            assert (digest(before), digest(after)) == (motion['before_hash'], motion['after_hash'])
            osp, nsp = physical_spans(before), physical_spans(after)
            oi, ni = old['row_index']-1, new['row_index']-1
            old_ids, new_ids = state(old['session_id'], old['last_frame']), state(new['session_id'], new['first_frame'])
            assert old_ids[oi] == old['track_id'] and new_ids[ni] == new['track_id']
            assert len(old_ids) == len(osp)-1 and len(new_ids) == len(nsp)-1
            def crop(gray, spans, index):
                top, bottom = padded_row_span(spans, index+1, gray.shape[0])
                return gray[top:bottom]
            old_crop = crop(before, osp, oi)
            new_crop = crop(after, nsp, ni)
            assert digest(old_crop) == old['pixel_hash'] and digest(new_crop) == new['pixel_hash']
            target = compare_rows(old_crop, new_crop)
            foreground = target['best_alignment']['foreground_union']['pixels']
            xor_ratio = target['best_alignment']['support_xor_pixels']/foreground if foreground else None
            neighbors = []
            for n in independent[r['pair_id']]['neighbors']:
                if n['center_residual'] > 1:
                    continue
                result = compact(compare_rows(crop(before, osp, n['old_index']), crop(after, nsp, n['new_index'])))
                score = result['best_alignment']
                count = score['foreground_union']['pixels']
                neighbors.append(dict(old_index=n['old_index'], new_index=n['new_index'], old_id=n['old_id'], new_id=n['new_id'],
                    same_id=n['old_id'] == n['new_id'], center_residual=n['center_residual'],
                    support_equal=support_equal(score), support_xor_ratio=score['support_xor_pixels']/count if count else None,
                    measurements=result))
            controls = [n for n in neighbors if n['same_id'] and n['support_xor_ratio'] is not None
                        and not n['measurements']['best_alignment']['missing_foreground_before']
                        and not n['measurements']['best_alignment']['missing_foreground_after']]
            max_mae = max((n['measurements']['best_alignment']['foreground_union']['mae'] for n in controls), default=None)
            max_xor = max((n['support_xor_ratio'] for n in controls), default=None)
            ts = target['best_alignment']
            within = 'UNKNOWN' if len(controls) < 2 or xor_ratio is None else 'PASS' if (
                ts['foreground_union']['mae'] <= max_mae and xor_ratio <= max_xor
                and not ts['missing_foreground_before'] and not ts['missing_foreground_after']) else 'FAIL'
            eligible = []
            candidates = []
            if motion['aligned']:
                for index, span in enumerate(nsp[1:]):
                    residual = abs(sum(osp[oi+1])/2-sum(span)/2-motion['shift_px'])
                    if residual > 7:
                        continue
                    measurements = compact(target) if index == ni else compact(compare_rows(old_crop, crop(after, nsp, index)))
                    match = support_equal(measurements['best_alignment'])
                    entry = dict(new_index=index, new_id=new_ids[index], center_residual=residual,
                                 is_intended_candidate=index == ni, support_equal=match, measurements=measurements)
                    eligible.append(entry)
                    if match:
                        candidates.append(dict(new_index=index, new_id=new_ids[index], is_intended_candidate=index == ni))
            context = bool(old['table_id'] == new['table_id'] and old['session_id'] == new['session_id']
                           and not r['features']['causal_visibility_overlap'])
            all_geometry = [dict(new_index=index, new_id=new_ids[index], tight_span=list(span),
                tight_center=sum(span)/2,
                center_residual=abs(sum(osp[oi+1])/2-sum(span)/2-motion['shift_px']))
                for index, span in enumerate(nsp[1:])]
            status = 'NO_ALIGNMENT' if not motion['aligned'] else 'CONTEXT_CONFLICT' if not context else (
                'CONTENT_CANDIDATE_UNIQUE' if len(candidates) == 1 else 'CONTENT_CANDIDATE_MULTIPLE' if candidates else 'CONTENT_CANDIDATE_NONE')
            results.append(dict(pair_id=r['pair_id'], decision_frame=r['decision_frame'],
                endpoint_frames=[old['last_frame'], new['first_frame']], old_track_id=old['track_id'], new_track_id=new['track_id'],
                old_mapping_present=motion['target_new_index'] is not None, original_verdict=r['link_status'],
                same_context_without_overlap=context, aligned=motion['aligned'], shift_px=motion['shift_px'],
                old_target_tight_center=sum(osp[oi+1])/2, all_new_row_geometry=all_geometry,
                target=target, target_support_equal_after_alignment=support_equal(ts), support_xor_ratio=xor_ratio,
                geometry_eligible_rows=eligible, content_candidates=candidates, content_mapping_status=status,
                content_neighbor_count=sum(n['support_equal'] for n in neighbors), coherent_neighbors=neighbors,
                causal_neighbor_control_count=len(controls), control_max_foreground_mae=max_mae, control_max_support_xor_ratio=max_xor,
                within_causal_neighbor_control_envelope=within, normalized_text_equal=old['normalized_text']==new['normalized_text'],
                causal_immediate_link=r['features']['delta_frame']==1 or motion['local_loss_bridge'],
                boundary_frames=r['features']['hand_boundary_frames'], classification_applied=False))
            compare_seconds += time.perf_counter()-tick
            del old_crop, new_crop, before, after
            if time.perf_counter()-last > 25:
                print(json.dumps(dict(phase='CAUSAL_CONTENT', pairs=i+1, decode_seconds=decode_seconds, comparison_seconds=compare_seconds)), flush=True)
                last = time.perf_counter()
    finally:
        end_images = reader.close()
        stop.set()
        sampler.join()
    print(json.dumps({'phase': 'HASH_IMAGE_POSTFLIGHT'}), flush=True)
    assert hashes == {str(p): sha(p) for p in paths}
    assert image_inventory() == inventory
    summary = dict(pairs=len(results), source_hashes=hashes, hashes_unchanged=True, image_writes=0,
        image_inventory_count=len(inventory), rss_start_bytes=rss_start, rss_peak_bytes=peak[0],
        decode_seconds=decode_seconds, comparison_seconds=compare_seconds,
        target_pairs_per_comparison_second=len(results)/compare_seconds,
        total_seconds=time.perf_counter()-started, roi_requests=reader.requests, decoded_seek_frames=reader.decoded,
        roi_peak_images=reader.peak_images, roi_peak_bytes=reader.peak_bytes, roi_images_end=end_images,
        classification_applied=False, content_mapping_status=dict(Counter(r['content_mapping_status'] for r in results)),
        content_neighbor_counts=dict(Counter(str(r['content_neighbor_count']) for r in results)),
        distributions=distributions(results), edge_gate_unchanged='6685 -> 6685 UNRESOLVED_EDGE')
    write(args.output/'pairs.jsonl', results)
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')
    print(json.dumps({'phase':'DONE', 'pairs':len(results), 'content_mapping_status':summary['content_mapping_status'], 'total_seconds':summary['total_seconds']}), flush=True)


if __name__ == '__main__':
    main()
