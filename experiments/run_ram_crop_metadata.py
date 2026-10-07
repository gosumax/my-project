"""Capture-only experiment using the pinned production FirstStage and saved OCR.

This does not launch/replace production; production launch remains run_production.py.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.capture_temporal_evidence import read, sha, write
from experiments.ram_crop_metadata import RamMetadata, watermark
from parser_core.ram_first_stage import FirstStage


def image_inventory():
    found = {}
    for parent in ('runs','outputs'):
        for folder,dirs,files in os.walk(ROOT/parent):
            for name in files:
                if Path(name).suffix.lower() not in {'.png','.jpg','.jpeg','.webp'}:
                    continue
                path = Path(folder)/name
                stat = path.stat()
                found[str(path.absolute())] = (stat.st_size,stat.st_mtime_ns)
    return found


def deep_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(deep_bytes(k,seen)+deep_bytes(v,seen) for k,v in value.items())
    elif isinstance(value, (tuple,list,set)):
        size += sum(deep_bytes(v,seen) for v in value)
    return size


def main():
    process_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',type=int)
    p.add_argument('--mode',choices=['baseline','metadata'],default='metadata')
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    experiment_hashes = {str(Path(__file__).relative_to(ROOT)):sha(__file__),
        'experiments/ram_crop_metadata.py':sha(ROOT/'experiments/ram_crop_metadata.py')}
    print(json.dumps(dict(phase='PREFLIGHT',mode=args.mode)),flush=True)
    baseline = ROOT/'runs/production_smoke17_20261001_013909'
    temporal = ROOT/'runs/temporal_context_smoke17_20261001'
    manifest = json.loads((ROOT/'production_manifest.json').read_text('utf-8'))
    pinned = {name: sha(ROOT/name) for name in manifest['hashes']}
    assert pinned == manifest['hashes'], 'Production drift'
    receipt = json.loads((temporal/'capture_summary.json').read_text('utf-8'))
    assert receipt['production_manifest_sha256'] == sha(ROOT/'production_manifest.json')
    baseline_hashes = {name: sha(baseline/name) for name in receipt['baseline_hashes']}
    assert baseline_hashes == receipt['baseline_hashes']
    meta = json.loads((baseline/'run_metadata.json').read_text('utf-8'))
    assert sha(meta['video']) == receipt['source_video_sha256']
    count = args.frames or meta['frames_requested']
    assert 0 < count <= meta['frames_requested']
    input_names = ['frame_times.jsonl','tracks.jsonl','ordered_timeline_temporal.jsonl',
                   'confirmed_starts.jsonl','persistent_previous_tails.jsonl','duplicate_analysis.jsonl']
    inputs = {name: sha(temporal/name) for name in input_names}
    times = read(temporal/'frame_times.jsonl')[:count]
    physical = read(baseline/'physical_rows.jsonl')
    expected = {r['row_key']:r for r in physical if r['frame_id'] < count}
    del physical
    saved_tracks = {t['track_id']:t for t in read(temporal/'tracks.jsonl')}
    saved_rows = {r['stable_row_id']:r for r in read(temporal/'ordered_timeline_temporal.jsonl')}
    print(json.dumps(dict(phase='INPUTS_VERIFIED',seconds=time.perf_counter()-process_started)),flush=True)
    inventory = image_inventory()
    args.output.mkdir(parents=True,exist_ok=False)
    layer = RamMetadata() if args.mode == 'metadata' else None
    process = psutil.Process()
    gc.collect()
    rss_start = process.memory_info().rss
    rss_peak = [rss_start]
    metadata_peak = [0]
    stop = threading.Event()
    def sample():
        while not stop.wait(.1):
            rss_peak[0] = max(rss_peak[0],process.memory_info().rss)
    sampler = threading.Thread(target=sample,daemon=True)
    sampler.start()
    last_progress = [time.perf_counter()]
    class Clock:
        def add(self,name,seconds,objects):
            if name == 'frame_to_ndarray':
                fid = stage.seen_frames
                if layer is not None and fid:
                    complete(fid-1)
                    if fid % 1000 == 0:
                        metadata_peak[0] = max(metadata_peak[0], deep_bytes(layer.tracks)+
                            deep_bytes(layer.active)+deep_bytes(layer.pending))
                if time.perf_counter()-last_progress[0] > 30:
                    print(json.dumps(dict(frames=fid,rows=len(seen),crops=layer.crops if layer else None,
                        wall_seconds=time.perf_counter()-started)),flush=True)
                    last_progress[0] = time.perf_counter()
    def complete(fid):
        changed = [sid for sid in meta['table_sessions'].values() if (fid,sid,'chat') in stage.changed]
        layer.complete_frame(fid,times[fid],changed)
    def observe(m,pixels):
        with layer.borrow(m,pixels,times[m['frame_id']]) as view:
            assert view.image is pixels
    stage = FirstStage(Path(meta['video']),meta['table_sessions'],count,root=ROOT,
                       telemetry=Clock(),identity_observer=observe if layer else None)
    seen = []
    io_start = process.io_counters().write_bytes
    started = time.perf_counter()
    preflight_seconds = started-process_started
    print(json.dumps(dict(phase='CAPTURE_START',frames=count,preflight_seconds=preflight_seconds)),flush=True)
    try:
        for row in stage.rows():
            key = f'{row.table_id}:{row.frame_id}:{row.source_span_index}'
            saved = expected[key]
            assert (row.stable_row_id,row.pixel_sha256,row.y1,row.origin,row.epoch,
                    row.source_pts,row.time_base_num,row.time_base_den) == (
                saved['row_id'],saved['row_pixel_sha256'],saved['y1'],saved['origin'],saved['epoch'],
                saved['source_pts'],saved['time_base_num'],saved['time_base_den'])
            assert row.pixels.tobytes() == base64.b64decode(saved['row_pixels_base64'])
            seen.append(key)
            if layer is not None:
                observe(dict(frame_id=row.frame_id,table_id=row.table_id,session_id=row.session_id,
                    stable_row_id=row.stable_row_id,source_pts=row.source_pts,
                    source_span_index=row.source_span_index,y1=row.y1,origin=row.origin),row.pixels)
            del row
        if layer is not None:
            complete(stage.seen_frames-1)
            layer.finish()
    finally:
        stop.set(); sampler.join()
    elapsed = time.perf_counter()-started
    capture_write_bytes = process.io_counters().write_bytes-io_start
    assert stage.seen_frames == count and len(seen) == len(set(seen)) == len(expected)
    def capture_order(key):
        table,frame,span = map(int,key.split(':'))
        return frame,table,span
    assert seen == sorted(seen,key=capture_order), 'Capture chronology changed'
    assert sorted(seen,key=lambda key:tuple(map(int,key.split(':')))) == list(expected), 'Export order changed'
    result = dict(mode=args.mode,frames=count,physical_rows=len(seen),wall_seconds=elapsed,
        preflight_seconds=preflight_seconds,
        capture_row_order_sha256=hashlib.sha256('\n'.join(seen).encode()).hexdigest(),
        rss_start_bytes=rss_start,rss_peak_bytes=rss_peak[0],rss_growth_bytes=rss_peak[0]-rss_start,
        capture_process_write_bytes=capture_write_bytes,
        source_video_sha256=receipt['source_video_sha256'],baseline_hashes=baseline_hashes,
        temporal_input_hashes=inputs,production_manifest_sha256=sha(ROOT/'production_manifest.json'),
        experiment_hashes=experiment_hashes,
        exact_row_pixels_hash_identity_pts_order_parity=True,production_code_changed=False,
        ocr_runs=0,measurement_scope='capture-only; saved OCR; RSS sampled at 100ms; no export/OCR workers')
    if layer is not None:
        fields = ['first_seen_frame','first_seen_ms','first_seen_row_index','last_seen_frame','last_seen_ms',
                  'visible_spans','observation_count','explicit_view_count','visibility_gap_count',
                  'left_censored','right_censored']
        if count == meta['frames_requested']:
            assert set(layer.tracks) == set(saved_tracks)
            for rid,t in layer.tracks.items():
                assert all(t[k] == saved_tracks[rid][k] for k in fields), (rid,t,saved_tracks[rid])
                r = saved_rows[rid]
                # Reuse the frozen existing analysis label and its boundary context;
                # this is cached analysis, not a new boundary inference.
                t.update(ocr_text=r['raw_ocr_text'],ocr_confidence=r['confidence'],
                    normalized_text=r['normalized_text'],temporal_class=r['temporal_class'],
                    temporal_reason=r['temporal_reason'],analysis_source_sha256=inputs['ordered_timeline_temporal.jsonl'],
                    source_row_key=r['row_key'],source_pixel_hash=r['row_pixel_sha256'],
                    source_frame=r['frame_id'],source_timestamp_ms=r['timestamp_ms'],
                    confirmed_start_row_id=r['confirmed_start_row_id'],
                    next_confirmed_start_row_id=r['next_confirmed_start_row_id'])
            result['exact_lifetime_parity_tracks'] = len(layer.tracks)
            review=[]
            for label in ('PREVIOUS_HAND_TAIL','CURRENT_HAND','AMBIGUOUS'):
                candidates = [t for t in layer.tracks.values() if t['temporal_class']==label]
                for t in candidates[:20]:
                    review.append(dict(t,watermark=watermark(t)))
            assert len(review)==60
            write(args.output/'temporal_review.jsonl',review)
            tails=read(temporal/'persistent_previous_tails.jsonl')
            duplicates=read(temporal/'duplicate_analysis.jsonl')
            result['edge_gate_usefulness']=dict(class_counts=dict(Counter(t['temporal_class'] for t in layer.tracks.values())),
                persistent_previous_track_boundary_pairs=len(tails),
                repeated_observations=sum(t['observation_count']-1 for t in layer.tracks.values()),
                repeated_explicit_views=layer.crops-len(layer.tracks),
                unresolved_identity_tracks=sum(t['origin']=='UNRESOLVED_IDENTITY' for t in layer.tracks.values()),
                visibility_gap_tracks=sum(t['visibility_gap_count']>0 for t in layer.tracks.values()),
                cross_track_candidate_pairs=len(duplicates),
                cross_track_ambiguous_pairs=sum(d['result']=='AMBIGUOUS_CROSS_TRACK_IDENTITY' for d in duplicates),
                decisions_changed=0,review_rows=len(review))
        metadata_bytes = deep_bytes(layer.tracks)+deep_bytes(layer.active)+deep_bytes(layer.pending)
        metadata_peak[0] = max(metadata_peak[0],metadata_bytes)
        result.update(crops=layer.crops,metadata_objects=len(layer.tracks),
            max_layer_image_objects=layer.peak_images,max_layer_image_payload_bytes=layer.peak_image_bytes,
            live_layer_images_at_end=layer.live_images,layer_image_cache_bytes=0,
            final_metadata_deep_bytes=metadata_bytes,
            sampled_peak_metadata_deep_bytes=metadata_peak[0],
            metadata_peak_scope='Python deep-size every 1000 frames and final; not RSS or exact allocation peak',
            lifecycle='borrow synchronously -> hash/metadata -> release; no image cache',
            crop_scope='physical grayscale rows, inherited changed-state views; unchanged ROI frames extend metadata only')
        write(args.output/'tracks.jsonl',layer.tracks.values())
    assert pinned == {name:sha(ROOT/name) for name in pinned}
    assert experiment_hashes == {name:sha(ROOT/name) for name in experiment_hashes}
    assert baseline_hashes == {name:sha(baseline/name) for name in baseline_hashes}
    assert inputs == {name:sha(temporal/name) for name in inputs}
    after=image_inventory()
    created=set(after)-set(inventory)
    modified={name for name in set(after)&set(inventory) if after[name]!=inventory[name]}
    image_counts=Counter(Path(name).suffix.lower() for name in created)
    result.update(new_png_files=image_counts['.png'],new_jpg_files=image_counts['.jpg']+image_counts['.jpeg'],
        new_webp_files=image_counts['.webp'],modified_image_files=len(modified),
        filesystem_audit_scope='all runs/ and outputs/ before/after; existing files excluded by full path',
        image_files_created=len(created),text_output_bytes=sum(f.stat().st_size for f in args.output.iterdir()))
    (args.output/'image_inventory_audit.json').write_text(json.dumps(dict(before_count=len(inventory),
        after_count=len(after),new_files=sorted(created),modified_files=sorted(modified)),indent=2),'utf-8')
    assert not created and not modified, 'Image filesystem audit FAIL'
    result['total_process_seconds_before_summary'] = time.perf_counter()-process_started
    (args.output/'summary.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps(result),flush=True)

if __name__ == '__main__':
    main()
