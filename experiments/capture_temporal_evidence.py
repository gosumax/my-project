"""Evidence-only replay of manifest-pinned FirstStage. No OCR or row filtering.

Track identity is the EXISTING stable_row_id. A changed viewport is observed
through its existing copy-only observer. Unchanged ROI frames inherit the last
observed viewport, bounded by the next chat change (including empty states).
First seen means first visible observation, never the physical server onset.
"""
from __future__ import annotations
import argparse
import base64
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time
import av
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.ram_first_stage import FirstStage


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, rows):
    with Path(path).open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def source_ms(pts, num, den):
    if pts is None or num is None or den is None:
        return None
    return float(Fraction(pts * num * 1000, den))


def timestamp_text(ms):
    if ms is None:
        return None
    value = round(ms)
    sign = '-' if value < 0 else ''
    value = abs(value)
    hours, value = divmod(value, 3600000)
    minutes, value = divmod(value, 60000)
    seconds, millis = divmod(value, 1000)
    return f'{sign}{hours:02}:{minutes:02}:{seconds:02}.{millis:03}'


def lifetime(states, changes, frame_times, count):
    """Retain disjoint visibility spans; never interpolate across changed ROIs."""
    tracks = {}
    for session, frames in changes.items():
        frames = sorted(frames)
        for index, frame in enumerate(frames):
            end = frames[index+1]-1 if index+1 < len(frames) else count-1
            for view in states.get((session, frame), []):
                rid = view['track_id']
                track = tracks.setdefault(rid, {'track_id': rid, 'session_id': session,
                    'table_id': view['table_id'], 'first_seen_frame': frame,
                    'first_seen_ms': frame_times[frame]['timestamp_ms'],
                    'first_seen_row_index': view['row_index'], 'origin': view['origin'],
                    'visible_spans': [], 'observation_count': 0, 'explicit_view_count': 0})
                spans = track['visible_spans']
                if spans and spans[-1][1]+1 == frame:
                    spans[-1][1] = end
                else:
                    spans.append([frame, end])
                track['last_seen_frame'] = end
                track['last_seen_ms'] = frame_times[end]['timestamp_ms']
                track['observation_count'] += end-frame+1
                track['explicit_view_count'] += 1
                track['visibility_gap_count'] = len(spans)-1
                track['first_seen_kind'] = 'FIRST_VISIBLE_OBSERVATION_NOT_SERVER_ONSET'
                track['left_censored'] = track['origin'] != 'NEW_CANDIDATE'
                track['right_censored'] = end == count-1
    return tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT/'production_manifest.json').read_text('utf-8'))
    pinned = {p: sha(ROOT/p) for p in manifest['hashes']}
    assert pinned == manifest['hashes'], 'Production hashes must match before replay'
    metadata = json.loads((args.baseline/'run_metadata.json').read_text('utf-8'))
    summary = json.loads((args.baseline/'final_summary.json').read_text('utf-8'))
    assert metadata['production_manifest_sha256'] == sha(ROOT/'production_manifest.json')
    assert metadata['production_modified_files'] == []
    count = metadata['frames_requested']
    assert summary['frames'] == count
    video = Path(metadata['video'])
    assert sha(video) == metadata['source_video_sha256']
    physical = read(args.baseline/'physical_rows.jsonl')
    expected = {r['row_key']: r for r in physical}
    assert len(expected) == len(physical)
    states = defaultdict(list)
    frame_times = []
    start = time.perf_counter()
    # Decode PTS metadata, without converting/writing frames or using fps.
    with av.open(str(video)) as source:
        stream = source.streams.video[0]
        for fid, frame in enumerate(source.decode(stream)):
            if fid >= count:
                break
            base = frame.time_base or stream.time_base
            frame_times.append({'source_frame': fid, 'source_pts': frame.pts,
                'time_base_num': base.numerator, 'time_base_den': base.denominator,
                'timestamp_ms': source_ms(frame.pts, base.numerator, base.denominator)})
    assert len(frame_times) == count
    write(args.output/'frame_times.jsonl', frame_times)
    print(json.dumps({'phase': 'PTS_COMPLETE', 'frames': count,
                      'wall_seconds': time.perf_counter()-start}), flush=True)
    observations = (args.output/'track_views.jsonl').open('w', encoding='utf-8')
    last_pixels = {}
    view_count = 0

    def record(meta, pixels, inherited):
        nonlocal view_count
        fid = meta['frame_id']
        rid = meta['stable_row_id']
        stamp = frame_times[fid]
        assert meta['source_pts'] == stamp['source_pts']
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        prior = last_pixels.get(rid)
        similarity = (1-float(np.mean(np.abs(pixels.astype(np.int16)-prior.astype(np.int16))))/255
                      if prior is not None and prior.shape == pixels.shape else None)
        last_pixels[rid] = pixels.copy()
        view = {'track_id': rid, 'table_id': f"T{meta['table_id']:02}",
            'session_id': meta['session_id'], 'source_frame': fid,
            'source_pts': stamp['source_pts'], 'timestamp_ms': stamp['timestamp_ms'],
            'row_index': meta['source_span_index'], 'y1': meta['y1'],
            'origin': meta['origin'], 'pixel_hash': digest,
            'pixel_similarity_prev': similarity, 'inherited': inherited}
        states[(meta['session_id'], fid)].append(view)
        observations.write(json.dumps(view, ensure_ascii=False, allow_nan=False)+'\n')
        view_count += 1

    def observe(meta, pixels):
        record(meta, pixels, True)

    stage = FirstStage(video, metadata['table_sessions'], count, root=ROOT,
                       identity_observer=observe)
    seen = set()
    last_progress = time.perf_counter()
    try:
        for row in stage.rows():
            key = f'{row.table_id}:{row.frame_id}:{row.source_span_index}'
            saved = expected[key]
            assert key not in seen
            assert (row.stable_row_id, row.pixel_sha256, row.y1, row.origin, row.epoch,
                    row.source_pts, row.time_base_num, row.time_base_den) == (
                saved['row_id'], saved['row_pixel_sha256'], saved['y1'], saved['origin'],
                saved['epoch'], saved['source_pts'], saved['time_base_num'], saved['time_base_den'])
            assert row.pixels.tobytes() == base64.b64decode(saved['row_pixels_base64'])
            seen.add(key)
            record({'frame_id': row.frame_id, 'table_id': row.table_id,
                'session_id': row.session_id, 'source_span_index': row.source_span_index,
                'stable_row_id': row.stable_row_id, 'y1': row.y1, 'origin': row.origin,
                'source_pts': row.source_pts}, row.pixels, False)
            if time.perf_counter()-last_progress > 30:
                progress = {'phase': 'CAPTURE_REPLAY', 'frames': stage.seen_frames,
                    'rows': len(seen), 'views': view_count, 'wall_seconds': time.perf_counter()-start}
                print(json.dumps(progress), flush=True)
                (args.output/'progress.json').write_text(json.dumps(progress), 'utf-8')
                last_progress = time.perf_counter()
    finally:
        observations.close()
    assert stage.seen_frames == count and seen == set(expected)
    changes = defaultdict(list)
    for fid, session, roi in stage.changed:
        if roi == 'chat':
            changes[session].append(fid)
    tracks = lifetime(states, changes, frame_times, count)
    assert set(tracks) == {r['row_id'] for r in physical}
    write(args.output/'tracks.jsonl', tracks.values())
    write(args.output/'chat_states.jsonl', ({'session_id': sid, 'source_frame': fid,
        'track_ids': [v['track_id'] for v in states.get((sid, fid), [])]}
        for sid in sorted(changes) for fid in sorted(changes[sid])))
    assert pinned == {p: sha(ROOT/p) for p in pinned}
    result = {'baseline': str(args.baseline.resolve()), 'frames': count, 'physical_rows': len(seen),
        'explicit_views': view_count, 'tracks': len(tracks), 'chat_changed_states': sum(map(len, changes.values())),
        'wall_seconds': time.perf_counter()-start, 'pixel_identity_pts_set_parity': True,
        'production_unchanged': True, 'source_video_sha256': metadata['source_video_sha256'],
        'production_manifest_sha256': sha(ROOT/'production_manifest.json'),
        'baseline_hashes': {name: sha(args.baseline/name) for name in
            ('physical_rows.jsonl', 'raw_ocr_results.jsonl', 'ordered_timeline.jsonl', 'visual_evidence.jsonl')},
        'experiment_sha256': sha(Path(__file__)),
        'observation_count_definition': 'decoded frames visible; explicit changed states plus proven unchanged ROI frames',
        'limitation': 'Existing tracker IDs; no cross-ID equivalence asserted; excluded first visible row has no track'}
    (args.output/'capture_summary.json').write_text(json.dumps(result, indent=2), 'utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
