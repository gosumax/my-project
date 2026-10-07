"""Read-only bounded diagnosis: publication order, native views and identity gallery.

No deletion and no new dedup output. Explicit target pairs are diagnostic probes,
never matching rules. Uses unchanged golden FirstStage and its copy-only observer.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.temporal_overlap_dedup import read_jsonl, sha, signature
from parser_core.chat_rows import canonical_row_distance
from parser_core.ram_first_stage import FirstStage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, default=1197)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Fresh diagnostic output required')
    manifest = json.loads((ROOT / 'production_manifest.json').read_text('utf-8'))
    pinned = {k: sha(ROOT / k) for k in manifest['hashes']}
    assert pinned == manifest['hashes']
    m = json.loads((args.old / 'run_metadata.json').read_text('utf-8'))
    video = Path(m['video'])
    assert sha(video) == manifest['golden_benchmark']['source_sha256']
    timeline = read_jsonl(args.old / 'ordered_timeline.jsonl')
    physical = read_jsonl(args.old / 'physical_rows.jsonl')
    facts = {r['row_id']: r for r in physical}
    byid = {r['stable_row_id']: r for r in timeline}
    t1 = [r for r in timeline if r['table_id'] == 'T01']
    positions = {r['stable_row_id']: i for i, r in enumerate(t1)}
    # Four user-supplied known duplicate pairs, for diagnosis only.
    suffix_pairs = {'R000019': 'R000002', 'R000025': 'R000009',
                    'R000026': 'R000022', 'R000030': 'R000016'}
    targets = {rid: rid.rsplit(':', 1)[0] + ':' + suffix_pairs[rid.rsplit(':', 1)[-1]]
               for rid in positions if rid.rsplit(':', 1)[-1] in suffix_pairs}
    gallery = defaultdict(list)
    blocks = defaultdict(list)
    probes = []

    def record(frame, rid, index, y, pixels, inherited):
        if rid not in positions:
            return
        sig = signature(pixels)
        item = {'frame_id': frame, 'row_id': rid, 'index': index, 'y1': y,
                'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                'inherited': inherited, 'signature': sig}
        if frame in (196, 898, 1196):
            blocks[frame].append(item)
        if rid in targets and not inherited:
            previous_id = targets[rid]
            seen = [v for v in gallery[previous_id] if v['frame_id'] < frame]
            distances = [(canonical_row_distance(v['signature'], sig), v) for v in seen]
            first = facts[previous_id]
            first_pixels = np.frombuffer(base64.b64decode(first['row_pixels_base64']), np.uint8).reshape(first['pixel_shape'])
            canonical_first = canonical_row_distance(signature(first_pixels), sig)
            best_d, best = min(distances, key=lambda x: x[0]) if distances else (float('inf'), None)
            native_a = first_pixels[:, :-12].astype(np.float32)
            native_b = pixels[:, :-12].astype(np.float32)
            native_ncc = float(cv2.matchTemplate(native_a, native_b, cv2.TM_CCOEFF_NORMED)[0, 0]) if native_a.shape == native_b.shape else None
            probes.append({'current_row_id': rid, 'previous_row_id': previous_id,
                'frame_id': frame, 'first_view_canonical_mae': canonical_first,
                'previous_views': len(seen), 'best_previous_view_mae': best_d,
                'best_previous_view_frame': best['frame_id'] if best else None,
                'latest_previous_view_mae': distances[-1][0] if distances else None,
                'latest_previous_view_frame': seen[-1]['frame_id'] if seen else None,
                'first_view_native_ncc': native_ncc})
        gallery[rid].append(item)

    def observer(meta, pixels):
        if meta['table_id'] == 1:
            record(meta['frame_id'], meta['stable_row_id'], meta['source_span_index'], meta['y1'], pixels, True)

    start = time.perf_counter()
    stage = FirstStage(video, m['table_sessions'], args.frames, root=ROOT, identity_observer=observer)
    emitted = 0
    expected = {r['row_key']: r for r in physical if r['frame_id'] < args.frames}
    for row in stage.rows():
        saved = expected[f'{row.table_id}:{row.frame_id}:{row.source_span_index}']
        assert row.pixel_sha256 == saved['row_pixel_sha256'] and row.stable_row_id == saved['row_id']
        emitted += 1
        if row.table_id == 1:
            record(row.frame_id, row.stable_row_id, row.source_span_index, row.y1, row.pixels, False)
    diagnostics = []
    for frame, block in sorted(blocks.items()):
        ordered = sorted(block, key=lambda v: v['index'])
        anchors = [{'row_id': v['row_id'], 'visible_index': i, 'source_span_index': v['index'],
                    'published_history_position': positions[v['row_id']],
                    'offset': positions[v['row_id']] - i} for i, v in enumerate(ordered) if v['inherited']]
        diagnostics.append({'frame_id': frame, 'visible_order': [v['row_id'] for v in ordered],
                            'anchor_offsets': dict(Counter(v['offset'] for v in anchors)), 'anchors': anchors})
    assert emitted == len(expected) and stage.seen_frames == args.frames
    assert pinned == {k: sha(ROOT / k) for k in pinned}
    result = {'scope': 'Read-only diagnostic, not an A/B acceptance run',
              'frames': stage.seen_frames, 'rows': emitted, 'pixel_identity_parity': True,
              'wall_seconds': time.perf_counter() - start, 'production_unchanged': True,
              'target_pairs': probes, 'target_visible_blocks': diagnostics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'target_visible_blocks'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
