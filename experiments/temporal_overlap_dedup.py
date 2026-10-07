"""EXPERIMENT ONLY: anchored visible-block replay; never a production launcher.

Keep all physical/OCR/card evidence. Suppress only unresolved timeline rows
supported by a unique contiguous historical block with two inherited identities.
No timestamp cutoff, OCR-only match, singleton match, or hand-specific rule.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.chat_rows import canonical_row, canonical_row_distance, physical_spans
from parser_core.ram_first_stage import FirstStage


@dataclass
class View:
    row_id: str
    signature: np.ndarray
    pixel_hash: str
    y1: int
    index: int
    inherited: bool
    origin: str
    text: str


def signature(pixels):
    spans = physical_spans(pixels)
    return canonical_row(pixels, spans[0]) if len(spans) == 1 else np.empty((0, 0), np.uint8)


def anchored_duplicates(history: list[View], block: list[View]):
    """Align by known stable identities first; visual/text checks only confirm.

    At least two inherited anchors must imply the same diagonal mapping into
    the published tail. Conflicting anchors reject the entire proposal. Rows
    beyond the anchored contiguous visual run are kept, including new suffixes.
    """
    positions = {v.row_id: i for i, v in enumerate(history)}
    offsets = [positions[v.row_id] - i for i, v in enumerate(block)
               if v.inherited and v.row_id in positions]
    if len(offsets) < 2 or len(set(offsets)) != 1:
        return []
    offset = offsets[0]
    runs, run = [], []
    for i, v in enumerate(block):
        j = offset + i
        old = history[j] if 0 <= j < len(history) else None
        distance = canonical_row_distance(old.signature, v.signature) if old else float('inf')
        valid = old is not None and distance <= 9.0
        if v.inherited:
            valid = valid and old.row_id == v.row_id
        elif v.text and old and old.text:
            valid = valid and v.text == old.text
        if valid:
            run.append((i, j, distance))
        else:
            if run:
                runs.append(run)
            run = []
    if run:
        runs.append(run)
    out = []
    for run in runs:
        anchors = [block[i].row_id for i, _, _ in run if block[i].inherited]
        if len(anchors) < 2:
            continue
        for i, j, distance in run:
            v = block[i]
            if not v.inherited and v.origin == 'UNRESOLVED_IDENTITY':
                out.append((v, history[j], {
                    'reason': 'UNIQUE_ANCHORED_CONTIGUOUS_PUBLISHED_HISTORY_OVERLAP',
                    'exact_hash_match': v.pixel_hash == history[j].pixel_hash,
                    'visual_match': True, 'canonical_mae': distance,
                    'visual_threshold': 9.0, 'overlap_length': len(run),
                    'anchor_row_ids': anchors,
                    'overlap_pairs': [{'current_row_id': block[a].row_id,
                                       'previous_row_id': history[b].row_id,
                                       'canonical_mae': d} for a, b, d in run],
                }))
    return out


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text('utf-8').splitlines()]


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), 'utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Experiment output must be fresh')
    manifest_bytes = (ROOT / 'production_manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    pinned = {k: sha(ROOT / k) for k in manifest['hashes']}
    if pinned != manifest['hashes']:
        raise ValueError('Golden integrity mismatch')
    metadata = json.loads((args.old / 'run_metadata.json').read_text('utf-8'))
    summary = json.loads((args.old / 'final_summary.json').read_text('utf-8'))
    assert metadata['production_modified_files'] == []
    assert metadata['production_manifest_sha256'] == hashlib.sha256(manifest_bytes).hexdigest()
    assert summary['frames'] == metadata['frames_requested'] == 2040
    assert metadata['vl_enabled'] is False and metadata['cards_enabled'] is True
    video = Path(metadata['video'])
    assert sha(video) == metadata['source_video_sha256'] == manifest['golden_benchmark']['source_sha256']
    physical = read_jsonl(args.old / 'physical_rows.jsonl')
    timeline = read_jsonl(args.old / 'ordered_timeline.jsonl')
    by_id = {r['stable_row_id']: r for r in timeline}
    by_key = {r['row_key']: r for r in physical}
    history, aliases, removals = {}, {}, []
    pending, pending_key = [], None
    emitted, blocks, dedup_seconds = 0, 0, 0.0

    def resolved(rid):
        while rid in aliases:
            rid = aliases[rid]
        return rid

    def flush():
        nonlocal blocks, dedup_seconds
        if not pending:
            return
        started = time.perf_counter()
        table, frame = pending_key
        tail = history.setdefault(table, [])
        block = sorted(pending, key=lambda v: v.index)
        for v in block:
            if v.inherited:
                v.row_id = resolved(v.row_id)
        found = anchored_duplicates(tail[-256:], block)
        removed = set()
        for current, previous, evidence in found:
            old_row = by_id[current.row_id]
            aliases[current.row_id] = previous.row_id
            removed.add(current.row_id)
            removals.append({'table_id': f'T{table:02d}', 'frame_id': frame,
                'timestamp_ms': old_row['video_time'] * 1000 if old_row['video_time'] is not None else None,
                'y1': current.y1, 'text': current.text, 'old_row_id': current.row_id,
                'matched_previous_row_id': previous.row_id,
                'current_pixel_sha256': current.pixel_hash,
                'previous_pixel_sha256': previous.pixel_hash, **evidence})
        tail.extend(v for v in block if not v.inherited and v.row_id not in removed)
        blocks += 1
        dedup_seconds += time.perf_counter() - started
        pending.clear()

    def observe(m, pixels):
        nonlocal pending_key
        key = (m['table_id'], m['frame_id'])
        if pending_key != key:
            flush()
            pending_key = key
        rid = m['stable_row_id']
        pending.append(View(rid, signature(pixels), hashlib.sha256(pixels.tobytes()).hexdigest(),
                            m['y1'], m['source_span_index'], True, m['origin'], by_id[rid]['raw_ocr_text']))

    started = time.perf_counter()
    stage = FirstStage(video, metadata['table_sessions'], summary['frames'], root=ROOT,
                       identity_observer=observe)
    for row in stage.rows():
        key = f'{row.table_id}:{row.frame_id}:{row.source_span_index}'
        expected = by_key[key]
        assert (row.stable_row_id, row.pixel_sha256, row.y1, row.origin, row.epoch) == (
            expected['row_id'], expected['row_pixel_sha256'], expected['y1'], expected['origin'], expected['epoch'])
        assert row.pixels.tobytes() == base64.b64decode(expected['row_pixels_base64'])
        group = (row.table_id, row.frame_id)
        if group != pending_key:
            flush()
            pending_key = group
        pending.append(View(row.stable_row_id, signature(row.pixels), row.pixel_sha256,
                            row.y1, row.source_span_index, False, row.origin,
                            by_id[row.stable_row_id]['raw_ocr_text']))
        emitted += 1
    flush()
    replay_wall = time.perf_counter() - started
    assert emitted == len(physical) and stage.seen_frames == summary['frames']
    assert pinned == {k: sha(ROOT / k) for k in pinned}
    removed_ids = {r['old_row_id'] for r in removals}
    new = [r for r in timeline if r['stable_row_id'] not in removed_ids]
    args.output.mkdir(parents=True)
    for name in ('physical_rows.jsonl', 'raw_ocr_results.jsonl', 'visual_evidence.jsonl'):
        shutil.copyfile(args.old / name, args.output / name)
    write_jsonl(args.output / 'ordered_timeline.jsonl', new)
    write_jsonl(args.output / 'removed_duplicates.jsonl', removals)
    write_jsonl(args.output / 'T01_before.jsonl', [r for r in timeline if r['table_id'] == 'T01'])
    write_jsonl(args.output / 'T01_after.jsonl', [r for r in new if r['table_id'] == 'T01'])
    result = {'experiment': 'ANCHORED_TEMPORAL_OVERLAP_V1', 'old': str(args.old.resolve()),
        'frames': stage.seen_frames, 'visible_blocks': blocks, 'physical_parity': True,
        'old_physical_rows': len(physical), 'new_physical_rows': len(physical),
        'old_ocr_rows': len(read_jsonl(args.old / 'raw_ocr_results.jsonl')),
        'new_ocr_rows': len(read_jsonl(args.old / 'raw_ocr_results.jsonl')),
        'old_timeline_rows': len(timeline), 'new_timeline_rows': len(new),
        'removed_count': len(removals), 'old_wall_seconds': summary['wall_seconds'],
        'replay_wall_seconds': replay_wall, 'dedup_seconds': dedup_seconds,
        'performance_scope': 'Full capture replay plus dedup; OCR/cards reused byte-for-byte, not a second end-to-end benchmark',
        'input_hashes': {n: sha(args.old / n) for n in ('physical_rows.jsonl', 'raw_ocr_results.jsonl', 'visual_evidence.jsonl', 'ordered_timeline.jsonl', 'run_metadata.json')},
        'production_unchanged': True, 'retained_timeline_rows_unchanged': new == [r for r in timeline if r['stable_row_id'] not in removed_ids],
        'raw_evidence_hash_parity': {n: sha(args.old / n) == sha(args.output / n) for n in ('physical_rows.jsonl', 'raw_ocr_results.jsonl', 'visual_evidence.jsonl')}}
    (args.output / 'experiment_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
