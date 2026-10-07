"""Offline acceptance of the promoted live CHAT entry against immutable v4.

Frozen pixel reference mappings are post-decision audit evidence only. They
cannot influence runtime tracking, OCR admission, or object allocation.
"""
import argparse
import base64
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time
import zlib

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.ram_first_stage import FirstStage
from experiments.vertical_chat_id_transfer_audit import get_metric

EXPECTED_QUEUE_SHA = '50cfcf8da2dbbf463c886dc313b96dfb2892376ede7c4740ada74c59ef622d9e'
FIELDS = ('table frame_from frame_to candidate_classification classification reason SHIFT_PX '
          'ROW_STEP_PX symmetry zones removed_top_ids carried_rows old_ids old_model_spans '
          'old_observed_spans new_observed_spans new_ids new_model_spans OCR_queue_row_ids '
          'old_edge_spans new_edge_spans checkpoint_committed safe_frame_from '
          'SAFE_ID_STATE_UNCHANGED PREVIOUS_FRAME_UPDATED old_link_states new_link_states '
          'visible_unknown_links').split()


def check(source, output):
    output.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    frozen = source / 'adjacent_safe_v4'
    refs = {(r['table'], r['frame_to']): r for r in map(json.loads,
            (frozen / 'symmetric_transitions.jsonl').open(encoding='utf-8'))}
    cut = json.loads((source / 'shift_rule_summary.json').read_text('utf-8'))['calibration']['calibration_end_frame']
    events = []
    stage = FirstStage(Path('unused'), {}, 0, root=ROOT, tracker_observer=events.append)
    stats, metric_counts = Counter(), Counter()
    queue = []
    previous_frames = {}
    coordinate_checks = 0
    with (output / 'production_transitions.jsonl').open('w', encoding='utf-8') as trace:
        for line in (source / 'states.jsonl').open(encoding='utf-8'):
            state = json.loads(line)
            table, frame = state['table'], state['frame']
            gray = np.frombuffer(zlib.decompress(base64.b64decode(state['zlib_gray'])), np.uint8).reshape(state['shape']).copy()
            assert hashlib.sha256(gray.tobytes()).hexdigest() == state['sha256']
            rows = list(stage.chat_rows(gray, frame_id=frame, table_id=table,
                        session_id=f'ADJ:T{table}', source_pts=state['pts'],
                        time_base=Fraction(*state['time_base']), source_chat_box=state.get('box')))
            actual = events.pop()
            # JSON canonicalization reconciles tuple/list container types only.
            actual = json.loads(json.dumps(actual))
            if table not in previous_frames:
                assert actual['classification'] == 'INITIAL_VISIBLE' and not rows
                previous_frames[table] = frame
                continue
            assert actual['frame_from'] == previous_frames[table]
            previous_frames[table] = frame
            ref = refs[(table, frame)]
            for field in FIELDS:
                assert actual[field] == ref[field], (table, frame, field, actual[field], ref[field])
            stats['TRANSITIONS_COMPARED'] += 1
            assert [row.stable_row_id for row in rows] == actual['OCR_queue_row_ids']
            if actual['classification'] != 'GLOBAL_SCROLL':
                assert not rows and actual['SAFE_ID_STATE_UNCHANGED']
            else:
                n = actual['symmetry']['N']
                assert len(rows) == len(actual['removed_top_ids']) == n
                for carry in actual['carried_rows']:
                    assert carry['translated_span'] == [v-actual['SHIFT_PX'] for v in carry['old_span']]
                    assert actual['new_ids'][carry['new_index']] == actual['old_ids'][carry['old_index']]
                    assert not carry['OCR_submitted']
                    coordinate_checks += 1
                proxy = ref['local_pixel_reference']
                mapping = {int(j): i for j, i in proxy['mapping'].items()} if proxy else {}
                metrics, details = get_metric(actual['old_ids'], actual['new_ids'], mapping)
                metrics['ID_TRANSFER_UNRESOLVED'] += sum(c['new_index'] not in mapping for c in actual['carried_rows'])
                for key in ('ID_TRANSFER_CORRECT', 'ID_TRANSFER_UNRESOLVED', 'ID_TRANSFER_WRONG', 'DUPLICATE_ID', 'LOST_ID'):
                    assert metrics[key] == ref['ID_transfer_metrics'].get(key, 0), (table, frame, key)
                if frame > cut:
                    metric_counts.update(metrics)
                for row in rows:
                    j = actual['new_ids'].index(row.stable_row_id)
                    ok, packed = cv2.imencode('.png', row.pixels)
                    assert ok
                    queue.append({'item_id': row.stable_row_id, 'table': table, 'frame': frame,
                        'frame_from': actual['frame_from'], 'source_pts': row.source_pts,
                        'time_base': [row.time_base_num, row.time_base_den], 'classification': 'GLOBAL_SCROLL',
                        'row_index': j, 'source_span_index': row.source_span_index,
                        'row_span': actual['new_observed_spans'][j],
                        'padded_span': (row.y1, row.y1+row.pixels.shape[0]),
                        'N_OUT': n, 'N_IN': n, 'new_rows_in_transition': n,
                        'SHIFT_PX': actual['SHIFT_PX'], 'ROW_STEP_PX': actual['ROW_STEP_PX'],
                        'source_chat_sha256': actual['source_chat_sha256'],
                        'source_sha256': row.pixel_sha256, 'png_base64': base64.b64encode(packed).decode()})
            if frame > cut:
                stats[actual['classification']+'_COUNT'] += 1
            trace.write(json.dumps(actual, ensure_ascii=False)+'\n')
    queue_path = output / 'production_ocr_queue.jsonl'
    with queue_path.open('w', encoding='utf-8') as stream:
        for item in queue:
            stream.write(json.dumps(item, ensure_ascii=False)+'\n')
    queue_bytes = queue_path.read_bytes()
    assert queue_bytes == (frozen / 'zonal_ocr_queue.jsonl').read_bytes()
    assert hashlib.sha256(queue_bytes).hexdigest() == EXPECTED_QUEUE_SHA
    assert len(queue) == 442 and len({r['item_id'] for r in queue}) == 442
    assert stats['GLOBAL_SCROLL_COUNT'] == 181 and stats['LOCAL_CHANGE_COUNT'] == 3850 and stats['UNRESOLVED_COUNT'] == 13
    assert stats['TRANSITIONS_COMPARED'] == len(refs) == 6266
    assert not any(metric_counts[k] for k in ('ID_TRANSFER_WRONG', 'DUPLICATE_ID', 'LOST_ID'))
    result = {'PASS': True, 'statistics': dict(stats), 'ID_transfer_metrics': dict(metric_counts),
        'coordinate_transfer_checks': coordinate_checks, 'coordinate_transfer_wrong': 0,
        'OCR_NEW_ROWS_ONLY': True, 'OCR_queue_byte_identical': True, 'OCR_queue_rows': len(queue),
        'OCR_queue_sha256': EXPECTED_QUEUE_SHA, 'elapsed_seconds': time.perf_counter()-tick,
        'ID_audit_reference': 'Frozen post-decision ordered pixel mappings; no physical-history certification',
        'source_sha256': hashlib.sha256((source / 'states.jsonl').read_bytes()).hexdigest()}
    for key in ('ID_TRANSFER_CORRECT', 'ID_TRANSFER_UNRESOLVED', 'ID_TRANSFER_WRONG', 'DUPLICATE_ID', 'LOST_ID'):
        result['ID_transfer_metrics'].setdefault(key, 0)
    (output / 'frozen_regression.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / 'outputs/vertical_chat_shift_20261002')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/chat_tracker_promotion_20261003_checks/frozen')
    args = parser.parse_args()
    check(args.input, args.output)
