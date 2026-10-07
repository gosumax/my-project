"""Parallel/storage changes must preserve hand fences and complete evidence."""
import hashlib
import multiprocessing
import zlib
import cv2
import numpy as np
import pytest
from parser_core.temporal_overlap import TemporalObservations


def pixels(text):
    out = np.zeros((16, 228), np.uint8)
    cv2.putText(out, text, (1, 11), cv2.FONT_HERSHEY_SIMPLEX, .3, 220, 1)
    return out


def sample():
    collector = TemporalObservations()
    timeline = []
    statuses = {}
    sessions = {'1': 'S', '2': 'T'}
    for frame in (0, 1, 2):
        for table, session in ((1, 'S'), (2, 'T')):
            header = '123456' if frame < 2 else '123457'
            rows = [(f'{session}:m' if frame < 2 else f'{session}:m2', 'Hand #' + header, frame == 1),
                    (f'{session}:a' if frame == 0 else f'{session}:a{frame}', 'ante A', False),
                    (f'{session}:b', 'ante B', frame > 0),
                    (f'{session}:c', 'ante C', frame > 0)]
            for index, (rid, text, inherited) in enumerate(rows):
                collector.record(table, frame, rid, index, index*20, 'UNRESOLVED_IDENTITY', pixels(text), inherited)
                if not inherited:
                    timeline.append(dict(table_id=f'T{table:02d}', stable_row_id=rid,
                        raw_ocr_text=text, status='UNRESOLVED', video_time=frame/30))
                    statuses[rid] = 'OK'
    return collector, timeline, sessions, statuses


def test_storage_sharing_preserves_distinct_observations_and_source_bytes():
    collector = TemporalObservations()
    raw = pixels('same pixels')
    for table, frame, rid in ((1, 0, 'S:H1'), (2, 3, 'T:H9'), (1, 9, 'S:H2')):
        collector.record(table, frame, rid, 2, 40, 'OBSERVED', raw, False)
    assert len(collector.records) == 3
    assert len({r[2] for r in collector.records}) == 3
    assert collector.records[0][8] is collector.records[1][8] is collector.records[2][8]
    assert collector.payload_cache_hits == 2
    assert collector.compressed_bytes == collector.unique_compressed_bytes*3
    assert all(zlib.decompress(r[8]) == raw.tobytes() for r in collector.records)


def test_cache_hit_requires_exact_bytes_not_only_digest():
    collector = TemporalObservations()
    raw = pixels('native')
    digest = hashlib.sha256(raw.tobytes()).hexdigest()
    collector._payloads[(raw.shape, digest)] = zlib.compress(bytes(raw.size))
    with pytest.raises(ValueError, match='digest collision'):
        collector.record(1, 0, 'S:a', 0, 0, 'OBSERVED', raw, False)
    assert not collector.records


@pytest.mark.parametrize('failed_ocr', [False, True])
def test_parallel_preserves_interleaved_tables_unknown_barriers_and_proof_order(failed_ocr):
    serial, timeline, sessions, statuses = sample()
    if failed_ocr:
        statuses['T:a1'] = 'ERROR'
    parallel = TemporalObservations()
    parallel.records = list(serial.records)
    observations = len(parallel.records)
    expected, baseline = serial.reduce(timeline, sessions, ocr_status=statuses)
    retained, reduced = parallel.reduce(timeline, sessions, ocr_status=statuses, workers=2)
    assert retained == expected
    assert reduced.removals == baseline.removals
    assert reduced.decisions == baseline.decisions
    assert reduced.birth == baseline.birth
    assert reduced.active == baseline.active
    assert reduced.proof == baseline.proof
    assert reduced.block_count == baseline.block_count
    assert reduced.barriers == baseline.barriers
    assert sum(row['records'] for row in reduced.worker_timings) == observations
    assert all(r['old_row_id'] not in ('S:a2', 'T:a2') for r in reduced.removals)
    assert not parallel.records and not parallel._payloads


def test_bad_parallel_input_cannot_silently_drop_duplicated_physical_identity():
    collector, timeline, sessions, statuses = sample()
    timeline.append(dict(timeline[0]))
    with pytest.raises(ValueError, match='Duplicate physical'):
        collector.reduce(timeline, sessions, ocr_status=statuses, workers=2)
    assert collector.records


def test_worker_failure_preserves_input_and_leaves_no_pool_processes():
    collector, timeline, sessions, statuses = sample()
    timeline = [row for row in timeline if row['stable_row_id'] != 'T:b']
    count = len(collector.records)
    before = {p.pid for p in multiprocessing.active_children()}
    with pytest.raises(ValueError, match='physical/OCR provenance'):
        collector.reduce(timeline, sessions, ocr_status=statuses, workers=2)
    assert len(collector.records) == count
    assert {p.pid for p in multiprocessing.active_children()} <= before
