import json
import queue
from types import SimpleNamespace

import pytest

from parser_core.bounded_storage import BoundedStats, DiskIntervals, DiskRecords
from parser_core.performance_diagnostics import Diagnostics, union_seconds


def test_reservoir_above_capacity_keeps_exact_aggregates_and_last():
    stats = BoundedStats(capacity=17)
    other = BoundedStats(capacity=17)
    values = [i / 4 for i in range(-100, 1000)]
    for value in values:
        stats.append(value)
        other.append(value)
    report = stats.distribution()
    assert len(stats) == report['calls'] == len(values)
    assert len(stats.values) == report['quantile_sample_size'] == 17
    assert report['total'] == sum(values)
    assert report['mean'] == sum(values) / len(values)
    assert (report['min'], report['max'], stats[-1]) == (min(values), max(values), values[-1])
    assert report['quantiles_approximate'] is True
    assert stats.values == other.values


def test_under_capacity_quantiles_are_exact_and_empty_is_explicit():
    stats = BoundedStats(capacity=5)
    assert stats.distribution()['mean'] is None
    for value in [5, 1, 3, 2, 4]:
        stats.append(value)
    report = stats.distribution()
    assert report['median'] == 3
    assert report['p95'] == report['p99'] == 5
    assert report['quantiles_approximate'] is False


@pytest.mark.parametrize('lo,hi', [(-10, 100), (2, 3), (3, 8), (10, 10), (100, 110)])
def test_disk_interval_union_matches_old_overlap_clipping(tmp_path, lo, hi):
    intervals = [(0, 4), (1, 2), (3, 5), (5, 5), (4, 9), (8, 11), (20, 22)]
    spool = DiskIntervals(tmp_path / 'intervals.sqlite')
    try:
        for a, b in reversed(intervals):
            spool.append((a, b, 'capture', 'compute'))
        assert spool.union(lo, hi) == union_seconds(intervals, lo, hi)
        assert not spool.pending
    finally:
        spool.close()


def test_interval_batch_is_bounded_and_filters_preserve_exact_union(tmp_path):
    spool = DiskIntervals(tmp_path / 'intervals.sqlite')
    entries = [(i / 2, i / 2 + 1, 'header/queue_put_wait' if i % 2 else 'capture',
                'wait' if i % 2 else 'compute') for i in range(5000)]
    try:
        for entry in entries:
            spool.append(entry)
            assert len(spool.pending) < 2048
        assert len(spool) == 5000
        wait = [(a, b) for a, b, _, kind in entries if kind == 'wait']
        expected = union_seconds(wait, 10, 120)
        assert spool.union(10, 120, kinds=('wait',)) == expected
        assert spool.union(10, 120, suffix='/queue_put_wait') == expected
        assert spool.union(10, 120, names=('header/queue_put_wait',)) == expected
        assert set(spool.names()) == {'capture', 'header/queue_put_wait'}
    finally:
        spool.close()


def test_disk_records_preserve_all_ordered_payloads_with_only_two_in_ram(tmp_path):
    spool = DiskRecords(tmp_path / 'jobs.jsonl')
    expected = [dict(job=i, raw='Строка', status='UNKNOWN' if i % 2 else 'OK') for i in range(3000)]
    try:
        for record in expected:
            spool.append(record)
        assert len(spool) == 3000 and len(spool.tail) == 2
        assert spool[-1] == expected[-1]
        assert list(spool) == expected
    finally:
        spool.close()
    assert [json.loads(line) for line in (tmp_path / 'jobs.jsonl').read_text('utf-8').splitlines()] == expected


def test_completed_diagnostic_jobs_leave_inflight_map_and_spool_every_record(tmp_path):
    profile = Diagnostics(tmp_path)
    owner = SimpleNamespace(jobs=queue.Queue(maxsize=8))
    profile.queues[id(owner.jobs)] = ('stack', owner.jobs, owner)
    try:
        for frame in range(2500):
            row = SimpleNamespace(table_id=1, frame_id=frame, source_span_index=0)
            profile.submitted(owner, [row])
            started = profile.batch_started(owner, [row])
            assert len(profile.running_jobs) == 1
            profile.batch_done(owner, [row], started)
            assert not profile.running_jobs and not profile.job_submit
        records = list(profile.jobs)
        assert len(records) == 2500
        assert [record['frame'] for record in records] == list(range(2500))
        assert all(record['callback_end'] >= record['start'] >= record['submit'] for record in records)
        assert profile.counters['stack/jobs_completed'] == 2500
        assert len(profile.jobs.tail) == 2
        assert len(profile.queue_depth['stack'].values) <= 1024
    finally:
        for records in (profile.jobs, profile.samples, profile.ocr_compute):
            records.close()
        profile.intervals.close()
