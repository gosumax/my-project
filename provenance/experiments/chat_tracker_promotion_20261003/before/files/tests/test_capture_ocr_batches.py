from dataclasses import dataclass
import pytest
from parser_core.parallel_ocr import CaptureTableBatches, row_key


@dataclass
class Row:
    table_id: int
    frame_id: int
    source_span_index: int
    session_id: str = 'S'


class Queue:
    def __init__(self):
        self.batches = []
    def submit_batch(self, rows):
        self.batches.append(list(rows))


def test_every_table_dispatches_during_capture_and_tails_preserve_source_order():
    queue = Queue()
    scheduler = CaptureTableBatches(queue, 2)
    rows = [Row(table, frame, 0) for frame in (0, 1, 2) for table in (1, 2, 3)]
    for row in rows[:6]:
        scheduler.submit(row)
    assert [[row.table_id for row in batch] for batch in queue.batches] == [[1,1],[2,2],[3,3]]
    for row in rows[6:]:
        scheduler.submit(row)
    scheduler.finish_capture(rows)
    assert [[r.frame_id for r in b] for b in queue.batches] == [[0,1],[0,1],[0,1],[2],[2],[2]]
    assert sorted(key for b in scheduler.batch_row_keys for key in b) == sorted(map(row_key, rows))


def test_distinct_sessions_cannot_share_an_ocr_batch():
    queue = Queue();scheduler = CaptureTableBatches(queue, 2)
    rows = [Row(1,0,0,'A'),Row(1,1,0,'B'),Row(1,2,0,'A'),Row(1,3,0,'B')]
    for row in rows:scheduler.submit(row)
    scheduler.finish_capture(rows)
    assert [[r.session_id for r in b] for b in queue.batches] == [['A','A'],['B','B']]


def test_duplicates_and_reordered_rows_fail_before_silent_double_submission():
    scheduler = CaptureTableBatches(Queue(), 2)
    scheduler.submit(Row(1,10,0))
    with pytest.raises(ValueError, match='nonmonotonic'):
        scheduler.submit(Row(1,9,0))
    with pytest.raises(ValueError, match='Duplicate'):
        scheduler.submit(Row(1,10,0))


def test_incomplete_capture_coverage_is_rejected():
    scheduler = CaptureTableBatches(Queue(), 2)
    scheduler.submit(Row(1,0,0))
    with pytest.raises(ValueError, match='exactly once'):
        scheduler.finish_capture([Row(1,0,0),Row(2,0,0)])
