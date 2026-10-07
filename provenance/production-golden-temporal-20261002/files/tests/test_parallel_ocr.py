"""Concurrency and ordering contracts for the RAM OCR handoff."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from parser_core.parallel_ocr import ParallelRowOCR, row_key


@dataclass
class Row:
    table_id: int
    frame_id: int
    source_span_index: int
    pixels: np.ndarray
    pixel_sha256: str = "test"


def row(frame):
    return Row(1, frame, 1, np.zeros((4, 8), dtype=np.uint8))


def test_ocr_starts_while_producer_is_active_and_finishes_out_of_order():
    started = threading.Event()
    release = threading.Event()
    completion = []

    class Client:
        def read_batch(self, items, batch_size):
            frame = int(items[0]["item_id"].split(":")[1])
            started.set()
            if frame == 1:
                assert release.wait(2)
            completion.append(frame)
            return {"results": [{"item_id": item["item_id"], "raw_text": str(frame),
                                  "status": "OK", "score": 1.0} for item in items]}

        def close(self):
            pass

    ocr = ParallelRowOCR(lambda _: Client(), workers=2, queue_size=1, batch_size=1)
    first, second = row(1), row(2)
    ocr.submit(first)
    assert started.wait(2)  # Capture has not reached its end.
    ocr.submit(second)
    deadline = time.monotonic() + 2
    while 2 not in completion and time.monotonic() < deadline:
        time.sleep(.005)
    assert completion == [2]
    release.set()
    ocr.finish()
    assert completion == [2, 1]
    ordered = [ocr.results[row_key(item)]["raw_text"] for item in (first, second)]
    assert ordered == ["1", "2"]
    assert ocr.max_queue <= 1


def test_worker_failure_becomes_explicit_error_without_hanging():
    ocr = ParallelRowOCR(lambda _: (_ for _ in ()).throw(RuntimeError("unavailable")),
                         workers=1, queue_size=1, batch_size=1)
    item = row(1)
    ocr.submit(item)
    ocr.finish()
    assert ocr.results[row_key(item)]["status"] == "ERROR"
    assert ocr.errors[0]["startup_error"] == "RuntimeError('unavailable')"
