"""Concurrency and ordering contracts for the RAM OCR handoff."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import base64
import hashlib
import ast
from pathlib import Path
import cv2
import pytest

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


@pytest.mark.parametrize('width', [9, 226, 230])
def test_chat_ocr_after_trim_rollback_transports_full_original_pixels(width):
    """A width change can lose All-in without cutting glyphs; preserve the input."""
    pixels = np.arange(16 * width, dtype=np.uint16).reshape(16, width).astype(np.uint8)
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    item = Row(2, 1451, 18, pixels.copy(), digest)
    received = []

    class Client:
        def read_batch(self, items, batch_size):
            for request in items:
                decoded = cv2.imdecode(np.frombuffer(base64.b64decode(request['png_base64']),
                                                     dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                received.append((decoded, request['source_sha256']))
            return {'results': [{'item_id': request['item_id'], 'status': 'OK',
                                 'raw_text': 'All-in', 'score': .93} for request in items]}

        def close(self):
            pass

    ocr = ParallelRowOCR(lambda _: Client(), workers=1, batch_size=1)
    ocr.submit(item)
    ocr.finish()
    assert not ocr.errors
    assert len(received) == 1
    assert received[0][0].shape == pixels.shape
    assert np.array_equal(received[0][0], pixels)
    assert received[0][1] == digest
    assert np.array_equal(item.pixels, pixels)


def test_ordinary_chat_runner_does_not_enable_rejected_width_trim():
    runner = Path(__file__).resolve().parents[1] / 'scripts/run_smoke14_ram.py'
    tree = ast.parse(runner.read_text(encoding='utf-8'))
    ordinary = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == 'ocr' for target in node.targets)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == 'ParallelRowOCR']
    assert len(ordinary) == 1
    assert all(keyword.arg != 'chat_ocr_right_trim' or
               isinstance(keyword.value, ast.Constant) and keyword.value.value is False
               for keyword in ordinary[0].keywords)
