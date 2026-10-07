"""Bounded OCR handoff for rows emitted by the unchanged first stage."""

from __future__ import annotations

import base64
import queue
import threading
import time

import cv2


def row_key(row):
    return f"{row.table_id}:{row.frame_id}:{row.source_span_index}"


class ParallelRowOCR:
    def __init__(self, client_factory, *, workers=2, queue_size=32, batch_size=8):
        if min(workers, queue_size, batch_size) < 1:
            raise ValueError("OCR workers, queue size, and batch size must be positive")
        self.jobs = queue.Queue(maxsize=queue_size)
        self.results = {}
        self.errors = []
        self.spans = []
        self.max_queue = 0
        self.first_started = None
        self._lock = threading.Lock()
        self._end = object()
        self._batch_size = batch_size
        self._threads = [threading.Thread(target=self._run, args=(client_factory, i), daemon=True)
                         for i in range(workers)]
        for thread in self._threads:
            thread.start()

    def submit(self, row):
        self.submit_batch([row])

    def submit_batch(self, rows):
        if not rows or len(rows) > self._batch_size:
            raise ValueError("Invalid OCR batch")
        self.jobs.put(list(rows))
        self.max_queue = max(self.max_queue, self.jobs.qsize())

    def _run(self, client_factory, worker_id):
        client = None
        try:
            client = client_factory(worker_id)
        except Exception as exc:
            with self._lock:
                self.errors.append({"worker": worker_id, "startup_error": repr(exc)})
        try:
            while True:
                batch = self.jobs.get()
                if batch is self._end:
                    self.jobs.task_done()
                    return
                try:
                    prepared = []
                    for item in batch:
                        ok, packed = cv2.imencode(".png", item.pixels)
                        if not ok:
                            raise ValueError("RAM row PNG encode failed")
                        prepared.append({"item_id": row_key(item),
                                         "png_base64": base64.b64encode(packed).decode("ascii"),
                                         "source_sha256": item.pixel_sha256})
                    if client is None:
                        raise RuntimeError("OCR_WORKER_STARTUP_FAILED")
                    started = time.perf_counter()
                    with self._lock:
                        if self.first_started is None:
                            self.first_started = started
                    response = client.read_batch(prepared, self._batch_size)
                    found = {item["item_id"]: item for item in response["results"]}
                    with self._lock:
                        self.spans.append((started, time.perf_counter(), len(batch)))
                except Exception as exc:
                    with self._lock:
                        self.errors.append({"worker": worker_id, "error": repr(exc)})
                    found = {}
                with self._lock:
                    for item in batch:
                        key = row_key(item)
                        self.results[key] = found.get(key, {
                            "status": "ERROR", "raw_text": "", "score": None,
                            "error": "WORKER_BATCH_FAILED_OR_RESULT_MISSING"})
                self.jobs.task_done()
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception as exc:
                    with self._lock:
                        self.errors.append({"worker": worker_id, "close_error": repr(exc)})

    def finish(self):
        for _ in self._threads:
            self.jobs.put(self._end)
        self.jobs.join()
        for thread in self._threads:
            thread.join()
