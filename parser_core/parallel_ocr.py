"""Bounded OCR handoff for rows emitted by the unchanged first stage."""

from __future__ import annotations

import base64
import queue
import threading
import time

import cv2
from parser_core import performance_diagnostics as diagnostics


def row_key(row):
    return f"{row.table_id}:{row.frame_id}:{row.source_span_index}"


class ParallelRowOCR:
    def __init__(self, client_factory, *, workers=2, queue_size=32, batch_size=8, result_observer=None, diagnostic_lane=None):
        if min(workers, queue_size, batch_size) < 1:
            raise ValueError("OCR workers, queue size, and batch size must be positive")
        self.jobs = queue.Queue(maxsize=queue_size)
        self.results = {}
        self.result_observer = result_observer
        self.errors = []
        self.spans = []
        self.max_queue = 0
        self.first_started = None
        self._lock = threading.Lock()
        self._end = object()
        self._batch_size = batch_size
        self.worker_timings = {}
        self.diagnostics = diagnostics.ACTIVE
        if self.diagnostics:
            callback_owner = getattr(result_observer, '__self__', None)
            callback_name = type(callback_owner).__name__ if callback_owner else ''
            lane = diagnostic_lane or {'HandHeaderAdmission':'header', 'StartStackLiveConnection':'stack',
                    'SeatForcedBinding':'forced'}.get(callback_name, 'chat')
            self.diagnostics.register_queue(self, lane)
        self._threads = [threading.Thread(target=self._run, args=(client_factory, i), daemon=True)
                         for i in range(workers)]
        for thread in self._threads:
            thread.start()

    def submit(self, row):
        self.submit_batch([row])

    def submit_batch(self, rows):
        if not rows or len(rows) > self._batch_size:
            raise ValueError("Invalid OCR batch")
        if self.diagnostics:
            self.diagnostics.submitted(self, rows)
            with self.diagnostics.span(self.diagnostics.queues[id(self.jobs)][0]+'/queue_put_wait',kind='wait'):
                self.jobs.put(list(rows))
            self.diagnostics.queue_sample(self)
        else:
            self.jobs.put(list(rows))
        self.max_queue = max(self.max_queue, self.jobs.qsize())

    def _run(self, client_factory, worker_id):
        lifetime_started = time.perf_counter()
        metrics = {'worker_id': worker_id, 'thread_id': threading.get_native_id(),
                   'launcher_pid': None, 'batches': 0, 'rows': 0,
                   'prepare_seconds': 0.0, 'request_seconds': 0.0,
                   'inference_seconds': 0.0, 'startup_seconds': None}
        self.worker_timings[worker_id] = metrics
        client = None
        try:
            client = client_factory(worker_id)
        except Exception as exc:
            with self._lock:
                self.errors.append({"worker": worker_id, "startup_error": repr(exc)})
        try:
            while True:
                if self.diagnostics:
                    with self.diagnostics.span(self.diagnostics.queues[id(self.jobs)][0]+'/worker_idle_get_wait',kind='wait'):
                        batch = self.jobs.get()
                else:batch = self.jobs.get()
                if batch is self._end:
                    self.jobs.task_done()
                    return
                try:
                    diagnostic_started=self.diagnostics.batch_started(self,batch) if self.diagnostics else None
                    prepare_started = time.perf_counter()
                    diagnostic_prepare=self.diagnostics.tick() if self.diagnostics else None
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
                    metrics['prepare_seconds'] += time.perf_counter()-prepare_started
                    if self.diagnostics:self.diagnostics.record(self.diagnostics.queues[id(self.jobs)][0]+'/OCR_prepare',diagnostic_prepare,len(batch))
                    started = time.perf_counter()
                    with self._lock:
                        if self.first_started is None:
                            self.first_started = started
                    if self.diagnostics:
                        with self.diagnostics.span(self.diagnostics.queues[id(self.jobs)][0]+'/OCR_request',len(batch),kind='container'):
                            response = client.read_batch(prepared, self._batch_size)
                        diagnostic_callback_ready=time.perf_counter()
                        worker_detail=response.get('performance_diagnostic')
                        self.diagnostics.compute_received(self,batch,worker_detail)
                        if worker_detail:
                            with self.diagnostics.lock:
                                self.diagnostics.ocr_compute.append(dict(lane=self.diagnostics.queues[id(self.jobs)][0],worker=worker_id,
                                    batch_size=len(batch),**worker_detail))
                    else:response = client.read_batch(prepared, self._batch_size)
                    metrics['request_seconds'] += time.perf_counter()-started
                    metrics['batches'] += 1
                    metrics['rows'] += len(batch)
                    inference = response.get('inference_seconds')
                    if isinstance(inference, (int, float)):
                        metrics['inference_seconds'] += inference
                    metrics['startup_seconds'] = getattr(client, 'ready', {}).get('startup_seconds')
                    metrics['launcher_pid'] = getattr(getattr(client, 'process', None), 'pid', None)
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
                if self.result_observer is not None:
                    for item in batch:
                        try:
                            if self.diagnostics:
                                lane=self.diagnostics.queues[id(self.jobs)][0]
                                self.diagnostics.latencies[lane+'/response_received_to_callback'].append(time.perf_counter()-locals().get('diagnostic_callback_ready',time.perf_counter()))
                                self.diagnostics.callback_boundary(self,item,'start')
                                with self.diagnostics.span(lane+'/callback_admission',kind='compute'):
                                    self.result_observer(item, dict(self.results[row_key(item)]))
                                self.diagnostics.callback_boundary(self,item,'return')
                            else:self.result_observer(item, dict(self.results[row_key(item)]))
                        except Exception as exc:
                            with self._lock:
                                self.errors.append({'worker':worker_id,'sidecar_error':repr(exc),'row_key':row_key(item)})
                if self.diagnostics:self.diagnostics.batch_done(self,batch,diagnostic_started)
                self.jobs.task_done()
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception as exc:
                    with self._lock:
                        self.errors.append({"worker": worker_id, "close_error": repr(exc)})
            metrics['lifetime_seconds'] = time.perf_counter()-lifetime_started

    def finish(self):
        tick=self.diagnostics.tick() if self.diagnostics else None
        for _ in self._threads:
            self.jobs.put(self._end)
        self.jobs.join()
        for thread in self._threads:
            thread.join()
        if self.diagnostics:self.diagnostics.record(self.diagnostics.queues[id(self.jobs)][0]+'/final_drain_shutdown_wait',tick,kind='wait')


class CaptureTableBatches:
    """Dispatch deterministic per-table/session batches while capture advances.

    Completion order cannot change chronology: results use stable row keys.
    Every real response, including its measured score, remains unmodified.
    """
    def __init__(self, ocr, batch_size):
        if batch_size < 1:
            raise ValueError('OCR batch size must be positive')
        self.ocr = ocr
        self.batch_size = batch_size
        self.pending = {}
        self.seen = set()
        self.last = {}
        self.batch_row_keys = []

    def submit(self, row):
        key = row_key(row)
        table = (row.table_id, row.session_id)
        position = (row.frame_id, row.source_span_index)
        if key in self.seen or position <= self.last.get(table, (-1, -1)):
            raise ValueError('Duplicate or nonmonotonic capture OCR row')
        self.seen.add(key)
        self.last[table] = position
        batch = self.pending.setdefault(table, [])
        batch.append(row)
        if len(batch) == self.batch_size:
            self._dispatch(batch)
            self.pending[table] = []

    def _dispatch(self, batch):
        self.ocr.submit_batch(batch)
        self.batch_row_keys.append([row_key(row) for row in batch])

    def finish_capture(self, rows):
        for table in sorted(self.pending):
            batch = self.pending[table]
            if batch:
                self._dispatch(batch)
        self.pending.clear()
        keys = [row_key(row) for row in rows]
        submitted = [key for batch in self.batch_row_keys for key in batch]
        if len(keys) != len(set(keys)) or self.seen != set(keys) or len(submitted) != len(keys):
            raise ValueError('Capture OCR coverage is not exactly once')
