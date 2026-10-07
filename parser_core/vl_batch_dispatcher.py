"""One persistent GPU client shared by bounded parallel CHAT callers."""
from __future__ import annotations

import hashlib
import queue
import threading
import time
from concurrent.futures import Future


class _ChatClientProxy:
    def __init__(self, dispatcher):
        self.dispatcher = dispatcher

    def read_bytes(self, source, source_sha256, right_trim_px):
        return self.dispatcher.read_bytes(source, source_sha256, right_trim_px)

    def close(self):
        # The owner closes the shared GPU process after all CHAT callers finish.
        pass


class VLBatchDispatcher:
    def __init__(self, client_factory, *, batch_size=8, queue_size=32, batch_wait_ms=30):
        if not 1 <= batch_size <= 8 or queue_size < 1 or batch_wait_ms < 0:
            raise ValueError("Invalid VL batch dispatcher configuration")
        self.batch_size = batch_size
        self.batch_wait_s = batch_wait_ms/1000
        self.jobs = queue.Queue(maxsize=queue_size)
        self.batch_sizes = []
        self.spans = []
        self.max_queue = 0
        self.startup_seconds = None
        self.errors = []
        self._end = object()
        self._closed = False
        self._submit_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, args=(client_factory,), daemon=True)
        self._thread.start()

    def proxy(self, _worker_id=None):
        return _ChatClientProxy(self)

    def read_bytes(self, source, source_sha256, right_trim_px):
        if hashlib.sha256(source).hexdigest() != source_sha256:
            raise ValueError("VL RAM crop SHA mismatch")
        if right_trim_px != 12:
            raise ValueError("VL dispatcher accepts only CHAT crop trim 12")
        future = Future()
        with self._submit_lock:
            if self._closed:
                raise RuntimeError("VL dispatcher is closed")
            self.jobs.put(({"source": source, "source_sha256": source_sha256,
                            "right_trim_px": right_trim_px}, future))
            self.max_queue = max(self.max_queue, self.jobs.qsize())
        return future.result()

    def _run(self, client_factory):
        client = None
        startup_error = None
        try:
            started = time.perf_counter()
            client = client_factory()
            if hasattr(client, "_start"):
                client._start()
            self.startup_seconds = time.perf_counter()-started
        except Exception as exc:
            startup_error = exc
            self.errors.append({"startup_error": repr(exc)})
        try:
            stop = False
            while not stop:
                item = self.jobs.get()
                if item is self._end:
                    self.jobs.task_done()
                    break
                batch = [item]
                deadline = time.perf_counter()+self.batch_wait_s
                while len(batch) < self.batch_size:
                    try:
                        item = self.jobs.get(timeout=max(0, deadline-time.perf_counter()))
                    except queue.Empty:
                        break
                    if item is self._end:
                        self.jobs.task_done()
                        stop = True
                        break
                    batch.append(item)
                started = time.perf_counter()
                try:
                    if startup_error is not None:
                        raise RuntimeError("VL worker startup failed") from startup_error
                    results = client.read_batch_bytes([request for request, _ in batch])
                    if len(results) != len(batch):
                        raise RuntimeError("VL batch result count mismatch")
                    for (request, _), result in zip(batch, results):
                        if result.get("source_sha256") != request["source_sha256"]:
                            raise RuntimeError("VL batch result crop SHA mismatch")
                    for (_, future), result in zip(batch, results):
                        future.set_result(result)
                except Exception as exc:
                    self.errors.append({"error": repr(exc)})
                    for _, future in batch:
                        future.set_exception(exc)
                finally:
                    self.spans.append((started, time.perf_counter()))
                    self.batch_sizes.append(len(batch))
                    for _ in batch:
                        self.jobs.task_done()
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception as exc:
                    self.errors.append({"close_error": repr(exc)})

    def close(self):
        with self._submit_lock:
            if not self._closed:
                self._closed = True
                self.jobs.put(self._end)
        self.jobs.join()
        self._thread.join()
