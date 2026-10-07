"""Bounded, nonblocking handoff of changed seat stack crops.

The caller owns hand boundaries. A dropped candidate invalidates certification
for that seat because it might have been the first visible stack.
"""

from __future__ import annotations

import hashlib
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from parser_core.start_stack import HandStackTracker, StackReading


@dataclass(frozen=True)
class CropJob:
    sequence: int
    table_id: str
    table_context: str
    seat: str
    frame_index: int
    source_pts: int | None
    timestamp: float | None
    pixels: np.ndarray
    sha256: str


class AsyncStackCapture:
    def __init__(self, tracker: HandStackTracker, boxes: dict[str, tuple[int, int, int, int]],
                 recognizer: Callable[[np.ndarray], tuple[str | None, float | None]],
                 *, crop_dir: Path, queue_size: int = 32, workers: int = 1):
        if queue_size < 1 or workers < 1:
            raise ValueError("Queue size and workers must be positive")
        if set(boxes) != set(tracker.seats):
            raise ValueError("Every bound seat needs one stack ROI")
        self.tracker = tracker
        self.boxes = boxes
        self.recognizer = recognizer
        self.crop_dir = crop_dir
        self.jobs: queue.Queue[CropJob | None] = queue.Queue(maxsize=queue_size)
        self.previous: dict[str, np.ndarray] = {}
        self.readings: list[StackReading] = []
        self.dropped: set[str] = set()
        self.max_queue = 0
        self._lock = threading.Lock()
        self._closed = False
        self._submitted = 0
        self._next_result = 0
        self._pending: dict[int, StackReading] = {}
        self._threads = [threading.Thread(target=self._run, daemon=True) for _ in range(workers)]
        for thread in self._threads:
            thread.start()

    def offer_table(self, table_id: str, table_context: str, frame_index: int,
                    source_pts: int | None, timestamp: float | None,
                    table_rgb: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Stack capture already closed")
        if table_id != self.tracker.table_id or table_context != self.tracker.table_context:
            raise ValueError("Wrong table context")
        if frame_index < self.tracker.start_frame:
            raise ValueError("Stack crop predates hand start")
        height, width = table_rgb.shape[:2]
        for seat, (x1, y1, x2, y2) in self.boxes.items():
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Invalid stack ROI for {seat}")
            crop = table_rgb[y1:y2, x1:x2]
            old = self.previous.get(seat)
            if old is not None and np.array_equal(old, crop):
                continue
            self.previous[seat] = np.ascontiguousarray(crop)
            with self._lock:
                if self.tracker.seats[seat].phase == "STACK_CAPTURED":
                    continue
            pixels = self.previous[seat]
            job = CropJob(self._submitted, table_id, table_context, seat, frame_index, source_pts,
                          timestamp, pixels, hashlib.sha256(pixels.tobytes()).hexdigest())
            try:
                self.jobs.put_nowait(job)
                self._submitted += 1
            except queue.Full:
                self.dropped.add(seat)
            self.max_queue = max(self.max_queue, self.jobs.qsize())

    def _run(self) -> None:
        while True:
            job = self.jobs.get()
            try:
                if job is None:
                    return
                path = None
                raw = None
                score = None
                status = "OK"
                try:
                    with self._lock:
                        captured = self.tracker.seats[job.seat].phase == "STACK_CAPTURED"
                    if captured:
                        status = "SKIPPED_AFTER_CAPTURE"
                    else:
                        self.crop_dir.mkdir(parents=True, exist_ok=True)
                        path_obj = self.crop_dir / (f"{job.table_context}_{job.seat}_"
                                    f"{job.frame_index}_{job.sha256[:12]}.png")
                        ok, encoded = cv2.imencode(".png", cv2.cvtColor(job.pixels, cv2.COLOR_RGB2BGR))
                        if not ok:
                            raise ValueError("Stack crop PNG encode failed")
                        path_obj.write_bytes(encoded.tobytes())
                        path = str(path_obj)
                        raw, score = self.recognizer(job.pixels)
                except Exception as exc:
                    status = "ERROR"
                    raw = f"{type(exc).__name__}: {exc}"
                reading = StackReading(job.table_id, job.table_context, job.seat,
                    job.frame_index, job.source_pts, job.timestamp, job.sha256,
                    path, raw, score, status)
                with self._lock:
                    self.readings.append(reading)
                    self._pending[job.sequence] = reading
                    while self._next_result in self._pending:
                        ordered = self._pending.pop(self._next_result)
                        self._next_result += 1
                        if ordered.seat not in self.dropped and ordered.recognition_status != "SKIPPED_AFTER_CAPTURE":
                            try:
                                self.tracker.observe(ordered)
                            except Exception:
                                self.dropped.add(ordered.seat)
                                self.tracker.seats[ordered.seat].reason = "STACK_OBSERVATION_INVALID"
            finally:
                self.jobs.task_done()

    def finish(self) -> list[StackReading]:
        if self._closed:
            return self.readings
        self._closed = True
        self.jobs.join()
        for _ in self._threads:
            self.jobs.put(None)
        for thread in self._threads:
            thread.join()
        # Worker scheduling cannot change which visual transition is first.
        self.readings.sort(key=lambda r: (r.frame_index, r.seat))
        for seat in self.dropped:
            self.tracker.seats[seat].reason = "STACK_CANDIDATE_DROPPED"
            self.tracker.seats[seat].phase = "INCOMPLETE"
            self.tracker.seats[seat].reading = None
            self.tracker.seats[seat].visible_stack = None
        return self.readings
