"""Bounded independent VL reader for physical rows containing card tiles."""

from __future__ import annotations

import hashlib
import base64
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from parser_core.board_tiles import read_board_tiles_bytes
from parser_core.card_hybrid import CardHybrid
from parser_core.parallel_ocr import row_key
from parser_core.visual_cards import parse_cards
from parser_core import performance_diagnostics as diagnostics


def card_tile_candidate(pixels: np.ndarray) -> bool:
    return (pixels.ndim == 2 and pixels.shape[1] >= 200 and
            pixels.shape[0] <= 40 and
            int(np.count_nonzero(pixels[:, :100] > 180)) >= 300)


class VisualTileQueue:
    def __init__(self, client_factory, output: Path, *, workers=1, queue_size=16,
                 card_hybrid_config: Path | None = None,
                 suit_backup_config: Path | None = None,
                 rank_ncc_config: Path | None = None,
                 image_transport: str = "file", card_workers: int = 2):
        if min(workers, queue_size, card_workers) < 1:
            raise ValueError("VL workers and queue size must be positive")
        self.output = output
        if image_transport not in {"file", "ram"}:
            raise ValueError("Unsupported VL image transport")
        self.image_transport = image_transport
        self.jobs = queue.Queue(maxsize=queue_size)
        self.results = {}
        self.max_queue = 0
        self.spans = []
        self.hybrid_spans = []
        self.errors = []
        self._lock = threading.Lock()
        self._end = object()
        self.diagnostics=diagnostics.ACTIVE
        if self.diagnostics:self.diagnostics.register_queue(self,'card')
        self._card_slots = threading.BoundedSemaphore(card_workers)
        self.hybrid = (CardHybrid(card_hybrid_config, suit_backup_config, rank_ncc_config=rank_ncc_config)
                       if card_hybrid_config is not None and suit_backup_config is not None else None)
        self.pixel_only = self.hybrid is not None and self.hybrid.rank_mode == "PIXEL_ONLY"
        self._threads = [threading.Thread(target=self._run, args=(client_factory, i), daemon=True)
                         for i in range(workers)]
        for thread in self._threads:
            thread.start()

    def submit(self, row):
        if not card_tile_candidate(row.pixels):
            return False
        if self.diagnostics:
            self.diagnostics.submitted(self,[row])
            with self.diagnostics.span('card/queue_put_wait',kind='wait'):self.jobs.put(row)
            self.diagnostics.queue_sample(self)
        else:self.jobs.put(row)
        self.max_queue = max(self.max_queue, self.jobs.qsize())
        return True

    def _run(self, client_factory, worker_id):
        client = None
        try:
            client = None if self.pixel_only else client_factory(worker_id)
        except Exception as exc:
            with self._lock:
                self.errors.append({"worker": worker_id, "startup_error": repr(exc)})
        try:
            while True:
                if self.diagnostics:
                    with self.diagnostics.span('card/worker_idle_get_wait',kind='wait'):job = self.jobs.get()
                else:job = self.jobs.get()
                if job is self._end:
                    self.jobs.task_done()
                    break
                row = job
                diagnostic_started=self.diagnostics.batch_started(self,[row]) if self.diagnostics else None
                key = row_key(row)
                result = {"row_key": key, "table_id": row.table_id,
                          "table_session_id": row.session_id, "frame_id": row.frame_id,
                          "source_span_index": row.source_span_index,
                          "video_time": row.video_time,
                          "source_pts": row.source_pts,
                          "time_base_num": row.time_base_num,
                          "time_base_den": row.time_base_den,
                          "row_pixel_sha256": row.pixel_sha256,
                          "row_id": getattr(row, "stable_row_id", None),
                          "epoch": getattr(row, "epoch", None),
                          "crop_type": "CHAT_ROW_CARDS", "street": None,
                          "status": "UNRESOLVED"}
                try:
                    path = self.output / "visual_crops" / row.session_id / f"{row.frame_id:09d}_{row.source_span_index:03d}.png"
                    ok, packed = cv2.imencode(".png", row.pixels)
                    if not ok:
                        raise ValueError("Card row PNG encode failed")
                    data = packed.tobytes()
                    digest = hashlib.sha256(data).hexdigest()
                    if self.image_transport == "file":
                        path.parent.mkdir(parents=True, exist_ok=True)
                        if path.exists() and path.read_bytes() != data:
                            raise ValueError("Visual crop path conflicts with existing pixels")
                        path.write_bytes(data)
                        result.update(crop_path=str(path.resolve()))
                    else:
                        result.update(crop_path=None, crop_png_base64=base64.b64encode(data).decode("ascii"))
                    result.update(crop_sha256=digest, image_transport=self.image_transport)
                    trim = 12
                    if client is None and not self.pixel_only:
                        raise RuntimeError("VL_WORKER_STARTUP_FAILED")
                    started = time.perf_counter()
                    response = ({"status": "OK", "raw_text": "", "source_sha256": digest,
                                 "request_id": "pixel_"+hashlib.sha256((key+digest).encode()).hexdigest()[:24],
                                 "recognizer": "FROZEN_NATIVE_PIXEL_CARD_READER",
                                 "rank_config_sha256": self.hybrid.rank_ncc_config_sha256,
                                 "suit_config_sha256": self.hybrid.suit_backup.config_sha256}
                                if self.pixel_only else
                                client.read_bytes(data, digest, trim) if self.image_transport == "ram"
                                else client.read(path, digest, trim))
                    finished = time.perf_counter()
                    if self.image_transport == "ram" and response.get("source_sha256") != digest:
                        raise ValueError("CHAT response crop SHA mismatch")
                    result.update(raw_vl_response=response, vl_raw_text=response.get("raw_text", ""),
                                  vl_status=response.get("status"), vl_elapsed_seconds=finished-started)
                    if self.pixel_only:
                        result.update(raw_vl_response=None, raw_pixel_response=response,
                                      recognition_mode="PIXEL_ONLY", vl_status="NOT_USED",
                                      pixel_elapsed_seconds=finished-started)
                    if response.get("source_sha256") == digest:
                        identity = f"{row.session_id}:{row.frame_id}:{row.source_span_index}:{digest}"
                        result.update(identity_status="CHAT_ROW_SHA_ROUNDTRIP_VERIFIED",
                                      response_source_sha256=digest,
                                      card_evidence_id="chatcard_" + hashlib.sha256(identity.encode()).hexdigest()[:24])
                    with self._lock:
                        self.spans.append((started, finished))
                    if response.get("status") == "OK":
                        for expected in (2, 3, 4, 5):
                            vl_text_cards = parse_cards(response.get("raw_text", ""), expected)
                            if vl_text_cards is not None:
                                result["vl_text_cards_unverified"] = vl_text_cards
                                break
                        reading = None if self.pixel_only else read_board_tiles_bytes(data, digest, response.get("raw_text", ""))
                        primary_cards = None
                        if reading is not None:
                            suit_codes = {"♠": "s", "♥": "h", "♦": "d", "♣": "c"}
                            cards = [("T" if card[:-1] == "10" else card[:-1]) +
                                     suit_codes[card[-1]] for card in reading.cards]
                            primary_cards = cards
                            result.update(cards=cards, status="RAW_UNREVIEWED",
                                          legacy_board_cards=cards,
                                          card_reader="board_tiles_fixed_chat_v1",
                                          template_sha256=reading.template_sha256)
                        if self.hybrid is not None:
                            if self.diagnostics:
                                with self.diagnostics.span('card/semaphore_wait',kind='wait'):self._card_slots.acquire()
                            else:self._card_slots.acquire()
                            hybrid_started = time.perf_counter()
                            try:
                                if self.image_transport == "ram":
                                    hybrid = self.hybrid.read_bytes(data, digest, response.get("raw_text", ""),
                                                                    primary_cards=primary_cards)
                                else:
                                    hybrid = self.hybrid.read(path, digest, response.get("raw_text", ""),
                                                              self.output / "card_tiles" / row.session_id /
                                                              f"{row.frame_id:09d}_{row.source_span_index:03d}",
                                                              primary_cards=primary_cards)
                            finally:
                                self._card_slots.release()
                            hybrid_finished = time.perf_counter()
                            result["hybrid"] = hybrid
                            result["hybrid_elapsed_seconds"] = hybrid_finished-hybrid_started
                            with self._lock:
                                self.hybrid_spans.append((hybrid_started, hybrid_finished))
                            if self.hybrid.suit_backup.mode == "BACKUP" and hybrid.get("cards") is not None:
                                result.update(cards=hybrid["cards"], status="RAW_UNREVIEWED",
                                              card_reader=hybrid["card_reader"])
                            elif self.hybrid.suit_backup.mode == "BACKUP":
                                result.update(cards=None, status="UNRESOLVED",
                                              reason=hybrid.get("reason") or "HYBRID_CARD_READING_UNRESOLVED",
                                              hybrid_authoritative_rejection=True)
                    if result["status"] == "UNRESOLVED":
                        result.setdefault("reason", "VL_CARD_READING_UNRESOLVED")
                except Exception as exc:
                    result.update(cards=None, status="UNRESOLVED", reason="VL_JOB_FAILED", error=repr(exc),
                                  hybrid_authoritative_rejection=True)
                    with self._lock:
                        self.errors.append({"row_key": key, "error": repr(exc)})
                with self._lock:
                    self.results[key] = result
                if self.diagnostics:self.diagnostics.batch_done(self,[row],diagnostic_started)
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

    def write_evidence(self, path: Path):
        with path.open("w", encoding="utf-8") as stream:
            for result in sorted(self.results.values(), key=lambda item: (
                    str(item.get("table_id", "")), item.get("frame_id", -1),
                    item.get("source_span_index", -1), item.get("crop_type", ""))):
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
