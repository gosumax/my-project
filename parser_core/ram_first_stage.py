"""Shared, file-free first stage for the Smoke14 benchmark and pipeline."""

from __future__ import annotations

import hashlib
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np

from parser_core.chat_rows import (Alignment, align_upward, canonical_row,
                                   exclude_first_visible_span, match_row_blocks,
                                   match_rows, padded_row_span, physical_spans)
from parser_core.layout import load_fixed_layout


HISTORY_STATES = 12


@dataclass(frozen=True)
class RamRow:
    frame_id: int
    video_time: float | None
    table_id: int
    session_id: str
    source_span_index: int
    y1: int
    origin: str
    epoch: int
    pixels: np.ndarray
    pixel_sha256: str
    stable_row_id: str


class FirstStage:
    def __init__(self, video: Path, sessions: dict[str, str], frames: int, *,
                 root: Path, telemetry=None):
        self.video = video
        self.sessions = sessions
        self.frames = frames
        self.root = root
        self.telemetry = telemetry
        self.metrics = Counter()
        self.changed: set[tuple[int, str, str]] = set()
        self.detected: dict[tuple[int, str, int], tuple[str, str]] = {}
        self.seen_frames = 0
        self.last_video_time = None

    def _measure(self, name: str, started: float, objects: int = 1):
        elapsed = time.perf_counter() - started
        self.metrics[name + "_seconds"] += elapsed
        if self.telemetry is not None:
            self.telemetry.add(name, elapsed, objects)

    @staticmethod
    def _stable_id(session_id: str, counters: Counter) -> str:
        counters[session_id] += 1
        return f"{session_id}:R{counters[session_id]:06d}"

    def rows(self):
        layout = load_fixed_layout(self.root / "configs/nine_table_layout.json",
                                   self.root / "configs/layout_847x404.json",
                                   self.root / "configs/chat_log_bottom_rois.json")
        previous_rgb = {}
        chat_history: dict[str, deque] = {}
        stable_counters = Counter()
        with av.open(str(self.video)) as source:
            stream = source.streams.video[0]
            iterator = enumerate(source.decode(stream))
            while self.seen_frames < self.frames:
                tick = time.perf_counter()
                try:
                    fid, frame = next(iterator)
                except StopIteration:
                    break
                self._measure("decode", tick)
                tick = time.perf_counter()
                rgb = frame.to_ndarray(format="rgb24")
                self._measure("frame_to_ndarray", tick)
                video_time = float(frame.pts * (frame.time_base or stream.time_base)) if frame.pts is not None else None
                self.last_video_time = video_time
                for slot in layout.slots:
                    sid = self.sessions[str(slot.slot_id)]
                    for roi, box in (("table", slot.table_box), ("chat", slot.chat_box)):
                        tick = time.perf_counter()
                        x1, y1, x2, y2 = box
                        crop = rgb[y1:y2, x1:x2]
                        self._measure(roi + "_roi_slice", tick)
                        key = (sid, roi)
                        tick = time.perf_counter()
                        same = key in previous_rgb and np.array_equal(crop, previous_rgb[key])
                        self._measure("change_comparison", tick)
                        if same:
                            self.metrics["unchanged_rois"] += 1
                            continue
                        previous_rgb[key] = np.ascontiguousarray(crop)
                        self.changed.add((fid, sid, roi))
                        self.metrics[f"changed_{roi}_rois"] += 1
                        if roi == "table":
                            continue

                        tick = time.perf_counter()
                        # Pixel-compatible with PIL RGB PNG -> OpenCV grayscale,
                        # encoded and decoded entirely in process memory.
                        ok, packed = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                                                  [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        if not ok:
                            raise ValueError("RAM chat conversion failed")
                        gray = cv2.imdecode(packed, cv2.IMREAD_GRAYSCALE)
                        if gray is None:
                            raise ValueError("RAM chat grayscale decode failed")
                        self._measure("ram_gray_conversion", tick)

                        tick = time.perf_counter()
                        all_spans = physical_spans(gray)
                        self._measure("physical_spans", tick)
                        tick = time.perf_counter()
                        spans, suppressed = exclude_first_visible_span(all_spans)
                        self._measure("exclude_first_visible_span", tick)

                        history = chat_history.setdefault(sid, deque(maxlen=HISTORY_STATES))
                        previous = history[-1] if history else None
                        if previous is None:
                            matches: dict[int, int] = {}
                            alignment = None
                            epoch = 0
                        else:
                            old_spans = previous["spans"]
                            tick = time.perf_counter()
                            zero_matches = match_rows(previous["gray"], old_spans, gray, spans, 0)
                            self._measure("match_rows_zero", tick)
                            if len(zero_matches) == len(spans) == len(old_spans):
                                alignment = Alignment(0, 0.0, 0.0, True)
                            else:
                                tick = time.perf_counter()
                                alignment = align_upward(previous["gray"], gray)
                                self._measure("align_upward", tick)
                            if alignment.shift_px == 0:
                                proposed = zero_matches
                            else:
                                tick = time.perf_counter()
                                proposed = match_rows(previous["gray"], old_spans, gray,
                                                      spans, alignment.shift_px)
                                self._measure("match_rows_shifted", tick)
                            minimum = (max(1, min(len(old_spans), len(spans)) // 2)
                                       if alignment.shift_px == 0 else 3)
                            if alignment.proven and len(proposed) < minimum:
                                alignment = Alignment(alignment.shift_px, alignment.score,
                                                      alignment.zero_score, False)
                            matches = proposed if alignment.proven else {}
                            epoch = previous["epoch"] + (0 if alignment.proven else 1)

                        # First carry identities proven against the immediately preceding state.
                        identity: dict[int, tuple[str, str]] = {}
                        if previous is not None:
                            for new_index, old_index in matches.items():
                                identity[new_index] = (previous["stable_ids"][old_index],
                                                       previous["origins"][old_index])

                        # Hard transitions are exactly where the old tracker produced large
                        # UNRESOLVED blocks.  Search a short history for a *contiguous sequence*
                        # of already-visible rows.  Single-line matches are deliberately rejected
                        # so a legitimate repeated poker line cannot be erased by itself.
                        history_proven = False
                        current_canonical = None
                        unmatched_count = len(spans) - len(identity)
                        need_history = (previous is not None and unmatched_count > 0 and
                                        (alignment is None or not alignment.proven or
                                         unmatched_count > max(2, len(spans) // 4)))
                        if need_history and spans:
                            tick = time.perf_counter()
                            current_canonical = [canonical_row(gray, span) for span in spans]
                            used_stable_ids = {stable_id for stable_id, _ in identity.values()}
                            for depth, snapshot in enumerate(reversed(history), 1):
                                if len(identity) == len(spans):
                                    break
                                old_canonical = snapshot.get("canonical")
                                if old_canonical is None:
                                    old_canonical = [canonical_row(snapshot["gray"], span)
                                                     for span in snapshot["spans"]]
                                    snapshot["canonical"] = old_canonical
                                blocks = match_row_blocks(old_canonical, current_canonical)
                                accepted_in_snapshot = 0
                                for block in blocks:
                                    for new_index, old_index in block.mapping.items():
                                        if new_index in identity:
                                            continue
                                        stable_id = snapshot["stable_ids"][old_index]
                                        if stable_id in used_stable_ids:
                                            continue
                                        identity[new_index] = (stable_id,
                                                               snapshot["origins"][old_index])
                                        used_stable_ids.add(stable_id)
                                        accepted_in_snapshot += 1
                                if accepted_in_snapshot:
                                    history_proven = True
                                    self.metrics["history_matched_rows"] += accepted_in_snapshot
                                    self.metrics[f"history_depth_{depth}_rows"] += accepted_in_snapshot
                            self._measure("history_row_identity", tick, len(spans))
                            self.metrics["history_match_calls"] += 1

                        if history_proven and previous is not None and not alignment.proven:
                            # Identity was recovered despite the viewport alignment failure;
                            # do not start a new epoch merely because the immediate state jumped.
                            epoch = previous["epoch"]

                        current_spans: list[tuple[int, int]] = []
                        current_origins: list[str] = []
                        current_stable_ids: list[str] = []
                        matched_indices = sorted(identity)
                        last_matched_index = matched_indices[-1] if matched_indices else -1

                        for index, span in enumerate(spans):
                            source_index = index + len(suppressed)
                            if index in identity:
                                stable_id, inherited_origin = identity[index]
                                current_spans.append(span)
                                current_origins.append(inherited_origin)
                                current_stable_ids.append(stable_id)
                                self.metrics["stable_rows_suppressed"] += 1
                                continue

                            if previous is None:
                                origin = "INITIAL_VISIBLE"
                            else:
                                if alignment is not None and alignment.proven:
                                    newly_exposed = (
                                        (alignment.shift_px > 0 and
                                         span[0] >= gray.shape[0] - alignment.shift_px - 8)
                                        or (alignment.shift_px == 0 and previous["spans"] and
                                            span[0] > max(s[1] for s in previous["spans"]) + 5)
                                    )
                                elif history_proven:
                                    # PokerDom chat appends new text at the bottom.  Once a
                                    # historical overlap has been re-established, an unmatched
                                    # suffix below that overlap is genuinely newly exposed.
                                    newly_exposed = index > last_matched_index
                                else:
                                    newly_exposed = False
                                origin = "NEW_CANDIDATE" if newly_exposed else "UNRESOLVED_IDENTITY"

                            tick = time.perf_counter()
                            row_span = padded_row_span(all_spans, source_index, gray.shape[0])
                            self._measure("padded_row_span", tick)
                            pixels = np.ascontiguousarray(gray[row_span[0]:row_span[1]])
                            tick = time.perf_counter()
                            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
                            self._measure("row_hash", tick)
                            stable_id = self._stable_id(sid, stable_counters)
                            self.detected[(fid, sid, source_index)] = (origin, digest)
                            current_spans.append(span)
                            current_origins.append(origin)
                            current_stable_ids.append(stable_id)
                            yield RamRow(fid, video_time, slot.slot_id, sid, source_index,
                                         row_span[0], origin, epoch, pixels, digest, stable_id)

                        history.append({"gray": gray,
                                        "spans": current_spans,
                                        "origins": current_origins,
                                        "stable_ids": current_stable_ids,
                                        "epoch": epoch,
                                        "frame_id": fid,
                                        "canonical": current_canonical})
                self.seen_frames += 1
