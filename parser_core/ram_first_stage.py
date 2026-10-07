"""Shared, file-free first stage for the Smoke14 benchmark and pipeline."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np

from parser_core.chat_rows import padded_row_span
from parser_core.chat_tracker import AdjacentChatTracker, CONFIRMED_LINK_STATES
from parser_core.layout import load_fixed_layout
from parser_core.visual_crops import snapshot_ram_board
from parser_core import performance_diagnostics as diagnostics


HISTORY_STATES = 12


def png_compatible_gray(rgb: np.ndarray) -> np.ndarray:
    """Exact integer grayscale used by this pipeline's PNG decoder."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected native uint8 RGB pixels")
    wide = rgb.astype(np.uint32)
    return ((wide[:, :, 0]*9797 + wide[:, :, 1]*19234 + wide[:, :, 2]*3737) >> 15).astype(np.uint8)


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
    source_pts: int | None = None
    time_base_num: int | None = None
    time_base_den: int | None = None
    identity_status: str | None = None
    admission_ref: str | None = None


class FirstStage:
    def __init__(self, video: Path, sessions: dict[str, str], frames: int, *,
                 root: Path, telemetry=None, retain_board: bool = False,
                 board_profile: Path | None = None, identity_observer=None,
                 decode_threads: int = 0, tracker_observer=None, frame_observer=None,
                 physical_admission=False, admission_observer=None):
        if decode_threads < 0:
            raise ValueError('Decode thread count must be nonnegative')
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
        self.retain_board = retain_board
        self.board_profile = board_profile or root / "configs/layout_847x404.json"
        self.board_snapshots = {}
        self.identity_observer = identity_observer
        self.decode_threads = decode_threads
        self.tracker_observer = tracker_observer
        self.frame_observer = frame_observer
        self.chat_trackers = {}
        self.chat_epochs = Counter()
        self.physical_admission = physical_admission
        self.admission_observer = admission_observer
        self.physical_admission_lanes = {}

    def _measure(self, name: str, started: float, objects: int = 1):
        if isinstance(started,tuple):
            diagnostics.ACTIVE.record(name,started,objects,
                kind='container' if name in {'frame_loop_total','table_loop_total'} else 'compute')
            started=started[0]
        elapsed = time.perf_counter() - started
        self.metrics[name + "_seconds"] += elapsed
        if self.telemetry is not None:
            self.telemetry.add(name, elapsed, objects)

    def _tick(self):
        return diagnostics.ACTIVE.tick() if diagnostics.ACTIVE else time.perf_counter()

    @staticmethod
    def _stable_id(session_id: str, counters: Counter) -> str:
        counters[session_id] += 1
        return f"{session_id}:R{counters[session_id]:06d}"

    def chat_rows(self, gray, *, frame_id, table_id, session_id,
                  source_pts=None, time_base=None, video_time=None, source_chat_box=None):
        """Shared live/frozen entry into the promoted tracker and OCR admission.

        The legacy admission path remains intact. When explicitly enabled,
        the separate physical lane may emit proved new rows on rejected gates.
        Diagnostic callbacks receive copies and cannot mutate tracker state.
        """
        tracker = self.chat_trackers.get(session_id)
        if tracker is None:
            tracker = self.chat_trackers[session_id] = AdjacentChatTracker(table_id, session_id)
        if tracker.table_id != table_id:
            raise ValueError('One CHAT session cannot contain different tables')
        tick = self._tick()
        transition = tracker.process(gray, frame_id)
        self._measure('chat_tracker', tick, len(transition['new_ids']))
        classification = transition['classification']
        self.metrics[classification + '_COUNT'] += 1
        self.metrics['OCR_NEW_BOTTOM_ROWS'] += len(transition['OCR_queue_row_ids'])
        if classification == 'UNRESOLVED':
            self.chat_epochs[session_id] += 1
        epoch = self.chat_epochs[session_id]
        digest = hashlib.sha256(gray.tobytes()).hexdigest()
        transition.update(source_pts=source_pts,
                          time_base=[time_base.numerator, time_base.denominator] if time_base is not None else None,
                          source_chat_sha256=digest, source_chat_box=source_chat_box,
                          session_id=session_id, epoch=epoch)
        if self.tracker_observer is not None:
            self.tracker_observer(deepcopy(transition))
        supplemental = {}
        if self.physical_admission:
            from parser_core.physical_chat_admission import PhysicalChatAdmission
            lane = self.physical_admission_lanes.get(session_id)
            if lane is None:
                lane = self.physical_admission_lanes[session_id] = PhysicalChatAdmission(table_id, session_id)
            tick = self._tick()
            receipt = lane.process(gray, transition)
            self._measure('physical_chat_admission',tick)
            if receipt is not None:
                receipt.update(source_pts=source_pts, time_base=transition['time_base'], source_chat_box=source_chat_box)
                supplemental = {a['index']: a for a in receipt['admissions']}
                self.metrics['OCR_PHYSICAL_NEW_ROWS'] += len(supplemental)
                if self.admission_observer is not None:
                    self.admission_observer(receipt)
        admitted = set(transition['OCR_queue_row_ids'])
        all_spans = transition['all_spans']
        edge_count = len(transition['new_edge_spans'])
        for index, (span, stable_id, link_state) in enumerate(zip(
                transition['new_observed_spans'], transition['new_ids'], transition['new_link_states'])):
            source_index = index + edge_count
            extra = supplemental.get(index)
            if extra is not None:
                stable_id = extra['row_id']
            if extra is None and stable_id not in admitted:
                self.metrics['stable_rows_suppressed'] += 1
                if (stable_id is not None and self.identity_observer is not None
                        and not tracker.journal[stable_id]['initial']):
                    view_span = padded_row_span(all_spans, source_index, gray.shape[0])
                    confirmed = link_state in CONFIRMED_LINK_STATES
                    origin = 'NEW_CANDIDATE' if confirmed else 'UNRESOLVED_OBJECT_LINK'
                    self.identity_observer({
                        'frame_id': frame_id, 'table_id': table_id, 'session_id': session_id,
                        'source_span_index': source_index, 'stable_row_id': stable_id,
                        'origin': origin, 'epoch': epoch, 'source_pts': source_pts,
                        'y1': view_span[0], 'view_type': 'INHERITED_STABLE_ROW' if confirmed else 'UNRESOLVED_OBJECT_LINK',
                        'identity_confirmed': confirmed, 'identity_status': link_state},
                        np.array(gray[view_span[0]:view_span[1]], copy=True, order='C'))
                continue
            tick = self._tick()
            row_span = padded_row_span(all_spans, source_index, gray.shape[0])
            self._measure('padded_row_span', tick)
            pixels = np.ascontiguousarray(gray[row_span[0]:row_span[1]])
            tick = self._tick()
            pixel_digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            self._measure('row_hash', tick)
            origin = 'NEW_CANDIDATE'
            self.detected[(frame_id, session_id, source_index)] = (origin, pixel_digest)
            yield RamRow(frame_id, video_time, table_id, session_id, source_index,
                         row_span[0], origin, epoch, pixels, pixel_digest, stable_id,
                         source_pts, time_base.numerator if time_base is not None else None,
                         time_base.denominator if time_base is not None else None,
                         'UNKNOWN' if extra else None,
                         f'{session_id}:{frame_id}:physical_new:{index}' if extra else None)

    def rows(self):
        layout = load_fixed_layout(self.root / 'configs/nine_table_layout.json',
                                   self.root / 'configs/layout_847x404.json',
                                   self.root / 'configs/chat_log_bottom_rois.json')
        previous_rgb = {}
        profile = json.loads(self.board_profile.read_text('utf-8')) if self.retain_board else None
        with av.open(str(self.video)) as source:
            stream = source.streams.video[0]
            if self.decode_threads:
                stream.thread_type = 'FRAME'
                stream.codec_context.thread_count = self.decode_threads
            iterator = enumerate(source.decode(stream))
            while self.seen_frames < self.frames:
                frame_tick=self._tick() if diagnostics.ACTIVE else None
                tick = self._tick()
                try:
                    fid, frame = next(iterator)
                except StopIteration:
                    break
                self._measure('decode', tick)
                tick = self._tick()
                rgb = frame.to_ndarray(format='rgb24')
                if (frame.width, frame.height) != (layout.source_width, layout.source_height):
                    raise ValueError('Source frame dimensions differ from fixed layout')
                self._measure('frame_to_ndarray', tick)
                base = frame.time_base or stream.time_base
                video_time = float(frame.pts*base) if frame.pts is not None else None
                self.last_video_time = video_time
                for slot in layout.slots:
                    table_tick=self._tick() if diagnostics.ACTIVE else None
                    if diagnostics.ACTIVE:diagnostics.ACTIVE.local.table=slot.slot_id
                    sid = self.sessions[str(slot.slot_id)]
                    if self.frame_observer is not None:
                        tx1,ty1,tx2,ty2=slot.table_box;cx1,cy1,cx2,cy2=slot.chat_box
                        self.frame_observer(slot.slot_id,sid,fid,frame.pts,
                            [base.numerator,base.denominator],rgb[ty1:ty2,tx1:tx2],
                            png_compatible_gray(rgb[cy1:cy2,cx1:cx2]))
                    for roi, box in (('table', slot.table_box), ('chat', slot.chat_box)):
                        tick = self._tick()
                        x1, y1, x2, y2 = box
                        crop = rgb[y1:y2, x1:x2]
                        self._measure(roi + '_roi_slice', tick)
                        key = (sid, roi)
                        tick = self._tick()
                        same = key in previous_rgb and np.array_equal(crop, previous_rgb[key])
                        if diagnostics.ACTIVE:
                            diagnostics.ACTIVE.count('change_comparison_operand_bytes',crop.nbytes*2 if key in previous_rgb else 0)
                            diagnostics.ACTIVE.count('change_comparison_operand_pixels',crop.shape[0]*crop.shape[1])
                            diagnostics.ACTIVE.count('change_comparison_no_change' if same else 'change_comparison_changed')
                            diagnostics.ACTIVE.count(f'T{slot.slot_id:02d}/changed_{roi}_ROI',not same)
                            diagnostics.ACTIVE.record('change_comparison_unchanged' if same else 'change_comparison_changed',tick)
                        self._measure('change_comparison', tick)
                        if same:
                            self.metrics['unchanged_rois'] += 1
                            continue
                        previous_rgb[key] = np.ascontiguousarray(crop)
                        self.changed.add((fid, sid, roi))
                        self.metrics[f'changed_{roi}_rois'] += 1
                        if roi == 'table':
                            continue
                        tick = self._tick()
                        gray = png_compatible_gray(crop)
                        self._measure('ram_gray_conversion', tick)
                        for row in self.chat_rows(gray, frame_id=fid, table_id=slot.slot_id,
                                                  session_id=sid, source_pts=frame.pts,
                                                  time_base=base, video_time=video_time, source_chat_box=list(box)):
                            if self.retain_board and (sid, fid) not in self.board_snapshots:
                                tick = self._tick()
                                tx1, ty1, tx2, ty2 = slot.table_box
                                self.board_snapshots[(sid, fid)] = snapshot_ram_board(
                                    rgb[ty1:ty2, tx1:tx2], profile['slots'][f'slot_{slot.slot_id:02d}'],
                                    table_id=slot.slot_id, table_session_id=sid, frame_id=fid,
                                    source_pts=frame.pts, time_base_num=base.numerator,
                                    time_base_den=base.denominator, source_table_box=list(slot.table_box))
                                self._measure('ram_board_snapshot', tick)
                            yield row
                    if diagnostics.ACTIVE:self._measure('table_loop_total',table_tick)
                self.seen_frames += 1
                if diagnostics.ACTIVE:
                    diagnostics.ACTIVE.local.table=None
                    self._measure('frame_loop_total',frame_tick)
