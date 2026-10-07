"""Experimental sidecar over the registered FirstStage; never a production runner.

Images are borrowed for one synchronous call. Only metadata survives that call.
Frame completion (including empty/unchanged chat states) is explicit.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import cv2
import numpy as np
from experiments.capture_temporal_evidence import source_ms, timestamp_text
from experiments.analyze_temporal_evidence import classify


def watermark(meta):
    return (f"T{meta['table_id']:02} | SESSION={meta['session_id']} | "
            f"F={meta['frame_id']} | {timestamp_text(meta['timestamp_ms'])}\n"
            f"ROW={meta['row_key']} | TRACK={meta['stable_row_id']}\n"
            f"FIRST={timestamp_text(meta['first_seen_ms'])} | "
            f"LAST={timestamp_text(meta['last_seen_ms'])} | N={meta['observation_count']}\n"
            f"CLASS={meta.get('temporal_class', 'AMBIGUOUS')}")


def debug_overlay(pixels, metadata):
    """Return an independent RAM canvas. No IO; original pixels are untouched."""
    text = watermark(metadata).splitlines()
    canvas = np.zeros((pixels.shape[0]+22*len(text), max(pixels.shape[1], 1000), 3), np.uint8)
    rgb = cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB) if pixels.ndim == 2 else pixels
    canvas[:pixels.shape[0], :pixels.shape[1]] = rgb
    for i, line in enumerate(text):
        cv2.putText(canvas, line, (4, pixels.shape[0]+18+22*i),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
    return canvas


class CropView:
    __slots__ = ('image', 'metadata')
    def __init__(self, image, metadata):
        self.image = image
        self.metadata = metadata
    def release(self):
        self.image = None


class RamMetadata:
    def __init__(self):
        self.tracks = {}
        self.pending = {}
        self.active = {}
        self.crops = self.live_images = self.peak_images = 0
        self.image_bytes = self.peak_image_bytes = 0
        self.completed_frame = -1

    @contextmanager
    def borrow(self, meta, pixels, stamp):
        """No image cache/copy. Caller must finish this view before capture advances."""
        if self.live_images:
            raise RuntimeError('RAM metadata accepts one synchronous crop at a time')
        fid, sid, rid = meta['frame_id'], meta['session_id'], meta['stable_row_id']
        if fid != self.completed_frame+1:
            raise ValueError('Crop frame must follow completed frame')
        if stamp['source_frame'] != fid or stamp['source_pts'] != meta['source_pts']:
            raise ValueError('Source frame/PTS mismatch')
        ms = source_ms(stamp['source_pts'], stamp['time_base_num'], stamp['time_base_den'])
        track = self.tracks.setdefault(rid, dict(stable_row_id=rid, track_id=rid,
            session_id=sid, table_id=meta['table_id'], first_seen_frame=fid,
            first_seen_ms=ms, first_seen_row_index=meta['source_span_index'],
            origin=meta['origin'], left_censored=meta['origin'] != 'NEW_CANDIDATE',
            visible_spans=[], visibility_gap_count=0, observation_count=0, explicit_view_count=0))
        if (track['session_id'], track['table_id']) != (sid, meta['table_id']):
            raise ValueError('Track identity crossed a table/session')
        state = self.pending.setdefault(sid, {})
        if rid in state:
            raise ValueError('Duplicate track observation in a changed state')
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        track.update(frame_id=fid, source_pts=stamp['source_pts'],
            time_base_num=stamp['time_base_num'], time_base_den=stamp['time_base_den'],
            timestamp_ms=ms, row_key=f"{meta['table_id']}:{fid}:{meta['source_span_index']}",
            source_span_index=meta['source_span_index'], y1=meta['y1'], pixel_hash=digest)
        track['explicit_view_count'] += 1
        state[rid] = True
        # Per-crop metadata is a snapshot; lifetime aggregate is separate.
        view = CropView(pixels, dict(track, last_seen_ms=ms,last_seen_frame=fid,
            visible_spans=tuple(tuple(span) for span in track['visible_spans']),
            observation_count=track['observation_count']+1,
            temporal_class='AMBIGUOUS', temporal_reason='NO_CONFIRMED_CONTEXT_FOR_LIVE_VIEW'))
        self.crops += 1
        self.live_images = 1
        self.image_bytes = pixels.nbytes
        self.peak_images = max(self.peak_images, self.live_images)
        self.peak_image_bytes = max(self.peak_image_bytes, self.image_bytes)
        try:
            yield view
        finally:
            view.release()
            self.live_images = self.image_bytes = 0

    def complete_frame(self, fid, stamp, changed_sessions):
        if self.live_images:
            raise ValueError('Cannot complete frame during an active crop borrow')
        if fid != self.completed_frame+1 or stamp['source_frame'] != fid:
            raise ValueError('Frame completion must be contiguous and ordered')
        for sid in changed_sessions:
            self.active[sid] = tuple(self.pending.pop(sid, {}))
        if self.pending:
            raise ValueError('Observation without a changed chat state')
        for ids in self.active.values():
            for rid in ids:
                track = self.tracks[rid]
                spans = track['visible_spans']
                if spans and spans[-1][1]+1 == fid:
                    spans[-1][1] = fid
                else:
                    spans.append([fid,fid])
                track.update(last_seen_frame=fid, last_seen_ms=stamp['timestamp_ms'],
                             visibility_gap_count=len(spans)-1)
                track['observation_count'] += 1
        self.completed_frame = fid

    def finish(self):
        if self.live_images or self.pending:
            raise ValueError('Unfinished RAM crop/state')
        for track in self.tracks.values():
            track['right_censored'] = track['last_seen_frame'] == self.completed_frame
        self.active.clear()

    def temporal(self, rid, start=None, following=None, **context):
        label, reason = classify(self.tracks[rid], start, following, **context)
        self.tracks[rid].update(temporal_class=label, temporal_reason=reason)
        return label, reason
