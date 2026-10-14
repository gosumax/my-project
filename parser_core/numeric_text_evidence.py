"""Reuse one saved OCR only for byte-identical text in its native temporal frames."""
import base64
import hashlib
import json
import math
import re

import cv2
import numpy as np


RULE = 'SAVED_OCR_EXACT_TEXT_RECTANGLE_NATIVE_TEMPORAL_V1'


def exact_text_temporal_binding(gray, state, observation, frame, gate):
    """Return source equivalence, never a second OCR or first-visibility proof.

    The caller must separately reproduce capture ROI/time/context, frozen config
    hashes and the original SQLite source. No alignment or normalization is used.
    """
    try:
        original = np.frombuffer(base64.b64decode(state['raw_gray_base64'], validate=True), np.uint8).reshape(
            state['height'], state['width'])
        sha = lambda a: hashlib.sha256(a.tobytes()).hexdigest()
        if (gray.dtype != np.uint8 or gray.shape != original.shape
                or sha(original) != state['exact_hash'] or state['raw_gray_sha256'] != state['exact_hash']):
            return None
        first = state['frame_index']
        if (type(frame) is not int or type(first) is not int or frame not in (first + 1, first + 2)
                or observation['frame_index'] != first or observation['confirmed_frame'] != first + 2
                or observation['last_seen_frame'] < first + 2
                or observation['state_id'] != state['state_id'] or state['source_state_id'] != state['state_id']
                or any(observation[k] != state[k] for k in ('table_id', 'seat_id'))):
            return None
        payload = json.loads(state['provenance'])['state_payload']
        saved = json.loads(state['digit_gate_evidence'])
        if (saved != payload['digit_gate_evidence'] or saved['temporal_status'] != 'CONFIRMED'
                or state['status'] != 'CONFIRMED_NUMERIC' or state['digit_gate_status'] != 'DIGITS_VISIBLE'
                or state['ocr_called'] != 1 or type(state['ocr_confidence']) not in (int, float)
                or not math.isfinite(state['ocr_confidence']) or not .95 <= state['ocr_confidence'] <= 1
                or re.fullmatch(r'\d+(?:[ \u00a0]\d{3})*', state['ocr_raw'].strip()) is None
                or state['ocr_raw'].strip().replace(' ', '').replace('\u00a0', '') != state['ocr_numeric']):
            return None
        if any(payload[k] != state[k] for k in ('state_id', 'table_id', 'seat_id', 'frame_index',
                                                'exact_hash', 'ocr_called', 'ocr_raw', 'ocr_numeric', 'ocr_confidence')):
            return None
        temporal = saved['temporal_frames']
        if (len(temporal) != 2 or [r['frame'] for r in temporal] != [first + 1, first + 2]
                or any(r['status'] != 'DIGITS_VISIBLE' for r in temporal)
                or temporal[frame - first - 1]['hash'] != sha(gray)):
            return None
        old_gate = gate.evaluate(cv2.cvtColor(original, cv2.COLOR_GRAY2BGR))
        new_gate = gate.evaluate(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        if old_gate['status'] != 'DIGITS_VISIBLE' or new_gate != old_gate:
            return None
        if any(saved.get(k) != v for k, v in old_gate['evidence'].items() if k != 'temporal_status'):
            return None
        # Use every foreground pixel, including the six edge columns which the
        # frozen presence gate excludes. Preserve spaces and original positions.
        old_mask = original > gate.config['binary_threshold']
        new_mask = gray > gate.config['binary_threshold']
        if not np.array_equal(old_mask, new_mask) or not old_mask.any():
            return None
        ys, xs = np.where(old_mask)
        x, y = max(0, int(xs.min()) - 1), max(0, int(ys.min()) - 1)
        X, Y = min(gray.shape[1], int(xs.max()) + 2), min(gray.shape[0], int(ys.max()) + 2)
        if not np.array_equal(original[y:Y, x:X], gray[y:Y, x:X]):
            return None
        return dict(rule=RULE, evidence_kind='SOURCE_TEXT_PIXELS_IDENTICAL', original_ocr_frame=first,
                    source_state_id=state['state_id'], source_observation_id=observation['observation_id'],
                    original_pixel_sha256=sha(original), target_frame=frame, target_pixel_sha256=sha(gray),
                    text_rectangle=[x, y, X, Y], text_pixel_sha256=sha(original[y:Y, x:X]),
                    full_foreground_mask_sha256=sha(old_mask.astype(np.uint8)),
                    ocr_inference_count=1, new_ocr_performed=False)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, cv2.error):
        return None
