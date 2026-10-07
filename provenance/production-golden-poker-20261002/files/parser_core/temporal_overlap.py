"""Native temporal overlap with immutable hand/session fences.

Promoted from the source-backed boundary replay. No OCR/pixel normalization,
no cross-hand deletion, and UNKNOWN context is always KEEP.
"""
from __future__ import annotations
import base64
import hashlib
import re
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass
import cv2
import numpy as np

GLYPH_NCC_MIN = 0.90
MAX_VIEWS = 256
HAND = re.compile(r'^(?:Dealer:\s*)?(?:Hand\s*)?#(\d{6,})$', re.I)
START = re.compile(r'^(?:Dealer:\s*)?Starting a new hand$', re.I)

@dataclass
class View:
    row_id: str
    signature: np.ndarray
    pixel_hash: str
    y1: int
    index: int
    inherited: bool
    origin: str
    text: str

def anchored_duplicates(history: list[View], block: list[View]):
    """Align by known stable identities first; visual/text checks only confirm.

    At least two inherited anchors must imply the same diagonal mapping into
    the published tail. Conflicting anchors reject the entire proposal. Rows
    beyond the anchored contiguous visual run are kept, including new suffixes.
    """
    positions = {v.row_id: i for i, v in enumerate(history)}
    offsets = [positions[v.row_id] - i for i, v in enumerate(block)
               if v.inherited and v.row_id in positions]
    if len(offsets) < 2 or len(set(offsets)) != 1:
        return []
    offset = offsets[0]
    runs, run = [], []
    for i, v in enumerate(block):
        j = offset + i
        old = history[j] if 0 <= j < len(history) else None
        distance = native_distance(old.signature, v.signature) if old else float('inf')
        valid = old is not None and distance <= 9.0
        if v.inherited:
            valid = valid and old.row_id == v.row_id
        elif v.text and old and old.text:
            valid = valid and v.text == old.text
        if valid:
            run.append((i, j, distance))
        else:
            if run:
                runs.append(run)
            run = []
    if run:
        runs.append(run)
    out = []
    for run in runs:
        anchors = [block[i].row_id for i, _, _ in run if block[i].inherited]
        if len(anchors) < 2:
            continue
        for i, j, distance in run:
            v = block[i]
            if not v.inherited and v.origin == 'UNRESOLVED_IDENTITY':
                out.append((v, history[j], {
                    'reason': 'UNIQUE_ANCHORED_CONTIGUOUS_PUBLISHED_HISTORY_OVERLAP',
                    'exact_hash_match': v.pixel_hash == history[j].pixel_hash,
                    'visual_match': True, 'canonical_mae': distance,
                    'visual_threshold': 9.0, 'overlap_length': len(run),
                    'anchor_row_ids': anchors,
                    'overlap_pairs': [{'current_row_id': block[a].row_id,
                                       'previous_row_id': history[b].row_id,
                                       'canonical_mae': d} for a, b, d in run],
                }))
    return out

def native_signature(pixels):
    if pixels.ndim != 2 or pixels.size == 0:
        return np.empty((0, 0), np.uint8)
    # The same existing scrollbar exclusion, without modifying source pixels.
    return pixels[:, :-12].copy() if pixels.shape[1] >= 50 else pixels.copy()

def native_distance(old, new):
    if old.size == 0 or new.size == 0 or old.shape != new.shape:
        return float('inf')
    h = old.shape[0]
    distances = []
    for dy in (-1, 0, 1):
        a = old[max(0, dy):min(h, h + dy)]
        b = new[max(0, -dy):min(h, h - dy)]
        if a.size:
            distances.append(float(np.mean(cv2.absdiff(a, b))))
    return min(distances, default=float('inf'))

def glyph_metrics(old, new):
    if old.shape != new.shape:
        return {'shape_match': False}
    h = old.shape[0]
    comparisons = []
    for dy in (-1, 0, 1):
        a = old[max(0, dy):min(h, h + dy)]
        b = new[max(0, -dy):min(h, h - dy)]
        dropped = ((old[:dy] > 70).any() or (new[h-dy:] > 70).any()) if dy > 0 else (
            ((old[h+dy:] > 70).any() or (new[:-dy] > 70).any()) if dy < 0 else False)
        if dropped:
            continue
        union = np.uint8((a > 70) | (b > 70))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(union, 8)
        locals_ = []
        for x, y, w, ht, area in stats[1:]:
            if area < 5:
                continue
            x0, x1, y0, y1 = max(0, x-1), min(a.shape[1], x+w+1), max(0, y-1), min(a.shape[0], y+ht+1)
            aa, bb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
            ncc = float(cv2.matchTemplate(aa, bb, cv2.TM_CCOEFF_NORMED)[0, 0]) if aa.std() and bb.std() else -1.0
            binary_a, binary_b = aa > 70, bb > 70
            dice = 2 * np.count_nonzero(binary_a & binary_b) / max(1, np.count_nonzero(binary_a) + np.count_nonzero(binary_b))
            locals_.append({'x': int(x), 'y': int(y), 'width': int(w), 'height': int(ht),
                            'area': int(area), 'ncc': ncc, 'dice': dice})
        comparisons.append({'dy': dy, 'mae': float(np.mean(cv2.absdiff(a, b))),
                            'min_glyph_ncc': min((r['ncc'] for r in locals_), default=-1),
                            'min_glyph_dice': min((r['dice'] for r in locals_), default=0),
                            'glyphs': locals_})
    return min(comparisons, key=lambda r: r['mae']) if comparisons else {'all_shifts_discard_ink': True}

@dataclass
class Observation:
    view: View
    frame_id: int
    pixels: np.ndarray

def guarded_metrics(old, new):
    metrics = glyph_metrics(old, new)
    if 'mae' not in metrics or not metrics['glyphs']:
        return None
    if metrics['mae'] > 9 or metrics['min_glyph_ncc'] < GLYPH_NCC_MIN:
        return None
    return metrics

class Reducer:
    def __init__(self, gallery=False):
        self.use_gallery = gallery
        self.histories = defaultdict(list)
        self.aliases = {}
        self.gallery = defaultdict(dict)
        self.first = {}
        self.removals = []
        self.seconds = 0.0
        self.block_count = 0

    def resolve(self, rid):
        seen = set()
        while rid in self.aliases:
            if rid in seen:
                raise ValueError('Identity alias cycle')
            seen.add(rid)
            rid = self.aliases[rid]
        return rid

    def process(self, table, frame, observations):
        start = time.perf_counter()
        block = []
        by_original = {}
        for obs in sorted(observations, key=lambda o: o.view.index):
            v = obs.view
            rid = self.resolve(v.row_id) if v.inherited else v.row_id
            copy = View(rid, v.signature, v.pixel_hash, v.y1, v.index,
                                 v.inherited, v.origin, v.text)
            block.append(copy)
            by_original[copy.row_id] = obs
        history = self.histories[table]
        proposals = anchored_duplicates(history[-256:], block)
        removed = set()
        for current, previous, evidence in proposals:
            # Unknown or conflicting OCR cannot be used to confirm a new alias.
            if not current.text.strip() or not previous.text.strip() or current.text != previous.text:
                continue
            templates = list(self.gallery[previous.row_id].values()) if self.use_gallery else [self.first[previous.row_id]]
            # Rank cheaply, then validate glyphs. Never use current/future views.
            templates = [obs for obs in templates if obs.frame_id < frame]
            templates.sort(key=lambda obs: native_distance(obs.view.signature, current.signature))
            selected, metrics = None, None
            for obs in templates:
                if native_distance(obs.view.signature, current.signature) > 9:
                    break
                candidate = guarded_metrics(obs.view.signature, current.signature)
                if candidate is not None:
                    selected, metrics = obs, candidate
                    break
            if selected is None:
                continue
            self.aliases[current.row_id] = previous.row_id
            removed.add(current.row_id)
            chosen = selected.view
            evidence = dict(evidence)
            evidence['proposal_native_mae'] = evidence.pop('canonical_mae')
            for pair in evidence['overlap_pairs']:
                pair['native_overlap_mae'] = pair.pop('canonical_mae')
            self.removals.append({'table_id': f'T{table:02d}', 'frame_id': frame,
                'text': current.text, 'y1': current.y1, 'source_span_index': current.index,
                'old_row_id': current.row_id, 'matched_previous_row_id': previous.row_id,
                **evidence, 'exact_hash_match': current.pixel_hash == chosen.pixel_hash,
                'distance_metric': 'native_overlap_mae_and_per_component_ncc',
                'native_overlap_mae': metrics['mae'], 'matched_shift_y': metrics['dy'],
                'min_glyph_ncc': metrics['min_glyph_ncc'], 'glyph_ncc_min': GLYPH_NCC_MIN,
                'glyphs': metrics['glyphs'], 'nonempty_exact_ocr_match': True,
                'active_edge_pixels_discarded': False,
                'current_pixel_sha256': current.pixel_hash, 'previous_pixel_sha256': chosen.pixel_hash,
                'matched_previous_observation': {'frame_id': selected.frame_id,
                    'stable_row_id': chosen.row_id, 'source_span_index': chosen.index,
                    'y1': chosen.y1, 'inherited': chosen.inherited,
                    'pixel_shape': list(selected.pixels.shape),
                    'row_pixels_base64': base64.b64encode(selected.pixels.tobytes()).decode('ascii')},
                'gallery_enabled': self.use_gallery})
        history.extend(v for v in block if not v.inherited and v.row_id not in removed)
        # Add observations only AFTER this block's decisions; never self-match.
        for original in observations:
            rid = self.resolve(original.view.row_id)
            if rid not in self.first:
                self.first[rid] = original
            if self.use_gallery:
                views = self.gallery[rid]
                views.setdefault(original.view.pixel_hash, original)
                if len(views) > MAX_VIEWS:
                    # First observation remains available separately.
                    del views[next(iter(views))]
                views.setdefault(self.first[rid].view.pixel_hash, self.first[rid])
        self.seconds += time.perf_counter() - start
        self.block_count += 1

def hand_number(text):
    match = HAND.fullmatch(text.strip())
    return match.group(1) if match else None

@dataclass(frozen=True)
class Context:
    table: int
    session: str
    hand: str

    def evidence(self):
        return {'table_id': f'T{self.table:02d}', 'session_id': self.session,
                'hand_id': self.hand, 'hand_epoch': f'{self.session}:H{self.hand}'}

class BoundaryReducer:
    def __init__(self, sessions):
        self.sessions = {int(k): v for k, v in sessions.items()}
        self.birth = {}  # includes explicit None for UNKNOWN, never upgraded
        self.proof = {}
        self.marker_templates = {}
        self.active = {}
        self.ambiguous = defaultdict(bool)
        self.cores = {}
        self.removals = []
        self.decisions = []
        self.seconds = 0.0
        self.block_count = 0
        self.barriers = 0
        self.last_frames = {}

    def resolve(self, rid):
        context = self.birth.get(rid)
        core = self.cores.get(context)
        return core.resolve(rid) if core else rid

    def process(self, table, frame, observations):
        started = time.perf_counter()
        session = self.sessions[table]
        ordered = sorted(observations, key=lambda o: o.view.index)
        if frame <= self.last_frames.get(table, -1):
            raise ValueError('Nonmonotonic dedup observation frame')
        self.last_frames[table] = frame
        if len({o.view.index for o in ordered}) != len(ordered):
            raise ValueError('Duplicate visible span index')
        if any(o.frame_id != frame for o in ordered):
            raise ValueError('Mixed dedup observation frame')
        if any(not o.view.row_id.startswith(session + ':') for o in ordered):
            raise ValueError('Observation/session mismatch')
        numbers = []
        marker_metrics = []
        for obs in ordered:
            number = hand_number(obs.view.text)
            metrics = None
            if number and obs.view.inherited:
                original = self.marker_templates.get(obs.view.row_id)
                metrics = guarded_metrics(original.view.signature, obs.view.signature) if original else None
                if metrics is None:
                    number = None
            elif number:
                self.marker_templates.setdefault(obs.view.row_id, obs)
            numbers.append(number)
            marker_metrics.append(metrics)
        visible = [int(h) for h in numbers if h]
        inversion = any(a > b for a, b in zip(visible, visible[1:]))
        active = self.active.get(table)
        # Reused action IDs cannot prove that a markerless block is one hand.
        cursor = None
        evidence_class = 'UNKNOWN_CONTEXT'
        assignments = []
        for pos, obs in enumerate(ordered):
            v = obs.view
            number = numbers[pos]
            marker_like = (hand_number(v.text) is not None
                           or 'hand #' in v.text.lower() or v.text.strip().startswith('#'))
            if number and not inversion:
                ctx = Context(table, session, number)
                if active is None or int(number) > int(active.hand):
                    active = ctx
                    self.active[table] = ctx
                    self.ambiguous[table] = False
                cursor = ctx if ctx == active else None  # old scrollback cannot roll live state back
                evidence_class = 'SAME_HAND_SPATIAL_MARKER' if cursor else 'OLDER_VISIBLE_HAND'
                if cursor:
                    self.proof[ctx] = {'row_id': v.row_id, 'frame_id': frame,
                        'source_span_index': v.index, 'pixel_sha256': v.pixel_hash,
                        'pixel_shape': list(obs.pixels.shape),
                        'row_pixels_base64': base64.b64encode(obs.pixels.tobytes()).decode('ascii'),
                        'inherited': v.inherited,
                        'original_marker_sha256': self.marker_templates[v.row_id].view.pixel_hash,
                        'inherited_marker_min_glyph_ncc': marker_metrics[pos]['min_glyph_ncc'] if marker_metrics[pos] else None}
                    self.ambiguous[table] = False
            elif marker_like and not number:
                cursor = None
                self.ambiguous[table] = True
                evidence_class = 'MALFORMED_HAND_MARKER'
            elif not v.text.strip():
                # An unreadable row may itself be a new hand marker. Never
                # carry an earlier spatial marker across missing/failed OCR.
                cursor = None
                self.ambiguous[table] = True
                evidence_class = 'UNREADABLE_CONTEXT_BARRIER'
            elif START.fullmatch(v.text.strip()):
                # Starting is not an identifier. An immediately adjacent number
                # may corroborate an already observed SAME hand in this frame.
                adjacent_number = (pos + 1 < len(ordered)
                          and ordered[pos + 1].view.index == v.index + 1
                          and numbers[pos + 1])
                if cursor is None and active is not None and adjacent_number == active.hand:
                    cursor = active
                    evidence_class = 'SAME_FRAME_ADJACENT_NUMBER'
                paired = cursor is not None and adjacent_number == cursor.hand
                if not paired:
                    cursor = None
                    self.ambiguous[table] = True
                    evidence_class = 'UNNUMBERED_STARTING_BOUNDARY'
            if inversion:
                cursor = None
                self.ambiguous[table] = True
                evidence_class = 'NONMONOTONIC_VISIBLE_MARKERS'
            # Decide spatial context before resolving an inherited identity.
            if v.row_id not in self.birth and not v.inherited:
                self.birth[v.row_id] = cursor
            original_context = self.birth.get(v.row_id)
            compatible = cursor is not None and original_context == cursor
            reason = evidence_class if compatible else (
                'INHERITED_HAND_CONTEXT_CONFLICT' if cursor and original_context and v.inherited
                else 'KEEP_UNKNOWN_OR_CONFLICTING_HAND_CONTEXT')
            assignments.append((obs, cursor, compatible, reason))

        segment = []
        segment_context = None

        def flush():
            nonlocal segment, segment_context
            if not segment:
                return
            ctx = segment_context
            core = self.cores.setdefault(ctx, Reducer(gallery=True))
            before = len(core.removals)
            core.process(table, frame, segment)
            for removal in core.removals[before:]:
                rid, prior = removal['old_row_id'], removal['matched_previous_row_id']
                all_ids = [rid, prior, removal['matched_previous_observation']['stable_row_id'],
                           *removal['anchor_row_ids']]
                all_ids += [i for pair in removal['overlap_pairs']
                            for i in (pair['current_row_id'], pair['previous_row_id'])]
                if not all(self.birth.get(i) == ctx for i in all_ids):
                    raise ValueError('Dedup hand-context proof mismatch')
                removal.update(ctx.evidence())
                removal.update({'previous_hand_id': ctx.hand,
                                'previous_hand_epoch': ctx.evidence()['hand_epoch'],
                                'previous_session_id': ctx.session,
                                'context_evidence_class': 'KNOWN_SAME_HAND_STRONG_OVERLAP',
                                'hand_marker_evidence': dict(self.proof[ctx]),
                                'duplicate_reason': 'SAME_TABLE_SESSION_HAND_AND_UNCHANGED_VISUAL_OVERLAP_GUARDS'})
                self.removals.append(removal)
            removed_here = {r['old_row_id'] for r in core.removals[before:]}
            for obs in segment:
                if not obs.view.inherited:
                    self.decisions.append({'row_id': obs.view.row_id, 'table_id': f'T{table:02d}',
                        'frame_id': frame, **ctx.evidence(),
                        'decision': 'DUPLICATE' if obs.view.row_id in removed_here else 'KEEP',
                        'reason': 'SAME_HAND_GUARDS' if obs.view.row_id in removed_here else 'NO_PROVEN_SAME_HAND_DUPLICATE'})
            segment = []
            segment_context = None

        def barrier(ctx):
            if ctx in self.cores:
                self.barriers += 1
                # Retain physical separation in the historical matching tail.
                self.cores[ctx].histories[table].append(View(
                    f'BARRIER:{table}:{frame}:{self.barriers}', np.empty((0, 0), np.uint8),
                    '', -1, -1, True, 'CONTEXT_BARRIER', ''))

        for obs, ctx, compatible, reason in assignments:
            if not compatible:
                prior_segment = segment_context
                flush()
                barrier(prior_segment)
                barrier(ctx)
                if not obs.view.inherited:
                    self.decisions.append({'row_id': obs.view.row_id, 'table_id': f'T{table:02d}',
                        'frame_id': frame, 'session_id': session, 'hand_id': ctx.hand if ctx else None,
                        'hand_epoch': ctx.evidence()['hand_epoch'] if ctx else None,
                        'decision': 'KEEP', 'reason': reason})
                continue
            if segment_context is not None and segment_context != ctx:
                prior_segment = segment_context
                flush()
                barrier(prior_segment)
            segment_context = ctx
            segment.append(obs)
        flush()
        self.seconds += time.perf_counter() - started
        self.block_count += 1


class TemporalObservations:
    """Keep visible appearances in compressed RAM until real OCR has drained.

    Dedup sees the capture order, including inherited rows omitted by FirstStage.
    Each appearance retains its own native pixels, never a later/future crop.
    """
    def __init__(self):
        self.records = []
        self.compressed_bytes = 0

    def record(self, table, frame, rid, index, y, origin, pixels, inherited):
        raw = pixels.tobytes(order='C')
        packed = zlib.compress(raw, level=1)
        self.records.append((table, frame, rid, index, y, origin, pixels.shape,
                             hashlib.sha256(raw).hexdigest(), packed, inherited))
        self.compressed_bytes += len(packed)

    def observe(self, metadata, pixels):
        self.record(metadata['table_id'], metadata['frame_id'], metadata['stable_row_id'],
                    metadata['source_span_index'], metadata['y1'], metadata['origin'], pixels, True)

    def emitted(self, row):
        self.record(row.table_id, row.frame_id, row.stable_row_id, row.source_span_index,
                    row.y1, row.origin, row.pixels, False)

    def reduce(self, timeline, sessions, *, ocr_status=None):
        by_id = {row['stable_row_id']: row for row in timeline}
        if len(by_id) != len(timeline):
            raise ValueError('Duplicate physical stable row ID')
        reducer = BoundaryReducer(sessions)
        block, key = [], None
        for table, frame, rid, index, y, origin, shape, digest, packed, inherited in self.records:
            group = (table, frame)
            if key != group:
                if block:
                    reducer.process(*key, block)
                block, key = [], group
            pixels = np.frombuffer(zlib.decompress(packed), np.uint8).reshape(shape).copy()
            row = by_id.get(rid)
            if row is None:
                raise ValueError('Visible identity has no physical/OCR provenance: ' + rid)
            # ERROR/empty OCR is evidence of uncertainty, never confirmation.
            failed = (ocr_status.get(rid, 'ERROR') == 'ERROR' if ocr_status is not None
                      else row['status'] == 'ERROR')
            text = row['raw_ocr_text'] if not failed else ''
            block.append(Observation(View(rid, native_signature(pixels), digest, y,
                              index, inherited, origin, text), frame, pixels))
        if block:
            reducer.process(*key, block)
        removed = {row['old_row_id'] for row in reducer.removals}
        if not removed <= by_id.keys():
            raise ValueError('Removed identity missing physical provenance')
        retained = [row for row in timeline if row['stable_row_id'] not in removed]
        for removal in reducer.removals:
            stamp = by_id[removal['old_row_id']]['video_time']
            removal['timestamp_ms'] = stamp * 1000 if stamp is not None else None
        self.records.clear()
        # Matching is complete. Keep JSON evidence, release native galleries
        # before downstream analysis and workbook export allocate their RAM.
        reducer.cores.clear()
        reducer.marker_templates.clear()
        return retained, reducer
