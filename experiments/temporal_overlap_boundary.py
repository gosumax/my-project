"""EXPERIMENT ONLY: fence unchanged glyph/overlap dedup by observed poker hand.

Birth context is immutable. Tracking epoch is deliberately not a poker epoch.
Context conflicts are barriers, never inherited anchors or gallery templates.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import base64
import re
import time
import numpy as np
from experiments import temporal_overlap_dedup as baseline
from experiments.temporal_overlap_guarded import Observation, Reducer as VisualReducer, guarded_metrics

HAND = re.compile(r'^(?:Dealer:\s*)?(?:Hand\s*)?#(\d{6,})$', re.I)
START = re.compile(r'^(?:Dealer:\s*)?Starting a new hand$', re.I)


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

    def resolve(self, rid):
        context = self.birth.get(rid)
        core = self.cores.get(context)
        return core.resolve(rid) if core else rid

    def process(self, table, frame, observations):
        started = time.perf_counter()
        session = self.sessions[table]
        ordered = sorted(observations, key=lambda o: o.view.index)
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
            core = self.cores.setdefault(ctx, VisualReducer(gallery=True))
            before = len(core.removals)
            core.process(table, frame, segment)
            for removal in core.removals[before:]:
                rid, prior = removal['old_row_id'], removal['matched_previous_row_id']
                all_ids = [rid, prior, removal['matched_previous_observation']['stable_row_id'],
                           *removal['anchor_row_ids']]
                all_ids += [i for pair in removal['overlap_pairs']
                            for i in (pair['current_row_id'], pair['previous_row_id'])]
                assert all(self.birth.get(i) == ctx for i in all_ids)
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
                self.cores[ctx].histories[table].append(baseline.View(
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
