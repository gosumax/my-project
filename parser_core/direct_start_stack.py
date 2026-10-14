"""Source-replayed direct first-number proof; no occupancy epoch is created."""
import base64
import copy
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from parser_core.actor_seat_point import HEADER, prove_actor_point, sha, source_time, unique
from parser_core.constant_stack_balance import direct_readings, numeric_sources
from parser_core.first_stack_ocr import effective_numeric, proof_config
from parser_core.joint_stack_sources import FILES, load_contexts
from parser_core.joint_start_stack import own_posts, FORCED_PREFIX_OCR_POLICY
from parser_core.ram_first_stage import png_compatible_gray
from parser_core.start_stack_visibility import VisibilitySource, first_visible_proof
from parser_core.constant_stack_balance import SourceSealBatch, frozen_runtime_files
from parser_core.joint_start_stack import matches_current_hand

from parser_core.chat_nick_actor import prove_actor_point, numeric_nickname_sources, outer_rule
from parser_core.stack_timeline_storage import stack_timeline_source_name

RULE = 'DIRECT_SAVED_FIRST_NUMBER_AND_EXACT_ACTOR_POINT_V1'


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def native_key(row):
    return (int(str(row['table_id']).removeprefix('T')), row['session_id'], str(row['hand_id']))


class FreshDirectWindows:
    """A hand-key selects work; complete native files define every case input."""
    def __init__(self, source, hand_keys, *, actor_policy=False, native_who=None,
                 current_hand_numeric=False):
        self.native_who = native_who
        self.actor_policy = actor_policy
        self.current_hand_numeric = current_hand_numeric
        self.source = Path(source)
        self.database_name = stack_timeline_source_name(self.source)
        names = set(FILES) | {'start_stack_live_receipts.jsonl', 'run_metadata.json', self.database_name}
        if actor_policy:names.add('seat_forced_summary.json')
        self.hashes = {name: file_hash(self.source / name) for name in sorted(names)}
        targets = set(hand_keys)
        starts = {}
        with (self.source / 'ordered_timeline.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                header = HEADER.fullmatch(row.get('raw_ocr_text', '').strip())
                if header is None or row.get('source') in {'FORCED_SNAPSHOT', 'SIDECAR_HEADER'}:
                    continue
                key = (int(str(row['table_id']).removeprefix('T')), row['session_id'], header[1])
                if key in targets:
                    starts[key] = min(starts.get(key, row['frame_id']), row['frame_id'])
        if set(starts) != targets:
            raise ValueError('INDEPENDENT_NATIVE_HEADER_MISSING')
        windows = [dict(table_id=k[0], session_id=k[1], hand_id=k[2], header_frame=f,
                        capture_end_inclusive=f + 64) for k, f in sorted(starts.items())]
        if current_hand_numeric:
            # Independent current-hand bounds, never a serialized HH interval.
            # Initial inventories and successful historical proofs stay intact.
            for w in windows:
                key = native_key(w)
                if (native_who is None or key not in native_who.bounds or
                        not native_who.inventory_unchanged(key)):
                    raise ValueError('FIRST_SAVED_CURRENT_HAND_BOUNDS_UNKNOWN')
                low, high = native_who.bounds[key]
                if low != w['header_frame'] or high < low:
                    raise ValueError('FIRST_SAVED_CURRENT_HAND_BOUNDS_CONFLICT')
                w['capture_end_inclusive'] = high
        loaded = load_contexts(self.source, windows, actor_policy=actor_policy)
        if any(self.hashes[name] != value for name, value in loaded['source_hashes'].items()):
            raise ValueError('CONTEXT_SOURCE_CHANGED_DURING_LOAD')
        self.contexts = {native_key(c): c for c in loaded['contexts']}
        self.captures = defaultdict(list)
        self.direct = defaultdict(list)
        self.observations = defaultdict(list)
        self.states = {}
        for name, destination in [('start_stack_capture.jsonl', self.captures),
                                  ('start_stack_live_receipts.jsonl', self.direct)]:
            with (self.source / name).open(encoding='utf-8') as stream:
                for lineno, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    selected = row.get('field') == 'stack' if name == 'start_stack_capture.jsonl' else row.get('stage') == 'STACK_OCR'
                    if not selected:
                        continue
                    for k, c in self.contexts.items():
                        if ((row.get('table_id'), row.get('session_id')) == k[:2] and
                                c['header_frame'] <= row['source_frame'] <= c['capture_end_inclusive']):
                            destination[(*k, row['seat'])].append(dict(source_line=lineno, raw_record=row))
        metadata = json.loads((self.source / 'run_metadata.json').read_text('utf-8'))
        self.video_sha256 = metadata['source_video_sha256']
        if re.fullmatch('[0-9a-f]{64}', self.video_sha256) is None:
            raise ValueError('VIDEO_IDENTITY_UNKNOWN')
        database = self.source / self.database_name
        with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            self.metadata = [dict(r) for r in db.execute('select key,value from metadata order by key')]
            if [json.loads(r['value']) for r in self.metadata if r['key'] == 'video_sha256'] != [self.video_sha256]:
                raise ValueError('SQL_VIDEO_IDENTITY_CONFLICT')
            for k, c in self.contexts.items():
                observed = [dict(r) for r in db.execute(
                    'select * from observations where table_id=? and frame_index<=? and last_seen_frame>=? order by observation_id',
                    (k[0], c['capture_end_inclusive'], c['header_frame']))]
                for o in observed:
                    self.observations[(*k, o['seat_id'])].append(o)
                for state_id in sorted({o['state_id'] for o in observed}):
                    s = dict(db.execute('select * from visual_states where state_id=?', (state_id,)).fetchone())
                    pixels = s.pop('raw_gray')
                    s.update(raw_gray_base64=base64.b64encode(pixels).decode(), raw_gray_sha256=sha(pixels))
                    self.states[state_id] = s
        if not self.sources_unchanged():
            raise ValueError('SOURCE_CHANGED_DURING_LOAD')

    def sources_unchanged(self):
        return all(file_hash(self.source / name) == value for name, value in self.hashes.items()) and (
            self.native_who is None or self.native_who.sources_unchanged())

    def case_for(self, context_key, actor, seat):
        c = self.contexts[context_key]
        observed = self.observations[(*context_key, seat)]
        return copy.deepcopy(dict(
            table_id=context_key[0], session_id=context_key[1], hand_id=context_key[2], actor=actor, seat=seat,
            source_context=c, full_hand_source_context=c,
            raw_stack_captures=self.captures[(*context_key, seat)],
            raw_direct_stack_readings=[r['raw_record'] for r in self.direct[(*context_key, seat)]],
            raw_sql_observations=observed,
            raw_sql_visual_states=[self.states[s] for s in sorted({o['state_id'] for o in observed})],
            database_metadata=self.metadata,
            database_sha256=self.hashes[self.database_name], source_hashes=self.hashes,
            scope=('CURRENT_HAND_SAVED_NUMERIC_WINDOW; NO_FULL_HAND_LEDGER_CLAIM'
                   if getattr(self, 'current_hand_numeric', False) else
                   'COMPLETE_INITIAL_NATIVE_WINDOW; NO_FULL_HAND_LEDGER_CLAIM')))

    def current_hand_inventory(self, context_key):
        """Fresh current-hand sources, loaded once for UNKNOWN consumers."""
        if self.current_hand_numeric:
            return self
        if self.native_who is None or context_key not in self.native_who.bounds:
            raise ValueError('FIRST_SAVED_CURRENT_HAND_BOUNDS_UNKNOWN')
        if self.native_who.bounds[context_key][0] != self.contexts[context_key]['header_frame']:
            raise ValueError('FIRST_SAVED_CURRENT_HAND_BOUNDS_CONFLICT')
        expanded = getattr(self, '_current_hand_numeric_inventory', None)
        if expanded is None:
            keys = [k for k, c in self.contexts.items() if k in self.native_who.bounds and
                    self.native_who.bounds[k][0] == c['header_frame']]
            expanded = FreshDirectWindows(self.source, keys,
                actor_policy=self.actor_policy, native_who=self.native_who,
                current_hand_numeric=True)
            # The outer source-seal batch brackets the identical parent files.
            if expanded.hashes != self.hashes:
                raise ValueError('FIRST_SAVED_CURRENT_HAND_SOURCE_SEAL_CONFLICT')
            self._current_hand_numeric_inventory = expanded
        return expanded

    def matches(self, case):
        k = native_key(case)
        return k in self.contexts and case == self.case_for(k, case['actor'], case['seat'])


def verify_direct_item_ids(case):
    """Replay the submitted row key for every callback, including rejections."""
    _, slots = proof_config()
    seats = list(slots[f"slot_{case['table_id']:02d}"]['seats'])
    position = seats.index(case['seat'])
    for reading in case['raw_direct_stack_readings']:
        expected = f"{case['table_id']}:{reading['source_frame']}:{position}"
        if reading.get('ocr_response', {}).get('item_id') != expected:
            raise ValueError('DIRECT_OCR_CALLBACK_ITEM_CONFLICT')


def full_readings(case):
    """Keep every captured frame; absence of OCR is explicitly NOT_READ."""
    verify_direct_item_ids(case)
    direct_readings(case)  # Independently join every accepted primary/scaled receipt.
    c = case['source_context']
    caps = unique([r['raw_record'] for r in case['raw_stack_captures']], lambda r: r['source_frame'])
    raw_direct = unique(case['raw_direct_stack_readings'], lambda r: r['source_frame'])
    native_nicks = unique([r for r in c['native_capture_nick_rows'] if r['seat'] == case['seat']], lambda r: r['source_frame'])
    readings = {}
    verifier = VisibilitySource()
    for frame, cap in sorted(caps.items()):
        if (any(cap.get(k) != case[k] for k in ('table_id', 'session_id', 'seat')) or cap.get('field') != 'stack' or
                not c['header_frame'] <= frame <= c['capture_end_inclusive']):
            raise ValueError('CAPTURE_CONTEXT_CONFLICT')
        nick = native_nicks[frame]
        if source_time(cap['source_pts'], cap['time_base']) != source_time(nick['source_pts'], nick['time_base']):
            raise ValueError('STACK_NICK_SOURCE_TIME_CONFLICT')
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(cap['png_base64'], validate=True), np.uint8), 1), cv2.COLOR_BGR2RGB)
        x, y, X, Y = cap['roi']
        a, b, A, B = verifier.rois[f"slot_{case['table_id']:02d}"]['seats'][case['seat']]['stack']
        if rgb.shape != (Y - y, X - x, 3) or sha(rgb.tobytes()) != cap['pixel_sha256'] or not x <= a < A <= X or not y <= b < B <= Y:
            raise ValueError('CAPTURE_PIXELS_OR_ROI_CONFLICT')
        if frame in raw_direct:
            row = copy.deepcopy(raw_direct[frame])
            if any(row.get(k) != cap.get(k) for k in ('table_id', 'session_id', 'seat', 'source_frame', 'source_pts', 'time_base', 'roi')):
                raise ValueError('RAW_DIRECT_CAPTURE_CONTEXT_CONFLICT')
            if (row.get('hand_id') != case['hand_id'] or row.get('status') == 'SOURCE_CONTEXT_UNKNOWN' or
                    row['visibility_source']['source_rgb_sha256'] != cap['pixel_sha256']):
                raise ValueError('RAW_DIRECT_CAPTURE_IDENTITY_CONFLICT')
            gray = cv2.imdecode(np.frombuffer(base64.b64decode(row['png_base64'], validate=True), np.uint8), 0)
            if gray is None or not np.array_equal(gray, png_compatible_gray(rgb)):
                raise ValueError('RAW_DIRECT_NATIVE_PIXELS_CONFLICT')
        else:
            gray = png_compatible_gray(rgb)
            ok, png = cv2.imencode('.png', gray)
            if not ok:
                raise ValueError('SOURCE_ENCODING_FAILED')
            projected = cv2.cvtColor(rgb[b-y:B-y, a-x:A-x], cv2.COLOR_RGB2GRAY)
            row = dict(table_id=case['table_id'], session_id=case['session_id'], hand_id=case['hand_id'], seat=case['seat'],
                       source_frame=frame, source_pts=cap['source_pts'], time_base=cap['time_base'], roi=cap['roi'],
                       png_base64=base64.b64encode(png).decode(), pixel_sha256=sha(gray.tobytes()),
                       raw_text='', confidence=None, ocr_status='NOT_READ', source_only=True,
                       visibility_source=dict(source_rgb_roi=cap['roi'], source_rgb_sha256=cap['pixel_sha256'],
                           source_rgb_png_base64=cap['png_base64'], timeline_roi=[a,b,A,B],
                           timeline_pixel_sha256=sha(projected.tobytes()), timeline_shape=list(projected.shape),
                           digit_gate=verifier.gate.evaluate(cv2.cvtColor(projected, cv2.COLOR_GRAY2BGR))))
        if verifier.verify(row) is None:
            raise ValueError('SOURCE_VISIBILITY_PROJECTION_CONFLICT')
        readings[frame] = row
    return readings


def verify_sql_conflicts(sql_readings, first_frame, frames, displayed):
    checked = []
    for frame, row in sorted(sql_readings.items()):
        if not first_frame <= frame <= frames[-1] or row['ocr_status'] != 'OK':
            continue
        value = int(row['raw_text'].replace(' ', '').replace('\u00a0', ''))
        if frame < frames[0]:
            raise ValueError('INDEPENDENT_SQL_EARLIER_ACCEPTED_NUMBER')
        if value != displayed:
            raise ValueError('INDEPENDENT_SQL_NUMERIC_CONFLICT')
        checked.append(dict(frame=frame, value=value, evidence=row['numeric_source_evidence']))
    return checked


def _prove_body(case, inventory, check_seals):
    def unknown(reason):
        return dict(status='UNKNOWN', rule=RULE, reason=reason, actor=case.get('actor'), scope='SOURCE_REPLAYED_DIRECT_FIRST_NUMBER_NO_OCCUPANCY_EPOCH')
    try:
        if inventory is None or not inventory.matches(case):
            return unknown('FRESH_COMPLETE_WINDOW_INVENTORY_REQUIRED')
        if check_seals() is not None:
            return unknown('SOURCE_SEAL_CHANGED')
        context = case['source_context']
        point = prove_actor_point(context, case['actor'], native_who=getattr(inventory, 'native_who', None))
        if point['status'] != 'PROVEN':
            return unknown('ACTOR_POINT_' + point['reason'])
        if any(point.get(k) != case[k] for k in ('table_id', 'session_id', 'hand_id', 'actor', 'seat')):
            return unknown('ACTOR_CONTEXT_CONFLICT')
        posts = own_posts(context, case['actor'])
        readings = full_readings(case)
        from parser_core.first_stack_ocr import direct_numeric_policy
        policy=direct_numeric_policy()
        from parser_core.chat_nick_actor import (reuses_cross_identity,
            first_consecutive_frames, cross_identity_frame_binding)
        identity_binding = None
        if reuses_cross_identity(point):
            from parser_core.start_stack_visibility import NUMERIC
            frames = first_consecutive_frames([f for f, r in readings.items()
                if NUMERIC.fullmatch((r.get('raw_text') or '').strip()) and
                effective_numeric(r, policy['min_confidence'], allow_scaled=True,
                                  numeric_policy=policy) is not None],
                'FIRST_NUMERIC_TRIPLE_UNKNOWN')
            identity_binding = cross_identity_frame_binding(context, point, frames,
                start_frame=context['header_frame'], end_frame=max(frames[-1], point['anchor_frame']),
                native_who=getattr(inventory, 'native_who', None))
        else:
            frames = [r['source_frame'] for r in numeric_nickname_sources(case['source_context'],point)]
        from parser_core.chat_nick_actor import assert_native_identity_interval
        native_who=getattr(inventory,'native_who',None)
        assert_native_identity_interval(context,point,frames,start_frame=context['header_frame'],
            end_frame=max(frames[-1],point['anchor_frame']) if native_who is not None else frames[-1],native_who=native_who)
        first = first_visible_proof(readings, frames, start_frame=context['header_frame'],
                                    confidence=policy['min_confidence'],numeric_policy=policy)
        if first['status'] != 'PROVEN':
            return unknown('FIRST_VISIBLE_' + first['reason'])
        displayed = int(first['displayed_stack'])
        if displayed <= 0:
            return unknown('POSITIVE_NUMBER_REQUIRED')
        sql, _, _, _ = numeric_sources(case, case['database_metadata'])
        checked = verify_sql_conflicts(sql, context['header_frame'], frames, displayed)
        if check_seals() is not None:
            return unknown('SOURCE_SEAL_CHANGED')
        return dict(status='PROVEN', rule=outer_rule(RULE,point), reason=None, actor=case['actor'], seat=case['seat'],
                    table_id=case['table_id'], session_id=case['session_id'], hand_id=case['hand_id'],
                    displayed_stack=displayed, forced_amount_added_back=sum(p['amount'] for p in posts),
                    calculated_start_stack=displayed + sum(p['amount'] for p in posts),
                    identity_proof=point, own_forced_posts=posts, first_visible_proof=first,
                    **({'identity_frame_binding': identity_binding} if identity_binding is not None else {}),
                    forced_prefix_ocr_policy=copy.deepcopy(FORCED_PREFIX_OCR_POLICY),
                    numeric_frames=frames, independent_sql_sources_checked=checked,
                    source_hashes=copy.deepcopy(inventory.hashes), source_case=copy.deepcopy(case),
                    numeric_authority='DIRECT_SAVED_STACK_OCR_NATIVE_PIXELS', occupancy_epoch=None,
                    manufactured_sql_observation_ids=False, scope='SOURCE_REPLAYED_DIRECT_FIRST_NUMBER_NO_OCCUPANCY_EPOCH')
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError, ZeroDivisionError, OSError, cv2.error) as error:
        return unknown(str(error) if isinstance(error, ValueError) else type(error).__name__ + ':' + str(error))


def prove(case, inventory):
    """Standalone use rehashes the complete source inventory before and after."""
    def seal():
        return None if inventory.sources_unchanged() else 'SOURCE_SEAL_CHANGED'
    return _prove_body(case, inventory, seal)


def proof_id(proof):
    return sha(json.dumps(proof, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8'))


def join_receipt(proof):
    from parser_core.first_visible_single_numeric import is_single_proof,join_receipt as single_join_receipt
    if is_single_proof(proof):
        return single_join_receipt(proof)
    readings = {r['source_frame']: r for r in proof['source_case']['raw_direct_stack_readings']}
    sources = [dict(source_frame=f, source_pts=readings[f]['source_pts'], time_base=readings[f]['time_base'],
                    item_id=readings[f]['ocr_response']['item_id'], pixel_sha256=readings[f]['pixel_sha256'],
                    source_rgb_sha256=readings[f]['visibility_source']['source_rgb_sha256'])
               for f in proof['numeric_frames']]
    return dict(rule=proof['rule'], proof_id=proof_id(proof), actor=proof['actor'], seat=proof['seat'],
                table_id=proof['table_id'], session_id=proof['session_id'], hand_id=proof['hand_id'],
                calculated_start_stack=proof['calculated_start_stack'], displayed_stack=proof['displayed_stack'],
                forced_amount_added_back=proof['forced_amount_added_back'], numeric_frames=proof['numeric_frames'],
                forced_prefix_ocr_policy=copy.deepcopy(proof['forced_prefix_ocr_policy']),
                numeric_ocr_policy=copy.deepcopy(proof['first_visible_proof']['numeric_ocr_policy']),
                native_numeric_sources=sources, source_files_sha256=proof['source_hashes'],
                scope='DIRECT_FIRST_NUMBER_NO_SQL_OBSERVATION_OR_OCCUPANCY_EPOCH')


class DirectStartStackProvider:
    """Saved FULL only: independently replay native direct OCR first numbers."""
    def __init__(self, source, records, joint_provider, registry, *, sitout_provider=None, actor_policy=False,
                 single_first_numeric=False, native_who=None):
        self.single_first_numeric=single_first_numeric
        self.registry = registry
        self.runtime_files = frozen_runtime_files()
        self.cases = {}; self.proofs = {}; self.audit = []; self.inventory = None
        by_key = {native_key(r['observed_hand']): copy.deepcopy(r) for r in records}
        if len(by_key) != len(records):
            raise ValueError('DIRECT_DUPLICATE_HAND_CONTEXT')
        candidates = defaultdict(set)
        for row in joint_provider.rows:
            if row.get('point_status') != 'PROVEN' or row.get('status') != 'UNKNOWN':
                continue
            context = (row['table_id'], row['session_id'], row['hand_id'])
            if sitout_provider is not None and sitout_provider.excludes_stack_work(context,row['actor']):
                continue
            if context in by_key and by_key[context]['lifecycle']['status'] == 'COMPLETE':
                candidates[context].add(row['actor'])
        if not candidates:
            return
        self.inventory = FreshDirectWindows(source, sorted(candidates), actor_policy=actor_policy, native_who=native_who)
        batch = SourceSealBatch(self.registry, self.inventory, self.runtime_files)
        if not batch.begin():
            raise ValueError('DIRECT_BATCH_PRE_SOURCE_SEAL_CHANGED')
        pending_cases = {}; pending_proofs = {}; pending_audit = []
        for context, actors in candidates.items():
            for actor in sorted(actors):
                point = prove_actor_point(self.inventory.contexts[context], actor, native_who=self.inventory.native_who)
                if point['status'] != 'PROVEN':
                    pending_audit.append(dict(hand_key=list(context), actor=actor, status='UNKNOWN',
                                              reason='ACTOR_POINT_' + point['reason']))
                    continue
                case = self.inventory.case_for(context, actor, point['seat'])
                record = by_key[context]
                result = self._prove(case, record, batch=batch)
                pending_audit.append(dict(hand_key=list(context), actor=actor, seat=point['seat'],
                                          status=result['status'], reason=result['reason'],
                                          calculated_start_stack=result.get('calculated_start_stack')))
                if result['status'] == 'PROVEN':
                    pending_cases[(context, actor)] = (case, record)
                    pending_proofs[(context, actor)] = result
        if not batch.finish():
            raise ValueError('DIRECT_BATCH_POST_SOURCE_SEAL_CHANGED')
        self.cases = pending_cases; self.proofs = pending_proofs; self.audit = pending_audit

    def _prove(self, case, record, *, batch=None):
        if batch is None:
            result = prove(case, self.inventory)
        elif batch.registry is self.registry and batch.inventory is self.inventory:
            result = _prove_body(case, self.inventory, lambda: batch.check_phase('proof'))
        else:
            raise ValueError('DIRECT_BATCH_INVENTORY_CONFLICT')
        if getattr(self,'single_first_numeric',False) and result['status']=='UNKNOWN':
            from parser_core.first_visible_single_numeric import prove as single_prove
            # Keep the normal source-seal bracket in standalone and batched calls.
            if batch is None:
                seal=lambda:None if self.inventory.sources_unchanged() else 'SOURCE_SEAL_CHANGED'
            else:
                seal=lambda:batch.check_phase('proof')
            result=single_prove(case,self.inventory,seal)
            if result.get('reason') == 'FIRST_SINGLE_NUMERIC_SOURCE_UNKNOWN':
                try:
                    expanded = self.inventory.current_hand_inventory(native_key(case))
                    expanded_case = expanded.case_for(native_key(case), case['actor'], case['seat'])
                    result = single_prove(expanded_case, expanded, seal)
                except (ValueError, OSError) as error:
                    result = dict(status='UNKNOWN', rule=RULE, actor=case['actor'],
                                  reason=str(error))
        if result['status'] == 'PROVEN' and not matches_current_hand(result, record['observed_hand']):
            return dict(status='UNKNOWN', rule=RULE, reason='CURRENT_NATIVE_HAND_SOURCE_CONFLICT', actor=case['actor'])
        return result

    def for_hand(self, hand):
        context = native_key(hand)
        cases = [(actor, case, record) for (saved, actor), (case, record) in self.cases.items()
                 if saved == context and record['observed_hand'] == hand]
        if not cases:
            return []
        batch = SourceSealBatch(self.registry, self.inventory, self.runtime_files)
        if not batch.begin():
            return []
        result = []
        for actor, case, record in cases:
            proof = self._prove(case, record, batch=batch)
            if proof['status'] == 'PROVEN' and proof == self.proofs[(context, actor)]:
                result.append(proof)
        return result if batch.finish() else []

    def write_audit(self, output):
        from collections import Counter
        output = Path(output)
        with (output / 'direct_start_stack_proofs.jsonl').open('w', encoding='utf-8') as stream:
            for proof in self.proofs.values():
                stream.write(json.dumps(proof, ensure_ascii=False) + '\n')
        (output / 'direct_start_stack_audit.json').write_text(json.dumps(dict(
            scope='SOURCE_REPLAYED_DIRECT_FIRST_NUMBER_NO_OCCUPANCY_EPOCH',
            counts=dict(Counter(r['status'] for r in self.audit)), rows=self.audit,
            source_hashes=self.inventory.hashes if self.inventory else {}), ensure_ascii=False, indent=2), 'utf-8')
