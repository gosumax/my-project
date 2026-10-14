"""Sealed native NICK observations provide WHO within an identity segment.

This saved-source inventory has no numeric, status, occupancy or event authority.
It never widens a numeric capture window and never requires a second RGB copy.
"""
import copy
import hashlib
import json
import re
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import cv2

from .actor_seat_point import HEADER, response_text, source_time
from .chat_nick_actor import (RULE, CROSS_RULE, NATIVE_NICK_RULE, SOURCE_KEY,
                             actual_conflict, literal_constraint, matches_hand)
from .current_chat_sources import FILES as CHAT_FILES, load_current_chat
from .nickname_roster_binding import _seat_rois, _verify_gray_source, matches
from .table_service_events import describe_service, service_view
from .nickname_policy import nickname_witness_floor

FILES = (*CHAT_FILES, 'seat_forced_inputs.jsonl', 'seat_forced_summary.json', 'run_metadata.json')
IDENTITY_KINDS = {'EMPTY', 'REPLACE', 'LEAVE', 'JOIN', 'FINISHED',
                  'has left the table', 'has joined the table'}
STATUS_ONLY = {'SIT_IN', 'SIT_OUT', 'sit in', 'sits in', 'sit out', 'sits out'}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def hand_key(row):
    return int(str(row['table_id']).removeprefix('T')), row['session_id'], row['hand_id']


def validate_native_nick(row):
    """Reproduce original gray pixels, fixed geometry, callback key and clock."""
    table = row['table_id']
    if type(table) is not int or row['kind'] != 'NICK':
        raise ValueError('NATIVE_NICK_KIND_OR_TABLE_CONFLICT')
    if type(row['source_frame']) is not int or row['source_frame'] < 0:
        raise ValueError('NATIVE_NICK_FRAME_INVALID')
    if not all(isinstance(row[k], str) and row[k] for k in ('session_id', 'hand_id', 'seat')):
        raise ValueError('NATIVE_NICK_CONTEXT_INVALID')
    seats = _seat_rois()[f'slot_{table:02d}']['seats']
    roi = seats[row['seat']]['nick']
    pixels = _verify_gray_source(row)
    if row['roi'] != roi or pixels.shape != (roi[3]-roi[1], roi[2]-roi[0]):
        raise ValueError('NATIVE_NICK_FIXED_ROI_CONFLICT')
    response = row['ocr_response']
    expected = f"{table}:{row['source_frame']}:{100+list(seats).index(row['seat'])}"
    if response.get('item_id') != expected:
        raise ValueError('NATIVE_NICK_CALLBACK_KEY_CONFLICT')
    stamp = source_time(row['source_pts'], row['time_base'])
    text = response_text(response) if response.get('status') == 'OK' else None
    return stamp, literal_constraint(text) if text is not None else None


class NativeNickWhoSource:
    """Load complete independent parents; retained proof subsets are not inputs."""
    def __init__(self, source, hands, *, expected_hashes=None, nickname_witness_policy=None):
        self.nickname_witness_policy=nickname_witness_policy
        self.nickname_confidence=nickname_witness_floor(nickname_witness_policy)
        self.source = Path(source)
        self.hashes = {name: file_hash(self.source/name) for name in FILES}
        if expected_hashes is not None and any(self.hashes[n] != value
                for n, value in expected_hashes.items() if n in self.hashes):
            raise ValueError('NATIVE_WHO_PARENT_SEAL_CONFLICT')
        metadata = json.loads((self.source/'run_metadata.json').read_text('utf8'))
        self.video_sha256 = metadata['source_video_sha256']
        if not isinstance(self.video_sha256, str) or not re.fullmatch('[0-9a-f]{64}', self.video_sha256):
            raise ValueError('NATIVE_WHO_VIDEO_IDENTITY_UNKNOWN')
        self.hands = {hand_key(h): copy.deepcopy(h) for h in hands}
        if len(self.hands) != len(hands):
            raise ValueError('NATIVE_WHO_DUPLICATE_HAND')
        chat = load_current_chat(self.source, {n: self.hashes[n] for n in CHAT_FILES})
        self.records = defaultdict(list)
        self.checkpoints = defaultdict(list)
        for row in chat['records']:
            self.records[hand_key(row)].append(row)
        for row in chat['context_checkpoints']:
            self.checkpoints[(row['table_id'], row['session_id'])].append(row)
        for entries in self.checkpoints.values():
            frames = [r['frame'] for r in entries]
            if frames != sorted(set(frames)):
                raise ValueError('NATIVE_WHO_CHECKPOINT_ORDER_CONFLICT')
        self.summary = json.loads((self.source/'seat_forced_summary.json').read_text('utf8'))
        self.services, self.service_rejections = self._services(chat)
        self.nicks = defaultdict(list)
        self.errors = defaultdict(list)
        self.source_lines = {}
        ends = defaultdict(lambda: -1)
        clocks = defaultdict(dict)
        # Independent native CHAT pixels carry their own verified source clock.
        # A second RGB rendering of a NICK is not needed for this join.
        for packet in chat['packets'].values():
            for row in packet['ordinary_prefix_physical_rows']:
                owner = (row['table_id'], row['table_session_id'])
                stamp = source_time(row['source_pts'], [row['time_base_num'], row['time_base_den']])
                prior = clocks[owner].get(row['frame_id'])
                if prior is not None and prior != stamp:
                    raise ValueError('NATIVE_WHO_CHAT_FRAME_CLOCK_CONFLICT')
                clocks[owner][row['frame_id']] = stamp
        seen = set()
        with (self.source/'seat_forced_inputs.jsonl').open('rb') as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('kind') != 'NICK':
                    continue
                key = hand_key(row)
                try:
                    stamp, label = validate_native_nick(row)
                    identity = (*key[:2], row['seat'], row['source_frame'])
                    if identity in seen:
                        raise ValueError('NATIVE_WHO_DUPLICATE_NICK')
                    seen.add(identity)
                    prior = clocks[key[:2]].get(row['source_frame'])
                    if prior is not None and prior != stamp:
                        raise ValueError('NATIVE_WHO_FRAME_CLOCK_CONFLICT')
                    clocks[key[:2]][row['source_frame']] = stamp
                    ends[key[:2]] = max(ends[key[:2]], row['source_frame'])
                    self.nicks[key].append((row, label))
                    self.source_lines[(*key, row['seat'], row['source_frame'])] = dict(
                        source_line=lineno, source_line_sha256=hashlib.sha256(line).hexdigest())
                except (KeyError, TypeError, ValueError, IndexError, cv2.error) as error:
                    self.errors[key].append(dict(source_line=lineno, reason=str(error)))
        for table, frames in clocks.items():
            ordered = [frames[f] for f in sorted(frames)]
            if ordered != sorted(set(ordered)):
                for key in self.hands:
                    if key[:2] == table:
                        self.errors[key].append(dict(reason='NATIVE_WHO_CLOCK_ORDER_CONFLICT'))
        self.bounds = {}
        positive = {(r['table_id'], r['session_id'], r['hand_id'], r['frame'])
                    for r in chat['positive_headers']}
        for key, hand in self.hands.items():
            starts = [r['frame_id'] for m in hand['messages'] for r in m['source_rows']
                if HEADER.fullmatch(r['raw_ocr_text'].strip()) and
                HEADER.fullmatch(r['raw_ocr_text'].strip())[1] == key[2] and
                (':R' in str(r.get('stable_row_id')) or ':F' in str(r.get('stable_row_id')))]
            if not starts or (*key, min(starts)) not in positive:
                continue
            low = min(starts)
            checkpoints = self.checkpoints[key[:2]]
            barriers = [r for r in checkpoints if r['frame'] > low and r['hand_id'] != key[2]]
            high = (barriers[0]['frame']-1 if barriers else
                    max([low, ends[key[:2]]] + [r['frame'] for r in checkpoints]))
            self.bounds[key] = (low, high)
        if not self.sources_unchanged():
            raise ValueError('NATIVE_WHO_SOURCE_CHANGED_DURING_LOAD')
        self._inventory_seals = {key: self._inventory_digest(key) for key in self.hands}

    def _services(self, chat):
        """Bind service semantics to verified original rows and current ownership."""
        parents = {}
        for packet in chat['packets'].values():
            for row in packet['ordinary_prefix_timeline']:
                parents[(int(row['table_id'][1:]), row['session_id'], row['frame_id'],
                         row['row_index'], row['stable_row_id'])] = row
        owners = {(r['table_id'], r['session_id'], r['frame'], r['row_index'], r['row_id']): r
                  for r in chat['retained_row_ownership']}
        accepted, rejected = [], []
        for event in service_view(list(self.hands.values()))['events']:
            for observation in event['observations']:
                if observation['kind'] not in {'LEAVE', 'JOIN', 'SIT_IN', 'SIT_OUT', 'FINISHED'}:
                    continue
                rows = observation['source_rows']
                parsed = describe_service(' '.join(r['raw_ocr_text'].strip() for r in rows))
                reasons = []
                if parsed is None or any(parsed[k] != observation[k] for k in ('actor', 'kind')):
                    reasons.append('SERVICE_NATIVE_SEMANTICS_CONFLICT')
                for row in rows:
                    identity = (int(row['table_id'][1:]), row['session_id'], row['frame_id'],
                                row['row_index'], row['stable_row_id'])
                    parent, owner = parents.get(identity), owners.get(identity)
                    if parent is None or any(parent.get(k) != row.get(k) for k in
                            ('raw_ocr_text', 'confidence', 'row_pixel_sha256', 'epoch')):
                        reasons.append('SERVICE_RAW_PARENT_UNKNOWN')
                    if (owner is None or owner['hand_id'] != observation['hand_id'] or
                            owner['first_proven_hand_id'] not in (None, observation['hand_id'])):
                        reasons.append('SERVICE_CURRENT_HAND_OWNER_UNKNOWN')
                    if type(row['confidence']) not in (int, float) or not .95 <= row['confidence'] <= 1:
                        reasons.append('SERVICE_CONFIDENCE_UNKNOWN')
                if len({r['frame_id'] for r in rows}) != 1:
                    reasons.append('SERVICE_FRAME_SPAN_UNKNOWN')
                indices = [r['row_index'] for r in rows]
                if indices != list(range(indices[0], indices[0]+len(indices))):
                    reasons.append('SERVICE_ROW_ADJACENCY_UNKNOWN')
                result = dict(table_id=int(observation['table'][1:]), session_id=observation['session'],
                    hand_id=observation['hand_id'], actor=observation['actor'], kind=observation['kind'],
                    frame=min(r['frame_id'] for r in rows), source_rows=copy.deepcopy(rows),
                    reasons=sorted(set(reasons)))
                (rejected if reasons else accepted).append(result)
        return accepted, rejected

    def sources_unchanged(self):
        try:
            return all(file_hash(self.source/n) == value for n, value in self.hashes.items())
        except OSError:
            return False

    def _inventory_digest(self, key):
        """Bind the cached inputs used for this hand to their original load."""
        payload = dict(hand=self.hands.get(key), bounds=self.bounds.get(key),
            nicks=self.nicks[key], errors=self.errors[key], records=self.records[key],
            checkpoints=self.checkpoints[key[:2]], hashes=self.hashes, video=self.video_sha256,
            summary={name: [r for r in self.summary[name] if (r['table'], r['session']) == key[:2]]
                for name in ('forced_snapshots', 'bound_posts', 'state_receipts', 'unresolved_state_events')},
            services=[r for r in self.services if hand_key(r) == key],
            source_lines=[(identity, value) for identity, value in self.source_lines.items()
                          if identity[:3] == key])
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()

    def inventory_unchanged(self, key):
        try:
            return self._inventory_seals.get(key) == self._inventory_digest(key)
        except (KeyError, TypeError, ValueError, IndexError, AttributeError):
            return False

    def _labels(self, key):
        low, high = self.bounds[key]
        checkpoints = self.checkpoints[key[:2]]
        frames = [r['frame'] for r in checkpoints]
        actors = {r['actor'] for r in self.records[key]}
        result = []
        for row, label in self.nicks[key]:
            frame = row['source_frame']
            if not low <= frame <= high or label is None:
                continue
            index = bisect_right(frames, frame)-1
            if index < 0 or checkpoints[index]['hand_id'] != key[2]:
                continue
            if row['ocr_response']['score'] >= getattr(self, 'nickname_confidence', .95) or any(matches(a, label) for a in actors):
                result.append((row, label))
        return result

    def _barriers(self, key, actor, seat):
        low, high = self.bounds[key]
        result = []
        epoch = None
        for row in sorted(self.summary['state_receipts'], key=lambda r: r['frame']):
            if (row['table'], row['session']) != key[:2] or row['frame'] > high:
                continue
            if row.get('status') != 'APPLIED' or row.get('evidence_status') != 'PROVEN':
                continue
            target = row['seat'] == seat
            other = row.get('actor')
            changed = (target and (row['kind'] in IDENTITY_KINDS or row.get('hand_id') != key[2]
                       or other not in (None, actor))) or (not target and other == actor)
            if target and row.get('epoch') is not None:
                changed = changed or epoch is not None and epoch != row['epoch']
                epoch = row['epoch']
            if changed and row['frame'] >= low:
                result.append(dict(frame=row['frame'], kind=row['kind'], source='STATE', receipt=copy.deepcopy(row)))
        for row in self.summary['unresolved_state_events']:
            if ((row['table'], row['session']) == key[:2] and low <= row['frame'] <= high and
                    (row.get('seat') == seat or row.get('actor') == actor) and
                    (row.get('kind') not in STATUS_ONLY or
                     row.get('seat') == seat and row.get('actor') not in (None, actor) or
                     row.get('actor') == actor and row.get('seat') not in (None, seat))):
                result.append(dict(frame=row['frame'], kind=row['kind'], source='UNRESOLVED_STATE', receipt=copy.deepcopy(row)))
        for row in self.services:
            if (hand_key(row) == key and row['actor'] == actor and
                    row['kind'] not in STATUS_ONLY and low <= row['frame'] <= high):
                result.append(dict(frame=row['frame'], kind=row['kind'], source='NATIVE_SERVICE', receipt=copy.deepcopy(row)))
        return sorted(result, key=lambda r: (r['frame'], r['source']))

    def interval_for(self, key, actor, seat, anchor):
        low, high = self.bounds[key]
        barriers = self._barriers(key, actor, seat)
        if any(row['frame'] == anchor for row in barriers):
            raise ValueError('NATIVE_WHO_ANCHOR_AT_IDENTITY_BARRIER')
        left = [row['frame'] for row in barriers if row['frame'] < anchor]
        right = [row['frame'] for row in barriers if row['frame'] > anchor]
        return [max(left)+1 if left else low, min(right)-1 if right else high], barriers

    def segment_conflict(self, key, actor, seat, interval):
        """Only qualified observations inside this identity segment may conflict."""
        low, high = interval
        inventory = {name: [r for r in self.summary[name]
            if (r['table'], r['session'], r['hand_id']) == key]
            for name in ('forced_snapshots', 'bound_posts')}
        inventory['state_receipts'] = [r for r in self.summary['state_receipts']
            if (r['table'], r['session']) == key[:2] and low <= r['frame'] <= high
            and not (r.get('kind') in STATUS_ONLY and r.get('seat') == seat and
                     r.get('actor') in (None, actor) and r.get('hand_id') == key[2])]
        # Raw status rows remain sealed source evidence. They do not themselves
        # replace a player; different actor/seat and true epoch breaks still do.
        local = {SOURCE_KEY: dict(conflict_inventory=inventory),
                 **dict(zip(('table_id', 'session_id', 'hand_id'), key))}
        known = {r['actor'] for r in self.records[key]}
        for roster in inventory['forced_snapshots']:
            known.update(roster['actors'])
        selected = [(row, label) for row, label in self._labels(key)
                    if low <= row['source_frame'] <= high]
        return actual_conflict(local, actor, seat, selected, known), known

    def for_actor(self, context, actor):
        """Pure replay from freshly loaded parents; callers bracket source seals."""
        def unknown(reason):
            return dict(status='UNKNOWN', rule=NATIVE_NICK_RULE, reason=reason, actor=actor)
        try:
            key = hand_key(context)
            if not self.inventory_unchanged(key):
                return unknown('NATIVE_WHO_CACHED_SOURCE_CHANGED')
            if key not in self.bounds or self.errors[key]:
                return unknown('NATIVE_WHO_HEADER_OR_SOURCE_UNKNOWN')
            source = context[SOURCE_KEY]
            if source.get('policy') != RULE or source.get('cross_source_policy') != CROSS_RULE:
                return unknown('NATIVE_WHO_SOURCE_POLICY_CONFLICT')
            if any(self.hashes[n] != value for n, value in
                    context[SOURCE_KEY]['source_sha256'].items() if n in self.hashes):
                return unknown('NATIVE_WHO_CURRENT_PARENT_CONFLICT')
            labels = self._labels(key)
            matched = [(r, label) for r, label in labels if matches(actor, label)]
            if not matched:
                return unknown('QUALIFIED_NICK_EXACT_OR_PREFIX_UNKNOWN')
            seats = {r['seat'] for r, _ in matched}
            if len(seats) != 1:
                return unknown('OBSERVED_SECOND_SEAT_MATCH')
            seat = next(iter(seats))
            strict = [(r, label) for r, label in matched if r['ocr_response']['score'] >= getattr(self, 'nickname_confidence', .95)]
            candidates = sorted(strict or matched, key=lambda pair: pair[0]['source_frame'])
            last_reason = 'NATIVE_WHO_CURRENT_SEGMENT_CHAT_UNKNOWN'
            for anchor, constraint in candidates:
                try:
                    interval, barriers = self.interval_for(key, actor, seat, anchor['source_frame'])
                except ValueError as error:
                    last_reason = str(error)
                    continue
                low, high = interval
                own = [r for r in self.records[key] if r['actor'] == actor and low <= r['frame'] <= high]
                if not own:
                    continue
                reason, known = self.segment_conflict(key, actor, seat, interval)
                if reason:
                    last_reason = reason
                    continue
                result = dict(status='PROVEN', rule=NATIVE_NICK_RULE, reason=None, actor=actor, seat=seat,
                    **{name: context[name] for name in ('table_id', 'session_id', 'hand_id')},
                    anchor_frame=anchor['source_frame'], nickname_anchor_source=copy.deepcopy(anchor),
                    match_constraint=copy.deepcopy(constraint), chat_actor_sources=copy.deepcopy(own),
                    source_files_sha256=copy.deepcopy(self.hashes), observed_collision_actors=sorted(known),
                    identity_interval=interval, identity_barriers=barriers,
                    native_nick_source=copy.deepcopy(self.source_lines[(*key, seat, anchor['source_frame'])]),
                    source_video_sha256=self.video_sha256, paired_start_stack_rgb_required=False,
                    nick_matching_rule=RULE if strict else CROSS_RULE,
                    start_stack_authority=False, historical_occupancy_authority=False,
                    complete_roster_authority=False, poker_action_authority=False,
                    event_uniqueness_authority=False, new_event_source_required=False,
                    scope='CURRENT_HAND_NATIVE_NICK_IDENTITY_SEGMENT')
                if getattr(self, 'nickname_witness_policy', None) is not None:
                    result['nickname_witness_policy']=self.nickname_witness_policy
                    result['nickname_confidence_min']=self.nickname_confidence
                if matches_hand(result, self.hands[key]):
                    return result
                last_reason = 'CURRENT_HAND_CHAT_SOURCE_JOIN_UNKNOWN'
            return unknown(last_reason)
        except (KeyError, TypeError, ValueError, IndexError, AttributeError, cv2.error) as error:
            return unknown(str(error) if isinstance(error, ValueError) else 'NATIVE_WHO_RECEIPT_INVALID:'+type(error).__name__)

    def assert_use_interval(self, context, point, frames, *, start_frame, end_frame):
        """Validate WHO use independently of the retained numeric payload."""
        key = hand_key(context)
        if (key != hand_key(point) or not self.inventory_unchanged(key) or
                key not in self.bounds or self.errors[key] or point.get('status') != 'PROVEN'):
            raise ValueError('NATIVE_WHO_USE_SOURCE_UNKNOWN')
        if any(self.hashes[n] != value for n, value in point['source_files_sha256'].items()
               if n in self.hashes):
            raise ValueError('NATIVE_WHO_USE_PARENT_CONFLICT')
        anchor = point['anchor_frame']
        original = point['nickname_anchor_source']
        if (original.get('source_frame') != anchor or original.get('seat') != point['seat'] or
                original not in [row for row, _ in self.nicks[key]]):
            raise ValueError('NATIVE_WHO_USE_ANCHOR_SOURCE_CONFLICT')
        interval, _ = self.interval_for(key, point['actor'], point['seat'], anchor)
        conflict, _ = self.segment_conflict(key, point['actor'], point['seat'], interval)
        if conflict:
            raise ValueError(conflict)
        low, high = interval
        if (not frames or any(type(frame) is not int for frame in frames) or
                not low <= start_frame <= min([anchor]+frames) <=
                    max([anchor]+frames) <= end_frame <= high):
            raise ValueError('NATIVE_WHO_IDENTITY_INTERVAL_CONFLICT')

    def frame_binding(self, context, point, frames, *, start_frame, end_frame, single):
        if point != self.for_actor(context, point.get('actor')) or point.get('status') != 'PROVEN':
            raise ValueError('FRESH_NATIVE_WHO_PROOF_REQUIRED')
        count = 1 if single else 3
        if (len(frames) != count or any(type(f) is not int for f in frames) or
                frames != list(range(frames[0], frames[0]+count)) or
                not context['header_frame'] <= frames[0] <= frames[-1] <= context['capture_end_inclusive']):
            raise ValueError('NATIVE_WHO_NUMERIC_WINDOW_CONFLICT')
        low, high = point['identity_interval']
        anchor = point['anchor_frame']
        if not low <= start_frame <= min(frames[0], anchor) <= max(frames[-1], anchor) <= end_frame <= high:
            raise ValueError('NATIVE_WHO_IDENTITY_INTERVAL_CONFLICT')
        return dict(rule='CURRENT_NATIVE_NICK_IDENTITY_NUMERIC_FRAME_BINDING_V1',
            identity_rule=NATIVE_NICK_RULE, actor=point['actor'], seat=point['seat'],
            **{k: point[k] for k in ('table_id', 'session_id', 'hand_id')},
            anchor_frame=anchor, selected_frames=list(frames), valid_interval=[start_frame, end_frame],
            identity_interval=copy.deepcopy(point['identity_interval']),
            native_nick_source=copy.deepcopy(point['native_nick_source']),
            nickname_ocr_rechecked=False, numeric_authority=False, status_authority=False)

    def scope_for_point(self, hand, point):
        """Separate use receipt: an immutable old proof is not hand-wide lineage."""
        actor = point.get('actor')
        result = dict(rule='CURRENT_HAND_IDENTITY_USE_INTERVAL_V1', actor=actor,
                      status='UNKNOWN', reason='NATIVE_WHO_USE_CONTEXT_UNKNOWN')
        try:
            key = hand_key(hand)
            if (hand != self.hands.get(key) or key != hand_key(point) or
                    not self.inventory_unchanged(key) or
                    key not in self.bounds or self.errors[key] or point.get('status') != 'PROVEN'):
                return result
            interval, barriers = self.interval_for(key, actor, point['seat'], point['anchor_frame'])
            conflict, _ = self.segment_conflict(key, actor, point['seat'], interval)
            if conflict:
                result.update(reason=conflict, seat=point['seat'], identity_interval=interval,
                              identity_barriers=barriers)
                return result
            low, high = interval
            own = [r for r in self.records[key] if r['actor'] == actor and low <= r['frame'] <= high]
            if not own or not matches_hand(dict(point, chat_actor_sources=own), hand):
                return result
            # HH assigns one seat to a player. Every source-backed action of
            # that player must fit the same uninterrupted identity interval.
            outside = []
            for message in hand['messages']:
                if message.get('player') != actor:
                    continue
                rows = message.get('source_rows', [])
                if not rows or any(type(r.get('frame_id')) is not int or
                                   not low <= r['frame_id'] <= high for r in rows):
                    outside.append(copy.deepcopy(message))
            statuses = [copy.deepcopy(r) for r in self.services if hand_key(r) == key and
                        r['actor'] == actor and r['kind'] in STATUS_ONLY and low <= r['frame'] <= high]
            result.update(same_actor_status_observations=statuses,
                          status_observations_create_seat_binding=False,
                          seat=point['seat'], identity_interval=interval,
                          identity_barriers=barriers, outside_interval_actions=outside,
                          source_files_sha256=copy.deepcopy(self.hashes),
                          status='UNKNOWN' if outside else 'PROVEN',
                          reason='ACTOR_ACTION_OUTSIDE_IDENTITY_INTERVAL' if outside else None)
            return result
        except (KeyError, TypeError, ValueError, IndexError, AttributeError):
            return result
