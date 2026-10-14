"""Source-bound SB absence from reviewed evidence or a complete physical prefix."""
from __future__ import annotations

import copy
import base64
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np

RULE = 'AUTHORIZED_LEGITIMATE_NO_SB_V1'
REVIEW_SCHEMA = 'legitimate-no-sb-review-v1'
NOMINAL_RULE = 'POKERDOM_NOMINAL_SB_HALF_NORMAL_BB_V1'
PHYSICAL_PREFIX_RULE = 'SINGLE_SNAPSHOT_CONTIGUOUS_CHAT_FORCED_PREFIX_V1'
SOURCE_METADATA_FIELDS = ('source_id','source_video_sha256','layout_id',
                          'layout_sha256','profile_sha256','calibration_sha256')


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _stamp(path):
    s = Path(path).stat()
    return s.st_size, s.st_mtime_ns, s.st_ctime_ns


def prefix_digest(rows):
    """Pin native observations, independently of derived economic messages."""
    fields = ('table_id', 'session_id', 'frame_id', 'row_index', 'stable_row_id',
              'raw_ocr_text', 'confidence', 'row_pixel_sha256', 'epoch', 'video_time')
    return hashlib.sha256(json.dumps([{k:r.get(k) for k in fields} for r in rows],
        sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class LegitimateNoSBSource:
    """Native prefix and current WHO, with manual or physical absence evidence.

    Reviews are separate input data, never hand flags or synthetic native pixels.
    This first contract covers one vacant position immediately after the button,
    followed by one normal BB, a complete observed ante prefix, and no all-in post.
    Without a review file, only a contiguous single-snapshot rendered prefix
    at the registered fixed line cadence can establish absence automatically.
    That path supplies neither a button nor an open-seat assertion.
    """
    def __init__(self, source, actor_provider, review_path=None):
        self.source = Path(source).resolve()
        self.actor_provider = actor_provider
        self.review_path = Path(review_path).resolve() if review_path is not None else None
        self.initial_error = None
        self.reviews = []
        if self.review_path is not None:
            try:
                self.review_hash = _hash(self.review_path)
                self.review_stamp = _stamp(self.review_path)
                self.reviews = [json.loads(line) for line in self.review_path.read_text('utf8').splitlines() if line.strip()]
            except (OSError,ValueError) as error:
                self.initial_error = str(error)
        self.loaded = False
        self.last_rejection = None

    def _metadata_path(self):
        capture=self.source/'capture_metadata.json'
        return capture if capture.exists() else self.source/'run_metadata.json'

    def _check_metadata_scope(self):
        if self.metadata_path.name=='capture_metadata.json':
            current=json.loads((self.source/'run_metadata.json').read_text('utf8'))
            if any(current.get(k)!=self.metadata.get(k) for k in SOURCE_METADATA_FIELDS):
                raise ValueError('NO_SB_CAPTURE_METADATA_SCOPE_CONFLICT')

    def _load(self):
        from .current_chat_sources import FILES, load_current_chat, observation_source_rows
        from .simple_saved_policy import LineageActorProvider, saved_nick_observations
        from .simple_seat_who import SeatIdentityLineage
        if self.loaded:
            if self._metadata_path()!=self.metadata_path:
                raise ValueError('NO_SB_METADATA_SOURCE_CHANGED')
            if any(_stamp(p) != v for p,v in self.stamps.items()):
                raise ValueError('NO_SB_SOURCE_CHANGED')
            self._check_metadata_scope()
            return
        if not isinstance(self.actor_provider, LineageActorProvider):
            raise ValueError('CURRENT_SIMPLE_WHO_REQUIRED')
        self.metadata_path=self._metadata_path()
        paths = [self.source/n for n in (*FILES, 'seat_forced_inputs.jsonl',
                                         'seat_forced_summary.json')]+[self.metadata_path]
        self.hashes = {str(p):_hash(p) for p in paths}
        self.stamps = {p:_stamp(p) for p in paths}
        self.metadata = json.loads(self.metadata_path.read_text('utf8'))
        self._check_metadata_scope()
        self.inventory = load_current_chat(self.source, {n:self.hashes[str(self.source/n)] for n in FILES})
        self.native = {}
        self.physical = {}
        for packet in self.inventory['packets'].values():
            for row, physical, response, pixels in observation_source_rows(packet):
                key = (row['table_id'], row['session_id'], row['frame_id'], row['row_index'], row['stable_row_id'])
                if key in self.native:raise ValueError('DUPLICATE_NATIVE_ROW')
                self.native[key] = row
                self.physical[key] = physical
        states = json.loads((self.source/'seat_forced_summary.json').read_text('utf8')).get('state_receipts', [])
        a = self.actor_provider
        self.fresh_actors = LineageActorProvider(SeatIdentityLineage(saved_nick_observations(self.source), states),
            a.bounds, a.hands.values(), a.associations, a.manual_reviews)
        if any(_hash(p) != self.hashes[str(p)] for p in paths):raise ValueError('NO_SB_SOURCE_CHANGED_DURING_LOAD')
        self.loaded = True

    def _physical_prefix(self, rows, key):
        """Complete rendered lines at the registered fixed CHAT line cadence.

        One native snapshot covers numbered header through DEAL. Independent
        detector spans and birth admissions must cover every line, including
        continuations. A missing whole line leaves a pitch gap. This grants no
        completeness for compact/other layouts, hidden text or player seating.
        """
        from .chat_rows import padded_row_span
        from .chat_tracker import ROW_STEP_PX, GEOMETRY_TOLERANCE_PX
        frames = {r['frame_id'] for r in rows}
        if len(frames) != 1:raise ValueError('PHYSICAL_PREFIX_NOT_SINGLE_SNAPSHOT')
        frame = next(iter(frames))
        packets = [p for p in self.inventory['packets'].values()
                   if (p['table_id'],p['session_id'],p['header_frame']) == (*key[:2],frame)]
        if len(packets) != 1:raise ValueError('PHYSICAL_PREFIX_PACKET_NOT_UNIQUE')
        receipts = packets[0]['retained_event_admission_receipts']
        receipts = [r for r in receipts if (r['table'],r['session_id'],r['frame_after']) == (*key[:2],frame)]
        if len(receipts) != 1:raise ValueError('PHYSICAL_PREFIX_RECEIPT_NOT_UNIQUE')
        receipt = receipts[0]
        if (receipt['schema'] != 'physical-chat-admission-v1' or
                receipt['proof']['status'] != 'PROVEN_PHYSICAL_NEW'):
            raise ValueError('PHYSICAL_PREFIX_BIRTH_NOT_PROVEN')
        height,width = receipt['geometry']['shape']
        box = receipt['source_chat_box']
        if (box[2]-box[0],box[3]-box[1]) != (width,height):
            raise ValueError('PHYSICAL_PREFIX_LAYOUT_CONFLICT')
        if re.fullmatch('[0-9a-f]{64}',receipt['source_pixel_hash_after']) is None:
            raise ValueError('PHYSICAL_PREFIX_CROP_HASH_REQUIRED')
        edge = receipt['geometry']['top_edge_spans']
        spans = edge + receipt['new_spans']
        indices = [r['row_index'] for r in rows]
        if indices != list(range(indices[0],indices[-1]+1)):
            raise ValueError('PHYSICAL_PREFIX_MISSING_SPAN')
        chosen = [spans[i] for i in indices]
        if any(not 1 < a < b < height-1 for a,b in chosen):
            raise ValueError('PHYSICAL_PREFIX_CLIPPED_SPAN')
        centers = [(a+b)/2 for a,b in chosen]
        if any(abs(b-a-ROW_STEP_PX) > GEOMETRY_TOLERANCE_PX
               for a,b in zip(centers,centers[1:])):
            raise ValueError('PHYSICAL_PREFIX_LINE_GAP_OR_LAYOUT')
        admissions = {a['index']:a for a in receipt['admissions']}
        if len(admissions) != len(receipt['admissions']):
            raise ValueError('PHYSICAL_PREFIX_DUPLICATE_ADMISSION')
        for row,index,(top,bottom) in zip(rows,indices,chosen):
            relative = index-len(edge)
            admission = admissions[relative]
            route = (row['table_id'],row['session_id'],frame,index,row['stable_row_id'])
            physical = self.physical[route]
            y1,y2 = padded_row_span(spans,index,height)
            if (relative not in receipt['proof']['new_indices'] or
                    admission['OCR_admission'] is not True or admission['span'] != [top,bottom] or
                    admission['row_id'] != row['stable_row_id'] or physical['y1'] != y1 or
                    physical['pixel_shape'] != [y2-y1,width] or
                    physical['source_pts'] != receipt['source_pts'] or
                    [physical['time_base_num'],physical['time_base_den']] != receipt['time_base'] or
                    row.get('confidence') is None or row['confidence'] < .95):
                raise ValueError('PHYSICAL_PREFIX_NATIVE_ADMISSION_CONFLICT')
            pixels = np.frombuffer(base64.b64decode(physical['row_pixels_base64'],validate=True),np.uint8).reshape(physical['pixel_shape'])
            if (hashlib.sha256(pixels.tobytes()).hexdigest() != physical['row_pixel_sha256'] or
                    hashlib.sha256(pixels[top-y1:bottom-y1].tobytes()).hexdigest() != admission['row_pixel_sha256']):
                raise ValueError('PHYSICAL_PREFIX_TIGHT_PIXEL_CONFLICT')
        low,high = self.actor_provider.bounds[key]
        following = sorted((h for h in self.inventory['positive_headers']
            if (h['table_id'],h['session_id']) == key[:2] and h['frame'] > high),key=lambda h:h['frame'])
        if not following or following[0]['hand_id'] == key[2]:
            raise ValueError('PHYSICAL_PREFIX_NEXT_HAND_BOUNDARY_NOT_PROVEN')
        return dict(rule=PHYSICAL_PREFIX_RULE,scope='RENDERED_FORCED_PREFIX_AT_REGISTERED_LINE_CADENCE',
            frame=frame,source_pts=receipt['source_pts'],time_base=receipt['time_base'],
            source_chat_box=box,source_chat_sha256=receipt['source_pixel_hash_after'],
            row_indices=indices,detector_spans=chosen,next_header=copy.deepcopy(following[0]),
            source_receipt_sha256=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode()).hexdigest())

    def _who(self, hand, actor, prefix_frame):
        from .simple_saved_policy import hand_key, _capture_record, _source_version
        from .native_nick_who import validate_native_nick
        from .actor_seat_point import source_time
        key = hand_key(hand)
        old = self.actor_provider.for_actor(hand, actor)
        fresh = self.fresh_actors.for_actor(hand, actor)
        for p in (old, fresh):
            if (p.get('status') != 'PROVEN' or hand_key(p) != key or
                    p.get('canonical_actor', p.get('actor')) != actor):
                raise ValueError('CURRENT_WHO_NOT_PROVEN')
        if old['seat'] != fresh['seat']:raise ValueError('WHO_SEAT_CONFLICT')
        anchor = fresh['anchor']
        for p in (old, fresh):
            begin, finish = p['identity_interval']
            if anchor['frame'] < begin or finish is not None and anchor['frame'] > finish:
                raise ValueError('STALE_WHO_INTERVAL')
        low, high = self.actor_provider.bounds[key]
        if not low <= anchor['frame'] <= high:raise ValueError('FOREIGN_WHO_ANCHOR')
        checkpoints = [c for c in self.inventory['context_checkpoints']
            if (c['table_id'],c['session_id']) == key[:2] and c['frame'] <= anchor['frame']]
        if not checkpoints or max(checkpoints, key=lambda c:c['frame'])['hand_id'] != key[2]:
            raise ValueError('WHO_CURRENT_HEADER_UNKNOWN')
        path = self.source/'seat_forced_inputs.jsonl'
        if Path(anchor['source_file']).resolve() != path:raise ValueError('FOREIGN_WHO_SOURCE')
        row = _capture_record(*_source_version(path), anchor['source_line'])
        if (hand_key(row) != key or row['seat'] != fresh['seat'] or row['source_frame'] != anchor['frame']):
            raise ValueError('WHO_NATIVE_SCOPE_CONFLICT')
        stamp, label = validate_native_nick(row)
        response = row['ocr_response']
        actual = dict(raw_text=response.get('raw_text'),confidence=response.get('score'),
            ocr_status=response.get('status'),item_id=response.get('item_id'),
            pixel_sha256=row['pixel_sha256'],source_pts=row['source_pts'],time_base=row['time_base'],roi=row['roi'])
        if response.get('status') != 'OK' or any(anchor.get(k) != v for k,v in actual.items()):
            raise ValueError('WHO_NATIVE_ANCHOR_CHANGED')
        clocks = {}
        for p in self.physical.values():
            if (p['table_id'],p['table_session_id']) == key[:2]:
                value = source_time(p['source_pts'], [p['time_base_num'],p['time_base_den']])
                if p['frame_id'] in clocks and clocks[p['frame_id']] != value:raise ValueError('CHAT_CLOCK_CONFLICT')
                clocks[p['frame_id']] = value
        if anchor['frame'] in clocks and clocks[anchor['frame']] != stamp:raise ValueError('WHO_CLOCK_CONFLICT')
        clocks[anchor['frame']] = stamp
        values = [clocks[f] for f in sorted(clocks)]
        if values != sorted(set(values)):raise ValueError('WHO_CLOCK_ORDER_CONFLICT')
        for boundary in fresh.get('boundaries', []):
            if min(prefix_frame,anchor['frame']) <= boundary.get('frame',-1) <= max(prefix_frame,anchor['frame']) and boundary.get('kind') in {
                    'EMPTY','OPEN SEAT','OPENSEAT','REPLACE','LEFT','LEAVE','HAS LEFT THE TABLE'}:
                raise ValueError('WHO_OCCUPANT_BOUNDARY')
        return dict(actor=actor,seat=fresh['seat'],rule=fresh.get('rule_used'),anchor=copy.deepcopy(anchor))

    def authorize(self, hand):
        from .simple_saved_policy import hand_key
        from .poker_hand_assembly import assemble_messages, BETTING, parse_text
        self.last_rejection = None
        try:
            if self.initial_error is not None:raise ValueError('NO_SB_REVIEW_UNREADABLE: '+self.initial_error)
            key = hand_key(hand)
            matching = [r for r in self.reviews if (r.get('table_id'),r.get('session_id'),r.get('hand_id')) == key]
            review = None
            if self.review_path is not None:
                if not matching:return None
                if len(matching) != 1:raise ValueError('CONFLICTING_NO_SB_REVIEWS')
                review = matching[0]
                if (_stamp(self.review_path) != self.review_stamp or _hash(self.review_path) != self.review_hash):
                    raise ValueError('NO_SB_REVIEW_CHANGED')
                if review.get('schema') != REVIEW_SCHEMA:raise ValueError('UNSUPPORTED_NO_SB_REVIEW')
            if not hand.get('start_observed') or hand.get('boundary_conflicts') or hand.get('excluded_message_indices'):
                raise ValueError('HAND_BOUNDARY_NOT_PROVEN')
            messages = hand['messages']
            if any(m['action'] == 'SMALL_BLIND' for m in messages):raise ValueError('OBSERVED_SB_CONFLICT')
            deal = [i for i,m in enumerate(messages) if m['action'] == 'DEAL']
            if len(deal) != 1:raise ValueError('DEAL_MARKER_NOT_UNIQUE')
            prefix = messages[:deal[0]+1]
            if any(m['action'] in BETTING for m in prefix):raise ValueError('VOLUNTARY_ACTION_BEFORE_PREFIX_END')
            if any(m['action'] in {'ANTE','BIG_BLIND','SMALL_BLIND'} for m in messages[deal[0]+1:]):
                raise ValueError('LATE_FORCED_POST')
            rows = [r for m in prefix for r in m['source_rows']]
            if not rows:raise ValueError('EMPTY_NATIVE_PREFIX')
            manual = None
            if review is not None:
                if prefix_digest(rows) != review.get('prefix_sha256'):raise ValueError('REVIEW_PREFIX_CHANGED')
                manual_path = Path(review['manual_source']['path']).resolve()
                if _hash(manual_path) != review['manual_source']['sha256']:raise ValueError('MANUAL_SOURCE_CHANGED')
                manual = json.loads(manual_path.read_text('utf8'))
            if manual is not None and (manual.get('authority') != 'USER_MANUAL_SOURCE_VIDEO_VERIFICATION' or
                    int(str(manual.get('TABLE')).removeprefix('T')) != key[0] or str(manual.get('HAND')) != key[2] or
                    manual.get('SB') != 'NOT_POSTED_OPEN_BLIND_POSITION' or manual.get('NEXT_SEAT') != 'OPEN_SEAT' or
                    manual.get('first_video_partial_hand_used') is not False or manual.get('raw_machine_receipt_created') is not False):
                raise ValueError('EXPLICIT_MANUAL_ABSENCE_NOT_PROVEN')
            self._load()
            video_hash = self.metadata.get('source_video_sha256')
            if (not isinstance(video_hash,str) or re.fullmatch('[0-9a-f]{64}',video_hash) is None or
                    review is not None and video_hash != review.get('source_video_sha256')):
                raise ValueError('REVIEW_VIDEO_SCOPE_CONFLICT')
            # The anchor is the first native numbered-hand observation. A
            # source-exact reconstruction may split NEW_HAND + its corroborating
            # numbered row; that service grouping grants no extra authority.
            anchor=prefix[0]
            if anchor['action']!='HAND_HEADER' or str(anchor.get('hand_number'))!=key[2]:
                raise ValueError('NUMBERED_HAND_ANCHOR_NOT_PROVEN')
            headers = [r for r in self.inventory['positive_headers']
                if (r['table_id'],r['session_id'],r['hand_id']) == key and r['row_id'] in anchor['source_row_ids']]
            if len(headers) != 1:raise ValueError('CURRENT_HEADER_NOT_UNIQUE')
            start = headers[0]['frame']
            end = rows[-1]['frame_id']
            owners = {(r['table_id'],r['session_id'],r['frame'],r['row_index'],r['row_id']):r
                for r in self.inventory['retained_row_ownership']}
            native_rows = []
            for row in rows:
                route = (row['table_id'],row['session_id'],row['frame_id'],row['row_index'],row['stable_row_id'])
                parent = self.native[route]
                owner = owners[(key[0],key[1],*route[2:])]
                if (route[:2] != (f'T{key[0]:02d}',key[1]) or
                        owner.get('first_proven_hand_id') not in (None,key[2]) or owner.get('hand_id') not in (None,key[2]) or
                        owner.get('hand_id') is None and row['frame_id']!=start or
                        prefix_digest([row]) != prefix_digest([parent]) or row.get('ocr_status') != 'OK' or
                        row.get('canonical_conflict') or row.get('canonical_duplicate_of')):
                    raise ValueError('NATIVE_PREFIX_CONTEXT_CONFLICT')
                physical=self.physical[route]
                for field,value in (('source_pts',physical['source_pts']),
                                    ('time_base',[physical['time_base_num'],physical['time_base_den']])):
                    if field in row and row[field]!=value:raise ValueError('PREFIX_CLOCK_CONFLICT')
                native_rows.append(copy.deepcopy(parent))
            first, last = (rows[0]['frame_id'],rows[0]['row_index']), (rows[-1]['frame_id'],rows[-1]['row_index'])
            complete = sorted((r for route,r in self.native.items() if route[:2] == (f'T{key[0]:02d}',key[1]) and
                              first <= (r['frame_id'],r['row_index']) <= last),key=lambda r:(r['frame_id'],r['row_index']))
            if [r['stable_row_id'] for r in complete] != [r['stable_row_id'] for r in rows]:
                raise ValueError('NATIVE_PREFIX_COVERAGE_NOT_EXACT')
            low, high = self.actor_provider.bounds[key]
            if any((r['table_id'],r['session_id']) == key[:2] and low <= r['frame'] <= high
                   for r in self.inventory['frame_failures']):raise ValueError('FAILED_CURRENT_HAND_FRAME')
            for route,row in self.native.items():
                if route[:2] != (f'T{key[0]:02d}',key[1]) or not low <= row['frame_id'] <= high:continue
                if parse_text(row['raw_ocr_text'])['action'] == 'SMALL_BLIND':raise ValueError('NATIVE_SB_CONFLICT')
            fresh=[];offset=0
            for message in prefix:
                count=len(message['source_rows'])
                parsed=assemble_messages(native_rows[offset:offset+count],require_adjacency=True)
                if len(parsed)!=1:raise ValueError('PREFIX_SOURCE_GROUP_NOT_REPRODUCIBLE')
                fresh.extend(parsed);offset+=count
            aliases = self.actor_provider.actor_map(hand)
            fields = ('action','player','amount','all_in','hand_number')
            for m in fresh:m['player'] = aliases.get(m.get('player'),m.get('player'))
            if [{k:m.get(k) for k in fields} for m in fresh] != [{k:m.get(k) for k in fields} for m in prefix]:
                raise ValueError('PREFIX_SEMANTICS_CHANGED')
            if any((m['action'] in {'HAND_HEADER','HAND_ID'} or m['action']=='NEW_HAND' and m.get('hand_number') is not None)
                   and str(m.get('hand_number'))!=key[2] for m in fresh):
                raise ValueError('FOREIGN_PREFIX_HAND_MARKER')
            if any(m['action'] not in {'HAND_HEADER','HAND_ID','NEW_HAND','ANTE','BIG_BLIND','DEAL'} for m in fresh):
                raise ValueError('UNRESOLVED_PREFIX_MESSAGE')
            ante = [m for m in fresh if m['action'] == 'ANTE']
            bb = [m for m in fresh if m['action'] == 'BIG_BLIND']
            if (not ante or len({m['player'] for m in ante}) != len(ante) or len(bb) != 1 or
                    any(m.get('all_in') for m in ante+bb) or any(int(m['amount'])<=0 for m in ante) or
                    int(bb[0]['amount']) <= 0 or int(bb[0]['amount']) % 2):
                raise ValueError('NORMAL_FORCED_PREFIX_NOT_PROVEN')
            actors = {m['player'] for m in ante}
            if ({m['player'] for m in messages if m.get('player')} != actors or bb[0]['player'] not in actors or
                    any(int(m['amount']) != int(ante[0]['amount']) for m in ante) or
                    manual is not None and (manual.get('ANTE') != dict(count=len(ante),amount=int(ante[0]['amount'])) or
                    manual.get('BB') != dict(actor=bb[0]['player'],amount=int(bb[0]['amount'])) or manual.get('BTN') not in actors)):
                raise ValueError('MANUAL_SOURCE_PREFIX_CONFLICT')
            completeness = None
            if manual is None:
                if not hand.get('next_boundary_observed'):raise ValueError('HAND_END_BOUNDARY_NOT_PROVEN')
                completeness = self._physical_prefix(rows,key)
            identities = {a:self._who(hand,a,start) for a in sorted(actors)}
            seats = [p['seat'] for p in identities.values()]
            if len(set(seats)) != len(seats):raise ValueError('WHO_DUPLICATE_SEAT')
            # Relative button/open-position facts come from explicit manual
            # review. Later WHO anchors do not establish historical seat geometry
            # or a vacant seat number at the forced prefix.
            if any(_stamp(p) != v for p,v in self.stamps.items()):raise ValueError('NO_SB_SOURCE_CHANGED')
            result = dict(rule=RULE,table_id=key[0],session_id=key[1],hand_id=key[2],
                sb_status='NOT_POSTED',reason='OPEN_BLIND_POSITION' if manual is not None else 'COMPLETE_PHYSICAL_FORCED_PREFIX_NO_SB',sb_actor=None,posted_sb_total=0,
                button_actor=manual['BTN'] if manual is not None else None,
                next_button_position='MANUALLY_CONFIRMED_OPEN' if manual is not None else 'UNKNOWN',
                bb_actor=bb[0]['player'],participants=sorted(actors),identities=identities,
                nominal_blinds=dict(SB=int(bb[0]['amount'])//2,BB=int(bb[0]['amount']),rule=NOMINAL_RULE),
                native_prefix_sha256=prefix_digest(rows),prefix_source_row_ids=[r['stable_row_id'] for r in rows],
                source=str(self.source),source_sha256=self.hashes,
                synthetic_post_authority=False,occupancy_authority=False,deal_complete_authority=False,
                start_stack_timing_authority=False,who_authority=False,historical_seat_geometry_authority=False)
            if manual is not None:
                result.update(review_source=dict(path=str(self.review_path),sha256=self.review_hash),
                    manual_source=copy.deepcopy(review['manual_source']),manual_authority=manual['authority'])
            else:
                result.update(chat_completeness=completeness,authority=PHYSICAL_PREFIX_RULE)
            return result
        except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError,cv2.error) as error:
            self.last_rejection = str(error)
            return None


def prove_no_sb(hand, source):
    """Cached flags/receipts and arbitrary providers grant no absence authority."""
    return source.authorize(hand) if type(source) is LegitimateNoSBSource else None
