"""User-authorized fixed-layout CHAT actor / NICK ROI identity policy.

A single qualified observation proves identity. Unread seats, amounts and
roster completeness have no authority here. Numeric witnesses stay separate.
"""
import copy
import hashlib
import json
import math
import re
from bisect import bisect_right
from collections import defaultdict

import cv2
import numpy as np

from .actor_seat_point import frame_times, response_text, prove_actor_point as legacy_point
from .nickname_roster_binding import literal_constraint as legacy_constraint, matches
from .nickname_roster_binding import _seat_rois, _verify_gray_source
from .normalize_chat import normalize

RULE = 'USER_20261009_DIRECT_CHAT_NICK_ROI_ACTOR_V1'
CROSS_RULE = 'CROSS_SOURCE_NICK_CHAT_PROOF'
NATIVE_NICK_RULE = 'CURRENT_HAND_NATIVE_NICK_WHO_V1'
IDENTITY_RULES = {RULE, CROSS_RULE, NATIVE_NICK_RULE}
SOURCE_KEY = 'current_chat_nick_actor_sources'
from .nickname_policy import FIRST_NICK_POLICY, nickname_witness_floor


def literal_constraint(text):
    raw = text.strip()
    stem = raw.rstrip('.…')
    if not stem or re.fullmatch(r'(?:PENSEAT|OPENSEAT|OPEN\s+SEAT|EMPTY)', stem, re.I):
        return None
    if legacy_constraint(stem) is None:
        return None
    return dict(kind='PREFIX' if stem != raw else 'EXACT', text=stem)


# Finite saved-corpus aliases. These are NICK pixel hypotheses, never CHAT aliases.
NICK_UI_VARIANTS = {
    'FOLD': frozenset({'5OLD'}),
    'ANTE': frozenset({'ANTT', 'ANITE', 'AANTE', 'AATE', 'ANTR', 'ANTF',
        'ARTR', 'ANT5', 'AATR', 'AANTR', 'NTE', 'OTE', '0TE', 'ZTE',
        'ANHE', 'UONT', 'ONT', 'ONTT'}),
}
# Native clipped glyph cores measured from saved 83x17 grayscale crops.
# Both ink and background are required; ordinary nickname glyphs fail this proof.
NICK_UI_CORES = {'FOLD': {'ink': ['00000000000000000000000000000000000000000000000000', '00000000100100000100100001000000111010000000000000', '00000000100000001000000001000000100001000000000000', '00000000100000000000010001000000100001000000000000', '00000000100000000000000001000000100001000000000000'], 'background': ['11111110000000110000011100111110000001111111111111', '11111110000000100000001100011110000000111111111111', '11111110001111100011000100011110001000011111111111', '11111110001111100011000100011110001100011111111111', '11111110000001100011000100011110001100011111111111']}, 'ANTE': {'ink': ['00000000001100000000001000001000001111100000000000', '00000000001100000000001000011100001111100000000000', '00000000011100000000001000011000000000000000000000', '00000000011100000000001000011000001000000000000000', '00000000011000000000001000011000001111000000000000'], 'background': ['00011111000001110001100000000000000000011111111111', '00011110000000110000000000000000000000011111111111', '00111100000000110000000001000011000111111111111111', '00111100000000110000000011100011000111111111111111', '00111000000000010000000011100011000000111111111111']}}


# Exact native pixel twins of corpus-observed canonical UI crops.
NICK_UI_PIXEL_TWINS = {'FOLD': ['cabc9b5779ba734630f4c3d16cd4c26d334cd0ace7d2695edfaf9c9abf2efa87'], 'ANTE': ['0575c3526344fa8bd908ac42f29ea52d619ac66b3185785334e428ab2ef4fe24', '09f57704ab8f51229a31151a24f1c8d45ac5f70549bc6793cd86387c4ae749b5', '105ac3830290879d7d0e10486d95d9fe12ca4719356ea600df35a8575b180a07', '12e2d62f873e24f1c715eb9b698bc24f3bd37245761e00acebb4f4a7d1687ccc', '1386573a3e1a8fc6318ccdf798aec52d64c44320001985e168e1abeb367c8596', '1e22f25f26a7975be3eaa3b1ae43b926f7fe64c43b613af33b2b8c98bff7e1c2', '31b654e5ee0f40b26de63e97536af85d14691c9f1a4de31027836c225e6312e8', '358c2dfc1b8ef37e1dfb7ac11393b03a95cda69cc515d59ca8ed815cb739eba5', '41b3a3de6d931f3533314886ddd18c511eb258fcf771daf2691e159d1c7d2816', '44b73e695c0a0e791a9ddb55d34d4d514371142af12c300a42c2f7b061f00419', '58ae86e8550226d417d4d5c392f67614bb4447c0a499fb1b8452b4a63b75152f', '5a137bd43946e47690b5ad8146dbf93ab628572675079f422128acb67c53d680', '5f1ce78f7e4533ab4fc944b252394d4194b7971224d5c99eb503e52d0b4afdd3', '634de3c94af4a43aa07ef94f971900bf5a2655973da4ff2ad8f294b544e4c06d', 'b7b39c95c8c305d0b3f25249dd50133e4e1aa6f16ce4d4ff51657206a8a71d80', 'c8368b2108dfe5f6065728655d1f5df92535d7f6dac20b743b9a893cb3d08373', 'e2a86776f54ffcf318b2b3cdaa857f0be22a615602c419423709188ef3381c78', 'e4c5528b1a12c085520ce744f2ad8da04efdf94ac1c49d80a4b6de188a69ed03', 'f44b8af6c545f4ca689cfb3b391f0a9f208d0716d3998c864277b2b1465b96fd', 'fd3f3b11df36bcdc5edff884edb9ae5f8fe8989bcc10abe25b25dab76ba2bec1']}

def nick_ui_variant(raw):
    if isinstance(raw, str):
        return next((name for name, variants in NICK_UI_VARIANTS.items()
                     if raw.strip().upper() in variants), None)
    return None


# Fixed native four-row ANTE stencil: interior ink/background from independently
# retained canonical sources. Every required point is tested, without resizing.
NICK_UI_CLIPPED_ANTE_CORE = {'ink': ['00000000000000000000000011110001111000000000000000', '00000000000000000000000000000000000000000000000000', '00000000010000010000000000000000000000000000000000', '00000000110000010000000000000000000000000000000000'], 'background': ['11111110000010000000000000000000000001111111111111', '11111100000010000000000000000000000001111111111111', '11111100000000000000000000000000000001111111111111', '11111100000000000000001100001000001111111111111111']}


def _prove_clipped_ui_contrast(pixels):
    """Prove an unscaled, clipped native glyph profile at the fixed ROI bottom."""
    height, width = pixels.shape
    if width < 55 or height < 8:
        return None
    contrast = pixels.astype(float) - np.median(pixels[:8, :], axis=0)
    matches = []
    # New fallback always tests both physical families, regardless of OCR text.
    for name, core in (('FOLD', NICK_UI_CORES['FOLD']),
                       ('ANTE', NICK_UI_CLIPPED_ANTE_CORE)):
        ink = np.array([[v == '1' for v in line] for line in core['ink']])
        background = np.array([[v == '1' for v in line] for line in core['background']])
        rows = ink.shape[0]
        y = height - rows
        for dx in range(-5, 6):
            x = 20 + dx
            if x < 0 or x + 50 > width:
                continue
            band = contrast[y:height, x:x+50]
            ink_min = float(band[ink].min())
            background_max = max(0.0, float(band[background].max()))
            if ink_min <= background_max:
                continue
            threshold = (ink_min + background_max) / 2
            upper = contrast[:y, x:x+50] >= threshold
            # A full horizontal line in the native ROI's first two rows is
            # the seat frame, not glyph ink. Preserve its source and receipt.
            border_rows = [r for r in range(min(2, y)) if upper[r].all()]
            upper_count = int(upper.sum()) - len(border_rows) * 50
            band_count = int((band >= threshold).sum())
            if upper_count > band_count:
                continue
            matches.append(dict(canonical_ui=name, glyph_rows=rows,
                glyph_band=[x, y, x+50, height], ink_contrast_min=ink_min,
                background_contrast_max=background_max, contrast_threshold=threshold,
                contrast_separation=ink_min-background_max,
                upper_ink_count=upper_count, glyph_ink_count=band_count,
                excluded_horizontal_border_rows=border_rows,
                ink_core_count=int(ink.sum()), background_core_count=int(background.sum())))
    if len({m['canonical_ui'] for m in matches}) != 1:
        return None
    return max(matches, key=lambda m: (m['ink_core_count'], m['contrast_separation']))


def _prove_bottom_clipped_service_text(pixels):
    """Fixed NICK region: bottom-edge glyphs without any upper continuation."""
    height,width=pixels.shape
    if height<8 or width<30:return None
    contrast=pixels.astype(float)-np.median(pixels[:8,:],axis=0)
    mask=contrast>=32
    split=height-4
    upper=mask[:split].copy()
    borders=[r for r in range(min(2,split)) if upper[r].all()]
    for r in borders:upper[r]=False
    if upper.any():return None
    band=mask[split:]
    if not band[-1].any() or int(band.sum())<8:return None
    columns=np.flatnonzero(band.any(axis=0))
    if len(columns)<6 or columns[-1]-columns[0]+1>=width-4:return None
    return dict(canonical_ui='BOTTOM_CLIPPED_SERVICE_TEXT',glyph_band=[int(columns[0]),split,
        int(columns[-1])+1,height],upper_ink_count=0,glyph_ink_count=int(band.sum()),
        contrast_threshold=32,excluded_horizontal_border_rows=borders)


def prove_nick_ui_overlay(observation):
    """Prove native UI pixels independently of OCR spelling; otherwise retain WHO input."""
    embedded = observation.get('native_nick_source')
    row = embedded if embedded is not None else observation
    response = row.get('ocr_response', {})
    raw = observation.get('raw_text', response.get('raw_text'))
    canonical = nick_ui_variant(raw)
    # Existing aliases retain their original single-family proof. Unlisted OCR
    # spellings may test both physical families, but text grants no authority.
    families = (canonical,) if canonical is not None else tuple(NICK_UI_CORES)
    if not isinstance(raw, str) or not raw.strip() or 'png_base64' not in row:
        return None
    if embedded is not None:
        pairs = [('table_id','table_id'), ('session_id','session_id'), ('seat','seat'),
                 ('hand_id','hand_id'), ('frame','source_frame'), ('roi','roi'),
                 ('pixel_sha256','pixel_sha256'), ('source_pts','source_pts'),
                 ('time_base','time_base')]
        if any(observation.get(a) != row.get(b) for a,b in pairs) or any(
                observation.get(a) != response.get(b) for a,b in
                [('raw_text','raw_text'),('confidence','score'),
                 ('ocr_status','status'),('item_id','item_id')]):
            raise ValueError('NICK_UI_SOURCE_CONTEXT_CONFLICT')
    if row.get('kind') != 'NICK' or response.get('raw_text') != raw:
        raise ValueError('NICK_UI_SOURCE_KIND_OR_TEXT_CONFLICT')
    if response.get('status') != 'OK':
        return None
    # The live producer may retain a partial source without PTS. Unlisted OCR
    # previously remained a literal candidate; missing clock grants no UI proof.
    if canonical is None and row.get('source_pts') is None:
        return None
    if not all(row.get(k) is not None for k in
               ('table_id','session_id','hand_id','seat','source_frame','source_pts','time_base')):
        raise ValueError('NICK_UI_SOURCE_CLOCK_UNKNOWN')
    tb = row['time_base']
    if not isinstance(row['source_frame'],int) or not math.isfinite(row['source_pts']) or (
            not isinstance(tb,(list,tuple)) or len(tb)!=2 or
            any(not isinstance(v,int) or v<=0 for v in tb)):
        raise ValueError('NICK_UI_SOURCE_CLOCK_INVALID')
    roi = _seat_rois()[f"slot_{int(row['table_id']):02d}"]['seats'][row['seat']]['nick']
    if list(row.get('roi',[])) != list(roi):
        raise ValueError('NICK_UI_SOURCE_ROI_CONFLICT')
    pixels = _verify_gray_source(row)
    if pixels.shape != (roi[3]-roi[1],roi[2]-roi[0]):
        raise ValueError('NICK_UI_SOURCE_SHAPE_CONFLICT')
    twins = [name for name in families
             if row['pixel_sha256'] in NICK_UI_PIXEL_TWINS.get(name, [])]
    if len(twins) == 1:
        return dict(rule='POKERDOM_NATIVE_NICK_UI_PIXEL_TWIN_V1', canonical_ui=twins[0],
                    raw_text=raw, pixel_sha256=row['pixel_sha256'], roi=list(roi),
                    source_frame=row['source_frame'], source_pts=row['source_pts'],
                    time_base=row['time_base'])
    if twins:
        return None
    if pixels.shape == (17,83):
        mask = pixels.astype(float)-np.median(pixels[:8,:],axis=0) >= 32
        band = mask[12:17,20:70]
        proofs = []
        for name in families:
            core = NICK_UI_CORES[name]
            ink = np.array([[v=='1' for v in line] for line in core['ink']])
            background = np.array([[v=='1' for v in line] for line in core['background']])
            if not band[ink].all() or band[background].any() or mask[:12,20:70].sum()>band.sum():
                continue
            proofs.append(dict(rule='POKERDOM_NATIVE_CLIPPED_NICK_UI_V1', canonical_ui=name,
                        raw_text=raw, pixel_sha256=row['pixel_sha256'], roi=list(roi),
                        source_frame=row['source_frame'], source_pts=row['source_pts'],
                        time_base=row['time_base'], ink_core_count=int(ink.sum()),
                        background_core_count=int(background.sum())))
        if len(proofs) == 1:
            return proofs[0]
        if proofs:
            return None
    clipped = _prove_clipped_ui_contrast(pixels)
    if clipped is None:
        clipped=_prove_bottom_clipped_service_text(pixels)
        if clipped is None:return None
        rule='POKERDOM_FIXED_NICK_BOTTOM_ONLY_SERVICE_REGION_V1'
    else:rule='POKERDOM_NATIVE_CLIPPED_NICK_UI_CONTRAST_V3'
    return dict(clipped, rule=rule,
                raw_text=raw, pixel_sha256=row['pixel_sha256'], roi=list(roi),
                source_frame=row['source_frame'], source_pts=row['source_pts'],
                time_base=row['time_base'])


def chat_actor_candidates(sources):
    result = []
    for row, physical, response, pixels in sources:
        text = response_text(response)
        if re.match(r'^(?:Dealer|Дилер)\s*:\s*', text, re.I) is None:
            continue
        parsed = normalize([text])
        if parsed.event_type not in {'ANTE', 'SMALL_BLIND', 'BIG_BLIND', 'RAISE',
                'CALL', 'CHECK', 'FOLD', 'BET', 'WIN', 'UNCALLED_RETURN', 'SHOW', 'NO_SHOW'}:
            continue
        actor = parsed.actor
        body = re.sub(r'^(?:Dealer|Дилер)\s*:\s*', '', text, flags=re.I)
        if not actor or not body.startswith(actor + ' '):
            continue
        if literal_constraint(actor) != dict(kind='EXACT', text=actor):
            continue
        reason = 'CHAT_ACTOR_SOURCE_CONFIDENCE_UNKNOWN' if response['score'] < .95 else None
        if parsed.reason and ('ACTOR_' in parsed.reason or 'MULTIPLE_ACTION_CLAUSES' in parsed.reason):
            reason = 'CHAT_ACTOR_GRAMMAR_UNKNOWN'
        result.append(dict(actor=actor, source=row, reason=reason))
    return result


def qualified_nicks(context, *, cross_source_actors=None):
    nick_confidence=nickname_witness_floor(context.get(SOURCE_KEY,{}).get('nickname_witness_policy'))
    frame_times(context)
    result = []
    seen = set()
    for row in context['nickname_observations_all_within_window']:
        if any(row.get(k) != context[k] for k in ('table_id', 'session_id', 'hand_id')):
            raise ValueError('NICK_SOURCE_CONTEXT_CONFLICT')
        if row['kind'] != 'NICK':
            raise ValueError('NICK_SOURCE_KIND_CONFLICT')
        key = (row['seat'], row['source_frame'])
        if key in seen:
            raise ValueError('DUPLICATE_NICK_SOURCE')
        seen.add(key)
        response = row.get('ocr_response', {})
        if response.get('status') != 'OK':
            continue
        text = response_text(response)
        if prove_nick_ui_overlay(row) is not None:
            continue
        constraint = literal_constraint(text)
        if constraint is None:
            continue
        if response['score'] < nick_confidence and (cross_source_actors is None or
                not any(matches(actor, constraint) for actor in cross_source_actors)):
            continue
        if SOURCE_KEY in context:
            checkpoints = context[SOURCE_KEY]['context_checkpoints']
            index = bisect_right([r['frame'] for r in checkpoints], row['source_frame'])-1
            if index < 0 or checkpoints[index]['hand_id'] != context['hand_id']:
                continue
        roi = _seat_rois()[f"slot_{context['table_id']:02d}"]['seats'][row['seat']]['nick']
        pixels = _verify_gray_source(row)
        if row['roi'] != roi or pixels.shape != (roi[3]-roi[1], roi[2]-roi[0]):
            raise ValueError('NICK_FIXED_SEAT_PIXELS_CONFLICT')
        result.append((row, constraint))
    return result


def actual_conflict(context, actor, seat, labels, known_actors):
    # Only observed contradictions have authority. No closed-roster premise.
    for row, label in labels:
        if row['seat'] == seat:
            if not matches(actor, label):
                return 'OBSERVED_TARGET_NICK_CONFLICT'
            if any(other != actor and matches(other, label) for other in known_actors):
                return 'OBSERVED_CHAT_PREFIX_COLLISION'
        elif matches(actor, label):
            return 'OBSERVED_SECOND_SEAT_MATCH'
    summary = context[SOURCE_KEY]['conflict_inventory']
    for row in summary['bound_posts']:
        if ((row['seat'] == seat and row['actor'] != actor) or
                (row['seat'] != seat and row['actor'] == actor)):
            return 'OBSERVED_ROSTER_SEAT_CONFLICT'
    for row in summary['state_receipts']:
        if row.get('status') != 'APPLIED' or row.get('evidence_status') != 'PROVEN':
            continue
        other = row.get('actor')
        if row['seat'] == seat and row.get('hand_id') != context['hand_id']:
            return 'OBSERVED_STATE_HAND_CONTEXT_CONFLICT'
        if row['seat'] == seat and row['kind'] in {'EMPTY', 'REPLACE', 'SIT_IN'}:
            return 'OBSERVED_TARGET_SEAT_TRANSITION'
        if other and ((row['seat'] == seat and other != actor) or
                      (row['seat'] != seat and other == actor)):
            return 'OBSERVED_STATE_IDENTITY_CONFLICT'
    return None


def prove_direct_actor(context, actor):
    def unknown(reason):
        return dict(status='UNKNOWN', rule=RULE, reason=reason, actor=actor)
    try:
        source = context[SOURCE_KEY]
        nick_policy=source.get('nickname_witness_policy')
        nick_confidence=nickname_witness_floor(nick_policy)
        if source['policy'] != RULE:
            return unknown('ACTOR_POLICY_CONFLICT')
        records = source['chat_records']
        own = [r for r in records if r['actor'] == actor]
        if not own:
            return unknown('SOURCE_BOUND_CURRENT_HAND_CHAT_ACTOR_UNKNOWN')
        if any(any(r[k] != context[k] for k in ('table_id', 'session_id', 'hand_id')) for r in records):
            return unknown('CHAT_ACTOR_CONTEXT_CONFLICT')
        cross_enabled = source.get('cross_source_policy') == CROSS_RULE
        # Only the independently source-bound, current-hand CHAT inventory can
        # corroborate a weak NICK. Roster and amounts cannot supply this set.
        strong_chat = [r for r in records if type(r.get('confidence')) in (int, float)
            and math.isfinite(r['confidence']) and .95 <= r['confidence'] <= 1
            and r.get('source_row', {}).get('confidence') == r['confidence']]
        labels = (qualified_nicks(context, cross_source_actors={r['actor'] for r in strong_chat})
                  if cross_enabled else qualified_nicks(context))
        matched = [(r, label) for r, label in labels if matches(actor, label)]
        if not matched:
            return unknown('QUALIFIED_NICK_EXACT_OR_PREFIX_UNKNOWN')
        seats = {r['seat'] for r, _ in matched}
        if len(seats) != 1:
            return unknown('OBSERVED_SECOND_SEAT_MATCH')
        seat = next(iter(seats))
        known = {r['actor'] for r in records}
        for roster in source['conflict_inventory']['forced_snapshots']:
            known.update(roster['actors'])
        conflict = actual_conflict(context, actor, seat, labels, known)
        if conflict:
            return unknown(conflict)
        strict = ([(r, label) for r, label in matched if r['ocr_response']['score'] >= nick_confidence]
                  if cross_enabled else matched)
        anchor, constraint = min(strict or matched, key=lambda pair: pair[0]['source_frame'])
        identity_rule = RULE if strict else CROSS_RULE
        if identity_rule == CROSS_RULE:
            own = [r for r in strong_chat if r['actor'] == actor]
            if not own:
                return unknown('SOURCE_BOUND_CURRENT_HAND_CHAT_ACTOR_UNKNOWN')
        result = dict(status='PROVEN', rule=identity_rule, reason=None, actor=actor, seat=seat,
            table_id=context['table_id'], session_id=context['session_id'], hand_id=context['hand_id'],
            anchor_frame=anchor['source_frame'], nickname_anchor_source=copy.deepcopy(anchor),
            match_constraint=constraint, chat_actor_sources=copy.deepcopy(own),
            source_files_sha256=copy.deepcopy(source['source_sha256']),
            observed_collision_actors=sorted(known),
            start_stack_authority=False, historical_occupancy_authority=False,
            complete_roster_authority=False, poker_action_authority=False,
            event_uniqueness_authority=False, new_event_source_required=False,
            scope='CURRENT_TABLE_SESSION_HAND_FIXED_NICK_ROI')
        if identity_rule == CROSS_RULE:
            result['cross_source_proof'] = dict(rule=CROSS_RULE,
                raw_nick_text=anchor['ocr_response']['raw_text'],
                nick_confidence=anchor['ocr_response']['score'],
                chat_confidence_threshold=.95, global_nick_confidence_threshold=nick_confidence,
                numeric_confidence_authority=False, matched_constraint=constraint)
        if nick_policy is not None:
            result['nickname_witness_policy']=nick_policy
            result['nickname_confidence_min']=nick_confidence
        return result
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError,
            ZeroDivisionError, cv2.error) as error:
        return unknown(str(error) if isinstance(error, ValueError) else 'SOURCE_RECEIPT_INVALID:' + type(error).__name__)


def prove_actor_point(context, actor, *, native_who=None):
    """Explicit context policy dispatch; historical cases keep their contract."""
    point = prove_direct_actor(context, actor) if SOURCE_KEY in context else legacy_point(context, actor)
    if point['status'] == 'PROVEN' or native_who is None or SOURCE_KEY not in context:
        return point
    native = native_who.for_actor(context, actor)
    return native if native['status'] == 'PROVEN' else point


def numeric_nickname_sources(context, point):
    if point.get('rule') not in IDENTITY_RULES:
        return point['nickname_sources']
    labels = {r['source_frame']: (r, label) for r, label in qualified_nicks(context)
              if r['seat'] == point['seat'] and matches(point['actor'], label)}
    for first in sorted(labels):
        triple = [labels.get(f) for f in range(first, first+3)]
        if all(item is not None for item in triple) and all(item[1] == triple[0][1] for item in triple):
            rows = [item[0] for item in triple]
            if [r['source_pts'] for r in rows] == sorted({r['source_pts'] for r in rows}):
                return rows
    raise ValueError('NUMERIC_NICKNAME_THREE_FRAME_WITNESS_UNKNOWN')


IDENTITY_REUSE_RULE = 'CURRENT_HAND_CROSS_SOURCE_IDENTITY_REUSE_V1'


def reuses_cross_identity(point):
    return point.get('rule') in {CROSS_RULE, NATIVE_NICK_RULE} and point.get('status') == 'PROVEN'


def first_consecutive_frames(frames, reason):
    """Select source frames only; the caller must prove their money/status fact."""
    available = set(frames)
    for first in sorted(available):
        triple = list(range(first, first + 3))
        if all(frame in available for frame in triple):
            return triple
    raise ValueError(reason)


def cross_identity_frame_binding(context, point, frames, *, start_frame, end_frame, native_who=None):
    """Attach independently selected seat sources to current-hand WHO.

    This consumes a freshly reproduced CROSS proof, not nickname OCR at the
    selected frames. Sparse source checkpoints and actual seat transitions
    constrain the whole interval, including the WHO anchor. Money/status
    sources and their verifiers retain all independent authority.
    """
    return _identity_frame_binding(context, point, frames, start_frame=start_frame,
                                   end_frame=end_frame, single=False, native_who=native_who)


def single_numeric_identity_binding(context, point, frame, *, start_frame, end_frame, native_who=None):
    """Reuse fresh current-hand DIRECT/CROSS WHO for one independent numeric source."""
    return _identity_frame_binding(context, point, [frame], start_frame=start_frame,
                                   end_frame=end_frame, single=True, native_who=native_who)


def assert_native_identity_interval(context, point, frames, *, start_frame, end_frame, native_who=None):
    """Additional identity guard; never adds numeric authority or proof fields."""
    if native_who is not None and point.get('rule') in IDENTITY_RULES:
        native_who.assert_use_interval(context, point, frames, start_frame=start_frame, end_frame=end_frame)


def _identity_frame_binding(context, point, frames, *, start_frame, end_frame, single, native_who=None):
    if point.get('rule') == NATIVE_NICK_RULE:
        if native_who is None:
            raise ValueError('FRESH_NATIVE_WHO_SOURCE_REQUIRED')
        return native_who.frame_binding(context, point, frames, start_frame=start_frame,
                                        end_frame=end_frame, single=single)
    assert_native_identity_interval(context, point, frames, start_frame=start_frame,
                                    end_frame=end_frame, native_who=native_who)
    permitted = (point.get('status') == 'PROVEN' and point.get('rule') in IDENTITY_RULES
                 if single else reuses_cross_identity(point))
    if not permitted:
        raise ValueError('CROSS_IDENTITY_PROOF_REQUIRED')
    fields = ('table_id', 'session_id', 'hand_id')
    if any(point.get(k) != context[k] for k in fields):
        raise ValueError('CROSS_IDENTITY_CONTEXT_CONFLICT')
    source = context[SOURCE_KEY]
    if (source.get('policy') != RULE or source.get('cross_source_policy') != CROSS_RULE or
            source['source_sha256'] != point['source_files_sha256']):
        raise ValueError('CROSS_IDENTITY_SOURCE_POLICY_CONFLICT')
    anchor = point['anchor_frame']
    count = 1 if single else 3
    if (len(frames) != count or any(type(f) is not int for f in frames) or
            frames != list(range(frames[0], frames[0]+count)) or
            not context['header_frame'] <= start_frame <= min(anchor, frames[0]) or
            not max(anchor, frames[-1]) <= end_frame <= context['capture_end_inclusive']):
        raise ValueError('CROSS_IDENTITY_VALID_INTERVAL_CONFLICT')
    checkpoints = source['context_checkpoints']
    clock = [r['frame'] for r in checkpoints]
    if clock != sorted(set(clock)) or any(
            (r['table_id'], r['session_id']) != (context['table_id'], context['session_id'])
            for r in checkpoints):
        raise ValueError('CROSS_IDENTITY_CHECKPOINT_CONTEXT_CONFLICT')
    index = bisect_right(clock, start_frame)-1
    if index < 0:
        raise ValueError('CROSS_IDENTITY_INTERVAL_HEADER_UNKNOWN')
    relevant = [checkpoints[index]] + [r for r in checkpoints[index+1:] if r['frame'] <= end_frame]
    if any(r['hand_id'] != context['hand_id'] for r in relevant):
        raise ValueError('CROSS_IDENTITY_HAND_INTERVAL_BARRIER')
    inventory = source['conflict_inventory']
    transitions = []
    for row in inventory['state_receipts']:
        if not start_frame <= row['frame'] <= end_frame:
            continue
        if row.get('status') != 'APPLIED' or row.get('evidence_status') != 'PROVEN':
            continue
        target = row['seat'] == point['seat']
        other = row.get('actor')
        same_status = (native_who is not None and row['kind'] in {'SIT_IN', 'SIT_OUT'} and
                       other in (None, point['actor']))
        if ((target and (row.get('hand_id') != point['hand_id'] or
                         row['kind'] in {'EMPTY', 'REPLACE'} or
                         row['kind'] == 'SIT_IN' and not same_status or
                         other not in (None, point['actor']))) or
                (not target and other == point['actor'])):
            raise ValueError('CROSS_IDENTITY_SEAT_INTERVAL_CONFLICT')
        if target:
            transitions.append(copy.deepcopy(row))
    for row in inventory['unresolved_state_events']:
        same_status = (native_who is not None and row.get('kind') in {'SIT_IN', 'SIT_OUT'} and
                       row.get('actor') in (None, point['actor']))
        if (row.get('seat') == point['seat'] and start_frame <= row['frame'] <= end_frame and
                (row.get('kind') in {'EMPTY', 'REPLACE'} or
                 row.get('kind') == 'SIT_IN' and not same_status or
                 row.get('actor') not in (None, point['actor']))):
            raise ValueError('CROSS_IDENTITY_TARGET_STATE_UNRESOLVED')
    return dict(rule=('CURRENT_HAND_PROVEN_IDENTITY_SINGLE_NUMERIC_FRAME_V1' if single else IDENTITY_REUSE_RULE),
                identity_rule=point['rule'] if single else CROSS_RULE,
                actor=point['actor'], seat=point['seat'],
                **{k: point[k] for k in fields}, anchor_frame=anchor,
                selected_frames=list(frames), valid_interval=[start_frame, end_frame],
                hand_checkpoints=copy.deepcopy(relevant), target_state_receipts=transitions,
                nickname_ocr_rechecked=False, numeric_authority=False, status_authority=False)


def outer_rule(legacy, point):
    return legacy + '__DIRECT_CHAT_NICK_V1' if point['rule'] in IDENTITY_RULES else legacy


def proof_id(proof):
    return hashlib.sha256(json.dumps(proof, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def attach_current_sources(source, loaded):
    """Build policy inputs from whole independent parents, never proof subsets."""
    from pathlib import Path
    from .current_chat_sources import FILES, load_current_chat
    source = Path(source)
    chat = load_current_chat(source, {n: loaded['source_hashes'][n] for n in FILES})
    raw = (source/'seat_forced_summary.json').read_bytes()
    summary = json.loads(raw)
    loaded['source_hashes']['seat_forced_summary.json'] = hashlib.sha256(raw).hexdigest()
    for c in loaded['contexts']:
        key = (c['table_id'], c['session_id'], c['hand_id'])
        def current(r):
            return (r['table_id'], r['session_id'], r['hand_id']) == key
        def summary_current(r):
            return (r['table'], r['session'], r['hand_id']) == key
        inventory = {n: [r for r in summary[n] if summary_current(r)]
                     for n in ('forced_snapshots', 'bound_posts')}
        # A later seat change is relevant only if captured in this NICK window.
        for name in ('state_receipts','unresolved_state_events'):
            inventory[name] = [r for r in summary[name]
                if (r['table'],r['session']) == key[:2] and
                c['header_frame'] <= r['frame'] <= c['capture_end_inclusive']]
        c[SOURCE_KEY] = dict(policy=RULE, cross_source_policy=CROSS_RULE,
            nickname_witness_policy=FIRST_NICK_POLICY,
            source_sha256=dict(loaded['source_hashes']),
            chat_records=[r for r in chat['records'] if current(r)],
            context_checkpoints=[r for r in chat['context_checkpoints']
                if (r['table_id'], r['session_id']) == key[:2]],
            conflict_inventory=inventory)
    if hashlib.sha256((source/'seat_forced_summary.json').read_bytes()).hexdigest() != loaded['source_hashes']['seat_forced_summary.json']:
        raise ValueError('ACTOR_CONFLICT_PARENT_CHANGED_DURING_LOAD')
    return loaded


def matches_hand(proof, hand):
    key = (int(str(hand['table_id']).removeprefix('T')), hand['session_id'], hand['hand_id'])
    if key != (proof['table_id'], proof['session_id'], proof['hand_id']):
        return False
    # R IDs persist across observations. Join the exact occurrence rather than
    # letting a later, possibly weaker reading overwrite an earlier witness.
    def occurrence(row):
        return (row['stable_row_id'], row['frame_id'], row['row_index'])
    rows = {}
    for message in hand['messages']:
        for row in message['source_rows']:
            rows.setdefault(occurrence(row), []).append(row)
    fields = ('stable_row_id', 'frame_id', 'row_index', 'raw_ocr_text', 'confidence', 'row_pixel_sha256',
              'table_id', 'session_id', 'epoch', 'admission_ref', 'video_time')
    for r in proof['chat_actor_sources']:
        current = rows.get(occurrence(r['source_row']), [])
        header = rows.get(occurrence(r['header_source']['row']), [])
        if not current or not header:
            continue
        if any(row.get(k) != r['source_row'].get(k) for row in current for k in fields):
            continue
        if any(row.get(k) != r['header_source']['row'].get(k) for row in header for k in fields):
            continue
        # Event deduplication is not an identity-evidence rejection. A repeated
        # source can confirm WHO, but has no authority to insert an action.
        if any(row.get('canonical_conflict') or row.get('status')=='ERROR' for row in current):
            continue
        return True
    return False


def point_matches_hand(point,hand):
    return point.get('rule') not in IDENTITY_RULES or matches_hand(point,hand)


class ActorSeatSourceProvider:
    """Identity-only channel; independently loaded sources bracket each use."""
    def __init__(self, source, hands, *, native_nick=False):
        from pathlib import Path
        from .actor_seat_point import HEADER
        from .joint_stack_sources import load_contexts
        self.source = Path(source)
        self.hands = {}; windows = []
        for hand in hands:
            key = (int(str(hand['table_id']).removeprefix('T')), hand['session_id'], hand['hand_id'])
            if key in self.hands:
                raise ValueError('ACTOR_DUPLICATE_HAND_CONTEXT')
            self.hands[key] = copy.deepcopy(hand)
            headers = [r['frame_id'] for m in hand['messages'] for r in m['source_rows']
                if HEADER.fullmatch(r['raw_ocr_text'].strip()) and
                HEADER.fullmatch(r['raw_ocr_text'].strip())[1] == hand['hand_id'] and
                (':R' in str(r.get('stable_row_id')) or ':F' in str(r.get('stable_row_id')))]
            if headers:
                windows.append(dict(table_id=key[0], session_id=key[1], hand_id=key[2],
                    header_frame=min(headers), capture_end_inclusive=min(headers)+64))
        loaded = load_contexts(self.source, windows, actor_policy=True)
        self.hashes = loaded['source_hashes']
        self.native_who = None
        if native_nick:
            from .native_nick_who import NativeNickWhoSource
            self.native_who = NativeNickWhoSource(self.source, hands, expected_hashes=self.hashes,
                nickname_witness_policy=FIRST_NICK_POLICY)
        self.contexts = {(c['table_id'], c['session_id'], c['hand_id']): c for c in loaded['contexts']}
        self.proofs = {}; self.rows = []
        for key, hand in self.hands.items():
            context = self.contexts.get(key)
            for actor in sorted({m['player'] for m in hand['messages'] if m.get('player')}):
                p = prove_actor_point(context, actor, native_who=self.native_who) if context else dict(status='UNKNOWN',
                    rule=RULE, reason='NATIVE_CURRENT_HEADER_UNKNOWN', actor=actor)
                if p['status'] == 'PROVEN' and not matches_hand(p, hand):
                    p = dict(status='UNKNOWN', rule=RULE, reason='CURRENT_HAND_CHAT_SOURCE_JOIN_UNKNOWN', actor=actor)
                self.rows.append(dict(table_id=key[0], session_id=key[1], hand_id=key[2],
                    actor=actor, status=p['status'], reason=p['reason'], seat=p.get('seat'),
                    proof_rule=p.get('rule')))
                if p['status'] == 'PROVEN':
                    self.proofs[(key, actor)] = p
        if not self.sources_unchanged():
            raise ValueError('ACTOR_SOURCE_CHANGED_DURING_LOAD')

    def sources_unchanged(self):
        return all(hashlib.sha256((self.source/n).read_bytes()).hexdigest() == digest
                   for n, digest in self.hashes.items()) and (
                       self.native_who is None or self.native_who.sources_unchanged())

    def for_hand(self, hand):
        key = (int(str(hand['table_id']).removeprefix('T')), hand['session_id'], hand['hand_id'])
        if self.hands.get(key) != hand or key not in self.contexts or not self.sources_unchanged():
            return []
        result = []
        for (saved, actor), prior in self.proofs.items():
            if saved != key:
                continue
            fresh = prove_actor_point(self.contexts[key], actor, native_who=self.native_who)
            if fresh == prior and matches_hand(fresh, hand):
                result.append(fresh)
        return result if self.sources_unchanged() else []

    def for_hand_identity_uses(self, hand):
        """Bracket use scopes together; keep each retained proof unchanged."""
        points = self.for_hand(hand)
        uses = [(point, self.native_who.scope_for_point(hand, point)
                 if self.native_who is not None else None) for point in points]
        return uses if self.sources_unchanged() else []

    def write_audit(self, output):
        from pathlib import Path
        from collections import Counter
        output = Path(output)
        (output/'actor_seat_audit.json').write_text(json.dumps(dict(rule=RULE, identity_rules=sorted(IDENTITY_RULES),
            counts=dict(Counter(r['status'] for r in self.rows)), rows=self.rows,
            source_hashes=self.hashes), ensure_ascii=False, indent=2), 'utf8')
        with (output/'actor_seat_proofs.jsonl').open('w', encoding='utf8') as stream:
            for p in self.proofs.values():
                stream.write(json.dumps(p, ensure_ascii=False)+'\n')
