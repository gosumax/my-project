"""Independent single-crop first-number proof; historical SQL admission is unchanged."""
import base64
import copy
import json
from fractions import Fraction

import cv2
import numpy as np

from parser_core.actor_seat_point import sha
from parser_core.chat_nick_actor import prove_actor_point,outer_rule,single_numeric_identity_binding
from parser_core.constant_stack_balance import numeric_sources
from parser_core.first_stack_ocr import (
    single_first_numeric_policy,single_first_numeric,direct_numeric_policy,
    effective_numeric,bound_scaled_numeric,numeric)
from parser_core.joint_start_stack import own_posts,FORCED_PREFIX_OCR_POLICY
from parser_core.start_stack_visibility import VisibilitySource,first_visible_proof,SINGLE_CONTRACT

RULE='FIRST_VISIBLE_SINGLE_NUMERIC'
SAVED_SINGLE_CONTRACT='POKERDOM_FIRST_SAVED_NUMERIC_CURRENT_HAND_V1'


def first_saved_numeric_proof(readings, frames, *, start_frame):
    """First accepted saved ROI number; preceding pixel classes grant no authority.

    Source/identity/forced-post joins belong to the caller. This contract keeps
    the existing original-primary numeric policy and checks the selected pixels
    and clock. Missing or unclassified pre-number crops are not rejection rules.
    """
    def unknown(reason):
        return dict(contract=SAVED_SINGLE_CONTRACT,status='UNKNOWN',reason=reason)
    if (not isinstance(start_frame,int) or isinstance(start_frame,bool) or
            len(frames)!=1 or frames[0]<start_frame):
        return unknown('FIRST_SAVED_NUMERIC_SOURCE_RANGE_UNKNOWN')
    selected=readings.get(frames[0])
    if selected is None:
        return unknown('FIRST_SAVED_NUMERIC_SOURCE_UNKNOWN')
    context=tuple(selected.get(k) for k in ('table_id','session_id','hand_id','seat'))
    if any(v is None for v in context) or selected.get('source_frame')!=frames[0]:
        return unknown('FIRST_SAVED_NUMERIC_CONTEXT_UNKNOWN')
    value=single_first_numeric(selected)
    if value is None:
        return unknown('FIRST_SINGLE_NUMBER_NOT_PROVEN')
    # Firstness refers to available saved recognizer results, not a claim that
    # every earlier video frame was observed or visually classified.
    earlier=[]
    for frame,row in sorted(readings.items()):
        if not start_frame<=frame<frames[0]:
            continue
        earlier.append(frame)
        if single_first_numeric(row) is not None:
            return unknown('EARLIER_ACCEPTED_SAVED_NUMBER')
    if VisibilitySource().verify(selected) is None:
        return unknown('FIRST_SAVED_NUMERIC_PIXELS_UNPROVEN')
    try:
        stamp=Fraction(selected['source_pts']*selected['time_base'][0],selected['time_base'][1])
    except (KeyError,TypeError,ValueError,IndexError,ZeroDivisionError):
        return unknown('FIRST_SAVED_NUMERIC_TIME_UNKNOWN')
    return dict(contract=SAVED_SINGLE_CONTRACT,status='PROVEN',
        reason='FIRST_ACCEPTED_SAVED_STACK_ROI_NUMBER_AFTER_HAND_START',
        start_frame=start_frame,first_numeric_frame=frames[0],frames=list(frames),
        displayed_stack=value,context=list(context),source_evidence=[selected],
        source_time=[stamp.numerator,stamp.denominator],
        pre_numeric_saved_frames=earlier,pre_numeric_pixel_classification='NOT_REQUIRED',
        complete_pre_numeric_coverage='NOT_REQUIRED',required_numeric_frames=1,
        single_first_numeric_policy=single_first_numeric_policy(),
        authority='USER_FIRST_SAVED_NUMERIC_POLICY_20261010')


def is_single_proof(proof):
    return proof.get('rule') in {RULE,RULE+'__DIRECT_CHAT_NICK_V1'}


def original_numeric_readings(case,direct,observations,states):
    """Bind original OCR to its exact crop; SQL stores raw OCR, not authority here."""
    readings=copy.deepcopy(direct)
    sources={}
    verifier=VisibilitySource()
    for frame,row in direct.items():
        if single_first_numeric(row) is not None:
            sources.setdefault(frame,[]).append(dict(
                kind='NATIVE_DIRECT_OCR_CALLBACK',source_frame=frame,
                raw_text=row['raw_text'],confidence=row['confidence'],
                item_id=row['ocr_response']['item_id'],
                source_rgb_sha256=row['visibility_source']['source_rgb_sha256']))
    # numeric_sources has verified the full frozen database, metadata, raw
    # provenance, observation context/time and captured pixel representations.
    # The original global numeric/status fields below remain unchanged.
    for observation in observations.values():
        state=states[observation['state_id']]
        frame=state['frame_index']
        if frame not in direct or state['ocr_called']!=1 or state['digit_gate_status']!='DIGITS_VISIBLE':
            continue
        view=copy.deepcopy(direct[frame])
        # This is an explicit proof-local view of original raw OCR. Keep the
        # original callback and UNKNOWN SQL row intact in source_case.
        view.pop('ocr_response',None)
        view.pop('numeric_source_evidence',None)
        view.update(raw_text=state['ocr_raw'],confidence=state['ocr_confidence'],ocr_status='OK',
                    single_numeric_raw_source='SAVED_RAW_OCR_IN_SQL_CONTAINER')
        value=single_first_numeric(view)
        if value is None:
            continue
        if state['ocr_numeric'] not in (None,value):
            raise ValueError('RAW_OCR_SQL_VALUE_CONFLICT')
        projected=verifier.verify(view)
        raw=base64.b64decode(state['raw_gray_base64'],validate=True)
        pixels=np.frombuffer(raw,np.uint8).reshape(state['height'],state['width'])
        if (projected is None or not np.array_equal(projected['gray'],pixels) or
                sha(raw)!=state['exact_hash'] or sha(raw)!=state['raw_gray_sha256']):
            continue
        source=dict(kind='NATIVE_CROP_BOUND_RAW_OCR_IN_SQL_CONTAINER',
            source_frame=frame,raw_text=state['ocr_raw'],confidence=state['ocr_confidence'],
            raw_state_id=state['state_id'],raw_observation_id=observation['observation_id'],
            refined_gray_sha256=state['exact_hash'],
            source_rgb_sha256=view['visibility_source']['source_rgb_sha256'],
            original_sql_status=state['status'],original_sql_numeric=state['ocr_numeric'],
            sql_numeric_authority=False,new_ocr_performed=False)
        prior=sources.setdefault(frame,[])
        if any(numeric(p['raw_text'])!=value for p in prior):
            raise ValueError('SINGLE_NATIVE_OCR_VALUE_CONFLICT')
        prior.append(source)
        if single_first_numeric(readings[frame]) is None:
            readings[frame]=view
    return readings,sources


def prove(case,inventory,check_seals):
    """Only fresh current source inventory can grant this distinct direct proof."""
    def unknown(reason):
        return dict(status='UNKNOWN',rule=RULE,reason=reason,actor=case.get('actor'),
                    scope='FIRST_VISIBLE_SINGLE_NUMERIC')
    try:
        if inventory is None or not inventory.matches(case):
            return unknown('FRESH_COMPLETE_WINDOW_INVENTORY_REQUIRED')
        if check_seals() is not None:
            return unknown('SOURCE_SEAL_CHANGED')
        context=case['source_context']
        point=prove_actor_point(context,case['actor'],native_who=getattr(inventory,'native_who',None))
        if point['status']!='PROVEN':
            return unknown('ACTOR_POINT_'+point['reason'])
        if any(point.get(k)!=case[k] for k in ('table_id','session_id','hand_id','actor','seat')):
            return unknown('ACTOR_CONTEXT_CONFLICT')
        posts=own_posts(context,case['actor'])
        from parser_core.direct_start_stack import full_readings,verify_sql_conflicts
        direct=full_readings(case)
        # This verifier and its .95 numeric route are not relaxed or rewritten.
        sql,_,observations,states=numeric_sources(case,case['database_metadata'])
        readings,sources=original_numeric_readings(case,direct,observations,states)
        frames=sorted(f for f,r in readings.items()
            if f>=context['header_frame'] and single_first_numeric(r) is not None)
        if not frames:
            return unknown('FIRST_SINGLE_NUMERIC_SOURCE_UNKNOWN')
        first_frame=frames[0]
        frames=[first_frame]
        identity=single_numeric_identity_binding(context,point,first_frame,
            start_frame=context['header_frame'],end_frame=max(first_frame,point['anchor_frame']),native_who=getattr(inventory,'native_who',None))
        first=first_visible_proof(readings,frames,start_frame=context['header_frame'],
            confidence=.93,single_first_policy=single_first_numeric_policy())
        # Preserve successful historical proofs exactly. Only UNKNOWN invokes
        # the user-authorized saved-number contract; no pre-number pixel gate.
        if first['status']!='PROVEN':
            first=first_saved_numeric_proof(readings,frames,start_frame=context['header_frame'])
        if first['status']!='PROVEN':
            return unknown('FIRST_VISIBLE_'+first['reason'])
        displayed=int(first['displayed_stack'])
        if displayed<=0:
            return unknown('POSITIVE_NUMBER_REQUIRED')
        checked_sql=verify_sql_conflicts(sql,context['header_frame'],frames,displayed)
        checked_direct=[]
        policy=direct_numeric_policy()
        for frame,row in sorted(direct.items()):
            if not context['header_frame']<=frame<=first_frame:
                continue
            primary=numeric(row.get('raw_text'))
            scaled=bound_scaled_numeric(row)
            if scaled is not None and scaled[1]>=policy['min_confidence'] and primary!=scaled[0]:
                return unknown('SOURCE_BOUND_SCALED_NUMERIC_CONFLICT')
            value=effective_numeric(row,policy['min_confidence'],allow_scaled=True,numeric_policy=policy)
            if value is None:
                continue
            if frame<first_frame:
                return unknown('INDEPENDENT_DIRECT_EARLIER_ACCEPTED_NUMBER')
            if int(value)!=displayed:
                return unknown('INDEPENDENT_DIRECT_NUMERIC_CONFLICT')
            checked_direct.append(dict(frame=frame,value=int(value),pixel_sha256=row['pixel_sha256']))
        if check_seals() is not None:
            return unknown('SOURCE_SEAL_CHANGED')
        return dict(status='PROVEN',rule=outer_rule(RULE,point),reason=None,
            actor=case['actor'],seat=case['seat'],table_id=case['table_id'],
            session_id=case['session_id'],hand_id=case['hand_id'],
            displayed_stack=displayed,forced_amount_added_back=sum(p['amount'] for p in posts),
            calculated_start_stack=displayed+sum(p['amount'] for p in posts),
            identity_proof=point,identity_frame_binding=identity,own_forced_posts=posts,
            forced_prefix_ocr_policy=copy.deepcopy(FORCED_PREFIX_OCR_POLICY),
            single_first_numeric_policy=single_first_numeric_policy(),first_visible_proof=first,
            numeric_frames=frames,native_numeric_sources=copy.deepcopy(sources[first_frame]),
            independent_sql_sources_checked=checked_sql,independent_direct_sources_checked=checked_direct,
            source_hashes=copy.deepcopy(inventory.hashes),source_case=copy.deepcopy(case),
            numeric_authority='FIRST_VISIBLE_SINGLE_NUMERIC_NATIVE_CROP_RAW_OCR',
            sql_acceptance_threshold_unchanged=.95,sql_numeric_authority=False,
            occupancy_epoch=None,manufactured_sql_observation_ids=False,scope=RULE)
    except (KeyError,TypeError,ValueError,IndexError,AttributeError,OverflowError,
            ZeroDivisionError,OSError,cv2.error) as error:
        return unknown(str(error) if isinstance(error,ValueError) else type(error).__name__+':'+str(error))


def join_receipt(proof):
    if (not is_single_proof(proof) or proof.get('status')!='PROVEN' or
            proof.get('single_first_numeric_policy')!=single_first_numeric_policy() or
            proof['first_visible_proof']['contract'] not in {SINGLE_CONTRACT,SAVED_SINGLE_CONTRACT} or len(proof['numeric_frames'])!=1):
        raise ValueError('SINGLE_FIRST_NUMERIC_RECEIPT_CONFLICT')
    from parser_core.direct_start_stack import proof_id
    return dict(rule=proof['rule'],proof_id=proof_id(proof),actor=proof['actor'],seat=proof['seat'],
        table_id=proof['table_id'],session_id=proof['session_id'],hand_id=proof['hand_id'],
        calculated_start_stack=proof['calculated_start_stack'],displayed_stack=proof['displayed_stack'],
        forced_amount_added_back=proof['forced_amount_added_back'],numeric_frames=proof['numeric_frames'],
        forced_prefix_ocr_policy=copy.deepcopy(proof['forced_prefix_ocr_policy']),
        single_first_numeric_policy=copy.deepcopy(proof['single_first_numeric_policy']),
        native_numeric_sources=copy.deepcopy(proof['native_numeric_sources']),
        source_files_sha256=proof['source_hashes'],scope='FIRST_VISIBLE_SINGLE_NUMERIC_NO_SQL_AUTHORITY')
