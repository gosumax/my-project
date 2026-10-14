"""Fresh whole-file inventory of source-bound current-hand CHAT actor rows.

Uses the registered native physical-row join at EVERY retained source frame.
It never proves a complete participant roster or actor-to-seat identity.
"""
import base64
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from parser_core.actor_seat_point import HEADER, START, source_time, unique, response_text
from parser_core.chat_nick_actor import chat_actor_candidates

FILES=('ordered_timeline.jsonl','physical_rows.jsonl','raw_ocr_results.jsonl',
       'physical_chat_admission.jsonl','hand_header_ocr.jsonl','hand_state_receipts.jsonl')
RULE='NATIVE_PHYSICAL_CHAT_ACTOR_WITH_CURRENT_CONFIRMED_HEADER_V1'
OBSERVATION_RULE='CURRENT_NATIVE_CHAT_OBSERVATION_IDENTITY_ONLY_V1'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def canonical_sha(value):
    return sha(json.dumps(value,ensure_ascii=False,sort_keys=True,
                         separators=(',',':'),allow_nan=False).encode('utf-8'))


def table_key(row):
    return int(str(row.get('table_id',row.get('table'))).removeprefix('T')),row.get('session_id',row.get('table_session_id'))


def observation_source_rows(packet):
    """Prove what was read at this frame, without asserting a new event.

    R is a tracker identity, not proof of physical novelty or a poker hand.
    The independent header state below supplies the hand ownership separately.
    """
    table,session,frame=packet['table_id'],packet['session_id'],packet['header_frame']
    rows=packet['ordinary_prefix_timeline']
    physical=unique(packet['ordinary_prefix_physical_rows'],lambda r:r['row_id'])
    ocr=unique(packet['ordinary_prefix_raw_ocr'],lambda r:r['row_id'])
    native=unique(rows,lambda r:r['stable_row_id'])
    if set(native)!=set(physical) or set(native)!=set(ocr):
        raise ValueError('OBSERVATION_PARENT_SET_CONFLICT')
    indices=[r['row_index'] for r in rows]
    if not indices or any(type(i) is not int or i<0 for i in indices) or indices!=sorted(set(indices)):
        raise ValueError('OBSERVATION_ROW_ORDER_CONFLICT')
    result=[]
    for row in rows:
        rid=row['stable_row_id'];p=physical[rid];raw=ocr[rid]
        row_key=f"{table}:{frame}:{row['row_index']}"
        if (row['table_id']!=f'T{table:02d}' or row['session_id']!=session or row['frame_id']!=frame or
            p['table_id']!=table or p['table_session_id']!=session or p['frame_id']!=frame or
            p['source_span_index']!=row['row_index'] or p['row_key']!=row_key or
            p['epoch']!=row['epoch'] or p['origin']!=row['origin'] or p['pixel_dtype']!='uint8'):
            raise ValueError('OBSERVATION_CONTEXT_CONFLICT')
        fields=('row_key','row_id','table_id','table_session_id','frame_id','source_span_index',
                'epoch','source_pts','time_base_num','time_base_den','row_pixel_sha256')
        if any(raw[k]!=p[k] for k in fields):raise ValueError('OBSERVATION_RAW_SOURCE_CONFLICT')
        b=base64.b64decode(p['row_pixels_base64'],validate=True)
        pixels=np.frombuffer(b,np.uint8).reshape(p['pixel_shape'])
        if pixels.ndim!=2 or not pixels.size or sha(b)!=p['row_pixel_sha256'] or row['row_pixel_sha256']!=sha(b):
            raise ValueError('OBSERVATION_PIXEL_CONFLICT')
        stamp=source_time(p['source_pts'],[p['time_base_num'],p['time_base_den']])
        if row['video_time']!=float(stamp):raise ValueError('OBSERVATION_CLOCK_CONFLICT')
        response=raw['raw_ocr_response'];text=response_text(response)
        if response.get('item_id')!=row_key or text!=row['raw_ocr_text'].strip() or response['score']!=row['confidence']:
            raise ValueError('OBSERVATION_OCR_CALLBACK_CONFLICT')
        result.append((row,p,response,pixels))
    return result


def confirmed_header(row,p,response,pixels,headers,states):
    """Same independent positive header checks as the registered point route."""
    number=HEADER.fullmatch(response_text(response))
    if number is None or response['score']<.95:return None
    table,session=table_key(row)
    hs=[r for r in headers if r.get('source_frame')==p['frame_id'] and
        r.get('source_span_index')==p['source_span_index']]
    admitted=[r for r in hs if r['status']=='ADMITTED']
    done=[r for r in hs if r['status']=='HAND_HEADER_CONFIRMED']
    if len(admitted)!=1 or len(done)!=1:return None
    a,d=admitted[0],done[0]
    if any(r.get('table_id')!=table or r.get('session_id')!=session or
           r.get('pixel_sha256')!=p['row_pixel_sha256'] or r.get('source_pts')!=p['source_pts'] or
           r.get('time_base')!=[p['time_base_num'],p['time_base_den']] for r in (a,d)):
        return None
    png=cv2.imdecode(np.frombuffer(base64.b64decode(a['png_base64'],validate=True),np.uint8),0)
    match=HEADER.fullmatch(response_text(d['ocr_response']))
    if (png is None or not np.array_equal(png,pixels) or a['gate']['candidate'] is not True or
            match is None or match[1]!=number[1] or d['ocr_response']['score']<.95 or
            d['current_hand_id']!=number[1]):return None
    state=[r for r in states if r.get('status')=='HAND_HEADER_CONFIRMED' and
        all(r.get(k)==d.get(k) for k in ('table_id','session_id','source_frame','source_pts','source_span_index')) and
        r.get('hand_id')==number[1]]
    if not state:return None
    return dict(hand_id=number[1],row=row,admitted=a,confirmed=d,state_receipts=state)


def load_current_chat(source,expected_source_sha256):
    """Read full independent parents; expected pins must not come from a proof."""
    source=Path(source)
    if set(expected_source_sha256)!=set(FILES):raise ValueError('COMPLETE_CHAT_PARENT_SET_REQUIRED')
    parents={name:(source/name).read_bytes() for name in FILES}
    hashes={name:sha(raw) for name,raw in parents.items()}
    if hashes!=expected_source_sha256:raise ValueError('CHAT_PARENT_SEAL_CONFLICT')
    timeline=defaultdict(list);physical=defaultdict(list);ocr=defaultdict(list)
    admissions=defaultdict(list);headers=defaultdict(list);states=defaultdict(list)
    raw_counts={}
    for name,raw in parents.items():
        count=0
        for line_number,line in enumerate(raw.splitlines(),1):
            if not line.strip():continue
            row=json.loads(line);count+=1
            if row.get("status")=="SUMMARY":continue
            key=table_key(row)
            if name=='ordered_timeline.jsonl':
                if row.get('source') in {'FORCED_SNAPSHOT','SIDECAR_HEADER'}:continue
                timeline[(*key,row['frame_id'])].append(row)
            elif name=='physical_rows.jsonl':physical[(*key,row['frame_id'])].append(row)
            elif name=='raw_ocr_results.jsonl':ocr[(*key,row['frame_id'])].append(row)
            elif name=='physical_chat_admission.jsonl':admissions[(*key,row['frame_after'])].append(row)
            elif name=='hand_header_ocr.jsonl':headers[key].append(row)
            elif name=='hand_state_receipts.jsonl':states[key].append(row)
        raw_counts[name]=count
    packets={};records=[];failures=[];unassigned=[];positive_headers=[]
    active={}; checkpoints=[];ownership=[];first_owner={}
    last_positive={};closed_hands=defaultdict(set)
    def retain_owner(table,session,row,current,join_status):
        number=current['header']['hand_id'] if current else None
        key=(table,session,row['stable_row_id'])
        prior=first_owner.get(key)
        ownership.append(dict(table_id=table,session_id=session,row_id=row['stable_row_id'],
            frame=row['frame_id'],row_index=row['row_index'],hand_id=number,
            first_proven_hand_id=prior,source_join_status=join_status))
        if number is not None and prior is None:first_owner[key]=number
        return prior is not None and number is not None and prior!=number
    barriers=defaultdict(list)
    for key, entries in headers.items():
        grouped=defaultdict(list)
        for r in entries:grouped[(r['source_frame'],r['source_span_index'])].append(r)
        for (frame,index), group in grouped.items():
            stale=any(r.get('status')=='STALE_HEADER_IGNORED' and r['source_frame']==frame and
                      r['source_span_index']==index for r in states[key])
            if stale:continue
            done=[r for r in group if r['status']=='HAND_HEADER_CONFIRMED']
            admitted=[r for r in group if r['status']=='ADMITTED']
            number=None
            if len(done)==len(admitted)==1 and len(group)==2:
                m=HEADER.fullmatch(response_text(done[0]['ocr_response']))
                # This is a negative constraint only, never a positive header.
                # A same-number repeated reading cannot erase an independently
                # proven hand merely because the repeated OCR score is lower.
                if m and done[0]['current_hand_id']==m[1]:number=m[1]
            barriers[(*key,frame)].append(number)
    for key, entries in states.items():
        for r in entries:
            if r['status']=='STALE_HEADER_IGNORED':continue
            barriers[(*key,r['source_frame'])].append(r.get('hand_id') if r['status']=='HAND_HEADER_CONFIRMED' else None)
    for table,session,frame in sorted(set(timeline)|set(barriers)):
        rows=timeline[(table,session,frame)]
        key=(table,session);frame_key=(*key,frame)
        # Reobserving a positively closed hand is historical identity evidence,
        # not a new hand transition. Hand IDs are scoped to this table/session.
        frame_barriers=set(barriers[frame_key])-closed_hands[key]
        for number in frame_barriers:
            previous=active.get(key)
            if number is None or previous is None or previous['header']['hand_id']!=number:
                active[key]=None
        if not rows:
            checkpoints.append(dict(table_id=table,session_id=session,frame=frame,
                hand_id=active[key]['header']['hand_id'] if active.get(key) else None))
            continue
        ids={r['stable_row_id'] for r in rows}
        packet=dict(table_id=table,session_id=session,header_frame=frame,
            ordinary_prefix_timeline=rows,
            ordinary_prefix_physical_rows=[r for r in physical[frame_key] if r['row_id'] in ids],
            ordinary_prefix_raw_ocr=[r for r in ocr[frame_key] if r['row_id'] in ids])
        joined=None;reason=None
        try:
            packet['retained_event_admission_receipts']=admissions[frame_key]
            packet['identity_observation_rule']=OBSERVATION_RULE
            joined=observation_source_rows(packet)
        except (KeyError,ValueError,TypeError,IndexError,AttributeError,OverflowError,ZeroDivisionError,cv2.error) as e:
            reason=type(e).__name__+':'+str(e)
            failures.append(dict(table_id=table,session_id=session,frame=frame,reason=reason,
                                 retained_native_row_ids=[r['stable_row_id'] for r in rows]))
        if joined is None:
            # A failed frame may contain a new hand. Never carry an old label
            # through an unverified marker merely because row joining failed.
            if any(HEADER.fullmatch(r['raw_ocr_text'].strip()) or START.fullmatch(r['raw_ocr_text'].strip()) or
                   r.get('event_type') in {'HAND_HEADER','HAND_ID','NEW_HAND'} for r in rows):
                active[key]=None
            for row in rows:retain_owner(table,session,row,active.get(key),'SOURCE_JOIN_FAILED')
            checkpoints.append(dict(table_id=table,session_id=session,frame=frame,hand_id=None if not active.get(key) else active[key]['header']['hand_id']))
            continue
        packet_id=canonical_sha(packet)
        packets[packet_id]=packet
        candidates={r['source']['stable_row_id']:r for r in chat_actor_candidates(joined)}
        historical_region=False;historical_context=None
        for row,p,response,pixels in joined:
            text=response_text(response)
            marker=HEADER.fullmatch(text)
            prior_owner=first_owner.get((table,session,row['stable_row_id']))
            historical_row=prior_owner in closed_hands[key]
            if START.fullmatch(text) and not historical_row and not historical_region:active[key]=None
            if marker:
                if marker[1] in closed_hands[key]:
                    # Rows below a visible historical header cannot become
                    # current-hand observations just because the live epoch is
                    # newer. Keep frame-local CHAT region apart from live state.
                    historical_region=True
                    old_header=confirmed_header(row,p,response,pixels,headers[key],states[key])
                    historical_context=dict(header=old_header) if old_header else None
                    retain_owner(table,session,row,historical_context,'HISTORICAL_HEADER_OBSERVATION')
                    continue
                header=confirmed_header(row,p,response,pixels,headers[key],states[key])
                if frame_barriers and frame_barriers!={marker[1]}:header=None
                active[key]=dict(header=header,packet_ref=packet_id) if header else None
                historical_region=False;historical_context=None
                if header:
                    old=last_positive.get(key)
                    if old is not None and old!=header['hand_id']:closed_hands[key].add(old)
                    last_positive[key]=header['hand_id']
                    positive_headers.append(dict(table_id=table,session_id=session,
                        hand_id=header['hand_id'],frame=frame,row_id=row['stable_row_id'],packet_ref=packet_id))
            elif row.get('event_type') in {'HAND_HEADER','HAND_ID','NEW_HAND'} and not historical_row and not historical_region:
                active[key]=None
            current=historical_context if historical_region else active.get(key)
            foreign_owner=retain_owner(table,session,row,current,'SOURCE_OBSERVATION_PROVEN')
            candidate=candidates.get(row['stable_row_id'])
            if candidate is None:continue
            if current is None or candidate['reason'] is not None or foreign_owner or historical_region:
                unassigned.append(dict(table_id=table,session_id=session,frame=frame,
                    actor=candidate['actor'],reason=candidate['reason'] or
                    ('HISTORICAL_CHAT_REGION' if historical_region else
                     'OBSERVED_FOREIGN_HAND_ROW_OWNER' if foreign_owner else 'CURRENT_HEADER_UNKNOWN'),
                    row_id=row['stable_row_id'],packet_ref=packet_id))
                continue
            header=current['header']
            records.append(dict(rule=RULE,status='SOURCE_BOUND_CHAT_ACTOR',
                table_id=table,session_id=session,hand_id=header['hand_id'],actor=candidate['actor'],
                row_id=row['stable_row_id'],frame=frame,row_index=row['row_index'],
                raw_text=row['raw_ocr_text'],confidence=row['confidence'],
                packet_ref=packet_id,header_packet_ref=current['packet_ref'],
                header_row_id=header['row']['stable_row_id'],header_frame=header['row']['frame_id'],
                header_source=header,source_row=row,
                observation_rule=OBSERVATION_RULE,new_event_source_required=False,
                poker_action_authority=False,event_uniqueness_authority=False,
                whole_roster_authority=False,actor_seat_authority=False))
        checkpoints.append(dict(table_id=table,session_id=session,frame=frame,hand_id=active[key]['header']['hand_id'] if active.get(key) else None))
    if any(sha((source/name).read_bytes())!=hashes[name] for name in FILES):
        raise ValueError('CHAT_PARENT_CHANGED_DURING_LOAD')
    return dict(schema='full-retained-current-chat-actor-inventory-v1',source_sha256=hashes,
        raw_counts=raw_counts,records=records,packets=packets,frame_failures=failures,context_checkpoints=checkpoints,
        retained_row_ownership=ownership,
        unassigned_actor_rows=unassigned,positive_headers=positive_headers,
        complete_participant_roster_authority=False,actor_seat_authority=False,
        coverage='ALL_RETAINED_NATIVE_TIMELINE_FRAMES; NOT_EXHAUSTIVE_PHYSICAL_VIDEO_OR_CHAT_TEXT_COVERAGE')
