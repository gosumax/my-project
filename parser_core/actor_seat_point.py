"""Positive current-hand actor at a fixed seat, never occupancy/stack authority."""
import base64
import copy
import hashlib
import math
import re
from collections import defaultdict
from fractions import Fraction

import cv2
import numpy as np

from parser_core.nickname_roster_binding import literal_constraint, matches, _seat_rois, _verify_gray_source
from parser_core.ram_first_stage import png_compatible_gray

RULE='NATIVE_ANTE_EXACT_NICKNAME_SEAT_POINT_V1'
SCOPE='CURRENT_HAND_PARTICIPANT_AT_OBSERVED_ANCHOR_ONLY'
HEADER=re.compile(r'(?:Hand\s*)?#\s*(\d+)',re.I)
ANTE=re.compile(r'(?:Dealer|Дилер)\s*:\s*(.+?) posts Ante (\d+(?:[ \u00a0]\d{3})*)',re.I)
START=re.compile(r'(?:Dealer|Дилер)\s*:\s*Starting a new hand[.!]?',re.I)


def ante_candidates(rows):
    """Literal own-ANTE inventory, without seat, occupancy or stack authority.

    Reuse the registered forced-post grammar for an amount on the immediately
    following physical row. Keep low-confidence/malformed candidates visible
    for rejection by the point prover; never replace either source string.
    """
    from parser_core.seat_forced_binding import POST, NUMERIC, adjacent_forced_rows
    result=[]
    for index,row in enumerate(rows):
        text=row['raw_ocr_text'].strip();complete=ANTE.fullmatch(text);head=POST.fullmatch(text)
        wrapped=bool(head and head[2].lower()=='ante' and head[3] is None and head[4] is None)
        if not complete and not wrapped:continue
        actor=complete[1] if complete else head[1];sources=[row];reason=None
        amount=complete[2] if complete else None
        if wrapped:
            tail=rows[index+1] if index+1<len(rows) else None
            def physical(r):
                return dict(r,source_frame=r.get('frame_id'),index=r.get('row_index'))
            if (tail is None or type(row.get('frame_id')) is not int or type(row.get('row_index')) is not int or
                    any(row.get(k) is None for k in ('table_id','session_id','epoch')) or
                    not adjacent_forced_rows(physical(row),physical(tail)) or
                    any(row.get(k)!=tail.get(k) for k in ('source_pts','time_base','video_time')) or
                    not NUMERIC.fullmatch(tail['raw_ocr_text'].strip())):
                reason='OWN_ANTE_AMOUNT_WRAP_UNPROVEN'
            else:sources.append(tail);amount=tail['raw_ocr_text'].strip()
        if any(type(r.get('confidence')) not in (int,float) or not math.isfinite(r['confidence']) or
               not .95<=r['confidence']<=1 for r in sources):reason='OWN_ANTE_SOURCE_CONFIDENCE_UNKNOWN'
        result.append(dict(actor=actor,source_rows=sources,reason=reason,
                           amount=int(amount.replace(' ','').replace('\u00a0','')) if amount is not None else None))
    return result


def sha(value):return hashlib.sha256(value).hexdigest()


def response_text(response):
    if (response.get('status')!='OK' or type(response.get('score')) not in (int,float)
            or not math.isfinite(response['score']) or not 0<=response['score']<=1):
        raise ValueError('OCR_RESPONSE_INVALID')
    return response['raw_text'].strip()


def unique(rows,key):
    result={}
    for row in rows:
        k=key(row)
        if k in result:raise ValueError('DUPLICATE_SOURCE')
        result[k]=row
    return result


def source_time(pts,base):
    if (type(pts) is not int or pts<0 or len(base)!=2 or
            any(type(v) is not int or v<=0 for v in base)):
        raise ValueError('SOURCE_TIME_INVALID')
    return Fraction(pts*base[0],base[1])


def frame_times(c):
    """Independent native capture owns the frame/PTS/pixel join."""
    frames={}
    def add(frame,pts,base):
        if type(frame) is not int or not c['header_frame']<=frame<=c['capture_end_inclusive']:
            raise ValueError('SOURCE_FRAME_OUTSIDE_WINDOW')
        time=source_time(pts,base)
        if frame in frames and frames[frame]!=time:raise ValueError('SAME_FRAME_PTS_CONFLICT')
        frames[frame]=time
    captures=unique(c['native_capture_nick_rows'],lambda r:(r['seat'],r['source_frame']))
    for r in captures.values():
        if (r['table_id'],r['session_id'],r['field'])!=(c['table_id'],c['session_id'],'nick'):
            raise ValueError('NATIVE_CAPTURE_CONTEXT_CONFLICT')
        add(r['source_frame'],r['source_pts'],r['time_base'])
    for r in c['nickname_observations_all_within_window']:
        cap=captures[(r['seat'],r['source_frame'])]
        for k in ('table_id','session_id','source_frame','source_pts','time_base','seat','roi'):
            if r[k]!=cap[k]:raise ValueError('NICK_NATIVE_CAPTURE_JOIN_CONFLICT')
        native=cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(cap['png_base64'],validate=True),np.uint8),1),cv2.COLOR_BGR2RGB)
        if sha(native.tobytes())!=cap['pixel_sha256'] or not np.array_equal(png_compatible_gray(native),_verify_gray_source(r)):
            raise ValueError('NICK_NATIVE_RGB_TO_GRAY_CONFLICT')
        add(r['source_frame'],r['source_pts'],r['time_base'])
    for r in c['header_interval_records']:
        add(r['source_frame'],r['source_pts'],r['time_base'])
    for r in c['state_interval_receipts']:
        add(r['source_frame'],r['source_pts'],c['ordinary_prefix_admission']['time_base'])
    ordered=[frames[f] for f in sorted(frames)]
    if ordered!=sorted(set(ordered)):raise ValueError('SOURCE_TIME_ORDER_CONFLICT')
    return frames


def source_rows(c):
    """Join two coordinate domains without asserting global row identity."""
    native=c['ordinary_prefix_timeline']
    physical=unique(c['ordinary_prefix_physical_rows'],lambda r:r['row_id'])
    ocr=unique(c['ordinary_prefix_raw_ocr'],lambda r:r['row_id'])
    admission=c['ordinary_prefix_admission']
    admitted=unique(admission['admissions'],lambda r:r['row_id'])
    table,session,frame=c['table_id'],c['session_id'],c['header_frame']
    source_time(admission['source_pts'],admission['time_base'])
    if (admission['table']!=table or admission['session_id']!=session or
            admission['frame_after']!=frame or admission['proof']['status']!='PROVEN_PHYSICAL_NEW' or
            admission['old_identity_reused'] is not False):
        raise ValueError('PHYSICAL_NEW_SOURCE_UNKNOWN')
    offset=len(admission['geometry']['top_edge_spans'])
    rows=unique(native,lambda r:r['stable_row_id'])
    if set(rows)!=set(physical) or set(rows)!=set(ocr):raise ValueError('NATIVE_SOURCE_SET_CONFLICT')
    indices=[r['row_index'] for r in native]
    if not indices or indices!=list(range(indices[0],indices[0]+len(indices))):
        raise ValueError('PREFIX_SOURCE_GAP')
    result=[]
    for row in native:
        rid=row['stable_row_id'];p=physical[rid];raw=ocr[rid];a=admitted[rid]
        if (row['table_id']!=f'T{table:02d}' or row['session_id']!=session or row['frame_id']!=frame
                or p['table_id']!=table or p['table_session_id']!=session or p['frame_id']!=frame
                or p['source_span_index']!=row['row_index'] or a['index']+offset!=row['row_index']
                or a['OCR_admission'] is not True or a['index'] not in admission['proof']['new_indices']
                or admission['new_spans'][a['index']]!=a['span']
                or p['origin']!='NEW_CANDIDATE' or row['origin']!='NEW_CANDIDATE'
                or p['admission_ref']!=f'{session}:{frame}:physical_new:{a["index"]}'
                or row['admission_ref']!=p['admission_ref']
                or p['epoch']!=row['epoch'] or p['pixel_dtype']!='uint8'):
            raise ValueError('PHYSICAL_NEW_ROW_JOIN_CONFLICT')
        fields=('row_key','table_id','table_session_id','frame_id','source_span_index','epoch',
                'source_pts','time_base_num','time_base_den','row_pixel_sha256')
        if any(raw[k]!=p[k] for k in fields):raise ValueError('RAW_OCR_SOURCE_CONFLICT')
        b=base64.b64decode(p['row_pixels_base64'],validate=True)
        pixels=np.frombuffer(b,np.uint8).reshape(p['pixel_shape'])
        if (pixels.ndim!=2 or sha(b)!=p['row_pixel_sha256'] or row['row_pixel_sha256']!=sha(b)
                or pixels.shape[1]!=admission['geometry']['shape'][1]):
            raise ValueError('PHYSICAL_ROW_PIXELS_CONFLICT')
        lo,hi=[v-p['y1'] for v in a['span']]
        if not 0<=lo<hi<=pixels.shape[0] or sha(pixels[lo:hi].tobytes())!=a['row_pixel_sha256']:
            raise ValueError('ADMISSION_UNPADDED_PIXELS_CONFLICT')
        if (p['source_pts']!=admission['source_pts'] or
                [p['time_base_num'],p['time_base_den']]!=admission['time_base'] or
                row['video_time']!=float(source_time(p['source_pts'],[p['time_base_num'],p['time_base_den']]))):
            raise ValueError('NATIVE_PTS_CONFLICT')
        text=response_text(raw['raw_ocr_response'])
        if text!=row['raw_ocr_text'].strip() or raw['raw_ocr_response']['score']!=row['confidence']:
            raise ValueError('NATIVE_TEXT_CONFLICT')
        result.append((row,p,raw['raw_ocr_response'],pixels))
    return result


def prove_actor_point(c,actor):
    def unknown(reason):return dict(status='UNKNOWN',reason=reason,rule=RULE,actor=actor,scope=SCOPE)
    try:
        table,session,hand=c['table_id'],c['session_id'],c['hand_id']
        frame_times(c)
        sources=source_rows(c)
        ante_by_row={p['source_rows'][0]['stable_row_id']:p for p in ante_candidates([r[0] for r in sources])}
        headers=[];posts=[];number=None
        for row,p,response,pixels in sources:
            text=response_text(response);header=HEADER.fullmatch(text);post=ante_by_row.get(row['stable_row_id'])
            if START.fullmatch(text):number=None
            if header:
                if header[1]!=hand:return unknown('FOREIGN_PREFIX_HEADER')
                number=header[1]
                if response['score']>=.95:headers.append((row,p,pixels))
            if post and post['actor']==actor:
                if number!=hand or post['reason']:return unknown('ANTE_CONTEXT_OR_OCR_UNKNOWN')
                posts.append(post['source_rows'])
        if len(posts)!=1 or not headers:return unknown('POSITIVE_OWN_ANTE_OR_NUMBER_UNKNOWN')
        # Independent header admission and callback must reproduce the same
        # positive source. A hand_id assigned by another consumer is insufficient.
        confirmed=[]
        for row,p,pixels in headers:
            hs=[r for r in c['header_receipts_admitted_and_confirmed'] if
                r['source_frame']==p['frame_id'] and r['source_span_index']==p['source_span_index']]
            admitted=[r for r in hs if r['status']=='ADMITTED'];done=[r for r in hs if r['status']=='HAND_HEADER_CONFIRMED']
            if len(admitted)!=1 or len(done)!=1:continue
            a,d=admitted[0],done[0]
            if any(r.get('table_id')!=table or r.get('session_id')!=session or
                   r.get('pixel_sha256')!=p['row_pixel_sha256'] or r.get('source_pts')!=p['source_pts'] or
                   r.get('time_base')!=[p['time_base_num'],p['time_base_den']] for r in (a,d)):continue
            png=cv2.imdecode(np.frombuffer(base64.b64decode(a['png_base64'],validate=True),np.uint8),0)
            match=HEADER.fullmatch(response_text(d['ocr_response']))
            if (png is None or not np.array_equal(png,pixels) or a['gate']['candidate'] is not True or
                    match is None or match[1]!=hand or d['ocr_response']['score']<.95 or
                    d['current_hand_id']!=hand):continue
            state=[r for r in c['hand_state_receipts'] if r.get('status')=='HAND_HEADER_CONFIRMED' and
                   all(r.get(k)==d.get(k) for k in ('table_id','session_id','source_frame','source_pts','source_span_index'))
                   and r.get('hand_id')==hand]
            if state:confirmed.append(row)
        if not confirmed:return unknown('INDEPENDENT_POSITIVE_HEADER_UNKNOWN')
        labels=defaultdict(dict);all_labels=[]
        for row in c['nickname_observations_all_within_window']:
            if ((row.get('table_id'),row.get('session_id'),row.get('hand_id'))!=(table,session,hand)
                    or row['kind']!='NICK' or type(row['source_frame']) is not int or
                    not c['header_frame']<=row['source_frame']<=c['capture_end_inclusive']):
                return unknown('NICK_SOURCE_CONTEXT_CONFLICT')
            response=row.get('ocr_response',{})
            if response.get('status')!='OK':continue
            text=response_text(response)
            if response['score']<.95:continue
            constraint=literal_constraint(text)
            if constraint is None:continue
            seat=row['seat'];pixels=_verify_gray_source(row)
            roi=_seat_rois()[f'slot_{table:02d}']['seats'][seat]['nick']
            if row['roi']!=roi or pixels.shape!=(roi[3]-roi[1],roi[2]-roi[0]):
                return unknown('NICK_FIXED_SEAT_PIXELS_CONFLICT')
            if (row['time_base']!=c['ordinary_prefix_admission']['time_base'] or
                    type(row['source_pts']) is not int or row['source_pts']<=c['ordinary_prefix_admission']['source_pts']):
                return unknown('NICK_SOURCE_TIME_CONFLICT')
            if row['source_frame'] in labels[seat]:return unknown('DUPLICATE_NICK_SOURCE')
            labels[seat][row['source_frame']]=(row,constraint);all_labels.append((row,constraint))
        triples=[]
        for seat,frames in labels.items():
            for end in frames:
                triple=[frames.get(f) for f in range(end-2,end+1)]
                if any(r is None for r in triple):continue
                if (all(r[1]==dict(kind='EXACT',text=actor) for r in triple) and
                        [r[0]['source_pts'] for r in triple]==sorted({r[0]['source_pts'] for r in triple})):
                    triples.append((end,seat,[r[0] for r in triple]))
        if not triples:return unknown('OWN_EXACT_THREE_FRAME_NICK_UNKNOWN')
        anchor=min(t[0] for t in triples);first=[t for t in triples if t[0]==anchor]
        if len(first)!=1:return unknown('ACTOR_MULTIPLE_SEAT_CONFLICT')
        _,seat,nicks=first[0]
        if any((r['seat']==seat and not matches(actor,label)) or (r['seat']!=seat and matches(actor,label))
               for r,label in all_labels if r['source_frame']<=anchor):
            return unknown('POSITIVE_NICKNAME_CONTRADICTION_BEFORE_ANCHOR')
        # Complete interval inputs are supplied afresh by the consumer. No
        # occupancy transition or historical epoch is manufactured here.
        interval=[r for r in c['header_interval_records'] if r['source_frame']<=anchor]
        if not interval or not c['state_interval_receipts']:return unknown('HAND_INTERVAL_UNKNOWN')
        pending=set()
        for r in interval:
            if r.get('table_id')!=table or r.get('session_id')!=session:return unknown('HEADER_INTERVAL_CONTEXT_CONFLICT')
            k=(r['source_frame'],r['source_span_index'])
            if r['status']=='ADMITTED':pending.add(k)
            elif r['status']=='HAND_HEADER_CONFIRMED':
                m=HEADER.fullmatch(response_text(r['ocr_response']))
                if m is None or m[1]!=hand or r['current_hand_id']!=hand:return unknown('FOREIGN_HEADER_IN_INTERVAL')
                pending.discard(k)
            else:return unknown('UNRESOLVED_HEADER_IN_INTERVAL')
        if pending:return unknown('PENDING_HEADER_IN_INTERVAL')
        for r in c['state_interval_receipts']:
            if r['source_frame']>anchor:continue
            if (r.get('table_id')!=table or r.get('session_id')!=session or r.get('hand_id')!=hand or
                    r.get('status') not in {'HAND_HEADER_CONFIRMED','STALE_HEADER_IGNORED'}):
                return unknown('HAND_STATE_INTERVAL_CONFLICT')
        current=unique(c['ordinary_rows_in_capture_window'],lambda r:r['stable_row_id'])
        for r in current.values():
            if ((r['table_id'],r['session_id'])!=(f'T{table:02d}',session) or
                    not c['header_frame']<=r['frame_id']<=c['capture_end_inclusive'] or
                    type(r['row_index']) is not int):return unknown('CURRENT_NATIVE_WINDOW_CONFLICT')
        if any(current[r['stable_row_id']]!=r for r in c['ordinary_prefix_timeline']):
            return unknown('CURRENT_NATIVE_PREFIX_CONFLICT')
        expected={r['stable_row_id'] for r in current.values() if
                  HEADER.fullmatch(r['raw_ocr_text'].strip()) or START.fullmatch(r['raw_ocr_text'].strip()) or
                  r.get('event_type') in {'HAND_HEADER','HAND_ID','NEW_HAND'}}
        retained=unique(c['ordinary_hand_markers_in_capture_window'],lambda r:r['stable_row_id'])
        if set(retained)!=expected:return unknown('ORDINARY_MARKER_SET_CONFLICT')
        for rid,r in retained.items():
            if any(current[rid].get(k)!=v for k,v in r.items()):return unknown('ORDINARY_MARKER_SOURCE_CONFLICT')
        active=None
        markers=sorted([current[rid] for rid in expected],key=lambda r:(r['frame_id'],r['row_index']))
        for r in markers:
            if r['frame_id']>anchor:continue
            text=r['raw_ocr_text'].strip();m=HEADER.fullmatch(text)
            if m:
                if m[1]!=hand:return unknown('FOREIGN_ORDINARY_HAND_MARKER')
                active=hand
            elif START.fullmatch(text):active=None
            else:return unknown('UNRESOLVED_ORDINARY_HAND_MARKER')
        if active!=hand:return unknown('ORDINARY_CURRENT_HAND_UNKNOWN')
        result=dict(status='PROVEN',reason=None,rule=RULE,scope=SCOPE,actor=actor,seat=seat,
            table_id=table,session_id=session,hand_id=hand,anchor_frame=anchor,
            ante_source=copy.deepcopy(posts[0][0]),header_source=copy.deepcopy(confirmed[-1]),
            nickname_sources=copy.deepcopy(nicks),source_inputs=copy.deepcopy(c),
            historical_occupancy_authority=False,start_stack_authority=False,
            event_uniqueness_authority=False,complete_roster_authority=False)
        if len(posts[0])>1:result['ante_source_rows']=copy.deepcopy(posts[0])
        return result
    except (KeyError,TypeError,ValueError,IndexError,AttributeError,OverflowError,ZeroDivisionError,cv2.error):
        return unknown('SOURCE_RECEIPT_INVALID')
