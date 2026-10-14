"""Independent first-stack proof from a named participant's own sources."""
import base64
import copy
import json
import math
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from parser_core.actor_seat_point import prove_actor_point, source_rows, source_time, sha, unique, HEADER, START
from parser_core.ram_first_stage import png_compatible_gray
from parser_core.start_stack_visibility import VisibilitySource, first_visible_proof
from parser_core.numeric_text_evidence import exact_text_temporal_binding

ROOT=Path(__file__).resolve().parents[1]
from parser_core.chat_nick_actor import prove_actor_point, numeric_nickname_sources, outer_rule, point_matches_hand

RULE='OWN_FORCED_PREFIX_FIRST_NUMBER_AND_EXACT_ACTOR_POINT_V1'
FORCED_PREFIX_OCR_POLICY=dict(
    policy_id='USER_APPROVED_FORCED_PREFIX_OCR_093_20261008_V1',
    min_confidence=.93,small_blind_tail_min_confidence=.93,
    scope='SAVED_OWN_FORCED_PREFIX_ONLY')
NUMBER=r'\d+(?:[ \u00a0]\d{3})*'
POST=re.compile(r'(?:Dealer|Дилер):\s*(.+?) posts (Ante|Small Blind|Big Blind) ('+NUMBER+')',re.I)
POST_HEAD=re.compile(r'(?:Dealer|Дилер):\s*(.+?) posts (Small Blind|Big Blind)',re.I)
DEAL=re.compile(r'(?:Dealer|Дилер):\s*Dealing cards[.!●]?',re.I)


def own_posts(context,actor):
    """Re-use the registered forced-post grammar without occupancy authority."""
    from parser_core.seat_forced_binding import parse_forced_snapshot
    sources=source_rows(context)
    native=[r[0] for r in sources]
    rows=[dict(table_id=context['table_id'],session_id=context['session_id'],hand_id=context['hand_id'],
        epoch=r['epoch'],source_frame=p['frame_id'],source_pts=p['source_pts'],
        time_base=[p['time_base_num'],p['time_base_den']],index=r['row_index'],
        source_ref=r['row_pixel_sha256'],text=r['raw_ocr_text'],score=r['confidence'])
        for r,p,_,_ in sources]
    # This count checks the source prefix's distinct ante inventory only. It
    # neither counts occupied seats nor makes the returned roster authoritative.
    participants={re.fullmatch(r'(?:Dealer|Дилер):\s*(.+?) posts Ante(?: .*|)',r['text'],re.I)[1]
                  for r in rows if re.fullmatch(r'(?:Dealer|Дилер):\s*(.+?) posts Ante(?: .*|)',r['text'],re.I)}
    snapshot,reason=parse_forced_snapshot(rows,context['hand_id'],len(participants),
        confidence=FORCED_PREFIX_OCR_POLICY['min_confidence'])
    if snapshot is None:raise ValueError('OWN_PREFIX_'+str(reason))
    if snapshot['state_events'] or snapshot['deal_index']!=rows[-1]['index']:
        raise ValueError('OWN_PREFIX_STATE_OR_TERMINAL_CONFLICT')
    covered={r['index'] for p in snapshot['posts'] for r in p['source_rows']}
    for row in rows:
        text=row['text'].strip()
        if row['index'] in covered or DEAL.fullmatch(text):continue
        header=HEADER.fullmatch(text)
        if header and header[1]==context['hand_id'] or START.fullmatch(text):continue
        raise ValueError('UNEXPLAINED_FORCED_PREFIX_SOURCE')
    by_index={r['row_index']:r for r in native}
    posts=[dict(actor=actor,role={'ANTE':'ANTE','SB':'SMALL_BLIND','BB':'BIG_BLIND'}[p['role']],
        amount=p['amount'],source_rows=[copy.deepcopy(by_index[r['index']]) for r in p['source_rows']])
        for p in snapshot['posts'] if p['actor']==actor]
    counts=Counter(p['role'] for p in posts)
    if counts['ANTE']!=1 or counts['SMALL_BLIND']+counts['BIG_BLIND']>1 or any(p['amount']<=0 for p in posts):
        raise ValueError('OWN_FORCED_POST_SET_UNKNOWN')
    return posts


def prove(case,metadata):
    def unknown(reason):return dict(status='UNKNOWN',rule=RULE,reason=reason,actor=case.get('actor'))
    try:
        c=case['source_context'];identity=prove_actor_point(c,case['actor'])
        if identity['status']!='PROVEN':return unknown('ACTOR_POINT_'+str(identity['reason']))
        if any(case.get(k)!=identity.get(k) for k in ('actor','seat','table_id','session_id','hand_id')):
            return unknown('ACTOR_STACK_CONTEXT_CONFLICT')
        posts=own_posts(c,case['actor'])
        meta={k:json.loads(r['value']) for k,r in unique(metadata,lambda r:r['key']).items()}
        if meta['ocr_acceptance_threshold']!=.95 or meta['schema']!='stack-evidence-timeline-v1':
            return unknown('SQL_NUMERIC_CONTRACT_CONFLICT')
        if (set(meta['config_hashes'])!={'stack_digit_gate.json','stack_roi_refined.json',
                'stack_sitout_mask.json','ocr_tiny_local.json'} or
                any(not isinstance(v,str) or re.fullmatch('[0-9a-f]{64}',v) is None
                    for v in [meta['algorithm_sha256'],*meta['config_hashes'].values()])):
            return unknown('SQL_FROZEN_CONFIG_RECEIPTS_INCOMPLETE')
        for name,digest in meta['config_hashes'].items():
            if sha((ROOT/'configs'/name).read_bytes())!=digest:return unknown('SQL_CONFIG_HASH_CONFLICT')
        if sha((ROOT/'parser_core/stack_timeline_algorithms.py').read_bytes())!=meta['algorithm_sha256']:
            return unknown('SQL_ALGORITHM_HASH_CONFLICT')
        captures=unique([r['raw_record'] for r in case['raw_stack_captures']],lambda r:r['source_frame'])
        states=unique(case['raw_sql_visual_states'],lambda r:r['state_id'])
        observations=unique(case['raw_sql_observations'],lambda r:r['observation_id'])
        state_pixels={}
        for state in states.values():
            if (state['table_id'],state['seat_id'])!=(case['table_id'],case['seat']):
                return unknown('SQL_STATE_CONTEXT_CONFLICT')
            b=base64.b64decode(state['raw_gray_base64'],validate=True)
            pixels=np.frombuffer(b,np.uint8).reshape(state['height'],state['width'])
            if sha(b)!=state['exact_hash'] or sha(b)!=state['raw_gray_sha256']:return unknown('SQL_STATE_PIXELS_CONFLICT')
            payload=json.loads(state['provenance'])['state_payload']
            for k in ('state_id','table_id','seat_id','frame_index','exact_hash','ocr_called','ocr_raw','ocr_numeric','ocr_confidence'):
                if state[k]!=payload[k]:return unknown('SQL_OCR_PROVENANCE_CONFLICT')
            state_pixels[state['state_id']]=pixels
        numeric=[];readings={};bindings=[];vs=VisibilitySource();verified_ocr_sources=set()
        nick_times={r['source_frame']:source_time(r['source_pts'],r['time_base']) for r in c['native_capture_nick_rows']}
        for f,cap in sorted(captures.items()):
            if (any(cap[k]!=case[k] for k in ('table_id','session_id','seat')) or cap['field']!='stack'
                    or source_time(cap['source_pts'],cap['time_base'])!=nick_times[f]):
                return unknown('STACK_NICK_SAME_FRAME_CONFLICT')
            rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(cap['png_base64'],validate=True),np.uint8),1),cv2.COLOR_BGR2RGB)
            x,y,X,Y=cap['roi'];a,b,A,B=vs.rois[f"slot_{case['table_id']:02d}"]['seats'][case['seat']]['stack']
            if (rgb.shape!=(Y-y,X-x,3) or not x<=a<A<=X or not y<=b<B<=Y or sha(rgb.tobytes())!=cap['pixel_sha256']):
                return unknown('STACK_RGB_SOURCE_CONFLICT')
            gray=cv2.cvtColor(rgb[b-y:B-y,a-x:A-x],cv2.COLOR_RGB2GRAY);digest=sha(gray.tobytes());matches=[]
            for o in observations.values():
                state=states[o['state_id']]
                if (any(o[k]!=case[k] for k in ('table_id','session_id')) or o['seat_id']!=case['seat']
                        or o['frame_index']!=state['frame_index'] or o['state_id']!=state['source_state_id']
                        or not o['frame_index']<=o['confirmed_frame']<=o['last_seen_frame']):
                    return unknown('SQL_OBSERVATION_CONTEXT_CONFLICT')
                if o['frame_index']==f and (source_time(o['source_pts'],json.loads(o['source_time_base']))!=nick_times[f]):
                    return unknown('SQL_CAPTURE_TIMESTAMP_CONFLICT')
                if not o['frame_index']<=f<=o['last_seen_frame']:continue
                numeric_source=None
                if np.array_equal(gray,state_pixels[o['state_id']]):
                    numeric_source=dict(evidence_kind='SOURCE_GRAY_PIXELS_IDENTICAL',original_ocr_frame=state['frame_index'],
                        source_state_id=state['state_id'],source_observation_id=o['observation_id'],
                        original_pixel_sha256=state['exact_hash'],target_frame=f,target_pixel_sha256=digest,
                        ocr_inference_count=1,new_ocr_performed=False)
                elif state['state_id'] in verified_ocr_sources:
                    numeric_source=exact_text_temporal_binding(gray,state,o,f,vs.gate)
                if numeric_source is None:continue
                if (state['status']=='CONFIRMED_NUMERIC' and state['digit_gate_status']=='DIGITS_VISIBLE'
                        and state['ocr_called']==1 and type(state['ocr_confidence']) in (int,float)
                        and math.isfinite(state['ocr_confidence']) and .95<=state['ocr_confidence']<=1
                        and re.fullmatch(NUMBER,state['ocr_raw'].strip())
                        and state['ocr_raw'].strip().replace(' ','').replace('\u00a0','')==state['ocr_numeric']):
                    matches.append((o,state,numeric_source))
                    if state['frame_index']==f:verified_ocr_sources.add(state['state_id'])
            if len({s['ocr_numeric'] for _,s,_ in matches})>1:return unknown('SQL_NUMERIC_VALUE_CONFLICT')
            selected=matches[0] if matches else None
            direct=png_compatible_gray(rgb);ok,png=cv2.imencode('.png',direct)
            if not ok:return unknown('SOURCE_ENCODING_FAILED')
            readings[f]=dict(table_id=case['table_id'],session_id=case['session_id'],hand_id=case['hand_id'],
                seat=case['seat'],source_frame=f,source_pts=cap['source_pts'],time_base=cap['time_base'],roi=cap['roi'],
                pixel_sha256=sha(direct.tobytes()),png_base64=base64.b64encode(png).decode(),
                raw_text=selected[1]['ocr_raw'] if selected else '',confidence=selected[1]['ocr_confidence'] if selected else None,
                ocr_status='OK' if selected else 'NOT_READ',numeric_source_evidence=selected[2] if selected else None,
                visibility_source=dict(source_rgb_roi=cap['roi'],
                source_rgb_sha256=cap['pixel_sha256'],source_rgb_png_base64=cap['png_base64'],timeline_roi=[a,b,A,B],
                timeline_pixel_sha256=digest,timeline_shape=list(gray.shape),digit_gate=vs.gate.evaluate(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR))))
            if selected:
                numeric.append(f);bindings.append(dict(frame=f,observation_id=selected[0]['observation_id'],
                    state_id=selected[0]['state_id'],pixel_sha256=digest,numeric_source_evidence=selected[2]))
        triples=[list(range(f,f+3)) for f in numeric if all(n in numeric for n in range(f,f+3))]
        if not triples:return unknown('FIRST_NUMERIC_TRIPLE_UNKNOWN')
        frames=triples[0]
        if [r['source_frame'] for r in numeric_nickname_sources(c,identity)]!=frames:
            return unknown('FIRST_NUMBER_EXACT_NICKNAME_TRIPLE_NOT_SAME_SOURCE')
        proof=first_visible_proof(readings,frames,start_frame=c['header_frame'])
        if proof['status']!='PROVEN':return unknown('FIRST_VISIBLE_'+proof['reason'])
        display=int(proof['displayed_stack']);forced=sum(p['amount'] for p in posts)
        return dict(status='PROVEN',rule=outer_rule(RULE,identity),reason=None,actor=case['actor'],seat=case['seat'],
            table_id=case['table_id'],session_id=case['session_id'],hand_id=case['hand_id'],
            displayed_stack=display,forced_amount_added_back=forced,calculated_start_stack=display+forced,
            identity_proof=identity,own_forced_posts=posts,first_visible_proof=proof,sql_numeric_bindings=bindings,
            forced_prefix_ocr_policy=copy.deepcopy(FORCED_PREFIX_OCR_POLICY),
            source_case=copy.deepcopy(case),database_metadata=copy.deepcopy(metadata),
            authority='CURRENT_OWN_FORCED_PREFIX_FIRST_NUMBER_CONTRACT_EXACT_NICKNAME_POINT',
            global_occupancy_authority=False,historical_epoch_authority=False)
    except (KeyError,TypeError,ValueError,IndexError,AttributeError,OverflowError,ZeroDivisionError,OSError,cv2.error):
        return unknown('JOINT_SOURCE_RECEIPT_INVALID')


def matches_current_hand(proof,hand):
    """An immutable retained proof cannot be transplanted to another source."""
    try:
        if ((int(str(hand['table_id']).lstrip('T')),hand['session_id'],hand['hand_id'])!=
                (proof['table_id'],proof['session_id'],proof['hand_id'])):return False
        if not point_matches_hand(proof.get('identity_proof',{}),hand):return False
        context=proof['source_case']['source_context']
        fields=('table_id','session_id','frame_id','row_index','stable_row_id',
                'raw_ocr_text','row_pixel_sha256','video_time','epoch','admission_ref')
        def native(row):
            rid=row.get('stable_row_id')
            return bool(rid and ':F' in rid or rid and ':R' in rid)
        rows=[r for m in hand['messages'] for r in m['source_rows'] if native(r) and
              context['header_frame']<=r['frame_id']<=context['capture_end_inclusive']]
        expected=context['ordinary_rows_in_capture_window']
        sources={r['row_id']:r for r in context['ordinary_prefix_physical_rows']}
        for row in rows:
            if (row.get('status') in {'ERROR','UNRESOLVED'} or row.get('origin')=='UNRESOLVED_IDENTITY'
                    or row.get('canonical_conflict') or row.get('canonical_duplicate_of')):return False
            if row.get('source_pts') is not None or row.get('time_base') is not None:
                p=sources.get(row['stable_row_id'])
                if p is None or source_time(row['source_pts'],row['time_base'])!=source_time(
                        p['source_pts'],[p['time_base_num'],p['time_base_den']]):return False
        signature=lambda rs:[[r.get(k) for k in fields] for r in rs]
        return signature(rows)==signature(expected)
    except (KeyError,TypeError,ValueError,ZeroDivisionError):return False


def verify_retained(proof,hand,current_context=None):
    try:
        if proof.get('rule')==RULE+'__DIRECT_CHAT_NICK_V1' and current_context!=proof['source_case']['source_context']:
            return False
        return (proof.get('rule') in {RULE,RULE+'__DIRECT_CHAT_NICK_V1'} and proof.get('status')=='PROVEN'
                and matches_current_hand(proof,hand)
                and proof==prove(proof['source_case'],proof['database_metadata']))
    except (KeyError,TypeError,ValueError):return False


def proof_id(proof):
    return sha(json.dumps(proof,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8'))


def join_receipt(proof):
    return dict(rule=proof['rule'],proof_id=proof_id(proof),actor=proof['actor'],seat=proof['seat'],
        table_id=proof['table_id'],session_id=proof['session_id'],hand_id=proof['hand_id'],
        START_STACK=proof['calculated_start_stack'],displayed_stack=proof['displayed_stack'],
        forced_amount_added_back=proof['forced_amount_added_back'],
        forced_prefix_ocr_policy=copy.deepcopy(proof['forced_prefix_ocr_policy']),
        source_observation_ids=sorted({b['observation_id'] for b in proof['sql_numeric_bindings']}),
        source_state_ids=sorted({b['state_id'] for b in proof['sql_numeric_bindings']}),
        numeric_frames=proof['first_visible_proof']['frames'],scope='OWN_FORCED_FIRST_NUMBER_NO_OCCUPANCY_EPOCH')
