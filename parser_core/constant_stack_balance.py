"""Independent saved-source balance proof; the first-visible rule is unchanged."""
import base64
import copy
import hashlib
import json
import math
import re
import sqlite3
from collections import defaultdict, Counter
from pathlib import Path
from dataclasses import replace
import cv2
import numpy as np
from parser_core.actor_seat_point import prove_actor_point, source_time, sha, unique, HEADER, response_text
from parser_core.joint_start_stack import own_posts, NUMBER, FORCED_PREFIX_OCR_POLICY
from parser_core.nickname_roster_binding import literal_constraint, matches
from parser_core.ram_first_stage import png_compatible_gray
from parser_core.start_stack_visibility import VisibilitySource, ForcedOverlayPatterns
from parser_core.stack_timeline_storage import stack_timeline_source_name
from parser_core.numeric_text_evidence import exact_text_temporal_binding
from parser_core.adjacent_stack_carry import rows, key
from parser_core.poker_reconstruction.assembly import candidate_assemblies
from parser_core.poker_reconstruction.timeline import parse_rows
from parser_core.poker_reconstruction.reconstruct import reconstruct_hand
from parser_core.reconstruction_validation import validate_reconstruction
from parser_core.table_service_events import describe_service
from parser_core.joint_stack_sources import load_contexts, FILES as CONTEXT_FILES
from parser_core.adjacent_stack_carry import file_sha256
from parser_core.first_stack_ocr import effective_numeric, proof_config
from parser_core.nickname_actor_binding import _deal_empty_refs
ROOT=Path(__file__).resolve().parents[1]
from parser_core.chat_nick_actor import prove_actor_point, numeric_nickname_sources, outer_rule, point_matches_hand

RULE='CONSTANT_POST_FORCED_BALANCE_EXACT_ACTOR_SOURCE_V1'


def native_hand_bounds(hand,registry):
    """The source headers, not a possibly shortened logical hand, bound time."""
    table,session,number=key(hand);table=int(str(table).removeprefix('T'))
    headers=[r for r in registry.headers if (r['table_id'],r['session_id'])==(table,session)]
    def raw_number(r):
        m=HEADER.fullmatch(str(r.get('ocr_response',{}).get('raw_text','')).strip())
        return m[1] if m else None
    done=[r for r in headers if r['status']=='HAND_HEADER_CONFIRMED']
    starts=[r['source_frame'] for r in done if r.get('current_hand_id')==number or raw_number(r)==number]
    if not starts:raise ValueError('INDEPENDENT_NATIVE_HAND_START_UNKNOWN')
    low=min(starts)
    later=[r for r in done if r['source_frame']>low and
           (r.get('current_hand_id') not in (None,number) or raw_number(r) not in (None,number))]
    if not later:raise ValueError('INDEPENDENT_NEXT_HAND_BOUNDARY_UNKNOWN')
    boundary=min(r['source_frame'] for r in later)
    next_numbers=set()
    for r in [h for h in done if h['source_frame']==boundary]:
        m=HEADER.fullmatch(response_text(r['ocr_response']))
        if (m is None or m[1]==number or r.get('current_hand_id')!=m[1] or r['ocr_response']['score']<.95):
            raise ValueError('INDEPENDENT_NEXT_HAND_HEADER_CONFLICT')
        next_numbers.add(m[1])
        admissions=[a for a in headers if a['status']=='ADMITTED' and
            (a['source_frame'],a['source_span_index'])==(r['source_frame'],r['source_span_index'])]
        if len(admissions)!=1:raise ValueError('INDEPENDENT_NEXT_HEADER_ADMISSION_UNKNOWN')
        a=admissions[0]
        if any(a.get(k)!=r.get(k) for k in ('table_id','session_id','source_frame','source_span_index','source_pts','time_base','pixel_sha256')):
            raise ValueError('INDEPENDENT_NEXT_HEADER_SOURCE_CONFLICT')
        pixels=cv2.imdecode(np.frombuffer(base64.b64decode(a['png_base64'],validate=True),np.uint8),0)
        if pixels is None or sha(pixels.tobytes())!=a['pixel_sha256'] or a['gate']['candidate'] is not True:
            raise ValueError('INDEPENDENT_NEXT_HEADER_PIXELS_CONFLICT')
        registry.verify_header_source(r,a,pixels,require_physical=True)
        if not any(s.get('status')=='HAND_HEADER_CONFIRMED' and s.get('hand_id')==m[1] and
            all(s.get(k)==r.get(k) for k in ('table_id','session_id','source_frame','source_span_index','source_pts'))
            for s in registry.states):raise ValueError('INDEPENDENT_NEXT_HEADER_STATE_UNKNOWN')
    if len(next_numbers)!=1:raise ValueError('INDEPENDENT_NEXT_HAND_NOT_UNIQUE')
    return low,boundary-1


class FreshConstantSources:
    """Independent full-hand inventory; never construct this from a proof case.

    Records come from the caller's current complete table registry. Retained
    cases must exactly match fresh native context, capture and SQLite rows.
    Source files and database are sealed before and after loading/consumption.
    """
    def __init__(self, source, records, registry, *, actor_policy=False, native_who=None):
        self.native_who = native_who
        self.source=Path(source);self.records=copy.deepcopy(records)
        self.contexts={};self.full_contexts={};self.captures=defaultdict(list)
        self.observations={};self.states={};self.bounds={};self.direct=defaultdict(list)
        self.database_name=stack_timeline_source_name(self.source)
        names=set(CONTEXT_FILES)|{self.database_name,'run_metadata.json','start_stack_live_receipts.jsonl'}
        if actor_policy:names.add('seat_forced_summary.json')
        self.hashes={name:file_sha256(self.source/name) for name in sorted(names)}
        windows=[]
        for record in self.records:
            hand=record['observed_hand'];table,session,number=key(hand)
            context=(int(str(table).removeprefix('T')),session,number)
            if context in self.bounds:raise ValueError('FRESH_DUPLICATE_HAND_CONTEXT')
            low,high=native_hand_bounds(hand,registry)
            self.bounds[context]=(low,high)
            for end in (low+64,high):
                windows.append(dict(table_id=context[0],session_id=session,hand_id=number,
                                    header_frame=low,capture_end_inclusive=end))
        loaded=load_contexts(self.source,windows,actor_policy=actor_policy)
        if any(self.hashes[n]!=v for n,v in loaded['source_hashes'].items()):
            raise ValueError('FRESH_CONTEXT_CHANGED_DURING_LOAD')
        for index,context in enumerate(self.bounds):
            self.contexts[context]=loaded['contexts'][2*index]
            self.full_contexts[context]=loaded['contexts'][2*index+1]
        with (self.source/'start_stack_capture.jsonl').open('rb') as stream:
            for line_number,line in enumerate(stream,1):
                if not line.strip():continue
                r=json.loads(line)
                if r.get('field')!='stack':continue
                for context,(low,high) in self.bounds.items():
                    if (r['table_id'],r['session_id'])==context[:2] and low<=r['source_frame']<=high:
                        self.captures[(*context,r['seat'])].append(dict(source_line=line_number,raw_record=r))
        with (self.source/'start_stack_live_receipts.jsonl').open('rb') as stream:
            for line in stream:
                if not line.strip():continue
                r=json.loads(line)
                if r.get('stage')!='STACK_OCR':continue
                for context,(low,high) in self.bounds.items():
                    if (r['table_id'],r['session_id'])==context[:2] and low<=r['source_frame']<=high:
                        self.direct[(*context,r['seat'])].append(r)
        metadata=json.loads((self.source/'run_metadata.json').read_text('utf-8'))
        self.video_sha256=metadata['source_video_sha256']
        if re.fullmatch('[0-9a-f]{64}',self.video_sha256) is None:
            raise ValueError('FRESH_VIDEO_IDENTITY_UNKNOWN')
        database=self.source/self.database_name
        self.database_sha256=self.hashes[self.database_name]
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row
            self.metadata=[dict(r) for r in db.execute('select key,value from metadata order by key')]
            if [json.loads(r['value']) for r in self.metadata if r['key']=='video_sha256']!=[self.video_sha256]:
                raise ValueError('FRESH_SQL_VIDEO_IDENTITY_CONFLICT')
            for context,(low,high) in self.bounds.items():
                # Include all seats, and states born before this hand if their
                # observed lifetime intersects it. No actor-case filtering.
                obs=[dict(r) for r in db.execute('select * from observations where table_id=? '
                    'and frame_index<=? and last_seen_frame>=? order by observation_id',(context[0],high,low))]
                for o in obs:self.observations.setdefault((*context,o['seat_id']),[]).append(o)
                for sid in sorted({o['state_id'] for o in obs}):
                    s=dict(db.execute('select * from visual_states where state_id=?',(sid,)).fetchone())
                    pixels=s.pop('raw_gray')
                    s.update(raw_gray_base64=base64.b64encode(pixels).decode(),raw_gray_sha256=sha(pixels))
                    if s['raw_gray_sha256']!=s['exact_hash']:raise ValueError('FRESH_SQL_PIXELS_CONFLICT')
                    self.states[sid]=s
        if not self.sources_unchanged():raise ValueError('FRESH_SOURCES_CHANGED_DURING_LOAD')

    def sources_unchanged(self):
        return all(file_sha256(self.source/name)==value for name,value in self.hashes.items()) and (
            self.native_who is None or self.native_who.sources_unchanged())

    def matches(self,case,metadata,record,registry):
        context=(case['table_id'],case['session_id'],case['hand_id']);seat=case['seat']
        if self.source.resolve()!=registry.source.resolve():return False
        if any(registry.hashes.get(n)!=v for n,v in self.hashes.items() if n in registry.hashes):return False
        current=[r for r in self.records if key(r['observed_hand'])==key(record['observed_hand'])]
        if current!=[record] or context not in self.bounds:return False
        obs=self.observations.get((*context,seat),[])
        states=[self.states[sid] for sid in sorted({o['state_id'] for o in obs})]
        return (case['source_context']==self.contexts[context] and
                case['full_hand_source_context']==self.full_contexts[context] and
                case['raw_stack_captures']==self.captures[(*context,seat)] and
                case['raw_direct_stack_readings']==self.direct[(*context,seat)] and
                case['raw_sql_observations']==obs and case['raw_sql_visual_states']==states and
                metadata==self.metadata and case['database_sha256']==self.database_sha256 and
                case['constant_interval']==list(self.bounds[context]))


def numeric_sources(case,metadata):
    # Exact source/OCR verification copied from the registered joint verifier;
    # no first-visible/fade inference occurs in this helper.
    def unknown(reason):raise ValueError(reason)
    c=case['full_hand_source_context']
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
    nicks=unique([r for r in c['native_capture_nick_rows'] if r['seat']==case['seat']],lambda r:r['source_frame'])
    nick_times={f:source_time(r['source_pts'],r['time_base']) for f,r in nicks.items()}
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
    return readings,bindings,observations,states


def complete_hand_interval(record,registry,actor,seat):
    hand=record['observed_hand'];context=key(hand)
    matches_records=[r for r in registry.records if key(r['observed_hand'])==context]
    if matches_records!=[record]:raise ValueError('CURRENT_FULL_HAND_REGISTRY_CONFLICT')
    if (record['lifecycle']['status']!='COMPLETE' or not hand['start_observed'] or
        not hand['end_observed'] or not hand['next_boundary_observed'] or hand['boundary_conflicts']):
        raise ValueError('COMPLETE_HAND_BOUNDARIES_UNPROVEN')
    native_rows=rows(hand);low,high=native_hand_bounds(hand,registry)
    fields=('stable_row_id','table_id','session_id','frame_id','row_index','raw_ocr_text',
            'row_pixel_sha256','source_pts','time_base','video_time','epoch')
    project=lambda rs:[[r.get(f) for f in fields] for r in rs]
    expected=[r for r in registry.native if (r['table_id'],r['session_id'])==context[:2] and low<=r['frame_id']<=high]
    if project(native_rows)!=project(expected):raise ValueError('COMPLETE_NATIVE_HAND_COVERAGE_CONFLICT')
    if any(r.get('status') in {'ERROR','UNRESOLVED'} or r.get('origin')=='UNRESOLVED_IDENTITY' or
        r.get('canonical_conflict') or r.get('canonical_duplicate_of') for r in native_rows):
        raise ValueError('UNRESOLVED_NATIVE_HAND_SOURCE')
    table=int(str(context[0]).removeprefix('T'));session=context[1];number=context[2]
    headers=[r for r in registry.headers if (r['table_id'],r['session_id'])==(table,session) and low<=r['source_frame']<=high]
    pending={};confirmed=set()
    for r in headers:
        k=(r['source_frame'],r['source_span_index'])
        if r['status']=='ADMITTED':
            if k in pending:raise ValueError('DUPLICATE_HAND_HEADER_ADMISSION')
            pending[k]=r
        elif r['status']=='HAND_HEADER_CONFIRMED':
            m=HEADER.fullmatch(response_text(r['ocr_response']));score=r['ocr_response']['score']
            if m is None or m[1]!=number or r['current_hand_id']!=number or score<.95:
                raise ValueError('FOREIGN_OR_UNQUALIFIED_HAND_HEADER')
            a=pending.get(k);join=('table_id','session_id','source_frame','source_span_index','source_pts','time_base','pixel_sha256')
            if a is None or any(a.get(f)!=r.get(f) for f in join):raise ValueError('HEADER_ADMISSION_JOIN_CONFLICT')
            pixels=cv2.imdecode(np.frombuffer(base64.b64decode(a['png_base64'],validate=True),np.uint8),0)
            lo,hi=a['row_bounds']
            if (pixels is None or type(lo) is not int or type(hi) is not int or not 0<=lo<hi or
                pixels.shape[0]!=hi-lo or sha(pixels.tobytes())!=a['pixel_sha256'] or a['gate']['candidate'] is not True):
                raise ValueError('HEADER_NATIVE_PIXELS_CONFLICT')
            registry.verify_header_source(r,a,pixels,require_physical=r['source_frame']==low)
            states=[s for s in registry.states if s.get('status')=='HAND_HEADER_CONFIRMED' and
                all(s.get(f)==r.get(f) for f in ('table_id','session_id','source_frame','source_span_index','source_pts')) and s.get('hand_id')==number]
            if not states:raise ValueError('HEADER_STATE_JOIN_UNKNOWN')
            confirmed.add(r['source_frame']);pending.pop(k)
        else:raise ValueError('UNRESOLVED_HAND_HEADER')
    if pending or low not in confirmed:raise ValueError('NATIVE_HAND_INTERVAL_UNKNOWN')
    for r in registry.states:
        if (r.get('table_id'),r.get('session_id'))!=(table,session) or not low<=r['source_frame']<=high:continue
        if r.get('status') not in {'HAND_HEADER_CONFIRMED','STALE_HEADER_IGNORED'} or r.get('hand_id')!=number:
            raise ValueError('HAND_STATE_INTERVAL_CONFLICT')
        admitted=[a for a in headers if a['status']=='ADMITTED' and
                  (a['source_frame'],a['source_span_index'])==(r['source_frame'],r['source_span_index'])]
        if len(admitted)!=1 or r['source_pts']!=admitted[0]['source_pts']:raise ValueError('HAND_STATE_SOURCE_CONFLICT')
    epochs={};empty_seats=set()
    for r in registry.summary['state_receipts']:
        if (r['table'],r['session'])!=(table,session) or not low<=r['frame']<=high:continue
        if r.get('kind')=='EMPTY':
            if r['seat'] in epochs or r['seat'] in empty_seats or not positive_other_empty(r,registry,context,seat):
                raise ValueError('SEAT_CONTINUITY_BARRIER')
            empty_seats.add(r['seat']);continue
        if (r.get('status')!='APPLIED' or r.get('evidence_status')!='PROVEN' or r.get('hand_id')!=number or
                r.get('kind') not in {'OCCUPIED','SIT_IN','SIT_OUT'} or r['seat'] in empty_seats):
            raise ValueError('SEAT_CONTINUITY_BARRIER')
        old=epochs.setdefault(r['seat'],r.get('epoch'))
        if old is None or old!=r.get('epoch'):raise ValueError('SEAT_EPOCH_CHANGED')
        if r['seat']==seat and r.get('actor') not in (None,actor):raise ValueError('SEAT_ACTOR_CHANGED')
    for r in registry.summary['unresolved_state_events']:
        if (r['table'],r['session'])==(table,session) and low<=r['frame']<=high:raise ValueError('UNRESOLVED_SEAT_STATE')
    for r in registry.inputs:
        if (r.get('table_id'),r.get('session_id'))!=(table,session) or r.get('kind')!='NICK' or not low<=r['source_frame']<=high:continue
        response=r.get('ocr_response',{})
        if response.get('status')!='OK':continue
        text=response_text(response)
        if response['score']<.95:continue
        label=literal_constraint(text)
        if label and ((r['seat']==seat and not matches(actor,label)) or (r['seat']!=seat and matches(actor,label))):
            raise ValueError('INDEPENDENT_HAND_NICKNAME_CONFLICT')
    return low,high


def positive_other_empty(event,registry,context,target_seat):
    """Replay an initial positive empty partition, never an actor replacement."""
    table,session,number=context;table=int(str(table).removeprefix('T'))
    if (event.get('kind')!='EMPTY' or event.get('status')!='APPLIED' or
            event.get('evidence_status')!='PROVEN' or event.get('seat')==target_seat or
            (event.get('table'),event.get('session'),event.get('hand_id'))!=(table,session,number)):
        return False
    deals=[d for d in registry.deals if (d.get('table_id'),d.get('session_id'),d.get('hand_id'))==(table,session,number)]
    if len(deals)!=1:return False
    deal=deals[0]
    if (deal.get('evidence_status')!='PROVEN' or deal['source_frame']!=event['frame'] or
            target_seat not in deal['occupied_seats'] or event['seat'] in deal['occupied_seats']):return False
    empty_refs=_deal_empty_refs(deal)
    if not event.get('source_ref') or empty_refs.get(event['seat'])!=event['source_ref']:return False
    witnesses=[r for r in registry.summary['state_receipts'] if
        (r.get('table'),r.get('session'),r.get('hand_id'),r.get('seat'),r.get('frame'))==
        (table,session,number,target_seat,deal['source_frame'])]
    return (len(witnesses)==1 and witnesses[0].get('status')=='APPLIED' and
            witnesses[0].get('evidence_status')=='PROVEN' and witnesses[0].get('kind')=='OCCUPIED' and
            witnesses[0].get('epoch') is not None and witnesses[0].get('source_ref')==deal['table_pixel_sha256'])


def direct_readings(case):
    """Replay primary and scaled native OCR without altering its raw response."""
    captures=unique([r['raw_record'] for r in case['raw_stack_captures']],lambda r:r['source_frame'])
    readings=unique(case['raw_direct_stack_readings'],lambda r:r['source_frame'])
    _,slots=proof_config()
    seat_index=list(slots[f"slot_{case['table_id']:02d}"]['seats']).index(case['seat'])
    accepted=[]
    for f,r in readings.items():
        if (any(r.get(k)!=case.get(k) for k in ('table_id','session_id','seat','hand_id')) or
                r.get('status')=='SOURCE_CONTEXT_UNKNOWN'):
            raise ValueError('DIRECT_STACK_SOURCE_CONTEXT_CONFLICT')
        response=r['ocr_response']
        if response.get('item_id')!=f"{case['table_id']}:{f}:{seat_index}":
            raise ValueError('DIRECT_OCR_CALLBACK_ITEM_CONFLICT')
        if any(r.get(k)!=response.get(v) for k,v in (('raw_text','raw_text'),('confidence','score'),('ocr_status','status'))):
            raise ValueError('DIRECT_STACK_RESPONSE_CONFLICT')
        value=effective_numeric(r,.95,allow_scaled=True)
        # The complete fresh inventory retains rejected readings. They cannot
        # supply a number (or a numeric conflict) under the frozen OCR contract.
        # Any accepted value still requires the independent native RGB join.
        if value is None:continue
        cap=captures.get(f)
        if cap is None:raise ValueError('DIRECT_STACK_NATIVE_CAPTURE_MISSING')
        if (any(r.get(k)!=cap.get(k) for k in ('table_id','session_id','seat','source_frame','source_pts','time_base','roi')) or
            r.get('hand_id')!=case['hand_id'] or r.get('status')=='SOURCE_CONTEXT_UNKNOWN'):
            raise ValueError('DIRECT_STACK_SOURCE_CONTEXT_CONFLICT')
        pixels=cv2.imdecode(np.frombuffer(base64.b64decode(r['png_base64'],validate=True),np.uint8),0)
        rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(cap['png_base64'],validate=True),np.uint8),1),cv2.COLOR_BGR2RGB)
        if (sha(rgb.tobytes())!=cap['pixel_sha256'] or pixels is None or
                sha(pixels.tobytes())!=r['pixel_sha256'] or not np.array_equal(pixels,png_compatible_gray(rgb))):
            raise ValueError('DIRECT_STACK_NATIVE_PIXELS_CONFLICT')
        accepted.append(dict(frame=f,value=int(value),pixel_sha256=r['pixel_sha256']))
    return accepted


def independent_ledger(hand):
    native=copy.deepcopy(rows(hand))
    for i,row in enumerate(native,1):row['source_index']=i
    assemblies,exhausted,_=candidate_assemblies(native)
    meanings=lambda e:replace(e,semantics='UNKNOWN') if e.kind=='CALL' and e.semantics=='DELTA' else e
    # No initial-stack seed, including the proposed value, is an assumption.
    result=reconstruct_hand([meanings(e) for e in parse_rows(native)],table=hand['table_id'],hand_id=hand['hand_id'],
        start_stacks={},max_nodes=20000,strict_source_uniqueness=True,
        assemblies=[[meanings(e) for e in assembly] for assembly in assemblies],assembly_exhausted=exhausted)
    if (result['solution_count']!=1 or result['exhausted'] or result['dropped_events'] or result['inferred_events'] or
        result.get('chosen_identity_hypothesis',{}).get('aliases') or result.get('pot_delta')!=0 or result.get('payout_eligible') is not True):
        raise ValueError('INDEPENDENT_COMPLETE_LEDGER_UNPROVEN')
    validation=validate_reconstruction(hand,result)
    if validation.get('accepted') is not True:
        raise ValueError('INDEPENDENT_LEDGER_SOURCE_REVALIDATION_UNPROVEN:'+str(validation.get('reason')))
    # The registered verifier has reproduced each literal physical source group,
    # preserving exactly every row and its order. Only that separately validated
    # view can establish that a numeric/Blind fragment is part of an action.
    # Raw UNKNOWN messages remain untouched in the original observed hand.
    for message in validation['candidate_hand']['messages']:
        if message['action']=='UNKNOWN_EVENT':raise ValueError('UNPARSED_HAND_SOURCE')
        service=describe_service(message['raw_text'])
        if service and service['kind'] in {'JOIN','LEAVE','FINISHED','REBUY'}:
            raise ValueError('HAND_ROSTER_OR_BANKROLL_TRANSFER')
        if message['action']=='TABLE_INFO' and service is None:raise ValueError('UNCLASSIFIED_HAND_SERVICE')
    return result


def verify_numeric_triple_frames(frames,low,high):
    if (not frames or frames!=list(range(frames[0],frames[0]+3)) or
            frames[0]<=low+2 or frames[-1]>high):
        raise ValueError('NUMERIC_TRIPLE_CONTEXT_CONFLICT')


def verify_numeric_agreement(case,value,readings,bindings,observations,states,direct,overlay_end,frames,high):
    """Compare every independently accepted current reading after the overlay.

    Inputs must first pass their source verifiers and fresh inventory join.
    This pure final predicate is shared by native proof and mutation tests.
    """
    for f,r in readings.items():
        if overlay_end<f<=high and r['ocr_status']=='OK':
            if int(r['raw_text'].replace(' ','').replace('\u00a0',''))!=value:
                raise ValueError('INDEPENDENT_NATIVE_NUMERIC_CONFLICT')
    checked=[]
    for o in observations.values():
        state=states[o['state_id']]
        if ((o['table_id'],o['session_id'],o['seat_id'])!=(case['table_id'],case['session_id'],case['seat']) or
            o['frame_index']!=state['frame_index'] or state['source_state_id']!=o['state_id']):
            raise ValueError('HAND_NUMERIC_SOURCE_CONTEXT_CONFLICT')
        score=state.get('ocr_confidence')
        accepted=(state['status']=='CONFIRMED_NUMERIC' and state['digit_gate_status']=='DIGITS_VISIBLE' and state['ocr_called']==1 and
            type(score) in (int,float) and math.isfinite(score) and .95<=score<=1 and re.fullmatch(NUMBER,state['ocr_raw'].strip()) and
            state['ocr_raw'].strip().replace(' ','').replace('\u00a0','')==state['ocr_numeric'])
        if accepted and overlay_end<o['frame_index']<=high:
            checked.append(o['observation_id'])
            if int(state['ocr_numeric'])!=value:raise ValueError('INDEPENDENT_ACCEPTED_NUMERIC_CONFLICT')
    if not checked:raise ValueError('POST_FORCED_NATIVE_NUMERIC_BIRTH_UNKNOWN')
    if any(not overlay_end<states[b['state_id']]['frame_index']<=frames[0] for b in bindings):
        raise ValueError('SELECTED_NUMERIC_NATIVE_BIRTH_NOT_POST_FORCED')
    direct_checked=[r for r in direct if overlay_end<r['frame']<=high]
    if any(r['value']!=value for r in direct_checked):raise ValueError('INDEPENDENT_DIRECT_NUMERIC_CONFLICT')
    return checked,direct_checked


def _prove_body(case,metadata,record,registry,inventory,check_seals):
    """Reproduce all evidence guards; authority requires the caller's seals."""
    def unknown(reason):return dict(status='UNKNOWN',rule=RULE,reason=reason,actor=case.get('actor'))
    try:
        if inventory is None:return unknown('FRESH_INDEPENDENT_INVENTORY_REQUIRED')
        if not inventory.matches(case,metadata,record,registry):return unknown('FRESH_NATIVE_INVENTORY_CONFLICT')
        seal_error=check_seals('before')
        if seal_error:return unknown(seal_error)
        identity=prove_actor_point(case['source_context'],case['actor'],native_who=getattr(inventory,'native_who',None))
        if identity['status']!='PROVEN':return unknown('ACTOR_POINT_'+identity['reason'])
        if any(case.get(k)!=identity.get(k) for k in ('actor','seat','table_id','session_id','hand_id')):return unknown('ACTOR_CONTEXT_CONFLICT')
        hand=record['observed_hand']
        if not point_matches_hand(identity,hand):return unknown('CURRENT_HAND_ACTOR_SOURCE_CONFLICT')
        if (int(str(hand['table_id']).removeprefix('T')),hand['session_id'],hand['hand_id'])!=(case['table_id'],case['session_id'],case['hand_id']):
            return unknown('HAND_CONTEXT_CONFLICT')
        low,high=complete_hand_interval(record,registry,case['actor'],case['seat'])
        c=case['source_context']
        if c['header_frame']!=low:return unknown('FORCED_PREFIX_HEADER_CONFLICT')
        posts=own_posts(c,case['actor']);forced=sum(p['amount'] for p in posts)
        ledger=independent_ledger(hand);player=ledger['players'][case['actor']]
        actor_events=[e for e in ledger['clean_events'] if e.get('player')==case['actor']]
        if (not actor_events or [e['kind'] for e in actor_events].count('FOLD')!=1 or actor_events[-1]['kind']!='FOLD' or
            any(e['kind'] not in {'ANTE','SB','BB','FOLD'} or e.get('all_in') for e in actor_events)):
            return unknown('ACTOR_NOT_CONSTANT_POST_FORCED')
        if (player['status']!='FOLDED' or player['returned'] or player['payout'] or
            player['gross_commit']!=forced or player['net_commit']!=forced):return unknown('ACTOR_LEDGER_FORCED_SET_CONFLICT')
        readings,bindings,observations,states=numeric_sources(case,metadata)
        direct=direct_readings(case)
        from parser_core.chat_nick_actor import (reuses_cross_identity,
            first_consecutive_frames, cross_identity_frame_binding)
        identity_binding = None
        if reuses_cross_identity(identity):
            frames = first_consecutive_frames([f for f, r in readings.items()
                if low <= f <= high and r['ocr_status'] == 'OK' and
                type(r.get('confidence')) in (int, float) and math.isfinite(r['confidence']) and
                .95 <= r['confidence'] <= 1 and len([b for b in bindings if b['frame'] == f]) == 1],
                'SOURCE_BOUND_NUMERIC_TRIPLE_UNKNOWN')
            identity_binding = cross_identity_frame_binding(case['full_hand_source_context'],
                identity, frames, start_frame=low, end_frame=high, native_who=getattr(inventory,'native_who',None))
        else:
            frames=[r['source_frame'] for r in numeric_nickname_sources(case['source_context'],identity)]
        from parser_core.chat_nick_actor import assert_native_identity_interval
        assert_native_identity_interval(case['full_hand_source_context'],identity,frames,
            start_frame=low,end_frame=high,native_who=getattr(inventory,'native_who',None))
        verify_numeric_triple_frames(frames,low,high)
        # A constant balance does not require proof of first visibility. The
        # earliest retained current-hand crop must instead start a positive
        # consecutive overlay triple; no missing earlier pixels are inferred.
        overlay_start=min(readings);overlay_end=overlay_start+2
        if overlay_start<low or overlay_end>=frames[0]:return unknown('CURRENT_OVERLAY_ANCHOR_UNKNOWN')
        if any(f not in readings for f in range(overlay_start,overlay_end+1)):
            return unknown('CURRENT_OVERLAY_TRIPLE_SOURCE_MISSING')
        verifier=VisibilitySource();overlays=ForcedOverlayPatterns();overlay_proofs=[]
        for f in range(overlay_start,overlay_end+1):
            verified=verifier.verify(readings[f])
            if verified is None:return unknown('FORCED_OVERLAY_SOURCE_INVALID')
            label=overlays.identify(verified['gray'][2:-2,6:-6].astype(np.float64))
            if label is None:return unknown('POSITIVE_CURRENT_FORCED_OVERLAY_UNKNOWN')
            overlay_proofs.append(dict(frame=f,identity=label,pixel_sha256=verified['timeline_pixel_sha256']))
        values=[];numeric_bindings=[]
        for f in frames:
            r=readings[f];verified=verifier.verify(r);selected=[b for b in bindings if b['frame']==f]
            if verified is None or len(selected)!=1 or r['ocr_status']!='OK' or r['confidence']<.95:return unknown('SOURCE_BOUND_NUMERIC_TRIPLE_UNKNOWN')
            if overlays.identify(verified['gray'][2:-2,6:-6].astype(np.float64)) is not None:return unknown('NUMERIC_SOURCE_STILL_FORCED_OVERLAY')
            values.append(int(r['raw_text'].replace(' ','').replace('\u00a0','')));numeric_bindings+=selected
        if len(set(values))!=1 or values[0]<=0:return unknown('NUMERIC_TRIPLE_CONFLICT')
        value=values[0]
        conflict_checked,direct_checked=verify_numeric_agreement(case,value,readings,numeric_bindings,
            observations,states,direct,overlay_end,frames,high)
        seal_error=check_seals('after')
        if seal_error:return unknown(seal_error)
        return dict(status='PROVEN',rule=outer_rule(RULE,identity),reason=None,actor=case['actor'],seat=case['seat'],table_id=case['table_id'],
            session_id=case['session_id'],hand_id=case['hand_id'],displayed_stack=value,forced_amount_added_back=forced,
            calculated_start_stack=value+forced,identity_proof=identity,own_forced_posts=posts,overlay_proofs=overlay_proofs,
            **({'identity_frame_binding': identity_binding} if identity_binding is not None else {}),
            forced_prefix_ocr_policy=copy.deepcopy(FORCED_PREFIX_OCR_POLICY),
            numeric_frames=frames,numeric_sources=[readings[f] for f in frames],numeric_bindings=numeric_bindings,
            independent_numeric_observations_checked=conflict_checked,actor_ledger=player,
            independent_direct_readings_checked=direct_checked,native_interval=[low,high],
            pre_overlay_missing_frames=list(range(low,overlay_start)),
            retained_capture_count=len(readings),missing_capture_count=high-low+1-len(readings),
            ledger_seed_start_stacks={},first_visible_authority=False,scope='SOURCE_REPLAYED_CONSTANT_BALANCE_NO_OCCUPANCY_EPOCH')
    except (KeyError,TypeError,ValueError,IndexError,AttributeError,OverflowError,ZeroDivisionError,OSError,cv2.error) as error:
        return unknown(str(error) if isinstance(error,ValueError) else type(error).__name__+':'+str(error))


def prove(case,metadata,record,registry,inventory=None):
    """Standalone proof keeps the original per-case before/after source seals."""
    def check_seals(phase):
        checks=[(inventory,'CURRENT_NUMERIC_SOURCE_SEAL_CHANGED'),(registry,'CURRENT_SOURCE_SEAL_CHANGED')]
        if phase=='after':checks.reverse()
        for source,reason in checks:
            if not source.sources_unchanged():return reason
        return None
    return _prove_body(case,metadata,record,registry,inventory,check_seals)


def frozen_runtime_files():
    """Keep the constructor's expected code/config/model hashes for all replay."""
    manifest_path=ROOT/'production_manifest.json'
    raw=manifest_path.read_bytes();manifest=json.loads(raw)
    expected={str((ROOT/name).resolve()):value for name,value in manifest['hashes'].items()}
    expected[str(manifest_path.resolve())]=sha(raw)
    return tuple(sorted(expected.items()))


class SourceSealBatch:
    """Full content hashes bracket unpublished proofs from loaded inventories.

    No stat cache or retained proof supplies source authority. The union keeps
    every independent expected digest, rejects disagreement, and covers the
    frozen runtime files read by pixel, card, and ledger verification.
    """
    def __init__(self,registry,inventory,runtime_files):
        if registry.source.resolve()!=inventory.source.resolve():
            raise ValueError('BATCH_SOURCE_DIRECTORY_CONFLICT')
        self.registry=registry;self.inventory=inventory;self.open=False
        expected={}
        def add(path,value):
            path=str(Path(path).resolve())
            if path in expected and expected[path]!=value:
                raise ValueError('BATCH_EXPECTED_SOURCE_HASH_CONFLICT')
            expected[path]=value
        for source in (registry,inventory):
            for name,value in source.hashes.items():add(source.source/name,value)
        native = getattr(inventory, 'native_who', None)
        if native is not None:
            if native.source.resolve() != inventory.source.resolve():
                raise ValueError('BATCH_NATIVE_WHO_SOURCE_DIRECTORY_CONFLICT')
            for name,value in native.hashes.items():add(native.source/name,value)
        add(registry.config_path,registry.config_sha256)
        for path,value in runtime_files:add(path,value)
        self.expected=tuple(sorted(expected.items()))

    def _unchanged(self):
        try:
            return all(file_sha256(path)==value for path,value in self.expected)
        except OSError:
            return False

    def begin(self):
        if self.open:raise ValueError('BATCH_ALREADY_OPEN')
        if not self._unchanged():return False
        self.open=True
        return True

    def check_phase(self,phase):
        return None if self.open else 'BATCH_SOURCE_SEAL_NOT_OPEN'

    def finish(self):
        if not self.open:return False
        self.open=False
        return self._unchanged()


def proof_id(proof):
    return sha(json.dumps(proof,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8'))


def join_receipt(proof):
    return dict(rule=proof['rule'],proof_id=proof_id(proof),actor=proof['actor'],seat=proof['seat'],
        table_id=proof['table_id'],session_id=proof['session_id'],hand_id=proof['hand_id'],
        calculated_start_stack=proof['calculated_start_stack'],displayed_stack=proof['displayed_stack'],
        forced_amount_added_back=proof['forced_amount_added_back'],numeric_frames=proof['numeric_frames'],
        forced_prefix_ocr_policy=copy.deepcopy(proof['forced_prefix_ocr_policy']),
        source_observation_ids=sorted({r['observation_id'] for r in proof['numeric_bindings']}),
        source_state_ids=sorted({r['state_id'] for r in proof['numeric_bindings']}),
        scope='CONSTANT_POST_FORCED_BALANCE_NO_OCCUPANCY_EPOCH')


class ConstantStackBalanceProvider:
    """Saved FULL only. Reproduce every consumed proof from a fresh inventory.

    The ordinary first-visible audit selects possible missing actors as a work
    bound, never as authority. Every boundary, identity, ledger, numeric source
    and rejected observation is independently checked by this contract.
    """
    def __init__(self,source,records,joint_provider,*,sitout_provider=None,actor_policy=False,native_who=None):
        from .adjacent_stack_carry import SavedCarrySources
        self.registry=SavedCarrySources(source,records,[])
        self.cases={};self.proofs={};self.audit=[];self.inventory=None
        self.runtime_files=frozen_runtime_files()
        by_key={key(r['observed_hand']):r for r in self.registry.records}
        if len(by_key)!=len(self.registry.records):raise ValueError('CONSTANT_DUPLICATE_HAND_CONTEXT')
        candidates=defaultdict(set)
        for row in joint_provider.rows:
            if row.get('point_status')!='PROVEN' or row.get('status')!='UNKNOWN':continue
            context=(f"T{row['table_id']:02d}",row['session_id'],row['hand_id'])
            if sitout_provider is not None and sitout_provider.excludes_stack_work(context,row['actor']):continue
            record=by_key.get(context)
            if not record or record['lifecycle']['status']!='COMPLETE':continue
            own=[m for m in record['observed_hand']['messages'] if m.get('player')==row['actor']]
            if (not any(m['action']=='FOLD' for m in own) or
                    any(m['action'] not in {'ANTE','SMALL_BLIND','BIG_BLIND','FOLD'} for m in own)):continue
            candidates[context].add(row['actor'])
        selected=[]
        for context,actors in list(candidates.items()):
            try:native_hand_bounds(by_key[context]['observed_hand'],self.registry)
            except (KeyError,TypeError,ValueError,IndexError,AttributeError,cv2.error) as error:
                self.audit.extend(dict(hand_key=list(context),actor=actor,status='UNKNOWN',reason=str(error)) for actor in sorted(actors))
                del candidates[context];continue
            selected.append(by_key[context])
        if not selected:return
        self.inventory=FreshConstantSources(source,selected,self.registry,actor_policy=actor_policy,native_who=native_who)
        batch=SourceSealBatch(self.registry,self.inventory,self.runtime_files)
        if not batch.begin():raise ValueError('CONSTANT_BATCH_PRE_SOURCE_SEAL_CHANGED')
        pending_cases={};pending_proofs={};pending_audit=[]
        for context,actors in candidates.items():
            native=(int(context[0].removeprefix('T')),context[1],context[2]);record=by_key[context]
            for actor in sorted(actors):
                point=prove_actor_point(self.inventory.contexts[native],actor,native_who=self.inventory.native_who)
                if point['status']!='PROVEN':
                    pending_audit.append(dict(hand_key=list(context),actor=actor,status='UNKNOWN',reason='ACTOR_POINT_'+point['reason']));continue
                seat=point['seat'];obs=self.inventory.observations.get((*native,seat),[])
                case=copy.deepcopy(dict(actor=actor,seat=seat,table_id=native[0],session_id=native[1],hand_id=native[2],
                    source_context=self.inventory.contexts[native],full_hand_source_context=self.inventory.full_contexts[native],
                    raw_stack_captures=self.inventory.captures[(*native,seat)],
                    raw_direct_stack_readings=self.inventory.direct[(*native,seat)],
                    raw_sql_observations=obs,raw_sql_visual_states=[self.inventory.states[s] for s in sorted({o['state_id'] for o in obs})],
                    database_sha256=self.inventory.database_sha256,constant_interval=list(self.inventory.bounds[native])))
                result=self._prove(case,record,batch=batch)
                pending_audit.append(dict(hand_key=list(context),actor=actor,seat=seat,status=result['status'],
                    reason=result['reason'],calculated_start_stack=result.get('calculated_start_stack')))
                if result['status']=='PROVEN':
                    pending_cases[(context,actor)]=(case,record)
                    pending_proofs[(context,actor)]=result
        if not batch.finish():raise ValueError('CONSTANT_BATCH_POST_SOURCE_SEAL_CHANGED')
        self.cases=pending_cases;self.proofs=pending_proofs;self.audit.extend(pending_audit)

    def _prove(self,case,record,*,batch=None):
        if batch is None:result=prove(case,self.inventory.metadata,record,self.registry,self.inventory)
        elif batch.registry is self.registry and batch.inventory is self.inventory:
            result=_prove_body(case,self.inventory.metadata,record,self.registry,self.inventory,batch.check_phase)
        else:raise ValueError('CONSTANT_BATCH_INVENTORY_CONFLICT')
        if result['status']=='PROVEN':
            result.update(source_files_sha256={**self.registry.hashes,**self.inventory.hashes},
                          database_sha256=self.inventory.database_sha256)
        return result

    def for_hand(self,hand):
        context=key(hand);result=[]
        cases=[(actor,case,record) for (saved_context,actor),(case,record) in self.cases.items()
               if saved_context==context and record['observed_hand']==hand]
        if not cases:return result
        batch=SourceSealBatch(self.registry,self.inventory,self.runtime_files)
        if not batch.begin():return []
        for actor,case,record in cases:
            proof=self._prove(case,record,batch=batch)
            if proof['status']=='PROVEN' and proof==self.proofs[(context,actor)]:result.append(proof)
        if not batch.finish():return []
        return result

    def write_audit(self,output):
        output=Path(output)
        with (output/'constant_stack_balance_proofs.jsonl').open('w',encoding='utf-8') as stream:
            for proof in self.proofs.values():stream.write(json.dumps(proof,ensure_ascii=False)+'\n')
        (output/'constant_stack_balance_audit.json').write_text(json.dumps(dict(
            scope='SAVED_COMPLETE_UNIQUE_LEDGER_CONSTANT_AFTER_FORCED; FIRST_VISIBLE_UNCHANGED',
            counts=dict(Counter(r['status'] for r in self.audit)),rows=self.audit,
            source_hashes={**self.registry.hashes,**(self.inventory.hashes if self.inventory else {})}),
            ensure_ascii=False,indent=2),'utf-8')
