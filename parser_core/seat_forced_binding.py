"""Source-bound vision facts only; arithmetic belongs to poker_math_engine."""
import base64
import copy
import hashlib
import json
import math
from .nickname_policy import NICK_CONFIDENCE_MIN, FIRST_NICK_POLICIES
import re
import threading
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from parser_core.chat_rows import physical_spans, padded_row_span
from parser_core.dealing_start_stack import HEADER, NUMERIC, MONEY_ACTION
from parser_core.parallel_ocr import ParallelRowOCR, row_key
from parser_core.ram_first_stage import RamRow, png_compatible_gray
from parser_core.seat_epoch_state import SeatEpochMachine
from parser_core.tiny_chat_client import TinyChatClient
from parser_core.poker_hand_assembly import split_small_blind, parse_text

# Strict current-project English POST grammar, including an adjacent amount wrap.
POST = re.compile(r'^Dealer:\s*(.+?)\s+posts\s+(Ante|Small Blind|Big Blind)(?:\s+(\d+(?:[ \u00a0]\d{3})*))?(?:\s+(and is All-in))?\s*$', re.I)
DEAL = re.compile(r'^Dealer:\s*Dealing cards\s*$', re.I)
STATE = re.compile(r'^Dealer:\s*(?:Player\s+)?(.+?)\s+(sits? in|sits? out|has left the table|has joined the table)\s*$', re.I)
LABEL_OVERLAYS = {'ANTE','SB','BB','FOLD','CHECK','CALL','BET','RAISE','ALL-IN','ALL IN','SITOUT','SIT OUT','OPEN SEAT'}
SPLIT_SMALL_BLIND_TAIL_CONFIDENCE = .94


def adjacent_forced_rows(first, second):
    """A continuation belongs to the same adjacent physical source/context."""
    return (first.get('source_frame') is not None and
            first['source_frame']==second.get('source_frame') and
            isinstance(first.get('index'),int) and second.get('index')==first['index']+1 and
            all(first.get(k)==second.get(k) for k in ('table_id','session_id','hand_id','epoch')))


def parse_forced_snapshot(rows, hand_id, occupied_count, confidence=.95):
    """Return one complete physical pre-deal snapshot, never sum duplicate posts."""
    def valid_score(value):
        return type(value) in (int,float) and math.isfinite(value) and 0<=value<=1
    if not valid_score(confidence) or confidence==0:
        return None,'FORCED_CONFIDENCE_POLICY_INVALID'
    if any(not valid_score(r.get('score')) for r in rows):
        return None,'FORCED_ROW_CONFIDENCE_VALUE_INVALID'
    small_tail_confidence=min(confidence,SPLIT_SMALL_BLIND_TAIL_CONFIDENCE)
    ordered = sorted(rows, key=lambda r:r['index'])
    if len({r.get('source_frame') for r in ordered})>1:
        return None,'FORCED_SOURCE_FRAME_CONFLICT'
    headers = [i for i,r in enumerate(ordered) if r['score'] >= confidence
               and (m:=HEADER.fullmatch(r['text'].strip())) and m.group(1)==hand_id]
    if not headers:
        return None,'PHYSICAL_HAND_HEADER_UNKNOWN'
    lo = min(headers)
    ends = [i for i in range(lo+1,len(ordered)) if ordered[i]['score'] >= confidence and DEAL.fullmatch(ordered[i]['text'].strip())]
    if len(ends) != 1:
        return None,'PHYSICAL_DEAL_ROW_AMBIGUOUS'
    hi = ends[0]
    posts = {}
    state_events = []
    i = lo+1
    while i < hi:
        r = ordered[i]
        text = r['text'].strip()
        header = HEADER.fullmatch(text)
        if header and header.group(1)!=hand_id:
            return None,'PHYSICAL_HAND_HEADER_CONFLICT'
        s = STATE.fullmatch(text)
        if s:
            if r['score'] < confidence:
                return None,'SEAT_STATE_ROW_CONFIDENCE_UNKNOWN'
            state_events.append(dict(actor=s.group(1),kind=s.group(2).lower(),index=r['index']))
        refs = [r['source_ref']]
        source_rows = [dict(r)]
        if i+1 < hi and split_small_blind(r, ordered[i+1]):
            if r['score'] < confidence or ordered[i+1]['score'] < small_tail_confidence:
                return None,'FORCED_ROW_CONFIDENCE_UNKNOWN'
            i += 1
            text += ' ' + ordered[i]['text'].strip()
            refs.append(ordered[i]['source_ref'])
            source_rows.append(dict(ordered[i]))
        elif (i+1 < hi and adjacent_forced_rows(r,ordered[i+1]) and
              re.fullmatch(r'Dealer:\s*.+?\s+posts Big',text,re.I) and
              re.fullmatch(r'Blind \d+(?:[ \u00a0]\d{3})*',ordered[i+1]['text'].strip(),re.I)):
            # Both Big/Blind source rows use the requested prefix threshold.
            if r['score'] < confidence or ordered[i+1]['score'] < confidence:
                return None,'FORCED_ROW_CONFIDENCE_UNKNOWN'
            i += 1
            text += ' ' + ordered[i]['text'].strip()
            refs.append(ordered[i]['source_ref'])
            source_rows.append(dict(ordered[i]))
        m = POST.fullmatch(text)
        if m:
            actor,role,amount,all_in_suffix = m.groups()
            if r['score'] < confidence:
                return None,'FORCED_ROW_CONFIDENCE_UNKNOWN'
            if all_in_suffix and amount is None:
                return None,'FORCED_AMOUNT_WRAP_UNKNOWN'
            if amount is None:
                if i+1 >= hi or not adjacent_forced_rows(r,ordered[i+1]) or ordered[i+1]['score'] < confidence or not NUMERIC.fullmatch(ordered[i+1]['text'].strip()):
                    return None,'FORCED_AMOUNT_WRAP_UNKNOWN'
                i += 1
                amount = ordered[i]['text'].strip()
                refs.append(ordered[i]['source_ref'])
                source_rows.append(dict(ordered[i]))
            all_in=bool(all_in_suffix)
            if (not all_in and i+1 < hi and
                    re.fullmatch(r'and is All-in',ordered[i+1]['text'].strip(),re.I) and
                    adjacent_forced_rows(source_rows[-1],ordered[i+1])):
                if ordered[i+1]['score'] < confidence:
                    return None,'FORCED_ROW_CONFIDENCE_UNKNOWN'
                i += 1
                all_in=True
                refs.append(ordered[i]['source_ref'])
                source_rows.append(dict(ordered[i]))
            role = {'ante':'ANTE','small blind':'SB','big blind':'BB'}[role.lower()]
            value = int(amount.replace(' ','').replace('\u00a0',''))
            if value<=0:
                return None,'FORCED_AMOUNT_NOT_POSITIVE'
            k = (actor,role)
            if k in posts:
                return None,'FORCED_DUPLICATE_ROW_RISK'
            posts[k] = dict(actor=actor,role=role,amount=value,source_refs=refs,
                            source_rows=source_rows,source_index=r['index'],phase='BEFORE_DEAL')
            if all_in:posts[k]['all_in']=True
        elif 'posts' in text.lower():
            return None,'FORCED_GRAMMAR_UNKNOWN'
        elif parse_text(text)['action'] in {'FOLD','CHECK','CALL','BET','RAISE','FLOP','TURN','RIVER','WIN','UNCALLED_RETURN','SHOWDOWN','SHOW','HAND_END'}:
            return None,'VOLUNTARY_OR_TERMINAL_IN_FORCED_PREFIX'
        elif (not s and not header and parse_text(text)['action'] not in {'TABLE_INFO','BLIND_LEVEL'}
              and not re.fullmatch(r'Dealer:\s*Starting a new hand[.!]?',text,re.I)):
            return None,'INTERVENING_FORCED_PREFIX_ROW_UNKNOWN'
        elif r['score'] < confidence:
            return None,'INTERVENING_FORCED_PREFIX_CONFIDENCE_UNKNOWN'
        i += 1
    antes = {a:p for (a,k),p in posts.items() if k=='ANTE'}
    sb = [p for (a,k),p in posts.items() if k=='SB']
    bb = [p for (a,k),p in posts.items() if k=='BB']
    if len(antes)!=occupied_count or len(sb)!=1 or len(bb)!=1:
        return None,'FORCED_SNAPSHOT_COVERAGE_UNKNOWN'
    if sb[0]['actor']==bb[0]['actor'] or sb[0]['actor'] not in antes or bb[0]['actor'] not in antes:
        return None,'FORCED_ROLE_CONFLICT'
    return dict(posts=list(posts.values()),actors=list(antes),state_events=state_events,
                standard_ante=next(iter({p['amount'] for p in antes.values()})) if len({p['amount'] for p in antes.values()})==1 else None,
                standard_ante_proven=len({p['amount'] for p in antes.values()})==1,
                forced_snapshot_complete=True,
                header_index=ordered[lo]['index'],deal_index=ordered[hi]['index'],
                money_rows_after_deal=[dict(index=r['index'],text=r['text'],source_frame=r['source_frame'],source_ref=r['source_ref']) for r in ordered[hi+1:] if MONEY_ACTION.match(r['text'].strip())]),None


class PendingSourceJobs(dict):
    """Pending source indexes; never scan all retained jobs for every label."""
    missing=object()

    def __init__(self):
        super().__init__();self.nick_frames=defaultdict(Counter)
        self.earliest_pending={};self.pending_prefix=Counter()

    def _track(self,value,delta):
        key,meta=value;kind=meta.get('kind')
        if kind=='NICK':
            route=(*key,meta.get('seat'));frame=meta['source_frame'];frames=self.nick_frames[route]
            frames[frame]+=delta
            if not frames[frame]:del frames[frame]
            if frames:self.earliest_pending[route]=min(frames)
            else:self.earliest_pending.pop(route,None);self.nick_frames.pop(route,None)
        elif kind in {'CHAT','CHAT_CANDIDATE'}:
            self.pending_prefix[key]+=delta
            if not self.pending_prefix[key]:del self.pending_prefix[key]

    def __setitem__(self,key,value):
        if key in self:self.pop(key)
        super().__setitem__(key,value);self._track(value,1)

    def pop(self,key,default=missing):
        if key not in self:
            if default is self.missing:raise KeyError(key)
            return default
        value=super().pop(key);self._track(value,-1);return value

    def __delitem__(self,key):self.pop(key)

    def clear(self):
        super().clear();self.nick_frames.clear();self.earliest_pending.clear();self.pending_prefix.clear()

    def update(self,other=(),**kwargs):
        for key,value in dict(other,**kwargs).items():self[key]=value


class SeatForcedBinding:
    def __init__(self, root, output, sidecar, stack_live, config, *, independent_occupancy=False,
                 roster_prefix_binding=False,empty_card_contract=False):
        self.root=Path(root);self.output=Path(output);self.sidecar=sidecar;self.stack_live=stack_live
        self.lock=threading.RLock();self.machine=SeatEpochMachine();self.counts=Counter()
        self.seen_deals=set();self.jobs=PendingSourceJobs();self.snapshots=defaultdict(dict);self.snapshot_sizes={}
        self.seen_prefix_candidates=set();self.prefix_candidates=Counter()
        self.candidate_sizes={};self.candidate_rows=defaultdict(dict)
        self.pending_snapshots={};self.prefix_reuse={};self.observed_frames={}
        self.nick_candidates=defaultdict(dict)
        self.forced={};self.blockers={};self.nick_schedule={};self.snapshot_occupied={}
        self.nick_runs={};self.nick_labels=defaultdict(list);self.state_events=[];self.closed=False
        self.roster_prefix_binding=roster_prefix_binding
        self.roster_nick_rows=defaultdict(list);self.roster_forced_rows=defaultdict(dict)
        from parser_core.seat_occupancy import SeatOccupancyEvidence
        self.occupancy = SeatOccupancyEvidence(sidecar.flags['stack_confidence'],
            reject_empty_fragments=empty_card_contract,
            nickname_confidence=NICK_CONFIDENCE_MIN) if independent_occupancy else None
        self.occupancy_expired=set()
        self.log=(self.output/'seat_forced_inputs.jsonl').open('w',encoding='utf-8')
        def factory(worker):
            return TinyChatClient(Path(config['python']),Path(config['model_dir']),config['model_name'],
                config['weights_sha256'],config['device'],self.output/f'tiny_seat_forced_{worker}.stderr.log')
        self.ocr=ParallelRowOCR(factory,workers=1,queue_size=32,batch_size=8,result_observer=self.completed)

    def write(self,r):self.log.write(json.dumps(r,ensure_ascii=False)+'\n')

    def _hand_end_frame(self,key):
        own=[t['source_frame'] for t in self.sidecar.triggers if
             (t['table_id'],t['session_id'],t['hand_id'])==key]
        start=min(own)
        later=[t['source_frame'] for t in self.sidecar.triggers if
            (t['table_id'],t['session_id'])==key[:2] and t['hand_id']!=key[2]
            and t['source_frame']>start]
        observed=getattr(self,'observed_frames',{}).get(key)
        if observed is None:
            observed=max([self.sidecar.windows[key].deal['source_frame']]+[
                f for route,labels in self.nick_labels.items() if route[:3]==key
                for label in labels for f in label['frames']]+[
                r['source_frame'] for r in self.roster_nick_rows[key]])
        return min(observed,min(later)-1) if later else observed

    def _resolve_nick_labels(self):
        """Choose FIRST by source order, after earlier pending OCR has returned."""
        from .nickname_roster_binding import FIRST_NICK_POLICY
        if isinstance(self.jobs,PendingSourceJobs):
            earliest_pending=self.jobs.earliest_pending;pending_prefix=self.jobs.pending_prefix
        else:
            earliest_pending={};pending_prefix=set()
            for key,meta in self.jobs.values():
                if meta.get('kind') in {'CHAT','CHAT_CANDIDATE'}:pending_prefix.add(key)
                if meta.get('kind')!='NICK':continue
                pending_route=(*key,meta.get('seat'));pending_frame=meta['source_frame']
                earliest_pending[pending_route]=min(earliest_pending.get(pending_route,pending_frame),pending_frame)
        for route,candidates in getattr(self,'nick_candidates',{}).items():
            if self.nick_labels[route]:continue
            for frame,original in sorted(candidates.items()):
                if earliest_pending.get(route,frame)<frame:break
                text=(original['ocr_response'].get('raw_text') or '').strip()
                if text.isdecimal():
                    snapshot=self.forced.get(route[:3],{})
                    if not snapshot.get('forced_snapshot_complete'):
                        if (route[:3] not in self.seen_deals or route[:3] in self.pending_snapshots or
                                route[:3] in pending_prefix):break
                        continue
                    if text not in snapshot['actors']:continue
                self.nick_labels[route].append(dict(actor=text,frames=[frame],
                    source_ref=original['pixel_sha256'],source_pts=original.get('source_pts'),
                    time_base=original.get('time_base'),nickname_witness_policy=FIRST_NICK_POLICY))
                break

    def _resolve_prefix_snapshots(self, *, final=False):
        """Return retained fallback batches for submission outside locks/callbacks."""
        fallback=[]
        for key,rows in list(self.pending_snapshots.items()):
            expected=self.candidate_sizes.get(key,0)
            if expected and len(self.candidate_rows[key])<expected and not final:continue
            w=self.sidecar.windows[key]
            candidates=list(self.candidate_rows[key].values())
            snapshot,blocker=parse_forced_snapshot(candidates,key[2],len(w.deal['occupied_seats']),
                self.sidecar.flags['stack_confidence']) if expected and len(candidates)==expected else (
                    None,'EARLY_OCR_COVERAGE_INCOMPLETE')
            if snapshot:
                original=candidates[0]
                for post in snapshot['posts']:
                    post.update(source_frame=original['source_frame'],source_pts=original['source_pts'],
                        time_base=original['time_base'])
                snapshot['early_prefix_reused']=True
                self.forced[key]=snapshot
                self.prefix_reuse[key]=dict(status='REUSED_COMPLETE_EARLY_PREFIX',
                    source_frame=original['source_frame'],source_pts=original['source_pts'],
                    time_base=original['time_base'],rows=expected,
                    historical_epoch_authority=False)
                self.counts['FORCED_CHAT_OCR_ROWS_AVOIDED']+=len(rows)
                for row in rows:self.jobs.pop(row_key(row))
            else:
                self.prefix_reuse[key]=dict(status='LATER_SNAPSHOT_REQUIRED',reason=blocker)
                fallback.extend(rows)
            del self.pending_snapshots[key]
        self._resolve_nick_labels()
        return fallback

    def observe_frame(self,table,session,frame,pts,time_base,rgb,chat):
        batch=[]
        occupancy=getattr(self,'occupancy',None)
        with self.sidecar.lock,self.lock:
            h=self.sidecar.current.get(table)
            if not h or h['session_id']!=session:return
            key=(table,session,h['hand_id']);w=self.sidecar.windows.get(key)
            if w is None:return
            trigger=next(t for t in self.sidecar.triggers if (t['table_id'],t['session_id'],t['hand_id'])==key)
            if frame<trigger['source_frame']:return
            self.observed_frames[key]=frame
            batch.extend(self._resolve_prefix_snapshots())
            if w.deal is not None:self.machine.hand(table,session,key[2],trigger['source_frame'])
            source=dict(table_id=table,session_id=session,hand_id=key[2],source_frame=frame,
                        source_pts=pts,time_base=time_base)
            # Preserve source candidates before the independent table/occupancy
            # gate. These rows cannot establish a complete forced snapshot or
            # mutate canonical posts, nicknames, epochs, stacks or hand status.
            if w.deal is None:
                if key not in self.seen_prefix_candidates:
                    self.seen_prefix_candidates.add(key)
                    spans=physical_spans(chat)
                    self.candidate_sizes[key]=len(spans)
                    for i in range(len(spans)):
                        a,b=padded_row_span(spans,i,len(chat));pixels=np.ascontiguousarray(chat[a:b]).copy()
                        row=self.row(table,session,frame,pts,time_base,2000+i,a,pixels,'FORCED_PREFIX_CANDIDATE')
                        meta=dict(source,kind='CHAT_CANDIDATE',index=i,row_bounds=[a,b],source_ref=row.pixel_sha256,
                                  marker_source_frame=trigger['source_frame'],authority='CANDIDATE_ONLY',
                                  admission_rule='CONFIRMED_HAND_MARKER_WITHOUT_DEAL_COMPLETE')
                        self.jobs[row_key(row)]=(key,meta);batch.append(row)
                return_batch=batch
            else:
                return_batch=None
            if w.deal is not None and key not in self.seen_deals:
                self.seen_deals.add(key)
                if occupancy is not None:
                    # Occupancy needs current pixels; a participant's first
                    # qualified NICK stays fixed for this hand.
                    for route in list(self.nick_schedule):
                        if route[:3] == key:self.nick_schedule.pop(route)
                for seat in w.deal.get('occupancy_receipt',{}).get('empty_seats',[]):
                    evidence=w.deal['occupancy_receipt']['seats'][seat]
                    self.machine.observe(table,session,seat,'EMPTY',w.deal['source_frame'],
                        hand_id=key[2],source_ref=evidence['current_pixel_sha256'])
                for seat in w.deal['occupied_seats']:
                    self.machine.observe(table,session,seat,'OCCUPIED',w.deal['source_frame'],
                        hand_id=key[2],source_ref=w.deal['table_pixel_sha256'])
                spans=physical_spans(chat)
                self.snapshot_sizes[key]=len(spans)
                self.snapshot_occupied[key]=len(w.deal['occupied_seats'])
                fallback=[]
                for i in range(len(spans)):
                    a,b=padded_row_span(spans,i,len(chat));pixels=np.ascontiguousarray(chat[a:b]).copy()
                    row=self.row(table,session,frame,pts,time_base,1000+i,a,pixels,'FORCED_SNAPSHOT')
                    meta=dict(source,kind='CHAT',index=i,row_bounds=[a,b],source_ref=row.pixel_sha256)
                    self.jobs[row_key(row)]=(key,meta);fallback.append(row)
                self.pending_snapshots[key]=fallback
                batch.extend(self._resolve_prefix_snapshots())
            # Nick labels are independent observations of fixed seats, not IDs.
            if not self.roster_prefix_binding and w.deal is not None and frame>w.deal['source_frame']+self.sidecar.flags['capture_frames']:return_batch=batch
            elif self.roster_prefix_binding or w.deal is not None or (occupancy is not None and w.closed is None
                    and frame<=trigger['source_frame']+self.sidecar.flags['capture_frames']):
                boxes=self.sidecar.profile['slots'][f'slot_{table:02d}']['seats']
                for i,(seat,b) in enumerate(boxes.items()):
                    x,y,X,Y=b['nick'];pixels=png_compatible_gray(np.ascontiguousarray(rgb[y:Y,x:X])).copy()
                    digest=hashlib.sha256(pixels.tobytes()).hexdigest();bk=(*key,seat)
                    if occupancy is not None:occupancy.observe(key,seat,frame,digest)
                    schedule=self.nick_schedule.get(bk)
                    if (not schedule or schedule['sha']!=digest or
                            occupancy is not None and schedule.get('last_frame')!=frame-1):
                        schedule=dict(sha=digest,count=0);self.nick_schedule[bk]=schedule
                    schedule['last_frame']=frame
                    if schedule['count']<1 and not (w.deal is not None and self.nick_labels[bk]):
                        schedule['count']+=1
                        row=self.row(table,session,frame,pts,time_base,100+i,y,pixels,'NICK_ONLY')
                        meta=dict(source,kind='NICK',seat=seat,roi=b['nick'],source_ref=digest)
                        if occupancy is not None:meta['pre_deal']=w.deal is None
                        self.jobs[row_key(row)]=(key,meta);batch.append(row)
                return_batch=batch
            elif occupancy is not None and w.deal is None and key not in self.occupancy_expired:
                self.occupancy_expired.add(key)
                self.write(dict(source,kind='OCCUPANCY',status='UNKNOWN',
                    reason='PRE_DEAL_OCCUPANCY_SOURCE_WINDOW_EXPIRED_OR_CLOSED',
                    source_frame_cap=trigger['source_frame']+self.sidecar.flags['capture_frames']))
            self.counts['OCR_ATTEMPTED']+=len(return_batch)
        for start in range(0,len(return_batch),8):self.ocr.submit_batch(return_batch[start:start+8])

    def current_occupancy(self,key,frame,seats):
        with self.lock:
            return self.occupancy.receipt(key,frame,seats) if self.occupancy is not None else None

    def current_occupancy_labels(self,key,frame,seats):
        with self.lock:
            return self.occupancy.current_labels(key,frame,seats) if self.occupancy is not None else {}

    @staticmethod
    def row(table,session,frame,pts,time_base,index,y,pixels,kind):
        return RamRow(frame,None,table,session,index,y,kind,0,pixels,
            hashlib.sha256(pixels.tobytes()).hexdigest(),f'{session}:{kind}:{frame}:{index}',pts,*time_base)

    def completed(self,row,response):
        with self.sidecar.lock,self.lock:
            key,meta=self.jobs.pop(row_key(row));text=(response.get('raw_text') or '').strip()
            score=response.get('score');score=score if isinstance(score,(int,float)) else 0
            ok,packed=cv2.imencode('.png',row.pixels)
            if not ok:raise ValueError('SEAT_INPUT_RAM_ENCODING_FAILED')
            original=dict(meta,ocr_response=dict(response),pixel_sha256=row.pixel_sha256,
                png_base64=base64.b64encode(packed).decode('ascii'))
            self.write(original)
            self.counts['OCR_COMPLETED']+=1
            if response.get('status')!='OK':score=0
            if getattr(self,'roster_prefix_binding',False):
                if meta['kind']=='CHAT':
                    self.roster_forced_rows[key][meta['index']]=dict(original,text=text,score=score)
                elif meta['kind']=='NICK':
                    self.roster_nick_rows[key].append(original)
            if meta['kind']=='CHAT_CANDIDATE':
                self.prefix_candidates[meta['table_id']]+=1
                self.candidate_rows[key][meta['index']]=dict(original,text=text,score=score)
                return
            if meta['kind']=='CHAT':
                self.snapshots[key][meta['index']]=dict(meta,text=text,score=score,
                    pixel_sha256=row.pixel_sha256,ocr_response=dict(response),source_frame=row.frame_id)
                if len(self.snapshots[key])==self.snapshot_sizes[key]:
                    snapshot,blocker=parse_forced_snapshot(list(self.snapshots[key].values()),key[2],
                        self.snapshot_occupied[key],self.sidecar.flags['stack_confidence'])
                    if snapshot:
                        for post in snapshot['posts']:
                            post.update(source_frame=meta['source_frame'],source_pts=meta['source_pts'],time_base=meta['time_base'])
                        self.forced[key]=snapshot
                        for e in snapshot['state_events']:
                            self.state_events.append(dict(table=key[0],session=key[1],hand_id=key[2],
                                frame=row.frame_id,source_index=e['index'],actor=e['actor'],kind=e['kind']))
                    else:self.blockers[key]=blocker
            else:
                if getattr(self,'occupancy',None) is not None:
                    self.occupancy.completed(key,meta['seat'],row.frame_id,row.pixel_sha256,response,
                                            pts=row.source_pts,time_base=(row.time_base_num,row.time_base_den))
                bk=(*key,meta['seat'])
                if type(score) not in (int,float) or not math.isfinite(score) or score<NICK_CONFIDENCE_MIN or score>1 or not text:
                    self._resolve_nick_labels();return
                if text.upper()=='OPEN SEAT':
                    if score<self.sidecar.flags['stack_confidence']:
                        self._resolve_nick_labels();return
                    self.machine.observe(key[0],key[1],meta['seat'],'EMPTY',row.frame_id,hand_id=key[2],source_ref=row.pixel_sha256)
                elif text.upper() not in LABEL_OVERLAYS:
                    from .chat_nick_actor import literal_constraint, prove_nick_ui_overlay
                    if literal_constraint(text) is not None and (
                            'roi' not in original or prove_nick_ui_overlay(original) is None):
                        self.nick_candidates[bk][row.frame_id]=original
            self._resolve_nick_labels()

    def ordinary_ocr(self,row,response):
        text=(response.get('raw_text') or '').strip();m=STATE.fullmatch(text)
        if not m or response.get('status')!='OK' or (response.get('score') or 0)<self.sidecar.flags['stack_confidence']:return
        with self.sidecar.lock,self.lock:
            h=[h for h in self.sidecar.histories[(row.table_id,row.session_id)] if h['source_frame']<=row.frame_id]
            hand=max(h,key=lambda h:tuple(h['source_position']))['hand_id'] if h else None
            self.state_events.append(dict(table=row.table_id,session=row.session_id,hand_id=hand,
                frame=row.frame_id,source_index=row.source_span_index,actor=m.group(1),kind=m.group(2).lower(),
                source_ref=row.pixel_sha256))

    def label_matches_epoch(self,key,seat,label,epoch):
        from .nickname_roster_binding import FIRST_NICK_POLICY, nickname_epoch_matches
        if label.get('nickname_witness_policy') in FIRST_NICK_POLICIES:
            window=self.sidecar.windows.get(key)
            return bool(epoch and window and window.deal and all(nickname_epoch_matches(
                self.machine,key,seat,f,window.deal['source_frame'],epoch) for f in label['frames']))
        return self.machine.epoch_at(*key[:2],seat,label['frames'][-1])==epoch

    def seats_for_actor(self,key,actor,limit):
        from .nickname_roster_binding import FIRST_NICK_POLICY, nickname_epoch_matches
        deal_frame=self.sidecar.windows[key].deal['source_frame']
        def membership_frame(r):
            return max(r['frames'][-1],deal_frame) if r.get('nickname_witness_policy') in FIRST_NICK_POLICIES else r['frames'][-1]
        return [seat for seat in self.sidecar.windows[key].deal['occupied_seats']
            if any(r['actor']==actor and r['frames'][-1]<=limit
                and self.machine.epoch_at(key[0],key[1],seat,membership_frame(r)) is not None
                and self.machine.epoch_at(key[0],key[1],seat,membership_frame(r)) == self.machine.epoch_at(key[0],key[1],seat,limit)
                and (r.get('nickname_witness_policy') not in FIRST_NICK_POLICIES or
                     nickname_epoch_matches(self.machine,key,seat,r['frames'][-1],deal_frame,
                         self.machine.epoch_at(key[0],key[1],seat,limit)))
                for r in self.nick_labels[(*key,seat)])]

    def historical_forced_binding(self,key,actor,posts,unresolved_events):
        """Bind a physical post at its first exact label proof, never a later retry.

        Labels are already admitted by completed(). Source confirmation time is
        retained separately from the earlier forced snapshot; no alias is inferred.
        """
        w=self.sidecar.windows[key]
        from .nickname_roster_binding import FIRST_NICK_POLICY, nickname_epoch_matches
        sources={p.get('source_frame') for p in posts}
        if len(sources)!=1 or None in sources:return None,'FORCED_SOURCE_FRAME_UNKNOWN'
        source=next(iter(sources))
        source_limit=w.deal['source_frame']+self.sidecar.flags['capture_frames']
        if not w.deal['source_frame']<=source<=source_limit:return None,'FORCED_SOURCE_OUTSIDE_CAPTURE_HORIZON'
        limit=self._hand_end_frame(key)
        labels=[(r['frames'][-1],seat,r) for seat in w.deal['occupied_seats']
            for r in self.nick_labels[(*key,seat)] if r['actor']==actor and r['frames'][-1]<=limit]
        if not labels:return None,'FORCED_ACTOR_LABEL_UNKNOWN'
        confirmation=min(f for f,seat,r in labels)
        anchor=max(source,confirmation)
        first=[(seat,r) for f,seat,r in labels if f==confirmation]
        seats={seat for seat,r in first}
        if len(seats)!=1:return None,'FORCED_ACTOR_SEAT_CONFLICT'
        seat=next(iter(seats))
        epoch=self.machine.epoch_at(key[0],key[1],seat,source)
        if not epoch or self.machine.epoch_at(key[0],key[1],seat,anchor)!=epoch:
            return None,'FORCED_LABEL_EPOCH_CONFLICT'
        if any((not nickname_epoch_matches(self.machine,key,seat,f,w.deal['source_frame'],epoch)
                if r.get('nickname_witness_policy') in FIRST_NICK_POLICIES else
                self.machine.epoch_at(key[0],key[1],seat,f)!=epoch)
                for s,r in first for f in r['frames']):
            return None,'FORCED_LABEL_TRIPLE_CROSSES_EPOCH'
        actor_seats={s for s in w.deal['occupied_seats'] for r in self.nick_labels[(*key,s)]
            if r['actor']==actor and r['frames'][-1]<=anchor}
        reverse={r['actor'] for r in self.nick_labels[(*key,seat)] if r['frames'][-1]<=anchor}
        if actor_seats!={seat} or reverse!={actor}:return None,'FORCED_LITERAL_CONFLICT_AT_ANCHOR'
        if not self.machine.membership(key[0],key[1],seat,key[2],epoch)['participant']:
            return None,'FORCED_HAND_PARTICIPANT_UNKNOWN'
        if any(e['table']==key[0] and e['session']==key[1] and e['frame']<=anchor for e in unresolved_events):
            return None,'FORCED_OCCUPANCY_LINEAGE_UNKNOWN'
        if any(e['table']==key[0] and e['session']==key[1] and e['frame']<=anchor
            and e['status']!='APPLIED' for e in self.machine.audit):
            return None,'FORCED_OCCUPANCY_EVIDENCE_UNKNOWN'
        policies={r.get('nickname_witness_policy') for s,r in first}
        policy=next(iter(policies)) if len(policies)==1 and policies<=FIRST_NICK_POLICIES else 'LEGACY_RETAINED_LABEL'
        return dict(table=key[0],session=key[1],hand_id=key[2],actor=actor,seat=seat,epoch=epoch,
            binding_rule='FIRST_EXACT_LABEL_HISTORICAL_EPOCH',forced_source_frame=source,
            nickname_witness_policy=policy,
            label_confirmation_frame=confirmation,binding_anchor_frame=anchor,
            label_evidence=[dict(r) for s,r in first],
            forced_source_refs=[p['source_refs'] for p in posts],
            later_literal_conflicts=[dict(r,seat=s) for s in w.deal['occupied_seats']
                for r in self.nick_labels[(*key,s)] if anchor<r['frames'][-1]<=limit
                and ((s==seat and r['actor']!=actor) or (s!=seat and r['actor']==actor))]),None

    def finish(self, *, retained_unresolved_events=None):
        if self.closed:return
        if getattr(self,'pending_snapshots',None):
            self.ocr.jobs.join()
            with self.sidecar.lock,self.lock:
                fallback=self._resolve_prefix_snapshots(final=True)
                self.counts['OCR_ATTEMPTED']+=len(fallback)
            for start in range(0,len(fallback),8):self.ocr.submit_batch(fallback[start:start+8])
        self.ocr.finish()
        with self.sidecar.lock,self.lock:
            unresolved_events=copy.deepcopy(retained_unresolved_events) if retained_unresolved_events is not None else []
            for t in self.sidecar.triggers:
                self.machine.hand(t['table_id'],t['session_id'],t['hand_id'],t['source_frame'])
            self.machine.build()
            for e in sorted(self.state_events,key=lambda e:(e['frame'],e['source_index'])):
                key=(e['table'],e['session'],e['hand_id'])
                if e['hand_id'] is None:
                    unresolved_events.append(dict(e,blocker='SEAT_EVENT_HAND_OR_TABLE_UNKNOWN'));continue
                seats=list({bk[3] for bk,labels in self.nick_labels.items() if bk[:2]==key[:2]
                    and any(r['actor']==e['actor'] and r['frames'][-1]<=e['frame']
                        and self.machine.epoch_at(key[0],key[1],bk[3],e['frame']) is not None
                        and self.label_matches_epoch(bk[:3],bk[3],r,
                            self.machine.epoch_at(key[0],key[1],bk[3],e['frame']))
                        for r in labels)})
                if len(seats)!=1:
                    unresolved_events.append(dict(e,blocker='SEAT_EVENT_ACTOR_UNRESOLVED'));continue
                kind={'sit in':'SIT_IN','sits in':'SIT_IN','sit out':'SIT_OUT','sits out':'SIT_OUT',
                      'has left the table':'EMPTY','has joined the table':'SIT_IN'}[e['kind']]
                self.machine.observe(e['table'],e['session'],seats[0],kind,e['frame'],actor=e['actor'],
                    hand_id=e['hand_id'],source_ref=e.get('source_ref'),source_index=e['source_index'])
                self.machine.build()
            self.machine.build()
            self.sidecar.logs['start_stack_readings'].flush()
            # Freeze original evidence before appending enriched replays to this log.
            readings=[json.loads(l) for l in (self.output/'start_stack_readings.jsonl').read_text('utf-8').splitlines()]
            index={(r['table_id'],r['session_id'],r['hand_id'],r['seat'],r['source_frame']):r for r in readings}
            results=[];bound=[]
            # Forced posts are evidence independently of numeric stack visibility.
            historical_receipts=[];historical_blockers=[]
            roster_proofs=[]
            for key,snapshot in self.forced.items():
                w=self.sidecar.windows.get(key)
                if w is None or w.deal is None:continue
                roster_proof=None
                roster_inputs=None
                if getattr(self,'roster_prefix_binding',False):
                    from parser_core.nickname_roster_binding import prove_roster_binding, binding_receipt
                    roster_inputs=dict(context=list(key),snapshot=dict(copy.deepcopy(snapshot),
                        table=key[0],session=key[1],hand_id=key[2]),
                        deal=copy.deepcopy(w.deal),forced_rows=list(self.roster_forced_rows[key].values()),
                        nick_rows=self.roster_nick_rows[key],
                        hand_triggers=[t for t in self.sidecar.triggers if (t['table_id'],t['session_id'])==key[:2]],
                        state_receipts=[e for e in self.machine.audit if (e['table'],e['session'])==key[:2]],
                        unresolved_events=[e for e in unresolved_events if (e['table'],e['session'])==key[:2]],
                        capture_frames=self.sidecar.flags['capture_frames'],confidence=self.sidecar.flags['stack_confidence'],
                        observed_capture_end_frame=self._hand_end_frame(key))
                    roster_proof=prove_roster_binding(roster_inputs)
                    roster_proofs.append(dict(table=key[0],session=key[1],hand_id=key[2],
                        **{k:v for k,v in roster_proof.items() if k not in {'source_inputs','label_evidence'}}))
                for actor in snapshot['actors']:
                    posts=[p for p in snapshot['posts'] if p['actor']==actor]
                    receipt,blocker=self.historical_forced_binding(key,actor,posts,unresolved_events)
                    if receipt is None and roster_proof and roster_proof['status']=='PROVEN':
                        receipt=binding_receipt(roster_proof,actor,posts)
                    if receipt is None and roster_inputs:
                        from parser_core.nickname_actor_binding import prove_actor_binding, binding_receipt as actor_receipt
                        actor_proof=prove_actor_binding(roster_inputs,actor)
                        receipt=actor_receipt(actor_proof,actor,posts)
                    if receipt is None:
                        historical_blockers.append(dict(table=key[0],session=key[1],hand_id=key[2],actor=actor,blocker=blocker))
                        continue
                    historical_receipts.append(receipt)
                    binding_metadata={'binding_rule':receipt['binding_rule']} if (
                        'roster_proof' in receipt or 'actor_prefix_proof' in receipt) else {}
                    bound.extend(dict(p,table=key[0],session=key[1],hand_id=key[2],seat=receipt['seat'],epoch=receipt['epoch'],
                                      **binding_metadata) for p in posts)

            for key,state in self.stack_live.states.items():
                for seat,triple in state.get('canonical_attempts', state['numeric_triples']).items():
                    record=dict(table=key[0],session=key[1],hand_id=key[2],seat=seat,**triple)
                    record.update(status='UNKNOWN',blocker=None,
                        displayed_stack_status='VALID_FIXED_SEAT_ROI' if triple.get('displayed_stack') is not None and triple.get('status')!='UNKNOWN' else 'UNKNOWN')
                    record['deal_complete_frame']=self.sidecar.windows[key].deal['source_frame'] if self.sidecar.windows[key].deal else None
                    epoch=self.machine.epoch_at(key[0],key[1],seat,triple['frames'][-1])
                    member=self.machine.membership(key[0],key[1],seat,key[2],epoch)
                    record['occupancy_epoch']=epoch
                    record['participant_proven']=bool(member['participant'])
                    snapshot=self.forced.get(key)
                    if triple.get('status') == 'UNKNOWN':
                        record['blocker']=triple['blocker']
                    elif len({self.machine.epoch_at(key[0],key[1],seat,f) for f in triple['frames']})!=1:
                        record['blocker']='STACK_TRIPLE_CROSSES_EPOCH'
                    elif not epoch or not member['participant']:record['blocker']=member['blocker']
                    elif not snapshot:record['blocker']=self.blockers.get(key,'FORCED_SNAPSHOT_UNKNOWN')
                    else:
                        limit=triple['frames'][-1]
                        standard=snapshot.get('standard_ante') if snapshot.get('standard_ante_proven') else None
                        special_posts=[p for p in snapshot['posts'] if p['role'] in {'SB','BB'}
                            or (p['role']=='ANTE' and standard is not None and p['amount']!=standard)]
                        special={};special_unknown=False
                        for p in special_posts:
                            seats=self.seats_for_actor(key,p['actor'],limit)
                            if len(seats)!=1:
                                special_unknown=True;continue
                            target=seats[0]
                            target_epoch=self.machine.epoch_at(key[0],key[1],target,limit)
                            reverse={r['actor'] for r in self.nick_labels[(*key,target)] if r['frames'][-1]<=limit
                                and self.label_matches_epoch(key,target,r,target_epoch)}
                            uncertain=any(e['table']==key[0] and e['session']==key[1] and e['actor']==p['actor'] and e['frame']<=limit for e in unresolved_events)
                            if reverse!={p['actor']} or uncertain or not self.machine.membership(key[0],key[1],target,key[2],target_epoch)['participant']:
                                special_unknown=True;continue
                            special.setdefault(target,{})[p['role']]=p['amount']
                        auto_all_in=any(r and (r.get('raw_text') or '').strip().upper() in {'ALL-IN','ALL IN','AUTO ALL-IN','AUTO ALL IN'}
                            for f,r in state.get('source_readings',{}).get(seat,{}).items() if f<=limit)
                        record['displayed_stack_status']='VALID_FIXED_SEAT_ROI'
                        record['standard_ante']=standard
                        record['standard_ante_source_refs']=[p['source_refs'] for p in snapshot['posts'] if p['role']=='ANTE']
                        record['special_forced_source_refs']=[p['source_refs'] for p in special_posts]
                        record['special_forced_seats_proven']=not special_unknown
                        record['auto_all_in_observed']=auto_all_in
                        record['sb_seat']=next((s for s,b in special.items() if 'SB' in b),None)
                        record['bb_seat']=next((s for s,b in special.items() if 'BB' in b),None)
                        record['sb_amount']=next((b['SB'] for b in special.values() if 'SB' in b),None)
                        record['bb_amount']=next((b['BB'] for b in special.values() if 'BB' in b),None)
                        actors=[a for a in snapshot['actors'] if self.seats_for_actor(key,a,limit)==[seat]]
                        labels_at_seat={r['actor'] for r in self.nick_labels[(*key,seat)] if r['frames'][-1]<=limit
                            and self.label_matches_epoch(key,seat,r,epoch)}
                        bets=None
                        if standard is not None and snapshot.get('forced_snapshot_complete') and not auto_all_in:
                            if any(e['table']==key[0] and e['session']==key[1] and e['frame']<=limit for e in unresolved_events):
                                record['blocker']='SEAT_EVENT_LINEAGE_OR_BINDING_UNKNOWN'
                            elif special_unknown:
                                record['blocker']='SPECIAL_FORCED_SEAT_UNKNOWN'
                            else:
                                bets=dict(ANTE=standard,**special.get(seat,{}))
                                record['binding_rule']='FIXED_SEAT_STANDARD_ANTE_WITH_PROVEN_SPECIAL_SEATS'
                        else:
                            if len(actors)!=1 or labels_at_seat!=set(actors):
                                record['blocker']='AUTO_ALL_IN_ACTUAL_FORCED_UNKNOWN' if auto_all_in else 'FORCED_ACTOR_TO_SEAT_UNKNOWN'
                            else:
                                actor=actors[0]
                                uncertain=[e for e in unresolved_events if e['table']==key[0] and e['session']==key[1]
                                    and e['actor']==actor and e['frame']<=limit]
                                if uncertain:record['blocker']='SEAT_EVENT_LINEAGE_OR_BINDING_UNKNOWN'
                                else:
                                    posts=[p for p in snapshot['posts'] if p['actor']==actor]
                                    bets={p['role']:p['amount'] for p in posts}
                                    record.update(actor=actor,forced_source_refs=[p['source_refs'] for p in posts])
                        if bets is not None:
                            record['forced_bets']=bets
                            record['forced_bets_debited_before_snapshot_proven']=all(
                                p.get('phase')=='BEFORE_DEAL' and p.get('source_frame',limit)<=limit
                                for p in snapshot['posts'])
                            from .start_stack_visibility import valid_numeric_witness
                            if not valid_numeric_witness(triple):
                                record['blocker']='FIRST_VISIBLE_STACK_TRIPLE_UNPROVEN'
                            elif record['deal_complete_frame'] is None or triple['frames'][0] <= record['deal_complete_frame']:
                                record['blocker']='DEAL_COMPLETE_SOURCE_ORDER_UNKNOWN'
                            else:
                                record.update(status='PROVEN_FOR_MATH',blocker=None)
                    results.append(record)
                w=self.sidecar.windows[key]
                if w.deal is not None:
                    existing={r['seat'] for r in results if (r['table'],r['session'],r['hand_id'])==key}
                    for seat in w.deal['occupied_seats']:
                        if seat not in existing:
                            results.append(dict(table=key[0],session=key[1],hand_id=key[2],seat=seat,
                                occupancy_epoch=self.machine.epoch_at(key[0],key[1],seat,w.deal['source_frame']),
                                deal_complete_frame=w.deal['source_frame'],standard_ante=self.forced.get(key,{}).get('standard_ante'),
                                first_numeric_visible_frame=None,frames=[],
                                ocr_values=[],displayed_stack=None,displayed_stack_status='UNKNOWN',status='UNKNOWN',blocker='NUMERIC_STACK_NOT_VISIBLE_OR_FIRST_TRIPLE_INCOMPLETE'))
            summary=dict(epochs_proven=len({e['epoch'] for e in self.machine.audit if e.get('epoch')}),
                states=dict(Counter(r['state'] or 'ACTIVITY_UNKNOWN' for r in self.machine.rows.values())),
                numeric_triples=sum(len(s['numeric_triples']) for s in self.stack_live.states.values()),forced_bets_bound=len(bound),bound_posts=bound,
                facts_proven_for_math=sum(r['status']=='PROVEN_FOR_MATH' for r in results),
                displayed_stack_valid=sum(r['displayed_stack_status']=='VALID_FIXED_SEAT_ROI' for r in results),results=results,
                state_receipts=self.machine.audit,unresolved_state_events=unresolved_events,
                historical_binding_receipts=historical_receipts,historical_binding_blockers=historical_blockers,
                forced_snapshots=[dict(table=k[0],session=k[1],hand_id=k[2],**snapshot) for k,snapshot in self.forced.items()],
                forced_snapshot_blockers={str(k):v for k,v in self.blockers.items()},counts=dict(self.counts),errors=self.ocr.errors,
                candidate_prefix_contexts=len(self.seen_prefix_candidates),
                candidate_prefix_rows_by_table=dict(self.prefix_candidates),candidate_prefix_authority='CANDIDATE_ONLY',
                observed_hand_ends=[dict(table=k[0],session=k[1],hand_id=k[2],source_frame=self._hand_end_frame(k))
                                    for k in getattr(self,'observed_frames',{})],
                early_prefix_reuse=[dict(table=k[0],session=k[1],hand_id=k[2],**v)
                                   for k,v in getattr(self,'prefix_reuse',{}).items()])
            if getattr(self,'roster_prefix_binding',False):summary['roster_binding_proofs']=roster_proofs
            (self.output/'seat_forced_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
            with (self.output/'start_stack_facts.jsonl').open('w',encoding='utf-8') as stream:
                for r in results:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
            self.log.close();self.closed=True
