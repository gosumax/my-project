"""OFF-by-default RAM capture sidecar. It cannot promote CHAT to TABLE proof."""
import base64,hashlib,json,os,re,threading,time
from collections import Counter,defaultdict,deque
from decimal import Decimal
from pathlib import Path
import cv2
import numpy as np
from parser_core.chat_rows import physical_spans,padded_row_span
from parser_core.dealing_cards_ncc import TwoStageModel
from parser_core.chat_tracker import CONFIRMED_LINK_STATES

HEADER=re.compile(r'^(?:Hand\s*)?#\s*(\d{8,})\s*$',re.I)
MONEY_ACTION=re.compile(r'^Dealer:\s+.+\s+(calls|bets|raises|is all-in|goes all-in)\b',re.I)
NUMERIC=re.compile(r'^\d+(?:[ \u00a0]\d{3})*$')

class CleanStackWindow:
    def __init__(self,confidence=.95):
        self.confidence=confidence;self.deal=None;self.closed=None;self.runs={};self.results={}
    def confirm_deal(self,receipt):
        if receipt.get('evidence_status')!='PROVEN' or not receipt.get('table_pixel_sha256') or not receipt.get('occupied_seats'):
            return 'DEAL_COMPLETE_UNKNOWN'
        self.deal=dict(receipt);return 'DEAL_COMPLETE'
    def action(self,kind,frame):
        if kind in {'CALL','BET','RAISE','RAISE_TO','ALL-IN','ALL_IN'}:
            self.closed=frame if self.closed is None else min(frame,self.closed)
            # A delayed action receipt can invalidate previously provisional
            # readings. It never repairs or silently retains a closed window.
            for seat,result in list(self.results.items()):
                if result['confirmation_frame']>=self.closed:del self.results[seat]
    def reading(self,seat,frame,text,confidence,forced_bets=None,occupancy_epoch=None,forced_bets_proven=False,occupancy_proven=False):
        if self.deal is None:return None,'DEAL_COMPLETE_UNKNOWN'
        if frame<self.deal['source_frame'] or (self.closed is not None and frame>=self.closed):return None,'WINDOW_CLOSED_OR_TIMING_UNKNOWN'
        if seat not in self.deal['occupied_seats'] or occupancy_epoch is None or not occupancy_proven:return None,'IDENTITY_UNKNOWN'
        if not NUMERIC.fullmatch((text or '').strip()) or confidence is None or confidence<self.confidence:
            self.runs.pop(seat,None);return None,'STACK_NOT_STABLE_NUMERIC'
        value=Decimal(text.replace(' ','').replace('\u00a0',''));old=self.runs.get(seat)
        same=old and old['value']==value and old['last']+1==frame and old['epoch']==occupancy_epoch
        state={'value':value,'epoch':occupancy_epoch,'first':old['first'] if same else frame,'last':frame,'count':old['count']+1 if same else 1};self.runs[seat]=state
        if state['count']<3:return None,'STACK_CONFIRMATION_PENDING'
        if not forced_bets_proven or not forced_bets or any(k not in {'ANTE','SB','BB'} for k in forced_bets):return None,'FORCED_BET_UNKNOWN'
        if any(not isinstance(v,(int,Decimal)) or isinstance(v,bool) or v<0 for v in forced_bets.values()):return None,'FORCED_BET_UNKNOWN'
        result={'seat':seat,'occupancy_epoch':occupancy_epoch,'snapshot_frame':state['first'],'confirmation_frame':frame,
                'displayed_stack':str(value),'forced_bets':{k:str(v) for k,v in forced_bets.items()},'start_stack':str(value+sum(forced_bets.values()))}
        self.results.setdefault(seat,result);return self.results[seat],'START_STACK_RECOVERED'

class DealingStartStackSidecar:
    def __init__(self,root,output,flags,hand_state_gate=None):
        self.root=Path(root);self.output=Path(output);self.flags=dict(flags);self.lock=threading.RLock()
        self.model=TwoStageModel(self.root/flags['ncc_config']);self.current={};self.active_sessions={};self.histories=defaultdict(list)
        self.frames=defaultdict(deque);self.fired=set();self.counts=Counter();self.capturing={};self.windows={};self.triggers=[]
        self.actions=defaultdict(list);self.outstanding=defaultdict(set);self.sealed={};self.context_proofs={}
        self.source_barriers=defaultdict(set);self.previous_cards_count={};self.capture_episodes=Counter()
        self.finished_summary=None
        self.hand_state_gate=hand_state_gate
        self.header_fences={}
        self.profile=json.loads((self.root/flags['roi_profile']).read_text('utf-8'))
        self.logs={name:(self.output/(name+'.jsonl')).open('w',encoding='utf-8') for name in
                   ('hand_state_receipts','dealing_ncc_receipts','dealing_cards_triggers','start_stack_capture','deal_complete_receipts')}
        self.logs['start_stack_readings']=(self.output/'start_stack_readings.jsonl').open('w',encoding='utf-8')
        # Opt-in bounded diagnostics only; never participates in admission.
        self.order_trace_scope=json.loads(os.environ.get('START_STACK_ORDER_TRACE','null'))
        self.order_trace_seq=0
        if self.order_trace_scope:
            self.logs['start_stack_order_trace']=(self.output/'start_stack_order_trace.jsonl').open('w',encoding='utf-8')
    def _log(self,name,value):self.logs[name].write(json.dumps(value,ensure_ascii=False)+'\n')
    def trace_order(self,event,table,session,frame,**details):
        scope=getattr(self,'order_trace_scope',None)
        if not scope or table!=scope['table_id'] or frame>scope['max_source_frame']:return
        # Callers already hold provider.lock: sequence describes actual lock order.
        self.order_trace_seq+=1
        current=self.current.get(table)
        fields=('hand_id','session_id','source_frame','source_position','source_pts','time_base')
        snapshot=lambda h:{k:h.get(k) for k in fields} if h else None
        history=[h for h in self.histories[(table,session)] if current and h['hand_id']==current['hand_id']]
        self._log('start_stack_order_trace',dict(event=event,sequence=self.order_trace_seq,
            monotonic_ns=time.perf_counter_ns(),thread_id=threading.get_ident(),
            table_id=table,session_id=session,source_frame=frame,
            capture_frame=self.counts.get('last_capture_frame'),current=snapshot(current),
            history_anchor=snapshot(history[0]) if history else None,
            outstanding_headers=sorted([list(p) for p in self.outstanding[(table,session)] if p[1]>=1000000]),
            **details))
    def header(self,table,session,hand,frame,pts,index=0,time_base=None,pixels_sha=None):
        key=(table,session);position=(frame,index);old=self.current.get(table)
        self.trace_order('HEADER_ENTER',table,session,frame,callback_hand_id=hand,
            source_pts=pts,source_span_index=index)
        if table in self.active_sessions and self.active_sessions[table]!=session:
            self.counts['STALE_SESSION_IGNORED']+=1
            self._log('hand_state_receipts',{'table_id':table,'session_id':session,'hand_id':hand,'source_frame':frame,'source_pts':pts,'status':'STALE_SESSION_IGNORED'})
            return 'STALE_SESSION_IGNORED'
        if old and old['session_id']!=session:
            self.current.pop(table,None);self.frames.pop(key,None);self.counts['SESSION_RESET']+=1;old=None
        status='HAND_HEADER_CONFIRMED'
        fence=self.header_fences.get(key)
        if fence and position<tuple(fence['source_position']):status='STALE_HEADER_IGNORED'
        elif fence and position==tuple(fence['source_position']) and (fence['unknown'] or fence['hand_id']!=hand):status='HEADER_CONFLICT_UNKNOWN';self.current.pop(table,None)
        elif fence and int(hand)<int(fence['hand_id']):status='HEADER_CONFLICT_UNKNOWN';self.current.pop(table,None)
        else:
            record={'table_id':table,'session_id':session,'hand_id':hand,'source_frame':frame,'source_position':list(position),
                    'source_pts':pts,'time_base':time_base,'header_pixel_sha256':pixels_sha,
                    'confirmation_monotonic_ns':time.perf_counter_ns(),
                    'confirmation_last_observed_frame':self.counts.get('last_capture_frame')}
            self.current[table]=record
            if not old or old['hand_id']!=hand:self.histories[key].append(record)
        if status!='STALE_HEADER_IGNORED':
            self.header_fences[key]={'source_position':list(position),
                'hand_id':hand if status=='HAND_HEADER_CONFIRMED' else fence['hand_id'],
                'unknown':status=='HEADER_CONFLICT_UNKNOWN'}
        if status=='HEADER_CONFLICT_UNKNOWN' and self.hand_state_gate is not None:
            self.hand_state_gate.header_conflict(table,session,position)
        self.trace_order('HEADER_CURRENT_AFTER',table,session,frame,status=status,
            previous_current=dict(old) if old else None,current_replaced=self.current.get(table) is not old)
        self.counts[status]+=1
        self._log('hand_state_receipts',{'table_id':table,'session_id':session,'hand_id':hand,'source_frame':frame,'source_pts':pts,'source_span_index':index,'status':status,
                  'callback_monotonic_ns':time.perf_counter_ns(),'last_observed_capture_frame':self.counts.get('last_capture_frame')})
        return status
    def ocr_complete(self,row,response):
        with self.lock:
            text=(response.get('raw_text') or '').strip();match=HEADER.fullmatch(text)
            if response.get('status')!='OK' or not text or ((text.lower().startswith('hand') or text.startswith('#')) and match is None):
                self.source_barriers[(row.table_id,row.session_id)].add((row.frame_id,row.source_span_index))
                self.counts['OCR_HAND_CONTEXT_BARRIER']+=1
            if match and response.get('status')=='OK':
                self.trace_order('HEADER_CALLBACK_ENTER',row.table_id,row.session_id,row.frame_id,
                    lane='ORDINARY',source_pts=row.source_pts,source_span_index=row.source_span_index)
                self.header(row.table_id,row.session_id,match.group(1),row.frame_id,row.source_pts,row.source_span_index,
                            [row.time_base_num,row.time_base_den],row.pixel_sha256)
                self.trace_order('HEADER_CALLBACK_RETURN',row.table_id,row.session_id,row.frame_id,lane='ORDINARY')
            action=MONEY_ACTION.match(text)
            if action:
                kinds={'calls':'CALL','bets':'BET','raises':'RAISE','is all-in':'ALL_IN','goes all-in':'ALL_IN'}
                self.actions[(row.table_id,row.session_id)].append({'frame':row.frame_id,'kind':kinds[action.group(1).lower()]})
                applicable=[h for h in self.histories[(row.table_id,row.session_id)] if h['source_frame']<=row.frame_id]
                if applicable:
                    header=max(applicable,key=lambda h:tuple(h['source_position']));key=(row.table_id,row.session_id,header['hand_id'])
                    if key in self.windows:self.windows[key].action(kinds[action.group(1).lower()],row.frame_id)
            self.outstanding[(row.table_id,row.session_id)].discard((row.frame_id,row.source_span_index))
            self._release(row.table_id,row.session_id)
    def note_ocr_submission(self,row):
        with self.lock:self.outstanding[(row.table_id,row.session_id)].add((row.frame_id,row.source_span_index))
    def tracker_transition(self,receipt):
        with self.lock:
            table=receipt.get('table_id',receipt.get('table'));frame=receipt.get('frame_id',receipt.get('frame_to'))
            if table is None or frame is None:raise ValueError('TRACKER_SOURCE_CONTEXT_MISSING')
            key=(table,receipt['session_id']);classification=receipt['classification']
            complete=classification not in {'INITIAL','INITIAL_VISIBLE','UNRESOLVED'} and not classification.startswith('LOCAL')
            complete=complete and all(state in CONFIRMED_LINK_STATES for state in receipt.get('new_link_states',[]))
            self.context_proofs[key]={'frame':frame,'complete':complete,'classification':classification}
            for record in self.frames[key]:
                if record['source_frame']==frame:record['context_complete']=complete
    def _release(self,table,session):
        if self.hand_state_gate is not None:
            selected=self.hand_state_gate.select(self,table,session)
            if selected is not None:
                header,record,candidate=selected
                self._commit_marker(header,record,candidate)
            return
        header=self.current.get(table)
        if not header or header['session_id']!=session:return
        key=(table,session,header['hand_id'])
        if key in self.fired:return
        for record in self.frames[(table,session)]:
            if record['source_frame']<header['source_frame']:continue
            if record['source_frame']>self.sealed.get((table,session),-1):continue
            if any(f<=record['source_frame'] for f,_ in self.outstanding[(table,session)]):continue
            if any(tuple(header['source_position'])<=position and position[0]<=record['source_frame'] for position in self.source_barriers[(table,session)]):
                self.counts['HAND_CONTEXT_OCR_FAILURE_UNKNOWN']+=1;continue
            if not record.get('context_complete'):
                self.counts['HAND_CONTEXT_COMPLETENESS_UNKNOWN']+=1;continue
            eligible=record['cards']
            visible=[r for r in record['rows'] if r['pixel_hash']==header['header_pixel_sha256']]
            if visible:eligible=[r for r in eligible if r['row_bounds'][0]>=max(v['row_bounds'][1] for v in visible)]
            elif record['source_frame']==header['source_frame']:
                eligible=[r for r in eligible if r['row_index']>header['source_position'][1]]
            if len(eligible)!=1:continue
            self._commit_marker(header,record,eligible[0])
            return
    def _commit_marker(self,header,record,candidate):
        table=header['table_id'];session=header['session_id'];key=(table,session,header['hand_id'])
        event=dict(header,trigger_source_frame=record['source_frame'],trigger_source_pts=record['source_pts'],
                   release_capture_frame=self.counts.get('last_capture_frame'),row_bounds=candidate['row_bounds'],
                   pixel_hash=candidate['pixel_hash'],marker='PREFLOP_DEALING_CARDS',
                   candidate_observation_monotonic_ns=record.get('observation_monotonic_ns'),
                   observed_current_hand_id=record.get('observed_current_hand_id'))
        self.fired.add(key);self.triggers.append(event);self.windows[key]=CleanStackWindow(self.flags['stack_confidence'])
        for action in self.actions[(table,session)]:
            if action['frame']>=header['source_frame']:self.windows[key].action(action['kind'],action['frame'])
        self.counts['PREFLOP_DEALING_CARDS']+=1;self._log('dealing_cards_triggers',event)
        self._log('deal_complete_receipts',dict(event,status='UNKNOWN',blocker='TABLE_DEAL_COMPLETE_PROVIDER_UNAVAILABLE'))
    def observe_frame(self,table,session,frame,pts,time_base,table_rgb,chat_gray):
        with self.lock:
            if table in self.active_sessions and self.active_sessions[table]!=session:
                self.current.pop(table,None);self.counts['SESSION_RESET']+=1
            self.active_sessions[table]=session
            self.sealed[(table,session)]=frame-1
            proof=self.context_proofs.get((table,session),{})
            for record in self.frames[(table,session)]:
                if record['source_frame']==proof.get('frame'):record['context_complete']=proof.get('complete',False)
            self.counts['last_capture_frame']=frame;rows=[];cards=[]
            for i in range(len(spans:=physical_spans(chat_gray))):
                a,b=padded_row_span(spans,i,len(chat_gray));gray=np.ascontiguousarray(chat_gray[a:b]);r=self.model.classify(gray)
                r.update(row_index=i,row_bounds=[a,b])
                self.counts['NCC_ROWS']+=1;self.counts['NCC_UNKNOWN']+=r['status']=='UNKNOWN'
                if r['gate_pass']:
                    self._log('dealing_ncc_receipts',dict(r,table_id=table,session_id=session,source_frame=frame,source_pts=pts,time_base=time_base))
                rows.append(r)
                if r['status']=='cards':cards.append(r)
            queue=self.frames[(table,session)];queue.append({'source_frame':frame,'source_pts':pts,'time_base':time_base,'rows':rows,'cards':cards,'context_complete':bool(proof.get('complete')),
                'observation_monotonic_ns':time.perf_counter_ns(),'observed_current_hand_id':self.current.get(table,{}).get('hand_id')})
            if len(queue)>self.flags['candidate_buffer_frames']:
                dropped=queue.popleft()
                if dropped['cards']:self.counts['NCC_CANDIDATE_BUFFER_EXPIRED']+=1
            capture_hand=self.current.get(table,{}).get('hand_id');arm=self.capturing.get((table,session))
            previous_cards=self.previous_cards_count.get((table,session),0)
            new_episode=bool(cards) and (previous_cards==0 or len(cards)>previous_cards)
            self.previous_cards_count[(table,session)]=len(cards)
            if cards and (new_episode or arm is None or arm['hand_id']!=capture_hand):
                self.capture_episodes[(table,session)]+=1
                self.capturing[(table,session)]={'hand_id':capture_hand,'until':frame+self.flags['capture_frames'],
                                               'episode':self.capture_episodes[(table,session)]}
            if self.flags['enable_start_stack_capture'] and frame<=self.capturing.get((table,session),{}).get('until',-1):
                seat_boxes=self.profile['slots'][f'slot_{table:02d}']['seats']
                for seat,boxes in seat_boxes.items():
                    for field in ('nick','stack'):
                        x1,y1,x2,y2=boxes[field];crop=np.ascontiguousarray(table_rgb[y1:y2,x1:x2])
                        ok,packed=cv2.imencode('.png',cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
                        if not ok:raise ValueError('STACK_RAM_ENCODING_FAILED')
                        self._log('start_stack_capture',{'table_id':table,'session_id':session,'source_frame':frame,'source_pts':pts,
                            'time_base':time_base,'seat':seat,'field':field,'roi':boxes[field],
                            'provisional_capture_episode':self.capturing[(table,session)]['episode'],
                            'pixel_sha256':hashlib.sha256(crop.tobytes()).hexdigest(),'png_base64':base64.b64encode(packed).decode('ascii'),
                            'status':'PROVISIONAL_UNKNOWN','blocker':'TABLE_DEAL_COMPLETE_AND_OCCUPANCY_UNPROVEN'})
                        self.counts['ROI_CROPS']+=1
            self._release(table,session)
    def confirm_deal_complete(self,key,receipt):
        with self.lock:
            if key not in self.windows:return 'HAND_TRIGGER_UNKNOWN'
            if (receipt.get('table_id'),receipt.get('session_id'),receipt.get('hand_id'))!=key:return 'DEAL_COMPLETE_BINDING_UNKNOWN'
            status=self.windows[key].confirm_deal(receipt);self._log('deal_complete_receipts',dict(receipt,status=status));return status
    def stack_reading(self,key,receipt):
        with self.lock:
            if key not in self.windows:return None,'HAND_TRIGGER_UNKNOWN'
            if (receipt.get('table_id'),receipt.get('session_id'),receipt.get('hand_id'))!=key:return None,'STACK_BINDING_UNKNOWN'
            result,status=self.windows[key].reading(receipt['seat'],receipt['source_frame'],receipt.get('raw_text'),receipt.get('confidence'),
                receipt.get('forced_bets'),receipt.get('occupancy_epoch'),receipt.get('forced_bets_proven',False),receipt.get('occupancy_proven',False))
            self._log('start_stack_readings',dict(receipt,status=status,result=result));return result,status
    def finish(self):
        with self.lock:
            if self.finished_summary is not None:return self.finished_summary
            for key,queue in self.frames.items():
                if queue:self.sealed[key]=queue[-1]['source_frame']
                self._release(*key)
            summary={'flags':self.flags,'counts':dict(self.counts),'hand_state_tables':len(self.current),
                     'marker_gate':'HAND_STATE' if self.hand_state_gate is not None else 'LEGACY_CHAT_COMPLETENESS',
                     'marker_gate_counts':dict(self.hand_state_gate.counts) if self.hand_state_gate is not None else {},
                     'triggers':len(self.triggers),'duplicate_triggers':len(self.triggers)-len({(t['table_id'],t['session_id'],t['hand_id']) for t in self.triggers}),
                     'deal_complete':sum(w.deal is not None for w in self.windows.values()),
                     'start_stack_recovered':sum(len(w.results) for w in self.windows.values()),
                     'first_unproven_stage':'TABLE_DEAL_COMPLETE','blockers':['TABLE_DEAL_COMPLETE_PROVIDER_UNAVAILABLE','OCCUPANCY_EPOCH_UNPROVEN','FORCED_BET_BINDING_UNPROVEN'],
                     'verdict':'NOT_PROVEN','authority':'SIDECAR_ONLY; original outputs unchanged'}
            (self.output/'dealing_start_stack_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
            for stream in self.logs.values():stream.close()
            self.finished_summary=summary
            return summary
