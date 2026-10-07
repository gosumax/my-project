"""Isolated persistent hand-state marker. No production imports or mutation.

The context provider must attest the same live table/session, independently of
the cards classifier. Backward/history viewport, replacement, or lost context
invalidate the stored hand. Header disappearance alone does not invalidate it.
"""

class PersistentHandStateTrigger:
    def __init__(self):
        self.contexts={};self.states={};self.header_hashes={};self.fired=set();self.transitions=[]

    def invalidate(self,table,frame,reason):
        self.states.pop(table,None)
        self.transitions.append({'table':table,'frame':frame,'transition':'UNKNOWN','reason':reason})

    def set_context(self,table,session_id,table_instance_id,frame,confirmed=True,replacement=False):
        if not confirmed or session_id is None or table_instance_id is None:
            self.contexts.pop(table,None);self.header_hashes.pop(table,None)
            self.invalidate(table,frame,'CONTEXT_UNKNOWN');return False
        identity=(session_id,table_instance_id);old=self.contexts.get(table)
        if old is None or old['identity']!=identity or replacement:
            generation=(old['generation']+1) if old else 1
            self.contexts[table]={'identity':identity,'generation':generation,'last_frame':None}
            self.header_hashes.pop(table,None)
            self.invalidate(table,frame,'TABLE_REPLACED' if replacement else 'SESSION_OR_TABLE_CONTEXT_STARTED')
        return True

    def confirm_hand(self,table,hand_id,frame,header_pixel_hash):
        context=self.contexts.get(table)
        if context is None:
            self.invalidate(table,frame,'CONTEXT_UNKNOWN');return False
        if not isinstance(hand_id,str) or not hand_id.isdigit() or not header_pixel_hash:
            self.invalidate(table,frame,'HEADER_CONFLICT');return False
        old=self.states.get(table)
        if old and frame<old['last_header_frame']:
            self.invalidate(table,frame,'HEADER_SOURCE_ORDER_CONFLICT');return False
        if old and old['hand_id']==hand_id:
            self.header_hashes.setdefault(table,{})[header_pixel_hash]=hand_id
            old['last_header_frame']=frame
            return True
        if old and int(hand_id)<int(old['hand_id']):
            self.invalidate(table,frame,'CONTRADICTING_NEW_HEADER');return False
        self.header_hashes.setdefault(table,{})[header_pixel_hash]=hand_id
        self.states[table]={'hand_id':hand_id,'header_hash':header_pixel_hash,'header_frame':frame,'last_header_frame':frame}
        self.transitions.append({'table':table,'frame':frame,'transition':'HAND_HEADER_CONFIRMED',
                                 'previous_hand_id':old['hand_id'] if old else None,'current_hand_id':hand_id,
                                 'header_pixel_hash':header_pixel_hash})
        return True

    def current_hand_id(self,table):
        state=self.states.get(table);return state['hand_id'] if state else None

    def observe(self,table,frame,rows,live_context_confirmed=False):
        context=self.contexts.get(table)
        if context is None or not live_context_confirmed:
            self.invalidate(table,frame,'LIVE_CONTEXT_UNCONFIRMED');return None,'CURRENT_HAND_UNKNOWN',set()
        previous=context['last_frame'];context['last_frame']=frame
        if previous is not None and frame!=previous+1:
            renewed=self.states.get(table)
            if renewed is None or renewed['last_header_frame']!=frame:
                self.invalidate(table,frame,'SOURCE_FRAME_GAP');return None,'CURRENT_HAND_UNKNOWN',set()
        state=self.states.get(table)
        if state is None:return None,'CURRENT_HAND_UNKNOWN',set()
        key=(table,context['identity'],context['generation'],state['hand_id'])
        # A header belonging to another hand is not silently assigned to the
        # saved hand. Known old headers may remain above the current one.
        headers=[(r,r.get('hand_header_id') or self.header_hashes.get(table,{}).get(r['pixel_hash'])) for r in rows]
        newer=[r for r,h in headers if h is not None and int(h)>int(state['hand_id'])]
        if newer:
            self.invalidate(table,frame,'NEW_HEADER_NEEDS_CONFIRMATION');return None,'CURRENT_HAND_UNKNOWN',set()
        if key in self.fired:return None,'ALREADY_TRIGGERED',set()
        current=[r for r,h in headers if h==state['hand_id']]
        # Hand #... and a following bare #... can both identify the same
        # hand. Their shared ID is not a contradictory hand transition.
        boundary=max(r['row_bounds'][1] for r in current) if current else None
        candidates=[]
        for row in rows:
            if row.get('status')!='cards':continue
            if boundary is not None and row['row_bounds'][0]<boundary:continue
            preceding=[(r,h) for r,h in headers if h is not None and r['row_bounds'][1]<=row['row_bounds'][0]]
            if preceding:
                _,nearest=max(preceding,key=lambda pair:pair[0]['row_bounds'][1])
                if nearest!=state['hand_id']:continue
            candidates.append(row)
        eligible={r['row_index'] for r in candidates}
        if len(candidates)!=1:
            return None,'NO_CURRENT_HAND_CARDS' if not candidates else 'MULTIPLE_CARDS_ROWS_UNKNOWN',eligible
        row=candidates[0];self.fired.add(key)
        return {'table':table,'session_id':context['identity'][0],'table_instance_id':context['identity'][1],
                'context_generation':context['generation'],'hand_id':state['hand_id'],
                'header_confirmed_frame':state['header_frame'],'header_pixel_hash':state['header_hash'],
                'trigger_frame':frame,'first_appearance_frame':frame,'row_index':row['row_index'],
                'row_bounds':row['row_bounds'],'pixel_hash':row['pixel_hash'],
                'marker':'PREFLOP_DEALING_CARDS','binding':'PERSISTENT_CONFIRMED_HAND_IN_LIVE_CONTEXT',
                'header_co_visible':bool(current),'row_birth_required':False},'TRIGGER',eligible
