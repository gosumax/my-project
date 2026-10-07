"""Hand-state readiness, separate from whole-window CHAT row lineage."""
from collections import Counter,defaultdict
from fractions import Fraction


class HandStateMarkerGate:
    def __init__(self):
        self.failed_headers=defaultdict(set)
        self.conflicts={}
        self.counts=Counter()

    def header_unknown(self,row):
        self.failed_headers[(row.table_id,row.session_id)].add((row.frame_id,row.source_span_index))

    def header_conflict(self,table,session,position):
        self.conflicts[(table,session)]=position

    def select(self,provider,table,session):
        current=provider.current.get(table)
        if not current or current['session_id']!=session or provider.active_sessions.get(table)!=session:
            self.counts['CURRENT_HAND_UNKNOWN']+=1;return None
        anchors=[h for h in provider.histories[(table,session)] if h['hand_id']==current['hand_id']]
        conflict=self.conflicts.get((table,session))
        if conflict is not None:
            anchors=[h for h in anchors if tuple(h['source_position'])>conflict]
        if not anchors:
            self.counts['HEADER_ANCHOR_UNKNOWN']+=1;return None
        # Repeated same-ID observations cannot move the hand's source boundary.
        header=min(anchors,key=lambda h:tuple(h['source_position']))
        if (table,session,header['hand_id']) in provider.fired:return None
        for record in provider.frames[(table,session)]:
            frame=record['source_frame']
            if frame<header['source_frame']:continue
            if record.get('source_pts') is not None and header.get('source_pts') is not None:
                hbase=header.get('time_base');rbase=record.get('time_base')
                if hbase and rbase and all(hbase+rbase):
                    ht=header['source_pts']*Fraction(*hbase);rt=record['source_pts']*Fraction(*rbase)
                    if rt<ht or (frame==header['source_frame'] and rt!=ht):
                        self.counts['SOURCE_ORDER_CONFLICT']+=1;continue
            if frame>provider.sealed.get((table,session),-1):continue
            if any(f<=frame and index>=1000000 for f,index in provider.outstanding[(table,session)]):
                self.counts['HEADER_OCR_PENDING']+=1;continue
            if any(tuple(header['source_position'])<=pos and pos[0]<=frame for pos in self.failed_headers[(table,session)]):
                self.counts['HEADER_OCR_UNKNOWN']+=1;continue
            eligible=record['cards']
            # Keep physical order checks, without asking other CHAT rows for IDs.
            witnesses=[h for h in (header,current) if h['source_frame']<=frame]
            visible=[r for r in record['rows'] if any(r['pixel_hash']==h['header_pixel_sha256'] for h in witnesses)]
            if visible:
                eligible=[r for r in eligible if r['row_bounds'][0]>=max(v['row_bounds'][1] for v in visible)]
            elif frame==header['source_frame']:
                eligible=[r for r in eligible if r['row_index']>header['source_position'][1]]
            if len(eligible)!=1:
                if eligible:self.counts['CARDS_ORDER_AMBIGUOUS']+=1
                continue
            self.counts['HAND_STATE_READY']+=1
            return header,record,eligible[0]
        return None
