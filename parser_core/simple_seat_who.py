"""Fixed NICK ROI owns its seat; identity persists until occupant replacement."""
from collections import defaultdict
from .chat_nick_actor import literal_constraint, matches, prove_nick_ui_overlay

POLICY = 'FIXED_NICK_ROI_OCCUPANT_LINEAGE_V8'
UI = {'CALL','RAISE','BET','CHECK','FOLD','ALL-IN','ALL IN','ALLIN','SIT OUT','SITOUT',
      'SITS OUT','SIT IN','SITS IN','SITIN','OPEN SEAT','OPENSEAT','PENSEAT','EMPTY',
      'TIME','TIME BANK','TIMEBANK','ANTE','SB','BB','WAIT','WAITING','AWAY','DEALER'}
CLEAR = {'EMPTY','OPEN SEAT','OPENSEAT','REPLACE','LEAVE','LEFT','FINISHED',
         'HAS LEFT THE TABLE'}


def nick_constraint(raw):
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.upper() in UI or not text:
        return None
    return literal_constraint(text)


def nick_observation_constraint(row):
    if prove_nick_ui_overlay(row) is not None:
        return None
    return nick_constraint(row.get("raw_text", row.get("raw_nick_OCR")))


def nick_matches(label, actor):
    return bool(label and matches(actor, label))


def compatible(a, b):
    if a['kind']=='EXACT':
        return nick_matches(b, a['text'])
    if b['kind']=='EXACT':
        return nick_matches(a, b['text'])
    return a['text'].startswith(b['text']) or b['text'].startswith(a['text'])


class SeatIdentityLineage:
    """One good observation establishes WHO; replacement needs sustained change.

    Two distinct frames of another nickname establish a replacement at the
    first such frame. A single anomalous OCR leaves the occupant unchanged.
    No hand/window/CHAT/confidence gate is imposed on fixed-seat observations.
    """
    def __init__(self, observations=(), state_events=()):
        self.observations = list(observations)
        self.ui_overlay_rejections = []
        self.segments = defaultdict(list)
        self.boundaries = defaultdict(list)
        routes = defaultdict(list)
        for row in self.observations:
            route = (int(str(row['table_id']).removeprefix('T')), row['session_id'], row['seat'])
            raw = row.get('raw_text', row.get('raw_nick_OCR'))
            if isinstance(raw,str) and raw.strip().upper() in {'OPEN SEAT','OPENSEAT','EMPTY'}:
                routes[route].append((row['frame'], 0, 'EMPTY', row))
            elif (proof:=prove_nick_ui_overlay(row)) is not None:
                self.ui_overlay_rejections.append(dict(source=row, proof=proof))
            elif (label:=nick_observation_constraint(row)) is not None:
                routes[route].append((row['frame'], 1, label, row))
        for event in state_events:
            kind = str(event.get('kind','')).upper()
            seat = event.get('seat')
            if not seat or kind not in CLEAR or event.get('evidence_status','PROVEN')!='PROVEN':
                continue
            route = (int(str(event.get('table_id',event.get('table'))).removeprefix('T')),
                     event.get('session_id',event.get('session')), seat)
            routes[route].append((event.get('frame',event.get('source_frame')),0,kind,event))
        for route, entries in routes.items():
            active = None
            low = 0
            pending = []
            for frame, _, label, row in sorted(entries,key=lambda e:(e[0],e[1])):
                if isinstance(label,str):
                    if active:
                        active['end'] = frame-1
                    self.boundaries[route].append(dict(frame=frame, kind=label, source=row))
                    active = None
                    pending = []
                    low = frame+1
                    continue
                if active is None:
                    active = dict(start=low,end=None,label=label,observations=[row],ignored=[])
                    self.segments[route].append(active)
                elif compatible(active['label'],label):
                    active['observations'].append(row)
                    if label['kind']=='EXACT':
                        active['label']=label
                    pending=[]
                else:
                    if not pending or not compatible(pending[0][1],label):
                        pending=[]
                    pending.append((frame,label,row))
                    active['ignored'].append(row)
                    if len({p[0] for p in pending})>=2:
                        low=pending[0][0]
                        active['end']=low-1
                        self.boundaries[route].append(dict(frame=low,kind='STABLE_NEW_NICK',
                                                         source=[p[2] for p in pending]))
                        active=dict(start=low,end=None,label=label,
                                    observations=[p[2] for p in pending],ignored=[])
                        self.segments[route].append(active)
                        pending=[]

    def for_seat(self, table, session, seat, frame):
        for segment in self.segments[(table,session,seat)]:
            if segment['start']<=frame and (segment['end'] is None or frame<=segment['end']):
                return segment
        return None

    def for_actor(self, table, session, actor, low, high, *, actors=()):
        candidates=[]
        for route, segments in self.segments.items():
            if route[:2]!=(table,session):
                continue
            for segment in segments:
                if segment['start']>high or (segment['end'] is not None and segment['end']<low):
                    continue
                if not nick_matches(segment['label'], actor):
                    continue
                matches=[r for r in segment['observations']
                         if nick_matches(nick_observation_constraint(r),actor)]
                if not matches:
                    continue
                exact=[r for r in matches if nick_observation_constraint(r)['kind']=='EXACT']
                if not exact and sum(nick_matches(segment['label'],a) for a in actors)>1:
                    continue
                current=[r for r in matches if low<=r['frame']<=high]
                exact_current=[r for r in current if r in exact]
                previous=[r for r in matches if r['frame']<low]
                chosen=min(exact_current or current or previous or matches,
                           key=lambda r:r['frame'])
                label=nick_observation_constraint(chosen)
                if chosen['frame']<low:
                    rule='CARRIED_OCCUPANT'
                elif chosen['frame']>low+64:
                    rule='LATE_NATIVE_NICK'
                else:
                    rule='EXACT_NICK_ROI' if label['kind']=='EXACT' else 'PREFIX_NICK_CHAT'
                candidates.append(dict(status='PROVEN',actor=actor,seat=route[2],rule_used=rule,
                    anchor=chosen,identity_interval=[segment['start'],segment['end']],
                    exact_match=label['kind']=='EXACT',prefix_match=label['kind']=='PREFIX',
                    previous_occupant=bool(previous),later_native_nick=any(r['frame']>low+64 for r in matches),
                    boundaries=self.boundaries[route]))
        if len({c['seat'] for c in candidates})!=1:
            return dict(status='UNKNOWN',actor=actor,seat=None,
                        reason='NO_LITERAL_NICK_MATCH' if not candidates else 'OBSERVED_MULTIPLE_SEATS')
        # Repeated appearances of the same literal actor at the same physical
        # seat do not create an identity conflict. Keep every interval visible.
        chosen=min(candidates,key=lambda c:(not low<=c['anchor']['frame']<=high,c['anchor']['frame']))
        return dict(chosen,identity_intervals=[c['identity_interval'] for c in candidates])
