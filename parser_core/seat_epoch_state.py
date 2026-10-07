"""Source-ordered seat epochs. Nickname labels never create replacements."""
from collections import defaultdict


class SeatEpochMachine:
    def __init__(self):
        self.events = []
        self.hands = {}
        self.rows = {}
        self.members = {}
        self.audit = []

    def hand(self, table, session, hand_id, frame):
        self.hands[(table, session, hand_id)] = frame

    def observe(self, table, session, seat, kind, frame, *, actor=None,
                hand_id=None, evidence_status='PROVEN', source_ref=None, source_index=100000):
        if kind not in {'OCCUPIED', 'EMPTY', 'SIT_IN', 'SIT_OUT', 'REPLACE'}:
            raise ValueError('SEAT_EVENT_KIND_UNKNOWN')
        event = dict(table=table, session=session, seat=seat, kind=kind, frame=frame,
                     actor=actor, hand_id=hand_id, evidence_status=evidence_status, source_ref=source_ref,
                     source_index=source_index)
        self.events.append(event)

    def build(self):
        self.rows = {}
        self.members = {}
        self.audit = []
        counters = defaultdict(int)
        for e in sorted(self.events, key=lambda e:(e['frame'], e['source_index'], e['table'], str(e['seat']))):
            k = (e['table'], e['session'], e['seat'])
            r = self.rows.setdefault(k, dict(epoch=None, state=None, occupant=None,
                aliases=set(), intervals=[], epoch_start=None, seated_mid_hand=None,last_empty=None))
            if e['evidence_status'] != 'PROVEN':
                self.audit.append(dict(e, status='SEAT_EVIDENCE_UNKNOWN'))
                continue
            kind = e['kind']
            if kind == 'EMPTY':
                if r['epoch'] is not None:
                    r['intervals'].append((r['epoch'],r['epoch_start'],e['frame']))
                r.update(epoch=None,state='EMPTY',occupant=None,aliases=set(),epoch_start=None,seated_mid_hand=None,last_empty=e['frame'])
            elif kind in {'OCCUPIED','REPLACE'}:
                # First positive physical occupancy or explicit confirmed replacement.
                if kind == 'REPLACE' and r['epoch'] is not None:
                    r['intervals'].append((r['epoch'],r['epoch_start'],e['frame']))
                    r['epoch'] = None
                if r['epoch'] is None:
                    counters[k] += 1
                    r.update(epoch=f'T{e["table"]:02d}_{e["seat"].upper()}_EPOCH_{counters[k]}',
                             epoch_start=e['frame'], occupant=e['actor'], aliases=set(),
                             state=None, seated_mid_hand=None)
                    if kind == 'REPLACE' and e['hand_id'] is not None:
                        r.update(state='SEATED_MID_HAND',seated_mid_hand=e['hand_id'])
                    elif e['hand_id'] is not None and r['last_empty'] is not None and r['last_empty'] >= self.hands.get((e['table'],e['session'],e['hand_id']),e['frame']):
                        r.update(state='SEATED_MID_HAND',seated_mid_hand=e['hand_id'])
                if e['actor']:
                    r['aliases'].add(e['actor'])
                if e['hand_id'] is not None:
                    hk = (*k, e['hand_id'],r['epoch'])
                    if r['seated_mid_hand'] == e['hand_id']:
                        self.members[hk] = dict(epoch=r['epoch'],participant=False,blocker='SEATED_MID_HAND')
                    else:
                        if r['state'] == 'SEATED_MID_HAND':
                            r.update(state='ACTIVE',seated_mid_hand=None)
                        self.members.setdefault(hk, dict(epoch=r['epoch'],participant=True,blocker=None))
            elif kind == 'SIT_OUT':
                if r['epoch'] is None:
                    if not e['actor']:
                        self.audit.append(dict(e,status='OCCUPANT_UNKNOWN'))
                        continue
                    counters[k] += 1
                    r.update(epoch=f'T{e["table"]:02d}_{e["seat"].upper()}_EPOCH_{counters[k]}',
                             epoch_start=e['frame'],occupant=e['actor'],aliases={e['actor']})
                r['state'] = 'SITOUT'
            elif kind == 'SIT_IN':
                if r['epoch'] is None:
                    counters[k] += 1
                    r.update(epoch=f'T{e["table"]:02d}_{e["seat"].upper()}_EPOCH_{counters[k]}',
                             epoch_start=e['frame'],occupant=e['actor'],aliases=set())
                    # Joining an empty seat during an already started hand cannot
                    # create retroactive membership in that hand.
                    if e['hand_id'] is not None and e['frame'] > self.hands.get((e['table'],e['session'],e['hand_id']),e['frame']):
                        r.update(state='SEATED_MID_HAND',seated_mid_hand=e['hand_id'])
                    else:
                        r.update(state='ACTIVE',seated_mid_hand=None)
                elif r['state'] == 'SEATED_MID_HAND' and r['seated_mid_hand'] == e['hand_id']:
                    pass
                else:
                    r['state'] = 'ACTIVE'
                if e['actor']:
                    r['aliases'].add(e['actor'])
            self.audit.append(dict(e,status='APPLIED',epoch=r['epoch'],state=r['state']))
        return self

    def membership(self, table, session, seat, hand_id, epoch=None):
        matches = [v for k,v in self.members.items() if k[:4] == (table,session,seat,hand_id)
                   and (epoch is None or k[4] == epoch)]
        return matches[0] if len(matches) == 1 else dict(epoch=None,participant=False,
            blocker='HAND_PARTICIPANT_UNKNOWN' if not matches else 'HAND_EPOCH_AMBIGUOUS')

    def epoch_at(self, table, session, seat, frame):
        r = self.rows.get((table,session,seat))
        if not r:
            return None
        for epoch,start,end in r['intervals']:
            if start <= frame < end:
                return epoch
        return r['epoch'] if r['epoch'] is not None and r['epoch_start'] <= frame else None
