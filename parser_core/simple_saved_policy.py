"""Production saved-source STACK and WHO adapter, without decode or OCR calls."""
import base64
import hashlib
from .chat_nick_actor import nick_ui_variant, prove_nick_ui_overlay
import copy
import json
from collections import defaultdict
from pathlib import Path
import sqlite3
from functools import lru_cache
from .performance_work import note as work_note
from .stack_timeline_storage import resolve_stack_timeline_path

from .simple_stack_policy import POLICY, corrected_start_stack, first_displayed_stack, forced_allin_prefix_total
from .simple_seat_who import POLICY as WHO_POLICY, SeatIdentityLineage
from .simple_seat_who import nick_constraint, nick_matches, nick_observation_constraint
from .nick_manual_review import select_nickname
from .nickname_policy import NICK_CONFIDENCE_MIN, FIRST_NICK_POLICY

ROOT = Path(__file__).resolve().parents[1]
ROLES = {'ANTE':'ANTE','SMALL_BLIND':'SB','BIG_BLIND':'BB','SB':'SB','BB':'BB'}
SIT_OUT_GATE_AUTHORITY = 'PROVEN_SIT_OUT_FOR_START_STACK_GATE'


def _identity_hand_messages(hand):
    """Post-payout table service cannot revoke participation in the played hand."""
    messages=hand['messages']
    if any(m.get('action') in {'WIN','HAND_END'} for m in messages):
        return [m for m in messages if m.get('action') not in {'TABLE_INFO','BLIND_LEVEL','INFO'}]
    return messages


class HistoricalHandAssociation(dict):
    """Same-video human association, never a global alias or canonical name."""


def confirmed_hand_associations(source, bounds):
    """Rebind registered human same-occupant data to identical hand coordinates."""
    path=ROOT/'configs/confirmed_nickname_associations.json'
    if not path.is_file():return []
    specification=json.loads(path.read_text('utf-8'))
    if specification.get('schema')!='poker.confirmed-hand-associations.v1':
        raise ValueError('CONFIRMED_HAND_ASSOCIATION_SCHEMA')
    metadata=Path(source)/'capture_metadata.json'
    if not metadata.is_file():metadata=Path(source)/'run_metadata.json'
    if not metadata.is_file():return []
    if json.loads(metadata.read_text('utf-8')).get('source_video_sha256')!=specification['source_video_sha256']:
        return []
    original=Path(specification['source_file'])
    if not original.is_absolute():original=ROOT/original
    if hashlib.sha256(original.read_bytes()).hexdigest()!=specification['source_sha256']:
        raise ValueError('CONFIRMED_HAND_ASSOCIATION_HISTORY_CHANGED')
    records=[r for _,r in read_rows(original)]
    if records!=specification['records']:raise ValueError('CONFIRMED_HAND_ASSOCIATION_DATA_CHANGED')
    result=[]
    for row in records:
        key=hand_key(row)
        if row.get('authority')!='USER_EXPLICIT_SAME_OCCUPANT' or key not in bounds:continue
        if tuple(bounds[key])!=(row['frame_start'],row['frame_end']):continue
        result.append(HistoricalHandAssociation(row))
    return result


def hand_key(hand):
    return (int(str(hand.get('table_id',hand.get('table'))).removeprefix('T')),
            hand.get('session_id',hand.get('session')),
            str(hand.get('hand_id',hand.get('hand_number'))))


def saved_hand_bounds(records):
    starts={hand_key(r['observed_hand']):r['lifecycle']['first_source_frame'] for r in records}
    result={}
    for record in records:
        key=hand_key(record['observed_hand'])
        later=[f for k,f in starts.items() if k[:2]==key[:2] and f>starts[key]]
        result[key]=(starts[key],min(later)-1 if later else record['lifecycle']['last_source_frame'])
    return result


def project_economic_actor_aliases(hand,mapping):
    """Carry proven message identity through the economic parser's own grammar.

    Derived aliases belong only to these same hand/message sources. They never
    enter the player database, and a competing observed actor prevents projection.
    """
    if not mapping:return {}
    from .poker_reconstruction.timeline import parse_rows
    targets=defaultdict(set);projections={}
    for message in hand['messages']:
        actor=message.get('player') or message.get('actor')
        target=mapping.get(actor,actor) if actor else None
        if actor:targets[actor].add(target)
        parsed={e.player for e in parse_rows(message.get('source_rows',[])) if e.player}
        for name in parsed:targets[name].add(target)
        if actor in mapping and len(parsed)==1:
            projections[next(iter(parsed))]=target
    result=dict(mapping)
    for name,target in projections.items():
        if targets[name]=={target} and (name not in mapping or mapping[name]==target):
            result[name]=target
    return result


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line, text in enumerate(stream,1):
            if text.strip():
                yield line,json.loads(text)


def saved_stack_observations(source):
    """Retain primary/scaled/raw SQL values, including rejected original OCR.

    An observation's actual frame is used, never last_seen as a synthetic crop.
    Container status/WHO are diagnostics. Saved confidence is retained for
    the simple numeric policy's >= 0.85 admission rule.
    """
    source=Path(source)
    rows=defaultdict(list)
    visibility_verifier=None
    path=source/'start_stack_live_receipts.jsonl'
    for line,raw in read_rows(path):
        if raw.get('stage')!='STACK_OCR':
            continue
        route=(int(raw['table_id']),raw['session_id'],raw['seat'])
        row=dict(table_id=route[0],session_id=route[1],seat=route[2],frame=raw['source_frame'],
            raw_text=raw.get('raw_text'),confidence=raw.get('confidence'),source_order=line*2,
            source_kind='SAVED_DIRECT_PRIMARY',source_file=str(path),source_line=line,
            roi=raw.get('roi'),pixel_sha256=raw.get('pixel_sha256'),source_pts=raw.get('source_pts'),
            time_base=raw.get('time_base'),saved_hand_context=raw.get('hand_id'),
            old_source_status=raw.get('status'),ocr_status=raw.get('ocr_status'),
            item_id=raw.get('ocr_response',{}).get('item_id'))
        scaled=raw.get('ocr_response',{}).get('same_source_scale_evidence',{}).get('response')
        if scaled is not None:
            from .first_stack_ocr import simple_scaled_numeric_evidence,valid_score
            # Existing source/pixel verification is needed only for a scaled
            # candidate that can cross this policy's admission floor.
            admission=None
            if valid_score(scaled.get('score')) and scaled['score']>=.85:
                if visibility_verifier is None:
                    from .start_stack_visibility import VisibilitySource
                    visibility_verifier=VisibilitySource()
                admission=simple_scaled_numeric_evidence(raw,visibility_source=visibility_verifier)
                if admission['primary_conflict']:row['numeric_admission']=admission['scaled']
            rows[route].append(row)
            rows[route].append(dict(row,raw_text=scaled.get('raw_text'),confidence=scaled.get('score'),
                source_kind='SAVED_DIRECT_SCALED',source_order=line*2+1,item_id=scaled.get('item_id'),
                numeric_admission=admission['scaled'] if admission is not None else dict(
                    status='UNKNOWN',reason='SCALED_CONFIDENCE_BELOW_POLICY')))
        else:rows[route].append(row)
    db=resolve_stack_timeline_path(source)
    with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as conn:
        conn.row_factory=sqlite3.Row
        query='''SELECT o.observation_id,o.table_id,o.session_id,o.seat_id,o.frame_index,
            o.last_seen_frame,o.source_pts,o.source_time_base,s.state_id,s.frame_index AS ocr_frame,
            s.ocr_raw,s.ocr_confidence,s.ocr_numeric,s.status,s.exact_hash,s.provenance
            FROM observations o JOIN visual_states s ON s.state_id=o.state_id
            WHERE s.ocr_raw IS NOT NULL ORDER BY o.frame_index,o.observation_id'''
        for index,raw in enumerate(conn.execute(query)):
            route=(int(raw['table_id']),raw['session_id'],raw['seat_id'])
            provenance=json.loads(raw['provenance'])
            rows[route].append(dict(table_id=route[0],session_id=route[1],seat=route[2],
                frame=raw['frame_index'],raw_text=raw['ocr_raw'],confidence=raw['ocr_confidence'],
                source_order=1000000000+index,source_kind='SAVED_SQL_RAW_OCR_OBSERVATION',
                source_file=str(db),observation_id=raw['observation_id'],state_id=raw['state_id'],
                ocr_origin_frame=raw['ocr_frame'],last_seen_frame=raw['last_seen_frame'],
                pixel_sha256=raw['exact_hash'],source_pts=raw['source_pts'],time_base=raw['source_time_base'],
                roi=provenance.get('roi'),old_source_status=raw['status'],original_sql_numeric=raw['ocr_numeric']))
    return rows


def saved_nick_observations(source):
    path=Path(source)/'seat_forced_inputs.jsonl'
    rows=[]
    for line,row in read_rows(path):
        if row.get('kind')!='NICK':
            continue
        response=row.get('ocr_response',{})
        rows.append(dict(table_id=row['table_id'],session_id=row['session_id'],seat=row['seat'],
            hand_id=row['hand_id'],frame=row['source_frame'],raw_text=response.get('raw_text'),
            confidence=response.get('score'),ocr_status=response.get('status'),roi=row['roi'],
            pixel_sha256=row.get('pixel_sha256'),source_pts=row.get('source_pts'),time_base=row.get('time_base'),
            source_file=str(path),source_line=line,item_id=response.get('item_id')))
        # Preserve legacy candidate payloads; retain newly proved UI sources only.
        # Genuine nickname observations keep their existing compact payloads.
        if (nick_ui_variant(response.get('raw_text')) is not None or
                prove_nick_ui_overlay(row) is not None):
            rows[-1]['native_nick_source'] = row
    return rows


def forced_correction(hand, *, actor=None, seat=None, bound_posts=(), no_sb_source=None):
    """Amounts only: missing roles keep START unknown, never erase DISPLAYED."""
    inventory=defaultdict(set)
    sources=[]
    if actor is not None:
        for message in hand['messages']:
            role=ROLES.get(message.get('action'))
            if role and message.get('amount') is not None:
                owner=message.get('player',message.get('actor'))
                amount=int(message['amount'])
                inventory[role].add((owner,amount))
                if owner==actor:
                    sources.append(dict(role=role,amount=amount,source_row_ids=message.get('source_row_ids')))
        owner=actor
    else:
        for post in bound_posts:
            if (int(post['table']),post['session'],str(post['hand_id']))!=hand_key(hand):
                continue
            role=ROLES.get(post['role'])
            if role:
                inventory[role].add((post['seat'],int(post['amount'])))
                if post['seat']==seat:
                    sources.append(dict(role=role,amount=int(post['amount']),source_refs=post.get('source_refs')))
        owner=seat
    amounts={}
    for role in ('ANTE','SB','BB'):
        own={amount for player,amount in inventory[role] if player==owner}
        if len(own)==1 and (role=='ANTE' or len(inventory[role])==1):
            amounts[role]=next(iter(own))
        elif not own and role!='ANTE' and len(inventory[role])==1:
            amounts[role]=0
        else:
            amounts[role]=None
    if not inventory['SB'] and actor is not None and no_sb_source is not None:
        from .legitimate_no_sb import prove_no_sb
        proof=prove_no_sb(hand,no_sb_source)
        if proof is not None and actor in proof['participants']:
            amounts['SB']=0
    return amounts,sources


class LineageActorProvider:
    def __init__(self, lineage, bounds, hands=(), associations=(), manual_reviews=(), *,
                 native_source=None, player_registry=None):
        self.lineage=lineage
        self.bounds=bounds
        self.hands={hand_key(h):h for h in hands}
        from .player_registry import CurrentHandReviewAssociation
        self.associations=[a for a in associations if a.get('authority')!='USER_CONFIRMED_CURRENT_HAND_SEAT' or
                           isinstance(a,CurrentHandReviewAssociation)]
        current_contexts={(a['table_id'],a['session_id'],str(a['hand_id']),a['seat']) for a in self.associations
                          if isinstance(a,CurrentHandReviewAssociation)}
        self.manual_reviews=[r for r in manual_reviews if r.get('authority')!='USER_CONFIRMED_CURRENT_HAND_SEAT' or
            (r.get('table',r.get('table_id')),r.get('session',r.get('session_id')),
             str(r.get('hand',r.get('hand_id'))),r['seat']) in current_contexts]
        self.cache={}
        self.seat_cache={}
        self.actor_maps={}
        self.native_source=Path(native_source).resolve() if native_source is not None else None
        self.player_registry=player_registry
        self.scope_observations={}
        self.scope_labels={}

    def _hand_scope(self, hand, actor, *, aliases=(), seat=None):
        """Lock an associated participant's name until this hand's last event.

        Later NICK OCR cannot rename that participant. Raw observations remain;
        physical CLEAR and competing current-hand players still fence the proof.
        """
        if self.native_source is None:return None
        import cv2
        from .simple_seat_who import CLEAR
        from .native_nick_who import validate_native_nick
        key=hand_key(hand);low,high=self.bounds[key];names=set(aliases) or {actor}
        messages=_identity_hand_messages(hand)
        frames=[r['frame_id'] for m in messages for r in m.get('source_rows',())]
        if not frames:return None
        # The capture window includes the post-WIN tail until the next header.
        # That tail cannot choose a participant anchor or invalidate this hand.
        high=min(high,max(frames))
        if high<low:return None
        actors={m.get('player') or m.get('actor') for m in messages} - {None}
        if key not in self.scope_observations:
            self.scope_observations[key]=sorted([r for r in self.lineage.observations if
                (int(str(r['table_id']).removeprefix('T')),r['session_id'])==key[:2] and
                low<=r['frame']<=high and r.get('ocr_status')=='OK'],key=lambda r:r['frame'])
            self.scope_labels[key]={id(r):nick_observation_constraint(r) for r in self.scope_observations[key]}
        observations=self.scope_observations[key];labels=self.scope_labels[key]
        candidates=[r for r in observations if (seat is None or r['seat']==seat) and
            (label:=labels[id(r)]) is not None and
            any(nick_matches(label,name) for name in names)]
        strong=[r for r in candidates if type(r.get('confidence')) in (int,float) and NICK_CONFIDENCE_MIN<=r['confidence']<=1]
        candidates=strong or (candidates if aliases else [])
        if len({r['seat'] for r in candidates})!=1:return None
        seat=candidates[0]['seat'];route=(*key[:2],seat)
        if any(low<=b['frame']<=high and b['kind'] in CLEAR for b in self.lineage.boundaries[route]):return None
        matching=[];uncertain=[]
        for row in observations:
            label=labels[id(row)]
            if label is None:continue
            score=row.get('confidence');qualified=type(score) in (int,float) and NICK_CONFIDENCE_MIN<=score<=1
            own=any(nick_matches(label,name) for name in names)
            if row['seat']!=seat:
                if own and qualified:return None
                continue
            if own:
                matching.append(row);continue
            # A weak, clipped prefix is not a second player's observed name.
            if (qualified or label['kind']=='EXACT') and any(
                    nick_matches(label,name) for name in actors-names):return None
            uncertain.append(row)
            # A participant cannot acquire another name in this hand, even
            # when the same UI/OCR spelling repeats with high confidence.
        if not matching:return None
        anchor=min(candidates,key=lambda r:r['frame'])
        try:
            path=Path(anchor['source_file']).resolve()
            if path!=self.native_source/'seat_forced_inputs.jsonl':return None
            raw=_capture_record(*_source_version(path),anchor['source_line'])
            validate_native_nick(raw)
            response=raw['ocr_response']
            fields={'table_id':raw['table_id'],'session_id':raw['session_id'],'hand_id':raw['hand_id'],
                'seat':raw['seat'],'frame':raw['source_frame'],'roi':raw['roi'],
                'pixel_sha256':raw['pixel_sha256'],'source_pts':raw['source_pts'],'time_base':raw['time_base'],
                'raw_text':response['raw_text'],'confidence':response['score'],
                'ocr_status':response['status'],'item_id':response['item_id']}
            if any(anchor.get(k)!=v for k,v in fields.items()) or str(raw['hand_id'])!=key[2]:return None
        except (OSError,ValueError,KeyError,TypeError,IndexError,cv2.error):return None
        return dict(status='PROVEN',actor=actor,seat=seat,anchor=anchor,
            rule_used='CURRENT_HAND_NATIVE_NICK_CHAT',identity_interval=[low,high],
            scope_nick_observations=[anchor],current_hand_identity=dict(
                rule='CURRENT_HAND_NATIVE_NICK_CHAT_V1',source=str(self.native_source),
                bounds=[low,high],aliases=sorted(names),uncertain_observations=uncertain,
                participant_nickname_locked=True,nickname_witness_policy=FIRST_NICK_POLICY,
                actual_replacement_crossed=False))

    def _proofs(self,hand):
        key=hand_key(hand)
        if key in self.cache:
            work_note("cache/who_proofs",category="cache/branch",key=key,hits=1)
            return self.cache[key]
        work_note("cache/who_proofs",category="cache/branch",key=key,misses=1)
        hand=self.hands.get(key,hand)
        low,high=self.bounds[key]
        actors={m.get('player') or m.get('actor') for m in hand['messages']}
        actors.discard(None)
        proofs={}
        groups=defaultdict(list)
        for actor in actors:
            proof=self.lineage.for_actor(*key[:2],actor,low,high,actors=actors)
            scoped=self._hand_scope(hand,actor)
            if scoped is not None:proof=scoped
            elif self.native_source is not None:
                proof=dict(status='UNKNOWN',actor=actor,seat=None,reason='CURRENT_HAND_FIRST_QUALIFIED_NICK_UNKNOWN')
            historical=next((a for a in self.associations if isinstance(a,HistoricalHandAssociation) and
                hand_key(a)==key and actor in a['aliases']),None)
            if historical is not None and historical in confirmed_hand_associations(self.native_source,self.bounds):
                scoped=self._hand_scope(hand,actor,aliases=historical['aliases'],seat=historical['seat'])
                if scoped is not None:
                    proof=dict(scoped,association=historical,rule_used='REPROVEN_HUMAN_HAND_ASSOCIATION')
            from .player_registry import CurrentHandReviewAssociation
            current=next((a for a in self.associations if isinstance(a,CurrentHandReviewAssociation) and
                (a['table_id'],a['session_id'],str(a['hand_id']))==key and actor in a['aliases']),None)
            if current is not None:
                proof=dict(status='PROVEN',actor=actor,seat=current['seat'],anchor=current['anchor'],
                    rule_used='USER_CONFIRMED_CURRENT_HAND_SEAT',identity_interval=current['hand_bounds'],
                    native_identity_intervals=current['registry_evidence']['native_intervals'],association=current)
            elif proof['status']!='PROVEN':
                for association in self.associations:
                    if isinstance(association,HistoricalHandAssociation):continue
                    if ((association['table_id'],association['session_id'])!=key[:2] or
                            actor not in association['aliases'] or
                            str(association.get('hand_id'))!=key[2]):
                        continue
                    candidates=[(segment,r) for segment in self.lineage.segments[(*key[:2],association['seat'])]
                        if segment['start']<=high and (segment['end'] is None or segment['end']>=low)
                        for r in segment['observations'] if r['raw_text'] in association['aliases']]
                    if not candidates:
                        continue
                    segment,anchor=min(candidates,key=lambda pair:(not low<=pair[1]['frame']<=high,pair[1]['frame']))
                    proof=dict(status='PROVEN',actor=actor,seat=association['seat'],
                        rule_used=('CONFIRMED_PLAYER_REGISTRY_SAME_OCCUPANT' if
                            association.get('authority')=='USER_CONFIRMED_PLAYER_REGISTRY' else
                            'EXPLICIT_SAME_OCCUPANT_ASSOCIATION'),anchor=anchor,
                        identity_interval=[segment['start'],segment['end']],association=association)
                    break
            proofs[actor]=dict(proof,rule=WHO_POLICY,table_id=key[0],session_id=key[1],hand_id=key[2])
            if proof['status']=='PROVEN':
                groups[proof['seat']].append(actor)
        for seat,members in groups.items():
            declared=next((a for a in self.associations if (a['table_id'],a['session_id'],a['seat'])==(*key[:2],seat)
                           and str(a.get('hand_id'))==key[2]
                           and set(members)<=set(a['aliases'])
                           and (not isinstance(a,HistoricalHandAssociation) or all(
                               proofs[m].get('association')==a and
                               proofs[m].get('rule_used')=='REPROVEN_HUMAN_HAND_ASSOCIATION' for m in members))),None)
            if len(members)>1 and declared is None:
                for actor in members:
                    proofs[actor].update(status='UNKNOWN',seat=None,reason='OBSERVED_MULTIPLE_ACTORS_ONE_SEAT')
                continue
            anchor=proofs[members[0]]['anchor']
            segment=self.lineage.for_seat(*key[:2],seat,anchor['frame'])
            nicks=proofs[members[0]].get('scope_nick_observations') or (
                [r for r in segment['observations'] if low<=r['frame']<=high] if segment else [])
            nicks=nicks or [anchor]
            nick=max(nicks,key=lambda r:r.get('confidence') if isinstance(r.get('confidence'),(int,float)) else -1)
            chats=[]
            for actor in members:
                matching=[m for m in hand['messages'] if (m.get('player') or m.get('actor'))==actor]
                sources=[r for m in matching for r in m.get('source_rows',())]
                strongest=max(sources,key=lambda r:r.get('confidence') if isinstance(r.get('confidence'),(int,float)) else -1,default={})
                chats.append(dict(raw_text=actor,confidence=strongest.get('confidence'),
                    original_raw_text=strongest.get('raw_ocr_text',strongest.get('raw_text')),
                    source_row=strongest,frame=strongest.get('frame_id')))
            manual=next((r for r in self.manual_reviews if
                (r.get('table',r.get('table_id')),r.get('session',r.get('session_id')),str(r.get('hand',r.get('hand_id'))),r['seat'])==(*key,seat)),None)
            anomalous=[r for r in self.lineage.observations if
                (int(str(r['table_id']).removeprefix('T')),r['session_id'],r['seat'])==(*key[:2],seat)
                and low<=r['frame']<=high and nick_observation_constraint(r) is not None
                and not any(nick_matches(nick_observation_constraint(r),a) for a in members)
                and (declared is None or r['raw_text'] not in declared['aliases'])]
            selection=select_nickname(nick,chats,context=dict(table=key[0],session=key[1],hand=key[2]),manual=manual,
                other_nick_observations=[r for r in nicks if r is not nick],
                anomalous_nick_observations=anomalous,
                associated_aliases=declared['aliases'] if declared is not None else ())
            for actor in members:
                proofs[actor].update(canonical_actor=selection['canonical_nick'],selection=selection,
                    status=selection['WHO_status'],aliases=members,needs_manual_review=selection['needs_manual_review'])
        self.cache[key]=proofs
        return proofs

    def for_actor(self,hand,actor):
        proofs=self._proofs(hand)
        if actor in proofs:
            return copy.deepcopy(proofs[actor])
        return copy.deepcopy(next((p for p in proofs.values() if p.get('canonical_actor')==actor),
                                 dict(status='UNKNOWN',actor=actor,seat=None,reason='NO_LITERAL_NICK_MATCH')))

    def for_hand(self,hand):
        byseat={}
        for p in self._proofs(hand).values():
            if p['status']=='PROVEN':
                byseat[p['seat']]=dict(copy.deepcopy(p),actor=p['canonical_actor'])
        return list(byseat.values())

    def for_seat(self,hand,seat):
        existing=next((p for p in self.for_hand(hand) if p['seat']==seat),None)
        if existing:
            return existing
        key=hand_key(hand);low,high=self.bounds[key]
        if self.native_source is not None and any(m.get('action') in {'WIN','HAND_END'} for m in hand['messages']):
            frames=[r['frame_id'] for m in _identity_hand_messages(hand) for r in m.get('source_rows',())]
            if frames:high=min(high,max(frames))
        if (*key,seat) in self.seat_cache:
            work_note("cache/who_seat",category="cache/branch",key=(*key,seat),hits=1)
            return copy.deepcopy(self.seat_cache[(*key,seat)])
        work_note("cache/who_seat",category="cache/branch",key=(*key,seat),misses=1)
        candidates=[(s,r) for s in self.lineage.segments[(*key[:2],seat)]
            if s['start']<=high and (s['end'] is None or s['end']>=low)
            for r in s['observations'] if (nick_observation_constraint(r) or {}).get('kind')=='EXACT']
        manual=next((r for r in self.manual_reviews if
            (r.get('table',r.get('table_id')),r.get('session',r.get('session_id')),str(r.get('hand',r.get('hand_id'))),r['seat'])==(*key,seat)),None)
        if self.native_source is not None and not (manual and manual.get('manual_confirmed')):
            candidates=[(s,r) for s,r in candidates if low<=r['frame']<=high
                and str(r.get('hand_id'))==key[2] and type(r.get('confidence')) in (int,float)
                and NICK_CONFIDENCE_MIN<=r['confidence']<=1 and r.get('ocr_status')=='OK']
        if not candidates:
            return None
        segment,nick=(min(candidates,key=lambda pair:pair[1]['frame']) if self.native_source is not None else
            max(candidates,key=lambda pair:pair[1].get('confidence')
                if isinstance(pair[1].get('confidence'),(int,float)) else -1))
        selection=select_nickname(nick,context=dict(table=key[0],session=key[1],hand=key[2]),
            manual=manual,other_nick_observations=[] if self.native_source is not None else
                [r for s,r in candidates if s is segment and r is not nick],
            anomalous_nick_observations=[] if self.native_source is not None else
                [r for r in segment['ignored'] if low<=r['frame']<=high])
        if self.native_source is None and len({s['label']['text'] for s,_ in candidates})>1:
            selection.update(WHO_status='MANUAL_REVIEW',canonical_nick=None,needs_manual_review=True,
                association_status='MULTIPLE_OCCUPANT_INTERVALS',
                occupant_interval_sources=[dict(start=s['start'],end=s['end'],label=s['label'],source=r)
                                          for s,r in candidates])
        result=dict(status=selection['WHO_status'],actor=selection['canonical_nick'],seat=seat,
            rule=WHO_POLICY,rule_used='STANDALONE_EXACT_NICK_ROI',anchor=nick,selection=selection,
            identity_interval=[segment['start'],segment['end']])
        self.seat_cache[(*key,seat)]=result
        return copy.deepcopy(result)

    def actor_map(self,hand):
        key=hand_key(hand)
        if key not in self.actor_maps:
            mapping={a:p['canonical_actor'] for a,p in self._proofs(hand).items()
                     if p['status']=='PROVEN' and p.get('canonical_actor')}
            self.actor_maps[key]=project_economic_actor_aliases(self.hands.get(key,hand),mapping)
        return dict(self.actor_maps[key])

    def canonical_hand(self,hand):
        result=copy.deepcopy(hand)
        mapping=self.actor_map(hand)
        for message in result['messages']:
            for field in ('player','actor'):
                actor=message.get(field)
                if actor in mapping:
                    message['original_'+field]=actor
                    message[field]=mapping[actor]
        result['nickname_canonicalization']=mapping
        return result

    def review_queue(self):
        unique={}
        for proofs in self.cache.values():
            for p in proofs.values():
                selection=p.get('selection')
                if selection and selection['needs_manual_review']:
                    unique[(selection['table'],selection['session'],selection['hand'],selection['seat'])]=selection
        for p in self.seat_cache.values():
            selection=p['selection']
            if selection['needs_manual_review']:
                unique[(selection['table'],selection['session'],selection['hand'],selection['seat'])]=selection
        return list(unique.values())

    def write_audit(self,output):
        # The parent writes per-hand proofs together with independent seat facts.
        return None


class SimpleSavedPolicyProvider:
    def __init__(self,hands,bounds,observations,nick_observations,*,state_events=(),
                 bound_posts=(),status_observations=(),layout=None,associations=(),manual_reviews=(),
                 candidate_source=None,candidate_records=(),no_sb_review_path=None,player_registry=None):
        self.hands={hand_key(h):h for h in hands}
        self.bounds=bounds
        self.layout=layout or json.loads((ROOT/'configs/stack_roi_refined.json').read_text('utf-8'))['slots']
        self.observations=observations
        self.bound_posts=bound_posts
        self.native_source=Path(candidate_source) if candidate_source is not None else None
        self.lineage=SeatIdentityLineage(nick_observations,state_events)
        self.player_registry=player_registry
        self.registry_rejections=[]
        if player_registry is not None:
            registry_associations,registry_reviews,self.registry_rejections=player_registry.hand_evidence(
                self.lineage,hands,bounds,source=self.native_source)
            associations=[*associations,*registry_associations]
            manual_reviews=[*manual_reviews,*registry_reviews]
        self.actor_sources=LineageActorProvider(self.lineage,bounds,hands,associations,manual_reviews,
            native_source=self.native_source,player_registry=player_registry)
        self.forced_prefix_admission=[]
        self.candidate_records=candidate_records
        if candidate_source is not None:
            from .canonical_events import admit_saved_candidate_forced_prefixes
            self.forced_prefix_admission=admit_saved_candidate_forced_prefixes(
                candidate_source,candidate_records,self.actor_sources)
            hands=[r['observed_hand'] for r in candidate_records]
            self.hands={hand_key(h):h for h in hands}
            self.actor_sources.hands.update(self.hands)
        self.no_sb_source=None
        if self.native_source is not None:
            review_path=Path(no_sb_review_path) if no_sb_review_path is not None else self.native_source/'legitimate_no_sb_reviews.jsonl'
            from .legitimate_no_sb import LegitimateNoSBSource
            self.no_sb_source=LegitimateNoSBSource(self.native_source,self.actor_sources,
                review_path if review_path.is_file() else None)
        self.rows={}
        self.statuses=defaultdict(list)
        for row in status_observations:
            self.statuses[(row['table_id'],row['session_id'],row['seat'])].append(row)
        for key,hand in self.hands.items():
            low,high=bounds[key]
            identities=self.actor_sources.for_hand(hand)
            hand=self.actor_sources.canonical_hand(hand)
            byseat={p['seat']:p for p in identities}
            seat_rows=[]
            for seat,box in self.layout[f'slot_{key[0]:02d}']['seats'].items():
                first=first_displayed_stack(observations.get((*key[:2],seat),()),
                    table_id=key[0],session_id=key[1],seat=seat,hand_start=low,hand_end=high)
                identity=byseat.get(seat) or self.actor_sources.for_seat(hand,seat)
                actor=identity['actor'] if identity else None
                forced,sources=forced_correction(hand,actor=actor,seat=seat,bound_posts=bound_posts,no_sb_source=self.no_sb_source)
                displayed=first['displayed_stack'] if first else None
                status=self.statuses[(*key[:2],seat)]
                row=apply_sitout_gate_status(dict(table_id=key[0],session_id=key[1],hand_id=key[2],seat=seat,
                    player=actor,WHO_status=identity['status'] if identity else 'UNKNOWN',identity_proof=identity,
                    hand_start=low,hand_end=high,fixed_stack_roi=box['stack'],DISPLAYED_STACK=displayed,
                    START_STACK=corrected_start_stack(displayed,forced),forced=forced,forced_sources=sources,
                    first_numeric_source=first,policy=POLICY,player_status=None,sitout_authority=None,
                    sitout_sources=[],sitout_numeric_conflict=False,
                    reason='NO_PARSEABLE_SAVED_NUMERIC' if first is None else None),status,state_events,
                    observations.get((*key[:2],seat),()))
                if self.no_sb_source is not None and actor is not None:
                    from .legitimate_no_sb import prove_no_sb
                    absence=prove_no_sb(hand,self.no_sb_source)
                    if absence is not None and actor in absence['participants']:
                        row['no_sb_authority']=absence
                apply_forced_allin_stack(row,hand,self.native_source)
                seat_rows.append(row)
            self.rows[key]=seat_rows

    @classmethod
    def from_saved(cls,source,records,*,bounds=None,associations=None,manual_reviews=None,no_sb_review_path=None,
                   player_database=None):
        source=Path(source)
        hands=[r['observed_hand'] for r in records]
        if bounds is None:
            bounds=saved_hand_bounds(records)
        observations=saved_stack_observations(source)
        summary=json.loads((source/'seat_forced_summary.json').read_text('utf-8'))
        states=[*summary.get('state_receipts',[])]
        statuses=saved_sitout_observations(source,bounds,observations)
        def optional(name):
            path=source/name
            return [r for _,r in read_rows(path)] if path.exists() else []
        from .player_registry import PlayerRegistry,DEFAULT_DATABASE
        registry=PlayerRegistry(DEFAULT_DATABASE if player_database is None else player_database)
        return cls(hands,bounds,observations,saved_nick_observations(source),state_events=states,
                   bound_posts=summary.get('bound_posts',()),status_observations=statuses,
                   associations=([*optional('nick_occupant_associations.jsonl'),
                                  *confirmed_hand_associations(source,bounds)] if associations is None else associations),
                   manual_reviews=optional('nick_manual_overrides.jsonl') if manual_reviews is None else manual_reviews,
                   candidate_source=source,candidate_records=records,no_sb_review_path=no_sb_review_path,
                   player_registry=registry)

    def seat_rows(self,hand):
        return copy.deepcopy(self.rows[hand_key(hand)])

    def for_hand(self,hand):
        result=[];canonical=self.actor_sources.canonical_hand(hand)
        has_observed_sb=any(m['action']=='SMALL_BLIND' for m in canonical['messages'])
        for r in self.seat_rows(hand):
            if r['player'] is None or r['START_STACK'] is None or r.get('forced_allin_stack_conflict'):continue
            # Absence dependence comes from the source hand, not a removable
            # receipt field. Forced-all-in stacks have their own native proof.
            if (r.get('no_sb_authority') or (not has_observed_sb and r.get('DISPLAYED_STACK') is not None)) and not no_sb_stack_row_valid(r,canonical,self.no_sb_source):continue
            receipt=r.get('forced_allin_stack_receipt')
            if (receipt or r.get('DISPLAYED_STACK') is None or r.get('first_numeric_source') is None) and not forced_allin_row_valid(r,canonical,self.native_source):continue
            result.append(simple_stack_receipt(r))
        return result
    def player_rows(self,hand,*,segment_id=None):
        hand=self.actor_sources.canonical_hand(hand)
        rows=self.seat_rows(hand)
        proven={p['actor']:p for p in self.actor_sources.for_hand(hand)}
        actors=sorted({m.get('player') or m.get('actor') for m in hand['messages']
                       if m.get('player') or m.get('actor')})
        proofs={p['actor']:p for p in self.for_hand(hand)}
        result=[]
        for actor in actors:
            identity=proven.get(actor)
            row=next((r for r in rows if identity and r['seat']==identity['seat']),None)
            p=proofs.get(actor)
            result.append(dict(segment_id=segment_id,table_id=hand['table_id'],session_id=hand['session_id'],
                hand_id=hand['hand_id'],player=actor,seat_id=identity['seat'] if identity else None,
                DISPLAYED_STACK=row['DISPLAYED_STACK'] if row else None,observed_stack=str(row['DISPLAYED_STACK'])
                    if row and row['DISPLAYED_STACK'] is not None else None,
                calculated_start_stack=p['calculated_start_stack'] if p else None,
                status='START_STACK_PROVEN' if p else 'START_STACK_UNKNOWN',
                reason=POLICY if p else row['reason'] or 'FORCED_CORRECTION_UNKNOWN' if row else 'WHO_UNKNOWN',
                player_status=row['player_status'] if row else None,
                forced_amount_added_back=p['forced_amount_added_back'] if p else None,
                voluntary_amount_added_back=0 if p else None,occupancy_epoch=None,
                source_observation_id=None,source_state_id=None,simple_binding_receipt=p))
        return result

    def write_audit(self,output):
        output=Path(output)
        if self.forced_prefix_admission:
            (output/'candidate_forced_prefix_admission.json').write_text(
                json.dumps(self.forced_prefix_admission,ensure_ascii=False,indent=2),encoding='utf-8')
            # Initial logical analysis precedes WHO. Persist the admitted logical
            # view used by reconstruction/HH, without rewriting ordinary rows.
            path=output/'logical_hand_analysis.jsonl'
            if path.exists():
                path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in self.candidate_records),encoding='utf-8')
            complete=output/'complete_hand_analysis.jsonl'
            if complete.exists():
                replacements={r['segment_id']:r for r in self.candidate_records}
                rows=[json.loads(line) for line in complete.read_text('utf-8').splitlines() if line.strip()]
                complete.write_text(''.join(json.dumps(replacements.get(r['segment_id'],r),ensure_ascii=False)+'\n'
                                            for r in rows),encoding='utf-8')
            canonical=output/'canonical_forced_events.jsonl'
            events=[json.loads(line) for line in canonical.read_text('utf-8').splitlines() if line.strip()] if canonical.exists() else []
            seen={r['canonical_event_id'] for r in events}
            for record in self.candidate_records:
                for message in record['observed_hand']['messages']:
                    for row in message['source_rows']:
                        identity=row.get('canonical_event_id')
                        if identity and identity not in seen and not row.get('canonical_duplicate_of'):
                            seen.add(identity);events.append(row)
            canonical.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in events),encoding='utf-8')
            summary=output/'canonical_join_summary.json'
            joined=json.loads(summary.read_text('utf-8')) if summary.exists() else {}
            joined['candidate_forced_prefix_admission']=self.forced_prefix_admission
            summary.write_text(json.dumps(joined,ensure_ascii=False,indent=2),encoding='utf-8')
        for name,rows in [('displayed_stack_by_seat.jsonl',[r for rows in self.rows.values() for r in rows]),
                          ('simple_start_stack_proofs.jsonl',[p for h in self.hands.values() for p in self.for_hand(h)]),
                          ('simple_who_proofs.jsonl',[p for h in self.hands.values() for p in self.actor_sources.for_hand(h)])]:
            (output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        (output/'nick_manual_review_queue.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'
            for r in self.actor_sources.review_queue()),encoding='utf-8')
        if self.player_registry is not None:
            (output/'player_registry_audit.json').write_text(json.dumps(dict(
                database=str(self.player_registry.path),revision=self.player_registry.revision,
                rejected=self.registry_rejections),ensure_ascii=False,indent=2),encoding='utf-8')
        if self.native_source is not None:
            from .nickname_review import write_cases
            write_cases(self,self.native_source,output)


def simple_stack_receipt(r):
    """Keep existing numeric receipts byte-compatible; attach only a proved fallback."""
    result=dict(status='PROVEN',rule=POLICY,actor=r['player'],seat=r['seat'],
        table_id=r['table_id'],session_id=r['session_id'],hand_id=r['hand_id'],
        displayed_stack=r['DISPLAYED_STACK'],calculated_start_stack=r['START_STACK'],
        forced_amount_added_back=(r['START_STACK'] if r.get('forced_allin_stack_receipt') and
            r['DISPLAYED_STACK'] is None else sum(r['forced'].values())),own_forced_posts=r['forced_sources'],
        first_numeric_source=r['first_numeric_source'],identity_proof=r['identity_proof'])
    if r.get('forced_allin_stack_receipt'):
        result['forced_allin_stack_receipt']=copy.deepcopy(r['forced_allin_stack_receipt'])
    if r.get('no_sb_authority'):
        result['no_sb_authority']=copy.deepcopy(r['no_sb_authority'])
    return result


def no_sb_stack_row_valid(row,hand,source):
    """Reprove absent SB; zero correction is never a cached observation."""
    from .legitimate_no_sb import prove_no_sb
    proof=prove_no_sb(hand,source)
    if (proof is None or row.get('no_sb_authority')!=proof or hand_key(row)!=hand_key(hand) or
            row.get('WHO_status')!='PROVEN' or row.get('player') not in proof['participants'] or
            row.get('seat')!=proof['identities'][row['player']]['seat'] or
            row.get('DISPLAYED_STACK') is None or row.get('first_numeric_source') is None):return False
    forced,refs=forced_correction(hand,actor=row['player'],no_sb_source=source)
    return (row.get('forced')==forced and row.get('forced_sources')==refs and
            row.get('START_STACK')==corrected_start_stack(row['DISPLAYED_STACK'],forced))


def forced_allin_row_valid(row,hand,source):
    """Reprove the fallback and bind its amount to the emitted seat fact."""
    if (row.get('WHO_status')!='PROVEN' or hand_key(row)!=hand_key(hand) or
            row.get('DISPLAYED_STACK') is not None or row.get('first_numeric_source') is not None):return False
    proof=prove_forced_allin_start_stack(hand,row.get('player'),row.get('seat'),source,row.get('identity_proof'))
    return (proof is not None and row.get('forced_allin_stack_receipt')==proof and
            row.get('START_STACK')==proof['calculated_start_stack'])


@lru_cache(maxsize=2)
def _forced_allin_native_inventory(versions):
    from .current_chat_sources import load_current_chat
    pins={Path(path).name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path,_,_ in versions}
    return load_current_chat(Path(versions[0][0]).parent,pins)


def prove_forced_allin_start_stack(hand,actor,seat,source,identity=None):
    """Reuse physical CHAT reproof and assembly; grant stack exhaustion only."""
    from .current_chat_sources import FILES, table_key
    from .poker_hand_assembly import assemble_messages
    from .canonical_semantics import PHYSICAL_ALL_IN
    if source is None or not hand.get('start_observed') or hand.get('boundary_conflicts'):return None
    plan=forced_allin_prefix_total(hand['messages'],actor)
    if plan is None:return None
    try:
        key=hand_key(hand);prefix=hand['messages'][:plan['end_index']+1]
        if (not isinstance(identity,dict) or identity.get('status')!='PROVEN' or
                hand_key(identity)!=key or identity.get('seat')!=seat or
                identity.get('canonical_actor',identity.get('actor'))!=actor):return None
        low,high=identity['identity_interval']
        own_frames=[r['frame_id'] for m in prefix if m.get('player')==actor for r in m['source_rows']]
        if any(f<low or high is not None and f>high for f in own_frames):return None
        inventory=_forced_allin_native_inventory(tuple(_source_version(Path(source)/name) for name in FILES))
        native={};owners={}
        for packet in inventory['packets'].values():
            for row in packet['ordinary_prefix_timeline']:
                native[(*table_key(row),row['frame_id'],row['row_index'],row['stable_row_id'])]=row
        for row in inventory['retained_row_ownership']:
            owners[(row['table_id'],row['session_id'],row['frame'],row['row_index'],row['row_id'])]=row
        headers=[r for r in inventory['positive_headers'] if
                 (r['table_id'],r['session_id'],r['hand_id'])==key]
        anchor=next((m for m in prefix if m['action']=='HAND_HEADER' and str(m.get('hand_number'))==key[2]),None)
        if anchor is None:return None
        anchors=[r for r in headers if r['row_id'] in anchor['source_row_ids']]
        if len(anchors)!=1:return None
        anchor=anchors[0];mapping=hand.get('nickname_canonicalization',{})
        checked=[];ids=[];receipts=[]
        for message in prefix:
            rows=[]
            for row in message['source_rows']:
                rid=row.get('stable_row_id');route=(*table_key(row),row['frame_id'],row['row_index'],rid)
                parent=native.get(route);owner=owners.get(route)
                if (parent is None or owner is None or route[:2]!=key[:2] or
                    owner.get('first_proven_hand_id') not in (None,key[2]) or owner.get('hand_id') not in (None,key[2]) or
                    owner.get('hand_id') is None and row['frame_id']!=anchor['frame'] or
                    row.get('canonical_conflict') or row.get('canonical_duplicate_of') or
                    any(row.get(k)!=parent.get(k) for k in
                        ('raw_ocr_text','confidence','row_pixel_sha256','epoch','video_time')) or rid in ids):return None
                rows.append(copy.deepcopy(parent));ids.append(rid)
            parsed=assemble_messages(rows,require_adjacency=True)
            if len(parsed)!=1:return None
            fresh=parsed[0];literal_actor=fresh.get('player')
            fresh['player']=mapping.get(literal_actor,literal_actor)
            if (any(fresh.get(k)!=message.get(k) for k in ('action','player','amount','all_in')) or
                fresh['action'] in {'HAND_HEADER','NEW_HAND','HAND_ID'} and str(fresh.get('hand_number'))!=key[2]):return None
            checked.append(fresh)
            if fresh['player']==actor and fresh['action'] in ROLES:
                receipts.append(dict(role=ROLES[fresh['action']],amount=int(fresh['amount']),
                    source_row_ids=[r['stable_row_id'] for r in rows],source_rows=rows))
        reproduced=forced_allin_prefix_total(checked,actor)
        literal=checked[-1]['raw_text'];match=PHYSICAL_ALL_IN.fullmatch(literal)
        if reproduced!=plan or match is None or mapping.get(match[1],match[1])!=actor:return None
        end_row=checked[-1]['source_rows'][-1]
        anchor_rows=[r for route,r in native.items() if route[:3]==(*key[:2],anchor['frame']) and
                     r['stable_row_id']==anchor['row_id']]
        if len(anchor_rows)!=1:return None
        first_order=(anchor['frame'],anchor_rows[0]['row_index'])
        end_order=(end_row['frame_id'],end_row['row_index'])
        if any((r['table_id'],r['session_id'])==key[:2] and
            first_order[0]<=r['frame']<=end_order[0] for r in inventory['frame_failures']):return None
        physical_prefix=sorted((r for route,r in native.items() if route[:2]==key[:2] and
            first_order<=(r['frame_id'],r['row_index'])<=end_order),key=lambda r:(r['frame_id'],r['row_index']))
        if ids!=[r['stable_row_id'] for r in physical_prefix]:return None
        if any((r['table_id'],r['session_id'])==key[:2] and r['hand_id']!=key[2] and
            anchor['frame']<=r['frame']<=end_order[0] for r in inventory['positive_headers']):return None
        return dict(rule=plan['rule'],table_id=key[0],session_id=key[1],hand_id=key[2],actor=actor,seat=seat,
            calculated_start_stack=plan['calculated_start_stack'],forced_contributions=receipts,
            prefix_source_row_ids=ids,
            header_source=anchor,source=str(Path(source).resolve()),source_sha256=inventory['source_sha256'],
            identity_proof=copy.deepcopy(identity),
            authority='START_STACK_ONLY; OBSERVED_FORCED_POST_EXHAUSTION',
            displayed_stack_authority=False,occupancy_authority=False,hand_boundary_authority=False)
    except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError):return None


def apply_forced_allin_stack(row,hand,source):
    if row.get('WHO_status')!='PROVEN' or row.get('player') is None:return row
    if hand_key(row)!=hand_key(hand):return row
    receipt=prove_forced_allin_start_stack(hand,row['player'],row['seat'],source,row.get('identity_proof'))
    if receipt is None:return row
    if row['DISPLAYED_STACK'] is not None:
        if row['START_STACK']!=receipt['calculated_start_stack']:
            row.update(forced_allin_stack_conflict=True,forced_allin_stack_claim=receipt,
                       reason='NUMERIC_FORCED_ALL_IN_STACK_CONFLICT')
        return row
    row.update(START_STACK=receipt['calculated_start_stack'],forced_allin_stack_receipt=receipt,
        player_status=None,sitout_authority=None,reason=receipt['rule'])
    return row


def apply_sitout_gate_status(row,statuses,state_events=(),numeric_observations=()):
    """Bind observed status to existing WHO; preserve every numeric fact."""
    matches=[r for r in statuses if valid_sitout_status(r,table_id=row['table_id'],
        session_id=row['session_id'],seat=row['seat'],hand_id=row['hand_id'],
        hand_start=row['hand_start'],hand_end=row['hand_end'])]
    if not matches:return row
    conflicts=[r for r in state_events if r.get('evidence_status')=='PROVEN' and
        (r.get('table'),r.get('session'),r.get('seat'))==(row['table_id'],row['session_id'],row['seat']) and
        type(r.get('frame')) is int and row['hand_start']<=r['frame']<=row['hand_end'] and
        r.get('kind') in {'EMPTY','SIT_IN'}]
    numeric=row['DISPLAYED_STACK'] is not None or row['first_numeric_source'] is not None
    state_observations=conflicts
    # A valid numeric fact owns this lane. Later/earlier seat transitions cannot
    # invalidate it merely because this hand also contains a SIT_OUT display.
    # Keep those transitions as provenance; simultaneous status/state claims
    # still enter the existing conflict policy.
    if numeric:conflicts=[r for r in conflicts if any(r['frame']==s['frame'] for s in matches)]
    # Sequential number/status displays are compatible. A contradiction needs
    # an admitted numeric reading of the same physical observation clock.
    numeric_sources=[r for r in numeric_observations if first_displayed_stack([r],
        table_id=row['table_id'],session_id=row['session_id'],seat=row['seat'],
        hand_start=row['hand_start'],hand_end=row['hand_end']) is not None]
    if row['first_numeric_source'] is not None:numeric_sources.append(row['first_numeric_source'])
    numeric_conflict=any(n.get('frame')==s['frame'] for n in numeric_sources for s in matches)
    bound=row.get('WHO_status')=='PROVEN' and row.get('player') is not None
    eligible=bound and not numeric and not conflicts
    row.update(sitout_sources=matches,sitout_numeric_conflict=numeric_conflict,sitout_state_conflicts=conflicts,
        sitout_state_observations=state_observations,
        player_status='SIT_OUT' if eligible else None,
        sitout_authority=SIT_OUT_GATE_AUTHORITY if eligible else None,
        reason='SIT_OUT_NUMERIC_CONFLICT' if numeric_conflict else 'SIT_OUT_STATE_CONFLICT' if conflicts else
               'NUMERIC_STACK_WITH_SIT_OUT_OBSERVATIONS' if numeric else
               'NORMAL_SIT_OUT_WITHOUT_NUMERIC' if bound else 'SIT_OUT_ACTOR_SEAT_UNPROVEN')
    return row


def _source_version(path):
    path=Path(path).resolve();stat=path.stat()
    return str(path),stat.st_mtime_ns,stat.st_size


@lru_cache(maxsize=4)
def _capture_offsets(path,mtime,size):
    offsets=[]
    with open(path,'rb') as stream:
        while True:
            offset=stream.tell()
            if not stream.readline():break
            offsets.append(offset)
    return offsets


@lru_cache(maxsize=256)
def _capture_record(path,mtime,size,line):
    if type(line) is not int or line<1:raise ValueError('SIT_OUT_SOURCE_LINE_INVALID')
    with open(path,'rb') as stream:
        stream.seek(_capture_offsets(path,mtime,size)[line-1])
        return json.loads(stream.readline())


@lru_cache(maxsize=4)
def _sql_status_records(path,mtime,size):
    with sqlite3.connect(Path(path).as_uri()+'?mode=ro',uri=True) as conn:
        conn.row_factory=sqlite3.Row
        query='''SELECT s.table_id AS state_table,s.seat_id AS state_seat,
            s.frame_index AS state_frame,s.raw_gray,s.height,s.width,s.state_id,
            s.exact_hash,s.ocr_raw,s.ocr_confidence,s.ocr_numeric,s.status,
            o.table_id,o.seat_id,o.frame_index,o.observation_id,o.session_id,
            o.source_pts,o.source_time_base
            FROM visual_states s JOIN observations o ON o.state_id=s.state_id
            ORDER BY o.frame_index,o.observation_id'''
        return {r['observation_id']:dict(r) for r in conn.execute(query)}


def _sitout_native_source(kind,path,*,line=None,observation_id=None,capture_record=None):
    """Reload the physical row, verify native hash/geometry, retain its clock."""
    import cv2
    import numpy as np
    layout=json.loads((ROOT/'configs/stack_roi_refined.json').read_text('utf-8'))['slots']
    version=_source_version(path)
    if kind=='SAVED_STACK_CAPTURE':
        # Discovery already read this physical line. Consumers always reload
        # its locator independently; no embedded receipt can supply these pixels.
        row=_capture_record(*version,line) if capture_record is None else capture_record
        route=(row['table_id'],row['session_id'],row['seat']);frame=row['source_frame']
        if row['field']!='stack':raise ValueError('SIT_OUT_SOURCE_FIELD_INVALID')
        buffer=np.frombuffer(base64.b64decode(row['png_base64'],validate=True),np.uint8)
        if not buffer.size:raise ValueError('SIT_OUT_CAPTURE_PNG_EMPTY')
        try:
            pixels=cv2.imdecode(buffer,cv2.IMREAD_UNCHANGED)
        except cv2.error as exc:
            raise ValueError('SIT_OUT_CAPTURE_PNG_INVALID') from exc
        x,y,X,Y=row['roi'];a,b,A,B=layout[f'slot_{route[0]:02d}']['seats'][route[2]]['stack']
        if (pixels is None or pixels.dtype!=np.uint8 or pixels.shape!=(Y-y,X-x,3) or
                not x<=a<A<=X or not y<=b<B<=Y or
                hashlib.sha256(pixels[:,:,::-1].tobytes()).hexdigest()!=row['pixel_sha256']):
            raise ValueError('SIT_OUT_CAPTURE_PIXELS_OR_ROI_CONFLICT')
        gray=cv2.cvtColor(pixels[b-y:B-y,a-x:A-x],cv2.COLOR_BGR2GRAY)
        ref=dict(kind=kind,file=version[0],line=line,parent_pixel_sha256=row['pixel_sha256'],
            pixel_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),roi=list(row['roi']),stack_roi=[a,b,A,B],
            source_frame=frame,source_pts=row.get('source_pts'),time_base=row.get('time_base'),
            saved_hand_context=row.get('hand_id'))
    elif kind=='SAVED_SQL_VISUAL_STATE':
        row=_sql_status_records(*version)[observation_id]
        route=(int(row['table_id']),row['session_id'],row['seat_id']);frame=row['frame_index']
        a,b,A,B=layout[f'slot_{route[0]:02d}']['seats'][route[2]]['stack']
        if (row['state_table']!=row['table_id'] or row['state_seat']!=row['seat_id'] or
                row['state_frame']>frame or row['width']!=A-a or row['height']!=B-b):
            raise ValueError('SIT_OUT_SQL_ROUTE_OR_GEOMETRY_CONFLICT')
        gray=np.frombuffer(row['raw_gray'],np.uint8).reshape(row['height'],row['width'])
        if hashlib.sha256(gray.tobytes()).hexdigest()!=row['exact_hash']:
            raise ValueError('SIT_OUT_SQL_NATIVE_HASH_CONFLICT')
        ref=dict(kind=kind,file=version[0],state_id=row['state_id'],observation_id=observation_id,
            state_origin_frame=row['state_frame'],source_frame=frame,source_pts=row['source_pts'],
            time_base=json.loads(row['source_time_base']),pixel_sha256=row['exact_hash'],stack_roi=[a,b,A,B],
            raw_text=row['ocr_raw'],confidence=row['ocr_confidence'],ocr_numeric=row['ocr_numeric'],old_status=row['status'])
    else:
        raise ValueError('SIT_OUT_SOURCE_KIND_INVALID')
    if (type(frame) is not int or not route[1] or type(ref['source_pts']) is not int or
            not isinstance(ref['time_base'],list) or len(ref['time_base'])!=2 or
            any(type(v) is not int or v<=0 for v in ref['time_base'])):
        raise ValueError('SIT_OUT_SOURCE_CLOCK_INVALID')
    return route,frame,gray,ref


def valid_sitout_status(row, *, table_id, session_id, seat, hand_id=None, hand_start=None, hand_end=None):
    """Reprove native pixels and the exact classifier receipt at every consumer."""
    from .sitout_text_pattern import SitOutTextPattern
    try:
        if (not isinstance(row,dict) or row.get('status')!='SIT_OUT' or
                row.get('authority')!=SIT_OUT_GATE_AUTHORITY or
                (row.get('table_id'),row.get('session_id'),row.get('seat'))!=(table_id,session_id,seat) or
                not isinstance(row.get('hand_id'),str) or type(row.get('frame')) is not int):return False
        low,high=row['hand_window']
        if (type(low) is not int or type(high) is not int or not low<=row['frame']<=high or
                hand_id is not None and row['hand_id']!=hand_id or
                hand_start is not None and (low,high)!=(hand_start,hand_end)):return False
        ref=row['source']
        route,frame,gray,fresh=_sitout_native_source(ref['kind'],ref['file'],line=ref.get('line'),
                                                  observation_id=ref.get('observation_id'))
        classifier=SitOutTextPattern().evaluate(gray)
        return (route==(table_id,session_id,seat) and frame==row['frame'] and fresh==ref and
                fresh.get('saved_hand_context') in (None,row['hand_id']) and
                classifier['matched'] and classifier==row['status_classifier'])
    except (OSError,ValueError,KeyError,TypeError,IndexError,sqlite3.Error):
        return False


def saved_sitout_observations(source,bounds,observations):
    """Classify current physical observations, including numeric/status conflicts."""
    import cv2
    import numpy as np
    from .sitout_text_pattern import SitOutTextPattern
    layout=json.loads((ROOT/'configs/stack_roi_refined.json').read_text('utf-8'))['slots']
    pending=defaultdict(list)
    statuses=[]
    for key,(low,high) in bounds.items():
        for seat in layout[f'slot_{key[0]:02d}']['seats']:
            route=(*key[:2],seat)
            pending[route].append((key[2],low,high))
    matcher=SitOutTextPattern()
    def add(route,frame,gray,ref):
        for hand_id,low,high in pending.get(route,()):
            if not low<=frame<=high or ref.get('saved_hand_context') not in (None,hand_id):
                continue
            digest=hashlib.sha256(gray.tobytes()).hexdigest()
            if digest != ref.get('pixel_sha256'):
                raise ValueError('SIT_OUT_SOURCE_PIXEL_HASH_CONFLICT')
            classifier=matcher.evaluate(gray)
            if classifier['matched']:
                statuses.append(dict(table_id=route[0],session_id=route[1],hand_id=hand_id,
                    seat=route[2],frame=frame,status='SIT_OUT',authority=SIT_OUT_GATE_AUTHORITY,
                    hand_window=[low,high],source=ref,status_classifier=classifier))
    path=Path(source)/'start_stack_capture.jsonl'
    for line,row in read_rows(path):
        route=(row.get('table_id'),row.get('session_id'),row.get('seat'))
        frame=row.get('source_frame')
        if row.get('field')!='stack' or not any(low<=frame<=high for _,low,high in pending.get(route,())):
            continue
        route,frame,gray,ref=_sitout_native_source('SAVED_STACK_CAPTURE',path,line=line,capture_record=row)
        add(route,frame,gray,ref)
    db=resolve_stack_timeline_path(source)
    for row in _sql_status_records(*_source_version(db)).values():
        route=(int(row['table_id']),row['session_id'],row['seat_id']);frame=row['frame_index']
        if any(low<=frame<=high for _,low,high in pending.get(route,())):
            route,frame,gray,ref=_sitout_native_source('SAVED_SQL_VISUAL_STATE',db,observation_id=row['observation_id'])
            add(route,frame,gray,ref)
    return statuses
