"""Proof-gated source joins shared by timeline, hand analysis and stack math.

Source frame/PTS is observation time. Prefix placement is a domain order,
never a manufactured action timestamp. Raw sidecars are immutable evidence.
"""
from __future__ import annotations
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from parser_core.hand_lifecycle import source_bound_headers, table_key
from parser_core.poker_hand_assembly import parse_text, assemble_messages
from parser_core.seat_forced_binding import parse_forced_snapshot

VERSION = 'canonical-source-join-v1'
ROLES = {'ANTE': 'ANTE', 'SB': 'SMALL_BLIND', 'BB': 'BIG_BLIND'}
FORCED_PREFIX_CAPABILITY = 'AUTHORIZED_FOR_FORCED_PREFIX'


def context(row):
    return table_key(row['table_id']), row['session_id'], row.get('hand_id')


def load_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text('utf-8').splitlines() if s.strip()] if Path(path).exists() else []


def _binding_rule_verified(receipt, snapshot, post):
    if post.get('binding_rule') and post['binding_rule'] != receipt.get('binding_rule'):
        return False
    if 'actor_prefix_proof' in receipt or post.get('binding_rule')=='UNIQUE_LITERAL_PREFIX_ACTOR_IN_COMPLETE_ROSTER_V1':
        from parser_core.nickname_actor_binding import RULE, verify_binding_receipt
        return receipt.get('binding_rule')==RULE and verify_binding_receipt(receipt,snapshot)
    if receipt.get('binding_rule') == 'FIRST_EXACT_LABEL_HISTORICAL_EPOCH':
        # Legacy source-bound exact receipts retain their original schema. A
        # prefix constraint cannot be relabelled as exact to bypass replay.
        return 'roster_proof' not in receipt and all(
            r.get('actor') == receipt.get('actor') and 'constraint' not in r
            for r in receipt.get('label_evidence', []))
    from parser_core.nickname_roster_binding import RULE, verify_binding_receipt
    return receipt.get('binding_rule') == RULE and verify_binding_receipt(receipt, snapshot)


def _ordinary_forced_conflict(posts, ordinary, rows):
    """Same economic conflict checks for legacy posts and atomic candidates."""
    actor_seats = defaultdict(set); seat_actors = defaultdict(set)
    for p, _ in posts:
        actor_seats[p['actor']].add(p['seat']); seat_actors[p['seat']].add(p['actor'])
    first_voluntary = next((r for r in rows if parse_text(r['raw_ocr_text'])['action'] in
        {'FOLD','CHECK','CALL','BET','RAISE','FLOP','TURN','RIVER','WIN'}), None)
    checked = []
    for p, receipt in posts:
        kind = ROLES[p['role']]
        matches = [m for m in ordinary if m['player']==p['actor'] and m['action']==kind]
        incompatible = [m for m in ordinary if
            (kind in {'SMALL_BLIND','BIG_BLIND'} and m['action']==kind and m['player']!=p['actor']) or
            (kind in {'SMALL_BLIND','BIG_BLIND'} and m['player']==p['actor'] and
             m['action'] in {'SMALL_BLIND','BIG_BLIND'} and m['action']!=kind)]
        conflict = (bool(incompatible) or sum(q['actor']==p['actor'] and q['role']==p['role'] for q,_ in posts)!=1
            or len(actor_seats[p['actor']])!=1 or len(seat_actors[p['seat']])!=1
            or any(m['amount'] is None or str(m['amount'])!=str(p['amount']) or
                any(r.get('seat') not in {None,p['seat']} for r in m['source_rows']) for m in matches))
        if first_voluntary and any((m['frame_id'],m['source_rows'][0]['row_index']) >=
                (first_voluntary['frame_id'],first_voluntary['row_index']) for m in matches):
            conflict = True
        checked.append((p, receipt, matches, incompatible, conflict))
    return checked


def prove_candidate_forced_prefix(rows, physical, hand, actor_provider, nick_rows, headers):
    """Prove observed posts only; never table occupancy or historical seat epochs.

    The saved physical enumeration proves the entire CHAT header -> DEAL interval.
    Existing current-hand WHO contracts supply compatible actor/seat identity.
    Their original intervals/boundaries remain unchanged, including late labels.
    """
    try:
        from parser_core.chat_rows import padded_row_span, physical_spans
        from parser_core.nickname_roster_binding import _verify_gray_source, _seat_rois
        key = (table_key(hand['table_id']), hand['session_id'], hand['hand_id'])
        ordered = sorted(rows, key=lambda r:r['index'])
        frame = ordered[0]['source_frame']; pts = ordered[0]['source_pts']; tb = ordered[0]['time_base']
        marker = ordered[0]['marker_source_frame']
        if (physical.get('schema')!='physical-chat-admission-v1' or
                (physical.get('table'),physical.get('session_id'),physical.get('frame_after'))!=(*key[:2],frame)
                or physical.get('source_pts')!=pts or physical.get('time_base')!=tb
                or type(frame) is not int or type(pts) is not int or pts<0
                or len(tb)!=2 or any(type(n) is not int or n<=0 for n in tb)
                or type(marker) is not int or marker>frame
                or not isinstance(physical.get('source_pixel_hash_after'),str)
                or len(physical['source_pixel_hash_after'])!=64
                or [r['index'] for r in ordered]!=list(range(len(physical['new_spans'])))
                or len({r['source_ref'] for r in ordered})!=len(ordered)
                or len({r.get('epoch') for r in ordered})!=1
                or any(r.get('epoch') is not None and r['epoch']!=physical.get('epoch') for r in ordered)):
            return None
        current_headers = sorted([h for h in headers if
            (table_key(h['table_id']),h['session_id'])==key[:2]], key=lambda h:(h['frame_id'],h['row_index']))
        def active(at):
            number = None
            for h in current_headers:
                if h['frame_id']>at:break
                number = parse_text(h['raw_ocr_text'])['hand_number'] if h.get('ocr_status')=='OK' else None
            return number
        if (active(frame)!=key[2] or not any(h['frame_id']==marker and h.get('ocr_status')=='OK'
                and parse_text(h['raw_ocr_text'])['hand_number']==key[2] and h.get('header_provenance')
                for h in current_headers)):
            return None
        packed=[]
        for r in ordered:
            response=r['ocr_response']; pixels=_verify_gray_source(r)
            bounds=list(padded_row_span(physical['new_spans'],r['index'],physical['geometry']['shape'][0]))
            if ((r.get('table_id'),r.get('session_id'),r.get('hand_id'),r.get('source_frame'))!=(*key,frame)
                    or r.get('kind')!='CHAT_CANDIDATE' or r.get('authority')!='CANDIDATE_ONLY'
                    or r.get('admission_rule')!='CONFIRMED_HAND_MARKER_WITHOUT_DEAL_COMPLETE'
                    or r.get('marker_source_frame')!=marker or r.get('source_pts')!=pts or r.get('time_base')!=tb
                    or response.get('status')!='OK' or r.get('row_bounds')!=bounds
                    or pixels.shape!=(bounds[1]-bounds[0],physical['geometry']['shape'][1])
                    or [[a+bounds[0],b+bounds[0]] for a,b in physical_spans(pixels)]!=[physical['new_spans'][r['index']]]):
                return None
            packed.append(dict(r,text=(response.get('raw_text') or '').strip(),score=response.get('score')))
        # Count is a syntax consistency check on a physically bounded CHAT prefix,
        # not evidence of table occupancy or unseen players' ANTE obligations.
        from parser_core.seat_forced_binding import POST
        actors={m[1] for r in packed if (m:=POST.fullmatch(r['text'])) and m[2].lower()=='ante'}
        snapshot, reason = parse_forced_snapshot(packed,key[2],len(actors))
        if snapshot is None:return None
        identities={}; seats=set()
        source_rows=[r for m in hand['messages'] for r in m['source_rows']]
        hand_frames=[r['frame_id'] for r in source_rows]
        for actor in snapshot['actors']:
            who=actor_provider.for_actor(hand,actor)
            anchor=who.get('anchor',{}); seat=who.get('seat')
            if (who.get('status')!='PROVEN' or who.get('actor')!=actor or seat in seats
                    or (table_key(who.get('table_id')),who.get('session_id'),who.get('hand_id'))!=key
                    or (table_key(anchor.get('table_id')),anchor.get('session_id'),anchor.get('hand_id'))!=key
                    or not min(hand_frames)<=anchor.get('frame',-1)<=max(hand_frames)
                    or active(anchor['frame'])!=key[2]):
                return None
            originals=[r for r in nick_rows if (r.get('table_id'),r.get('session_id'),r.get('hand_id'),
                r.get('seat'),r.get('source_frame'))==(*key,seat,anchor['frame'])]
            if len(originals)!=1:return None
            nick=originals[0]; response=nick['ocr_response']; pixels=_verify_gray_source(nick)
            roi=_seat_rois()[f'slot_{key[0]:02d}']['seats'][seat]['nick']
            if (nick.get('roi')!=roi or pixels.shape!=(roi[3]-roi[1],roi[2]-roi[0])
                    or nick['pixel_sha256']!=anchor.get('pixel_sha256') or nick['source_pts']!=anchor.get('source_pts')
                    or nick['time_base']!=anchor.get('time_base') or response.get('status')!='OK'
                    or (response.get('raw_text') or '').strip()!=anchor.get('raw_text')
                    or response.get('score')!=anchor.get('confidence')):
                return None
            # Retain WHO's boundaries; no historical ownership is inferred.
            # Explicit seat/actor changes in this hand cannot support a join.
            low, high=sorted((frame,anchor['frame']))
            for boundary in who.get('boundaries',[]):
                if low<=boundary.get('frame',-1)<=high and boundary.get('kind') in {
                        'EMPTY','OPEN SEAT','OPENSEAT','REPLACE','LEFT','LEAVE','HAS LEFT THE TABLE'}:
                    return None
            for r in source_rows:
                if low<=r['frame_id']<=high:
                    p=parse_text(r['raw_ocr_text'])
                    if p['action'] in {'SIT_IN','SIT_OUT','LEFT_TABLE','JOINED_TABLE'} and p.get('player')==actor:
                        return None
                    text=r['raw_ocr_text'].lower()
                    if actor.lower() in text and ('has left the table' in text or 'has joined the table' in text):
                        return None
            identities[actor]=copy.deepcopy(who);seats.add(seat)
        posts=[]
        for p in snapshot['posts']:
            p=copy.deepcopy(p)
            p.update(table=key[0],session=key[1],hand_id=key[2],seat=identities[p['actor']]['seat'],
                epoch=None,source_frame=frame,source_pts=pts,time_base=tb)
            for r in p['source_rows']:r.pop('png_base64',None)
            posts.append(p)
        return dict(capability=FORCED_PREFIX_CAPABILITY,table=key[0],session=key[1],hand_id=key[2],
            posts=posts,identities=identities,header_index=snapshot['header_index'],deal_index=snapshot['deal_index'],
            physical_source=copy.deepcopy(physical),source_refs=[r['source_ref'] for r in ordered],
            seat_authority='EXISTING_CURRENT_HAND_WHO_ONLY; NO_HISTORICAL_OCCUPANCY',
            occupancy_authority=False,deal_complete_authority=False,start_stack_timing_authority=False)
    except (KeyError,TypeError,ValueError,IndexError,AttributeError,OSError,ZeroDivisionError):
        return None


def join_forced_posts(timeline, summary, headers=(), *, candidate_inputs=()):
    """Join only existing historical actor/seat proofs and strict hand prefixes.

    No name aliases, majority vote or confidence tie-break invents identity.
    Conflicting ordinary observations remain visible but cannot debit a ledger.
    """
    stream = copy.deepcopy(timeline)
    diagnostics = []
    header_groups=defaultdict(list)
    unknown_headers=[]
    for h in headers:
        number=parse_text(h['raw_ocr_text'])['hand_number']
        if h.get('ocr_status')=='OK' and number:
            header_groups[(table_key(h['table_id']),h['session_id'],number)].append(h)
        else:unknown_headers.append(h)
    admitted_headers=[]
    for key,observations in header_groups.items():
        observations.sort(key=lambda r:(r['frame_id'],r['row_index']))
        first=copy.deepcopy(observations[0])
        first['corroborating_header_receipts']=[r['header_provenance'] for r in observations[1:]]
        admitted_headers.append(first)
    # Exact source duplicates corroborate an ordinary header; unknown headers
    # remain barriers. No receipt is allowed to replace a different raw header.
    for header in admitted_headers+unknown_headers:
        same = [r for r in stream if (table_key(r['table_id']), r['session_id'], r['frame_id'], r['row_index']) ==
                (table_key(header['table_id']), header['session_id'], header['frame_id'], header['row_index'])]
        if same and all(r.get('row_pixel_sha256') == header.get('row_pixel_sha256') and
                        parse_text(r['raw_ocr_text'])['hand_number'] == parse_text(header['raw_ocr_text'])['hand_number'] for r in same):
            for r in same:
                r['header_provenance'] = copy.deepcopy(header['header_provenance'])
                r['canonical_stream_version'] = VERSION
        else:
            stream.append(copy.deepcopy(header))
            stream[-1].update(canonical_stream_version=VERSION,
                event_type='HAND_HEADER' if header.get('ocr_status')=='OK' else 'UNKNOWN_EVENT',
                normalized_text=header['raw_ocr_text'].strip(),
                notes_error='SOURCE_BOUND_HEADER_RECEIPT',
                source_roi_change_id=f'{header["session_id"]}:{header["frame_id"]}:header')
    stream.sort(key=lambda r: (str(r['table_id']), r['session_id'], r['frame_id'], r['row_index'], r['stable_row_id']))
    # Physical snapshot/confirmed header anchors belong to this exact context.
    groups = defaultdict(list)
    current = {}
    anchors = {}
    for r in stream:
        ts = (table_key(r['table_id']), r['session_id'])
        p = parse_text(r['raw_ocr_text'])
        if r.get('source_kind') == 'SIDECAR_HEADER_RECEIPT' and r.get('ocr_status') != 'OK':
            current[ts] = None
        elif p['action'] in {'HAND_HEADER','HAND_ID','NEW_HAND'}:
            number = p['hand_number']
            if number is not None:
                current[ts] = number
                key = (*ts, number)
                anchors.setdefault(key, r)
            else:
                current[ts] = None
        key = (*ts, current.get(ts))
        groups[key].append(r)
    receipts = defaultdict(list)
    for receipt in summary.get('historical_binding_receipts', []):
        receipts[(receipt['table'], receipt['session'], receipt['hand_id'], receipt['actor'])].append(receipt)
    snapshots={}
    for snapshot in summary.get('forced_snapshots',[]):
        key=(snapshot['table'],snapshot['session'],snapshot['hand_id'])
        snapshots[key]=snapshot
    accepted = []
    for post in summary.get('bound_posts', []):
        key = (post['table'], post['session'], post['hand_id'])
        rs = receipts.get((*key,post['actor']), [])
        reason = None
        snapshot=snapshots.get(key)
        if not snapshot or not snapshot.get('forced_snapshot_complete') or not any(
            all(post.get(k)==p.get(k) for k in ('actor','role','amount','source_refs','source_frame','source_index')) for p in snapshot['posts']):
            reason = 'COMPLETE_FORCED_SNAPSHOT_PROOF_UNKNOWN'
        elif len(rs) != 1 or rs[0].get('seat') != post.get('seat') or rs[0].get('epoch') != post.get('epoch'):
            reason = 'ACTOR_SEAT_PROOF_UNKNOWN'
        elif (not _binding_rule_verified(rs[0],snapshot,post) or
              rs[0].get('forced_source_frame') != post.get('source_frame') or
              post.get('source_refs') not in rs[0].get('forced_source_refs', [])):
            reason = 'SOURCE_BINDING_PROOF_UNKNOWN'
        elif key not in anchors:
            reason = 'CANONICAL_HAND_ANCHOR_UNKNOWN'
        elif post.get('phase') != 'BEFORE_DEAL' or post.get('role') not in ROLES or not post.get('source_rows'):
            reason = 'FORCED_PREFIX_SOURCE_UNKNOWN'
        elif not isinstance(post.get('amount'), int) or isinstance(post['amount'], bool) or post['amount'] < 0:
            reason = 'FORCED_AMOUNT_INVALID'
        elif any((r.get('table_id'),r.get('session_id'),r.get('hand_id'),r.get('source_frame')) !=
                 (*key,post.get('source_frame')) for r in post['source_rows']):
            reason = 'SOURCE_CONTEXT_CONFLICT'
        elif any(r.get('ocr_response',{}).get('status') != 'OK' or not r.get('pixel_sha256') or
                 r.get('source_ref') != r.get('pixel_sha256') for r in post['source_rows']):
            reason = 'SOURCE_RECEIPT_UNKNOWN'
        else:
            # Receipt context cannot cross a subsequent numbered/unknown header.
            source_position = (post['source_frame'], post['source_index'])
            active = None
            for r in stream:
                if (table_key(r['table_id']),r['session_id']) != key[:2]: continue
                if (r['frame_id'],r['row_index']) > source_position: break
                p = parse_text(r['raw_ocr_text'])
                if r.get('source_kind')=='SIDECAR_HEADER_RECEIPT' and r.get('ocr_status')!='OK': active=None
                elif p['action'] in {'HAND_HEADER','HAND_ID','NEW_HAND'}: active=p['hand_number']
            if active != key[2]: reason = 'SOURCE_HAND_FENCE_CONFLICT'
        if reason:
            diagnostics.append(dict(status='UNRESOLVED',blocker=reason,post=copy.deepcopy(post)))
        else:
            accepted.append((key,post,rs[0]))
    for key,snapshot in snapshots.items():
        for p in snapshot['posts']:
            if not any((b['table'],b['session'],b['hand_id'],b['actor'],b['role'],b['source_refs'])==(*key,p['actor'],p['role'],p['source_refs']) for b in summary.get('bound_posts',[])):
                diagnostics.append(dict(status='UNRESOLVED',blocker='ACTOR_SEAT_BINDING_UNPROVEN',post=dict(p,table=key[0],session=key[1],hand_id=key[2])))
    # Match ordinary multi-row messages with the same parser used downstream.
    ordinary = defaultdict(list)
    for key, rows in groups.items():
        for m in assemble_messages(rows,require_adjacency=True):
            if m['action'] in ROLES.values():
                ordinary[key].append(m)
    candidate_conflicts=set()
    for inputs in candidate_inputs:
        proof=prove_candidate_forced_prefix(**inputs)
        if proof is None:
            diagnostics.append(dict(status='UNRESOLVED',blocker='CANDIDATE_FORCED_PREFIX_AUTHORITY_UNPROVEN'))
            continue
        key=(proof['table'],proof['session'],proof['hand_id'])
        if key in candidate_conflicts:
            diagnostics.append(dict(status='CONFLICT',blocker='CANDIDATE_FORCED_PREFIX_CONFLICT_RETAINED'))
            continue
        posts=[(p,dict(capability=FORCED_PREFIX_CAPABILITY,identity_proof=proof['identities'][p['actor']],
                      forced_prefix_proof=proof)) for p in proof['posts']]
        expected={(p['actor'],ROLES[p['role']],str(p['amount'])) for p,_ in posts}
        checks=_ordinary_forced_conflict(posts,ordinary[key],groups[key])
        negative_forced=any((r.get('canonical_conflict') or r.get('canonical_duplicate_of') or
            r.get('origin')=='UNRESOLVED_IDENTITY' or r.get('ocr_status')=='ERROR') and
            (parse_text(r['raw_ocr_text'])['action'] in ROLES.values() or ' posts ' in r['raw_ocr_text'].lower())
            for r in groups[key])
        if (key not in anchors or negative_forced or any(c[-1] for c in checks) or
                any((m['player'],m['action'],str(m['amount'])) not in expected for m in ordinary[key])):
            candidate_conflicts.add(key)
            accepted=[(k,p,r) for k,p,r in accepted if k!=key or r.get('capability')!=FORCED_PREFIX_CAPABILITY]
            diagnostics.append(dict(status='CONFLICT',blocker='CANDIDATE_FORCED_PREFIX_ATOMIC_CONFLICT'))
            continue
        existing=[(p,r) for k,p,r in accepted if k==key]
        if existing:
            # Never let candidate duplicates suppress prior accepted authority.
            # Legacy summary posts keep their original independent contract.
            if any(r.get('capability')!=FORCED_PREFIX_CAPABILITY for _,r in existing):
                diagnostics.append(dict(status='UNRESOLVED',blocker='EXISTING_SUMMARY_FORCED_AUTHORITY_RETAINED'))
                continue
            signature=lambda ps:{(p['actor'],p['role'],p['amount'],p['seat']) for p,_ in ps}
            if signature(existing)!=signature(posts):
                candidate_conflicts.add(key)
                accepted=[(k,p,r) for k,p,r in accepted if k!=key or r.get('capability')!=FORCED_PREFIX_CAPABILITY]
                diagnostics.append(dict(status='CONFLICT',blocker='CANDIDATE_FORCED_PREFIX_SOURCES_DISAGREE'))
            else:
                diagnostics.append(dict(status='CORROBORATED',capability=FORCED_PREFIX_CAPABILITY,hand_id=key[2]))
            continue
        if all(c[2] for c in checks):
            # An already complete ordinary prefix needs no new source authority.
            diagnostics.append(dict(status='CORROBORATED',capability=FORCED_PREFIX_CAPABILITY,
                                    hand_id=key[2],posts=len(posts)))
            continue
        accepted.extend((key,p,receipt) for p,receipt in posts)
    candidates = defaultdict(list)
    for key,p,receipt in accepted:
        candidates[key].append((p,receipt))
    inserted = defaultdict(list)
    decisions = []
    for key, posts in candidates.items():
        for p,receipt,matches,incompatible,conflict in _ordinary_forced_conflict(posts,ordinary[key],groups[key]):
            kind=ROLES[p['role']]
            if conflict:
                diagnostics.append(dict(status='CONFLICT',blocker='FORCED_SOURCES_DISAGREE_OR_IDENTITY_AMBIGUOUS',post=copy.deepcopy(p),ordinary_messages=copy.deepcopy(matches+incompatible)))
                for m in matches+incompatible:
                    for source in m['source_rows']:
                        r=next(r for r in stream if r['stable_row_id']==source['stable_row_id'])
                        r['canonical_conflict']='FORCED_SOURCES_DISAGREE_OR_IDENTITY_AMBIGUOUS'
                continue
            digest=hashlib.sha256(json.dumps([*key,p['actor'],p['seat'],kind,p['amount'],p['source_refs']],ensure_ascii=False).encode()).hexdigest()
            event_id='CANONICAL_FORCED:'+digest
            provenance=dict(source_lane='FORCED_SNAPSHOT',post=copy.deepcopy(p),binding_receipt=copy.deepcopy(receipt),
                            source_time_semantics='OBSERVED_IN_SNAPSHOT_NOT_ACTION_ONSET',order_rule='PROVEN_BEFORE_DEAL_FORCED_PREFIX')
            if receipt.get('capability')==FORCED_PREFIX_CAPABILITY:
                provenance.update(source_lane='CANDIDATE_FORCED_PREFIX',capability=FORCED_PREFIX_CAPABILITY,
                                  seat_authority='CURRENT_HAND_WHO_ONLY; NO_HISTORICAL_OCCUPANCY')
            if matches:
                # All corroborating ordinary rows retain exact original text.
                # Repeated messages are explicit observations, one money event.
                for i,m in enumerate(matches):
                    for j,source in enumerate(m['source_rows']):
                        r=next(r for r in stream if r['stable_row_id']==source['stable_row_id'])
                        r.update(canonical_event_id=event_id,canonical_provenance=provenance,canonical_stream_version=VERSION,
                                 seat=p['seat'],hand_id=key[2],player=p['actor'],amount=p['amount'],action_type=kind,
                                 binding_status='PROVEN_ACTOR_SEAT_HAND')
                        if i>0:r['canonical_duplicate_of']=matches[0]['source_row_ids'][0]
                decisions.append(dict(status='DEDUPLICATED',canonical_event_id=event_id,post=p,ordinary_refs=[m['source_row_ids'] for m in matches]))
                continue
            raw=' '.join(r['text'] for r in p['source_rows'])
            parsed=parse_text(raw)
            if (parsed['action'],parsed['player'],parsed['amount'])!=(kind,p['actor'],str(p['amount'])):
                diagnostics.append(dict(status='UNRESOLVED',blocker='CANONICAL_PARSER_DISAGREEMENT',post=p));continue
            tb=p.get('time_base');pts=p.get('source_pts')
            stamp=pts*tb[0]/tb[1] if isinstance(tb,list) and len(tb)==2 and tb[1] and pts is not None else None
            r=dict(table_id=f'T{key[0]:02d}',session_id=key[1],hand_id=key[2],stable_row_id=event_id,
                frame_id=p['source_frame'],source_pts=pts,time_base=tb,video_time=stamp,row_index=p['source_index'],
                epoch=p['epoch'],seat=p['seat'],player=p['actor'],amount=p['amount'],action_type=kind,
                raw_ocr_text=raw,normalized_text=raw,event_type=kind,confidence=min(r['score'] for r in p['source_rows']),
                status='BOUND_SOURCE',ocr_status='OK',origin='FORCED_SNAPSHOT',source_lane='FORCED_SNAPSHOT',
                row_pixel_sha256=p['source_refs'][0],source_roi_change_id=f'{key[1]}:{p["source_frame"]}:forced',
                notes_error='SOURCE_OBSERVATION_TIME; PROVEN_BEFORE_DEAL_PREFIX',canonical_event_id=event_id,
                canonical_provenance=provenance,canonical_stream_version=VERSION,binding_status='PROVEN_ACTOR_SEAT_HAND',
                action_onset_timestamp=None)
            if receipt.get('capability')==FORCED_PREFIX_CAPABILITY:
                r.update(origin='CANDIDATE_FORCED_PREFIX',source_lane='CANDIDATE_FORCED_PREFIX',
                         binding_status='PROVEN_OBSERVED_POST_CURRENT_HAND_WHO',
                         forced_prefix_capability=FORCED_PREFIX_CAPABILITY,
                         historical_occupancy_authority=False)
            inserted[anchors[key]['stable_row_id']].append(r)
            decisions.append(dict(status='JOINED',canonical_event_id=event_id,post=p))
    result=[]
    # A physical prefix already observed before DEAL/actions retains its row
    # order. Relocating selected corroborated posts alone splits literal wraps
    # and moves posts ahead of the adjacent Starting/header rows. Domain prefix
    # placement remains necessary for later snapshot observations.
    boundaries={}
    for key,rows in groups.items():
        stop=next((r for r in rows if parse_text(r['raw_ocr_text'])['action'] in
            {'DEAL','FOLD','CHECK','CALL','BET','RAISE','FLOP','TURN','RIVER','WIN'}),None)
        if stop:boundaries[key]=(stop['frame_id'],stop['row_index'])
    def in_physical_prefix(row):
        key=context(row);anchor=anchors[key]
        return (key in boundaries and (anchor['frame_id'],anchor['row_index'])<
                (row['frame_id'],row['row_index'])<boundaries[key])
    located=[]
    for anchor,rows in inserted.items():
        located.extend(r for r in rows if in_physical_prefix(r))
        inserted[anchor]=[r for r in rows if not in_physical_prefix(r)]
    relocated=set()
    for r in stream:
        if (r.get('canonical_event_id') and not r.get('canonical_duplicate_of')
                and not in_physical_prefix(r)):
            key=context(r)
            inserted[anchors[key]['stable_row_id']].append(r)
            relocated.add(r['stable_row_id'])
    ordered=sorted(stream+located,key=lambda r:(str(r['table_id']),r['session_id'],r['frame_id'],r['row_index'],r['stable_row_id']))
    for r in ordered:
        if r['stable_row_id'] in relocated:continue
        result.append(r)
        result.extend(sorted(inserted.get(r['stable_row_id'],[]),key=lambda x:(
            {'ANTE':0,'SMALL_BLIND':1,'BIG_BLIND':2}[x['action_type']],x['canonical_provenance']['post']['source_index'],x['row_index'])))
    for i,r in enumerate(result,1):r['sequence_id']=i
    counts=dict(Counter(d['status'] for d in decisions+diagnostics))
    return result, dict(version=VERSION,counts=counts,decisions=decisions,diagnostics=diagnostics,
                        bound=len(summary.get('bound_posts',[])),source_time_semantics='OBSERVATION_ONLY')


def admit_saved_candidate_forced_prefixes(source, records, actor_provider):
    """Resolve deferred source admission once existing current-hand WHO is ready.

    Reuses canonical projection and the existing hand analyzer. Only successful
    observed-prefix views change; immutable capture/OCR/sidecar files are read.
    """
    source=Path(source)
    if not (source/'physical_chat_admission.jsonl').exists():return []
    hands={(table_key(r['observed_hand']['table_id']),r['observed_hand']['session_id'],
            r['observed_hand']['hand_id']):r for r in records}
    wanted=set()
    for r in records:
        for who in actor_provider.for_hand(r['observed_hand']):
            a=who.get('anchor',{})
            wanted.add((a.get('table_id'),a.get('session_id'),a.get('hand_id'),a.get('seat'),a.get('frame')))
    candidates=defaultdict(list); nicks=[]
    with (source/'seat_forced_inputs.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line)
            key=(row.get('table_id'),row.get('session_id'),row.get('hand_id'))
            if key not in hands:continue
            if row.get('kind')=='CHAT_CANDIDATE':candidates[(*key,row['source_frame'])].append(row)
            elif row.get('kind')=='NICK' and (*key,row.get('seat'),row.get('source_frame')) in wanted:
                nicks.append(row)
    if not candidates:return []
    physical={}
    with (source/'physical_chat_admission.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line); key=(row.get('table'),row.get('session_id'),row.get('frame_after'))
            if any((k[0],k[1],k[3])==key for k in candidates):
                physical.setdefault(key,[]).append(row)
    headers,_,_=source_bound_headers(source)
    reports=[]
    from parser_core.poker_pipeline import analyze_timeline
    for key,record in hands.items():
        hand=record['observed_hand']; inputs=[]
        for route,rows in candidates.items():
            if route[:3]!=key:continue
            receipts=physical.get((route[0],route[1],route[3]),[])
            if len(receipts)==1:
                inputs.append(dict(rows=rows,physical=receipts[0],hand=hand,
                    actor_provider=actor_provider,nick_rows=nicks,headers=headers))
        if not inputs:continue
        timeline=[r for m in hand['messages'] for r in m['source_rows']]
        joined,report=join_forced_posts(timeline,{},candidate_inputs=inputs)
        report.update(table=key[0],session=key[1],hand_id=key[2])
        promoted=[r for r in joined if r.get('forced_prefix_capability')==FORCED_PREFIX_CAPABILITY]
        report['new_observed_posts']=len(promoted)
        reports.append(report)
        if not promoted:continue
        analyzed,_=analyze_timeline(joined,ocr_status={r['stable_row_id']:r.get('ocr_status','OK') for r in joined},
                                    join_show_description=True)
        if len(analyzed)!=1 or context(analyzed[0]['observed_hand'])!=key:
            raise ValueError('CANDIDATE_FORCED_PREFIX_CHANGED_HAND_BOUNDARY')
        for field in ('observed_hand','validation','diagnostics','fully_validated'):
            record[field]=analyzed[0][field]
        record['forced_prefix_admission']=dict(capability=FORCED_PREFIX_CAPABILITY,new_posts=len(promoted))
    return reports


def canonical_join(output, timeline):
    output=Path(output)
    summary_path=output/'seat_forced_summary.json'
    if not summary_path.exists():return timeline, None
    headers,_,_=source_bound_headers(output)
    summary=json.loads(summary_path.read_text('utf-8'))
    if 'forced_snapshots' not in summary:
        # Older saved evidence did not export in-memory snapshots. Replay only
        # the same strict physical parser; never infer missing snapshot rows.
        grouped=defaultdict(list)
        deals={(r['table_id'],r['session_id'],r['hand_id']):r for r in load_jsonl(output/'start_stack_live_receipts.jsonl')
               if r.get('status')=='DEAL_COMPLETE' and r.get('occupied_seats') and r.get('table_pixel_sha256')}
        for r in load_jsonl(output/'seat_forced_inputs.jsonl'):
            if r.get('kind')=='CHAT':
                response=r['ocr_response']
                grouped[(r['table_id'],r['session_id'],r['hand_id'])].append(dict(r,text=response.get('raw_text') or '',score=response.get('score') or 0))
        summary['forced_snapshots']=[]
        for key,rows in grouped.items():
            if key not in deals:continue
            snapshot,_=parse_forced_snapshot(rows,key[2],len(deals[key]['occupied_seats']))
            if snapshot:
                for p in snapshot['posts']:
                    first=p['source_rows'][0]
                    p.update(source_frame=first['source_frame'],source_pts=first['source_pts'],time_base=first['time_base'])
                # strip RAM/base64 from parsed semantic source objects; originals
                # remain in immutable input evidence.
                for p in snapshot['posts']:
                    for r in p['source_rows']:r.pop('png_base64',None)
                summary['forced_snapshots'].append(dict(table=key[0],session=key[1],hand_id=key[2],**snapshot))
    result,report=join_forced_posts(timeline,summary,headers)
    (output/'canonical_join_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
    seen=set();events=[]
    for r in result:
        if r.get('canonical_event_id') and not r.get('canonical_duplicate_of') and r['canonical_event_id'] not in seen:
            seen.add(r['canonical_event_id']);events.append(r)
    with (output/'canonical_forced_events.jsonl').open('w',encoding='utf-8') as f:
        for r in events:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    return result,report


def _first_number_forced_chronology(facts, snapshot, matched, events):
    """Reproduce the complete roster and source order across a callback offset."""
    from fractions import Fraction
    from parser_core.nickname_roster_binding import RULE, verify_binding_receipt
    from parser_core.start_stack_visibility import verify_first_visible_fact
    try:
        if not verify_first_visible_fact(facts):return None
        key=(facts['table'],facts['session'],facts['hand_id'])
        receipts=[r['canonical_provenance']['binding_receipt'] for r in events if context(r)==key
            and not r.get('canonical_conflict') and not r.get('canonical_duplicate_of')
            and r.get('canonical_provenance',{}).get('binding_receipt',{}).get('binding_rule')==RULE]
        if not receipts:return None
        receipt=receipts[0];proof=receipt['roster_proof']
        if any(r['roster_proof']!=proof for r in receipts) or not verify_binding_receipt(receipt,snapshot):return None
        actors={r['player'] for r in matched}
        if (len(actors)!=1 or proof['context']!=list(key) or
                proof['matching'].get(facts['seat'])!=next(iter(actors)) or
                proof['epochs'].get(facts['seat'])!=facts['occupancy_epoch'] or
                proof['binding_anchor_frame']<facts['frames'][-1]):return None
        numeric=facts['evidence'][0]
        first_frame=numeric['source_frame'];first_pts=numeric['source_pts'];tb=numeric['time_base']
        if type(first_frame) is not int or type(first_pts) is not int:return None
        first_time=Fraction(first_pts*tb[0],tb[1])
        source=None
        for post in snapshot['posts']:
            current=(post['source_frame'],post['source_pts'],tuple(post['time_base']))
            if source is not None and current!=source:return None
            source=current
            frame,pts,base=current
            if type(frame) is not int or type(pts) is not int or len(base)!=2 or any(type(v) is not int or v<=0 for v in base):return None
            stamp=Fraction(pts*base[0],base[1])
            if post['phase']!='BEFORE_DEAL' or frame>first_frame or stamp>first_time:return None
            if (frame==first_frame)!=(stamp==first_time):return None
            for row in post['source_rows']:
                if ((row['table_id'],row['session_id'],row['hand_id'])!=key or
                        (row['source_frame'],row['source_pts'],tuple(row['time_base']))!=source):return None
        if source is None:return None
        return dict(rule='COMPLETE_FORCED_PREFIX_OBSERVED_BY_FIRST_POSTFORCED_NUMBER_V1',
            forced_source_frame=source[0],forced_source_pts=source[1],forced_time_base=list(source[2]),
            first_numeric_source_frame=first_frame,first_numeric_source_pts=first_pts,first_numeric_time_base=list(tb),
            roster_proof=copy.deepcopy(proof),first_visible_contract=facts['first_visible_proof']['contract'])
    except (KeyError,TypeError,ValueError,IndexError,ZeroDivisionError):return None


def reconcile_stack_facts(facts, events, *, forced_snapshots=(), first_visible_contract=False,
                          exact_binding_sources=None):
    """Use the same actor/seat money events; retain the original vision object."""
    result=copy.deepcopy(facts)
    key=(facts['table'],facts['session'],facts['hand_id'])
    forced={}
    refs=[]
    matched=[]
    for event in events:
        if context(event)!=key or event.get('seat')!=facts['seat'] or event.get('canonical_conflict') or event.get('canonical_duplicate_of'):continue
        if event['canonical_provenance']['post']['epoch']!=facts.get('occupancy_epoch'):continue
        p=event['canonical_provenance']['post']
        if p['role'] in forced and forced[p['role']]!=p['amount']:
            result.update(status='UNKNOWN',blocker='CANONICAL_FORCED_CONFLICT');return result
        forced[p['role']]=p['amount'];refs.append(event['canonical_event_id']);matched.append(event)
    result['vision_fact']=copy.deepcopy(facts)
    result['canonical_forced_event_ids']=refs
    if forced!=facts.get('forced_bets') or not forced:
        result.update(status='UNKNOWN',blocker='CANONICAL_FORCED_CORRECTIONS_UNPROVEN')
        # Global blind-seat mapping is unnecessary for an exactly bound actor
        # when the complete physical snapshot proves ALL of that actor's posts.
        # An ANTE in the bound subset alone is never such a proof.
        snapshots=[s for s in forced_snapshots if (s['table'],s['session'],s['hand_id'])==key
                   and s.get('forced_snapshot_complete')]
        actors={r.get('player') for r in matched}
        if (facts.get('blocker')=='SPECIAL_FORCED_SEAT_UNKNOWN' and
                facts.get('displayed_stack_status')=='VALID_FIXED_SEAT_ROI' and
                facts.get('participant_proven') and facts.get('occupancy_epoch') and
                isinstance(facts.get('deal_complete_frame'),int) and
                len(snapshots)==1 and len(actors)==1 and None not in actors):
            actor=next(iter(actors));snapshot=snapshots[0]
            own=[p for p in snapshot['posts'] if p['actor']==actor]
            signatures=lambda posts: Counter((p['role'],p['amount'],tuple(p['source_refs'])) for p in posts)
            bound=[r['canonical_provenance']['post'] for r in matched]
            seats={r.get('seat') for r in events if context(r)==key and r.get('player')==actor
                   and not r.get('canonical_conflict') and not r.get('canonical_duplicate_of')}
            chronology=None
            if first_visible_contract:
                chronology=_first_number_forced_chronology(facts,snapshot,matched,events)
                if chronology is None and exact_binding_sources:
                    from parser_core.exact_forced_chronology import prove
                    chronology=prove(facts,snapshot,matched,exact_binding_sources)
            prior_order=all(p.get('phase')=='BEFORE_DEAL' and p.get('source_frame') is not None and
                        p['source_frame']<=facts.get('deal_complete_frame',-1) for p in bound)
            order_proven=chronology is not None if first_visible_contract else prior_order
            if (own and signatures(own)==signatures(bound) and seats=={facts['seat']} and order_proven):
                result.update(status='PROVEN_FOR_MATH',blocker=None,forced_bets=forced,
                    forced_truth='CANONICAL_COMPLETE_ACTOR_POSTS',
                    forced_bets_debited_before_snapshot_proven=True,
                    correction_proof=dict(actor=actor,rule='COMPLETE_PHYSICAL_ACTOR_POST_SET',
                                          posts=copy.deepcopy(own),event_ids=refs))
                if chronology is not None:
                    result.update(actor=actor,forced_chronology_proof=chronology)
                    if chronology['rule']=='EXACT_LABEL_COMPLETE_ACTOR_POSTS_BEFORE_FIRST_NUMBER_V1':
                        result['forced_truth']='CANONICAL_COMPLETE_ACTOR_POSTS_EXACT_LABEL'
                        result['correction_proof']['rule']='COMPLETE_PHYSICAL_ACTOR_POST_SET_EXACT_LABEL'
                    elif chronology['rule']=='UNIQUE_ACTOR_PREFIX_COMPLETE_POSTS_BEFORE_FIRST_NUMBER_V1':
                        result['forced_truth']='CANONICAL_COMPLETE_ACTOR_POSTS_ACTOR_PREFIX'
                        result['correction_proof']['rule']='COMPLETE_PHYSICAL_ACTOR_POST_SET_ACTOR_PREFIX'
    else:
        result['forced_bets']=forced
        result['forced_truth']='CANONICAL_EVENTS'
    return result
