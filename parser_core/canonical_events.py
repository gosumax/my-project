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


def context(row):
    return table_key(row['table_id']), row['session_id'], row.get('hand_id')


def load_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text('utf-8').splitlines() if s.strip()] if Path(path).exists() else []


def join_forced_posts(timeline, summary, headers=()):
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
        elif (rs[0].get('binding_rule') != 'FIRST_EXACT_LABEL_HISTORICAL_EPOCH' or
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
    candidates = defaultdict(list)
    for key,p,receipt in accepted:
        candidates[key].append((p,receipt))
    inserted = defaultdict(list)
    decisions = []
    for key, posts in candidates.items():
        actor_seats = defaultdict(set); seat_actors = defaultdict(set)
        for p,_ in posts:
            actor_seats[p['actor']].add(p['seat']);seat_actors[p['seat']].add(p['actor'])
        for p,receipt in posts:
            kind=ROLES[p['role']]
            matches=[m for m in ordinary[key] if m['player']==p['actor'] and m['action']==kind]
            incompatible=[m for m in ordinary[key] if
                (kind in {'SMALL_BLIND','BIG_BLIND'} and m['action']==kind and m['player']!=p['actor']) or
                (kind in {'SMALL_BLIND','BIG_BLIND'} and m['player']==p['actor'] and m['action'] in {'SMALL_BLIND','BIG_BLIND'} and m['action']!=kind)]
            same_posts=[q for q,_ in posts if q['actor']==p['actor'] and q['role']==p['role']]
            conflict=(bool(incompatible) or len(same_posts)!=1 or len(actor_seats[p['actor']])!=1 or len(seat_actors[p['seat']])!=1 or
                any(m['amount'] is None or str(m['amount'])!=str(p['amount']) or
                    any(r.get('seat') not in {None,p['seat']} for r in m['source_rows']) for m in matches))
            # A repeated same actor/role after voluntary action is not a proven
            # duplicate of the prefix. Keep all raw rows, withhold money authority.
            first_voluntary=next((r for r in groups[key] if parse_text(r['raw_ocr_text'])['action'] in {'FOLD','CHECK','CALL','BET','RAISE','FLOP','TURN','RIVER','WIN'}),None)
            if first_voluntary and any((m['frame_id'],m['source_rows'][0]['row_index']) >=
                                     (first_voluntary['frame_id'],first_voluntary['row_index']) for m in matches):conflict=True
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


def reconcile_stack_facts(facts, events, *, forced_snapshots=()):
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
            if (own and signatures(own)==signatures(bound) and seats=={facts['seat']} and
                    all(p.get('phase')=='BEFORE_DEAL' and p.get('source_frame') is not None and
                        p['source_frame']<=facts.get('deal_complete_frame',-1) for p in bound)):
                result.update(status='PROVEN_FOR_MATH',blocker=None,forced_bets=forced,
                    forced_truth='CANONICAL_COMPLETE_ACTOR_POSTS',
                    forced_bets_debited_before_snapshot_proven=True,
                    correction_proof=dict(actor=actor,rule='COMPLETE_PHYSICAL_ACTOR_POST_SET',
                                          posts=copy.deepcopy(own),event_ids=refs))
    else:
        result['forced_bets']=forced
        result['forced_truth']='CANONICAL_EVENTS'
    return result
