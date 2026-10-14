"""Source-replayed forward balance transfer for saved FULL evidence.

The caller must supply the complete current table registry and native header /
state receipts afresh. A proof's own retained subset is never that registry.
"""
import copy
import base64
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import numpy as np

from parser_core.nickname_roster_binding import RULE as ROSTER_RULE, verify_binding_receipt, prove_roster_binding
from parser_core.poker_reconstruction_layer import proven_start_stacks
from parser_core.poker_reconstruction.assembly import candidate_assemblies
from parser_core.poker_reconstruction.timeline import parse_rows
from parser_core.poker_reconstruction.reconstruct import reconstruct_hand
from parser_core.reconstruction_validation import validate_reconstruction
from parser_core.table_service_events import describe_service
from parser_core.poker_hand_assembly import parse_text

RULE = 'ADJACENT_SAME_ROSTER_PROVEN_END_BALANCE_V1'


def rows(hand):
    return [r for m in hand['messages'] for r in m['source_rows']]


def key(hand):
    return hand['table_id'], hand['session_id'], hand['hand_id']


def file_sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def roster(hand,fresh_proof):
    """Only independently reproduced complete literal roster proofs for now."""
    events={}
    for row in rows(hand):
        if row.get('canonical_conflict') or row.get('canonical_duplicate_of'):
            raise ValueError('CURRENT_CANONICAL_CONFLICT')
        event=row.get('canonical_event_id')
        if not event:continue
        if event in events and events[event]['canonical_provenance']!=row['canonical_provenance']:
            raise ValueError('DUPLICATE_CANONICAL_PROVENANCE_CONFLICT')
        events[event]=row
    if not events:raise ValueError('CURRENT_COMPLETE_ROSTER_UNKNOWN')
    mapping={};snapshots=[];posts=[];verified=set()
    table,session,number=key(hand);table_number=int(str(table).removeprefix('T'))
    roles={'ANTE':'ANTE','SB':'SMALL_BLIND','BB':'BIG_BLIND'}
    # A complete-roster proof establishes every matching edge. Other canonical
    # posts may retain their older exact-label receipt; none may contradict the
    # independently reproduced whole-roster mapping or its source prefix.
    for row in events.values():
        receipt=row['canonical_provenance']['binding_receipt']
        if receipt.get('binding_rule')!=ROSTER_RULE:continue
        proof=receipt['roster_proof'];snapshot=proof['source_inputs']['snapshot']
        if proof!=fresh_proof:raise ValueError('ROSTER_FRESH_SOURCE_INVENTORY_CONFLICT')
        if proof['context']!=[table_number,session,number]:raise ValueError('ROSTER_CONTEXT_CONFLICT')
        signature=digest(receipt)
        if signature not in verified:
            if not verify_binding_receipt(receipt,snapshot):raise ValueError('ROSTER_PROOF_NOT_REPRODUCIBLE')
            verified.add(signature)
        current={a:s for s,a in proof['matching'].items()}
        if mapping and mapping!=current:raise ValueError('CURRENT_ROSTER_PROOFS_DISAGREE')
        mapping=current;snapshots.append(snapshot)
    if not mapping:raise ValueError('COMPLETE_ROSTER_SOURCE_RULE_UNSUPPORTED')
    snapshot=snapshots[0]
    if any(s!=snapshot for s in snapshots):raise ValueError('COMPLETE_PREFIX_SNAPSHOTS_DISAGREE')
    for event,row in events.items():
        provenance=row['canonical_provenance'];post=provenance['post'];receipt=provenance['binding_receipt']
        actor=post['actor'];seat=post['seat'];kind=roles[post['role']]
        expected='CANONICAL_FORCED:'+hashlib.sha256(json.dumps(
            [table_number,session,number,actor,seat,kind,post['amount'],post['source_refs']],ensure_ascii=False).encode()).hexdigest()
        if (event!=expected or receipt['actor']!=actor or receipt['seat']!=seat or mapping.get(actor)!=seat
                or (row['table_id'],row['session_id'],row['hand_id'])!=(table,session,number)
                or (row['player'],row['seat'],row['action_type'],row['amount'])!=(actor,seat,kind,post['amount'])):
            raise ValueError('CURRENT_CANONICAL_BINDING_CONFLICT')
        if not any(all(post.get(k)==p.get(k) for k in ('actor','role','amount','source_refs','source_frame','source_index'))
                   for p in snapshot['posts']):raise ValueError('POST_NOT_IN_REPRODUCED_PREFIX')
        posts.append((actor,post['role'],post['amount'],tuple(post['source_refs'])))
    expected=[(p['actor'],p['role'],p['amount'],tuple(p['source_refs'])) for p in snapshot['posts']]
    if sorted(posts)!=sorted(expected) or len(set(mapping.values()))!=len(mapping):
        raise ValueError('COMPLETE_CANONICAL_PREFIX_NOT_COVERED')
    players={m['player'] for m in hand['messages'] if m.get('player')}
    if players!=set(mapping):raise ValueError('CURRENT_PARTICIPANT_ROSTER_CONFLICT')
    return mapping


def bridge(case,source_registry):
    previous,current=case['previous'],case['current']
    a,b=previous['observed_hand'],current['observed_hand']
    if key(a)[:2]!=key(b)[:2] or key(a)[2]==key(b)[2]:raise ValueError('TABLE_OR_HAND_CONTEXT_CONFLICT')
    registry=[r for r in case['all_table_records'] if key(r['observed_hand'])[:2]==key(a)[:2]]
    keys=[key(r['observed_hand']) for r in registry]
    if len(set(keys))!=len(keys):raise ValueError('AMBIGUOUS_HAND_CONTEXT')
    index=keys.index(key(a))
    if index+1>=len(keys) or keys[index+1]!=key(b):raise ValueError('INTERVENING_HAND_CONTEXT')
    if registry[index]!=previous or registry[index+1]!=current:raise ValueError('CURRENT_SOURCE_REGISTRY_CONFLICT')
    if any(r['lifecycle']['status']!='COMPLETE' for r in (previous,current)):
        raise ValueError('PARTIAL_BOUNDARY')
    if not a.get('next_boundary_observed') or a.get('boundary_conflicts') or b.get('boundary_conflicts'):
        raise ValueError('BOUNDARY_CONTINUITY_UNKNOWN')
    ar,br=rows(a),rows(b)
    if not ar or not br or max(r['frame_id'] for r in ar)>=min(r['frame_id'] for r in br):
        raise ValueError('HAND_SOURCE_TIME_OVERLAP')
    low=min(r['frame_id'] for r in ar);high=min(r['frame_id'] for r in br)
    receipts=[r for r in case['header_receipts'] if r['table_id']==int(str(a['table_id']).removeprefix('T'))
              and r['session_id']==a['session_id'] and low<=r['source_frame']<=high]
    confirmed=set();pending={}
    for r in receipts:
        coordinate=(r['source_frame'],r['source_span_index'])
        if r['status']=='ADMITTED':
            if coordinate in pending:raise ValueError('DUPLICATE_HEADER_ADMISSION')
            pending[coordinate]=r
        elif r['status']=='HAND_HEADER_CONFIRMED':
            number=parse_text(r['ocr_response']['raw_text'])['hand_number']
            expected=a['hand_id'] if r['source_frame']<high else b['hand_id']
            score=r['ocr_response'].get('score')
            if (number!=expected or r.get('current_hand_id')!=expected or r['ocr_response'].get('status')!='OK'
                    or type(score) not in (int,float) or not math.isfinite(score) or not .95<=score<=1):
                raise ValueError('INTERVENING_HEADER_CONFLICT')
            admitted=pending.get(coordinate)
            fields=('table_id','session_id','source_frame','source_span_index','source_pts','time_base','pixel_sha256')
            if not admitted or any(admitted.get(k)!=r.get(k) for k in fields):
                raise ValueError('HEADER_ADMISSION_SOURCE_JOIN_CONFLICT')
            pixels=cv2.imdecode(np.frombuffer(base64.b64decode(admitted['png_base64'],validate=True),np.uint8),0)
            bounds=admitted['row_bounds']
            if (pixels is None or pixels.ndim!=2 or len(bounds)!=2 or
                    any(type(v) is not int for v in bounds) or not 0<=bounds[0]<bounds[1] or
                    pixels.shape[0]!=bounds[1]-bounds[0] or pixels.shape[1]<=0 or
                    hashlib.sha256(pixels.tobytes()).hexdigest()!=admitted['pixel_sha256'] or
                    admitted.get('gate',{}).get('candidate') is not True):
                raise ValueError('HEADER_NATIVE_PIXELS_CONFLICT')
            source_registry.verify_header_source(r,admitted,pixels,require_physical=r['source_frame'] in {low,high})
            states=[s for s in case['state_receipts'] if s.get('status')=='HAND_HEADER_CONFIRMED'
                and all(s.get(k)==r.get(k) for k in ('table_id','session_id','source_frame','source_span_index','source_pts'))
                and s.get('hand_id')==number]
            # The same source header may be confirmed on callback and pending
            # replay. Keep both observations; they prove one matching context.
            if not states:raise ValueError('HEADER_STATE_SOURCE_JOIN_CONFLICT')
            confirmed.add((r['source_frame'],number));pending.pop(coordinate)
        else:raise ValueError('UNRESOLVED_INTERHAND_HEADER')
    if pending or (low,a['hand_id']) not in confirmed or (high,b['hand_id']) not in confirmed:
        raise ValueError('NATIVE_ADJACENT_HEADERS_UNPROVEN')
    for r in case['state_receipts']:
        if (r.get('table_id'),r.get('session_id'))!=(int(str(a['table_id']).removeprefix('T')),a['session_id']):continue
        if not low<=r['source_frame']<=high:continue
        expected=a['hand_id'] if r['source_frame']<high else b['hand_id']
        if r.get('status') not in {'HAND_HEADER_CONFIRMED','STALE_HEADER_IGNORED'} or r.get('hand_id')!=expected:
            raise ValueError('INTERVENING_HAND_STATE_CONFLICT')
        matches=[h for h in receipts if h.get('status')=='ADMITTED' and
                 (h['source_frame'],h['source_span_index'])==(r['source_frame'],r['source_span_index'])]
        if len(matches)!=1 or r.get('source_pts')!=matches[0]['source_pts']:
            raise ValueError('HEADER_STATE_PHYSICAL_SOURCE_CONFLICT')
    for message in a['messages']+b['messages']:
        if message['action']=='UNKNOWN_EVENT':raise ValueError('UNPARSED_PREVIOUS_SOURCE')
        service=describe_service(message['raw_text'])
        if service and service['kind'] in {'JOIN','LEAVE','FINISHED','REBUY'}:
            raise ValueError('ROSTER_OR_BANKROLL_TRANSFER_OBSERVED')
        if message['action']=='TABLE_INFO' and service is None:
            raise ValueError('UNCLASSIFIED_SERVICE_SOURCE')
    old,new=roster(a,source_registry.roster_proof(a)),roster(b,source_registry.roster_proof(b))
    if old!=new:raise ValueError('ROSTER_CHANGED')
    source_registry.check_continuity(a,b,old)
    return old


def independent_stacks(hand,case,stack_provider):
    stacks,receipts,rejected=proven_start_stacks(hand,case['seed_math_facts'],
        first_visible_contract=True,stack_provider=stack_provider)
    if any('CONFLICT' in r.get('reason','') for r in rejected):
        raise ValueError('INDEPENDENT_CANONICAL_STACK_CONFLICT')
    stacks={r['actor']:int(r['START_STACK']) for r in receipts if r.get('START_STACK') is not None
            and r['actor'] in stacks and stack_provider.match_snapshot(r['math_fact'])['status']=='PROVEN'}
    joint=getattr(stack_provider,'joint_sources',None)
    if joint is None:raise ValueError('FRESH_INDEPENDENT_JOINT_SOURCE_INVENTORY_REQUIRED')
    for proof in joint.for_hand(hand):
        actor=proof['actor'];value=proof['calculated_start_stack']
        if actor in stacks and stacks[actor]!=value:raise ValueError('INDEPENDENT_JOINT_STACK_CONFLICT')
        stacks[actor]=value
    constant=getattr(stack_provider,'constant_sources',None)
    if constant is not None:
        for proof in constant.for_hand(hand):
            actor=proof['actor'];value=proof['calculated_start_stack']
            if actor in stacks and stacks[actor]!=value:raise ValueError('INDEPENDENT_CONSTANT_STACK_CONFLICT')
            # Carry still replays and requires its separate complete roster.
            mapping=roster(hand,constant.registry.roster_proof(hand))
            if mapping.get(actor)!=proof['seat']:raise ValueError('INDEPENDENT_CONSTANT_SEAT_CONFLICT')
            stacks[actor]=value
    direct=getattr(stack_provider,'direct_sources',None)
    if direct is not None:
        for proof in direct.for_hand(hand):
            actor=proof['actor'];value=proof['calculated_start_stack']
            if actor in stacks and stacks[actor]!=value:raise ValueError('INDEPENDENT_DIRECT_STACK_CONFLICT')
            mapping=roster(hand,direct.registry.roster_proof(hand))
            if mapping.get(actor)!=proof['seat']:raise ValueError('INDEPENDENT_DIRECT_SEAT_CONFLICT')
            stacks[actor]=value
    return stacks


def prove(case,stack_provider,source_registry,*,depth=0):
    def unknown(reason):return dict(status='UNKNOWN',rule=RULE,reason=reason)
    try:
        if depth>72:return unknown('FORWARD_PROOF_CHAIN_TOO_LONG')
        if not source_registry.matches(case):return unknown('CURRENT_SOURCE_INVENTORY_CONFLICT')
        sitout=getattr(stack_provider,'sitout_sources',None)
        if sitout is not None and any(sitout.excludes_stack_work(key(r['observed_hand']),m.get('player'))
                for r in (case['previous'],case['current']) for m in r['observed_hand']['messages'] if m.get('player')):
            # This route requires every numeric balance. An explicit SitOut
            # exemption never supplies a carried amount or a partial bridge.
            return unknown('SIT_OUT_NUMERIC_BALANCE_NOT_REQUIRED_OR_RECOVERED')
        mapping=bridge(case,source_registry);previous=case['previous']['observed_hand'];current=case['current']['observed_hand']
        if depth==0 and not source_registry.sources_unchanged():return unknown('CARRY_SOURCE_SNAPSHOT_CHANGED')
        if stack_provider.sha256!=case['database_sha256']:return unknown('SOURCE_DATABASE_CHANGED')
        stacks=independent_stacks(previous,case,stack_provider)
        predecessor=case.get('previous_carry')
        if predecessor:
            if predecessor.get('current_hand_key')!=list(key(previous)):
                return unknown('FORWARD_PREDECESSOR_CONTEXT_CONFLICT')
            checked=prove(predecessor['source_case'],stack_provider,source_registry,depth=depth+1)
            if checked!=predecessor or checked['status']!='PROVEN':return unknown('PREDECESSOR_PROOF_INVALID')
            if predecessor['source_case']['current']!=case['previous']:return unknown('PREDECESSOR_CURRENT_SOURCE_CHANGED')
            for actor,value in predecessor['start_stacks'].items():
                if actor in stacks and stacks[actor]!=value:return unknown('INDEPENDENT_START_STACK_CONFLICT')
                stacks[actor]=value
        if set(stacks)!=set(mapping):return unknown('ALL_PREVIOUS_START_STACKS_NOT_PROVEN')
        source=copy.deepcopy(rows(previous))
        for index,row in enumerate(source,1):row['source_index']=index
        assemblies,exhausted,_=candidate_assemblies(source)
        def meanings(event):return replace(event,semantics='UNKNOWN') if event.kind=='CALL' and event.semantics=='DELTA' else event
        result=reconstruct_hand([meanings(e) for e in parse_rows(source)],table=previous['table_id'],
            hand_id=previous['hand_id'],start_stacks=stacks,max_nodes=20000,strict_source_uniqueness=True,
            assemblies=[[meanings(e) for e in assembly] for assembly in assemblies],assembly_exhausted=exhausted)
        if (result['solution_count']!=1 or result['exhausted'] or result['dropped_events'] or result['inferred_events']
                or result.get('chosen_identity_hypothesis',{}).get('aliases') or result.get('pot_delta')!=0
                or result.get('payout_eligible') is not True):return unknown('PREVIOUS_COMPLETE_UNIQUE_LEDGER_UNPROVEN')
        validation=validate_reconstruction(previous,result)
        if validation.get('accepted') is not True:return unknown('PREVIOUS_SOURCE_REVALIDATION_UNPROVEN')
        if set(result['players'])!=set(mapping):return unknown('PREVIOUS_LEDGER_ROSTER_CONFLICT')
        end={actor:stacks[actor]-p['net_commit']+p['payout'] for actor,p in result['players'].items()}
        if any(type(v) is not int or v<=0 for v in end.values()):return unknown('BUST_OR_INVALID_END_BALANCE')
        direct=independent_stacks(current,case,stack_provider)
        if any(end[actor]!=value for actor,value in direct.items() if actor in end):
            return unknown('CURRENT_DIRECT_START_STACK_CONFLICT')
        if depth==0 and not source_registry.sources_unchanged():return unknown('CARRY_SOURCE_SNAPSHOT_CHANGED')
        return dict(status='PROVEN',rule=RULE,reason=None,previous_hand_key=list(key(previous)),
            current_hand_key=list(key(current)),roster=mapping,start_stacks=end,previous_start_stacks=stacks,
            ledger={actor:{k:p[k] for k in ('gross_commit','returned','net_commit','payout')} for actor,p in result['players'].items()},
            independent_current_agreements=direct,database_sha256=stack_provider.sha256,source_hashes=source_registry.hashes,
            source_case=copy.deepcopy(case),authority='FORWARD_END_BALANCE_ONLY; NO_NEW_OCCUPANCY_EPOCH')
    except (KeyError,TypeError,ValueError,IndexError,AttributeError,OverflowError,cv2.error,OSError) as error:
        return unknown(str(error) if isinstance(error,ValueError) else type(error).__name__+':'+str(error))


def verify_current(proof,current_case,stack_provider,source_registry):
    return (proof.get('status')=='PROVEN' and proof.get('source_case')==current_case
            and prove(current_case,stack_provider,source_registry)==proof)


class SavedCarrySources:
    """Build complete current source inventories independently of a saved proof."""
    FILES=('seat_forced_inputs.jsonl','start_stack_live_receipts.jsonl','dealing_cards_triggers.jsonl',
           'seat_forced_summary.json','hand_header_ocr.jsonl','hand_state_receipts.jsonl','ordered_timeline.jsonl','physical_rows.jsonl')

    def __init__(self,source,all_table_records,seed_math_facts):
        self.source=Path(source);self.records=copy.deepcopy(all_table_records)
        self.math_facts=copy.deepcopy(seed_math_facts);self.hashes={};data={}
        contexts={(int(str(r['observed_hand']['table_id']).removeprefix('T')),r['observed_hand']['session_id']) for r in self.records}
        for name in self.FILES:
            raw=(self.source/name).read_bytes();self.hashes[name]=hashlib.sha256(raw).hexdigest()
            data[name]=json.loads(raw) if name.endswith('.json') else [json.loads(line) for line in raw.splitlines() if line.strip()]
        def context(r):return r.get('table_id',r.get('table')),r.get('session_id',r.get('session'))
        self.headers=[r for r in data['hand_header_ocr.jsonl'] if context(r) in contexts]
        self.states=[r for r in data['hand_state_receipts.jsonl'] if context(r) in contexts]
        self.inputs=[r for r in data['seat_forced_inputs.jsonl'] if context(r) in contexts]
        self.deals=[r for r in data['start_stack_live_receipts.jsonl'] if context(r) in contexts and r.get('status')=='DEAL_COMPLETE']
        self.triggers=[r for r in data['dealing_cards_triggers.jsonl'] if context(r) in contexts]
        self.summary=data['seat_forced_summary.json'];self.proofs={}
        self.physical=data['physical_rows.jsonl']
        self.native=[r for r in data['ordered_timeline.jsonl'] if r.get('source') not in {'FORCED_SNAPSHOT','SIDECAR_HEADER'}]
        root=Path(__file__).resolve().parents[1]
        self.admission_clocks=[];clock_hash=hashlib.sha256()
        with (self.source/'physical_chat_admission.jsonl').open('rb') as stream:
            for line in stream:
                clock_hash.update(line)
                if not line.strip():continue
                r=json.loads(line)
                if (r.get('table'),r.get('session_id')) in contexts:
                    self.admission_clocks.append({k:r.get(k) for k in
                        ('table','session_id','frame_after','source_pts','time_base','source_chat_box')})
        self.hashes['physical_chat_admission.jsonl']=clock_hash.hexdigest()
        self.config_path=root/'configs/dealing_start_stack.json'
        raw=self.config_path.read_bytes();self.config_sha256=hashlib.sha256(raw).hexdigest();self.flags=json.loads(raw)

    def matches(self,case):
        expected=self.context_inputs(case['current']['observed_hand'])
        return all(case[name]==value for name,value in expected.items())

    def context_inputs(self,hand):
        context=key(hand)[:2];table=int(str(context[0]).removeprefix('T'))
        return dict(all_table_records=[r for r in self.records if key(r['observed_hand'])[:2]==context],
            header_receipts=[r for r in self.headers if (r['table_id'],r['session_id'])==(table,context[1])],
            state_receipts=[r for r in self.states if (r['table_id'],r['session_id'])==(table,context[1])],
            seed_math_facts=[r for r in self.math_facts if (r['table'],r['session'])==(table,context[1])])

    def roster_proof(self,hand):
        context=(int(str(hand['table_id']).removeprefix('T')),hand['session_id'],hand['hand_id'])
        if context in self.proofs:return self.proofs[context]
        snapshots=[s for s in self.summary['forced_snapshots'] if (s['table'],s['session'],s['hand_id'])==context]
        deals=[r for r in self.deals if (r['table_id'],r['session_id'],r['hand_id'])==context]
        if len(snapshots)!=1 or len(deals)!=1:return dict(status='UNKNOWN',reason='FRESH_PREFIX_OR_DEAL_UNIQUE_SOURCE_UNKNOWN')
        native=[r for r in self.inputs if (r['table_id'],r['session_id'],r['hand_id'])==context]
        forced=[dict(r,text=(r['ocr_response'].get('raw_text') or '').strip(),
                     score=r['ocr_response'].get('score',0) if r['ocr_response'].get('status')=='OK' else 0)
                for r in native if r['kind']=='CHAT']
        nicks=[r for r in native if r['kind']=='NICK' and not r.get('pre_deal')]
        source=dict(context=list(context),snapshot=dict(snapshots[0],table=context[0],session=context[1],hand_id=context[2]),
            deal=deals[0],forced_rows=forced,nick_rows=nicks,
            hand_triggers=[r for r in self.triggers if (r['table_id'],r['session_id'])==context[:2]],
            state_receipts=[r for r in self.summary['state_receipts'] if (r['table'],r['session'])==context[:2]],
            unresolved_events=[r for r in self.summary['unresolved_state_events'] if (r['table'],r['session'])==context[:2]],
            capture_frames=self.flags['capture_frames'],confidence=self.flags['stack_confidence'])
        proof=prove_roster_binding(source);self.proofs[context]=proof
        return proof

    def sources_unchanged(self):
        return (hashlib.sha256(self.config_path.read_bytes()).hexdigest()==self.config_sha256 and
                all(file_sha256(self.source/name)==value for name,value in self.hashes.items()))

    def check_continuity(self,previous,current,mapping):
        context=key(previous)[:2];table=int(str(context[0]).removeprefix('T'))
        rr=rows(previous)+rows(current);low=min(r['frame_id'] for r in rr);high=max(r['frame_id'] for r in rr)
        # Reconcile the caller's logical partition with the immutable native row
        # inventory. A deleted service row cannot disappear from both cases.
        fields=('stable_row_id','table_id','session_id','frame_id','row_index','raw_ocr_text',
                'row_pixel_sha256','source_pts','time_base','video_time','epoch')
        native=[r for r in self.native if (r['table_id'],r['session_id'])==context and low<=r['frame_id']<=high]
        identifiers={r['stable_row_id'] for r in native}
        # Initial carry route requires entirely native hands. A generated row
        # needs its own independent replay contract before it can be admitted.
        if any(r['stable_row_id'] not in identifiers for r in rr):
            raise ValueError('UNSUPPORTED_NON_NATIVE_CARRY_ROW')
        actual=rr
        projection=lambda r:[r.get(f) for f in fields]
        if [projection(r) for r in actual]!=[projection(r) for r in native]:
            raise ValueError('COMPLETE_NATIVE_INTERVAL_COVERAGE_CONFLICT')
        seat_actors={seat:actor for actor,seat in mapping.items()};epochs={}
        for r in self.summary['state_receipts']:
            if (r['table'],r['session'])!=(table,context[1]) or not low<=r['frame']<=high:continue
            if (r.get('status')!='APPLIED' or r.get('evidence_status')!='PROVEN' or
                    r.get('kind') not in {'OCCUPIED','SIT_IN','SIT_OUT'} or r.get('seat') not in seat_actors or
                    r.get('actor') not in (None,seat_actors[r['seat']])):
                raise ValueError('OBSERVED_SEAT_CONTINUITY_BARRIER')
            prior=epochs.setdefault(r['seat'],r.get('epoch'))
            if prior is None or r.get('epoch')!=prior:raise ValueError('OBSERVED_OCCUPANT_EPOCH_CHANGED')
        for r in self.summary['unresolved_state_events']:
            if (r['table'],r['session'])==(table,context[1]) and low<=r['frame']<=high:
                raise ValueError('UNRESOLVED_SEAT_CONTINUITY_BARRIER')

    def verify_header_source(self,header,admitted,pixels,*,require_physical):
        table,session,frame=header['table_id'],header['session_id'],header['source_frame']
        native=[r for r in self.physical if
                (r['table_id'],r['table_session_id'],r['frame_id'])==(table,session,frame)]
        sources=[(r['source_pts'],[r['time_base_num'],r['time_base_den']]) for r in native]
        sources.extend((r['source_pts'],r['time_base']) for r in self.inputs
                       if (r['table_id'],r['session_id'],r['source_frame'])==(table,session,frame))
        clock=[r for r in self.admission_clocks if
               (r['table'],r['session_id'],r['frame_after'])==(table,session,frame)]
        sources.extend((r['source_pts'],r['time_base']) for r in clock)
        for r in clock:
            box=r['source_chat_box'];lo,hi=admitted['row_bounds']
            if not (0<=lo<hi<=box[3]-box[1] and pixels.shape[1]==box[2]-box[0]):
                raise ValueError('HEADER_NATIVE_CHAT_BOUNDS_CONFLICT')
        from parser_core.actor_seat_point import source_time
        value=source_time(header['source_pts'],header['time_base'])
        if not sources or any(source_time(pts,base)!=value for pts,base in sources):
            raise ValueError('HEADER_NATIVE_FRAME_PTS_CONFLICT')
        exact=[r for r in native if r['source_span_index']==header['source_span_index']]
        if not exact and not require_physical:return
        if len(exact)!=1:raise ValueError('HEADER_NATIVE_PHYSICAL_ROW_UNKNOWN')
        row=exact[0];raw=base64.b64decode(row['row_pixels_base64'],validate=True)
        if (row['pixel_dtype']!='uint8' or row['pixel_shape']!=list(pixels.shape) or
                raw!=pixels.tobytes() or hashlib.sha256(raw).hexdigest()!=row['row_pixel_sha256'] or
                admitted['row_bounds']!=[row['y1'],row['y1']+pixels.shape[0]]):
            raise ValueError('HEADER_NATIVE_PHYSICAL_ROW_CONFLICT')


def join_receipts(proof):
    """Derived balance authority has no synthetic observation or seat epoch."""
    identifier=digest(proof)
    return [dict(rule=RULE,proof_id=identifier,actor=actor,seat=proof['roster'][actor],
                 calculated_start_stack=value,previous_hand_key=proof['previous_hand_key'],
                 current_hand_key=proof['current_hand_key'],database_sha256=proof['database_sha256'])
            for actor,value in sorted(proof['start_stacks'].items())]


class AdjacentStackCarryProvider:
    """Forward-only candidate registry containing complete and partial hands.

    Every downstream request reproduces the proof against current retained
    sources. An output JSON proof or its identifier cannot grant authority.
    """
    def __init__(self,source,records,math_facts,stack_provider):
        self.registry=SavedCarrySources(source,records,math_facts)
        self.stack_provider=stack_provider;self.proofs={};self.cases={};self.audit=[]
        tables=defaultdict(list)
        for record in self.registry.records:tables[key(record['observed_hand'])[:2]].append(record)
        for ordered in tables.values():
            for previous,current in zip(ordered,ordered[1:]):
                current_key=key(current['observed_hand']);previous_key=key(previous['observed_hand'])
                case=dict(previous=previous,current=current,
                    **self.registry.context_inputs(current['observed_hand']),database_sha256=stack_provider.sha256)
                if previous_key in self.proofs:case['previous_carry']=self.proofs[previous_key]
                proof=prove(case,stack_provider,self.registry)
                self.audit.append(dict(previous_hand_key=list(previous_key),current_hand_key=list(current_key),
                    status=proof['status'],reason=proof['reason']))
                if proof['status']=='PROVEN':self.proofs[current_key]=proof;self.cases[current_key]=case

    def for_hand(self,hand):
        context=key(hand);case=self.cases.get(context);proof=self.proofs.get(context)
        if not case or not proof or case['current']['observed_hand']!=hand:return []
        if not verify_current(proof,case,self.stack_provider,self.registry):return []
        return [proof]

    def write_audit(self,output):
        output=Path(output)
        with (output/'adjacent_stack_carry_proofs.jsonl').open('w',encoding='utf-8') as stream:
            for proof in self.proofs.values():stream.write(json.dumps(proof,ensure_ascii=False,allow_nan=False)+'\n')
        (output/'adjacent_stack_carry_audit.json').write_text(json.dumps(dict(
            scope='FORWARD_END_BALANCE; COMPLETE_NATIVE_ROSTER_AND_SOURCE_INTERVAL; NO_NEW_EPOCH',
            source_hashes=self.registry.hashes,counts=dict(Counter(r['status'] for r in self.audit)),
            rows=self.audit),ensure_ascii=False,indent=2),'utf-8')
