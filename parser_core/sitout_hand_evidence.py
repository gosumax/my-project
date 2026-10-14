"""Source-bound SitOut status; never a numeric stack or an invented fold."""
import base64
import copy
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from .actor_seat_point import prove_actor_point, sha, source_time, unique
from .adjacent_stack_carry import SavedCarrySources
from .constant_stack_balance import (FreshConstantSources, SourceSealBatch, complete_hand_interval,
    frozen_runtime_files, independent_ledger, native_hand_bounds, numeric_sources, direct_readings)
from .joint_start_stack import own_posts, FORCED_PREFIX_OCR_POLICY
from .sitout_text_pattern import SitOutTextPattern
from .start_stack_visibility import (VisibilitySource, ForcedOverlayPatterns,
    OVERLAY_NCC_MIN, BLANK_GRAY_RANGE_MAX)

ROOT=Path(__file__).resolve().parents[1]
from parser_core.chat_nick_actor import prove_actor_point, numeric_nickname_sources, outer_rule, point_matches_hand

RULE='SOURCE_BOUND_SIT_OUT_WITH_OBSERVED_FORCED_AND_FOLD_V1'
HH_CONTRACT='USER_20261008_SIT_OUT_NUMERIC_STACK_NOT_REQUIRED'


def native_key(hand):
    return (int(str(hand['table_id']).removeprefix('T')),hand['session_id'],hand['hand_id'])


def case_for(inventory,context,actor,seat):
    obs=inventory.observations.get((*context,seat),[])
    return copy.deepcopy(dict(actor=actor,seat=seat,table_id=context[0],session_id=context[1],hand_id=context[2],
        source_context=inventory.contexts[context],full_hand_source_context=inventory.full_contexts[context],
        raw_stack_captures=inventory.captures[(*context,seat)],raw_direct_stack_readings=inventory.direct[(*context,seat)],
        raw_sql_observations=obs,raw_sql_visual_states=[inventory.states[s] for s in sorted({o['state_id'] for o in obs})],
        database_sha256=inventory.database_sha256,constant_interval=list(inventory.bounds[context])))


def source_image(case,frame):
    caps=unique([w['raw_record'] for w in case['raw_stack_captures']],lambda r:r['source_frame'])
    row=caps[frame];context=case['source_context']
    if (any(row.get(k)!=case[k] for k in ('table_id','session_id','seat')) or row.get('field')!='stack' or
            not context['header_frame']<=frame<=context['capture_end_inclusive']):
        raise ValueError('SIT_OUT_INITIAL_CAPTURE_CONTEXT_CONFLICT')
    nicks=unique([r for r in context['native_capture_nick_rows'] if r['seat']==case['seat']],lambda r:r['source_frame'])
    nick=nicks[frame]
    if source_time(row['source_pts'],row['time_base'])!=source_time(nick['source_pts'],nick['time_base']):
        raise ValueError('SIT_OUT_NICK_STACK_SOURCE_TIME_CONFLICT')
    rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(row['png_base64'],validate=True),np.uint8),1),cv2.COLOR_BGR2RGB)
    x,y,X,Y=row['roi'];verifier=VisibilitySource()
    a,b,A,B=verifier.rois[f"slot_{case['table_id']:02d}"]['seats'][case['seat']]['stack']
    if (rgb.shape!=(Y-y,X-x,3) or sha(rgb.tobytes())!=row['pixel_sha256'] or
            not x<=a<A<=X or not y<=b<B<=Y):
        raise ValueError('SIT_OUT_NATIVE_PIXELS_OR_ROI_CONFLICT')
    gray=cv2.cvtColor(rgb[b-y:B-y,a-x:A-x],cv2.COLOR_RGB2GRAY)
    return row,gray


def status_numeric_conflicts(case,inventory,last_frame):
    """Existing accepted numbers can contradict status; never recover a stack."""
    low=case['source_context']['header_frame']
    window=copy.deepcopy(case)
    window['raw_stack_captures']=[w for w in case['raw_stack_captures']
        if low<=w['raw_record']['source_frame']<=last_frame]
    window['raw_direct_stack_readings']=[r for r in case['raw_direct_stack_readings']
        if low<=r['source_frame']<=last_frame]
    _,bindings,_,_=numeric_sources(window,inventory.metadata)
    direct=direct_readings(window)
    return ([dict(kind='SQL',frame=b['frame'],pixel_sha256=b['pixel_sha256'],
                  observation_id=b['observation_id'],state_id=b['state_id']) for b in bindings]+
            [dict(kind='DIRECT',frame=r['frame'],pixel_sha256=r['pixel_sha256']) for r in direct])


def initial_status(case,point,inventory,*,interval=None):
    """Prove the first visible status after the current forced-post overlay.

    This is a complete visual transition, not a claim about action onset or
    arbitrary later status. Unknown/occluded frames withhold the exemption.
    """
    model=SitOutTextPattern();sources=[];receipts=[];phases=[]
    from parser_core.chat_nick_actor import reuses_cross_identity, cross_identity_frame_binding
    identity_binding = None
    if reuses_cross_identity(point):
        if interval is None:
            raise ValueError('CROSS_STATUS_VALID_HAND_INTERVAL_REQUIRED')
        low, high = interval
        context = case['source_context']
        candidates = sorted({w['raw_record']['source_frame'] for w in case['raw_stack_captures']
            if low <= w['raw_record']['source_frame'] <= context['capture_end_inclusive']})
        first = None
        for frame in candidates:
            _, gray = source_image(case, frame)
            if model.evaluate(gray)['matched']:
                first = frame
                break
        if first is None:
            raise ValueError('FIRST_SOURCE_SIT_OUT_TRIPLE_NOT_PROVEN')
        frames = list(range(first, first+3))
        identity_binding = cross_identity_frame_binding(case['full_hand_source_context'],
            point, frames, start_frame=low, end_frame=high, native_who=getattr(inventory,'native_who',None))
    else:
        frames=[r['source_frame'] for r in numeric_nickname_sources(case['source_context'],point)]
    if frames!=list(range(frames[0],frames[0]+3)):
        raise ValueError('SIT_OUT_NICKNAME_TRIPLE_REQUIRED')
    if status_numeric_conflicts(case,inventory,frames[-1]):
        raise ValueError('SIT_OUT_NUMERIC_SOURCE_CONFLICT')
    low=case['source_context']['header_frame']
    if frames[0]<=low+2:raise ValueError('SIT_OUT_CURRENT_FORCED_OVERLAY_UNPROVEN')
    references=[];blank_seen=False;last_time=None;overlays=ForcedOverlayPatterns()
    for frame in range(low,frames[-1]+1):
        row,gray=source_image(case,frame);match=model.evaluate(gray)
        stamp=source_time(row['source_pts'],row['time_base'])
        if last_time is not None and stamp<=last_time:raise ValueError('SIT_OUT_SOURCE_TIME_CONFLICT')
        last_time=stamp
        patch=gray[2:-2,6:-6].astype(np.float64)
        if not patch.size:raise ValueError('SIT_OUT_SOURCE_INTERIOR_EMPTY')
        if frame<low+3:
            identity=overlays.identify(patch)
            if identity is None:raise ValueError('SIT_OUT_CURRENT_FORCED_OVERLAY_UNPROVEN')
            references.append(patch)
        ref=np.mean(references,axis=0)
        a=ref-ref.mean();b=patch-patch.mean();denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
        ncc=float((a*b).sum()/denominator) if denominator else None
        gain=float((a*b).sum()/(a*a).sum()) if (a*a).sum() else None
        blank=float(np.ptp(patch))<=BLANK_GRAY_RANGE_MAX
        same_overlay=ncc is not None and ncc>=OVERLAY_NCC_MIN and gain is not None and gain>0
        if frame<frames[0]:
            if match['matched']:raise ValueError('EARLIER_SIT_OUT_WITHOUT_SAME_FRAME_ACTOR_POINT')
            if blank:blank_seen=True;phase='BLANK_WITHIN_CODEC_NOISE'
            elif same_overlay and not blank_seen:phase='POSITIVE_SAME_FORCED_OVERLAY_OR_FADE'
            else:raise ValueError('UNEXPLAINED_PRE_SIT_OUT_PIXELS')
        else:
            if not match['matched'] or blank or same_overlay:raise ValueError('FIRST_SOURCE_SIT_OUT_TRIPLE_NOT_PROVEN')
            phase='FIRST_VISIBLE_SIT_OUT'
            sources.append(row);receipts.append(dict(frame=frame,refined_gray_sha256=sha(gray.tobytes()),mask=match))
        phases.append(dict(frame=frame,phase=phase,ncc_to_forced_overlay=ncc,brightness_gain=gain,
                           refined_gray_sha256=sha(gray.tobytes())))
    return dict(authority='FIRST_VISIBLE_SIT_OUT_AFTER_CURRENT_FORCED_OVERLAY',
        frames=frames,sources=sources,mask_receipts=receipts,source_transition=phases,
        numeric_contradictions=[],initial_deal_status_authority=False,action_onset_authority=False,
        **({'identity_frame_binding': identity_binding} if identity_binding is not None else {}))


def prove(case,record,registry,inventory,check_seals):
    def unknown(reason):return dict(status='UNKNOWN',rule=RULE,actor=case.get('actor'),reason=reason)
    try:
        if not inventory.matches(case,inventory.metadata,record,registry):return unknown('FRESH_NATIVE_INVENTORY_CONFLICT')
        if check_seals('before'):return unknown('SIT_OUT_SOURCE_SEAL_NOT_OPEN')
        point=prove_actor_point(case['source_context'],case['actor'],native_who=getattr(inventory,'native_who',None))
        if point['status']!='PROVEN':return unknown('ACTOR_POINT_'+point['reason'])
        if any(point.get(k)!=case[k] for k in ('table_id','session_id','hand_id','actor','seat')):
            return unknown('SIT_OUT_ACTOR_CONTEXT_CONFLICT')
        if native_key(record['observed_hand'])!=native_key(case):return unknown('SIT_OUT_CURRENT_HAND_CONFLICT')
        if not point_matches_hand(point,record['observed_hand']):return unknown('CURRENT_HAND_ACTOR_SOURCE_CONFLICT')
        from parser_core.chat_nick_actor import reuses_cross_identity
        if reuses_cross_identity(point):
            low,high=complete_hand_interval(record,registry,case['actor'],case['seat'])
            status=initial_status(case,point,inventory,interval=(low,high))
        else:
            status=initial_status(case,point,inventory)
            low,high=complete_hand_interval(record,registry,case['actor'],case['seat'])
        if low!=case['source_context']['header_frame']:return unknown('SIT_OUT_HEADER_CONFLICT')
        native_who=getattr(inventory,'native_who',None)
        if native_who is not None:
            from parser_core.chat_nick_actor import assert_native_identity_interval
            assert_native_identity_interval(case['full_hand_source_context'],point,status['frames'],
                start_frame=low,end_frame=high,native_who=native_who)
        posts=own_posts(case['source_context'],case['actor'])
        ledger=independent_ledger(record['observed_hand'])
        events=[e for e in ledger['clean_events'] if e.get('player')==case['actor']]
        player=ledger['players'][case['actor']];forced=sum(p['amount'] for p in posts)
        if (not events or events[-1]['kind']!='FOLD' or sum(e['kind']=='FOLD' for e in events)!=1 or
                any(e['kind'] not in {'ANTE','SB','BB','FOLD'} or e.get('all_in') for e in events) or
                player['status']!='FOLDED' or player['returned'] or player['payout'] or
                player['gross_commit']!=forced or player['net_commit']!=forced):
            return unknown('SIT_OUT_OBSERVED_FORCED_AND_FOLD_CONFLICT')
        if check_seals('after'):return unknown('SIT_OUT_SOURCE_SEAL_NOT_OPEN')
        return dict(status='PROVEN',rule=outer_rule(RULE,point),reason=None,actor=case['actor'],seat=case['seat'],
            table_id=case['table_id'],session_id=case['session_id'],hand_id=case['hand_id'],
            player_status='SIT_OUT',initial_status_proof=status,identity_proof=point,
            own_forced_posts=posts,observed_actor_ledger=player,native_interval=[low,high],
            forced_prefix_ocr_policy=copy.deepcopy(FORCED_PREFIX_OCR_POLICY),
            ledger_seed_start_stacks={},numeric_stack_authority=False,synthetic_fold_authority=False,
            hh_contract=HH_CONTRACT,source_files_sha256={**registry.hashes,**inventory.hashes})
    except (KeyError,TypeError,ValueError,IndexError,AttributeError,OverflowError,ZeroDivisionError,OSError,cv2.error) as error:
        return unknown(str(error) if isinstance(error,ValueError) else type(error).__name__+':'+str(error))


def join_receipt(proof):
    return dict(rule=proof['rule'],proof_id=sha(json.dumps(proof,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()),
        actor=proof['actor'],seat=proof['seat'],table_id=proof['table_id'],session_id=proof['session_id'],
        hand_id=proof['hand_id'],player_status='SIT_OUT',numeric_stack_authority=False,
        synthetic_fold_authority=False,hh_contract=HH_CONTRACT,
        forced_prefix_ocr_policy=copy.deepcopy(proof['forced_prefix_ocr_policy']),
        status_source_authority=proof['initial_status_proof']['authority'],
        status_source_frames=proof['initial_status_proof']['frames'])


class SitOutSourceProvider:
    """A positive status selects an explicit HH exemption, never an amount."""
    def __init__(self,source,records,*,actor_policy=False,native_who=None):
        self.registry=SavedCarrySources(source,records,[])
        self.runtime_files=frozen_runtime_files()
        self.cases={};self.proofs={};self.audit=[];self.inventory=None
        selected=[];candidates={}
        for record in self.registry.records:
            if record['lifecycle']['status']!='COMPLETE':continue
            hand=record['observed_hand'];actors=set()
            for actor in {m.get('player') for m in hand['messages'] if m.get('player')}:
                own=[m for m in hand['messages'] if m.get('player')==actor]
                if any(m['action']=='FOLD' for m in own) and all(m['action'] in {'ANTE','SMALL_BLIND','BIG_BLIND','FOLD'} for m in own):
                    actors.add(actor)
            if not actors:continue
            try:native_hand_bounds(hand,self.registry)
            except (KeyError,TypeError,ValueError,IndexError,AttributeError,cv2.error) as error:
                self.audit.extend(dict(hand_key=list(native_key(hand)),actor=a,status='UNKNOWN',reason=str(error)) for a in sorted(actors))
                continue
            candidates[native_key(hand)]=(record,actors);selected.append(record)
        if not selected:return
        self.inventory=FreshConstantSources(source,selected,self.registry,actor_policy=actor_policy,native_who=native_who)
        batch=SourceSealBatch(self.registry,self.inventory,self.runtime_files)
        if not batch.begin():raise ValueError('SIT_OUT_PRE_SOURCE_SEAL_CHANGED')
        pending_cases={};pending_proofs={};pending_audit=[]
        for context,(record,actors) in candidates.items():
            for actor in sorted(actors):
                point=prove_actor_point(self.inventory.contexts[context],actor,native_who=self.inventory.native_who)
                if point['status']!='PROVEN':
                    pending_audit.append(dict(hand_key=list(context),actor=actor,status='UNKNOWN',reason='ACTOR_POINT_'+point['reason']))
                    continue
                case=case_for(self.inventory,context,actor,point['seat'])
                result=prove(case,record,self.registry,self.inventory,batch.check_phase)
                pending_audit.append(dict(hand_key=list(context),actor=actor,seat=point['seat'],status=result['status'],reason=result['reason']))
                if result['status']=='PROVEN':
                    pending_cases[(context,actor)]=(case,record);pending_proofs[(context,actor)]=result
        if not batch.finish():raise ValueError('SIT_OUT_POST_SOURCE_SEAL_CHANGED')
        self.cases=pending_cases;self.proofs=pending_proofs;self.audit.extend(pending_audit)

    def excludes_stack_work(self,context,actor):
        # Work suppression cannot supply HH authority; consumption replays below.
        context=(int(str(context[0]).removeprefix('T')),context[1],context[2])
        return (context,actor) in self.proofs

    def for_hand(self,hand):
        context=native_key(hand)
        cases=[(actor,case,record) for (saved,actor),(case,record) in self.cases.items()
               if saved==context and record['observed_hand']==hand]
        if not cases:return []
        batch=SourceSealBatch(self.registry,self.inventory,self.runtime_files)
        if not batch.begin():return []
        result=[]
        for actor,case,record in cases:
            proof=prove(case,record,self.registry,self.inventory,batch.check_phase)
            if proof['status']=='PROVEN' and proof==self.proofs[(context,actor)]:result.append(proof)
        return result if batch.finish() else []

    def write_audit(self,output):
        output=Path(output)
        with (output/'sitout_hand_proofs.jsonl').open('w',encoding='utf-8') as stream:
            for proof in self.proofs.values():stream.write(json.dumps(proof,ensure_ascii=False)+'\n')
        (output/'sitout_hand_audit.json').write_text(json.dumps(dict(
            scope='SOURCE_BOUND_STATUS_ONLY; NO_NUMERIC_STACK_OR_SYNTHETIC_FOLD',hh_contract=HH_CONTRACT,
            counts=dict(Counter(r['status'] for r in self.audit)),rows=self.audit),ensure_ascii=False,indent=2),'utf-8')
