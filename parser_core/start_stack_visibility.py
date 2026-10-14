"""PokerDom first-number contract with independently reproducible source crops.

The user-supplied client rule applies after the current forced-post overlay.
It never turns an arbitrary later numeric snapshot into an initial bankroll.
"""
import base64
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from .dealing_start_stack import NUMERIC
from .ram_first_stage import png_compatible_gray
from .stack_timeline_algorithms import excluding_gate_class

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'POKERDOM_FIRST_VISIBLE_POST_FORCED_V1'
SINGLE_CONTRACT = 'POKERDOM_SINGLE_FIRST_VISIBLE_POST_FORCED_GT_093_V1'
FIRST_QUALIFIED_CONTRACT = 'POKERDOM_FIRST_QUALIFIED_STACK_FRAME_085_V1'
FIRST_QUALIFIED_RULE = 'FIRST_QUALIFIED_NUMERIC_STACK_AFTER_DEAL_COMPLETE'
OVERLAY_NCC_MIN = .90
BLANK_GRAY_RANGE_MAX = 3


class ForcedOverlayPatterns:
    """Native, visually labelled static words; no digit/value recognition."""
    def __init__(self):
        data=(ROOT/'models/forced_stack_overlays.json').read_bytes()
        self.sha256=hashlib.sha256(data).hexdigest()
        self.templates=[]
        for row in json.loads(data)['templates']:
            pixels=np.frombuffer(base64.b64decode(row['pixels_base64']),np.uint8).reshape(row['shape'])
            if hashlib.sha256(pixels.tobytes()).hexdigest()!=row['pixel_sha256']:
                raise ValueError('FORCED_OVERLAY_TEMPLATE_HASH_CONFLICT')
            self.templates.append((row['label'],self.canvas(pixels)))

    @staticmethod
    def canvas(pixels,dy=0,dx=0):
        values=np.maximum(pixels.astype(np.float64)-float(np.median(pixels)),0)
        out=np.zeros((24,128),np.float64)
        h,w=values.shape;y=(24-h)//2+dy;x=(128-w)//2+dx
        if not (0<=x and x+w<=128 and 0<=y and y+h<=24):
            return out
        out[y:y+h,x:x+w]=values
        return out

    def identify(self,pixels):
        if not self.templates:
            return None
        # The translated input is independent of the template. Keep all reuse
        # local to this call and retain the original template/offset score order.
        candidates=[self.canvas(pixels,dy,dx) for dy in range(-2,3) for dx in range(-2,3)]
        candidate_norms=[np.linalg.norm(candidate) for candidate in candidates]
        scores={}
        for label,template in self.templates:
            best=0.
            template_norm=np.linalg.norm(template)
            for candidate,candidate_norm in zip(candidates,candidate_norms):
                denom=float(candidate_norm*template_norm)
                value=float((candidate*template).sum()/denom) if denom else 0.
                best=max(best,value)
            scores[label]=max(scores.get(label,0.),best)
        ranked=sorted(scores.items(),key=lambda item:item[1],reverse=True)
        if not ranked or ranked[0][1]<.98:
            return None
        # SB vs BB ambiguity has no amount/role authority here: both belong to
        # the same forced-overlay visibility class. CHAT binds actual posts.
        return dict(label='FORCED_POST_OVERLAY',matched_labels=[label for label,score in ranked if score>=.98],
                    score=ranked[0][1],scores=scores,model_sha256=self.sha256)


class VisibilitySource:
    def __init__(self):
        self.rois = json.loads((ROOT/'configs/stack_roi_refined.json').read_text('utf-8'))['slots']
        self.gate = excluding_gate_class()(ROOT/'configs/stack_digit_gate.json')

    def capture(self, rgb, table, seat, roi):
        target = self.rois[f'slot_{table:02d}']['seats'][seat]['stack']
        x,y,X,Y = roi
        a,b,A,B = target
        if not (x <= a < A <= X and y <= b < B <= Y):
            raise ValueError('STACK_PROJECTION_OUTSIDE_SOURCE_ROI')
        crop = np.ascontiguousarray(rgb[y:Y,x:X])
        projected = cv2.cvtColor(crop[b-y:B-y,a-x:A-x], cv2.COLOR_RGB2GRAY)
        ok, packed = cv2.imencode('.png', cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError('STACK_SOURCE_RGB_ENCODING_FAILED')
        return dict(source_rgb_roi=list(roi), source_rgb_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                    source_rgb_png_base64=base64.b64encode(packed).decode('ascii'),
                    timeline_roi=list(target), timeline_pixel_sha256=hashlib.sha256(projected.tobytes()).hexdigest(),
                    timeline_shape=list(projected.shape), digit_gate=self.gate.evaluate(cv2.cvtColor(projected,cv2.COLOR_GRAY2BGR)))

    def verify(self, reading):
        """Both gray representations must reproduce from the same RGB pixels."""
        try:
            p = reading['visibility_source']
            rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(p['source_rgb_png_base64'],validate=True),np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
            x,y,X,Y = reading['roi']
            target = self.rois[f"slot_{reading['table_id']:02d}"]['seats'][reading['seat']]['stack']
            a,b,A,B = target
            if (p['source_rgb_roi'] != reading['roi'] or p['timeline_roi'] != target
                    or rgb.shape != (Y-y,X-x,3) or not (x <= a < A <= X and y <= b < B <= Y)
                    or hashlib.sha256(rgb.tobytes()).hexdigest() != p['source_rgb_sha256']):
                return None
            direct = png_compatible_gray(rgb)
            saved = cv2.imdecode(np.frombuffer(base64.b64decode(reading['png_base64'],validate=True),np.uint8),cv2.IMREAD_GRAYSCALE)
            projected = cv2.cvtColor(rgb[b-y:B-y,a-x:A-x],cv2.COLOR_RGB2GRAY)
            digest = hashlib.sha256(projected.tobytes()).hexdigest()
            if (not np.array_equal(saved,direct) or hashlib.sha256(direct.tobytes()).hexdigest() != reading['pixel_sha256']
                    or digest != p['timeline_pixel_sha256'] or list(projected.shape) != p['timeline_shape']):
                return None
            gate = self.gate.evaluate(cv2.cvtColor(projected,cv2.COLOR_GRAY2BGR))
            if gate != p['digit_gate']:
                return None
            return dict(timeline_pixel_sha256=digest, digit_gate=gate, gray=projected)
        except (KeyError, TypeError, ValueError, cv2.error):
            return None


def first_qualified_stack_proof(readings, frames, *, start_frame, confidence, numeric_policy,
                                clean_window_end_frame=None):
    """One real qualifying frame; no numeric agreement or overlay sequence."""
    from .first_stack_ocr import effective_numeric, direct_numeric_policy
    unknown=lambda reason:dict(contract=FIRST_QUALIFIED_CONTRACT,status='UNKNOWN',reason=reason)
    if numeric_policy!=direct_numeric_policy() or confidence!=numeric_policy['min_confidence']:
        return unknown('FIRST_QUALIFIED_NUMERIC_POLICY_CONFLICT')
    if (type(start_frame) is not int or len(frames)!=1 or type(frames[0]) is not int
            or frames[0]<start_frame):
        return unknown('FIRST_QUALIFIED_SOURCE_RANGE_UNKNOWN')
    if clean_window_end_frame is not None and (type(clean_window_end_frame) is not int
            or frames[0]>=clean_window_end_frame):
        return unknown('FIRST_QUALIFIED_SOURCE_AFTER_WINDOW_CLOSED')
    selected=readings.get(frames[0])
    if selected is None:return unknown('FIRST_QUALIFIED_SOURCE_UNKNOWN')
    context=tuple(selected.get(k) for k in ('table_id','session_id','hand_id','seat'))
    if any(v is None for v in context):return unknown('FIRST_QUALIFIED_CONTEXT_UNKNOWN')
    sources=[];previous=None
    for frame,row in sorted(readings.items()):
        if not start_frame<=frame<=frames[0]:continue
        if row is None:return unknown('EARLIER_STACK_OCR_PENDING')
        if row.get('source_frame')!=frame or tuple(row.get(k) for k in
                ('table_id','session_id','hand_id','seat'))!=context:
            return unknown('FIRST_QUALIFIED_CONTEXT_CONFLICT')
        try:
            pts=row['source_pts'];num,den=row['time_base']
            if any(type(v) is not int for v in (pts,num,den)) or num<=0 or den<=0:
                return unknown('FIRST_QUALIFIED_TIME_UNKNOWN')
            stamp=Fraction(pts*num,den)
        except (KeyError,TypeError,ValueError,ZeroDivisionError):
            return unknown('FIRST_QUALIFIED_TIME_UNKNOWN')
        if previous is not None and stamp<=previous:return unknown('FIRST_QUALIFIED_TIME_CONFLICT')
        previous=stamp;sources.append(row)
        value=effective_numeric(row,confidence,allow_scaled=True,numeric_policy=numeric_policy)
        if frame<frames[0] and value is not None:return unknown('EARLIER_QUALIFIED_STACK_NUMBER')
    value=effective_numeric(selected,confidence,allow_scaled=True,numeric_policy=numeric_policy)
    if value is None:return unknown('FIRST_QUALIFIED_NUMBER_NOT_PROVEN')
    if VisibilitySource().verify(selected) is None:return unknown('FIRST_QUALIFIED_PIXELS_UNPROVEN')
    return dict(contract=FIRST_QUALIFIED_CONTRACT,status='PROVEN',reason='FIRST_QUALIFIED_OBSERVED_STACK_NUMBER',
        start_frame=start_frame,first_numeric_frame=frames[0],frames=list(frames),
        displayed_stack=value,context=list(context),source_evidence=sources,
        numeric_ocr_policy=direct_numeric_policy(),required_numeric_frames=1,
        clean_window_end_frame=clean_window_end_frame,complete_pre_numeric_coverage='NOT_REQUIRED',
        pre_numeric_overlay_sequence='NOT_REQUIRED',authority='USER_ONE_GOOD_STACK_FRAME_20261014')


def valid_numeric_witness(fact):
    frames=fact.get('frames',[])
    if fact.get('stack_witness_policy')==FIRST_QUALIFIED_CONTRACT:
        return (len(frames)==1 and type(frames[0]) is int and
                fact.get('clean_window_rule')==FIRST_QUALIFIED_RULE)
    return (len(frames)==3 and frames==list(range(frames[0],frames[0]+3)) and
            fact.get('clean_window_rule')=='FIRST_VISIBLE_STABLE_STACK_TRIPLE_AFTER_DEAL_COMPLETE')


def first_visible_proof(readings, frames, *, start_frame, confidence=.95, numeric_policy=None,
                        single_first_policy=None, qualified_single=False, clean_window_end_frame=None):
    """Require a positive forced-label/fade/blank sequence before the number.

    The digit gate is retained as independent evidence, not numeric-absence
    authority. Fixed interior geometry removes panel borders, as in the existing
    visual tracker. Every earlier source must positively match the same literal
    overlay's centered shape or have only the measured codec-noise gray range.
    """
    if qualified_single:
        return first_qualified_stack_proof(readings,frames,start_frame=start_frame,confidence=confidence,
            numeric_policy=numeric_policy,clean_window_end_frame=clean_window_end_frame)
    contract = SINGLE_CONTRACT if single_first_policy is not None else CONTRACT
    unknown = lambda reason: dict(contract=contract,status='UNKNOWN',reason=reason)
    from parser_core.first_stack_ocr import (direct_numeric_policy, valid_score,
        single_first_numeric_policy, single_first_numeric)
    single = single_first_policy is not None
    if single and (single_first_policy != single_first_numeric_policy() or
                   confidence != .93 or numeric_policy is not None):
        return unknown('SINGLE_FIRST_NUMERIC_POLICY_CONFLICT')
    if not valid_score(confidence):return unknown('NUMERIC_CONFIDENCE_INVALID')
    if numeric_policy is not None and (numeric_policy!=direct_numeric_policy() or
            confidence!=numeric_policy['min_confidence']):
        return unknown('NUMERIC_POLICY_CONFLICT')
    count = 1 if single else 3
    if len(frames)!=count or frames!=list(range(frames[0],frames[0]+count)) or start_frame is None:
        return unknown('FIRST_VISIBLE_SOURCE_RANGE_UNKNOWN')
    sources = [readings.get(f) for f in range(start_frame,frames[-1]+1)]
    if not sources or any(r is None for r in sources):
        return unknown('FIRST_VISIBLE_SOURCE_COVERAGE_UNKNOWN')
    verifier = VisibilitySource()
    overlays = ForcedOverlayPatterns()
    context = tuple(sources[0].get(k) for k in ('table_id','session_id','hand_id','seat'))
    if any(v is None for v in context):
        return unknown('FIRST_VISIBLE_SOURCE_CONTEXT_UNKNOWN')
    values=[]
    labels=[]
    phases=[]
    references=[]
    identities=[]
    blank_seen=False
    last_time=None
    for f,r in zip(range(start_frame,frames[-1]+1),sources):
        if r.get('source_frame')!=f or tuple(r.get(k) for k in ('table_id','session_id','hand_id','seat'))!=context:
            return unknown('FIRST_VISIBLE_SOURCE_CONTEXT_CONFLICT')
        try:
            stamp=Fraction(r['source_pts']*r['time_base'][0],r['time_base'][1])
        except (KeyError,TypeError,ValueError,IndexError,ZeroDivisionError):
            return unknown('FIRST_VISIBLE_SOURCE_TIME_UNKNOWN')
        if last_time is not None and stamp<=last_time:
            return unknown('FIRST_VISIBLE_SOURCE_TIME_CONFLICT')
        last_time=stamp
        projection=verifier.verify(r)
        if projection is None:
            return unknown('FIRST_VISIBLE_SOURCE_PIXELS_UNPROVEN')
        text=(r.get('raw_text') or '').strip()
        from parser_core.first_stack_ocr import effective_numeric
        number=(single_first_numeric(r) if single else
                effective_numeric(r,confidence,allow_scaled=True,numeric_policy=numeric_policy))
        good=number is not None
        patch=projection['gray'][2:-2,6:-6].astype(np.float64)
        if not patch.size:
            return unknown('STACK_TEXT_INTERIOR_EMPTY')
        if f<start_frame+3:
            identity=overlays.identify(patch)
            if not (f<frames[0] and identity is not None):
                return unknown('CURRENT_FORCED_OVERLAY_NOT_PROVEN')
            if identities and identity['label']!=identities[0]['label']:
                return unknown('CURRENT_FORCED_OVERLAY_CONFLICT')
            references.append(patch)
            identities.append(identity)
        ref=np.mean(references,axis=0) if references else None
        if ref is None:
            return unknown('CURRENT_FORCED_OVERLAY_NOT_PROVEN')
        a=ref-ref.mean(); b=patch-patch.mean()
        denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
        ncc=float((a*b).sum()/denominator) if denominator else None
        gain=float((a*b).sum()/(a*a).sum()) if (a*a).sum() else None
        blank=float(np.ptp(patch))<=BLANK_GRAY_RANGE_MAX
        same_overlay=ncc is not None and ncc>=OVERLAY_NCC_MIN and gain is not None and gain>0
        if f<frames[0]:
            if good and NUMERIC.fullmatch(text):
                return unknown('EARLIER_NUMERIC_OR_AMBIGUOUS_SOURCE')
            if blank:
                blank_seen=True
                phase='BLANK_WITHIN_CODEC_NOISE'
            elif same_overlay and not blank_seen:
                phase='POSITIVE_SAME_FORCED_OVERLAY_OR_FADE'
            else:
                return unknown('UNEXPLAINED_PRE_NUMERIC_PIXELS')
            if same_overlay and not blank:
                labels.append((f,identities[0]['label']))
        elif not (good and NUMERIC.fullmatch(text) and not blank and not same_overlay):
            return unknown('FIRST_SINGLE_NUMBER_NOT_PROVEN' if single else 'FIRST_NUMERIC_TRIPLE_NOT_PROVEN')
        else:
            phase='FIRST_SOURCE_NUMBER' if single else 'FIRST_STABLE_NUMBER'
            values.append(number)
        phases.append(dict(frame=f,phase=phase,ncc_to_forced_overlay=ncc,brightness_gain=gain,
                           gray_range=float(np.ptp(patch))))
    if len(set(values))!=1:
        return unknown('FIRST_NUMERIC_TRIPLE_NOT_STABLE')
    if not any(labels[i:i+3]==[(f,label),(f+1,label),(f+2,label)] for i,(f,label) in enumerate(labels)):
        return unknown('CURRENT_FORCED_OVERLAY_NOT_PROVEN')
    result=dict(contract=contract,status='PROVEN',reason='FIRST_NUMBER_AFTER_CURRENT_FORCED_OVERLAY',
                start_frame=start_frame,first_numeric_frame=frames[0],frames=list(frames),
                displayed_stack=values[0],context=list(context),
                forced_overlay_frames=[f for f,_ in labels],source_evidence=sources,phases=phases,
                forced_overlay_identities=identities,
                pattern_contract=dict(interior='gray[2:-2,6:-6]',overlay_ncc_min=OVERLAY_NCC_MIN,
                                      blank_gray_range_max=BLANK_GRAY_RANGE_MAX),
                authority='USER_CLIENT_RULE_20261007_PLUS_SOURCE_PIXELS_AND_OCR')
    if numeric_policy is not None:result['numeric_ocr_policy']=direct_numeric_policy()
    if single:result['single_first_numeric_policy']=single_first_numeric_policy()
    return result


def verify_first_visible_fact(fact):
    proof=fact.get('first_visible_proof') or {}
    if proof.get('contract')==FIRST_QUALIFIED_CONTRACT:
        sources=proof.get('source_evidence') or []
        if not sources or not valid_numeric_witness(fact):return False
        readings={r.get('source_frame'):r for r in sources}
        if len(readings)!=len(sources):return False
        checked=first_qualified_stack_proof(readings,fact.get('frames',[]),
            start_frame=proof.get('start_frame'),confidence=.85,numeric_policy=proof.get('numeric_ocr_policy'),
            clean_window_end_frame=proof.get('clean_window_end_frame'))
        return (checked==proof and checked.get('status')=='PROVEN' and
            checked.get('context')==[fact.get(k) for k in ('table','session','hand_id','seat')] and
            checked.get('displayed_stack')==fact.get('displayed_stack') and
            checked.get('first_numeric_frame')==fact.get('first_numeric_visible_frame') and
            proof['start_frame']==fact.get('deal_complete_frame',-2)+1 and
            fact.get('evidence')==sources[-1:])
    if proof.get('contract')!=CONTRACT or proof.get('status')!='PROVEN':
        return False
    sources=proof.get('source_evidence') or []
    readings={r.get('source_frame'):r for r in sources}
    if len(readings)!=len(sources):
        return False
    from parser_core.first_stack_ocr import direct_numeric_policy
    policy=proof.get('numeric_ocr_policy')
    if 'numeric_ocr_policy' in proof and policy!=direct_numeric_policy():return False
    # A legacy receipt can only reproduce its original .95 contract. Absence
    # of the new policy never inherits calibrated .85 admission.
    checked=first_visible_proof(readings,fact.get('frames',[]),start_frame=proof.get('start_frame'),
        confidence=.95 if policy is None else policy['min_confidence'],numeric_policy=policy)
    expected=[fact.get('table'),fact.get('session'),fact.get('hand_id'),fact.get('seat')]
    evidence=fact.get('evidence') or []
    return (checked==proof and checked.get('status')=='PROVEN' and checked.get('context')==expected
            and checked.get('displayed_stack')==fact.get('displayed_stack')
            and checked.get('first_numeric_frame')==fact.get('first_numeric_visible_frame')
            and proof['start_frame']==fact.get('deal_complete_frame',-2)+1
            and evidence==sources[-3:])
