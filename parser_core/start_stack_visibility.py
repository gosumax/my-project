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
        scores={}
        for label,template in self.templates:
            best=0.
            for dy in range(-2,3):
                for dx in range(-2,3):
                    candidate=self.canvas(pixels,dy,dx)
                    denom=float(np.linalg.norm(candidate)*np.linalg.norm(template))
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


def first_visible_proof(readings, frames, *, start_frame, confidence=.95):
    """Require a positive forced-label/fade/blank sequence before the number.

    The digit gate is retained as independent evidence, not numeric-absence
    authority. Fixed interior geometry removes panel borders, as in the existing
    visual tracker. Every earlier source must positively match the same literal
    overlay's centered shape or have only the measured codec-noise gray range.
    """
    unknown = lambda reason: dict(contract=CONTRACT,status='UNKNOWN',reason=reason)
    if len(frames)!=3 or frames!=list(range(frames[0],frames[0]+3)) or start_frame is None:
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
        number=effective_numeric(r,confidence,allow_scaled=True)
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
            return unknown('FIRST_NUMERIC_TRIPLE_NOT_PROVEN')
        else:
            phase='FIRST_STABLE_NUMBER'
            values.append(number)
        phases.append(dict(frame=f,phase=phase,ncc_to_forced_overlay=ncc,brightness_gain=gain,
                           gray_range=float(np.ptp(patch))))
    if len(set(values))!=1:
        return unknown('FIRST_NUMERIC_TRIPLE_NOT_STABLE')
    if not any(labels[i:i+3]==[(f,label),(f+1,label),(f+2,label)] for i,(f,label) in enumerate(labels)):
        return unknown('CURRENT_FORCED_OVERLAY_NOT_PROVEN')
    return dict(contract=CONTRACT,status='PROVEN',reason='FIRST_NUMBER_AFTER_CURRENT_FORCED_OVERLAY',
                start_frame=start_frame,first_numeric_frame=frames[0],frames=list(frames),
                displayed_stack=values[0],context=list(context),
                forced_overlay_frames=[f for f,_ in labels],source_evidence=sources,phases=phases,
                forced_overlay_identities=identities,
                pattern_contract=dict(interior='gray[2:-2,6:-6]',overlay_ncc_min=OVERLAY_NCC_MIN,
                                      blank_gray_range_max=BLANK_GRAY_RANGE_MAX),
                authority='USER_CLIENT_RULE_20261007_PLUS_SOURCE_PIXELS_AND_OCR')


def verify_first_visible_fact(fact):
    proof=fact.get('first_visible_proof') or {}
    if proof.get('contract')!=CONTRACT or proof.get('status')!='PROVEN':
        return False
    sources=proof.get('source_evidence') or []
    readings={r.get('source_frame'):r for r in sources}
    if len(readings)!=len(sources):
        return False
    checked=first_visible_proof(readings,fact.get('frames',[]),start_frame=proof.get('start_frame'))
    expected=[fact.get('table'),fact.get('session'),fact.get('hand_id'),fact.get('seat')]
    evidence=fact.get('evidence') or []
    return (checked==proof and checked.get('status')=='PROVEN' and checked.get('context')==expected
            and checked.get('displayed_stack')==fact.get('displayed_stack')
            and checked.get('first_numeric_frame')==fact.get('first_numeric_visible_frame')
            and proof['start_frame']==fact.get('deal_complete_frame',-2)+1
            and evidence==sources[-3:])
