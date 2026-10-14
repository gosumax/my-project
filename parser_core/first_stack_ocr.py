"""FULL-only same-source numeric evidence; immutable primary OCR stays intact."""
import base64
import hashlib
import json
import math
from pathlib import Path
import re
from functools import lru_cache

import cv2
import numpy as np

from parser_core import performance_diagnostics as diagnostics

RULE='SAME_SOURCE_NEAREST2_NUMERIC_AGREEMENT_V1'
NUMBER=re.compile(r'\d+(?:[ \u00a0]\d{3})*')
DIRECT_FIRST_VISIBLE_NUMERIC_POLICY = ('CALIBRATED_DIRECT_FIRST_VISIBLE_NUMERIC_085_20261009_V1', .85)
SIMPLE_SCALED_RULE = 'SAME_SOURCE_SCALED_ONLY_WITH_POSITIVE_NATIVE_DIGITS_V1'


def single_first_numeric_policy():
    """User-approved first native number; does not rewrite global OCR gates."""
    return dict(policy_id='FIRST_VISIBLE_SINGLE_NUMERIC', version=1,
                min_confidence=.93, comparison='STRICTLY_GREATER',
                required_numeric_frames=1, scope='FIRST_NATIVE_NUMBER_ONLY',
                bound_scaled_numeric_conflict='WITHHOLD')


def single_first_numeric(reading):
    """One original primary OCR must qualify; scaled OCR cannot raise its score."""
    score=reading.get('confidence')
    value=numeric(reading.get('raw_text'))
    if (reading.get('ocr_status')!='OK' or value is None or
            not valid_score(score) or not score>.93):
        return None
    scaled=bound_scaled_numeric(reading)
    if scaled is not None and scaled[1]>=DIRECT_FIRST_VISIBLE_NUMERIC_POLICY[1] and scaled[0]!=value:
        return None
    return value


def direct_numeric_policy():
    """A fresh receipt; this policy never lowers SQL, actor or occupancy gates."""
    return dict(policy_id=DIRECT_FIRST_VISIBLE_NUMERIC_POLICY[0],
                min_confidence=DIRECT_FIRST_VISIBLE_NUMERIC_POLICY[1],
                scope='DIRECT_FIRST_VISIBLE_NUMBER_ONLY',
                bound_scaled_numeric_conflict='WITHHOLD')


@lru_cache(maxsize=1)
def proof_config():
    root=Path(__file__).resolve().parents[1]
    return (json.loads((root/'configs/ocr_tiny_local.json').read_text('utf-8')),
            json.loads((root/'configs/layout_847x404.json').read_text('utf-8'))['slots'])


def numeric(text):
    text=(text or '').strip()
    return text.replace(' ','').replace('\u00a0','') if NUMBER.fullmatch(text) else None


def valid_score(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and 0<=value<=1


def scale_pixels(png_base64,source_sha256):
    gray=cv2.imdecode(np.frombuffer(base64.b64decode(png_base64,validate=True),np.uint8),cv2.IMREAD_GRAYSCALE)
    if gray is None or hashlib.sha256(gray.tobytes()).hexdigest()!=source_sha256:
        raise ValueError('FIRST_STACK_SCALE_SOURCE_HASH_CONFLICT')
    enlarged=np.repeat(np.repeat(gray,2,axis=0),2,axis=1)
    return gray,enlarged


class FirstStackOCR:
    def __init__(self,client,*,model_name,weights_sha256,confidence=.95,worker=0):
        self.client=client;self.model_name=model_name;self.weights_sha256=weights_sha256
        self.confidence=confidence;self.worker=worker

    def __getattr__(self,name):return getattr(self.client,name)

    def read_batch(self,items,batch_size):
        response=self.client.read_batch(items,batch_size)
        source={r['item_id']:r for r in items};pending=[];proofs={}
        for row in response['results']:
            score=row.get('score')
            if row.get('status')!='OK' or numeric(row.get('raw_text')) is None or not valid_score(score) or score>=self.confidence:
                continue
            original=source.get(row['item_id'])
            if original is None:continue
            proof=dict(rule=RULE,source_item_id=row['item_id'],source_pixel_sha256=original['source_sha256'],
                primary_response={k:row.get(k) for k in ('status','raw_text','score')},
                model_name=self.model_name,weights_sha256=self.weights_sha256)
            row['same_source_scale_evidence']=proof
            try:
                gray,enlarged=scale_pixels(original['png_base64'],original['source_sha256'])
                ok,png=cv2.imencode('.png',enlarged)
                if not ok:raise ValueError('FIRST_STACK_SCALE_PNG_ENCODING_FAILED')
                digest=hashlib.sha256(enlarged.tobytes()).hexdigest()
                proof.update(source_shape=list(gray.shape),scaled_shape=list(enlarged.shape),scaled_pixel_sha256=digest)
                pending.append(dict(item_id=row['item_id'],png_base64=base64.b64encode(png).decode('ascii'),source_sha256=digest))
                proofs[row['item_id']]=proof
            except (ValueError,TypeError,cv2.error) as exc:
                proof.update(status='UNKNOWN',reason=type(exc).__name__)
        if pending:
            try:
                extra=self.client.read_batch(pending,batch_size)
                for item_id,proof in proofs.items():
                    matches=[r for r in extra['results'] if r.get('item_id')==item_id]
                    proof.update(status='OBSERVED' if len(matches)==1 else 'UNKNOWN',
                                 response=matches[0] if len(matches)==1 else None)
                response['inference_seconds']=response.get('inference_seconds',0)+extra.get('inference_seconds',0)
                detail=extra.get('performance_diagnostic')
                if diagnostics.ACTIVE and detail:
                    with diagnostics.ACTIVE.lock:
                        diagnostics.ACTIVE.ocr_compute.append(dict(lane='START_STACK_SAME_SOURCE_SCALE',
                            worker=self.worker,batch_size=len(pending),**detail))
            except Exception as exc:
                for proof in proofs.values():proof.update(status='UNKNOWN',reason='SCALE_OCR_FAILED',error=repr(exc))
        return response


def bound_scaled_numeric(reading):
    """Reproduce a saved scale reading before giving it agreement/conflict authority."""
    score=reading.get('confidence')
    try:
        proof=reading['ocr_response']['same_source_scale_evidence']
        model,slots=proof_config()
        seats=list(slots[f"slot_{reading['table_id']:02d}"]['seats'])
        expected_id=f"{reading['table_id']}:{reading['source_frame']}:{seats.index(reading['seat'])}"
        if (proof.get('rule')!=RULE or proof.get('status')!='OBSERVED' or
                proof.get('model_name')!=model['model_name'] or proof.get('weights_sha256')!=model['weights_sha256'] or
                proof.get('primary_response')!=dict(status=reading['ocr_status'],raw_text=reading['raw_text'],score=score) or
                proof.get('source_item_id')!=reading['ocr_response'].get('item_id') or
                proof.get('source_item_id')!=expected_id or
                proof.get('source_pixel_sha256')!=reading['pixel_sha256']):return None
        gray,enlarged=scale_pixels(reading['png_base64'],reading['pixel_sha256'])
        if (proof.get('source_shape')!=list(gray.shape) or proof.get('scaled_shape')!=list(enlarged.shape) or
                proof.get('scaled_pixel_sha256')!=hashlib.sha256(enlarged.tobytes()).hexdigest()):return None
        result=proof['response']
        if (result.get('item_id')!=proof['source_item_id'] or result.get('status')!='OK' or
                not valid_score(result.get('score'))):return None
        value=numeric(result.get('raw_text'))
        return (value,result['score']) if value is not None else None
    except (KeyError,TypeError,ValueError,cv2.error):return None


def effective_numeric(reading,confidence=.95,*,allow_scaled=False,numeric_policy=None):
    """Retain primary bytes; the calibrated direct policy also checks conflicts."""
    if not valid_score(confidence):return None
    if numeric_policy is not None and (numeric_policy!=direct_numeric_policy() or
            confidence!=DIRECT_FIRST_VISIBLE_NUMERIC_POLICY[1]):return None
    primary=numeric(reading.get('raw_text'));score=reading.get('confidence')
    if reading.get('ocr_status')!='OK' or primary is None or not valid_score(score):return None
    scaled=None
    if numeric_policy is not None:
        scaled=bound_scaled_numeric(reading)
        if scaled is not None and scaled[1]>=confidence and scaled[0]!=primary:return None
    if score>=confidence:return primary
    if not allow_scaled:return None
    if scaled is None:scaled=bound_scaled_numeric(reading)
    return primary if scaled is not None and scaled[1]>=confidence and scaled[0]==primary else None


def simple_scaled_numeric_evidence(reading, *, visibility_source=None):
    """Scaling cannot independently promote weak OCR without positive pixels.

    A negative presence gate never vetoes an independently qualifying primary
    reading and never proves EMPTY. Only scaled-only rescue needs positive,
    reproduced native digit visibility. Bound confident disagreement withholds
    both readings, as in the existing calibrated direct numeric policy.
    """
    threshold=DIRECT_FIRST_VISIBLE_NUMERIC_POLICY[1]
    primary=numeric(reading.get('raw_text'));score=reading.get('confidence')
    evidence=dict(rule=SIMPLE_SCALED_RULE,status='UNKNOWN',reason='SCALED_SOURCE_NOT_PROVEN',
                  source_pixel_sha256=reading.get('pixel_sha256'),primary_value=primary,
                  primary_confidence=score)
    scaled=bound_scaled_numeric(reading)
    if scaled is None:return dict(primary_conflict=False,scaled=evidence)
    evidence.update(same_source_bound=True,scaled_value=scaled[0],scaled_confidence=scaled[1])
    if scaled[1]<threshold:
        evidence['reason']='SCALED_CONFIDENCE_BELOW_POLICY'
        return dict(primary_conflict=False,scaled=evidence)
    if primary is None or not valid_score(score) or reading.get('ocr_status')!='OK':
        evidence['reason']='PRIMARY_NUMERIC_SOURCE_NOT_PROVEN'
        return dict(primary_conflict=False,scaled=evidence)
    if primary!=scaled[0]:
        evidence['reason']='BOUND_PRIMARY_SCALED_NUMERIC_CONFLICT'
        return dict(primary_conflict=True,scaled=evidence)
    if score>=threshold:
        evidence.update(status='ADMITTED',reason='INDEPENDENT_PRIMARY_NUMERIC_AGREEMENT')
        return dict(primary_conflict=False,scaled=evidence)
    from .start_stack_visibility import VisibilitySource
    if visibility_source is None:visibility_source=VisibilitySource()
    projection=visibility_source.verify(reading)
    if projection is None:
        evidence['reason']='SCALED_ONLY_NATIVE_VISIBILITY_UNPROVEN'
    else:
        gate=projection['digit_gate']
        evidence.update(visibility_status=gate['status'],
                        timeline_pixel_sha256=projection['timeline_pixel_sha256'])
        if gate['status']=='DIGITS_VISIBLE':
            evidence.update(status='ADMITTED',reason='BOUND_AGREEING_SCALE_WITH_POSITIVE_NATIVE_DIGITS')
        else:evidence['reason']='SCALED_ONLY_WITHOUT_POSITIVE_NATIVE_DIGITS'
    return dict(primary_conflict=False,scaled=evidence)
