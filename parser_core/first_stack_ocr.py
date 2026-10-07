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


def effective_numeric(reading,confidence=.95,*,allow_scaled=False):
    """Return a value only from unchanged primary or reproduced agreement proof."""
    primary=numeric(reading.get('raw_text'));score=reading.get('confidence')
    if reading.get('ocr_status')!='OK' or primary is None or not valid_score(score):return None
    if score>=confidence:return primary
    if not allow_scaled:return None
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
                not valid_score(result.get('score')) or result['score']<confidence or numeric(result.get('raw_text'))!=primary):return None
        return primary
    except (KeyError,TypeError,ValueError,cv2.error):return None
