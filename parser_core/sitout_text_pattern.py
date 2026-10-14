"""Shared SitOut raster family; classification consumes pixels alone.

The old binary digit-exclusion mask and its historical SQL receipts stay frozen.
This separate status matcher never produces an amount, actor or hand identity.
"""
import hashlib
import json
import math
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models/sitout_text_gray.json'
METHOD='SIT_OUT_GRAY_TEMPLATE_BANK_V1'
SIT_OUT_MATCH_THRESHOLD=.86


class SitOutTextPattern:
    def __init__(self,model=MODEL):
        payload=Path(model).read_bytes();data=json.loads(payload)
        if (data.get('method')!='TM_CCOEFF_NORMED' or data.get('threshold')!=SIT_OUT_MATCH_THRESHOLD or
                data.get('generic_pixel_only') is not True):
            raise ValueError('SIT_OUT_TEXT_MODEL_CONTRACT_CONFLICT')
        self.templates=[]
        self.threshold=float(data['threshold'])
        self.model_sha256=hashlib.sha256(payload).hexdigest()
        self.code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        for row in data['templates']:
            original=np.asarray(row['raw_gray'])
            template=np.asarray(row['template_gray'])
            if (original.ndim!=2 or template.ndim!=2 or not template.size or
                    not np.issubdtype(original.dtype,np.integer) or not np.issubdtype(template.dtype,np.integer) or
                    original.min()<0 or original.max()>255 or template.min()<0 or template.max()>255):
                raise ValueError('SIT_OUT_TEXT_MODEL_PIXELS_INVALID')
            original=original.astype(np.uint8);template=template.astype(np.uint8)
            x,y,X,Y=row['template_xyxy']
            if (list(original.shape)!=row['raw_gray_shape'] or list(template.shape)!=row['template_shape'] or
                    hashlib.sha256(original.tobytes()).hexdigest()!=row['raw_gray_sha256'] or
                    hashlib.sha256(template.tobytes()).hexdigest()!=row['template_gray_sha256'] or
                    not 0<=x<X<=original.shape[1] or not 0<=y<Y<=original.shape[0] or
                    not np.array_equal(template,original[y:Y,x:X]) or float(template.std())==0):
                raise ValueError('SIT_OUT_TEXT_MODEL_SOURCE_CONFLICT')
            # Table/seat/frame fields are retained provenance only. No such
            # input is accepted by evaluate(), and no route selects a template.
            self.templates.append((row['template_gray_sha256'],template))
        if not self.templates:raise ValueError('SIT_OUT_TEXT_MODEL_EMPTY')

    def evaluate(self,image):
        gray=image if image.ndim==2 else cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if gray.ndim!=2 or gray.dtype!=np.uint8:raise ValueError('SIT_OUT_NATIVE_GRAY_REQUIRED')
        best=-1.;box=None;template_hash=None;scores=[]
        for digest,template in self.templates:
            h,w=template.shape
            if gray.shape[0]<h or gray.shape[1]<w:continue
            matrix=cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
            _,score,_,(x,y)=cv2.minMaxLoc(matrix)
            if not math.isfinite(score):raise ValueError('SIT_OUT_TEXT_SCORE_INVALID')
            scores.append(dict(template_sha256=digest,score=float(score)))
            if score>best:best=float(score);box=[x,y,x+w,y+h];template_hash=digest
        return dict(matched=best>=self.threshold,score=best,box=box,method=METHOD,threshold=self.threshold,
            template_sha256=template_hash,template_scores=scores,model_sha256=self.model_sha256,
            code_sha256=self.code_sha256,numeric_authority=False)
