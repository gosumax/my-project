"""Positive native EMPTY buttons; never infer a player from their absence."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

RULE='COMPLETE_EMPTY_BUTTON_AND_FROZEN_CARDS_V1'


def button_roi(boxes, width, height):
    x=boxes['nick'][0];bottom=boxes['stack'][3]
    return [max(0,x-40),max(0,bottom-42),min(width,x+88),min(height,bottom+6)]


def _ncc(a,b):
    a=a.astype(np.float64).ravel();b=b.astype(np.float64).ravel()
    a-=a.mean();b-=b.mean()
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/denominator) if denominator>0 else -1.


def _glyph_ncc(a,b):
    a=np.maximum(a.astype(np.float64)-np.median(a),0).ravel()
    b=np.maximum(b.astype(np.float64)-np.median(b),0).ravel()
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/denominator) if denominator>0 else -1.


class EmptySeatButtonModel:
    def __init__(self, root, model_path=None):
        self.root=Path(root)
        path=Path(model_path) if model_path else self.root/'models/empty_seat_buttons.json'
        raw=path.read_bytes();self.sha256=hashlib.sha256(raw).hexdigest()
        self.model=json.loads(raw)
        legacy=(self.model['schema']=='native-full-empty-seat-button-v1' and
                self.model['full_ncc_min']>=.98 and self.model['text_ncc_min']>=.98 and
                self.model['border_ncc_min']>=.98 and self.model.get('border_check_enabled',True) is True)
        calibrated=(self.model['schema']=='native-full-text-border-empty-seat-button-v2' and
                self.model['full_ncc_min']==.91 and self.model['text_ncc_min']==.95 and
                self.model['border_ncc_min']==.08 and self.model.get('border_check_enabled') is True)
        if (not (legacy or calibrated) or self.model['translation_radius']>2):
            raise ValueError('EMPTY_BUTTON_MODEL_CONTRACT_UNKNOWN')
        self.templates=[]
        for row in self.model['templates']:
            gray=np.frombuffer(base64.b64decode(row['pixels_base64'],validate=True),np.uint8).reshape(row['shape'])
            if row.get('label')!='EMPTY' or hashlib.sha256(gray.tobytes()).hexdigest()!=row['pixel_sha256']:
                raise ValueError('EMPTY_BUTTON_TEMPLATE_HASH_CONFLICT')
            self.templates.append((row['template_id'],gray))
        self.slots=json.loads((self.root/'configs/layout_847x404.json').read_text('utf-8'))['slots']

    def classify(self,gray):
        best=None
        radius=self.model['translation_radius'];x,y,X,Y=self.model['text_box']
        for name,template in self.templates:
            if gray.shape!=template.shape:continue
            height,width=gray.shape
            for dy in range(-radius,radius+1):
                for dx in range(-radius,radius+1):
                    a,b=max(0,-dx),max(0,-dy);A,B=min(width,width-dx),min(height,height-dy)
                    full=_ncc(template[b:B,a:A],gray[b+dy:B+dy,a+dx:A+dx])
                    if full<self.model['full_ncc_min']:continue
                    glyph=_glyph_ncc(template[y:Y,x:X],gray[y+dy:Y+dy,x+dx:X+dx])
                    if glyph<self.model['text_ncc_min']:continue
                    border=np.ones((B-b,A-a),dtype=bool)
                    border[max(y-b,0):Y-b,max(x-a,0):X-a]=False
                    outline=_ncc(template[b:B,a:A][border],gray[b+dy:B+dy,a+dx:A+dx][border])
                    if outline<self.model['border_ncc_min']:continue
                    candidate=dict(status='EMPTY',template_id=name,translation=[dx,dy],
                        full_ncc=full,text_ncc=glyph,border_ncc=outline,model_sha256=self.sha256)
                    if best is None or min(full,glyph)>min(best['full_ncc'],best['text_ncc']):best=candidate
        return best or dict(status='UNKNOWN',reason='COMPLETE_EMPTY_BUTTON_NOT_PROVEN',model_sha256=self.sha256)

    def capture(self,rgb,source,boxes):
        observations={}
        for seat,b in boxes.items():
            roi=button_roi(b,rgb.shape[1],rgb.shape[0]);x,y,X,Y=roi
            gray=cv2.cvtColor(np.ascontiguousarray(rgb[y:Y,x:X]),cv2.COLOR_RGB2GRAY)
            classified=self.classify(gray)
            row=dict(source,seat=seat,roi=roi,**classified)
            if classified['status']=='EMPTY':
                ok,png=cv2.imencode('.png',gray)
                if not ok:raise ValueError('EMPTY_BUTTON_RAM_ENCODING_FAILED')
                row.update(pixel_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),
                    png_base64=base64.b64encode(png).decode('ascii'),shape=list(gray.shape))
            observations[seat]=row
        return observations

    def verify(self,row,source):
        try:
            if any(row.get(k)!=source.get(k) for k in ('table_id','session_id','hand_id','source_frame','source_pts','time_base')):
                return False
            if row.get('model_sha256')!=self.sha256 or row.get('status')!='EMPTY':return False
            boxes=self.slots[f'slot_{source["table_id"]:02d}']['seats'][row['seat']]
            if row['roi']!=button_roi(boxes,847,404):return False
            pixels=cv2.imdecode(np.frombuffer(base64.b64decode(row['png_base64'],validate=True),np.uint8),cv2.IMREAD_GRAYSCALE)
            x,y,X,Y=row['roi']
            if (pixels is None or pixels.shape!=(Y-y,X-x) or row.get('shape')!=[Y-y,X-x] or
                    hashlib.sha256(pixels.tobytes()).hexdigest()!=row['pixel_sha256']):
                return False
            checked=self.classify(pixels)
            return (checked['status']=='EMPTY' and checked['template_id']==row.get('template_id') and
                    checked['translation']==row.get('translation') and
                    all(abs(checked[k]-row[k])<1e-9 for k in ('full_ncc','text_ncc','border_ncc')))
        except (KeyError,TypeError,ValueError,IndexError,cv2.error):
            return False


def frame_partition(source,seats,scores,threshold,empty_observations,labels=()):
    """One joint frame. No negative EMPTY inference can prove an occupied seat."""
    unknown=lambda reason:dict(status='UNKNOWN',reason=reason)
    positives={}
    for seat,row in (empty_observations or {}).items():
        if seat not in seats or row.get('status')!='EMPTY':continue
        if row.get('seat')!=seat or any(row.get(k)!=source.get(k) for k in
                ('table_id','session_id','hand_id','source_frame','source_pts','time_base')):
            return unknown('EMPTY_BUTTON_SOURCE_CONTEXT_UNKNOWN')
        positives[seat]=row
    cards={s for s in seats if scores.get(s) is not None and scores[s]>=threshold}
    occupied_labels=set()
    for seat,row in dict(labels).items():
        if seat not in seats or any(row.get(k)!=source.get(k) for k in
                ('table_id','session_id','hand_id','source_frame')):
            return unknown('NICKNAME_LABEL_SOURCE_CONTEXT_UNKNOWN')
        if row.get('state')=='OCCUPIED':occupied_labels.add(seat)
    if set(positives)&(cards|occupied_labels):
        return dict(status='CONFLICT',reason='EMPTY_BUTTON_CONTRADICTS_CARDS_OR_OCCUPIED_LABEL',
                    conflicting_seats=sorted(set(positives)&(cards|occupied_labels)))
    if not positives:return unknown('NO_POSITIVE_EMPTY_BUTTON')
    required=set(seats)-set(positives)
    if len(required)<2 or cards!=required:return unknown('ALL_OTHER_SEATS_REQUIRE_POSITIVE_FROZEN_CARDS')
    return dict(status='PROVEN_FRAME',reason=None,empty_seats=sorted(positives),occupied_seats=sorted(required),
                empty_evidence=positives,proof=RULE)
