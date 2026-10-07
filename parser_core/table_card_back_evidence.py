"""Additional positive fixed-ROI card backs; the frozen recognizer is unchanged."""
import base64
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


class CardBackSupplement:
    def __init__(self, root):
        root=Path(root)
        raw=(root/'models/table_card_back_supplement.json').read_bytes()
        self.sha256=hashlib.sha256(raw).hexdigest()
        model=json.loads(raw)
        if model['schema']!='source-bound-table-card-back-supplement-v1' or model['threshold']<.98:
            raise ValueError('CARD_BACK_SUPPLEMENT_CONTRACT_UNKNOWN')
        self.threshold=model['threshold']
        slots=json.loads((root/'configs/layout_847x404.json').read_text('utf-8'))['slots']
        self.templates={}
        for row in model['templates']:
            key=(row['table_id'],row['seat'])
            roi=slots[f'slot_{key[0]:02d}']['seats'][key[1]]['shown_cards']
            pixels=np.frombuffer(base64.b64decode(row['pixels_base64'],validate=True),np.uint8).reshape(row['shape'])
            x,y,X,Y=roi
            if (key in self.templates or row['label']!='TWO_VISIBLE_CARD_BACKS' or row['roi']!=roi or
                    pixels.shape!=(Y-y,X-x,3) or hashlib.sha256(pixels.tobytes()).hexdigest()!=row['pixel_sha256']):
                raise ValueError('CARD_BACK_SUPPLEMENT_SOURCE_CONFLICT')
            self.templates[key]=(row,pixels)

    def match(self,table,seat,roi,bgr):
        entry=self.templates.get((table,seat))
        if entry is None:return None
        row,pixels=entry
        if list(roi)!=row['roi'] or bgr.shape!=pixels.shape:return None
        score=float(cv2.matchTemplate(bgr,pixels,cv2.TM_CCOEFF_NORMED)[0,0])
        if not np.isfinite(score) or score<self.threshold:return None
        return dict(rule='POSITIVE_NATIVE_CARD_BACK_SUPPLEMENT_V1',score=score,
                    template_id=row['template_id'],model_sha256=self.sha256,roi=list(roi))
