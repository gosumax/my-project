"""Conservative native-component fallback for the pinned grayscale suit NCC.

Fixed-ROI SuitBackup stays primary. This adapter keeps the same TRAIN and
confidence gates; it never replaces an already confident fixed-ROI result.
"""
import hashlib
import io
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from parser_core.board_tiles import _glyph_boxes

def component_feature(rgb):
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.shape[0]!=16 or not 18<=rgb.shape[1]<=22:
        return None,None,'UNSUPPORTED_COMPONENT_GEOMETRY'
    gray=np.asarray(Image.fromarray(rgb).convert('L'))
    boxes=_glyph_boxes(gray,0,gray.shape[1]-1)
    if boxes is None:return None,None,'COMPONENT_COUNT_UNRESOLVED'
    x,y,w,h=boxes[1]
    if not (2<=w<=9 and 3<=h<=11 and x>=1 and y>=2 and x+w<=gray.shape[1] and y+h<=13):
        return None,None,'COMPONENT_BOUNDS_UNRESOLVED'
    box=[x,y,x+w,y+h]
    a=cv2.resize(gray[y:y+h,x:x+w],(16,16),interpolation=cv2.INTER_LINEAR).astype(np.float32)
    a-=a.mean();norm=float(np.linalg.norm(a))
    if norm<=1e-8:return None,box,'EMPTY_COMPONENT_EVIDENCE'
    return (a/norm).reshape(-1),box,None

class ComponentSuitBackup:
    def __init__(self,config_path):
        raw=Path(config_path).read_bytes();config=json.loads(raw)
        manifest=Path(config['train_manifest']);payload=manifest.read_bytes()
        self.manifest_sha=hashlib.sha256(payload).hexdigest()
        if self.manifest_sha!=config['train_manifest_sha256']:
            raise ValueError('Component suit TRAIN manifest changed')
        self.config_sha=hashlib.sha256(raw).hexdigest()
        self.implementation_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.threshold=float(config['best_score_threshold']);self.margin=float(config['margin_threshold'])
        self.rows=[json.loads(line) for line in payload.decode('utf8').splitlines()]
        features=[];labels=[];indices=[]
        for i,row in enumerate(self.rows):
            data=Path(row['tile_png']).read_bytes()
            if hashlib.sha256(data).hexdigest()!=row['tile_sha256']:raise ValueError('Component template PNG changed')
            rgb=np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))
            if hashlib.sha256(rgb.tobytes()).hexdigest()!=row['tile_pixel_sha256']:raise ValueError('Component template pixels changed')
            feature,box,reason=component_feature(rgb)
            if feature is None:continue
            features.append(feature);labels.append(row['ground_truth'][-1]);indices.append(i)
        if set(labels)!=set('shdc'):raise ValueError('Incomplete component suit bank')
        self.bank=np.stack(features);self.labels=np.array(labels);self.indices=np.array(indices)

    def read(self,rgb):
        feature,box,reason=component_feature(rgb)
        result=dict(template_suit='UNRESOLVED',template_confident=False,
                    suit_backup_mode='BACKUP',component_suit_roi=box,
                    component_suit_manifest_sha256=self.manifest_sha,
                    component_suit_implementation_sha256=self.implementation_sha,
                    suit_backup_config_sha256=self.config_sha)
        if feature is None:return {**result,'template_reason':reason}
        scores=self.bank@feature;classes=[]
        for suit in 'shdc':
            hits=np.flatnonzero(self.labels==suit);idx=int(hits[np.argmax(scores[hits])])
            classes.append((suit,float(scores[idx]),int(self.indices[idx])))
        classes.sort(key=lambda x:(-x[1],'shdc'.index(x[0])))
        best,second=classes[:2];margin=best[1]-second[1]
        x1,y1,x2,y2=box;native=rgb[y1:y2,x1:x2]
        return {**result,'template_suit':best[0],'template_best_score':best[1],
                'template_second_suit':second[0],'template_second_score':second[1],
                'template_margin':margin,'template_confident':best[1]>=self.threshold and margin>=self.margin,
                'template_reason':None if best[1]>=self.threshold and margin>=self.margin else 'TEMPLATE_LOW_CONFIDENCE',
                'component_suit_best_template_index':best[2],
                'component_suit_second_template_index':second[2],
                'component_suit_native_roi_pixel_sha256':hashlib.sha256(native.tobytes()).hexdigest()}
