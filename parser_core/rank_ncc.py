"""Native CHAT-rank adapter for the existing grayscale NCC template method.

No production caller is enabled by importing this module. SuitBackup remains
byte-for-byte unchanged. Geometry outside the verified CHAT domain fails closed.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from parser_core.board_tiles import _glyph_boxes, _glyphs

RANKS = tuple('23456789TJQKA')

def rank_feature(rgb: np.ndarray, method: str = 'component_ncc', size: int = 16):
    """Return centered unit-norm native grayscale ROI and its coordinates.

    Component bounds accommodate native origin shifts and 10's connected glyph;
    multiple/disconnected unexpected glyphs fail closed, never select by score.
    Resizing/mean subtraction/L2 norm match existing suit NCC preprocessing.
    """
    if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] != 16 or
            not 18 <= rgb.shape[1] <= 22):
        return None, None, 'UNSUPPORTED_CHAT_GEOMETRY'
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if method == 'fixed_ncc':
        if rgb.shape[1] not in (20,21):
            return None,None,'UNSUPPORTED_FIXED_ROI_GEOMETRY'
        box = [1,2,10,13]
    elif method == 'component_ncc':
        boxes = _glyph_boxes(gray,0,gray.shape[1]-1)
        if boxes is None:
            return None,None,'RANK_COMPONENT_COUNT_UNRESOLVED'
        x,y,w,h = boxes[0];sx,sy,sw,sh = boxes[1]
        if not (1 <= w <= 9 and 5 <= h <= 9 and 2 <= y and y+h <= 13 and x+w < sx):
            return None,None,'RANK_COMPONENT_BOUNDS_UNRESOLVED'
        box = [x,y,x+w,y+h]
    else:
        raise ValueError('Unknown rank NCC normalization')
    x1,y1,x2,y2=box
    crop = gray[y1:y2,x1:x2]
    feature = cv2.resize(crop,(size,size),interpolation=cv2.INTER_LINEAR).astype(np.float32)
    feature -= feature.mean()
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-8:
        return None,box,'EMPTY_RANK_EVIDENCE'
    feature /= norm
    return feature.reshape(-1),box,None


def rank_geometry_fallback(rgb: np.ndarray, result: dict) -> dict:
    """Resolve only an already rejected T/Q tie using tiny mask geometry.

    The gate is deliberately closed for every other candidate pair.  A dense
    central stem is a T receipt; a lower-right tail with no central stem is a
    Q receipt.  Borderline masks remain unresolved.
    """
    candidates = {result.get('rank_candidate'), result.get('rank_second_candidate')}
    if candidates != {'T', 'Q'} or result.get('rank_accepted'):
        return result
    # Keep the fallback inside the registered ambiguity envelope: a usable
    # primary match, two close candidates, and the ordinary margin gate as the
    # reason for abstention. Very weak or non-ambiguous T/Q scores stay closed.
    best = result.get('rank_best_score')
    second = result.get('rank_second_score')
    margin = result.get('rank_margin')
    if (best is None or second is None or margin is None or
            best < 0.71 or second < 0.65 or margin > 0.015):
        return {**result, 'rank_geometry_reason': 'TQ_GEOMETRY_OUTSIDE_ENVELOPE'}
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] != 16:
        return result
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    glyphs = _glyphs(gray, 0, gray.shape[1] - 1)
    if glyphs is None:
        return {**result, 'rank_geometry_reason': 'TQ_GEOMETRY_UNRESOLVED'}
    mask = glyphs[0].astype(bool)
    stem = float(mask[2:, 3:5].mean())
    tail = float(mask[5:, 5:].mean())
    # Thresholds are integer-grid boundaries measured on the saved native
    # corpus.  They intentionally leave the boundary cases unresolved.
    if stem >= 0.875:
        label = 'T'
    elif stem < 0.75 and tail > 0.75:
        label = 'Q'
    else:
        return {**result, 'rank_geometry_reason': 'TQ_GEOMETRY_UNRESOLVED',
                'rank_geometry_stem_ratio': stem, 'rank_geometry_tail_ratio': tail}
    return {**result, 'rank_geometry_fallback': True,
            'rank_geometry_method': 'NATIVE_GLYPH_8X8_STEM_TAIL_V1',
            'rank_geometry_stem_ratio': stem, 'rank_geometry_tail_ratio': tail,
            'rank_geometry_result': label, 'rank_result': label,
            'rank_accepted': True, 'rank_reason': 'RANK_GEOMETRY_FALLBACK_TQ'}

class RankNCC:
    def __init__(self, manifest: Path, *, method='component_ncc', threshold=1.01, margin=1.01,
                 expected_manifest_sha256: str | None = None):
        raw=manifest.read_bytes();self.manifest_sha256=hashlib.sha256(raw).hexdigest()
        if expected_manifest_sha256 and self.manifest_sha256!=expected_manifest_sha256:
            raise ValueError('Rank TRAIN manifest changed')
        self.method=method;self.threshold=float(threshold);self.margin=float(margin)
        self.rows=[json.loads(line) for line in raw.decode('utf8').splitlines()]
        vectors=[];labels=[];indices=[];self.rejected_templates=[]
        for i,row in enumerate(self.rows):
            payload=Path(row['tile_png']).read_bytes()
            if hashlib.sha256(payload).hexdigest()!=row['tile_sha256']:
                raise ValueError('Rank TRAIN PNG changed')
            rgb=np.asarray(Image.open(io.BytesIO(payload)).convert('RGB'))
            if hashlib.sha256(rgb.tobytes()).hexdigest()!=row['tile_pixel_sha256']:
                raise ValueError('Rank TRAIN pixels changed')
            label=row['ground_truth'][:-1]
            if label not in RANKS:raise ValueError('Invalid rank template label')
            vector,box,reason=rank_feature(rgb,method)
            if vector is None:
                self.rejected_templates.append(dict(index=i,reason=reason));continue
            vectors.append(vector);labels.append(label);indices.append(i)
        if set(labels)!=set(RANKS):raise ValueError('Rank TRAIN does not cover all13 classes')
        self.bank=np.stack(vectors);self.labels=np.array(labels);self.indices=np.array(indices)

    def read(self,rgb:np.ndarray,*,excluded_source: str | None = None) -> dict:
        vector,box,reason=rank_feature(rgb,self.method)
        result=dict(rank_candidate='UNRESOLVED',rank_second_candidate='UNRESOLVED',
                    rank_best_score=None,rank_second_score=None,rank_margin=None,
                    rank_accepted=False,rank_result='UNRESOLVED',rank_roi=box,
                    rank_normalization=self.method,rank_template_manifest_sha256=self.manifest_sha256)
        if vector is None:return {**result,'rank_reason':reason}
        similarities=self.bank@vector
        excluded_name=Path(excluded_source).name.lower() if excluded_source is not None else None
        allowed=np.array([excluded_name is None or Path(self.rows[int(i)]['source_video']).name.lower()!=excluded_name for i in self.indices])
        if not np.any(allowed):return {**result,'rank_reason':'NO_DISJOINT_TEMPLATES'}
        scores=[]
        for label in RANKS:
            hits=np.flatnonzero((self.labels==label)&allowed)
            if not len(hits):return {**result,'rank_reason':'SOURCE_HELD_OUT_CLASS_MISSING'}
            best=int(hits[np.argmax(similarities[hits])]);scores.append((label,float(similarities[best]),int(self.indices[best])))
        scores.sort(key=lambda x:(-x[1],RANKS.index(x[0])))
        first,second=scores[:2];margin=first[1]-second[1]
        accepted=first[1]>=self.threshold and margin>=self.margin
        return {**result,'rank_candidate':first[0],'rank_second_candidate':second[0],
                'rank_best_score':first[1],'rank_second_score':second[1],'rank_margin':margin,
                'rank_best_template_index':first[2],'rank_second_template_index':second[2],
                'rank_accepted':accepted,'rank_result':first[0] if accepted else 'UNRESOLVED',
                'rank_reason':None if accepted else 'RANK_NCC_LOW_CONFIDENCE'}
