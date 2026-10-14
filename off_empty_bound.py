"""OFF necessary-condition rejection for OLD's centered normalized correlation.

For unit vectors u,v, ||u-v||_2 >= ||u-v||_1/sqrt(n), hence
NCC <= 1 - ||u-v||_1**2/(2*n). Reject only if this upper bound is strictly
below OLD's full NCC threshold, with conservative float64 roundoff slack.
No learned threshold, template selection shortcut, or pixel resampling.
"""
from collections import Counter
import cv2
import numpy as np

ROUNDOFF_SLACK=1e-9
MAX_COMPONENTS=100000


class BoundedSadDispatcher:
    def __init__(self, original):
        self.original=original
        self.counts=Counter()
        self.prepared={}

    def _prepare(self, model):
        from off_empty_pixels import aligned_pairs
        groups={}
        for _,template in model.templates:
            groups.setdefault(template.shape,[])
        for shape in groups:
            dummy=np.zeros(shape,dtype=np.uint8)
            for name,dx,dy,a,b in aligned_pairs(model,dummy):
                x=a.astype(np.float64).ravel();x-=x.mean()
                norm=np.linalg.norm(x)
                unit=x/norm if norm>0 else None
                if unit is not None:unit.flags.writeable=False
                groups[shape].append((dx,dy,unit,a.size))
        self.prepared[model.sha256]=groups
        return groups

    def decide(self, model,gray):
        if gray.dtype!=np.uint8 or gray.ndim!=2 or not gray.size or gray.size>MAX_COMPONENTS:
            self.counts['OLD']+=1
            return self.original(model,gray)
        groups=self.prepared.get(model.sha256)
        if groups is None:groups=self._prepare(model)
        prepared=groups.get(gray.shape)
        if not prepared:
            self.counts['OLD']+=1
            return self.original(model,gray)
        height,width=gray.shape
        # Same overlap/normalization is reused for all templates at a given dx/dy.
        # It is computed from current input only and is never cached across calls.
        current={}
        for dx,dy,u,n in prepared:
            if u is None:continue # OLD full NCC = -1 for a constant template.
            key=(dx,dy)
            if key not in current:
                a,b=max(0,-dx),max(0,-dy);A,B=min(width,width-dx),min(height,height-dy)
                y=gray[b+dy:B+dy,a+dx:A+dx].astype(np.float64).ravel();y-=y.mean()
                norm=np.linalg.norm(y)
                current[key]=y/norm if norm>0 else None
            v=current[key]
            if v is None:continue # OLD full NCC = -1 for constant observed overlap.
            l1=float(cv2.norm(u,v,cv2.NORM_L1))
            upper=1.-l1*l1/(2.*n)+ROUNDOFF_SLACK
            if not np.isfinite(upper) or upper>=model.model['full_ncc_min']:
                self.counts['OLD']+=1
                return self.original(model,gray)
        self.counts['SAD_BOUND']+=1
        return dict(status='UNKNOWN',reason='COMPLETE_EMPTY_BUTTON_NOT_PROVEN',model_sha256=model.sha256)
