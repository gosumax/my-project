"""Top/middle/bottom Pearson shift consensus. Experiment only, no OCR inputs."""
from collections import Counter
import numpy as np


def zone_scores(old,new,y1,y2,min_overlap=48):
    a=old[:,:-12:3].astype(np.float64); b=new[:,:-12:3].astype(np.float64)
    h,w=a.shape; size=1<<(2*h-1).bit_length()
    masked=np.zeros_like(b); masked[y1:y2]=b[y1:y2]
    dots=np.fft.irfft(np.sum(np.fft.rfft(a,n=size,axis=0)*np.conj(np.fft.rfft(masked,n=size,axis=0)),axis=1),n=size)
    shifts=np.arange(max(0,h-y1-min_overlap+1)); end=np.minimum(y2,h-shifts)
    support=end-y1; valid=support>=min_overlap; shifts=shifts[valid];end=end[valid];support=support[valid]
    def pre(x):return np.r_[0.,np.cumsum(x.sum(axis=1))]
    pa,pb=pre(a),pre(b); paa,pbb=pre(a*a),pre(b*b)
    sa=pa[shifts+end]-pa[shifts+y1];sb=pb[end]-pb[y1];n=support*w
    va=paa[shifts+end]-paa[shifts+y1]-sa*sa/n;vb=pbb[end]-pbb[y1]-sb*sb/n
    den=np.sqrt(np.maximum(va,0)*np.maximum(vb,0))
    scores=np.divide(dots[shifts]-sa*sb/n,den,out=np.full(len(shifts),-1.),where=den>0)
    return shifts,scores,support


def measure_zone(old,new,y1,y2):
    shifts,scores,support=zone_scores(old,new,y1,y2)
    if not len(shifts):return {'accepted':False,'shift_px':None,'score':None,'reason':'NO_OVERLAP'}
    order=np.argsort(scores)[::-1];idx=int(order[0]);d=int(shifts[idx]);score=float(scores[idx])
    second=max((float(scores[i]) for i in range(len(scores)) if abs(int(shifts[i])-d)>2),default=-1.)
    return {'shift_px':d,'score':score,'margin':score-second,'support_px':int(support[idx]),
            'accepted':score>=.90 and score-second>=.02,'y1':y1,'y2':y2,
            'top_shifts':[[int(shifts[i]),float(scores[i])] for i in order[:3]]}


def classify_zones(zones,tolerance=2.):
    accepted=[(name,z['shift_px']) for name,z in zones.items() if z['accepted']]
    if len(accepted)>=2:
        shifts=[d for _,d in accepted]
        # A reliable contradicting lower zone vetoes a 2/3 moving majority.
        # Otherwise "whole CHAT" would silently include a stationary region.
        if max(shifts)-min(shifts)>tolerance:
            return {'classification':'LOCAL_CHANGE','shift_px':None,'reason':'CONFIDENT_ZONE_SHIFT_CONFLICT','support_zones':[name for name,_ in accepted]}
        d=int(round(float(np.median(shifts))))
        if d>2:
            return {'classification':'GLOBAL_SCROLL','shift_px':d,'reason':'NONZERO_ZONE_CONSENSUS','support_zones':[name for name,_ in accepted]}
        return {'classification':'LOCAL_CHANGE','shift_px':None,'reason':'ZERO_ZONE_CONSENSUS_CHANGED_CROP','support_zones':[name for name,_ in accepted]}
    return {'classification':'UNRESOLVED','shift_px':None,'reason':'INSUFFICIENT_CONFIDENT_ZONES','support_zones':[name for name,_ in accepted]}


def global_scroll(old,new,tolerance=2.):
    h=old.shape[0];cuts=[0,h//3,2*h//3,h]
    zones={name:measure_zone(old,new,cuts[i],cuts[i+1]) for i,name in enumerate(['top','middle','bottom'])}
    result=classify_zones(zones,tolerance);result['zones']=zones;result['tolerance_px']=tolerance
    return result
