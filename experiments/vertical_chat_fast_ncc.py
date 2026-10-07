"""Same overlap Pearson NCC, FFT evaluation; no new tracking heuristics."""
import numpy as np


def scores_fft(old, new, fixed=False, min_overlap=48):
    if fixed: old,new=old[:,:-12],new[:,:-12]
    a=old[:,::3].astype(np.float64); b=new[:,::3].astype(np.float64)
    h,w=a.shape; size=1<<(2*h-1).bit_length()
    products=np.fft.irfft(np.sum(np.fft.rfft(a,n=size,axis=0)*np.conj(np.fft.rfft(b,n=size,axis=0)),axis=1),n=size)[:h-min_overlap+1]
    def prefix(x): return np.r_[0.,np.cumsum(x.sum(axis=1))]
    pa,pb=prefix(a),prefix(b); paa,pbb=prefix(a*a),prefix(b*b)
    shifts=np.arange(len(products)); n=(h-shifts)*w
    sa=pa[h]-pa[shifts]; sb=pb[h-shifts]
    va=paa[h]-paa[shifts]-sa*sa/n; vb=pbb[h-shifts]-sb*sb/n
    denom=np.sqrt(np.maximum(va,0)*np.maximum(vb,0))
    return np.divide(products-sa*sb/n,denom,out=np.full(len(products),-1.),where=denom>0)


def correlation_fft(old,new,fixed=False,min_overlap=48):
    scores=scores_fft(old,new,fixed,min_overlap); order=np.argsort(scores)[::-1]; shift=int(order[0])
    second=max((float(scores[i]) for i in range(len(scores)) if abs(i-shift)>2),default=-1.)
    best=float(scores[shift])
    return {'shift':shift,'score':best,'margin':best-second,'runner_up_score':second,
            'top_shifts':[[int(i),float(scores[i])] for i in order[:5]],'accepted':best>=.90 and best-second>=.02}
