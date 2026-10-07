"""Frozen STACK algorithms extracted without threshold or recognition changes."""
import base64, hashlib, json, re, subprocess, sys, threading, queue, time
from collections import Counter, deque
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
MASK_CONFIG = ROOT / 'configs/stack_sitout_mask.json'
VERSION = 'sitout-fullword-mask-v1'
GATE_VERSION = 'stack-digit-presence-v1'
def metrics(a, b):
    a=a.astype(np.float32); b=b.astype(np.float32)
    raw_mae=float(np.abs(b-a).mean())
    # Outer two pixels contain panel borders/turn highlights; retain raw ROI,
    # but measure text changes in the fixed interior (no alignment or scaling).
    a=a[2:-2,2:-2]; b=b[2:-2,2:-2]
    delta=b-a
    residual=np.abs(delta-np.median(delta))
    x=a-a.mean(); y=b-b.mean()
    denom=float(np.linalg.norm(x)*np.linalg.norm(y))
    ncc=float((x*y).sum()/denom) if denom else float(np.array_equal(a,b))
    return dict(mae=raw_mae, residual_mae=float(residual.mean()),changed_pixels=int((residual>20).sum()),
                changed_fraction=float((residual>20).mean()), ncc=ncc,
                brightness_shift=float(np.median(delta)))

def different(m, threshold=.025):
    return (m['changed_fraction']>=threshold or m['changed_pixels']>=10) and m['ncc']<.995

class Track:
    def __init__(self, confirm=3, threshold=.025):
        self.confirm=confirm; self.threshold=threshold; self.stable=None; self.pending=None
        self.counts=Counter(); self.last_stable_seen=None
        self.rejected=[]
    def feed(self, gray, frame):
        m=metrics(self.stable,gray) if self.stable is not None else None
        if m is not None and not different(m,self.threshold):
            self.counts['unchanged_or_noise']+=1
            if m['mae']>0: self.counts['compression_noise_rejected']+=1
            if self.pending:
                self.counts['temporary_changes_rejected']+=1
                self.rejected.append(dict(first=self.pending['first'],last=frame-1,reason='RETURN_TO_STABLE'))
            self.pending=None; self.last_stable_seen=frame
            return None
        if self.pending is None or different(metrics(self.pending['gray'],gray),self.threshold):
            if self.pending:
                self.counts['unstable_candidates_rejected']+=1
                self.rejected.append(dict(first=self.pending['first'],last=frame-1,reason='CANDIDATE_CHANGED_BEFORE_CONFIRMATION'))
            self.pending=dict(gray=gray.copy(),first=frame,n=1,metrics=m)
        else: self.pending['n']+=1
        if self.pending['n']<self.confirm: return None
        p=self.pending; self.stable=p['gray']; self.pending=None; self.last_stable_seen=frame
        return dict(first_seen_frame=p['first'],confirmed_frame=frame,metrics=p['metrics'],gray=self.stable)

def segments(image, threshold=100):
    g=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    m=(g>threshold).astype('uint8');m[:,:6]=0;m[:,-6:]=0
    columns=m.sum(0);runs=[];start=None
    for x,v in enumerate(list(columns)+[0]):
        if v>=2 and start is None:start=x
        elif v<2 and start is not None:
            ys=np.where(m[:,start:x].sum(1)>0)[0]
            if len(ys):runs.append((start,int(ys[0]),x,int(ys[-1])+1,m[ys[0]:ys[-1]+1,start:x].copy()))
            start=None
    return runs,m

def normalized(mask):
    # Preserve relative width. A narrow '1' must not become a solid wide block.
    h,w=mask.shape; width=max(1,min(8,round(w*12/h)))
    a=cv2.resize(mask,(width,12),interpolation=cv2.INTER_NEAREST)
    out=np.zeros((12,8),np.uint8);out[:,(8-width)//2:(8-width)//2+width]=a
    return out

class DigitGate:
    def __init__(self, config):
        self.config=json.loads(Path(config).read_text('utf-8'))
        self.templates=np.array([t['mask'] for t in self.config['templates']],np.float32)
    def evaluate(self,image):
        cfg=self.config; runs,m=segments(image,cfg['binary_threshold']); glyphs=[]
        for x,y,X,Y,patch in runs:
            if cfg['height_min']<=Y-y<=cfg['height_max'] and y<=cfg['top_max'] and Y>=cfg['bottom_min'] and 1<=X-x<=cfg['width_max']:
                n=normalized(patch).astype('float32'); denom=self.templates.sum((1,2))+n.sum()
                scores=2*(self.templates*n).sum((1,2))/np.maximum(denom,1)
                glyphs.append(dict(box=[x,y,X,Y],digit_similarity=float(scores.max())))
        good=[g for g in glyphs if g['digit_similarity']>=cfg['similarity_min']]
        valid_pixels=sum(int(m[g['box'][1]:g['box'][3],g['box'][0]:g['box'][2]].sum()) for g in glyphs)
        coverage=valid_pixels/max(int(m.sum()),1)
        baseline_spread=max((g['box'][3] for g in glyphs),default=0)-min((g['box'][3] for g in glyphs),default=0)
        visible=(len(good)>=1 and len(good)/max(len(glyphs),1)>=cfg['matched_fraction_min'] and
            coverage>=cfg['coverage_min'] and baseline_spread<=cfg['baseline_spread_max'])
        return dict(status='DIGITS_VISIBLE' if visible else 'NOT_DIGITS_VISIBLE',
            score=float(min((g['digit_similarity'] for g in good),default=0)),method=GATE_VERSION,
            evidence=dict(glyphs=glyphs,good_count=len(good),glyph_count=len(glyphs),mask_coverage=coverage,baseline_spread=baseline_spread,
                temporal_status='STATIC_ONLY'))

class SitOutMask:
    def __init__(self,config=MASK_CONFIG):
        payload=Path(config).read_bytes();self.config=json.loads(payload.decode('utf-8'));self.template=np.array(self.config['template'],np.float32)
        self.config_sha256=hashlib.sha256(payload).hexdigest()
        self.code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def evaluate(self,image):
        gray=image if image.ndim==2 else cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        m=(gray>self.config['binary_threshold']).astype(np.float32);m[:,:6]=0;m[:,-6:]=0
        h,w=self.template.shape
        if m.shape[0]<h or m.shape[1]<w:return dict(matched=False,score=0.,box=None,method=VERSION)
        intersection=cv2.matchTemplate(m,self.template,cv2.TM_CCORR)
        foreground=cv2.matchTemplate(m,np.ones_like(self.template),cv2.TM_CCORR)
        scores=2*np.maximum(intersection,0)/np.maximum(foreground+self.template.sum(),1)
        y,x=np.unravel_index(np.argmax(scores),scores.shape);score=float(scores[y,x])
        return dict(matched=score>=self.config['min_dice'],score=score,box=[int(x),int(y),int(x+w),int(y+h)],method=VERSION,threshold=self.config['min_dice'],mask_config_sha256=self.config_sha256,mask_code_sha256=self.code_sha256)

def excluding_gate_class():
    # The frozen binary gate remains unchanged; this is a separate wrapper.
    class SitOutExcludingGate(DigitGate):
        def __init__(self,config):
            super().__init__(config);self.sitout=SitOutMask()
        def evaluate(self,image):
            result=super().evaluate(image);exclusion=self.sitout.evaluate(image)
            result['evidence']['sitout_exclusion']=exclusion
            if exclusion['matched']:
                result['evidence']['base_digit_gate_status']=result['status']
                result['status']='NOT_DIGITS_VISIBLE'
                result['evidence']['excluded_label']='SIT_OUT'
                result['evidence']['exclusion_reason']='FULL_WORD_VISUAL_MASK'
            return result
    return SitOutExcludingGate

class RawTiny:
    def __init__(self,out):
        cfg=json.loads((ROOT/'configs/ocr_tiny_local.json').read_text('utf-8'))
        self.log=(out/'ocr.stderr.log').open('w',encoding='utf-8');self.q=queue.Queue();self.seq=0
        self.p=subprocess.Popen([cfg['python'],'-u',str(Path(__file__).with_name('raw_worker.py')),str(ROOT/'configs/ocr_tiny_local.json')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        def reader():
            for line in self.p.stdout:self.q.put(line)
            self.q.put(None)
        threading.Thread(target=reader,daemon=True).start()
        try:
            self.ready=self.receive()
            if self.ready.get('type')!='ready':raise RuntimeError(self.ready)
        except Exception:
            self.close()
            raise
    def receive(self):
        deadline=time.monotonic()+180
        while True:
            line=self.q.get(timeout=max(.1,deadline-time.monotonic()))
            if line is None:raise RuntimeError('Tiny exited')
            try:return json.loads(line)
            except json.JSONDecodeError:self.log.write(line)
    def read(self,states):
        self.seq+=1
        items=[{k:s[k] for k in ('state_id','width','height','exact_hash')}|dict(raw_base64=base64.b64encode(s['_raw']).decode()) for s in states]
        self.p.stdin.write(json.dumps(dict(request_id=self.seq,items=items))+'\n');self.p.stdin.flush()
        reply=self.receive()
        if reply.get('request_id')!=self.seq or reply.get('error'):raise RuntimeError(reply)
        return reply
    def close(self):
        self.p.stdin.close()
        try:self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:self.p.terminate();self.p.wait(timeout=5)
        self.log.close()

class Timeline:
    def __init__(self,routes,gate,fps):
        self.routes=routes;self.gate=gate;self.fps=fps
        self.tracks=[Track() for _ in routes];self.pools=[{} for _ in routes]
        self.current=[None]*len(routes);self.recent=[deque(maxlen=2) for _ in routes]
        self.states=[];self.observations=[];self.counts=Counter()
    def feed(self,i,gray,frame):
        track=self.tracks[i];self.recent[i].append((frame,gray.copy()))
        if track.stable is None:
            track.stable=gray.copy();track.last_stable_seen=frame
            event=dict(gray=gray.copy(),first_seen_frame=frame,confirmed_frame=frame,metrics=None);origin='INITIAL'
        else:
            self.counts['comparisons']+=1;event=track.feed(gray,frame);origin='CHANGE'
        if event is None:
            if track.pending is None:self.current[i]['last_seen_frame']=frame
            return None
        raw=event['gray'].tobytes();digest=hashlib.sha256(raw).hexdigest()
        key=(gray.shape,digest);s=self.pools[i].get(key);recurrence=s is not None
        old=self.current[i]
        if recurrence:
            self.counts['exact_recurrences']+=1
        else:
            e=self.gate.evaluate(cv2.cvtColor(event['gray'],cv2.COLOR_GRAY2BGR))
            temporal=[]
            if origin=='CHANGE':
                for f,g in self.recent[i]:
                    v=self.gate.evaluate(cv2.cvtColor(g,cv2.COLOR_GRAY2BGR))
                    temporal.append(dict(frame=f,hash=hashlib.sha256(g.tobytes()).hexdigest(),status=v['status']))
                confirmed=len(temporal)==2 and temporal[1]['frame']==temporal[0]['frame']+1 and all(v['status']=='DIGITS_VISIBLE' for v in temporal)
                if not confirmed:e['status']='NOT_DIGITS_VISIBLE'
            e['evidence']['temporal_frames']=temporal;e['evidence']['temporal_status']='INITIAL_STATIC' if origin=='INITIAL' else ('CONFIRMED' if confirmed else 'WITHHELD')
            r=self.routes[i]
            s=dict(state_id=f'S{len(self.states)+1:05d}',table_id=r['table_id'],seat_id=r['seat_id'],frame_index=event['first_seen_frame'],timestamp=event['first_seen_frame']/self.fps,origin=origin,width=gray.shape[1],height=gray.shape[0],exact_hash=digest,previous_state_id=old['state_id'] if old else None,change_metrics=event['metrics'],digit_gate_status=e['status'],digit_gate_score=e['score'],digit_gate_evidence=e['evidence'],ocr_called=False,ocr_raw=None,ocr_numeric=None,ocr_confidence=None,_raw=raw)
            self.states.append(s);self.pools[i][key]=s
        if old and old['state_id']==s['state_id']:
            self.counts['same_canonical_confirmations']+=1;old['last_seen_frame']=frame
            return None
        o=dict(observation_id=len(self.observations)+1,table_id=s['table_id'],seat_id=s['seat_id'],state_id=s['state_id'],previous_state_id=old['state_id'] if old else None,frame_index=event['first_seen_frame'],confirmed_frame=frame,last_seen_frame=frame,timestamp=event['first_seen_frame']/self.fps,origin=origin,change_metrics=event['metrics'],observed_hash=digest,recurrence='EXACT' if recurrence else None)
        self.observations.append(o);self.current[i]=o
        return s if not recurrence and s['digit_gate_status']=='DIGITS_VISIBLE' else None

def numeric(raw,score):
    return raw.replace(' ','').replace('\u00a0','') if re.fullmatch(r'\d+(?:[ \u00a0]\d{3})*',raw.strip()) and score is not None and score>=.95 else None
