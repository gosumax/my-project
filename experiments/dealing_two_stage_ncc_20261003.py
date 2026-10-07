"""Isolated two-stage model and conservative causal row frontier.

Hand context is a separate input: only an independently observed, pixel-bound
hand header may supply it. Labels for Dealing cards are never model inputs.
"""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CLASSES=('cards','Turn','Flop','River')

def sha(data):return hashlib.sha256(data).hexdigest()

def freeze(out):
    import cv2,numpy as np
    old=ROOT/'runs/start_stack_frame_ledger_wide_smoke14_20261003/multi_template_classification'
    ref=Path('C:/ParserData/runs/smoke14_full_tiny_ssd_20260929_v3/crops/rows/table_95984db438dc55f85dee9067/000000681_018.png')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);folder=out/'templates';folder.mkdir(exist_ok=True)
    image=cv2.imdecode(np.fromfile(ref,dtype=np.uint8),0);prefix=image[:,4:91].copy()
    templates={'Dealer_Dealing':cv2.imencode('.png',prefix)[1].tobytes()}
    templates['Dealer_part']=cv2.imencode('.png',image[:,4:47].copy())[1].tobytes()
    templates['Dealing_part']=cv2.imencode('.png',image[:,47:91].copy())[1].tobytes()
    templates.update({word:(old/f'template_{word}.png').read_bytes() for word in CLASSES})
    manifest={}
    for word,data in templates.items():
        path=folder/f'{word}.png'
        if path.exists() and path.read_bytes()!=data:raise ValueError('FROZEN_TEMPLATE_CHANGED')
        if not path.exists():path.write_bytes(data)
        pixels=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),0)
        manifest[word]={'path':str(path.resolve()),'file_sha256':sha(data),'gray_sha256':sha(pixels.tobytes()),'shape':list(pixels.shape)}
    config={'schema':'experimental.dealing-two-stage.v2','prefix_threshold':.90,'margin_threshold':.20,'absolute_word_threshold_diagnostic':.95,'row_continuity_threshold':.985,'minimum_overlap_rows':3,'consecutive_confirmations':3,'prefix_search_x':[1,10+87],'word_search_x':[88,134],'gate_rule':'Dealer_part AND Dealing_part >=same0.90; common shift derived from Dealer. Full prefix score remains diagnostic.','templates':manifest,'source_prefix':{'path':str(ref),'file_sha256':sha(ref.read_bytes()),'rect':[4,0,91,16]},'margin_policy':'Retain prior0.20diagnostic value; no selection of0.213..0.251separating interval from old20hands.','freeze_policy':'No tuning on regression or newheldout. Structural gate conjunction fixes known v1 Dealer dominance; v1 evidence retained. No template bank, scale normalization or intensity correction.','production_modified':False}
    path=out/'frozen_config.json';raw=json.dumps(config,ensure_ascii=False,indent=2)+'\n'
    if path.exists() and path.read_text(encoding='utf-8')!=raw:raise ValueError('FROZEN_CONFIG_CHANGED')
    if not path.exists():path.write_text(raw,encoding='utf-8')
    return path

class TwoStageModel:
    def __init__(self,config_path):
        import cv2,numpy as np
        self.config=json.loads(Path(config_path).read_text(encoding='utf-8'));self.templates={};self.classifier_calls=0;self.cache={}
        for word,meta in self.config['templates'].items():
            data=Path(meta['path']).read_bytes()
            if sha(data)!=meta['file_sha256']:raise ValueError('TEMPLATE_HASH_MISMATCH')
            pixels=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),0)
            if sha(pixels.tobytes())!=meta['gray_sha256']:raise ValueError('TEMPLATE_PIXEL_HASH_MISMATCH')
            self.templates[word]=pixels
    def classify(self,gray):
        import cv2
        key=(gray.shape,sha(gray.tobytes()))
        if key in self.cache:return dict(self.cache[key])
        if gray.ndim!=2 or gray.shape[0]!=16 or gray.shape[1]<134:
            return {'status':'UNKNOWN','reason':'ROW_GEOMETRY_UNKNOWN','gate_pass':False,'gate_score':None,'scores':None,'winner':None,'runner_up':None,'margin':None,'pixel_hash':key[1]}
        cfg=self.config;a,b=cfg['prefix_search_x'];match=cv2.matchTemplate(gray[:,a:b],self.templates['Dealer_Dealing'],cv2.TM_CCOEFF_NORMED);_,gate,_,loc=cv2.minMaxLoc(match)
        part_scores=None
        if 'Dealer_part' in self.templates:
            match=cv2.matchTemplate(gray[:,1:53],self.templates['Dealer_part'],cv2.TM_CCOEFF_NORMED);_,dealer,_,dealer_loc=cv2.minMaxLoc(match)
            shift=1+dealer_loc[0]-4
            # Dealing must occupy its expected position relative to Dealer;
            # it cannot borrow a match from another word or physical row.
            left=max(0,47+shift-1);right=min(gray.shape[1],91+shift+2)
            match=cv2.matchTemplate(gray[:,left:right],self.templates['Dealing_part'],cv2.TM_CCOEFF_NORMED);_,dealing,_,_=cv2.minMaxLoc(match)
            part_scores={'Dealer':float(dealer),'Dealing':float(dealing)}
        passed=min(part_scores.values())>=cfg['prefix_threshold'] if part_scores else gate>=cfg['prefix_threshold']
        result={'status':'UNKNOWN','reason':'DEALING_PREFIX_NOT_CONFIRMED','gate_pass':passed,'gate_score':float(gate),'gate_part_scores':part_scores,'gate_shift':[a+loc[0],loc[1]],'scores':None,'winner':None,'runner_up':None,'margin':None,'pixel_hash':key[1]}
        if result['gate_pass']:
            self.classifier_calls+=1;scores={};shifts={};a,b=cfg['word_search_x']
            for word in CLASSES:
                match=cv2.matchTemplate(gray[:,a:b],self.templates[word],cv2.TM_CCOEFF_NORMED);_,score,_,xy=cv2.minMaxLoc(match);scores[word]=float(score);shifts[word]=[a+xy[0],xy[1]]
            order=sorted(scores,key=lambda k:(-scores[k],k));winner,runner=order[:2];margin=scores[winner]-scores[runner]
            result.update(scores=scores,shifts=shifts,winner=winner,runner_up=runner,margin=margin,absolute_95_pass=scores[winner]>=cfg['absolute_word_threshold_diagnostic'])
            if margin>=cfg['margin_threshold']:result.update(status=winner,reason='DEALING_PREFIX_AND_CLASS_CONFIRMED')
            else:result['reason']='CLASS_MARGIN_UNKNOWN'
        self.cache[key]=dict(result);return result

class CausalRowFrontier:
    """One table. Proves only ordered append supported by >=3 prior rows.

    Unknown transitions reset row identity, never manufacture fresh rows.
    Gray arrays remain borrowed references only until the next frame.
    """
    def __init__(self,threshold=.985,min_overlap=3):
        self.threshold=threshold;self.min_overlap=min_overlap;self.previous=[];self.counter=0;self.last_frame=None
    def _new(self,row,frame,confirmed):
        self.counter+=1
        return dict(row,track_id=self.counter,birth_frame=frame,birth_row_index=row.get('row_index'),append_proven=confirmed,freshness='FRESH_APPEND' if confirmed else 'UNKNOWN')
    def _same(self,a,b):
        import cv2
        aa=a['gray'];bb=b['gray']
        if aa.shape!=bb.shape:return False
        if a.get('pixel_hash') is not None and a.get('pixel_hash')==b.get('pixel_hash'):return True
        width=max(1,aa.shape[1]-12)
        value=float(cv2.matchTemplate(bb[:,:width],aa[:,:width],cv2.TM_CCOEFF_NORMED)[0,0])
        return value>=self.threshold
    def observe(self,frame,rows):
        old=self.previous
        if self.last_frame is not None and frame!=self.last_frame+1:old=[]
        self.last_frame=frame
        if not old or not rows:
            result=[self._new(r,frame,False) for r in rows]
        elif len(old)==len(rows) and all(self._same(a,b) for a,b in zip(old,rows)):
            result=[dict(r,**{k:a[k] for k in ['track_id','birth_frame','birth_row_index','append_proven']},freshness='INHERITED' if a['append_proven'] else 'UNKNOWN') for a,r in zip(old,rows)]
        else:
            matches=[]
            for count in range(self.min_overlap,min(len(old),len(rows))+1):
                if all(self._same(a,b) for a,b in zip(old[-count:],rows[:count])):matches.append(count)
            if len(matches)!=1:
                result=[self._new(r,frame,False) for r in rows]
            else:
                count=matches[0]
                result=[dict(r,**{k:a[k] for k in ['track_id','birth_frame','birth_row_index','append_proven']},freshness='INHERITED' if a['append_proven'] else 'UNKNOWN') for a,r in zip(old[-count:],rows[:count])]
                # NCC overlap alone is a candidate, not physical lifecycle
                # proof (historical/repeated viewport may share this block).
                result.extend(dict(self._new(r,frame,False),overlap_append_candidate=True) for r in rows[count:])
        self.previous=result;return result

def bind_current(row,hand_id,header_frame,header_row_index):
    if hand_id is None or header_frame is None:return False,'CURRENT_HAND_UNKNOWN'
    if not row.get('append_proven'):return False,'ROW_FRESHNESS_UNKNOWN'
    birth=row['birth_frame']
    if birth<header_frame:return False,'OLD_ROW_BEFORE_CURRENT_HAND'
    if birth==header_frame and (header_row_index is None or row.get('birth_row_index') is None or row['birth_row_index']<=header_row_index):return False,'SAME_FRAME_HAND_ROW_ORDER_UNKNOWN_OR_OLD'
    return True,'CONFIRMED_APPEND_AFTER_CURRENT_HAND_HEADER'

class OncePerHandLatch:
    def __init__(self,confirmations=3):self.confirmations=confirmations;self.fired=set();self.pending={}
    def observe(self,table,hand_id,frame,candidates):
        key=(table,hand_id)
        if hand_id is None:return None,'CURRENT_HAND_UNKNOWN'
        if key in self.fired:return None,'ALREADY_TRIGGERED'
        if len(candidates)!=1:
            self.pending.pop(key,None);return None,'NO_CONFIRMED_FRESH_CARDS' if not candidates else 'MULTIPLE_FRESH_CARDS_ROWS_UNKNOWN'
        row=candidates[0];prev=self.pending.get(key);same=bool(prev and prev['last']+1==frame and prev['track_id']==row['track_id'])
        state={'first':prev['first'] if same else frame,'last':frame,'count':prev['count']+1 if same else 1,'track_id':row['track_id']};self.pending[key]=state
        if state['count']<self.confirmations:return None,'CONFIRMATION_PENDING'
        self.fired.add(key);self.pending.pop(key,None)
        return {'table':table,'hand_id':hand_id,'first_appearance_frame':state['first'],'trigger_frame':frame,'track_id':row['track_id'],'row_index':row['row_index'],'pixel_hash':row['pixel_hash']},'TRIGGER'

class HandStateTrigger:
    """Current-hand PREFLOP marker; no physical-row birth or row ID required.

    Independently pixel-confirmed numbered headers open a hand epoch. Before
    admission the current header must be uniquely co-visible above the cards
    row. A cached y/index is never used after scrolling. Once admitted the
    latch stays closed for this hand through any viewport or action changes.
    """
    def __init__(self,confirmations=1):
        self.confirmations=confirmations;self.states={};self.fired=set();self.header_hashes={}
    def confirm_hand(self,table,hand_id,frame,header_pixel_hash):
        if hand_id is None or not header_pixel_hash:return False
        self.header_hashes.setdefault(table,{})[header_pixel_hash]=hand_id
        old=self.states.get(table)
        if old and old['hand_id']==hand_id:return True
        if old and int(hand_id)<=int(old['hand_id']):return False
        self.states[table]={'hand_id':hand_id,'header_hash':header_pixel_hash,
                            'header_frame':frame,'pending':None}
        return True
    def observe(self,table,frame,rows):
        state=self.states.get(table)
        if not state:return None,'CURRENT_HAND_UNKNOWN',set()
        key=(table,state['hand_id'])
        if key in self.fired:return None,'ALREADY_TRIGGERED',set()
        headers=[r for r in rows if r['pixel_hash']==state['header_hash']]
        if len(headers)!=1:
            state['pending']=None
            return None,'CURRENT_HAND_HEADER_NOT_UNIQUELY_VISIBLE',set()
        boundary=headers[0]['row_bounds'][1]
        other_headers=[r for r in rows if r['row_bounds'][0]>=boundary and
                       (r.get('hand_header_id') or self.header_hashes.get(table,{}).get(r['pixel_hash'])) not in (None,state['hand_id'])]
        upper=min((r['row_bounds'][0] for r in other_headers),default=float('inf'))
        candidates=[r for r in rows if r.get('status')=='cards' and r['row_bounds'][0]>=boundary and r['row_bounds'][1]<=upper]
        eligible={r['row_index'] for r in candidates}
        if len(candidates)!=1:
            state['pending']=None
            return None,'NO_CURRENT_HAND_CARDS' if not candidates else 'MULTIPLE_CURRENT_HAND_CARDS_UNKNOWN',eligible
        row=candidates[0];previous=state['pending']
        consecutive=bool(previous and previous['last']+1==frame)
        pending={'first':previous['first'] if consecutive else frame,'last':frame,
                 'count':previous['count']+1 if consecutive else 1}
        state['pending']=pending
        if pending['count']<self.confirmations:return None,'CONFIRMATION_PENDING',eligible
        self.fired.add(key);state['pending']=None
        return {'table':table,'hand_id':state['hand_id'],'first_appearance_frame':pending['first'],
                'trigger_frame':frame,'row_index':row['row_index'],'row_bounds':row['row_bounds'],
                'pixel_hash':row['pixel_hash'],'header_pixel_hash':state['header_hash'],
                'header_confirmed_frame':state['header_frame'],'marker':'PREFLOP_DEALING_CARDS',
                'binding':'CO_VISIBLE_CURRENT_HAND_HEADER_ORDER','row_birth_required':False},'TRIGGER',eligible
