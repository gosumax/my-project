"""Frozen two-stage NCC, transferred unchanged from reviewed offline model."""
import hashlib,json
from pathlib import Path
CLASSES=('cards','Turn','Flop','River')
def sha(data):return hashlib.sha256(data).hexdigest()

class TwoStageModel:
    def __init__(self,config_path):
        import cv2,numpy as np
        self.config=json.loads(Path(config_path).read_text(encoding='utf-8'));self.templates={};self.classifier_calls=0;self.cache={}
        for word,meta in self.config['templates'].items():
            template_path=Path(meta['path']); template_path=template_path if template_path.is_absolute() else Path(config_path).resolve().parents[1]/template_path
            data=template_path.read_bytes()
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
