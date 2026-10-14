"""Unchanged Golden finite score/order comparator, with canonical constants."""
import json
import math
import re
from types import SimpleNamespace
from parser_core import physical_chat_admission as physical
bridge = SimpleNamespace(study=physical)

SCORE=re.compile(r'^(?:candidates\[\d+\]\.)?(?:score|retained_rows\[\d+\]\.score|competing_retained_translations\[\d+\]\.score)$')

MARGIN=re.compile(r'^(?:candidates\[\d+\]\.)?margin$')

def normalized(value):
    def proof(v,path=''):
        if SCORE.fullmatch(path) or MARGIN.fullmatch(path):
            if type(v) is not float or not math.isfinite(v):
                raise ValueError('INVALID_INTERNAL_FLOAT:'+path+':'+type(v).__name__)
            threshold=bridge.study.NCC_MARGIN_MIN if MARGIN.fullmatch(path) else bridge.study.NCC_SCORE_MIN
            return dict(internal_float_threshold_side=v>=threshold)
        if isinstance(v,dict):return {k:proof(x,path+'.'+k if path else k) for k,x in v.items()}
        if isinstance(v,list):return [proof(x,path+f'[{i}]') for i,x in enumerate(v)]
        return v
    def walk(v):
        if isinstance(v,dict):
            if v.get('schema')=='physical-chat-admission-v1':
                return {k:proof(x) if k=='proof' else walk(x) for k,x in v.items()}
            return {k:walk(x) for k,x in v.items()}
        if isinstance(v,list):return [walk(x) for x in v]
        return v
    return walk(json.loads(json.dumps(value,allow_nan=False)))

def differences(a,b):
    result=[]
    def walk(x,y,p):
        if type(x)!=type(y):result.append((p,'type:'+type(x).__name__,'type:'+type(y).__name__));return
        if isinstance(x,dict):
            if x.keys()!=y.keys():result.append((p+'.keys',sorted(x),sorted(y)))
            for k in x.keys()&y.keys():walk(x[k],y[k],p+'.'+k)
        elif isinstance(x,list):
            if len(x)!=len(y):result.append((p+'.length',len(x),len(y)))
            for i,(xx,yy) in enumerate(zip(x,y)):walk(xx,yy,p+f'[{i}]')
        elif isinstance(x,float):
            if x.hex()!=y.hex():result.append((p,x,y))
        elif x!=y:result.append((p,x,y))
    walk(a,b,'root');return result

def compare(old,new):
    try:ds=differences(normalized(old),normalized(new))
    except ValueError as exc:ds=[('schema',str(exc),'INVALID')]
    return dict(semantic_exact=not ds,different_semantic_fields=len(ds),first_differences=ds[:15],
        numeric_whitelist='Only physical-chat-admission-v1 proof score and margin fields; finite float/type/field presence/threshold side preserved',
        ordering_scope='Full candidate order, visible competitor top3 order and every selected residual preserved (stronger than decision-only order)')
