"""Offline CARDS-only diagnostic and one trigger per confirmed hand.

No production imports or edits. Current-hand keys are supplied by reviewed
offline evidence, not inferred from a persistent CHAT word.
"""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from start_stack_wide_20261003 import ROOT,PARENT,database,table,read_jsonl,dump,jsonl,digest
THRESHOLD=.95
WORD_RECT=(91,0,121,16)  # Literal cards only: its foreground is x92..119.

class OncePerHandTrigger:
    def __init__(self):
        self.emitted=set();self.pending={}
    def observe(self,hand,frame,score,xy,confirmed=False,row_current_hand_confirmed=False):
        if not confirmed or not row_current_hand_confirmed or hand is None:
            if hand is not None:self.pending.pop(hand,None)
            return None
        if hand in self.emitted:return None
        previous=self.pending.get(hand)
        if score<THRESHOLD:
            self.pending.pop(hand,None);return None
        consecutive=bool(previous and previous['last']+1==frame and abs(previous['xy'][0]-xy[0])<=1 and abs(previous['xy'][1]-xy[1])<=1)
        state={'first':previous['first'] if consecutive else frame,'last':frame,'count':previous['count']+1 if consecutive else 1,'xy':xy}
        self.pending[hand]=state
        if state['count']<3:return None
        self.emitted.add(hand);self.pending.pop(hand,None)
        return {'first_appearance_frame':state['first'],'trigger_frame':frame,'confirmation':'3 consecutive cards-word hits at stable position + supplied confirmed current-hand context'}

def run(out):
    import cv2,numpy as np
    dest=out/'cards_word_trigger';dest.mkdir(exist_ok=True)
    reference=PARENT/'crops/rows/table_95984db438dc55f85dee9067/000000681_018.png'
    image=cv2.imdecode(np.fromfile(reference,dtype=np.uint8),0)
    template=image[0:16,91:121].copy()
    dump(dest/'template_manifest.json',{'source':str(reference),'file_sha256':digest(reference),'rect':WORD_RECT,'literal':'cards','threshold':THRESHOLD,'template_gray_sha256':hashlib.sha256(template.tobytes()).hexdigest(),'threshold_tuning':False,'production_changed':False})
    hands={(c['table'],c['hand_id']):c for c in read_jsonl(out/'selected_hands.jsonl')}
    profile=json.loads((ROOT/'configs/layout_847x404.json').read_text(encoding='utf-8'))
    original=read_jsonl(out/'ncc_dealing_cards_trigger/ncc_frame_scores.jsonl')
    # These positions are independent offline row-review annotations, not
    # inferred by the cards-word detector. Beyond the initial reviewed
    # appearance/triple, fresh-row association remains UNKNOWN.
    expected=read_jsonl(out/'t04_ncc_acceptance/recheck_expected_frames.jsonl')
    expected_y={(p['table'],p['hand_id']):p['match_chat_local_xy'][1] for p in expected}
    state=OncePerHandTrigger();probes=[];events=[];db=database()
    for p in original:
        key=(p['table'],p['hand_id']);c=hands[key];im,meta=table(db,c['session'],p['frame'])
        box=profile['slots'][f"slot_{int(c['table'][1:]):02d}"]['chat']['chat_log']
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);chat=gray[box[1]:,box[0]:box[2]]
        # Constrain search to the suffix's expected horizontal position. Only
        # the cards word is scored; Dealer/Dealing never enter correlation.
        x1=88;x2=125
        corr=cv2.matchTemplate(chat[:,x1:x2],template,cv2.TM_CCOEFF_NORMED)
        _,score,_,loc=cv2.minMaxLoc(corr);xy=[loc[0]+x1,loc[1]]
        current=p['frame']>=c['deal_chat_frame']
        fresh=current and p['frame']<=c['deal_chat_frame']+2 and abs(xy[1]-expected_y[key])<=1
        ev=state.observe(key,p['frame'],score,xy,confirmed=current,row_current_hand_confirmed=fresh)
        record=dict(meta,table=c['table'],hand_id=c['hand_id'],score=score,ncc_hit=score>=THRESHOLD,match_chat_local_xy=xy,current_hand_context_confirmed_offline=current,row_current_hand_confirmed_offline=fresh)
        probes.append(record)
        if ev:events.append(dict(ev,table=c['table'],hand_id=c['hand_id'],score=score,xy=xy))
    jsonl(dest/'frame_scores.jsonl',probes);jsonl(dest/'triggers.jsonl',events)
    results=[]
    for key,c in hands.items():
        ps=[p for p in probes if (p['table'],p['hand_id'])==key];best=max(ps,key=lambda p:p['score']);cur=[p for p in ps if p['current_hand_context_confirmed_offline']];bc=max(cur,key=lambda p:p['score']);ev=[e for e in events if (e['table'],e['hand_id'])==key]
        results.append({'table':key[0],'hand_id':key[1],'deal_chat_frame':c['deal_chat_frame'],'best_raw':best,'best_current':bc,'trigger_count':len(ev),'trigger':ev[0] if ev else None})
    jsonl(dest/'hand_results.jsonl',results)
    summary={'hands':len(hands),'source_frames':len(probes),'threshold':THRESHOLD,'raw_hit_hands':sum(any(p['ncc_hit'] and (p['table'],p['hand_id'])==key for p in probes) for key in hands),'hands_with_one_confirmed_trigger':len(events),'hands_without_confirmed_trigger':len(hands)-len(events),'duplicate_triggers':sum(max(0,r['trigger_count']-1) for r in results),'false_positives':'AWAITING_INDEPENDENT_PIXEL_REVIEW','live_current_hand_confirmation':'NOT_IMPLEMENTED; supplied reviewed offline hand keys','scope':'Fixed T01 cards-word template; fixed0.95; same1224source-frame baseline windows. No template bank or threshold adjustment.'}
    dump(dest/'summary.json',summary);db.close();return summary

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.output),ensure_ascii=False,indent=2))
