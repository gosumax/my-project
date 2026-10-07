"""Same-row four-word NCC diagnostics. No production admission changes."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from start_stack_wide_20261003 import ROOT,PARENT,database,table,read_jsonl,dump,jsonl,digest
from parser_core.chat_rows import physical_spans,padded_row_span
CLASSES=('cards','Turn','Flop','River')
ABS_THRESHOLD=.95
MARGIN_LEVELS=(0.,.05,.10,.20,.30)

def rank(scores):
    order=sorted(scores,key=lambda k:(-scores[k],k))
    return order[0],order[1],scores[order[0]]-scores[order[1]]

def run(out):
    import cv2,numpy as np
    dest=out/'multi_template_classification';dest.mkdir(exist_ok=True)
    profile=json.loads((ROOT/'configs/layout_847x404.json').read_text(encoding='utf-8'))
    negatives=json.loads((out/'cards_word_trigger/negative_controls_and_local_template_diagnostic.json').read_text(encoding='utf-8'))['negative_controls']
    db=database();hands={(c['table'],c['hand_id']):c for c in read_jsonl(out/'selected_hands.jsonl')}
    sources={'cards':{'session':hands[('T01','2875498927')]['session'],'frame':681,'y':355}}
    for n in negatives:
        word=n['raw_ocr_text'].split()[-1];sources[word]={'session':hands[('T01','2875498927')]['session'],'frame':n['frame'],'y':n['quality']['crop_span_local_y'][0]}
    templates={};manifest=[]
    for word in CLASSES:
        s=sources[word];im,meta=table(db,s['session'],s['frame']);box=profile['slots']['slot_01']['chat']['chat_log'];gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);template=gray[box[1]+s['y']:box[1]+s['y']+16,box[0]+91:box[0]+131].copy();templates[word]=template
        path=dest/f'template_{word}.png';cv2.imwrite(str(path),template) if str(path).isascii() else path.write_bytes(cv2.imencode('.png',template)[1].tobytes())
        manifest.append(dict(meta,word=word,chat_local_rect=[91,s['y'],131,s['y']+16],shape=[40,16],template_gray_sha256=hashlib.sha256(template.tobytes()).hexdigest(),file=str(path)))
    dump(dest/'template_manifest.json',{'templates':manifest,'threshold_unchanged':ABS_THRESHOLD,'margin_levels_diagnostic_only':MARGIN_LEVELS,'calibration':'Four frozen T01 observed literals; no template tuning on six misses.','common_canvas':'40x16 untouched grayscale; no resize/contrast correction; all4compare same physical row; small horizontal shifts88..94.'})
    original=read_jsonl(out/'ncc_dealing_cards_trigger/ncc_frame_scores.jsonl')
    expected={(p['table'],p['hand_id']):p for p in read_jsonl(out/'t04_ncc_acceptance/recheck_expected_frames.jsonl')}
    rows=[];frames=[];start=time.perf_counter();ncc_seconds=0.
    for p in original:
        key=(p['table'],p['hand_id']);c=hands[key];im,meta=table(db,c['session'],p['frame']);box=profile['slots'][f"slot_{int(c['table'][1:]):02d}"]['chat']['chat_log'];gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);chat=gray[box[1]:,box[0]:box[2]]
        spans=physical_spans(chat);frame_rows=[]
        for i,span in enumerate(spans):
            y1,y2=padded_row_span(spans,i,chat.shape[0],target_height=16)
            if y2-y1!=16:continue
            search=chat[y1:y2,88:134];scores={};shifts={}
            tick=time.perf_counter()
            for word,t in templates.items():
                corr=cv2.matchTemplate(search,t,cv2.TM_CCOEFF_NORMED);_,score,_,loc=cv2.minMaxLoc(corr);scores[word]=float(score);shifts[word]=[88+loc[0],y1]
            ncc_seconds+=time.perf_counter()-tick
            winner,second,margin=rank(scores)
            raw=im[box[1]+y1:box[1]+y2,box[0]:box[2]]
            record=dict(meta,table=c['table'],hand_id=c['hand_id'],row_index=i,row_chat_span=[y1,y2],scores=scores,shifts=shifts,winner=winner,second=second,margin=margin,winner_score=scores[winner],absolute_threshold_pass=scores[winner]>=ABS_THRESHOLD,fullrow_bgr_sha256=hashlib.sha256(raw.tobytes()).hexdigest())
            rows.append(record);frame_rows.append(record)
        # A frame can contain cards and Turn simultaneously. Preserve row
        # classifications; never compare maxima from different physical rows.
        best_cards=max(frame_rows,key=lambda r:r['scores']['cards']) if frame_rows else None
        fresh=next((r for r in frame_rows if abs(r['row_chat_span'][0]-expected[key]['match_chat_local_xy'][1])<=1),None) if c['deal_chat_frame']<=p['frame']<=c['deal_chat_frame']+2 else None
        chosen=fresh or best_cards
        frames.append(dict(meta,table=c['table'],hand_id=c['hand_id'],scores=chosen['scores'] if chosen else None,winner=chosen['winner'] if chosen else None,second=chosen['second'] if chosen else None,margin=chosen['margin'] if chosen else None,selected_row=chosen['row_index'] if chosen else None,selected_row_span=chosen['row_chat_span'] if chosen else None,selection='INDEPENDENT_OFFLINE_CURRENT_DEAL_ROW' if fresh else 'MAX_CARDS_ROW_DIAGNOSTIC_ONLY',fresh_row_bound=bool(fresh),classes_present=sorted({r['winner'] for r in frame_rows}),row_candidates=len(frame_rows)))
    jsonl(dest/'row_scores.jsonl',rows);jsonl(dest/'frame_scores.jsonl',frames)
    results=[]
    old=read_jsonl(out/'cards_word_trigger/hand_results.jsonl');old_miss={(r['table'],r['hand_id']) for r in old if not r['trigger_count']}
    for key,c in hands.items():
        initial=[f for f in frames if (f['table'],f['hand_id'])==key and c['deal_chat_frame']<=f['frame']<=c['deal_chat_frame']+2]
        valid=bool(len(initial)==3 and all(f['fresh_row_bound'] for f in initial))
        result={'table':key[0],'hand_id':key[1],'previous_miss':key in old_miss,'initial_frames':initial,'relative_cards_3_frames':valid and all(f['winner']=='cards' for f in initial),'margin_min':min(f['margin'] for f in initial) if valid else None,'absolute_cards_3_frames':valid and all(f['scores']['cards']>=ABS_THRESHOLD for f in initial),'margin_acceptance_diagnostic':{str(m):valid and all(f['winner']=='cards' and f['margin']>=m for f in initial) for m in MARGIN_LEVELS}}
        results.append(result)
    jsonl(dest/'hand_results.jsonl',results);jsonl(dest/'six_miss_results.jsonl',[r for r in results if r['previous_miss']])
    summary={'hands':20,'source_frames':len(frames),'physical_rows':len(rows),'absolute_threshold_unchanged':ABS_THRESHOLD,'relative_current_cards_hands':sum(r['relative_cards_3_frames'] for r in results),'relative_current_misses':sum(not r['relative_cards_3_frames'] for r in results),'absolute_cards_hands_common_canvas':sum(r['absolute_cards_3_frames'] for r in results),'margin_sensitivity_hands':{str(m):sum(r['margin_acceptance_diagnostic'][str(m)] for r in results) for m in MARGIN_LEVELS},'false_positives':'AWAITING_INDEPENDENT_PHYSICAL_ROW_PIXEL_REVIEW','margin_threshold_promoted':False,'sample_ncc_seconds':ncc_seconds,'scan_wall_seconds':time.perf_counter()-start,'scope':'Relative nearest-word classifier only. An arbitrary non-dealing line always has a nearest class: open-set false positives must be checked. Maxima across different rows never define a classification margin. Fresh-row binding remains offline reviewed evidence, not live implementation.'}
    dump(dest/'summary.json',summary);db.close();return summary

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.output),ensure_ascii=False,indent=2))
