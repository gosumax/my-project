"""Old20 regression; no threshold/model selection from these results."""
import argparse,collections,json,re,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from start_stack_wide_20261003 import ROOT,TL,database,table,read_jsonl,jsonl,dump
from dealing_two_stage_ncc_20261003 import TwoStageModel,CausalRowFrontier,OncePerHandLatch,HandStateTrigger,bind_current,sha
from parser_core.chat_rows import physical_spans,padded_row_span
from parser_core.ram_first_stage import png_compatible_gray

def run(out,config,hand_state_mode=False):
    import cv2
    old=ROOT/'runs/start_stack_frame_ledger_wide_smoke14_20261003';out.mkdir(parents=True,exist_ok=True)
    hands=read_jsonl(old/'selected_hands.jsonl');baseline=read_jsonl(old/'ncc_dealing_cards_trigger/ncc_frame_scores.jsonl')
    headers=collections.defaultdict(list);timeline=read_jsonl(TL)
    for r in timeline:
        text=r.get('raw_ocr_text','').strip();m=re.fullmatch(r'(?:Hand\s*)?#(\d+)',text)
        if m:headers[(r['table_id'],r['frame_id'])].append((m[1],r['row_index'],r.get('row_pixel_sha256')))
    profile=json.loads((ROOT/'configs/layout_847x404.json').read_text(encoding='utf-8'));bottoms=json.loads((ROOT/'configs/chat_log_bottom_rois.json').read_text(encoding='utf-8'))['slots']
    model=TwoStageModel(config);db=database();all_rows=[];all_frames=[];triggers=[];results=[];start=time.perf_counter();header_checks=[]
    for c in hands:
        key=(c['table'],c['hand_id']);probes=[p for p in baseline if (p['table'],p['hand_id'])==key];frontier=CausalRowFrontier();latch=OncePerHandLatch();state=HandStateTrigger();active=None;header_frame=None;header_index=None
        for p in probes:
            image,meta=table(db,c['session'],p['frame']);slot=f"slot_{int(c['table'][1:]):02d}";box=profile['slots'][slot]['chat']['chat_log'];gray=png_compatible_gray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));chat=gray[box[1]:bottoms[slot],box[0]:box[2]]
            spans=physical_spans(chat);rs=[]
            for i,_ in enumerate(spans):
                a,b=padded_row_span(spans,i,len(chat),target_height=16)
                if b-a!=16:continue
                pixels=chat[a:b].copy();rs.append({'gray':pixels,'row_index':i,'row_bounds':[a,b],'pixel_hash':sha(pixels.tobytes())})
            rs=frontier.observe(p['frame'],rs)
            for number,index,digest in sorted(headers.get((c['table'],p['frame']),[]),key=lambda x:x[1]):
                row=next((r for r in rs if r['row_index']==index),None);passed=bool(row and row['pixel_hash']==digest)
                header_checks.append({'table':c['table'],'frame':p['frame'],'hand_id':number,'row_index':index,'pixel_verified':passed})
                if passed and number==c['hand_id'] and active is None:
                    active=number;header_frame=p['frame'];header_index=index;state.confirm_hand(c['table'],number,p['frame'],digest)
            candidates=[];frame_rows=[]
            for row in rs:
                result=model.classify(row['gray']);fresh,reason=bind_current(row,active,header_frame,header_index)
                record=dict(meta,table=c['table'],sample_hand_id=c['hand_id'],hand_id=active,row_bounds=row['row_bounds'],row_index=row['row_index'],track_id=row['track_id'],birth_frame=row['birth_frame'],append_proven=row['append_proven'],freshness=row['freshness'],current_row_confirmed=fresh,current_reason=reason,**result)
                number=next((number for number,index,digest in headers.get((c['table'],p['frame']),[]) if digest==row['pixel_hash']),None)
                if number:record['hand_header_id']=number
                all_rows.append(record);frame_rows.append(record)
                if result['status']=='cards' and fresh:candidates.append(row)
            if hand_state_mode:
                event,reason,eligible=state.observe(c['table'],p['frame'],frame_rows)
                for record in frame_rows:
                    record['current_row_confirmed']=record['row_index'] in eligible
                    record['current_reason']='CO_VISIBLE_CURRENT_HAND_HEADER_ORDER' if record['current_row_confirmed'] else reason
                    record['physical_row_freshness_required']=False
            else:event,reason=latch.observe(c['table'],active,p['frame'],candidates)
            if event:triggers.append(event)
            all_frames.append(dict(meta,table=c['table'],sample_hand_id=c['hand_id'],hand_id=active,rows=len(rs),gate_pass=sum(r['gate_pass'] for r in frame_rows),classifier_rows=sum(r['scores'] is not None for r in frame_rows),latch_reason=reason))
        expected=[r for r in all_rows if r['sample_hand_id']==c['hand_id'] and r['table']==c['table'] and c['deal_chat_frame']<=r['frame']<=c['deal_chat_frame']+2 and r['status']=='cards']
        # Classification recall uses independent known DEAL frames only for
        # evaluation, not as classifier/freshness input.
        frames_with_cards={r['frame'] for r in expected}
        results.append({'table':c['table'],'hand_id':c['hand_id'],'expected_deal_frame':c['deal_chat_frame'],'classification_hit':all(c['deal_chat_frame']+i in frames_with_cards for i in range(3)),'trigger_hit':any(t['table']==c['table'] and t['hand_id']==c['hand_id'] for t in triggers),'current_hand_pixel_verified':active is not None})
    jsonl(out/'rows.jsonl',all_rows);jsonl(out/'frames.jsonl',all_frames);jsonl(out/'triggers.jsonl',triggers);jsonl(out/'hand_results.jsonl',results);jsonl(out/'header_pixel_checks.jsonl',header_checks)
    gate_fail=[r for r in all_rows if not r['gate_pass']]
    summary={'scope':'Old20hands regression only, never used to retune frozen model. Independent source-bound headers supply currenthand context; no manually asserted freshness.','hands':len(hands),'frames':len(all_frames),'rows':len(all_rows),'first_stage_excluded':len(gate_fail),'four_class_rows':sum(r['scores'] is not None for r in all_rows),'gate_fail_classifier_calls':sum(r['scores'] is not None for r in gate_fail),'classification_HITS':sum(r['classification_hit'] for r in results),'classification_MISSES':sum(not r['classification_hit'] for r in results),'trigger_HITS':sum(r['trigger_hit'] for r in results),'trigger_MISSES':sum(not r['trigger_hit'] for r in results),'UNKNOWN':sum(r['status']=='UNKNOWN' for r in all_rows),'DUPLICATE_TRIGGERS':len(triggers)-len({(t['table'],t['hand_id']) for t in triggers}),'header_pixel_checks_pass':sum(r['pixel_verified'] for r in header_checks),'header_pixel_checks_fail':sum(not r['pixel_verified'] for r in header_checks),'frontier_claimed_fresh_rows':sum(r['append_proven'] for r in all_rows),'wall_seconds':time.perf_counter()-start,'trigger_verdict':'NOT_PROVEN; independent physical-row lifecycle producer unavailable.'}
    summary['hand_state_mode']=hand_state_mode
    if hand_state_mode:summary['trigger_verdict']='AWAITING_INDEPENDENT_TRIGGER_REVIEW; current hand header order, no row-birth proof required.'
    dump(out/'summary.json',summary);db.close();return summary
if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--hand-state',action='store_true');a=p.parse_args();print(json.dumps(run(a.output,a.config,a.hand_state),ensure_ascii=False,indent=2))
