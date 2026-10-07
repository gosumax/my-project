"""Read-only saved receipt join and bounded registered tracker source replay."""
import sys,json,time,hashlib
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parser_core.ram_first_stage import FirstStage
from parser_core.chat_tracker import CONFIRMED_LINK_STATES

OUT=ROOT/'runs/hand_context_completeness_audit_20261003'
RUN=ROOT/'runs/header_admission_fix_20261003/acceptance_ON'
def readjson(p):return json.loads(p.read_text('utf-8'))
def jsonl(p):
    with p.open(encoding='utf-8') as f:
        for line in f:yield json.loads(line)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    manifest=readjson(ROOT/'production_manifest.json')
    before={p:digest(ROOT/p) for p in manifest['hashes']}
    groups=readjson(ROOT/'runs/start_stack_production_connection_20261003/historical_controls/known_new_hand_groups.json')['groups']
    headers=defaultdict(list)
    for r in jsonl(RUN/'hand_header_ocr.jsonl'):
        if r['status']=='HAND_HEADER_CONFIRMED':headers[(r['table_id'],r['current_hand_id'])].append(r)
    cards=defaultdict(list)
    for r in jsonl(RUN/'dealing_ncc_receipts.jsonl'):
        if r['status']=='cards':cards[r['table_id']].append(r)
    card_counts=Counter((table,r['source_frame']) for table,rr in cards.items() for r in rr)
    episodes={}
    for r in jsonl(RUN/'start_stack_capture.jsonl'):
        episodes.setdefault((r['table_id'],r['provisional_capture_episode']),r['source_frame'])
    episode_starts=defaultdict(list)
    for (table,episode),frame in episodes.items():episode_starts[table].append({'frame':frame,'episode':episode})
    groups.sort(key=lambda g:(g['table_id'],g['header_frame']))
    rows=[]
    for i,g in enumerate(groups):
        table=g['table_id'];hand=g['hand_id'];hs=sorted(headers[table,hand],key=lambda r:(r['source_frame'],r['source_span_index']))
        h=hs[0];end=next((x['header_frame'] for x in groups[i+1:] if x['table_id']==table),5401)
        eligible=[r for r in cards[table] if h['source_frame']<=r['source_frame']<end and (r['source_frame']!=h['source_frame'] or r['row_index']>h['source_span_index'])]
        c=min(eligible,key=lambda r:(r['source_frame'],r['row_index'])) if eligible else None
        rows.append({'table':table,'hand_id':hand,'header_source_frame':h['source_frame'],'header_source_pts':h['source_pts'],
          'header_source_span_index':h['source_span_index'],'header_pixel_sha256':h['pixel_sha256'],
          'header_ocr_completion_frame':'NOT_RECORDED','header_ocr_completion_time':'NOT_RECORDED',
          'header_confirmed_available_at_first_cards':'UNKNOWN_NOT_RECORDED',
          'first_valid_dealing_cards_frame':c['source_frame'] if c else None,'first_cards_receipt':c,
          'source_hand_interval_end_exclusive':end,'header_confirmations_for_hand':len(hs)})
        row=rows[-1];cf=c['source_frame'] if c else None
        es=[e for e in episode_starts[table] if cf is not None and cf<=e['frame']<end]
        changes=[e for e in es if e['frame']>cf and card_counts[table,e['frame']]>0 and card_counts[table,e['frame']-1]>=card_counts[table,e['frame']]]
        row['indirect_capture_episode_timing']={'episode_starts':es,'current_hand_change_only_episode_starts':changes,
          'interpretation':'New episode with existing arm, positive previous cards count and nonincreasing cards count requires changed current_hand_id. Attribution is to a hand-state change, not directly a stamped OCR callback.',
          'first_cards_count':card_counts[table,cf] if cf is not None else None}
        if changes and changes[0]['frame']==cf+1:
            row['header_confirmed_available_at_first_cards']='INFERRED_NO_AT_OBSERVE_FIRST_CARDS; HAND_CHANGE_BY_NEXT_FRAME'
    targets={(r['table'],r['first_valid_dealing_cards_frame']) for r in rows if r['first_cards_receipt']}
    targets.add((9,4360));proofs={};contexts={};observed={};frame_seen={}
    def frame(table,session,fid,pts,base,table_rgb,gray):
        if (table,fid) in targets:
            observed[table,fid]={'inherited':proofs.get(table),'source_chat_sha256':hashlib.sha256(gray.tobytes()).hexdigest()}
            frame_seen[table,fid]=fid
    def transition(r):
        table=r['table'];fid=r['frame_to']
        complete=r['classification'] not in {'INITIAL','INITIAL_VISIBLE','UNRESOLVED'} and not r['classification'].startswith('LOCAL')
        complete=complete and all(s in CONFIRMED_LINK_STATES for s in r['new_link_states'])
        proof={'complete':complete,'transition':r}
        proofs[table]=proof
        if (table,fid) in targets:contexts[table,fid]=proof
    meta=readjson(RUN/'run_metadata.json')
    replay_start=time.perf_counter()
    stage=FirstStage(ROOT/'video/smoke_17.mp4',meta['table_sessions'],5401,root=ROOT,decode_threads=3,
      frame_observer=frame,tracker_observer=transition)
    admitted=sum(1 for _ in stage.rows())
    replay_s=time.perf_counter()-replay_start
    def context(table,fid):
        proof=contexts.get((table,fid)) or observed.get((table,fid),{}).get('inherited')
        if not proof:return {'complete':False,'exact_blocker_reasons':['NO_TRACKER_CONTEXT_PROOF']}
        tr=proof['transition'];cls=tr['classification'];reasons=[]
        if cls in {'INITIAL','INITIAL_VISIBLE','UNRESOLVED'} or cls.startswith('LOCAL'):reasons.append('CLASSIFICATION_'+cls+':'+tr['reason'])
        bad=Counter(s for s in tr['new_link_states'] if s not in CONFIRMED_LINK_STATES)
        if bad:reasons.append('UNCONFIRMED_NEW_LINK_STATES:'+json.dumps(dict(bad),sort_keys=True))
        return {'complete':proof['complete'],'transition_frame':tr['frame_to'],'same_frame_transition':tr['frame_to']==fid,
          'classification':cls,'tracker_reason':tr['reason'],'new_link_states':tr['new_link_states'],
          'new_link_state_counts':dict(Counter(tr['new_link_states'])),'unconfirmed_link_state_counts':dict(bad),
          'exact_blocker_reasons':reasons,'source_chat_sha256':tr['source_chat_sha256'],
          'observed_source_chat_sha256':observed.get((table,fid),{}).get('source_chat_sha256')}
    distribution=Counter();overlap=Counter()
    for r in rows:
        ctx=context(r['table'],r['first_valid_dealing_cards_frame']);r['reconstructed_tracker_context']=ctx
        r['HAND_CONTEXT_COMPLETENESS']='PASS' if ctx['complete'] else 'UNKNOWN'
        r['exact_blocker_reason']='; '.join(ctx['exact_blocker_reasons'])
        distribution[r['exact_blocker_reason'] or 'PASS']+=1
        for reason in ctx['exact_blocker_reasons']:overlap[reason.split(':')[0]]+=1
    t09=next(r for r in rows if r['table']==9 and r['hand_id']=='2876720526')
    t09={'hand':t09,'context_at_4360':context(9,4360),'HAND_HEADER_CONFIRMED_available_before_or_at_4360':'UNKNOWN_NOT_RECORDED',
      'explanation':'Saved receipts bind source frames, not wall/capture completion frames; row order across distinct logs cannot prove availability.'}
    after={p:digest(ROOT/p) for p in before};changed=[p for p in before if before[p]!=after[p]]
    result={'scope':'SAVED_ACCEPTANCE_RECEIPTS + NEW_READ_ONLY_TRACKER_REPLAY; no OCR rerun',
      'hands':rows,'reason_distribution':dict(distribution),'overlapping_blocker_conditions':dict(overlap),'T09':t09,
      'counts':{'hands':len(rows),'completeness_PASS':sum(r['HAND_CONTEXT_COMPLETENESS']=='PASS' for r in rows),
        'completeness_UNKNOWN':sum(r['HAND_CONTEXT_COMPLETENESS']=='UNKNOWN' for r in rows),'completion_timing_NOT_RECORDED':len(rows),
        'replayed_frames':stage.seen_frames,'normal_admitted_rows':admitted},'replay_seconds':replay_s,
      'elapsed_seconds':time.perf_counter()-started,'registered_hashes_unchanged':not changed,'changed_registered_files':changed}
    (OUT/'all_38_hands.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    md=['# HAND_CONTEXT_COMPLETENESS read-only audit','',
      'Saved OCR completion timestamps/capture frames: NOT_RECORDED. Tracker context is a new deterministic source replay, not saved live timing.','',
      '| table | hand_id | header source | OCR completion frame/time | first valid cards | completeness | exact blocker |',
      '|---|---|---:|---|---:|---|---|']
    for r in rows:md.append(f"| T{r['table']:02} | {r['hand_id']} | {r['header_source_frame']} | NOT_RECORDED / NOT_RECORDED | {r['first_valid_dealing_cards_frame']} | {r['HAND_CONTEXT_COMPLETENESS']} | {r['exact_blocker_reason']} |")
    md+=['','Reason distribution: '+json.dumps(dict(distribution),ensure_ascii=False),'',
      'T09 #2876720526 header source frame: '+str(t09['hand']['header_source_frame'])+'. Availability by frame 4360: UNKNOWN_NOT_RECORDED.','',
      'Command: `python experiments/audit_hand_context_completeness_20261003.py`','',
      'Registered hashes unchanged: '+str(not changed)+'. Replay seconds: '+str(replay_s)]
    (OUT/'ALL_38_HANDS.md').write_text('\n'.join(md),'utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'hands','T09'}},ensure_ascii=False))
def augment_saved_episodes():
    started=time.perf_counter();saved=readjson(OUT/'all_38_hands.json')
    cc=Counter((r['table_id'],r['source_frame']) for r in jsonl(RUN/'dealing_ncc_receipts.jsonl') if r['status']=='cards')
    episodes={}
    for r in jsonl(RUN/'start_stack_capture.jsonl'):episodes.setdefault((r['table_id'],r['provisional_capture_episode']),r['source_frame'])
    results=[]
    for row in saved['hands']:
        table=row['table'];cf=row['first_valid_dealing_cards_frame'];end=row['source_hand_interval_end_exclusive']
        es=[{'frame':frame,'episode':ep,'cards_count':cc[table,frame],'previous_cards_count':cc[table,frame-1]} for (t,ep),frame in episodes.items() if t==table and cf<=frame<end]
        changes=[e for e in es if e['frame']>cf and e['cards_count']>0 and e['previous_cards_count']>=e['cards_count']]
        r={'table':table,'hand_id':row['hand_id'],'first_valid_cards':cf,'episode_starts':es,'hand_change_only_episode_starts':changes,
          'first_cards_availability':'UNKNOWN_NOT_RECORDED'}
        if changes and changes[0]['frame']==cf+1:r['first_cards_availability']='INFERRED_NOT_AVAILABLE_AT_FIRST_OBSERVE; HAND_STATE_CHANGE_BY_NEXT_FRAME'
        elif changes:r['first_cards_availability']='HAND_STATE_CHANGE_PROVEN_LATER; HAND_ATTRIBUTION_NOT_DIRECTLY_RECORDED'
        results.append(r);row['indirect_capture_episode_timing']=r
    t09=next(r for r in results if r['table']==9 and r['hand_id']=='2876720526')
    result={'scope':'POSTHOC_CAPTURE_EPISODE_INFERENCE_ONLY','hands':results,'T09':t09,
      'counts':dict(Counter(r['first_cards_availability'] for r in results)),
      'limitations':'An extra episode with existing arm and positive nonincreasing cards count proves current_hand_id changed between observations. An absent extra episode does not prove early completion: a hand change while cards are absent can be masked by the next visual episode.',
      'elapsed_seconds':time.perf_counter()-started}
    (OUT/'capture_episode_timing_inference.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    (OUT/'all_38_hands.json').write_text(json.dumps(saved,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({'counts':result['counts'],'T09':t09,'elapsed_seconds':result['elapsed_seconds']},ensure_ascii=False))
if __name__=='__main__':
    if '--augment-saved-episodes' in sys.argv:augment_saved_episodes()
    else:main()
