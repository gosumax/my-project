"""Offline Smoke14 frame-ledger sample, isolated from production.

No money-dependent sample selection, nickname aliases, duplicate removal, or
future actions in primary candidates. Manual review is explicit input evidence.
"""
from __future__ import annotations
import argparse,base64,collections,hashlib,html,json,sqlite3,sys,time
from pathlib import Path
from decimal import Decimal
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from parser_core.stack_temporal import hand_windows
from parser_core.poker_hand_assembly import assemble_messages
from parser_core.tiny_chat_client import TinyChatClient
sys.path.insert(0,str(Path(__file__).parent))
from start_stack_frame_ledger_20261003 import observed_ledger,read_jsonl,digest,dump,jsonl
SRC=ROOT/'runs/smoke14_start_stack_full_20260930_v1'
TL=ROOT/'runs/smoke14_rowid_stack_parity_20260930_v1/ordered_timeline.jsonl'
PARENT=Path('C:/ParserData/runs/smoke14_full_tiny_ssd_20260929_v3')
MONEY={'ANTE','SMALL_BLIND','BIG_BLIND','CALL','BET','RAISE','ALL_IN','UNCALLED_RETURN','WIN'}

def database():return sqlite3.connect((PARENT/'journal.sqlite3').as_uri()+'?mode=ro',uri=True)

def table(db,session,frame):
    import cv2,numpy as np
    row=db.execute("SELECT o.crop_path,o.crop_sha256,f.source_pts,f.time_base_num,f.time_base_den FROM observations o JOIN frames f ON f.source_id=o.source_id AND f.frame_id=o.frame_id WHERE o.roi_type='table' AND o.table_session_id=? AND o.frame_id=?",(session,frame)).fetchone()
    if not row:raise ValueError(f'MISSING_TABLE_FRAME {session} {frame}')
    path=PARENT/row[0]
    if digest(path)!=row[1]:raise ValueError('TABLE_FILE_HASH_MISMATCH')
    image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    return image,{'table_path':str(path),'table_file_sha256':row[1],'source_pts':row[2],'time_base':[row[3],row[4]],'frame':frame}

def prepare(out):
    obs=read_jsonl(SRC/'start_stack_observations.jsonl');rows=read_jsonl(TL);candidates=[]
    for w in hand_windows(rows):
        if not w['start_frame'] or w['end_frame'] is None:continue
        ms=[m for m in assemble_messages(w['rows'],require_adjacency=True) if m['frame_id']<w['end_frame']]
        deal=[m['frame_id'] for m in ms if m['action']=='DEAL' and m['frame_id']<w['start_frame']+150]
        win=[m for m in ms if m['action']=='WIN']
        nums=[o for o in obs if o['table_id']==w['table_id'] and o['roi_type']=='stack' and o['visual_kind']=='STACK' and w['start_frame']<=o['frame_index']<min(w['end_frame'],w['start_frame']+200)]
        counts=collections.Counter(o['frame_index'] for o in nums)
        gate=next((f for f,n in sorted(counts.items()) if n>=4 and deal and f>min(deal)),None)
        if gate is None or not win:continue
        session=next(o['table_context'] for o in nums if o['frame_index']==gate)
        candidates.append({'table':w['table_id'],'hand_id':w['hand_number'],'start':w['start_frame'],'end':w['end_frame'],'session':session,
            'deal_chat_frame':min(deal),'gate_candidate':gate,'win_frame':win[-1]['frame_id'],'payout_source_row_ids':win[-1]['source_row_ids'],
            'selection_policy':'first2 per table then first remaining chronological table/hand; payout+next header observed; gate candidate >=4 numeric saved seats after DEAL; never selected by agreement'})
    selected=[]
    for tableid in sorted({c['table'] for c in candidates}):selected.extend([c for c in candidates if c['table']==tableid][:2])
    selected.extend([c for c in candidates if c not in selected][:20-len(selected)])
    jsonl(out/'selected_hands.jsonl',selected);jsonl(out/'selection_pool.jsonl',candidates)
    from PIL import Image,ImageDraw
    db=database()
    for kind in ['gate','end']:
        for page in range(0,len(selected),4):
            sheet=Image.new('RGB',(1694,860),'white');draw=ImageDraw.Draw(sheet)
            for j,c in enumerate(selected[page:page+4]):
                f=c['gate_candidate'] if kind=='gate' else min(c['win_frame']+1,c['end']-1)
                _,meta=table(db,c['session'],f)
                im=Image.open(meta['table_path']);x=(j%2)*847;y=(j//2)*430
                draw.text((x,y),f"{page+j}: {c['table']} #{c['hand_id']} {kind} f{f}",fill='black');sheet.paste(im,(x,y+24))
                c[kind+'_evidence']=meta
            sheet.save(out/f'visual_{kind}_{page//4+1}.png')
    jsonl(out/'selected_hands.jsonl',selected);db.close()
    dump(out/'source_manifest.json',{'inputs':[{'path':str(p),'sha256':digest(p)} for p in [TL,SRC/'start_stack_observations.jsonl',ROOT/'configs/layout_847x404.json',ROOT/'production_manifest.json']],
        'journal':str(PARENT/'journal.sqlite3'),'journal_access':'READ_ONLY','production_modified_by_experiment':False})
    return {'selected':len(selected),'candidate_hands':len(candidates),'tables':len({c['table'] for c in selected})}

def collect(out):
    import cv2
    selected=read_jsonl(out/'selected_hands.jsonl');reviews=read_jsonl(out/'visual_reviews.jsonl')
    reviewmap={(r['table'],r['hand_id']):r for r in reviews}
    obs=read_jsonl(SRC/'start_stack_observations.jsonl');profile=json.loads((ROOT/'configs/layout_847x404.json').read_text(encoding='utf-8'))
    db=database();probes=[];payloads={}
    for c in selected:
        rv=reviewmap[(c['table'],c['hand_id'])]
        if rv['deal_complete_status']!='CONFIRMED':continue
        geometry=json.loads(db.execute('SELECT geometry_json FROM table_sessions WHERE table_session_id=?',(c['session'],)).fetchone()[0])
        if int(geometry['slot_id'])!=int(c['table'][1:]):raise ValueError('SESSION_SLOT_ROUTING_MISMATCH')
        if geometry.get('profile_sha256')!=digest(ROOT/'configs/layout_847x404.json'):raise ValueError('SOURCE_ROI_PROFILE_MISMATCH')
        gate=rv['deal_complete_frame'];frames=set(range(gate+1,min(gate+16,c['end'])))
        # ALL saved later numeric states, no selection by money equality.
        later=sorted({o['frame_index'] for o in obs if o['table_id']==c['table'] and o['roi_type']=='stack' and o['visual_kind']=='STACK' and gate<o['frame_index']<c['end']})
        for f in later:
            frames.update(range(f,min(f+3,c['end'])))
        for f in sorted(frames):
            image,meta=table(db,c['session'],f)
            for seat in rv['occupied_seats']:
                for kind in ['stack','nick']:
                    box=profile['slots'][f"slot_{int(c['table'][1:]):02d}"]['seats'][seat][kind]
                    crop=image[box[1]:box[3],box[0]:box[2]];key=hashlib.sha256(crop.tobytes()).hexdigest()
                    if key not in payloads:
                        packed=cv2.imencode('.png',crop)[1].tobytes()
                        payloads[key]={'item_id':key,'source_sha256':hashlib.sha256(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).tobytes()).hexdigest(),'png_base64':base64.b64encode(packed).decode()}
                    probes.append(dict(meta,table=c['table'],hand_id=c['hand_id'],session=c['session'],seat=seat,kind=kind,roi=box,crop_bgr_sha256=key))
    config=json.loads((ROOT/'configs/ocr_tiny_local.json').read_text(encoding='utf-8'));items=list(payloads.values());pred={}
    cache=out/'raw_ocr_cache.jsonl'
    if cache.exists():pred={p['item_id']:p for p in read_jsonl(cache)}
    items=[i for i in items if i['item_id'] not in pred]
    client=TinyChatClient(Path(config['python']),Path(config['model_dir']),config['model_name'],config['weights_sha256'],config['device'],out/'wide_tiny.stderr.log')
    try:
        with cache.open('a',encoding='utf-8') as stream:
            for start in range(0,len(items),8):
                for r in client.read_batch(items[start:start+8],8)['results']:
                    pred[r['item_id']]=r;stream.write(json.dumps(r,ensure_ascii=False)+'\n')
                stream.flush()
                if start%400==0:print('OCR',start,'/',len(items),flush=True)
    finally:client.close()
    for p in probes:p['ocr']=pred[p['crop_bgr_sha256']]
    jsonl(out/'numeric_nick_probes.jsonl',probes)
    db.close();return {'source_roi_probes':len(probes),'unique_exact_bgr_payloads':len(payloads),'new_ocr_payloads':len(items)}

def numeric(probe):
    if not probe:return None
    raw=probe['ocr']['raw_text'].replace(' ','').replace('\u00a0','')
    return raw if raw.isdigit() and probe['ocr']['score']>=.90 else None

def classify(observations):
    """No averaging, consensus choice or unknown-to-known promotion."""
    good=[o for o in observations if not o['blockers']]
    values={o['candidate_start_stack'] for o in good};states={(o['gross_committed_before_frame'],o['returns_before_frame'],o['payouts_before_frame']) for o in good}
    if len(values)>1:return 'CONFLICT'
    if len(good)>=2 and len(states)>=2:return 'MULTI_STATE_EXACT'
    if len(good)>=2:return 'SAME_STATE_REPEAT'
    blockers={b for o in observations for b in o['blockers']}
    if 'IDENTITY_UNKNOWN' in blockers:return 'IDENTITY_UNKNOWN'
    if 'OCR_STACK_CONFLICT' in blockers:return 'OCR_STACK_CONFLICT'
    if 'TIMING_UNKNOWN' in blockers:return 'TIMING_UNKNOWN'
    return 'LEDGER_UNKNOWN'

def uncertain_row_identity(message):
    return any('ROW_IDENTITY_UNRESOLVED' in r.get('message_reason','') for r in message.get('source_rows',[]))

def first_search_complete(gate,frame):return frame<=gate+13

def analyze(out):
    selected=read_jsonl(out/'selected_hands.jsonl');reviews=read_jsonl(out/'visual_reviews.jsonl');probes=read_jsonl(out/'numeric_nick_probes.jsonl')
    reviewmap={(r['table'],r['hand_id']):r for r in reviews};rows=read_jsonl(TL)
    windows={(w['table_id'],w['hand_number']):w for w in hand_windows(rows)}
    lookup={(p['table'],p['hand_id'],p['seat'],p['kind'],p['frame']):p for p in probes}
    observations=[];groups=[];snapshots=[];eventlog=[];hands=[]
    for c in selected:
        rv=reviewmap[(c['table'],c['hand_id'])];gate=rv['deal_complete_frame']
        if rv['deal_complete_status']!='CONFIRMED':continue
        key=(c['table'],c['hand_id']);w=windows[key]
        messages=[m for m in assemble_messages(w['rows'],require_adjacency=True) if m['frame_id']<c['end']]
        for m in messages:eventlog.append(dict(m,table=c['table'],hand_id=c['hand_id']))
        # Visual nicknames are hand-scoped labels; no cross-hand canonical ID.
        roster=rv.get('visually_read_nicknames',{});known={n for n in roster.values() if n}
        for seat in rv['occupied_seats']:
            seatframes=sorted({p['frame'] for p in probes if (p['table'],p['hand_id'],p['seat'],p['kind'])==(*key,seat,'stack')})
            stable=[]
            for f in seatframes:
                triple=[lookup.get((*key,seat,'stack',f+j)) for j in range(3)]
                numbers=[numeric(p) for p in triple]
                if None in numbers or len(set(numbers))!=1:continue
                # Fresh OCR of all three own TABLE source frames. Later
                # two frames confirm stability, never add money to cutoff f.
                first=triple[0];nick=lookup.get((*key,seat,'nick',f));player=roster.get(seat)
                identity=bool(player and nick and nick['ocr']['raw_text']==player and nick['ocr']['score']>=.90)
                scoped_id=f"{c['session']}:{c['hand_id']}:{seat}" if identity else None
                ledger=observed_ledger(messages,player,f) if identity else None
                blockers=[];details=[]
                if not identity:blockers.append('IDENTITY_UNKNOWN')
                if ledger:
                    for issue in ledger['issues']:
                        if issue!='PAYOUT_APPLIED_FORMULA_OUT_OF_SCOPE':blockers.append('LEDGER_UNKNOWN');details.append(issue)
                    for m in messages:
                        if m['frame_id']>f:continue
                        if m['frame_id']==f and m['player']==player and m['action'] in MONEY:
                            blockers.append('TIMING_UNKNOWN');details.append('SAME_FRAME_MONEY_ORDER_UNKNOWN')
                        if m['frame_id']==f and m['action'] in {'FLOP','TURN','RIVER'}:
                            blockers.append('TIMING_UNKNOWN');details.append('SAME_FRAME_STREET_ORDER_UNKNOWN')
                        if m['frame_id']<f and m['action'] in MONEY and m['player'] not in known:
                            blockers.append('LEDGER_UNKNOWN');details.append('UNBOUND_MONETARY_ACTOR:'+str(m['player']))
                        if m['frame_id']<f and m['action']=='UNKNOWN_EVENT':
                            blockers.append('LEDGER_UNKNOWN');details.append('UNPARSED_CHAT_ROW:'+str(m['source_row_ids']))
                        if m['frame_id']<f and m['action'] in MONEY and uncertain_row_identity(m):
                            blockers.append('LEDGER_UNKNOWN');details.append('MONEY_SOURCE_ROW_IDENTITY_UNRESOLVED')
                        if m['frame_id']<f and m['action'] in {'FLOP','TURN','RIVER'} and uncertain_row_identity(m):
                            blockers.append('LEDGER_UNKNOWN');details.append('STREET_SOURCE_ROW_IDENTITY_UNRESOLVED')
                    if any(m['frame_id']<f and m['action'] in {'WIN','UNCALLED_RETURN'} for m in messages):
                        blockers.append('LEDGER_UNKNOWN');details.append('SETTLEMENT_EXCLUDED_FROM_PRIMARY_ACCEPTANCE')
                candidate=str(Decimal(numbers[0])+Decimal(ledger['gross'])-Decimal(ledger['returns'])-Decimal(ledger['payouts'])) if ledger else None
                record={'table':c['table'],'hand_id':c['hand_id'],'session':c['session'],'seat':seat,'player':player,'player_id':scoped_id,
                    'frame':f,'snapshot_frame':f,'stable_confirmed_frame':f+2,'deal_complete_frame':gate,
                    'displayed_stack':numbers[0],'gross_committed_before_frame':ledger['gross'] if ledger else None,
                    'returns_before_frame':ledger['returns'] if ledger else None,'payouts_before_frame':ledger['payouts'] if ledger else None,
                    'candidate_start_stack':candidate,'blockers':sorted(set(blockers)),'blocker_details':sorted(set(details)),
                    'stack_raw_readings':[p['ocr'] for p in triple],'source_frames':[p['frame'] for p in triple],
                    'stack_hashes':[p['crop_bgr_sha256'] for p in triple],'table_source_paths':[p['table_path'] for p in triple],
                    'nick_raw':nick['ocr'] if nick else None,'ledger':ledger,'future_ledger_events_used':0,
                    'acceptance_scope':'PRE_SETTLEMENT' if ledger and Decimal(ledger['payouts'])==0 else 'POST_SETTLEMENT_DIAGNOSTIC_ONLY'}
                stable.append(record)
            if stable:
                first=stable[0];first['is_first_stable_snapshot']=True
                first['first_snapshot_search_complete']=first_search_complete(gate,first['frame'])
                if not first['first_snapshot_search_complete']:
                    first['blockers']=sorted(set(first['blockers']+['LEDGER_UNKNOWN']))
                    first['blocker_details'].append('FIRST_STABLE_SNAPSHOT_NOT_PROVEN_ACROSS_UNSAMPLED_GAP')
                    for record in stable[1:]:
                        record['blockers']=sorted(set(record['blockers']+['LEDGER_UNKNOWN']))
                        record['blocker_details'].append('BASELINE_FIRST_STABLE_SNAPSHOT_UNRESOLVED')
                snapshots.append(first)
                # Preserve all stable observations; adjacent repeats do not
                # increase independent commitment-state count.
                observations.extend(stable)
            else:
                snapshots.append({'table':c['table'],'hand_id':c['hand_id'],'seat':seat,'frame':None,'blockers':['LEDGER_UNKNOWN'],'blocker_details':['STACK_NUMERIC_UNAVAILABLE; NO_3_CONSECUTIVE_NUMERIC_READINGS_IN_SAVED_SOURCE_SAMPLE']})
            status=classify(stable) if stable else 'LEDGER_UNKNOWN'
            good=[o for o in stable if not o['blockers']]
            values=[Decimal(o['candidate_start_stack']) for o in good]
            diag=[Decimal(o['candidate_start_stack']) for o in stable if o['candidate_start_stack'] is not None]
            diagnostic_primary=[o for o in stable if o['player_id'] and o['acceptance_scope']=='PRE_SETTLEMENT' and 'SETTLEMENT_EXCLUDED_FROM_PRIMARY_ACCEPTANCE' not in o['blocker_details']]
            dvalues={o['candidate_start_stack'] for o in diagnostic_primary}
            dstates={(o['gross_committed_before_frame'],o['returns_before_frame']) for o in diagnostic_primary}
            groups.append({'table':c['table'],'hand_id':c['hand_id'],'seat':seat,'player':roster.get(seat),
                'status':status,'stable_observations':len(stable),'usable_observations':len(good),
                'commitment_states':len({(o['gross_committed_before_frame'],o['returns_before_frame'],o['payouts_before_frame']) for o in good}),
                'max_delta':str(max(values)-min(values)) if values else None,
                'diagnostic_max_delta':str(max(diag)-min(diag)) if diag else None,
                'excluded_blockers':dict(collections.Counter(b for o in stable for b in o['blockers'])),
                'fully_usable_group':bool(stable and len(good)==len(stable)),
                'pre_settlement_diagnostic_states':len(dstates),
                'pre_settlement_diagnostic_exact':len(dstates)>=2 and len(dvalues)==1,
                'pre_settlement_diagnostic_conflict':len(dstates)>=2 and len(dvalues)>1,
                'pre_settlement_diagnostic_max_delta':str(max(map(Decimal,dvalues))-min(map(Decimal,dvalues))) if dvalues else None,
                'different_commitment_diagnostic_states':len({(o['gross_committed_before_frame'],o['returns_before_frame'],o['payouts_before_frame']) for o in stable if o['player_id']})})
        hands.append(dict(c,visual_review=rv,stable_snapshots=sum(s['table']==c['table'] and s['hand_id']==c['hand_id'] and s['frame'] is not None for s in snapshots)))
    # Unreviewed hand-to-hand occupancy remains UNKNOWN even if spelling/math
    # agrees. Never use a next-hand result to choose the prior candidate.
    cross=[]
    for a in selected:
        b=next((b for b in selected if b['table']==a['table'] and b['start']==a['end']),None)
        if b:
            for seat in reviewmap[(a['table'],a['hand_id'])]['occupied_seats']:
                ra=reviewmap[(a['table'],a['hand_id'])].get('visually_read_nicknames',{}).get(seat)
                rb=reviewmap[(b['table'],b['hand_id'])].get('visually_read_nicknames',{}).get(seat)
                sa=next((s for s in snapshots if s['table']==a['table'] and s['hand_id']==a['hand_id'] and s['seat']==seat),None)
                sb=next((s for s in snapshots if s['table']==b['table'] and s['hand_id']==b['hand_id'] and s['seat']==seat),None)
                check={'table':a['table'],'seat':seat,'previous_hand':a['hand_id'],'next_hand':b['hand_id'],'previous_player':ra,'next_player':rb,'status':'CONTINUITY_UNKNOWN','reason':'CONTINUOUS_OCCUPANCY_AND_COMPLETE_SETTLEMENT_NOT_REVIEWED','used_to_choose_candidate':False,'comparison_computed':False,'diagnostic_only':True}
                if ra and ra==rb and sa and sb and sa.get('candidate_start_stack') is not None and sb.get('candidate_start_stack') is not None:
                    prior=[e for e in eventlog if e['table']==a['table'] and e['hand_id']==a['hand_id']]
                    ledger=observed_ledger(prior,ra,a['end'])
                    expected=Decimal(sa['candidate_start_stack'])-Decimal(ledger['net'])+Decimal(ledger['payouts'])
                    next_start=Decimal(sb['candidate_start_stack'])
                    check.update(comparison_computed=True,previous_start_diagnostic=sa['candidate_start_stack'],
                        previous_gross=ledger['gross'],previous_returns=ledger['returns'],previous_payouts=ledger['payouts'],
                        expected_end_diagnostic=str(expected),next_start_diagnostic=str(next_start),diagnostic_delta=str(next_start-expected),
                        diagnostic_agreement=expected==next_start,ledger_status='LEDGER_UNKNOWN',ledger_issues=ledger['issues'],
                        source_snapshot_frames=[sa['frame'],sb['frame']])
                cross.append(check)
    occluded=[dict(p,visibility_status='STACK_OCCLUDED_BY_ACTION') for p in probes if p['kind']=='stack' and p['ocr']['raw_text'].upper().strip() in {'CALL','BET','RAISE','RASE','BEI','FOLD','CHECK','WINNER','ALL-IN','ALL IN'}]
    jsonl(out/'occluded_stack_observations.jsonl',occluded)
    jsonl(out/'stack_observations.jsonl',observations);jsonl(out/'first_stable_snapshots.jsonl',snapshots)
    jsonl(out/'player_hand_groups.jsonl',groups);jsonl(out/'events.jsonl',eventlog);jsonl(out/'hands_reviewed.jsonl',hands);jsonl(out/'cross_hand_checks.jsonl',cross)
    counts=collections.Counter(g['status'] for g in groups)
    for k in ['MULTI_STATE_EXACT','SAME_STATE_REPEAT','CONFLICT','IDENTITY_UNKNOWN','TIMING_UNKNOWN','LEDGER_UNKNOWN','OCR_STACK_CONFLICT']:counts.setdefault(k,0)
    summary={'verdict':'NOT_PROVEN','hands_reviewed':len(hands),'tables':len({c['table'] for c in selected}),
        'snapshots':sum(s['frame'] is not None for s in snapshots),'occupied_seat_opportunities':len(snapshots),
        'identity_bound_snapshots':sum(bool(s.get('player_id')) for s in snapshots),
        'stable_numeric_observations':len(observations),'usable_observations':sum(not o['blockers'] for o in observations),
        'multi_state_groups':sum(g['commitment_states']>=2 for g in groups),'group_counts':dict(counts),
        'pre_settlement_diagnostic_multi_state_groups':sum(g['pre_settlement_diagnostic_states']>=2 for g in groups),
        'pre_settlement_diagnostic_exact_groups':sum(g['pre_settlement_diagnostic_exact'] for g in groups),
        'pre_settlement_diagnostic_conflict_groups':sum(g['pre_settlement_diagnostic_conflict'] for g in groups),
        'occluded_action_stack_observations':len(occluded),
        'max_delta':str(max(Decimal(g['max_delta']) for g in groups if g['max_delta'] is not None)) if any(g['max_delta'] is not None for g in groups) else None,
        'diagnostic_max_delta':str(max((Decimal(g['diagnostic_max_delta']) for g in groups if g['diagnostic_max_delta'] is not None),default=Decimal(0))),
        'pre_settlement_diagnostic_max_delta':str(max((Decimal(g['pre_settlement_diagnostic_max_delta']) for g in groups if g['pre_settlement_diagnostic_max_delta'] is not None),default=Decimal(0))),
        'cross_hand_opportunities':len(cross),'cross_hand_checks':sum(c['comparison_computed'] for c in cross),
        'cross_hand_diagnostic_agreements':sum(c.get('diagnostic_agreement',False) for c in cross),
        'cross_hand_status_counts':dict(collections.Counter(c['status'] for c in cross)),
        'money_messages':sum(e['action'] in MONEY for e in eventlog),
        'money_messages_with_unresolved_source_identity':sum(e['action'] in MONEY and uncertain_row_identity(e) for e in eventlog),
        'observation_blocker_counts':dict(collections.Counter(b for o in observations for b in o['blockers'])),
        'observation_detail_counts':dict(collections.Counter(d for o in observations for d in o['blocker_details'] if ':' not in d)),
        'future_ledger_events_used':0,'acceptance_min_multi_state_groups':30,
        'hidden_debit_onset_required':False,'overlay_rule':'STACK_OCCLUDED_BY_ACTION; no numeric triple, excluded; +46..48 not debit latency',
        'selection_policy':selected[0]['selection_policy'],
        'no_alias_merge':True,'no_money_based_candidate_selection':True,
        'scope_limitations':['Later saved numeric source states are sparse; only the first15 frames after gate were exhaustively searched. Later triple probes cover every saved numeric state, not every video frame.',
            'Hand-local player_id means session+hand+seat plus visually matched exact nickname, not persistent PokerDom account identity.',
            'Raw money-row unresolved identity, unknown CHAT rows, repeated posts and unbound actors remain blockers. No row removal.',
            'Visual gate and win+next boundary review proves completed visible hand sample, not complete trusted ledger.']}
    dump(out/'summary.json',summary)
    report(out,summary,groups,observations,snapshots);return summary

def report(out,s,groups,obs,snapshots):
    cols=['table','hand_id','seat','player','frame','displayed_stack','gross_committed_before_frame','returns_before_frame','payouts_before_frame','candidate_start_stack','blockers']
    def tab(items,cols):return '<table><tr>'+''.join('<th>'+c+'</th>' for c in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(c,'')))+'</td>' for c in cols)+'</tr>' for r in items)+'</table>'
    page='<meta charset="utf-8"><title>Wide Smoke14 START_STACK</title><style>body{font:14px system-ui;margin:20px}td,th{padding:5px;border:1px solid #ddd}table{border-collapse:collapse}th{background:#eee}pre{white-space:pre-wrap}</style><h1>Wide Smoke14 START_STACK — '+s['verdict']+'</h1><pre>'+html.escape(json.dumps(s,ensure_ascii=False,indent=2))+'</pre><h2>First stable snapshots</h2>'+tab(snapshots,cols)+'<h2>Groups</h2>'+tab(groups,['table','hand_id','seat','player','status','stable_observations','usable_observations','commitment_states','max_delta','diagnostic_max_delta','excluded_blockers'])+'<h2>All stable source observations</h2>'+tab(obs,cols)
    (out/'report.html').write_text(page,encoding='utf-8')

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['prepare','collect','analyze'],required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True);start=time.perf_counter()
    result={'prepare':prepare,'collect':collect,'analyze':analyze}[a.phase](a.output)
    result['wall_seconds']=time.perf_counter()-start;print(json.dumps(result,ensure_ascii=False,indent=2))
