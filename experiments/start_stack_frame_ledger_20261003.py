"""Read-only real Smoke14 evidence audit; never a production stack resolver.

Hand windows are observed-header windows, not independently certified epochs.
No OCR nickname aliases, row removals, inferred actions or future ledger repairs.
"""
from __future__ import annotations
import argparse, base64, collections, hashlib, html, io, json, sqlite3, sys, time
from pathlib import Path
from decimal import Decimal
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.poker_hand_assembly import assemble_messages
from parser_core.stack_temporal import hand_windows, POST, KIND

SOURCE = ROOT / 'runs/smoke14_start_stack_full_20260930_v1'
TIMELINE = ROOT / 'runs/smoke14_rowid_stack_parity_20260930_v1/ordered_timeline.jsonl'
PROOF = SOURCE / 'reviewed_start_stack_proof_postforced.json'
REVIEW = ROOT / 'runs/stack_proof_review_postforced_20260930.json'
PARENT = Path('C:/ParserData/runs/smoke14_full_tiny_ssd_20260929_v3')
OFFSETS = [-10,-5,-3,-2,-1,0,1,2,3,5,10]
MONEY = {'ANTE','SMALL_BLIND','BIG_BLIND','CALL','BET','RAISE','ALL_IN'}

def read_jsonl(p):
    return [json.loads(l) for l in p.open(encoding='utf-8') if l.strip()]

def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def dump(p, x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def jsonl(p,xs):
    with p.open('w',encoding='utf-8') as f:
        for x in xs: f.write(json.dumps(x,ensure_ascii=False)+'\n')

def resolved_path(s):
    return Path(s.replace('\\\\','\\'))

def observed_ledger(messages, player, cutoff):
    """Strict BEFORE cutoff; all amounts are observed deltas except RAISE_TO.
    Payout invalidates the requested formula: it must be subtracted separately.
    """
    gross=Decimal(0); returns=Decimal(0); paid=Decimal(0)
    street=Decimal(0); trace=[]; issues=[]; seen_forced=set()
    for m in messages:
        if m['frame_id'] >= cutoff: continue
        kind=m['action']
        if kind in {'FLOP','TURN','RIVER'}: street=Decimal(0)
        if m['player'] != player: continue
        if kind in MONEY | {'UNCALLED_RETURN','WIN'}:
            if m['amount'] is None:
                issues.append('MISSING_AMOUNT'); continue
            value=Decimal(m['amount']); delta=Decimal(0)
            if kind in MONEY:
                delta=value-street if kind=='RAISE' else value
                if delta<0: issues.append('NEGATIVE_RAISE_DELTA')
                gross+=delta
                if kind!='ANTE': street+=delta
                if kind in {'ANTE','SMALL_BLIND','BIG_BLIND'}:
                    if kind in seen_forced: issues.append('REPEATED_FORCED_POST_UNRESOLVED')
                    seen_forced.add(kind)
            elif kind=='UNCALLED_RETURN':
                returns+=value; street-=value
                if street<0: issues.append('RETURN_EXCEEDS_STREET_COMMITMENT')
            else: paid+=value; issues.append('PAYOUT_APPLIED_FORMULA_OUT_OF_SCOPE')
            trace.append({'frame':m['frame_id'],'action':kind,'amount':str(value),
                          'delta':str(delta),'source_row_ids':m['source_row_ids']})
    if 'ANTE' not in seen_forced: issues.append('ANTE_NOT_OBSERVED')
    return {'gross':str(gross),'returns':str(returns),'net':str(gross-returns),
            'payouts':str(paid),'trace':trace,'issues':sorted(set(issues))}

def unresolved_prefix(messages, known_players, cutoff):
    """Fail closed for unbound money actors or unparsed dealer evidence.
    Numeric/card continuation rows remain explicit diagnostics; they cannot
    silently supply money or erase a missing action.
    """
    issues=[]
    for m in messages:
        if m['frame_id']>=cutoff: continue
        if m['action'] in MONEY|{'UNCALLED_RETURN','WIN'} and m['player'] not in known_players:
            issues.append('UNBOUND_MONETARY_ACTOR')
        if m['action']=='UNKNOWN_EVENT':
            issues.append('UNPARSED_CHAT_ROW_LEDGER_COMPLETENESS_UNKNOWN')
    return sorted(set(issues))

def validate_supplement(s, rv, w):
    match=POST.fullmatch(s['raw_visible_text'])
    pairs={(v[f'{prefix}_frame'],v[f'{prefix}_table_sha256'])
           for v in rv.get('seats',{}).values() for prefix in ('stack','overlay')
           if f'{prefix}_frame' in v}
    if (not match or match.group(1)!=s['player'] or KIND[match.group(2).lower()]!=s['event_type']
        or Decimal(match.group(3).replace(' ',''))!=Decimal(s['amount'])
        or (s['frame_index'],s['table_sha256']) not in pairs
        or s['frame_index']<w['start_frame']
        or (w['end_frame'] is not None and s['frame_index']>=w['end_frame'])):
        raise ValueError('VISUAL_SUPPLEMENT_PROVENANCE_MISMATCH')

def audit(out):
    started=time.perf_counter()
    import cv2, numpy as np
    obs=read_jsonl(SOURCE/'start_stack_observations.jsonl'); rows=read_jsonl(TIMELINE)
    proof=json.loads(PROOF.read_text(encoding='utf-8'))
    review=json.loads(REVIEW.read_text(encoding='utf-8'))
    windows=hand_windows(rows)
    db=sqlite3.connect((PARENT/'journal.sqlite3').as_uri()+'?mode=ro',uri=True)
    errors=[]; checks=collections.Counter(); tables=[]; group_rows=collections.defaultdict(list)
    player_map={(h['table_id'],h['hand_id'],p['seat']):p
                for h in proof['hands'] for p in h['players'] if p['status']=='CERTIFIED'}
    review_map={(h['table_id'],h['hand_number']):h for h in review['hands']}
    window_map=collections.defaultdict(list)
    for w in windows: window_map[w['table_id']].append(w)
    messages_by_hand={}; events=[]
    for w in windows:
        msgs=assemble_messages(w['rows'],require_adjacency=True)
        # Same-frame tail rows can occur before next header in source order;
        # retain them as evidence, never assign their money across the boundary.
        msgs=[m for m in msgs if w['end_frame'] is None or m['frame_id']<w['end_frame']]
        rv=review_map.get((w['table_id'],w['hand_number']))
        if rv:
            for s in rv.get('visual_chat_supplement',[]):
                validate_supplement(s,rv,w)
                parent_path=PARENT/'crops'/'table_95984db438dc55f85dee9067'/f"{s['frame_index']:09d}_table.png"
                if digest(parent_path)!=s['table_sha256']: raise ValueError('VISUAL_SUPPLEMENT_TABLE_HASH_MISMATCH')
                if not any(m['action']==s['event_type'] and m['player']==s['player'] and m['frame_id']==s['frame_index'] for m in msgs):
                    msgs.append({'action':s['event_type'],'player':s['player'],'amount':s['amount'],
                        'frame_id':s['frame_index'],'source_row_ids':['VISUAL_TABLE_SHA256:'+s['table_sha256']],
                        'raw_text':s['raw_visible_text'],'evidence_origin':'EXISTING_HASH_PINNED_VISUAL_REVIEW'})
            msgs.sort(key=lambda m:m['frame_id'])
        messages_by_hand[(w['table_id'],w['hand_number'])]=msgs
        for m in msgs:
            events.append(dict(m,table=w['table_id'],hand_id=w['hand_number']))
    for i,o in enumerate(obs):
        if o['roi_type']!='stack': continue
        f=o['frame_index']; ws=window_map[o['table_id']]
        w=next((w for w in ws if w['start_frame']<=f and (w['end_frame'] is None or f<w['end_frame'])),None)
        hand=w['hand_number'] if w else None
        p=player_map.get((o['table_id'],hand,o['seat']))
        crop=SOURCE/o['crop_path']; reasons=[]
        if not crop.is_file(): reasons.append('CROP_MISSING')
        else:
            checks['crop_file_checks']+=1
            if digest(crop)!=o['crop_file_sha256']: reasons.append('CROP_FILE_HASH_MISMATCH')
            img=cv2.imdecode(np.fromfile(crop,dtype=np.uint8),cv2.IMREAD_COLOR); pixel=hashlib.sha256(img.tobytes()).hexdigest()
            checks['crop_pixel_checks']+=1
            # Stored crop hash is BGR raw array hash in the observer.
            if pixel!=o['crop_sha256']: reasons.append('CROP_PIXEL_HASH_MISMATCH')
        parent=db.execute("SELECT o.crop_sha256,f.source_pts,f.time_base_num,f.time_base_den FROM observations o JOIN frames f ON f.source_id=o.source_id AND f.frame_id=o.frame_id WHERE o.roi_type='table' AND o.table_session_id=? AND o.frame_id=?",(o['table_context'],f)).fetchone()
        checks['frame_mapping_checks']+=1
        if not parent or parent[0]!=o['table_crop_sha256'] or parent[1]!=o['source_pts'] or abs(parent[1]*parent[2]/parent[3]-o['timestamp'])>1e-8:
            reasons.append('FRAME_MAPPING_MISMATCH')
        if reasons: errors.append({'observation_id':i+1,'reasons':reasons})
        numeric=o.get('visual_kind')=='STACK' and o.get('visible_stack') is not None
        rec={'observation_id':i+1,'table':o['table_id'],'table_context':o['table_context'],'hand_id':hand,
             'seat':o['seat'],'player':p['nickname'] if p else None,'player_id':None,
             'frame':f,'source_pts':o['source_pts'],'timestamp':o['timestamp'],
             'displayed_stack':o['visible_stack'] if numeric else None,
             'raw_ocr':o['raw_recognition'],'confidence':o['confidence'],'visual_kind':o['visual_kind'],
             'crop_path':str(crop.resolve()),'crop_sha256':o['crop_sha256'],
             'table_crop_sha256':o['table_crop_sha256'], 'committed_before_frame':None,
             'returns_before_frame':None,'candidate_start_stack':None,'timing_offset':0,
             'status':'IDENTITY_UNKNOWN','reasons':reasons,'window_candidates':[]}
        if not p: rec['reasons'].append('NO_REVIEWED_SEAT_PLAYER_BINDING')
        elif not numeric: rec['status']='TIMING_UNKNOWN';rec['reasons'].append('STACK_NOT_NUMERIC')
        elif o['confidence']<.90: rec['status']='TIMING_UNKNOWN';rec['reasons'].append('STACK_OCR_LOW_CONFIDENCE')
        elif not reasons:
            msgs=messages_by_hand[(o['table_id'],hand)]
            l=observed_ledger(msgs,p['nickname'],f)
            candidate=Decimal(o['visible_stack'])+Decimal(l['net'])
            rec.update(committed_before_frame=l['gross'],returns_before_frame=l['returns'],
                       candidate_start_stack=str(candidate),ledger=l,status='PROBABLE')
            rec['reasons'].extend(l['issues'])
            known={q['nickname'] for key,q in player_map.items() if key[:2]==(o['table_id'],hand)}
            # Roster-only identity review may expose seats whose stack was
            # unreviewed. Their exact nick evidence prevents a known actor from
            # being mislabeled unknown without granting stack authority.
            known.update(q['nickname'] for h in proof['hands'] if h['table_id']==o['table_id'] and h['hand_id']==hand for q in h['players'])
            prefix_issues=unresolved_prefix(msgs,known,f)
            if prefix_issues: rec['reasons'].extend(prefix_issues);rec['status']='TIMING_UNKNOWN'
            # Identity mismatches block settlement; no actor alias is permitted.
            if any(m['frame_id']<f and m['action'] in MONEY|{'UNCALLED_RETURN','WIN'} and m['player'] and m['player'] not in {q['nickname'] for key,q in player_map.items() if key[:2]==(o['table_id'],hand)} and m['action'] in {'UNCALLED_RETURN','WIN'} for m in msgs):
                rec['reasons'].append('UNBOUND_SETTLEMENT_ACTOR'); rec['status']='IDENTITY_UNKNOWN'
            elif l['issues']: rec['status']='TIMING_UNKNOWN'
            if any(m['frame_id']==f and m['player']==p['nickname'] and m['action'] in MONEY|{'UNCALLED_RETURN','WIN'} for m in msgs):
                rec['reasons'].append('SAME_FRAME_ACTION_ORDER_UNKNOWN');rec['status']='TIMING_UNKNOWN'
            for offset in OFFSETS:
                wl=observed_ledger(msgs,p['nickname'],f+offset)
                rec['window_candidates'].append({'offset':offset,'candidate':str(Decimal(o['visible_stack'])+Decimal(wl['net'])),'issues':sorted(set(wl['issues']+unresolved_prefix(msgs,known,f+offset))),'diagnostic_only':offset>0})
        tables.append(rec); group_rows[(o['table_id'],hand,o['seat'])].append(rec)
    groups=[]
    for key,rs in group_rows.items():
        ns=[r for r in rs if r['candidate_start_stack'] is not None]
        good=[r for r in ns if r['status']=='PROBABLE']
        values=[Decimal(r['candidate_start_stack']) for r in good]
        states={(r['committed_before_frame'],r['returns_before_frame']) for r in good}
        # Agreement is diagnostic until CHAT/stack timing is actually bounded.
        status='IDENTITY_UNKNOWN'
        if ns:
            if any(r['status']=='IDENTITY_UNKNOWN' for r in ns): status='IDENTITY_UNKNOWN'
            elif len(set(values))>1: status='CONFLICT'
            elif any(r['status']=='TIMING_UNKNOWN' for r in ns): status='TIMING_UNKNOWN'
            elif len(states)>=2: status='TIMING_UNKNOWN'
            elif good: status='PROBABLE'
        spread=str(max(values)-min(values)) if values else None
        groups.append({'table':key[0],'hand_id':key[1],'seat':key[2],
            'player':ns[0]['player'] if ns else None,'numeric_observations':len(ns),
            'usable_observations':len(good),'independent_commitment_states':len(states),
            'exact_diagnostic_agreement':bool(len(states)>=2 and len(set(values))==1),
            'candidates':sorted({str(v) for v in values},key=Decimal),'max_difference':spread,
            'status':status,'observation_ids':[r['observation_id'] for r in rs],
            'reason':'NUMERIC_VISIBILITY_GAP_TIMING_UNPROVEN' if len(states)>=2 and len(set(values))==1 else None})
    continuity=[]
    for seat in [f'seat{i}' for i in range(1,9)]:
        prior=[r for r in tables if r['table']=='T01' and r['hand_id']=='2875498927' and r['seat']==seat and r['status']=='PROBABLE']
        later=[r for r in tables if r['table']=='T01' and r['hand_id']=='2875499893' and r['seat']==seat and r['status']=='PROBABLE']
        if not prior or not later: continue
        old=prior[0]; new=later[0]
        mm=messages_by_hand[('T01','2875498927')]
        folded=any(m['action']=='FOLD' and m['player']==old['player'] for m in mm)
        ledger=observed_ledger(mm,old['player'],3060)
        expected=Decimal(old['candidate_start_stack'])-Decimal(ledger['net'])+Decimal(ledger['payouts'])
        continuity.append({'table':'T01','seat':seat,'player':old['player'],
            'previous_hand':'2875498927','next_hand':'2875499893',
            'previous_start_diagnostic':old['candidate_start_stack'],
            'previous_gross':ledger['gross'],'previous_returns':ledger['returns'],'previous_payouts':ledger['payouts'],
            'expected_end_stack':str(expected),'next_start_diagnostic':new['candidate_start_stack'],
            'exact_agreement':expected==Decimal(new['candidate_start_stack']),
            'status':'DIAGNOSTIC_AGREEMENT' if folded and not ledger['issues'] and expected==Decimal(new['candidate_start_stack']) else 'UNRESOLVED',
            'fold_observed':folded,'source_observation_ids':[old['observation_id'],new['observation_id']],
            'limitation':'Reviewed seat/nickname at both hand starts; continuous occupancy and CHAT timing unproven; never used to set candidates.'})
    jsonl(out/'continuity_checks.jsonl',continuity)
    jsonl(out/'observations.jsonl',tables);jsonl(out/'events.jsonl',events);jsonl(out/'player_hands.jsonl',groups)
    counts=collections.Counter(g['status'] for g in groups)
    summary={'schema':'experimental.start-stack-frame-ledger.v1','verdict':'NOT_PROVEN',
        'dataset':'Smoke14 saved evidence, all 9 tables; deep sample T01 #2875498927/#2875499893',
        'total_stack_observations':len(tables),'numeric_stack_observations':sum(r['displayed_stack'] is not None for r in tables),
        'usable_observations':sum(r['status']=='PROBABLE' for r in tables),
        'reviewed_numeric_observations':sum(r['candidate_start_stack'] is not None for r in tables),
        'players_hands_with_at_least_two_usable_observations':sum(g['usable_observations']>=2 for g in groups),
        'groups_with_multiple_commitment_states':sum(g['independent_commitment_states']>=2 for g in groups),
        'group_status_counts':dict(counts),'exact_agreement_count':sum(g['exact_diagnostic_agreement'] for g in groups),
        'max_difference':str(max((Decimal(g['max_difference']) for g in groups if g['max_difference'] is not None),default=Decimal(0))),
        'provenance_checks':dict(checks),'provenance_errors':errors,
        'timing_result':'NOT_MEASURED_YET','future_actions_used_for_primary_candidates':0,
        'adjacent_hand_diagnostic_checks':len(continuity),'adjacent_hand_diagnostic_agreements':sum(c['status']=='DIAGNOSTIC_AGREEMENT' for c in continuity),
        'notes':['Usable means numeric/hash/frame/seat evidence and no observed ledger issues; timing NOT certified.',
                 'No alias m_a_j_o_r -> m_a_j_0_r. Repeated ANTE is kept and flagged.',
                 'Folded unchanged-stack repetitions do not supply different commitment states.',
                 'Post-payout formula needs -payouts; requested formula is scoped before winnings apply.',
                 'HAND_ID windows are provisional; later stale NEW_HAND rows remain in events.',
                 'No fully observed, trusted adjacent-hand settlement: cross-hand check NOT_PROVEN.'],
        'elapsed_seconds':time.perf_counter()-started}
    dump(out/'summary.json',summary)
    dump(out/'source_manifest.json',{'inputs':[{'path':str(p),'sha256':digest(p)} for p in [SOURCE/'start_stack_observations.jsonl',TIMELINE,PROOF,REVIEW,ROOT/'production_manifest.json',ROOT/'configs/layout_847x404.json']],
        'journal':str(PARENT/'journal.sqlite3'),'journal_access':'READ_ONLY_SQLITE_URI',
        'source_video_sha256':proof['source']['sha256'],'production_changes':False,'frozen_vendor_changes':False})
    render(out,tables,groups,summary)
    db.close()
    return summary

def render(out,rs,groups,summary):
    cols=['table','hand_id','seat','player','frame','displayed_stack','committed_before_frame','returns_before_frame','candidate_start_stack','timing_offset','status','raw_ocr','confidence','reasons']
    def tab(data,cols):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(c,'')))+'</td>' for c in cols)+'</tr>' for r in data)+'</tbody></table>'
    page='<!doctype html><meta charset="utf-8"><title>START_STACK frame ledger — Smoke14</title><style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse;margin:20px 0}td,th{border:1px solid #ccc;padding:5px;text-align:left}th{position:sticky;top:0;background:#eee}pre{white-space:pre-wrap}</style><h1>START_STACK frame ledger — NOT_PROVEN</h1><p>Experimental diagnostics. Timing is unproven; no production certification. Primary cutoff is strictly before frame; positive offsets are future-only diagnostics.</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><h2>Player / hand groups</h2>'+tab([g for g in groups if g['player']],['table','hand_id','seat','player','usable_observations','independent_commitment_states','candidates','max_difference','status','reason'])+'<h2>All stack observations</h2>'+tab(rs,cols)
    (out/'report.html').write_text(page,encoding='utf-8')

def timing(out):
    """Bounded fresh offline OCR of exact saved TABLE pixels, ±10 through +60.
    Deduplicate identical pixel payloads only; every observation remains saved.
    """
    import cv2, numpy as np
    from parser_core.tiny_chat_client import TinyChatClient
    events=read_jsonl(out/'events.jsonl'); groups=read_jsonl(out/'player_hands.jsonl')
    pmap={(g['hand_id'],g['player']):g['seat'] for g in groups if g['player']}
    targets=[e for e in events if e['table']=='T01' and e['hand_id'] in {'2875498927','2875499893'} and e['action'] in {'RAISE','CALL','BET','UNCALLED_RETURN'} and (e['hand_id'],e['player']) in pmap]
    config=json.loads((ROOT/'configs/ocr_tiny_local.json').read_text(encoding='utf-8'))
    profile=json.loads((ROOT/'configs/layout_847x404.json').read_text(encoding='utf-8'))
    db=sqlite3.connect((PARENT/'journal.sqlite3').as_uri()+'?mode=ro',uri=True)
    refs=[];unique={}; missing=[]
    for ti,e in enumerate(targets):
        seat=pmap[(e['hand_id'],e['player'])]; box=profile['slots']['slot_01']['seats'][seat]['stack']
        for f in range(e['frame_id']-10,e['frame_id']+61):
            row=db.execute("SELECT crop_path,crop_sha256 FROM observations WHERE roi_type='table' AND table_session_id='table_95984db438dc55f85dee9067' AND frame_id=?",(f,)).fetchone()
            if not row: missing.append([ti,f]);continue
            path=PARENT/row[0]; img=cv2.imread(str(path));
            if digest(path)!=row[1]: raise ValueError('TABLE_FILE_HASH_MISMATCH')
            crop=img[box[1]:box[3],box[0]:box[2]]
            key=hashlib.sha256(crop.tobytes()).hexdigest()
            gray_hash=hashlib.sha256(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).tobytes()).hexdigest()
            if key not in unique:
                packed=cv2.imencode('.png',crop)[1].tobytes()
                unique[key]={'item_id':key,'source_sha256':gray_hash,'png_base64':base64.b64encode(packed).decode()}
            refs.append({'target_index':ti,'hand_id':e['hand_id'],'seat':seat,'player':e['player'],
                         'event':e['action'],'amount':e['amount'],'chat_frame':e['frame_id'],
                         'frame':f,'offset':f-e['frame_id'],'table_path':str(path),'table_file_sha256':row[1],
                         'stack_roi':box,'stack_bgr_sha256':key,'stack_gray_sha256':gray_hash,'source_row_ids':e['source_row_ids']})
    client=TinyChatClient(Path(config['python']),Path(config['model_dir']),config['model_name'],config['weights_sha256'],config['device'],out/'timing_tiny.stderr.log')
    predictions={}; items=list(unique.values())
    try:
        for start in range(0,len(items),8):
            r=client.read_batch(items[start:start+8],8)
            for x in r['results']: predictions[x['item_id']]=x
            print('TIMING_OCR',min(start+8,len(items)),'/',len(items),flush=True)
    finally:client.close()
    for r in refs: r['ocr']=predictions[r['stack_bgr_sha256']]
    jsonl(out/'timing_observations.jsonl',refs)
    result={'targets':len(targets),'source_observations':len(refs),'unique_ocr_payloads':len(items),
            'missing_frames':missing,'tested_offsets':OFFSETS,'capture_step':1,'range':'CHAT frame-10..+60',
            'numeric_visibility_delay_is_not_stack_debit_delay':True,
            'chat_stack_debit_delay_frames':None,'status':'TIMING_UNKNOWN','targets_detail':[]}
    for ti,e in enumerate(targets):
        rr=[r for r in refs if r['target_index']==ti]
        def num(r):
            raw=r['ocr']['raw_text'].replace(' ','').replace('\u00a0','')
            return int(raw) if raw.isdigit() and r['ocr']['score']>=.90 else None
        before=[r for r in rr if r['offset']<0 and num(r) is not None]
        old=num(before[-1]) if before else None
        changed=[r for r in rr if r['offset']>=0 and num(r) is not None and num(r)!=old]
        first=changed[0] if changed else None
        result['targets_detail'].append({'target_index':ti,'hand_id':e['hand_id'],'player':e['player'],'event':e['action'],
             'chat_frame':e['frame_id'],'old_visible':old,'first_changed_numeric_frame':first['frame'] if first else None,
             'first_changed_numeric_offset':first['offset'] if first else None,'new_visible':num(first) if first else None,
             'chat_frame_stack_raw':next((r['ocr']['raw_text'] for r in rr if r['offset']==0),None),
             'interpretation':'NUMERIC_STACK_OCCLUDED_BY_ACTION_OVERLAY; actual debit onset unobserved'})
    delays=[x['first_changed_numeric_offset'] for x in result['targets_detail'] if x['first_changed_numeric_offset'] is not None]
    result['numeric_reappearance_delay_min']=min(delays) if delays else None
    result['numeric_reappearance_delay_max']=max(delays) if delays else None
    dump(out/'timing_summary.json',result)
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'));summary['timing_result']=result
    dump(out/'summary.json',summary); render(out,read_jsonl(out/'observations.jsonl'),groups,summary)
    # Compact raw evidence contact sheet; all crop source coordinates are retained above.
    from PIL import Image,ImageDraw
    shown=[-2,-1,0,1,2,3,5,10,44,45,46,47,48,49,50,60]
    sheet=Image.new('RGB',(180*len(shown),85*len(targets)), 'white'); draw=ImageDraw.Draw(sheet)
    for ti,e in enumerate(targets):
        for ci,offset in enumerate(shown):
            r=next((r for r in refs if r['target_index']==ti and r['offset']==offset),None)
            if not r:continue
            im=Image.open(r['table_path']);c=im.crop(r['stack_roi']);c=c.resize((c.width*2,c.height*2))
            sheet.paste(c,(ci*180,ti*85+22))
            draw.text((ci*180,ti*85),f"{ti} f{r['frame']} ({offset:+})",fill='black')
            draw.text((ci*180,ti*85+61),r['ocr']['raw_text'],fill='black')
    sheet.save(out/'timing_contact_sheet.png')
    db.close()
    return result

def postdeal(out):
    """Causal gate attested from real TABLE pixels; first exact stable ROI.
    Three consecutive numeric OCR states are the explicit experimental stability
    rule, not a claim about PokerDom timing. Evidence is confirmed at frame+2.
    """
    import cv2,numpy as np
    gates=[{'table':'T01','hand_id':'2875498927','frame':728,'end':3060},
           {'table':'T01','hand_id':'2875499893','frame':3106,'end':6261}]
    profile=json.loads((ROOT/'configs/layout_847x404.json').read_text(encoding='utf-8'))
    rs=read_jsonl(out/'observations.jsonl');events=read_jsonl(out/'events.jsonl')
    db=sqlite3.connect((PARENT/'journal.sqlite3').as_uri()+'?mode=ro',uri=True)
    from parser_core.tiny_chat_client import TinyChatClient
    config=json.loads((ROOT/'configs/ocr_tiny_local.json').read_text(encoding='utf-8'))
    payloads={};probes=[]
    for gate in gates:
        for f in range(gate['frame']+1,gate['frame']+13):
            row=db.execute("SELECT crop_path,crop_sha256 FROM observations WHERE roi_type='table' AND table_session_id='table_95984db438dc55f85dee9067' AND frame_id=?",(f,)).fetchone()
            if not row:raise ValueError('POSTDEAL_SOURCE_FRAME_MISSING')
            path=PARENT/row[0]
            if digest(path)!=row[1]:raise ValueError('POSTDEAL_TABLE_HASH_MISMATCH')
            img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
            for seat,boxes in profile['slots']['slot_01']['seats'].items():
                box=boxes['stack'];crop=img[box[1]:box[3],box[0]:box[2]]
                key=hashlib.sha256(crop.tobytes()).hexdigest()
                if key not in payloads:
                    payloads[key]={'item_id':key,'source_sha256':hashlib.sha256(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).tobytes()).hexdigest(),
                        'png_base64':base64.b64encode(cv2.imencode('.png',crop)[1].tobytes()).decode()}
                probes.append({'table':'T01','hand_id':gate['hand_id'],'seat':seat,'frame':f,'crop_sha256':key,'table_path':str(path),'table_file_sha256':row[1],'stack_roi':box})
    client=TinyChatClient(Path(config['python']),Path(config['model_dir']),config['model_name'],config['weights_sha256'],config['device'],out/'postdeal_tiny.stderr.log')
    predictions={};items=list(payloads.values())
    try:
        for start in range(0,len(items),8):
            for r in client.read_batch(items[start:start+8],8)['results']:predictions[r['item_id']]=r
    finally:client.close()
    for p in probes:p['ocr']=predictions[p['crop_sha256']]
    jsonl(out/'postdeal_numeric_probes.jsonl',probes)
    snapshots=[]
    for gate in gates:
        path=PARENT/'crops'/'table_95984db438dc55f85dee9067'/f"{gate['frame']:09d}_table.png"
        gate.update(table_path=str(path),table_file_sha256=digest(path),
            status='DEAL_COMPLETE_CONFIRMED',confirmation_basis='Direct visual inspection: all eight occupied seats show two dealt card backs; no card-deal overlay; hand ID matches CHAT. This is a confirmed completed state, not exact completion onset.')
        rr=[r for r in rs if r['table']=='T01' and r['hand_id']==gate['hand_id']]
        for seat in [f'seat{i}' for i in range(1,9)]:
            box=profile['slots']['slot_01']['seats'][seat]['stack']
            found=None
            seatprobes=[p for p in probes if p['hand_id']==gate['hand_id'] and p['seat']==seat]
            for index in range(len(seatprobes)-2):
                triple=seatprobes[index:index+3];numbers=[]
                for probe in triple:
                    raw=probe['ocr']['raw_text'].replace(' ','').replace('\u00a0','')
                    numbers.append(raw if raw.isdigit() and probe['ocr']['score']>=.90 else None)
                if None in numbers or len(set(numbers))!=1:continue
                start=triple[0]['frame']
                r=next((r for r in rr if r['seat']==seat and r.get('player')),None)
                if r is None:
                    r={'table':'T01','hand_id':gate['hand_id'],'seat':seat,'player':None,'status':'IDENTITY_UNKNOWN'}
                found=dict(r,snapshot_frame=start,stable_confirmed_frame=start+2,
                    stable_source_frames=[p['frame'] for p in triple],stable_table_paths=[p['table_path'] for p in triple],
                    deal_complete_confirmed_frame=gate['frame'],ocr_source_observation_frame=start,
                    stability_policy='3_CONSECUTIVE_IDENTICAL_NUMERIC_OCR_VALUES',future_leak=False,
                    displayed_stack=numbers[0],raw_ocr=triple[0]['ocr']['raw_text'],confidence=triple[0]['ocr']['score'],
                    frame=start,stable_crop_sha256=[p['crop_sha256'] for p in triple],status='IDENTITY_UNKNOWN')
                if r['player']:
                    mm=[e for e in events if e['table']=='T01' and e['hand_id']==gate['hand_id']]
                    ledger=observed_ledger(mm,r['player'],start)
                    roster=json.loads(PROOF.read_text(encoding='utf-8'))
                    known={p['nickname'] for h in roster['hands'] if h['table_id']=='T01' and h['hand_id']==gate['hand_id'] for p in h['players']}
                    issues=ledger['issues']+unresolved_prefix(mm,known,start)
                    found.update(committed_before_frame=ledger['gross'],returns_before_frame=ledger['returns'],
                        candidate_start_stack=str(Decimal(numbers[0])+Decimal(ledger['net'])),ledger=ledger,
                        reasons=issues,status='TIMING_UNKNOWN' if issues else 'PROBABLE')
                found['ocr_evidence_file']='postdeal_numeric_probes.jsonl'
                found.pop('window_candidates',None)
                found.pop('crop_path',None)
                found.pop('crop_sha256',None)
                found.pop('observation_id',None)
                snapshots.append(found);break
            if not found:snapshots.append({'table':'T01','hand_id':gate['hand_id'],'seat':seat,'status':'TIMING_UNKNOWN','reason':'NO_STABLE_NUMERIC_TRIPLE_IN_BOUNDED_FIRST_12_FRAMES','deal_complete_confirmed_frame':gate['frame']})
    jsonl(out/'postdeal_first_stable_snapshots.jsonl',snapshots)
    dump(out/'deal_complete_evidence.json',gates)
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    summary['postdeal_policy']={'gate':'Confirmed DEAL_COMPLETE only; forced-post or CHAT DEAL frame is not the gate.',
        'stable_definition':'3 consecutive identical numeric OCR values, confidence >=0.90; confirm causally at snapshot_frame+2.',
        'searched_source_frames_after_each_gate':12,'probe_count':len(probes),'exact_bgr_ocr_payloads':len(items),
        'reviewed_hands':2,'seat_opportunities':16,'stable_numeric_snapshots':sum('snapshot_frame' in s for s in snapshots),
        'identity_known_snapshots':sum(bool(s.get('player')) for s in snapshots),
        'snapshot_frames':sorted({s['snapshot_frame'] for s in snapshots if 'snapshot_frame' in s}),
        'snapshot_frame_is_forced_post_frame':False,
        'confirmation_is_completed_state_not_exact_deal_onset':True,
        'new_policy_certified':0,'verdict':'NOT_PROVEN'}
    summary['timing_result']=json.loads((out/'timing_summary.json').read_text(encoding='utf-8'))
    dump(out/'summary.json',summary);render(out,rs,read_jsonl(out/'player_hands.jsonl'),summary)
    db.close();return summary['postdeal_policy']

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--timing',action='store_true');parser.add_argument('--postdeal',action='store_true')
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    print(json.dumps(postdeal(a.output) if a.postdeal else timing(a.output) if a.timing else audit(a.output),ensure_ascii=False,indent=2))
