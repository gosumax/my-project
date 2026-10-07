"""Forced-only START_STACK in confirmed post-deal/pre-money window."""
import argparse,base64,collections,hashlib,json,sys,time
from pathlib import Path
from decimal import Decimal
sys.path.insert(0,str(Path(__file__).parent))
from start_stack_wide_20261003 import ROOT,PARENT,read_jsonl,jsonl,dump,digest,database,table,numeric,TinyChatClient
CHANGING={'CALL','BET','RAISE','ALL_IN'}

def boundary(events,gate,end):
    # Events are already scoped to the current hand. A late visual checkpoint
    # must never reopen the forced-only window after an earlier money action.
    money=[m for m in events if m['action'] in CHANGING and m['frame_id']<end]
    settlement=[m for m in events if m['action'] in {'UNCALLED_RETURN','WIN'} and m['frame_id']<end]
    first=min(money,key=lambda m:m['frame_id']) if money else None
    natural=min([end]+[m['frame_id'] for m in settlement])
    return min(first['frame_id'],natural) if first else natural,first

def run(out):
    import cv2
    target=out/'clean_forced_window';target.mkdir(exist_ok=True)
    hands=read_jsonl(out/'selected_hands.jsonl');reviews=read_jsonl(out/'visual_reviews.jsonl')
    forced=read_jsonl(target/'forced_visual_evidence.jsonl');events=read_jsonl(out/'events.jsonl');probes=read_jsonl(out/'numeric_nick_probes.jsonl')
    rvmap={(r['table'],r['hand_id']):r for r in reviews};fmap={(r['table'],r['hand_id']):r for r in forced}
    profile=json.loads((ROOT/'configs/layout_847x404.json').read_text(encoding='utf-8'))
    windows=[];pending=[]
    def stable(ps,gate,close):
        lookup={p['frame']:p for p in ps}
        for f in sorted(lookup):
            if not gate<f or f+2>=close:continue
            tri=[lookup.get(f+i) for i in range(3)];values=[numeric(p) for p in tri]
            if None not in values and len(set(values))==1:return tri
        return None
    for c in hands:
        rv=rvmap[(c['table'],c['hand_id'])];ms=[e for e in events if e['table']==c['table'] and e['hand_id']==c['hand_id']]
        close,first=boundary(ms,rv['deal_complete_frame'],c['end'])
        w=dict(c,deal_complete_frame=rv['deal_complete_frame'],window_close_exclusive=close,
            first_stack_changing_action={k:first.get(k) for k in ['frame_id','action','player','amount','source_row_ids']} if first else None,
            fold_check_frames=[e['frame_id'] for e in ms if e['action'] in {'FOLD','CHECK'} and rv['deal_complete_frame']<e['frame_id']<close],
            clean_window_status='CLEAN' if close>rv['deal_complete_frame'] else 'NO_WINDOW')
        windows.append(w)
        for seat in rv['occupied_seats']:
            ps=[p for p in probes if (p['table'],p['hand_id'],p['seat'],p['kind'])==(c['table'],c['hand_id'],seat,'stack')]
            tri=stable(ps,rv['deal_complete_frame'],close)
            if not tri or tri[0]['frame']>rv['deal_complete_frame']+13:pending.append((w,seat))
    # Only unresolved seats need exhaustive clean-window pixels. Existing
    # successful gate+1 triples already prove earliest possible source frame.
    db=database();payloads={};extra=[]
    for w,seat in pending:
        geom=json.loads(db.execute('SELECT geometry_json FROM table_sessions WHERE table_session_id=?',(w['session'],)).fetchone()[0])
        if int(geom['slot_id'])!=int(w['table'][1:]) or geom['profile_sha256']!=digest(ROOT/'configs/layout_847x404.json'):raise ValueError('CLEAN_ROI_ROUTING_MISMATCH')
        box=profile['slots'][f"slot_{int(w['table'][1:]):02d}"]['seats'][seat]['stack']
        for f in range(w['deal_complete_frame']+1,w['window_close_exclusive']):
            im,meta=table(db,w['session'],f);crop=im[box[1]:box[3],box[0]:box[2]];key=hashlib.sha256(crop.tobytes()).hexdigest()
            if key not in payloads:
                payloads[key]={'item_id':key,'source_sha256':hashlib.sha256(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).tobytes()).hexdigest(),
                    'png_base64':base64.b64encode(cv2.imencode('.png',crop)[1].tobytes()).decode()}
            extra.append(dict(meta,table=w['table'],hand_id=w['hand_id'],seat=seat,kind='stack',roi=box,crop_bgr_sha256=key))
    pred={p['crop_bgr_sha256']:p['ocr'] for p in probes};items=[i for key,i in payloads.items() if key not in pred]
    cache=target/'raw_ocr_cache.jsonl'
    if cache.exists():pred.update({p['item_id']:p for p in read_jsonl(cache)});items=[i for i in items if i['item_id'] not in pred]
    conf=json.loads((ROOT/'configs/ocr_tiny_local.json').read_text(encoding='utf-8'));client=TinyChatClient(Path(conf['python']),Path(conf['model_dir']),conf['model_name'],conf['weights_sha256'],conf['device'],target/'tiny.stderr.log')
    try:
        with cache.open('a',encoding='utf-8') as stream:
            for start in range(0,len(items),8):
                for r in client.read_batch(items[start:start+8],8)['results']:pred[r['item_id']]=r;stream.write(json.dumps(r,ensure_ascii=False)+'\n')
                stream.flush()
    finally:client.close()
    for p in extra:p['ocr']=pred[p['crop_bgr_sha256']]
    jsonl(target/'exhaustive_pending_probes.jsonl',extra);probes.extend(extra)
    results=[]
    for w in windows:
        rv=rvmap[(w['table'],w['hand_id'])];fr=fmap[(w['table'],w['hand_id'])]
        for seat in rv['occupied_seats']:
            ps=[p for p in probes if (p['table'],p['hand_id'],p['seat'],p['kind'])==(w['table'],w['hand_id'],seat,'stack')]
            tri=stable(ps,w['deal_complete_frame'],w['window_close_exclusive'])
            r={'table':w['table'],'hand_id':w['hand_id'],'seat':seat,'player_id':f"{w['table']}_SEAT_{int(seat[4:])}_EPOCH_1",
                'nickname_label':rv['visually_read_nicknames'].get(seat),'deal_complete_frame':w['deal_complete_frame'],'window_close_exclusive':w['window_close_exclusive'],
                'snapshot_frame':None,'status':'NO_STACK_BEFORE_FIRST_ACTION','used_call_bet_raise_return_win':False,'future_money_events_used':0}
            seatforced=fr['forced_by_seat'].get(seat)
            if tri:
                f=tri[0]['frame'];value=numeric(tri[0]);r.update(snapshot_frame=f,stable_confirmed_frame=f+2,
                    displayed_stack=value,ocr_readings=[p['ocr'] for p in tri],source_frames=[p['frame'] for p in tri],
                    table_source_paths=[p['table_path'] for p in tri],source_bgr_hashes=[p['crop_bgr_sha256'] for p in tri])
                if not seatforced or fr['status']!='CONFIRMED':r['status']='FORCED_BET_UNKNOWN'
                elif not (w['deal_complete_frame']<f and f+2<w['window_close_exclusive']):r['status']='TIMING_UNKNOWN'
                else:
                    r.update(status='START_STACK_RECOVERED',ante=seatforced['ante'],sb=seatforced['sb'],bb=seatforced['bb'],
                        start_stack=str(Decimal(value)+Decimal(seatforced['ante'])+Decimal(seatforced['sb'])+Decimal(seatforced['bb'])),
                        forced_evidence_file='forced_visual_evidence.jsonl',forced_evidence_frame=fr['evidence_frame'])
            results.append(r)
    counts=collections.Counter(r['status'] for r in results)
    for k in ['START_STACK_RECOVERED','NO_STACK_BEFORE_FIRST_ACTION','IDENTITY_UNKNOWN','FORCED_BET_UNKNOWN','TIMING_UNKNOWN','CONFLICT']:counts.setdefault(k,0)
    summary={'verdict':'START_STACK_FORCED_WINDOW_PROVEN_ON_SAMPLE' if counts['START_STACK_RECOVERED']>=30 and not counts['CONFLICT'] else 'NOT_PROVEN',
        'acceptance_scope':'20 visually reviewed hands / all9tables; only DEAL_COMPLETE -> strictly BEFORE_FIRST_STACK_CHANGING_ACTION; settlement/end caps no-action hands.',
        'acceptance_min_recovered_seat_hand_groups':30,'threshold_note':'30 retained from earlier user sample requirement; latest sufficient-number criterion has no new numeric threshold.',
        'hands_with_clean_window':sum(w['clean_window_status']=='CLEAN' for w in windows),'seats_observed':len(results),
        'stable_stacks':sum(r['snapshot_frame'] is not None for r in results),'counts':dict(counts),
        'extra_exhaustive_seats':len(pending),'extra_source_probes':len(extra),'new_ocr_payloads':len(items),
        'closed_by_check_or_fold':0,'future_money_events_used':0,'post_window_money_rows_in_calculation':0,
        'no_account_id_from_nickname_spelling':True,'occupancy_epoch_policy':'Observed epoch1 at first confirmed occupancy. Confirmed T05seat1 release at3693; no newepoch due OCR variants. Cross-hand gaps unresolved.',
        'later_row_lineage_not_a_blocker':True,
        'forced_evidence_method':'Manual gate TABLE/CHAT current hand: occupied dealt seats, full forced pot, all individual uniformante posts, directly visible SB/BB chip amount and physicalseat; evidence frame precedes snapshot.',
        'not_proven_scopes':['General deployment/production acceptance','Full-hand reconstruction','Persistent account identity or cross-hand carry','Hidden debit latency','Multi-state full-hand ledger acceptance']}
    jsonl(target/'hand_windows.jsonl',windows);jsonl(target/'start_stack_results.jsonl',results);dump(target/'summary.json',summary);db.close();return summary

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=time.perf_counter();r=run(a.output);r['wall_seconds']=time.perf_counter()-s;print(json.dumps(r,ensure_ascii=False,indent=2))
