"""Seat occupancy identity and independent CHAT lineage audit, offline only."""
import collections,hashlib,json,sys,sqlite3
from pathlib import Path
from decimal import Decimal
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(Path(__file__).parent))
from start_stack_wide_20261003 import ROOT,PARENT,MONEY,read_jsonl,dump,jsonl,digest,database,observed_ledger,classify,uncertain_row_identity

def run(out):
    import cv2,numpy as np
    target=out/'seat_identity_reclassification';target.mkdir(exist_ok=True)
    events=read_jsonl(out/'events.jsonl');reviews=read_jsonl(out/'visual_reviews.jsonl');selected=read_jsonl(out/'selected_hands.jsonl')
    probes=read_jsonl(out/'numeric_nick_probes.jsonl');prior=read_jsonl(out/'stack_observations.jsonl')
    reviewmap={(r['table'],r['hand_id']):r for r in reviews}
    hands={(h['table'],h['hand_id']):h for h in selected};db=database()
    # Epoch1 is the first confirmed observed occupant, not a historical account
    # ID. Unknown gaps do not spawn epochs or certify cross-hand continuity.
    occupancy=[]
    for table in sorted({r['table'] for r in reviews}):
        for seat in sorted({s for r in reviews if r['table']==table for s in r['occupied_seats']}):
            first=min(r['deal_complete_frame'] for r in reviews if r['table']==table and seat in r['occupied_seats'])
            occupancy.append({'table':table,'seat':seat,'occupancy_epoch':1,'player_id':f'{table}_SEAT_{int(seat[4:])}_EPOCH_1',
                'first_confirmed_occupied_frame':first,'historical_epoch_number_unknown':True,
                'confirmed_release_frame':3693 if table=='T05' and seat=='seat1' else None,
                'release_evidence':reviewmap[('T05','2875499590')]['completion_evidence'] if table=='T05' and seat=='seat1' else None,
                'cross_hand_continuity':'UNKNOWN_UNOBSERVED_GAPS','new_epoch_policy':'Only confirmed release/change/new occupant. OCR spelling never increments epoch.'})
    jsonl(target/'occupancy_epochs.jsonl',occupancy)
    nickprobes=collections.defaultdict(list)
    for p in probes:
        if p['kind']=='nick':nickprobes[(p['table'],p['hand_id'],p['seat'])].append(p)
    def bind_actor(m,cutoff):
        rv=reviewmap[(m['table'],m['hand_id'])];matches=[];evidence=[]
        for seat in rv['occupied_seats']:
            # Labels are evidence for the fixed seat ID, not canonical names.
            direct=rv['visually_read_nicknames'].get(seat)==m['player'] and rv['deal_complete_frame']<cutoff
            supporting=[p for p in nickprobes[(m['table'],m['hand_id'],seat)] if p['frame']<=cutoff and p['ocr']['score']>=.90 and p['ocr']['raw_text']==m['player'] and '...' not in p['ocr']['raw_text']]
            if direct or supporting:
                matches.append(seat);evidence.append({'seat':seat,'visual_gate_match':direct,'table_nickname_evidence_frames':[p['frame'] for p in supporting][:4]})
        if len(matches)!=1:return None,evidence
        return matches[0],evidence
    appearance_cache={}
    def appearances(session,frame):
        if (session,frame) not in appearance_cache:
            rows=db.execute("SELECT observation_id,crop_path,crop_sha256,quality_json FROM observations WHERE table_session_id=? AND frame_id=? AND roi_type='CHAT_ROW'",(session,frame)).fetchall()
            found=[]
            for oid,path,filehash,quality in rows:
                p=PARENT/path
                if digest(p)!=filehash:raise ValueError('CHAT_ROW_FILE_HASH_MISMATCH')
                pixels=cv2.imdecode(np.fromfile(p,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
                found.append({'observation_id':oid,'crop_path':str(p),'file_sha256':filehash,
                    'pixel_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),'ordinal':json.loads(quality)['source_span_index']})
            appearance_cache[(session,frame)]=found
        return appearance_cache[(session,frame)]
    audits=[];audit_by_sources={}
    for index,m in enumerate(e for e in events if e['action'] in MONEY):
        h=hands[(m['table'],m['hand_id'])];rv=reviewmap[(m['table'],m['hand_id'])]
        cutoff=max(m['frame_id'],rv['deal_complete_frame']+1)
        seat,binding=bind_actor(m,cutoff)
        sources=[];lineage=[];prev_hashes={r['pixel_sha256'] for r in appearances(h['session'],m['frame_id']-1)}
        current=appearances(h['session'],m['frame_id']);maxordinal=max((r['ordinal'] for r in current),default=0)
        for r in m['source_rows']:
            appearance=next((a for a in current if a['ordinal']==r['row_index'] and a['pixel_sha256']==r['row_pixel_sha256']),None)
            if appearance:
                source=dict(appearance,source_frame=r['frame_id'],timeline_row_id=r['stable_row_id'],present_in_previous_frame=r['row_pixel_sha256'] in prev_hashes)
                p=db.execute('SELECT p.physical_row_id,p.state,p.reason,p.first_source_pts,p.last_source_pts FROM row_observations ro JOIN physical_rows p ON p.physical_row_id=ro.physical_row_id WHERE ro.observation_id=?',(appearance['observation_id'],)).fetchone()
                source['legacy_lineage_evidence']=dict(zip(['physical_row_id','state','reason','first_source_pts','last_source_pts'],p)) if p else None
                sources.append(source)
            else:lineage.append('ROW_PIXEL_FRAME_MAPPING_UNRESOLVED')
        sameprior=[e for e in events if e['table']==m['table'] and e['hand_id']==m['hand_id'] and e['frame_id']<m['frame_id'] and e['action']==m['action'] and e['player']==m['player'] and e['amount']==m['amount']]
        if sameprior:lineage.append('REPEATED_MONEY_SIGNATURE_POSSIBLE_DUPLICATE_OR_REAL_REPEAT')
        if any(s['present_in_previous_frame'] for s in sources):lineage.append('MONEY_ROW_ALREADY_VISIBLE_BEFORE_EMISSION')
        if len(sources)!=len(m['source_rows']):lineage.append('INCOMPLETE_SOURCE_ROW_BINDING')
        grammar=m['action'] in MONEY and m['player'] is not None
        amount=m['amount'] is not None
        if sources:
            batch_forced=m['action'] in {'ANTE','SMALL_BLIND','BIG_BLIND'} and m['frame_id']==h['start']
            bottom_append=max(s['ordinal'] for s in sources)>=maxordinal-1
            if not(batch_forced or bottom_append):lineage.append('APPEND_GEOMETRY_NOT_PROVEN')
        else:lineage.append('SOURCE_APPEARANCE_UNKNOWN')
        # This certifies a local first visible occurrence / source association,
        # not every historical tracker link. Unresolved older tracker labels
        # stay visible independently. Repeated monetary signatures never pass.
        append_candidate=not lineage
        # Exact own-frame provenance is independently verified, but a row can
        # change rasterization/reappear or be a previous-hand residue. These
        # observations do not establish a neighboring-block lifecycle chain.
        lineage.append('ROW_LIFECYCLE_CONTINUITY_NOT_PROVEN')
        rowresolved=False
        if seat and rowresolved:bucket='PLAYER_RESOLVED_ROW_RESOLVED'
        elif seat:bucket='PLAYER_RESOLVED_ROW_UNRESOLVED'
        elif rowresolved:bucket='PLAYER_UNRESOLVED'
        else:bucket='BOTH_UNRESOLVED'
        audit={'money_event_id':index+1,'table':m['table'],'hand_id':m['hand_id'],'frame':m['frame_id'],
            'event':m['action'],'actor_raw':m['player'],'amount_raw':m['amount'],'seat':seat,
            'player_id':f"{m['table']}_SEAT_{int(seat[4:])}_EPOCH_1" if seat else None,
            'player_binding_evidence':binding,'row_identity_original_unresolved':uncertain_row_identity(m),
            'bucket':bucket,'row_duplicate_risk':lineage,'row_resolved_local':rowresolved,
            'source_frame_pixel_binding_verified':len(sources)==len(m['source_rows']),
            'local_append_candidate_only':append_candidate,
            'grammar_proven':grammar,'amount_unambiguous':amount,'source_rows':sources,'source_row_ids':m['source_row_ids'],
            'source_raw_text':m['raw_text'],'policy':'fresh own-frame pixel/hash/ordinal binding; absent immediately prior; newly appended bottom or new-hand forced batch; no earlier same monetary signature. No OCR text alone or legacy ID alone establishes identity.'}
        audits.append(audit);audit_by_sources[tuple(m['source_row_ids'])]=audit
    jsonl(target/'money_row_classification.jsonl',audits)
    observations=[];groups=[];firsts=[]
    for c in selected:
        key=(c['table'],c['hand_id']);rv=reviewmap[key];msgs=[e for e in events if (e['table'],e['hand_id'])==key]
        for seat in rv['occupied_seats']:
            records=[r for r in prior if (r['table'],r['hand_id'],r['seat'])==(*key,seat)]
            new=[]
            for old in records:
                f=old['frame'];pid=f"{c['table']}_SEAT_{int(seat[4:])}_EPOCH_1"
                remapped=[];blockers=[];details=[]
                for m in msgs:
                    copy=dict(m)
                    if m['action'] in MONEY and m['frame_id']<=f:
                        binding,_=bind_actor(m,f)
                        a=audit_by_sources[tuple(m['source_row_ids'])]
                        if binding is None:details.append('PLAYER_UNRESOLVED:'+str(a['money_event_id']));blockers.append('LEDGER_UNKNOWN')
                        if a['row_duplicate_risk']:details.extend(a['row_duplicate_risk']);blockers.append('LEDGER_UNKNOWN')
                        if not a['grammar_proven']:details.append('UNKNOWN_GRAMMAR');blockers.append('LEDGER_UNKNOWN')
                        if not a['amount_unambiguous']:details.append('AMOUNT_AMBIGUITY');blockers.append('LEDGER_UNKNOWN')
                        if binding:copy['player']=f"{c['table']}_SEAT_{int(binding[4:])}_EPOCH_1"
                        if m['frame_id']==f and binding==seat:blockers.append('TIMING_UNKNOWN');details.append('SAME_FRAME_MONEY_ORDER_UNKNOWN')
                    if m['frame_id']<f and m['action']=='UNKNOWN_EVENT':blockers.append('LEDGER_UNKNOWN');details.append('UNKNOWN_CHAT_GRAMMAR_OR_CONTINUATION')
                    if m['frame_id']<f and m['action'] in {'FLOP','TURN','RIVER'} and uncertain_row_identity(m):blockers.append('LEDGER_UNKNOWN');details.append('STREET_LINEAGE_UNRESOLVED')
                    if m['frame_id']==f and m['action'] in {'FLOP','TURN','RIVER'}:blockers.append('TIMING_UNKNOWN');details.append('SAME_FRAME_STREET_ORDER_UNKNOWN')
                    if m['frame_id']<f and m['action'] in {'WIN','UNCALLED_RETURN'}:blockers.append('LEDGER_UNKNOWN');details.append('SETTLEMENT_EXCLUDED_FROM_PRIMARY_ACCEPTANCE')
                    remapped.append(copy)
                ledger=observed_ledger(remapped,pid,f)
                for issue in ledger['issues']:
                    if issue!='PAYOUT_APPLIED_FORMULA_OUT_OF_SCOPE':blockers.append('LEDGER_UNKNOWN');details.append(issue)
                if any('FIRST_STABLE_SNAPSHOT_NOT_PROVEN' in d or 'BASELINE_FIRST_STABLE' in d for d in old['blocker_details']):
                    blockers.append('LEDGER_UNKNOWN');details.append('FIRST_STABLE_SEARCH_INCOMPLETE')
                candidate=str(Decimal(old['displayed_stack'])+Decimal(ledger['gross'])-Decimal(ledger['returns'])-Decimal(ledger['payouts']))
                r=dict(old,player_id=pid,player=rv['visually_read_nicknames'].get(seat),ledger=ledger,
                    gross_committed_before_frame=ledger['gross'],returns_before_frame=ledger['returns'],payouts_before_frame=ledger['payouts'],
                    candidate_start_stack=candidate,blockers=sorted(set(blockers)),blocker_details=sorted(set(details)),
                    identity_basis='CONFIRMED_OCCUPIED_FIXED_TABLE_SEAT_ROIS; spelling is a label; occupancy epoch1 observed; continuity gaps explicit')
                new.append(r)
            if new:firsts.append(new[0])
            observations.extend(new);good=[r for r in new if not r['blockers']]
            values=[Decimal(r['candidate_start_stack']) for r in good]
            diagnostic=[r for r in new if 'SETTLEMENT_EXCLUDED_FROM_PRIMARY_ACCEPTANCE' not in r['blocker_details'] and Decimal(r['payouts_before_frame'])==0]
            dv={r['candidate_start_stack'] for r in diagnostic};ds={(r['gross_committed_before_frame'],r['returns_before_frame']) for r in diagnostic}
            groups.append({'table':c['table'],'hand_id':c['hand_id'],'seat':seat,'player_id':f"{c['table']}_SEAT_{int(seat[4:])}_EPOCH_1",'nickname_label':rv['visually_read_nicknames'].get(seat),
                'status':classify(new) if new else 'LEDGER_UNKNOWN','observations':len(new),'usable_observations':len(good),
                'commitment_states':len({(r['gross_committed_before_frame'],r['returns_before_frame'],r['payouts_before_frame']) for r in good}),
                'max_delta':str(max(values)-min(values)) if values else None,'diagnostic_states':len(ds),
                'diagnostic_exact':len(ds)>=2 and len(dv)==1,'diagnostic_conflict':len(ds)>=2 and len(dv)>1,
                'blockers':dict(collections.Counter(b for r in new for b in r['blockers']))})
    jsonl(target/'stack_observations.jsonl',observations);jsonl(target/'first_stable_snapshots.jsonl',firsts);jsonl(target/'player_hand_groups.jsonl',groups)
    counts=collections.Counter(g['status'] for g in groups)
    for k in ['MULTI_STATE_EXACT','SAME_STATE_REPEAT','CONFLICT','IDENTITY_UNKNOWN','TIMING_UNKNOWN','LEDGER_UNKNOWN','OCR_STACK_CONFLICT']:counts.setdefault(k,0)
    summary={'verdict':'NOT_PROVEN','hands':len(selected),'money_messages':len(audits),'buckets':dict(collections.Counter(a['bucket'] for a in audits)),
        'blocker_counts_nonexclusive':{'player_binding':sum(a['seat'] is None for a in audits),'row_duplicate_lineage_risk':sum(bool(a['row_duplicate_risk']) for a in audits),
          'unknown_grammar':sum(not a['grammar_proven'] for a in audits),'amount_ambiguity':sum(not a['amount_unambiguous'] for a in audits),
          'money_events_coinciding_with_stable_numeric_frame':len({a['money_event_id'] for a in audits if any(r['table']==a['table'] and r['hand_id']==a['hand_id'] and r['frame']==a['frame'] and r['seat']==a['seat'] for r in observations)})},
        'original_unresolved_lineage_labels_retained':sum(a['row_identity_original_unresolved'] for a in audits),
        'duplicate_risk_detail_counts':dict(collections.Counter(d for a in audits for d in a['row_duplicate_risk'])),
        'seat_based_numeric_snapshots':len(firsts),'usable_observations':sum(not r['blockers'] for r in observations),'group_counts':dict(counts),
        'multi_state_groups':sum(g['commitment_states']>=2 for g in groups),'diagnostic_multi_state_groups':sum(g['diagnostic_states']>=2 for g in groups),
        'diagnostic_exact':sum(g['diagnostic_exact'] for g in groups),'diagnostic_conflict':sum(g['diagnostic_conflict'] for g in groups),
        'max_delta':str(max(Decimal(g['max_delta']) for g in groups if g['max_delta'] is not None)) if any(g['max_delta'] is not None for g in groups) else None,
        'future_money_events_used':0,'numeric_triples_unchanged':True,'no_nickname_normalization_or_math_alias_merge':True,
        'occupancy_epochs':len(occupancy),'new_epochs_from_ocr_spelling':0,'cross_hand_continuity':'UNKNOWN; prior78 checks remain unresolved',
        'row_policy_limit':'Append geometry is diagnostic only. Physical row lifecycle continuity remains unproven for all331 money messages; none is counted as row-resolved.'}
    dump(target/'summary.json',summary);db.close();return summary

if __name__=='__main__':
    import argparse
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(run(a.output),ensure_ascii=False,indent=2))
