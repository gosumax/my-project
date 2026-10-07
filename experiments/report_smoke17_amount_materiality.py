"""Report-only amount-kind counterfactuals over all frozen Smoke17 observations.

No source/event deletion, boundary repair, Jev call or validator modification.
The saved 199 raw transcript partitions are retained. Existing assemble_messages
creates an explicitly DERIVED current event view *inside* each saved partition.
"""
import copy
from collections import Counter,defaultdict
from decimal import Decimal,InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parser_core.poker_hand_assembly import assemble_messages,MONEY_ACTIONS

SOURCE=ROOT/'runs/production_smoke17_20261001_013909'
RAW=ROOT/'runs/poker_logic_math_jev_smoke17_final_20261001/hands.jsonl'
CORE=ROOT/'outputs/edge_hand_attribution_window_simulation_20261001_v2'
PRIOR=ROOT/'outputs/jev_structured_evidence_smoke17_20261001_v2'
OUT=ROOT/'outputs/smoke17_amount_materiality_20261001'
STREETS=['PREFLOP','FLOP','TURN','RIVER']
CAP=1024

def read(p):return [json.loads(l) for l in p.read_text('utf-8').splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=enc),encoding='utf-8')
def ids(h):return [rid for m in h['messages'] for rid in m['source_row_ids']]
def number(value):
    try:
        n=Decimal(str(value))
        return n if n.is_finite() and n>0 else None
    except (InvalidOperation,ValueError):return None
def enc(v):
    if isinstance(v,Decimal):return str(v)
    raise TypeError(type(v).__name__)
def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,default=enc,separators=(',',':'))

def initial(hand):
    known=bool(hand.get('start_observed')) and not hand.get('boundary_conflicts')
    return dict(street='PREFLOP',own={},ledger={},refunds={},payouts={},states={},bet=Decimal(0),
        ended=False,seen_start=False,seen_money=False,epoch=None,board=[],hand_id=str(hand.get('hand_id') or ''),
        precise=known,unknown=[] if known else ['MISSING_START_OR_BOUNDARY_CONFLICT'],
        errors=[],trace=[],choices={},hash='START')

def clone(s):
    r=dict(s)
    for k in ['own','refunds','payouts','states','choices']:r[k]=dict(s[k])
    r['ledger']={p:dict(v) for p,v in s['ledger'].items()}
    for k in ['unknown','errors','trace']:r[k]=list(s[k])
    return r

def totals(s):
    gross=sum((sum(v.values(),Decimal(0)) for v in s['ledger'].values()),Decimal(0))
    returned=sum(s['refunds'].values(),Decimal(0));paid=sum(s['payouts'].values(),Decimal(0))
    return dict(known_gross_subtotal=gross,known_return_subtotal=returned,known_payout_subtotal=paid,
        nominal_net=gross-returned,nominal_remaining_pot=gross-returned-paid,
        numeric_completeness='COMPLETE_IN_OBSERVED_EVENT_VIEW' if s['precise'] else 'UNKNOWN_TOTAL_PARTIAL_SUBTOTAL_ONLY')

def caps(s):
    net={p:sum(v.values(),Decimal(0))-s['refunds'].get(p,Decimal(0)) for p,v in s['ledger'].items()}
    previous=Decimal(0);tiers=[]
    for level in sorted({v for v in net.values() if v>0}):
        funded=[p for p,v in net.items() if v>=level]
        tiers.append(dict(cap=level,amount=(level-previous)*len(funded),funders=funded,
                          eligible=[p for p in funded if s['states'].get(p)!='FOLDED']))
        previous=level
    return dict(status='DERIVED_CONTRIBUTION_TIERS_NOT_OBSERVED_SIDE_POTS',tiers=tiers,
                completeness='CONDITIONAL_UNKNOWN' if not s['precise'] else 'NUMERIC_VIEW_ONLY')

def snapshot(s):
    return dict(street=s['street'],player_street_contributions=s['own'],per_player_ledger=s['ledger'],
        refunds=s['refunds'],payouts=s['payouts'],player_states=s['states'],current_bet=s['bet'],
        ended=s['ended'],board=s['board'],precise=s['precise'],unknown=s['unknown'],
        totals=totals(s),contribution_tiers=caps(s))

def disputed(m):
    return m['action'] in {'CALL','BET'} or (m['action']=='RAISE' and not
        re.search(r'raises\s+to|повышает\s+до|рейз\s+до',m['raw_text'],re.I))

def step(s,m,index,kind='ADDED'):
    s=clone(s);action=m['action'];p=m.get('player');amount=number(m.get('amount'))
    critical=action not in {'DEAL','TABLE_INFO','BLIND_LEVEL','NO_SHOW'}
    new_unknown=[]
    epochs={r.get('epoch') for r in m['source_rows']}
    if s['epoch'] is not None and epochs!={s['epoch']}:new_unknown.append('EPOCH_DISCONTINUITY_NO_SILENT_STATE_CARRY')
    if len(epochs)==1:s['epoch']=next(iter(epochs))
    if critical and any(r.get('origin')=='UNRESOLVED_IDENTITY' for r in m['source_rows']):
        new_unknown.append('EVENT_IDENTITY_OR_MULTIPLICITY_UNKNOWN')
    if any(r.get('status') in {'ERROR','UNREADABLE'} for r in m['source_rows']):new_unknown.append('OCR_FAILURE')
    if m.get('card_binding_conflict') and action in STREETS[1:]:new_unknown.append('STREET_EVENT_BINDING_CONFLICT')
    if new_unknown:s['precise']=False;s['unknown']=sorted(set(s['unknown']+new_unknown))
    before=copy.deepcopy(snapshot(s));own=s['own'].get(p,Decimal(0))
    own_known=own if s['precise'] else None
    delta=Decimal(0);errors=[];legal_unknown=[];street_changed=False
    if action=='NEW_HAND':
        if s['seen_start'] or s['seen_money']:
            s['precise']=False;s['unknown']=sorted(set(s['unknown']+['COMPETING_OR_REPEATED_START_NO_RESET']))
        s['seen_start']=True
    elif action in {'HAND_HEADER','HAND_ID'}:
        candidates=re.findall(r'#\s*(\d{6,})',m['raw_text'])
        if any(n!=s['hand_id'] for n in candidates) and s['hand_id']:
            s['precise']=False;s['unknown']=sorted(set(s['unknown']+['COMPETING_HAND_HEADER']))
    elif action in STREETS[1:]:
        expected=STREETS[STREETS.index(s['street'])+1] if s['street'] in STREETS and s['street']!='RIVER' else None
        if action!=expected or s['ended']:
            errors.append('UNPROVEN_STREET_BOUNDARY');s['precise']=False
            s['unknown']=sorted(set(s['unknown']+['REPEATED_SKIPPED_OR_AFTER_END_BOARD_EVENT_NO_RESET']))
        else:
            # Only an explicit board/street event changes street; DEAL never does.
            s['street']=action;s['own']={n:Decimal(0) for n in s['own']};s['bet']=Decimal(0)
            s['board']=list(m.get('cards',[]));street_changed=True
    elif action in {'UNKNOWN_EVENT','UNKNOWN','UNRESOLVED'}:
        s['precise']=False;s['unknown']=sorted(set(s['unknown']+['UNPARSED_ACTION_MAY_HAVE_MONEY_EFFECT']))
    elif action in {'DEAL','TABLE_INFO','BLIND_LEVEL','NO_SHOW','SHOW','SHOWDOWN'}:pass
    elif action=='HAND_END':s['ended']=True
    elif action in MONEY_ACTIONS and (amount is None or not p):
        s['precise']=False;s['unknown']=sorted(set(s['unknown']+['MISSING_OBSERVED_AMOUNT_OR_ACTOR']))
    else:
        if p:s['states'].setdefault(p,'ACTIVE')
        if action in {'CALL','BET','RAISE','CHECK','FOLD'}:
            if s['ended']:errors.append('ACTION_AFTER_HAND_END')
            if s['states'].get(p)=='FOLDED':errors.append('ACTION_AFTER_FOLD')
            if s['states'].get(p)=='ALL_IN':errors.append('ACTION_AFTER_ALLIN')
        if action in {'ANTE','SMALL_BLIND','BIG_BLIND'}:
            delta=amount;bucket=action
            if s['seen_money'] or s['street']!='PREFLOP':errors.append('STALE_OR_REPEATED_FORCED_EVENT')
            if bucket in s['ledger'].get(p,{}):errors.append('REPEATED_FORCED_EVENT')
            s['ledger'].setdefault(p,{})[bucket]=s['ledger'].get(p,{}).get(bucket,Decimal(0))+delta
            if action!='ANTE':
                s['own'][p]=own+delta;s['bet']=max(s['bet'],s['own'][p])
                if m.get('all_in'):
                    legal_unknown.append('NOMINAL_BLIND_TARGET_UNKNOWN_FOR_SHORT_ALLIN')
                    s['precise']=False;s['unknown']=sorted(set(s['unknown']+['NOMINAL_BLIND_TARGET_UNKNOWN_FOR_SHORT_ALLIN']))
        elif action in {'CALL','BET','RAISE'}:
            s['seen_money']=True
            total_kind=kind=='TOTAL' if disputed(m) else action=='RAISE'
            delta=amount-own if total_kind else amount
            if total_kind and not s['precise']:legal_unknown.append('TOTAL_INCREMENT_REQUIRES_UNKNOWN_PRIOR_CONTRIBUTION')
            if action=='CALL':
                required=s['bet']-own
                if delta<=0 or required<=0 or delta>required:errors.append('CALL_AMOUNT_EXCEEDS_TARGET_OR_NONPOSITIVE')
                elif delta<required and not m.get('all_in'):legal_unknown.append('SHORT_CALL_REQUIRES_UNOBSERVED_ALLIN_OR_STACK')
            elif action=='BET':
                if s['bet']!=0 or delta<=0:errors.append('BET_ON_NONEMPTY_STREET_OR_NONPOSITIVE')
            elif action=='RAISE':
                if own+delta<=s['bet'] or delta<=0:errors.append('RAISE_TARGET_NOT_ABOVE_CURRENT_BET')
            s['own'][p]=own+delta
            if action in {'BET','RAISE'}:s['bet']=s['own'][p]
            bucket=s['street'];s['ledger'].setdefault(p,{})[bucket]=s['ledger'].get(p,{}).get(bucket,Decimal(0))+delta
            if disputed(m):s['choices'][index]=kind
        elif action=='UNCALLED_RETURN':
            if amount>own:errors.append('RETURN_EXCEEDS_STREET_CONTRIBUTION')
            others=max((v for n,v in s['own'].items() if n!=p),default=Decimal(0))
            if amount!=own-others:errors.append('RETURN_NOT_EQUAL_UNMATCHED_CONTRIBUTION')
            s['refunds'][p]=s['refunds'].get(p,Decimal(0))+amount;s['own'][p]=own-amount
            s['bet']=max(s['own'].values(),default=Decimal(0))
        elif action=='WIN':
            if s['states'].get(p)=='FOLDED':errors.append('PAYOUT_TO_FOLDED_ACTOR')
            s['payouts'][p]=s['payouts'].get(p,Decimal(0))+amount;s['ended']=True
        elif action=='FOLD':s['states'][p]='FOLDED'
        elif action=='CHECK':
            if s['bet']!=own:errors.append('CHECK_FACING_BET')
        if m.get('all_in') and action in MONEY_ACTIONS:s['states'][p]='ALL_IN'
    s['errors']+= [dict(message_index=index,code=e,source_row_ids=m['source_row_ids']) for e in errors]
    after=copy.deepcopy(snapshot(s))
    missing_debit=action in MONEY_ACTIONS and (amount is None or not p)
    total_with_unknown_prior=action in {'CALL','BET','RAISE'} and (
        kind=='TOTAL' if disputed(m) else action=='RAISE') and own_known is None
    entry=dict(message_index=index,source_row_ids=m['source_row_ids'],action=action,player=p,kind_hypothesis=kind if disputed(m) else None,
        amount_reading=m.get('amount'),before=before,after=after,
        derived_increment='UNKNOWN' if missing_debit or total_with_unknown_prior else str(delta),
        prior_contribution=str(own_known) if own_known is not None else 'UNKNOWN',
        errors=errors,legality_unknown=legal_unknown,street_changed=street_changed,
        observed_pot='UNKNOWN',observed_stack='UNKNOWN')
    s['trace'].append(entry)
    # Entire numeric/state history is part of equivalence: later cancellation
    # cannot disguise different earlier pots, debits or next-action legality.
    invariant=dict(before=before,after=after,increment=entry['derived_increment'],errors=errors,legality_unknown=legal_unknown)
    s['hash']=hashlib.sha256((s['hash']+canonical(invariant)).encode()).hexdigest()
    return s

def run_arm(hand,meaning):
    state=initial(hand)
    for i,m in enumerate(hand['messages']):state=step(state,m,i,meaning)
    return state

def branch_view(hand):
    states=[initial(hand)];limited=False;max_states=1;action_cases=defaultdict(list);last_index=-1
    for i,m in enumerate(hand['messages']):
        expanded=[]
        for s in states:
            alternatives=['ADDED','TOTAL'] if disputed(m) else ['ADDED']
            results=[step(s,m,i,k) for k in alternatives]
            if disputed(m):
                a,t=results;ae,te=a['trace'][-1],t['trace'][-1]
                action_cases[i].append(dict(prefix_hash=s['hash'],ADDED=ae,TOTAL=te,
                    material=None if ae['prior_contribution']=='UNKNOWN' or te['prior_contribution']=='UNKNOWN' else a['hash']!=t['hash']))
            expanded+=results
        merged={}
        for s in expanded:
            if s['hash'] not in merged:merged[s['hash']]=s
        max_states=max(max_states,len(merged));last_index=i
        if len(merged)>CAP:limited=True;states=list(merged.values())[:CAP];break
        states=list(merged.values())
    return states,dict(action_cases),dict(incomplete=limited,budget=CAP,max_states=max_states,last_processed_message=last_index)

def analyze(hand,view_id,source_kind):
    added=run_arm(hand,'ADDED');total=run_arm(hand,'TOTAL');states,cases,search=branch_view(hand)
    amount_indices=[i for i,m in enumerate(hand['messages']) if disputed(m)]
    complete_numeric=bool(states) and all(s['precise'] for s in states) and not search['incomplete']
    local_equal=bool(amount_indices) and all(i in cases and all(c['material'] is False for c in cases[i]) for i in amount_indices)
    viable=[]
    if complete_numeric and hand.get('end_observed') and not hand.get('boundary_conflicts'):
        for s in states:
            if not s['errors'] and not any(t['legality_unknown'] for t in s['trace']) and totals(s)['nominal_remaining_pot']==0:
                viable.append(s)
    if complete_numeric and local_equal and len(states)==1:
        classification='A';reason='AMOUNT_KIND_UNRESOLVED_BUT_NON_MATERIAL: all per-action alternatives have identical whole-history numeric/state effects.'
    elif amount_indices and complete_numeric and len(states)>1 and len(viable)==1 and not any(
            t['legality_unknown'] for s in states for t in s['trace']):
        classification='B';reason='One complete source-bound branch satisfies local constraints, returns and observed closed payout; alternatives have explicit contradictions.'
    else:
        classification='C';reason=('NO_AMOUNT_KIND_DISPUTE: other source/math/sizing uncertainty remains; not solved by kind choice.' if not amount_indices else
            'Unknown prior/contribution/identity/street, inconsistent input, multiple feasible branches, or bounded search prevents proof.')
    result=dict(view_id=view_id,source_kind=source_kind,table_id=hand['table_id'],session_id=hand['session_id'],
        hand_number_candidate=hand.get('hand_id'),source_row_ids=ids(hand),
        start_observed=hand.get('start_observed'),end_observed=hand.get('end_observed'),
        boundary_conflicts=hand.get('boundary_conflicts'),class_=classification,reason=reason,
        amount_action_indices=amount_indices,branch_search=search,distinct_history_signatures=len(states),
        complete_numeric=complete_numeric,viable_branches=len(viable),
        equivalent_global_final_totals=totals(added)==totals(total),
        equivalent_global_full_histories=added['hash']==total['hash'],
        global_ADDED=dict(snapshot=snapshot(added),trace=added['trace'],constraints=added['errors']),
        global_TOTAL=dict(snapshot=snapshot(total),trace=total['trace'],constraints=total['errors']),
        per_action_all_prefix_cases=cases,unknown=sorted(set(added['unknown']+total['unknown'])),
        observed_pot='UNKNOWN',observed_stacks='UNKNOWN',
        interpretation_note='All values are DERIVED counterfactual bookkeeping. When incomplete, nominal numbers are partial/conditional, not observed pot/stack facts.',
        classification_does_not_discharge_other_blockers=True)
    return result

def display_effect(result,index,meaning):
    entry=result['global_'+meaning]['trace'][index];t=entry['after']['totals']
    own=entry['after']['player_street_contributions'].get(entry['player'])
    prefix='' if entry['after']['precise'] else 'CONDITIONAL; actual UNKNOWN; '
    return f'{prefix}increment={entry["derived_increment"]}; own-after={own}; net={t["nominal_net"]}; remaining-pot={t["nominal_remaining_pot"]}'

def full_transcript(spec,result,original):
    h=spec['hand'];label=f'{result["view_id"]}: {h["table_id"]} / {h.get("hand_id") or "UNKNOWN"}'
    out=[f'## {label}','',f'Class **{result["class_"]}**. {result["reason"]}',
        f'Source view `{result["source_kind"]}`; session `{h["session_id"]}`; start={h.get("start_observed")}, end={h.get("end_observed")}; conflicts={h.get("boundary_conflicts")}.',
        'No rows trimmed before start or after WIN. If start/end is absent, it remains absent. Saved partition is an observation scope, not a certified complete hand.','',
        '### Полный исходный OCR — все physical source rows','', '```text']
    for rid in ids(h):
        row=original[rid]
        out.append(f'{rid} | frame={row["frame_id"]} time={row.get("video_time")} epoch={row.get("epoch")} OCR-confidence={row.get("confidence")} origin={row.get("origin")}')
        out.append(row.get('raw_ocr_text',''))
    out+=['```','','### Полный DERIVED event transcript — без исключённых сообщений','','```text']
    for i,m in enumerate(h['messages']):
        marker='PREFLOP' if i==0 else m['action'] if m['action'] in STREETS[1:] else ''
        if marker:out.append(marker+' (board/event candidate; see boundary constraints)')
        out.append(f'{i:03d} {m.get("player") or "Dealer"} {m["action"]} {m.get("amount") or ""} '+
            (str(m['cards']) if m.get('cards') else '')+' | '+','.join(m['source_row_ids']))
        out.append('OCR joined: '+m['raw_text'])
    out+=['```','','### Forced bets и contributions по улицам — DERIVED','',
        'ANTE/SB/BB are separate ledger buckets. PREFLOP below is voluntary debit; full preflop contribution = SB + BB + PREFLOP, with ANTE separately added to pot. Missing buckets show only zero in this conditional ledger, never proof of absent source actions.',
        '| ARM | PLAYER | ANTE | SB | BB | PREFLOP | FLOP | TURN | RIVER | RETURN | PAYOUT | PRECISION |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for meaning in ['ADDED','TOTAL']:
        s=result['global_'+meaning]['snapshot']
        for p,ledger in s['per_player_ledger'].items():
            vals=[str(ledger.get(k,0)) for k in ['ANTE','SMALL_BLIND','BIG_BLIND']+STREETS]
            out.append('|'+meaning+'|'+p+'|'+'|'.join(vals)+f'|{s["refunds"].get(p,0)}|{s["payouts"].get(p,0)}|'+
                       ('DERIVED' if s['precise'] else 'UNKNOWN totals; partial/conditional numbers')+'|')
        forced={k:sum((ledger.get(k,Decimal(0)) for ledger in s['per_player_ledger'].values()),Decimal(0))
                for k in ['ANTE','SMALL_BLIND','BIG_BLIND']}
        out.append('Forced subtotal '+meaning+' (DERIVED; completeness='+str(s['precise'])+'): '+canonical(forced))
    out+=['','### Каждое спорное действие — pot before, обе гипотезы и последствия','']
    for i in result['amount_action_indices']:
        m=h['messages'][i];out+=[f'#### Event {i}: {m.get("player")} {m["action"]} {m.get("amount")}','',
            'Source: '+', '.join(m['source_row_ids']),
            'Observed pot/stack: UNKNOWN. Complete per-prefix branch cases are in analysis.json; the table below shows the two uniform global arms.','',
            '| ARM | prior contribution | DERIVED pot before | increment/result | next legality / return / payout / side-pot |',
            '|---|---|---|---|---|']
        for meaning in ['ADDED','TOTAL']:
            trace=result['global_'+meaning]['trace'];e=trace[i];before=e['before']['totals']
            pot=str(before['nominal_remaining_pot']) if e['before']['precise'] else f'UNKNOWN; conditional subtotal={before["nominal_remaining_pot"]}'
            following=next((t for t in trace[i+1:] if t['action'] in
                           {'CALL','BET','RAISE','CHECK','FOLD','UNCALLED_RETURN','WIN'}),None)
            effects=dict(current_errors=e['errors'],unknown=e['legality_unknown'],
                next_action_errors=following['errors'] if following else None,
                final_refunds=result['global_'+meaning]['snapshot']['refunds'],
                observed_payouts=result['global_'+meaning]['snapshot']['payouts'],
                stack_absolute='UNKNOWN',stack_debit=e['derived_increment'],
                final_side_pot_candidates=result['global_'+meaning]['snapshot']['contribution_tiers'])
            out.append(f'|{meaning}|{e["prior_contribution"]}|{pot}|{display_effect(result,i,meaning)}|'+canonical(effects).replace('|','\\|')+'|')
    return '\n'.join(out)+'\n'

def main():
    if OUT.exists():raise ValueError('Fresh report directory required')
    frozen=json.loads((PRIOR/'input_hashes_before.json').read_text('utf-8'))
    protected={p:sha(ROOT/p) for p in frozen}
    if protected!=frozen:raise ValueError('Frozen source drift')
    protected[str(RAW.relative_to(ROOT))]=sha(RAW)
    production=json.loads((ROOT/'production_manifest.json').read_text('utf-8'))['hashes']
    if any(sha(ROOT/p)!=value for p,value in production.items()):raise ValueError('Production drift')
    original=read(SOURCE/'ordered_timeline.jsonl');byid={r['stable_row_id']:r for r in original}
    raw=read(RAW);core=json.loads((CORE/'CORE_FIXED_WINDOWS.assembled_hands.json').read_text('utf-8'))
    core_records=read(CORE/'CORE_FIXED_WINDOWS.hands.jsonl');seen=[];specs=[]
    for i,receipt in enumerate(raw):
        saved=receipt['observed_hand'];rows=[r for m in saved['messages'] for r in m['source_rows']]
        for row in rows:
            rid=row['stable_row_id'];seen.append(rid)
            if any(row.get(k)!=v for k,v in byid[rid].items()):raise ValueError('Raw source field mutation')
        hand={k:copy.deepcopy(v) for k,v in saved.items() if k not in {'messages','semantics'}}
        hand['messages']=assemble_messages(rows)
        if ids(hand)!=ids(saved):raise ValueError('Message view source loss/order mutation')
        specs.append(dict(view_id=f'RAW_{i+1:03d}',source_kind='FULL_SOURCE_PARTITION_CURRENT_DERIVED_EVENT_VIEW',
            hand=hand,saved_hand=saved,historical_issues=receipt['initial']['issues']))
    if len(seen)!=8146 or len(set(seen))!=8146 or set(seen)!=set(byid):raise ValueError('Incomplete source coverage')
    for table in {r['table_id'] for r in original}:
        if [rid for rid in seen if byid[rid]['table_id']==table]!=[r['stable_row_id'] for r in original if r['table_id']==table]:
            raise ValueError('Per-table source order mutation')
    for i,hand in enumerate(core):specs.append(dict(view_id=f'CORE_{i+1:03d}',source_kind='CURRENT_CORE_FIXED_WINDOW_PROJECTION',
        hand=hand,record=core_records[i],saved_hand=hand))
    OUT.mkdir();results=[];tables=[];groups={k:[] for k in 'ABC'};coregroups={k:[] for k in 'ABC'}
    appendix=['# Полные OCR / event transcripts всех Smoke17 observation scopes','',
        'RAW_001..RAW_199 cover all8146 rows once. CORE_001..CORE_029 are separately labeled current projections of976 rows, not additional observations.',
        'Existing assemble_messages supplies only an explicit DERIVED report view inside each frozen raw partition. No hand boundaries, source fields, identities or event multiplicity repaired. Historical semantics/resolution not used.','']
    for spec in specs:
        result=analyze(spec['hand'],spec['view_id'],spec['source_kind'])
        exact=[f'CORE_{i+1:03d}' for i,h in enumerate(core) if ids(h)==result['source_row_ids']]
        result['exact_core_source_matches']=exact
        result['overlapping_core_candidates']=[i for i,h in enumerate(core) if set(ids(h)) & set(result['source_row_ids'])]
        results.append(result)
        entry=dict(view_id=result['view_id'],hand=result['hand_number_candidate'],table=result['table_id'],reason=result['reason'])
        (groups if result['view_id'].startswith('RAW') else coregroups)[result['class_']].append(entry)
        h=spec['hand']
        indices=result['amount_action_indices'] or [None]
        for i in indices:
            m=h['messages'][i] if i is not None else None
            tables.append(dict(HAND=f'{h.get("hand_id") or "UNKNOWN"} [{result["view_id"]}]',TABLE=h['table_id'],
                PLAYER_ACTION=f'{m.get("player")} {m["action"]} {m.get("amount")}' if m else 'NO CALL/BET KIND DISPUTE; inspect other uncertainty',
                ISSUE='AMOUNT_KIND_UNRESOLVED' if m else 'UNRESOLVED_SOURCE_OR_MATH_OR_SIZING_CONTEXT',
                ADDED_RESULT=display_effect(result,i,'ADDED') if m else canonical(result['global_ADDED']['snapshot']['totals']),
                TOTAL_RESULT=display_effect(result,i,'TOTAL') if m else canonical(result['global_TOTAL']['snapshot']['totals']),
                MATERIAL=('UNKNOWN' if not result['complete_numeric'] or result['branch_search']['incomplete'] else
                    'YES' if any(c['material'] for c in result['per_action_all_prefix_cases'].get(i,[])) else 'NO'),
                CAN_PROVE='EQUAL_EFFECTS' if result['class_']=='A' else 'UNIQUE_COMPLETE_BRANCH' if result['class_']=='B' else 'NO',
                FINAL_CLASS=result['class_'],source_row_ids=m['source_row_ids'] if m else result['source_row_ids']))
        appendix.append(full_transcript(spec,result,byid))
        print(json.dumps(dict(view=result['view_id'],class_=result['class_'],branches=result['distinct_history_signatures'],
                              incomplete=result['branch_search']['incomplete']),ensure_ascii=False),flush=True)
    write(OUT/'analysis.json',results);write(OUT/'table.json',tables)
    write(OUT/'observed_and_derived_views.json',specs)
    write(OUT/'groups.json',dict(full_source=groups,CORE_projection=coregroups))
    (OUT/'FULL_TRANSCRIPTS.md').write_text('\n'.join(appendix),encoding='utf-8')
    table=['# Итоговая таблица amount materiality','', '| HAND | TABLE | PLAYER/ACTION | ISSUE | ADDED RESULT | TOTAL RESULT | MATERIAL? | CAN PROVE? | FINAL CLASS |',
           '|---|---|---|---|---|---|---|---|---|']
    for row in tables:table.append('|'+ '|'.join(str(row[k]).replace('|','\\|') for k in
        ['HAND','TABLE','PLAYER_ACTION','ISSUE','ADDED_RESULT','TOTAL_RESULT','MATERIAL','CAN_PROVE','FINAL_CLASS'])+'|')
    (OUT/'TABLE.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    for category in 'ABC':
        lines=[f'# Category {category}','', '## Full source — raw partitions','']
        for item in groups[category]:lines.append(f'- {item["view_id"]}: {item["table"]} / {item["hand"] or "UNKNOWN"} — {item["reason"]}')
        lines+=['','## Current CORE projection — separate views, not new hands','']
        for item in coregroups[category]:lines.append(f'- {item["view_id"]}: {item["table"]} / {item["hand"] or "UNKNOWN"} — {item["reason"]}')
        (OUT/f'{category}_HANDS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    after={p:sha(ROOT/p) for p in protected}
    if after!=protected:raise ValueError('Source hash mutation')
    write(OUT/'input_hashes_before.json',protected);write(OUT/'input_hashes_after.json',after)
    summary=dict(raw_observation_partitions=199,raw_source_rows=8146,raw_current_event_messages=sum(len(s['hand']['messages']) for s in specs[:199]),
        core_views=29,core_admitted_source_rows=976,raw_group_counts={k:len(v) for k,v in groups.items()},
        core_group_counts={k:len(v) for k,v in coregroups.items()},table_rows=len(tables),
        search_incomplete=[r['view_id'] for r in results if r['branch_search']['incomplete']],
        source_hashes_unchanged=len(protected),production_hashes_unchanged=len(production),
        Jev_calls=0,validator_calls=0,production_code_changed=False,thresholds_changed=False,
        source_rewrites=0,event_dedup=0,boundary_changes=0,automatic_fixes=0,
        categories_are_amount_dimension_not_hand_certification=True,
        inventory_is_deliberately_unfiltered='All199 raw partitions exported, including73 old scopes without CALL/BET; no potentially related uncertainty omitted.')
    write(OUT/'summary.json',summary)
    report=['# Smoke17 — amount-kind materiality, report only','',
        'No implementation or certification change. Jev calls0; validator calls0; source/production/thresholds unchanged.','',
        '## Coverage / OBSERVED vs DERIVED vs UNKNOWN','',
        'All199 saved raw observation partitions cover8146 original rows exactly once. Their old event grouping and completeness flags are retained as source metadata; current unchanged assemble_messages makes a separately identified DERIVED event view inside each partition. The old global DERIVED ADDED contract and old resolver decisions are discarded as authority. Current CORE29 projections are separate views of976 existing source rows; their intervals stay frozen. Never add the raw and CORE counts as distinct hands.',
        'OBSERVED: exact raw OCR, source IDs/frame/time/epoch/confidence, parser input reader receipts. DERIVED: parsed action/amount candidates, street contributions, nominal pot, refunds, contribution tiers, branch constraints. UNKNOWN: observed stacks/pot, absent actions/amounts, uncertain multiplicity/chronology. Partial nominal sums are never reported as observed/complete pots.','',
        '## Method / A B C','',
        'A NON_MATERIAL_AMBIGUITY: every per-action ADDED/TOTAL alternative under every explored prefix gives identical complete numeric and state history. This preserves other logic/card/payout blockers; it is not Fully valid.',
        'B MATERIAL_BUT_RESOLVABLE: a complete identity-bound observed chronology yields exactly one full branch satisfying local constraints, observed returns and closed payout. Delta0 alone on dirty data is not proof. No guessed missing all-in, amount, stack or side pot.',
        'C TRUE_UNRESOLVED: material alternatives, unknown inputs/identity/street, mutually contradictory candidates, or incomplete bounded search. Scopes without any amount-kind dispute are retained here as source/math controls, explicitly marked NO_AMOUNT_KIND_DISPUTE; this does not manufacture an ADDED/TOTAL ambiguity.',
        'Only explicit FLOP/TURN/RIVER board/event context changes street; DEAL does not. Repeated/skipped/conflicting boards, starts and epoch changes do not silently reset state. Explicit RAISE TO is a known total target, not an ADDED/TOTAL candidate; its debit still depends on known prior contribution. Nominal blind and actual short all-in post are not conflated.','',
        f'Branch budget {CAP} distinct complete-history signatures per view. Mathematical equivalence merging applies to hypotheses only, never source events. Full trace signatures prevent final-pot cancellation from being mislabeled A. A/B denied after a capped or numerically unknown search.','',
        '## Counts','',canonical(summary),'',
        '## Full reports','',
        '- FULL_TRANSCRIPTS.md: every raw OCR row and every derived event, unabridged, with forced bets/per-player street ledger and both action hypotheses.',
        '- TABLE.md / table.json: requested comparison columns.',
        '- A_HANDS.md / B_HANDS.md / C_HANDS.md and groups.json: full lists, separating raw and CORE views.',
        '- analysis.json: complete per-action/per-prefix counterfactuals, pot before actions, stack debit implications, return/payout/tier effects and next-action constraints.',
        '- tests.json / verification.json: bounded independent verification.','',
        '## Stop','', 'Report only. No blocker removed, no new Fully valid count computed, no automatic fix or implementation.']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(canonical(summary),flush=True)

if __name__=='__main__':main()
