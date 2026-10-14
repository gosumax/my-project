"""Bounded START_STACK-only replay: retained crops/receipts, no recognition/search.

The completed replay fixes hands, actors, all other providers and ledgers. Only
the existing Direct provider is replayed; newly closed UNKNOWN entries are a
potential HH gate gain, never a newly exported/certified hand history.
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from parser_core.adjacent_stack_carry import SavedCarrySources
from parser_core.constant_stack_balance import SourceSealBatch, frozen_runtime_files
from parser_core.direct_start_stack import (DirectStartStackProvider, FreshDirectWindows,
    native_key, join_receipt, file_hash)
from parser_core.chat_nick_actor import prove_actor_point
from parser_core.native_nick_who import NativeNickWhoSource
from parser_core.poker_reconstruction.models import Event, Player, RuleError
from parser_core.poker_reconstruction.state_machine import State
from parser_core.poker_reconstruction.accounting import audit as audit_math

TARGETS=[(2,'2878366007','temazm',171250,625,171875),
 (2,'2878367229','tolbin3',398697,3125,401822),
 (7,'2878365900','bolshoi60',519264,625,519889),
 (7,'2878366487','avtobusvip01',200800,625,201425),
 (7,'2878366745','ubivashka88888',285564,625,286189),
 (8,'2878366523','franticma',125832,3125,128957)]


def jsonl(path):
    with path.open(encoding='utf8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path,value,*,lines=False):
    with path.open('x',encoding='utf8') as stream:
        if lines:
            for row in value:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        else:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def event_sequence(result):
    """Use only the exact saved chosen events and their saved interpretations."""
    if (result.get('solution_count')!=1 or result.get('exhausted') or
            not result.get('clean_events') or result.get('dropped_events') or
            result.get('chosen_identity_hypothesis',{}).get('aliases')):
        raise ValueError('SAVED_UNIQUE_COMPLETE_LEDGER_UNAVAILABLE')
    kept=[t for t in result.get('diagnostic_trace',[]) if t.get('decision')=='KEEP']
    raw=result['clean_events']
    if len(kept)!=len(raw):raise ValueError('SAVED_EVENT_TRACE_JOIN_UNKNOWN')
    events=[]
    for source,trace in zip(raw,kept):
        selected=trace['candidate_event']
        if any(selected.get(k)!=source.get(k) for k in ('kind','player','amount')):
            raise ValueError('SAVED_EVENT_TRACE_JOIN_CONFLICT')
        row=copy.deepcopy(source);row['semantics']=selected['semantics']
        events.append(Event(**row))
    return events


def math_recheck(result,start_stacks):
    """Existing deterministic state/math validator; no solver or new actions."""
    try:events=event_sequence(result)
    except ValueError as exc:return dict(status='NOT_VERIFIABLE',reason=str(exc))
    names={e.player for e in events if e.player}
    state=State(players={n:Player(start_stack=start_stacks.get(n)) for n in names})
    for index,event in enumerate(events):
        try:state.apply(event)
        except RuleError as exc:
            return dict(status='CONFLICT',reason=str(exc),event_index=index,
                player=event.player,event=event.evidence())
    checked=audit_math(state)
    if checked['pot_delta']!=0 or checked['payout_eligible'] is not True:
        return dict(status='CONFLICT',reason='EXISTING_ACCOUNTING_OR_PAYOUT_FAILURE',math=checked)
    return dict(status='PASS',math=checked)


def main():
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('--source-output',type=Path,required=True)
    args.add_argument('--output',type=Path,required=True)
    opts=args.parse_args();started=time.perf_counter()
    source_output=opts.source_output.resolve();out=opts.output.resolve();out.mkdir(exist_ok=False)
    replay=json.loads((source_output/'replay_receipt.json').read_text('utf8'))
    source=Path(replay['source'])
    # Seal every original completed replay artifact, including all saved ledgers.
    completed_hashes={p.name:file_hash(p) for p in source_output.iterdir() if p.is_file()}
    records=jsonl(source_output/'logical_hand_analysis.jsonl')
    by_key={native_key(r['observed_hand']):r for r in records}
    hh=[h for h in jsonl(source_output/'hand_histories.jsonl') if h['lifecycle']=='COMPLETE']
    complete_keys={(int(h['table'].removeprefix('T')),h['session'],h['hand_number']) for h in hh}
    old_players={(int(h['table'].removeprefix('T')),h['session'],h['hand_number'],p['player']):p
        for h in hh for p in h['players']}
    old_proven=sum(p['START_STACK_status']=='PROVEN' for p in old_players.values())
    old_unknown=sum(p['START_STACK_status']=='UNKNOWN' for p in old_players.values())
    old_proofs=[p for p in jsonl(source_output/'direct_start_stack_proofs.jsonl') if native_key(p) in complete_keys]
    old_direct={(native_key(p),p['actor']):p for p in old_proofs}
    rows=[r for r in json.loads((source_output/'direct_start_stack_audit.json').read_text('utf8'))['rows']
          if tuple(r['hand_key']) in complete_keys]
    original_direct_candidates={(tuple(r['hand_key']),r['actor']) for r in rows}
    # Include COMPLETE current WHO/seat UNKNOWN stacks absent from the provider
    # scheduler. Partial lifecycles never enter any measurement below.
    for key,player in old_players.items():
        context,actor=key[:3],key[3]
        if (player['START_STACK_status']=='UNKNOWN' and player['ACTOR_status']=='PROVEN'
                and player.get('seat') is not None and (context,actor) not in original_direct_candidates):
            rows.append(dict(hand_key=list(context),actor=actor))
    keys={tuple(r['hand_key']) for r in rows}
    print('LOAD_SAVED_SOURCE_INVENTORIES',flush=True)
    registry=SavedCarrySources(source,records,[])
    native_who=NativeNickWhoSource(source,[r['observed_hand'] for r in records])
    # Same production class, fresh native files; scheduler is fixed by old audit.
    inventory=FreshDirectWindows(source,sorted(keys),actor_policy=True,native_who=native_who)
    provider=object.__new__(DirectStartStackProvider)
    provider.registry=registry;provider.inventory=inventory;provider.single_first_numeric=True
    provider.runtime_files=frozen_runtime_files();provider.cases={};provider.proofs={};provider.audit=[]
    candidates={(tuple(r['hand_key']),r['actor']) for r in rows}
    target_keys=[]
    for table,number,actor,display,forced,total in TARGETS:
        matches=[(k,a) for k,a in candidates if k[0]==table and k[2]==number and a==actor]
        assert len(matches)==1;target_keys.append(matches[0])
    ordered=target_keys+sorted(candidates-set(target_keys))
    batch=SourceSealBatch(registry,inventory,provider.runtime_files)
    assert batch.begin(),'PRE_SOURCE_SEAL_CHANGED'
    measured=[];targets=[];old_direct_parity=[]
    for index,(key,actor) in enumerate(ordered):
        point=prove_actor_point(inventory.contexts[key],actor,native_who=native_who)
        if point['status']!='PROVEN':
            result=dict(status='UNKNOWN',reason='ACTOR_POINT_'+str(point['reason']))
        else:
            case=inventory.case_for(key,actor,point['seat'])
            result=provider._prove(case,by_key[key],batch=batch)
            if result['status']=='PROVEN':
                provider.cases[(key,actor)]=(case,by_key[key]);provider.proofs[(key,actor)]=result
        row=dict(table_id=key[0],session_id=key[1],hand_id=key[2],actor=actor,
            seat=point.get('seat'),status=result['status'],reason=result.get('reason'),
            displayed_stack=result.get('displayed_stack'),
            forced_amount_added_back=result.get('forced_amount_added_back'),
            calculated_start_stack=result.get('calculated_start_stack'),
            numeric_frames=result.get('numeric_frames'),
            contract=result.get('first_visible_proof',{}).get('contract'))
        measured.append(row)
        if index<6:
            expected=TARGETS[index]
            assert (row['status'],row['displayed_stack'],row['forced_amount_added_back'],row['calculated_start_stack'])==(
                'PROVEN',expected[3],expected[4],expected[5]),row
            targets.append(row);print('TARGET_PASS '+json.dumps(row),flush=True)
            if index==5:write(out/'target_cases.json',targets);print('TARGETED_6_OF_6_PASS',flush=True)
        if (key,actor) in old_direct:
            previous=old_direct[(key,actor)]
            old_direct_parity.append(dict(table_id=key[0],hand_id=key[2],actor=actor,
                old_amount=previous['calculated_start_stack'],new_amount=row['calculated_start_stack'],
                proof_exact_equal=result==previous,amount_equal=row['calculated_start_stack']==previous['calculated_start_stack']))
        if index%25==0:print(f'PROGRESS {index+1}/{len(ordered)}',flush=True)
    assert batch.finish(),'POST_SOURCE_SEAL_CHANGED'
    # Reproduce the identical consumer predicate in one fresh all-source bracket.
    consumer=SourceSealBatch(registry,inventory,provider.runtime_files);assert consumer.begin()
    print('CONSUMER_REPROOF_BEGIN',flush=True)
    for index,(key,(case,record)) in enumerate(provider.cases.items()):
        fresh=provider._prove(case,record,batch=consumer)
        assert fresh==provider.proofs[key],'CONSUMER_PROOF_CHANGED'
        assert join_receipt(fresh)==join_receipt(provider.proofs[key])
        if index%25==0:print(f'CONSUMER_REPROOF {index+1}/{len(provider.cases)}',flush=True)
    assert consumer.finish(),'CONSUMER_POST_SOURCE_SEAL_CHANGED'
    assert all(r['amount_equal'] and r['proof_exact_equal'] for r in old_direct_parity)
    assert len(old_direct_parity)==len(old_proofs)
    gains={(*key,actor):p for (key,actor),p in provider.proofs.items()
        if (*key,actor) in old_players and old_players[(*key,actor)]['START_STACK_status']=='UNKNOWN'}
    final_players=[];old_proven_parity=[]
    for key,old in old_players.items():
        new=copy.deepcopy(old)
        if key in gains:
            new.update(START_STACK_status='PROVEN',START_STACK=gains[key]['calculated_start_stack'])
        if old['START_STACK_status']=='PROVEN':old_proven_parity.append(dict(
            table_id=key[0],session_id=key[1],hand_id=key[2],actor=key[3],
            amount_equal=old['START_STACK']==new['START_STACK'],player_payload_equal=old==new))
        final_players.append(dict(table_id=key[0],session_id=key[1],hand_id=key[2],
            old=old,new=new,gained=key in gains))
    assert len(old_proven_parity)==old_proven and all(r['player_payload_equal'] for r in old_proven_parity)
    recon={ (int(r['table'].removeprefix('T')),r['session_id'],r['hand_id']):r
        for r in jsonl(source_output/'poker_reconstruction.jsonl')}
    math_rows=[];unblocked=[];stack_gate_closed=[]
    final={(*rkey[:3],rkey[3]):r for rkey,r in old_players.items()}
    for row in final_players:final[(row['table_id'],row['session_id'],row['hand_id'],row['new']['player'])]=row['new']
    for h in hh:
        key=(int(h['table'].removeprefix('T')),h['session'],h['hand_number'])
        added=[a for (*k,a) in gains if tuple(k)==key]
        if not added:continue
        old_stacks={p['player']:p['START_STACK'] for p in h['players'] if p['START_STACK_status']=='PROVEN'}
        new_stacks={**old_stacks,**{a:gains[(*key,a)]['calculated_start_stack'] for a in added}}
        result=recon.get(key)
        before=math_recheck(result,old_stacks) if result else dict(status='NOT_VERIFIABLE',reason='NO_SAVED_RECONSTRUCTION')
        after=math_recheck(result,new_stacks) if result else copy.deepcopy(before)
        mr=dict(table=h['table'],hand_id=h['hand_number'],session=h['session'],added_players=added,
            before=before,after=after,new_conflict=before['status']=='PASS' and after['status']=='CONFLICT',
            scope='FIXED_SAVED_UNIQUE_EVENTS_AND_INTERPRETATIONS_EXISTING_STATE_AND_ACCOUNTING; NO_SOLVER')
        math_rows.append(mr)
        closed=all(final[(*key,p['player'])].get('seat') is not None and
            final[(*key,p['player'])]['ACTOR_status']=='PROVEN' and
            final[(*key,p['player'])]['START_STACK_status']=='PROVEN' for p in h['players'])
        if closed:
            item=dict(table=h['table'],hand_id=h['hand_number'],added_players=added,
                remaining_saved_HH_blockers=[b for b in h['blockers'] if b!='ALL_PLAYER_SEATS_AND_START_STACKS_NOT_PROVEN'],
                math_after_status=after['status'])
            stack_gate_closed.append(item)
            if h['lifecycle']=='COMPLETE' and not item['remaining_saved_HH_blockers'] and after['status']=='PASS':unblocked.append(item)
    unchanged=all(file_hash(source_output/n)==v for n,v in completed_hashes.items())
    assert unchanged,'COMPLETED_REPLAY_ARTIFACT_CHANGED'
    summary=dict(scope='START_STACK_ONLY_SAVED_DIRECT_PROVIDER_REPLAY; OTHER_PROVIDERS_FIXED; NO_CARRY_PROPAGATION',
        full_production_replay=False,video_decoded=False,new_ocr=False,new_capture=False,reconstruction_search=False,
        lifecycle_scope='COMPLETE_ONLY; START_PARTIAL_AND_END_PARTIAL_EXCLUDED_FROM_ALL_COUNTS',
        total_hands=len(hh),total_players=len(old_players),old_START_STACK_proven=old_proven,old_START_STACK_unknown=old_unknown,
        new_START_STACK_proven=len(gains),total_START_STACK_proven=old_proven+len(gains),
        remaining_START_STACK_unknown=old_unknown-len(gains),old_proven_parity=len(old_proven_parity),
        old_direct_exact_proof_parity=len(old_direct_parity),consumer_exact_reproof=len(provider.proofs),
        target_pass=6,target_total=6,provider_counts=dict(Counter(r['status'] for r in measured)),
        original_direct_candidates=len(original_direct_candidates),scheduled_candidates=len(measured),
        additional_saved_WHO_proven_unknown_candidates=len(measured)-len(original_direct_candidates),
        unknown_reasons=dict(Counter(r['reason'] for r in measured if r['status']=='UNKNOWN')),
        stack_gate_closed=len(stack_gate_closed),potential_READY_gain=len(unblocked),potential_READY_hands=unblocked,
        math_new_conflicts=[r for r in math_rows if r['new_conflict']],
        math_after_counts=dict(Counter(r['after']['status'] for r in math_rows)),
        math_scope='ONLY_FIXED_SAVED_UNIQUE_LEDGER; UNRESOLVED_LEDGERS_REMAIN_NOT_VERIFIABLE',
        actual_HH_READY='NOT_MEASURED_NO_RECONSTRUCTION_OR_HH_EXPORT',
        source_window='LEGACY_INITIAL_64_THEN_FRESH_CURRENT_HAND_ON_NO_NUMERIC_SOURCE',
        source_output_unchanged=unchanged,completed_replay_hashes=completed_hashes,
        source_hashes=inventory.hashes,wall_seconds=time.perf_counter()-started)
    write(out/'provider_results.jsonl',measured,lines=True)
    write(out/'direct_start_stack_proofs.jsonl',list(provider.proofs.values()),lines=True)
    write(out/'new_start_stack_proofs.jsonl',list(gains.values()),lines=True)
    write(out/'player_comparison.jsonl',final_players,lines=True)
    write(out/'old_proven_parity.jsonl',old_proven_parity,lines=True)
    write(out/'old_direct_proof_parity.jsonl',old_direct_parity,lines=True)
    write(out/'math_validation.jsonl',math_rows,lines=True)
    write(out/'stack_gate_closed_hands.jsonl',stack_gate_closed,lines=True)
    write(out/'summary.json',summary)
    print('SUMMARY '+json.dumps({k:v for k,v in summary.items() if k not in {'completed_replay_hashes','source_hashes','potential_READY_hands','math_new_conflicts'}},ensure_ascii=True),flush=True)


if __name__=='__main__':main()
