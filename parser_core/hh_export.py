"""Source-preserving HH export with explicit incomplete evidence and proof gates."""
from collections import Counter
import copy
import json
from .performance_work import write_json_line, scope as work_scope
from pathlib import Path
from parser_core.poker_evidence_axes import evidence_axes


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def action_text(action):
    if not action.get('amount_proof'):
        return action['raw_text']
    verbs=dict(ANTE='posts Ante',SMALL_BLIND='posts Small Blind',BIG_BLIND='posts Big Blind',
               CALL='calls',BET='bets',RAISE='raises to',UNCALLED_RETURN='gets uncalled bet',WIN='wins')
    verb=verbs[action['action']]
    suffix=' back' if action['action']=='UNCALLED_RETURN' else ' and is All-in' if action.get('all_in') else ''
    return f"Dealer: {action['player']} {verb} {action['amount']}{suffix}"


def export_hand_histories(output, records, *, carry_provider=None, constant_provider=None, direct_provider=None,
                          sitout_provider=None, actor_provider=None, joint_provider=None, simple_provider=None):
    output = Path(output)
    no_sb_source=getattr(simple_provider,'no_sb_source',None)
    recon = {r['segment_id']: r for r in read_jsonl(output/'poker_reconstruction.jsonl')}
    stack_rows = read_jsonl(output/'stack_start_stack.jsonl')
    from .joint_start_stack import verify_retained,proof_id,join_receipt
    joint_proofs={proof_id(p):p for p in read_jsonl(output/'joint_start_stack_proofs.jsonl')}
    exported = []
    for record in records:
        hand = record['observed_hand']
        original_hand=hand
        if simple_provider is not None:
            hand=simple_provider.actor_sources.canonical_hand(hand)
        lifecycle = record['lifecycle']['status']
        result = recon.get(record['segment_id'], {})
        derived=result.get('derived_final_validation',{})
        if not derived.get('accepted') and result.get('call_context_validation'):
            record = {**record, **result['call_context_validation']}
        players = sorted({m['player'] for m in hand['messages'] if m.get('player')})
        stacks = [s for s in stack_rows if s.get('segment_id') == record['segment_id']]
        blockers = list(result.get('output_blockers', []))
        if lifecycle != 'COMPLETE':
            blockers.append(lifecycle)
        if not record['fully_validated'] and not derived.get('accepted'):
            blockers.append('FINAL_VALIDATION_UNRESOLVED')
        if not result.get('output_eligible'):
            blockers.append('RECONSTRUCTION_NOT_ELIGIBLE')
        # A label or arithmetical candidate cannot certify a seat or bankroll.
        proven = {}
        conflicts = set()
        carry_verified=[]
        if carry_provider is not None and any(s.get('carry_binding_receipt') for s in stacks):
            from .adjacent_stack_carry import join_receipts as carry_receipts
            carry_verified=[r for p in carry_provider.for_hand(hand) for r in carry_receipts(p)]
        constant_verified=[]
        if constant_provider is not None and any(s.get('constant_binding_receipt') for s in stacks):
            from .constant_stack_balance import join_receipt as constant_receipt
            constant_verified=[constant_receipt(p) for p in constant_provider.for_hand(hand)]
        direct_verified=[]
        simple_verified=simple_provider.for_hand(hand) if simple_provider is not None else []
        if direct_provider is not None and any(s.get('direct_binding_receipt') for s in stacks):
            from .direct_start_stack import join_receipt as direct_receipt
            direct_verified=[direct_receipt(p) for p in direct_provider.for_hand(hand)]
        joint_verified=[]
        if joint_provider is not None and any(s.get('joint_binding_receipt') for s in stacks):
            joint_verified=[join_receipt(p) for p in joint_provider.for_hand(hand)]
        identities={};identity_scopes={};identity_scope_conflicts=set()
        if actor_provider is not None:
            scope_method=getattr(actor_provider,'for_hand_identity_uses',None)
            identity_uses=(scope_method(hand) if scope_method is not None else
                           [(proof,None) for proof in actor_provider.for_hand(hand)])
            for proof,scope in identity_uses:
                if proof['actor'] in players:
                    if scope is not None:
                        identity_scopes[proof['actor']]=scope
                        if scope.get('status')!='PROVEN':
                            identity_scope_conflicts.add(proof['actor'])
                            continue
                    identities[proof['actor']]=dict(seat_id=proof['seat'],identity_proof=proof)
        from .hh_text import SIMPLE_SITOUT_CONTRACT, simple_sitout_state_valid
        from .sitout_hand_evidence import HH_CONTRACT
        sitout_verified=[];sitouts={}
        if sitout_provider is not None and any(s.get('sitout_binding_receipt') for s in stacks):
            from .sitout_hand_evidence import join_receipt as sitout_receipt,HH_CONTRACT
            sitout_verified=[sitout_receipt(p) for p in sitout_provider.for_hand(hand)]
        for s in stacks:
            receipt=s.get('sitout_binding_receipt')
            if (receipt and receipt in sitout_verified and receipt['actor']==s.get('player') and
                    receipt['seat']==s.get('seat_id') and s.get('status')=='START_STACK_UNKNOWN' and
                    all(s.get(k)==hand.get(k) for k in ('table_id','session_id','hand_id')) and
                    s.get('player_status')=='SIT_OUT' and s.get('stack_requirement')=='NOT_REQUIRED_SIT_OUT' and
                    all(s.get(k) is None for k in ('calculated_start_stack','observed_stack','occupancy_epoch',
                                                  'source_observation_id','source_state_id'))):
                prior=sitouts.get(s['player'])
                if prior and prior!=s:conflicts.add(s['player'])
                sitouts[s['player']]=s
        # The current policy already owns status/WHO classification. Consume
        # that saved state directly, without asking the legacy provider to
        # manufacture an additional exemption receipt or a numeric stack.
        seat_rows_method=getattr(simple_provider,'seat_rows',None)
        for state in seat_rows_method(hand) if seat_rows_method is not None else []:
            player=state.get('player');identity=identities.get(player)
            if state.get('forced_allin_stack_conflict') and identity and state.get('seat')==identity['seat_id']:
                from .simple_saved_policy import prove_forced_allin_start_stack
                claim=state.get('forced_allin_stack_claim') or {}
                fresh=prove_forced_allin_start_stack(hand,player,identity['seat_id'],claim.get('source'),
                                                    state.get('identity_proof'))
                if fresh==claim and fresh and state.get('START_STACK')!=fresh['calculated_start_stack']:
                    conflicts.add(player)
            if (state.get('sitout_numeric_conflict') or state.get('sitout_state_conflicts')) and identity:
                from .simple_saved_policy import valid_sitout_status
                if (state.get('table_id')==int(hand['table_id'].removeprefix('T')) and
                        state.get('session_id')==hand['session_id'] and state.get('hand_id')==hand['hand_id'] and
                        state.get('seat')==identity['seat_id'] and
                        any(valid_sitout_status(source,table_id=state['table_id'],session_id=hand['session_id'],
                            seat=identity['seat_id'],hand_id=hand['hand_id'],hand_start=state.get('hand_start'),
                            hand_end=state.get('hand_end')) for source in state.get('sitout_sources',[]))):
                    conflicts.add(player)
            if not identity or not simple_sitout_state_valid(state, table=hand['table_id'],
                    session=hand['session_id'], hand=hand['hand_id'], player=player, seat=identity['seat_id']):
                continue
            for s in stacks:
                if (s.get('player')==player and s.get('seat_id')==identity['seat_id'] and
                        all(s.get(k)==hand.get(k) for k in ('table_id','session_id','hand_id')) and
                        s.get('status')=='START_STACK_UNKNOWN' and s.get('player_status')=='SIT_OUT' and
                        all(s.get(k) is None for k in ('calculated_start_stack','DISPLAYED_STACK','observed_stack'))):
                    current=dict(s,stack_requirement='NOT_REQUIRED_SIT_OUT',
                                 simple_sitout_state=copy.deepcopy(state))
                    prior=sitouts.get(player)
                    if prior and prior['seat_id']!=current['seat_id']:conflicts.add(player)
                    sitouts[player]=current
        for s in stacks:
            if s.get('status') != 'START_STACK_PROVEN':
                continue
            player=s.get('player')
            binding=s.get('binding_receipt') or {}
            joint=s.get('joint_binding_receipt')
            joint_valid=False
            carry=s.get('carry_binding_receipt')
            carry_valid=bool(carry and carry in carry_verified and carry['actor']==player and
                carry['seat']==s.get('seat_id') and carry['calculated_start_stack']==s.get('calculated_start_stack') and
                s.get('occupancy_epoch') is None and s.get('source_observation_id') is None and s.get('source_state_id') is None)
            constant=s.get('constant_binding_receipt')
            constant_valid=bool(constant and constant in constant_verified and constant['actor']==player and
                constant['seat']==s.get('seat_id') and constant['calculated_start_stack']==s.get('calculated_start_stack') and
                s.get('occupancy_epoch') is None and s.get('source_observation_id') in constant['source_observation_ids'] and
                s.get('source_state_id') in constant['source_state_ids'])
            direct=s.get('direct_binding_receipt')
            simple=s.get('simple_binding_receipt')
            simple_valid=bool(simple and simple in simple_verified and simple['actor']==player and
                simple['seat']==s.get('seat_id') and simple['calculated_start_stack']==s.get('calculated_start_stack') and
                all(s.get(k)==hand.get(k) for k in ('table_id','session_id','hand_id')))
            direct_valid=bool(direct and direct in direct_verified and direct['actor']==player and
                direct['seat']==s.get('seat_id') and direct['calculated_start_stack']==s.get('calculated_start_stack') and
                s.get('occupancy_epoch') is None and s.get('source_observation_id') is None and s.get('source_state_id') is None)
            if joint:
                p=joint_proofs.get(joint.get('proof_id'))
                joint_valid=bool(p and (joint in joint_verified or verify_retained(p,hand)) and join_receipt(p)==joint and
                    p['actor']==player and p['seat']==s.get('seat_id') and
                    p['calculated_start_stack']==s.get('calculated_start_stack') and
                    s.get('source_observation_id') in joint['source_observation_ids'] and
                    s.get('source_state_id') in joint['source_state_ids'])
            context=dict(player=player,seat_id=s.get('seat_id'),table_id=hand['table_id'],
                         session_id=hand['session_id'],hand_id=hand['hand_id'],
                         occupancy_epoch=s.get('occupancy_epoch'))
            if not simple_valid and not joint_valid and not carry_valid and not constant_valid and not direct_valid and (any(v is None for v in context.values()) or
                    any(binding.get(k)!=v for k,v in context.items()) or
                    s.get('source_observation_id') is None or s.get('source_state_id') is None or
                    s.get('calculated_start_stack') is None):
                continue
            prior=proven.get(player)
            if prior and any(prior.get(k)!=s.get(k) for k in
                             ('seat_id','occupancy_epoch','calculated_start_stack')):
                conflicts.add(player)
            proven[player]=s
        for player in conflicts:
            proven.pop(player,None)
            sitouts.pop(player,None)
        if set(proven)&set(sitouts):
            conflicts.update(set(proven)&set(sitouts))
            for player in conflicts:
                proven.pop(player,None);sitouts.pop(player,None)
        if conflicts:
            blockers.append('START_STACK_CONFLICT')
        if identity_scope_conflicts:
            blockers.append('ACTOR_IDENTITY_INTERVAL_UNKNOWN')
            conflicts.update(identity_scope_conflicts)
            for player in identity_scope_conflicts:
                proven.pop(player,None);sitouts.pop(player,None)
        proven_seats={**proven,**sitouts}
        for player, identity in identities.items():
            if player in proven_seats and identity['seat_id']!=proven_seats[player]['seat_id']:
                conflicts.add(player)
        for player in conflicts:
            identities.pop(player,None);proven_seats.pop(player,None)
            proven.pop(player,None);sitouts.pop(player,None)
        all_seats={**identities,**proven_seats}
        if conflicts:blockers.append('ACTOR_IDENTITY_CONFLICT')
        duplicated={s['seat_id'] for s in all_seats.values()
                    if sum(other['seat_id']==s['seat_id'] for other in all_seats.values())>1}
        if duplicated:
            blockers.append('PLAYER_SEAT_CONFLICT')
            for player in [p for p,s in all_seats.items() if s['seat_id'] in duplicated]:
                conflicts.add(player)
                all_seats.pop(player,None);identities.pop(player,None)
                proven_seats.pop(player,None);proven.pop(player,None);sitouts.pop(player,None)
        hh_contract=(SIMPLE_SITOUT_CONTRACT if any(s.get('simple_sitout_state') for s in sitouts.values())
                     else HH_CONTRACT if sitouts else None)
        if not players or set(players) != set(proven_seats):
            blockers.append('ALL_PLAYER_SEATS_AND_START_STACKS_NOT_PROVEN')
        if len({s['seat_id'] for s in all_seats.values()}) != len(all_seats):
            blockers.append('PLAYER_SEAT_CONFLICT')
        no_sb_authority=None
        if not any(m['action']=='SMALL_BLIND' for m in hand['messages']):
            from .legitimate_no_sb import prove_no_sb
            no_sb_authority=prove_no_sb(derived.get('candidate_hand') or hand,no_sb_source)
            if no_sb_authority is None:
                blockers.append('LEGITIMATE_NO_SB_AUTHORITY_UNPROVEN')
        status = ('COMPLETE' if not blockers else lifecycle if lifecycle in {'START_PARTIAL','END_PARTIAL'}
                  else 'PARTIAL')
        exported.append(dict(schema='source-proven-hand-history-v1', segment_id=record['segment_id'],
            table=hand['table_id'], session=hand['session_id'], hand_number=hand['hand_id'],
            status=status, lifecycle=lifecycle, certified=status=='COMPLETE',
            **({'no_sb_authority':no_sb_authority} if no_sb_authority is not None else {}),
            **({'hh_contract':hh_contract} if hh_contract else {}),
            evidence_axes=evidence_axes(record, result, proven_stacks=proven,proven_seats=all_seats,
                                        certified=status=='COMPLETE',hh_contract=hh_contract),
            blockers=sorted(set(blockers)), players=[dict(player=p,
                seat=all_seats[p]['seat_id'] if p in all_seats else None,
                ACTOR_status='PROVEN' if p in all_seats else 'UNKNOWN',
                **(dict(actor_identity_proof=identities[p]['identity_proof']) if p in identities else {}),
                **(dict(actor_identity_use_scope=identity_scopes[p]) if p in identity_scopes else {}),
                START_STACK=proven[p]['calculated_start_stack'] if p in proven else None,
                DISPLAYED_STACK=next((s.get('DISPLAYED_STACK') for s in stacks if s.get('player')==p),None),
                START_STACK_status='PROVEN' if p in proven else 'UNKNOWN',
                **(dict(player_status='SIT_OUT',stack_requirement='NOT_REQUIRED_SIT_OUT',
                        **({'simple_sitout_state':sitouts[p]['simple_sitout_state']}
                           if sitouts[p].get('simple_sitout_state') else
                           {'sitout_binding_receipt':sitouts[p]['sitout_binding_receipt']})) if p in sitouts else {})) for p in players],
            stack_evidence=stacks, actions=(derived['candidate_hand']['messages'] if derived.get('accepted')
                                          and result.get('output_eligible') else hand['messages']),
            observed_actions=original_hand['messages'],validation=record['validation'],
            derived_final_validation=derived,
            diagnostics=record['diagnostics'],
            diagnostics_ledger_view=record.get('diagnostics_ledger_view', 'OBSERVED'), reconstruction=result,
            final_pot=result.get('calculated_pot') if result.get('payout_eligible') else None))
    with (output/'poker_evidence_axes.jsonl').open('w', encoding='utf-8') as stream:
        for row in exported:
            stream.write(json.dumps(dict(segment_id=row['segment_id'], hand_id=row['hand_number'],
                table=row['table'], **row['evidence_axes']), ensure_ascii=False)+'\n')
    with (output/'poker_money_ledger.jsonl').open('w', encoding='utf-8') as stream:
        for row in exported:
            ledgers = [(row['diagnostics_ledger_view'], row['diagnostics'].get('math', {}))]
            derived = row['derived_final_validation']
            if derived.get('accepted'):
                ledgers.append(('VALIDATED_DERIVED', derived['diagnostics']['math']))
            for view, ledger in ledgers:
                for step in ledger.get('steps', []):
                    step = dict(step)
                    known = row['reconstruction'].get('joined_start_stacks', {}).get(step.get('player'))
                    if type(known) is int:
                        from decimal import Decimal
                        for when in ('before', 'after'):
                            snapshot = step[when]
                            net = Decimal(snapshot['contributions_by_player'].get(step['player'], '0')) - Decimal(
                                snapshot['returns_by_player'].get(step['player'], '0'))
                            step['stack_'+when] = str(Decimal(known)-net)
                        step['stack_source'] = 'SOURCE_BOUND_JOINED_START_STACK'
                    stream.write(json.dumps(dict(segment_id=row['segment_id'], table=row['table'],
                        ledger_view=view, **step), ensure_ascii=False)+'\n')
    with (output/'hand_histories.jsonl').open('w',encoding='utf-8') as stream:
        for row in exported:
            write_json_line(stream,row,allow_nan=False)
    with (output/'hand_histories_audit.txt').open('w',encoding='utf-8') as stream:
        for hand in exported:
            stream.write(f"Hand #{hand['hand_number'] or 'UNKNOWN'} | {hand['table']} | {hand['status']}\n")
            for player in hand['players']:
                stream.write(f"Seat {player['seat'] or 'UNKNOWN'}: {player['player']} (START_STACK {player['START_STACK'] if player['START_STACK'] is not None else 'UNKNOWN'})\n")
            for action in hand['actions']:
                stream.write(action_text(action)+'\n')
                if action.get('amount_proof'):
                    stream.write('  Source OCR: '+action['raw_text']+'\n')
            stream.write('Evidence blockers: '+(', '.join(hand['blockers']) or 'NONE')+'\n\n')
    from .hh_text import render_complete
    ready_count=0;text_blockers=[]
    with (output/'hand_histories.txt').open('w',encoding='utf-8') as stream:
        for hand in exported:
            if hand['status']!='COMPLETE':continue
            try:
                rendered=render_complete(hand,no_sb_source=no_sb_source)
            except ValueError as error:
                text_blockers.append(dict(segment_id=hand['segment_id'],reason=str(error)))
                continue
            with work_scope('serialization/hh_text_write',category='io/write',
                    key=(str(output/'hand_histories.txt'),hand['segment_id']),characters=len(rendered),hands=1):
                stream.write(rendered)
            ready_count+=1
    summary=dict(export_enabled=True, hands=len(exported), statuses=dict(Counter(h['status'] for h in exported)),
        certified_hh=sum(h['certified'] for h in exported),
        blockers=dict(Counter(b for h in exported for b in h['blockers'])),
        physical_complete_hands=None, HH_COMPLETE_RATE=None,
        denominator_status='INDEPENDENT_SOURCE_AUDIT_REQUIRED',
        ready_hh_text_count=ready_count,ready_hh_text_blockers=text_blockers,
        ready_hh_text='hand_histories.txt',diagnostic_text='hand_histories_audit.txt')
    (output/'hh_export_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    return summary
