"""Source-preserving HH export with explicit incomplete evidence and proof gates."""
from collections import Counter
import json
from pathlib import Path


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


def export_hand_histories(output, records):
    output = Path(output)
    recon = {r['segment_id']: r for r in read_jsonl(output/'poker_reconstruction.jsonl')}
    stack_rows = read_jsonl(output/'stack_start_stack.jsonl')
    exported = []
    for record in records:
        hand = record['observed_hand']
        lifecycle = record['lifecycle']['status']
        result = recon.get(record['segment_id'], {})
        derived=result.get('derived_final_validation',{})
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
        for s in stacks:
            if s.get('status') != 'START_STACK_PROVEN':
                continue
            player=s.get('player')
            binding=s.get('binding_receipt') or {}
            context=dict(player=player,seat_id=s.get('seat_id'),table_id=hand['table_id'],
                         session_id=hand['session_id'],hand_id=hand['hand_id'],
                         occupancy_epoch=s.get('occupancy_epoch'))
            if (any(v is None for v in context.values()) or
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
        if conflicts:
            blockers.append('START_STACK_CONFLICT')
        if not players or set(players) != set(proven):
            blockers.append('ALL_PLAYER_SEATS_AND_START_STACKS_NOT_PROVEN')
        if len({s['seat_id'] for s in proven.values()}) != len(proven):
            blockers.append('PLAYER_SEAT_CONFLICT')
        status = ('COMPLETE' if not blockers else lifecycle if lifecycle in {'START_PARTIAL','END_PARTIAL'}
                  else 'PARTIAL')
        exported.append(dict(schema='source-proven-hand-history-v1', segment_id=record['segment_id'],
            table=hand['table_id'], session=hand['session_id'], hand_number=hand['hand_id'],
            status=status, lifecycle=lifecycle, certified=status=='COMPLETE',
            blockers=sorted(set(blockers)), players=[dict(player=p,
                seat=proven[p]['seat_id'] if p in proven else None,
                START_STACK=proven[p]['calculated_start_stack'] if p in proven else None,
                START_STACK_status='PROVEN' if p in proven else 'UNKNOWN') for p in players],
            stack_evidence=stacks, actions=(derived['candidate_hand']['messages'] if derived.get('accepted')
                                          and result.get('output_eligible') else hand['messages']),
            observed_actions=hand['messages'],validation=record['validation'],
            derived_final_validation=derived,
            diagnostics=record['diagnostics'], reconstruction=result,
            final_pot=result.get('calculated_pot') if result.get('payout_eligible') else None))
    with (output/'hand_histories.jsonl').open('w',encoding='utf-8') as stream:
        for row in exported:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    with (output/'hand_histories.txt').open('w',encoding='utf-8') as stream:
        for hand in exported:
            stream.write(f"Hand #{hand['hand_number'] or 'UNKNOWN'} | {hand['table']} | {hand['status']}\n")
            for player in hand['players']:
                stream.write(f"Seat {player['seat'] or 'UNKNOWN'}: {player['player']} (START_STACK {player['START_STACK'] if player['START_STACK'] is not None else 'UNKNOWN'})\n")
            for action in hand['actions']:
                stream.write(action_text(action)+'\n')
                if action.get('amount_proof'):
                    stream.write('  Source OCR: '+action['raw_text']+'\n')
            stream.write('Evidence blockers: '+(', '.join(hand['blockers']) or 'NONE')+'\n\n')
    summary=dict(export_enabled=True, hands=len(exported), statuses=dict(Counter(h['status'] for h in exported)),
        certified_hh=sum(h['certified'] for h in exported),
        blockers=dict(Counter(b for h in exported for b in h['blockers'])),
        physical_complete_hands=None, HH_COMPLETE_RATE=None,
        denominator_status='INDEPENDENT_SOURCE_AUDIT_REQUIRED')
    (output/'hh_export_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    return summary
