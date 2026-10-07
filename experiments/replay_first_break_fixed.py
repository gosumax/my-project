"""FIXED Smoke17 diagnostic replay and nine-sheet review data. Saved rows only."""
import argparse
import collections
import copy
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.poker_hand_assembly import assemble_hands
from parser_core.poker_diagnostics import diagnose_hand, revalidate_candidate
from parser_core.poker_logic_engine import validate_hand
from parser_core.jev_poker_resolver import JevResolver, JevHTTPClient, JevMCPClient, TypeSafeJevClient

INPUT = ROOT/'runs/temporal_overlap_boundary_smoke17_20261001/ordered_timeline.jsonl'
EXPECTED_SHA = '9f67b9b470bfbf333341e8c7155baeea5bd7a4a04fbe3c116f3fb4c5257ad77a'
BASE = ROOT/'runs/logic_math_fixed_smoke17_20261001'


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def numeric(value):
    if value is None:
        return None
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (ValueError, TypeError):
        return value


def source_rows(hand):
    return [r for m in hand['messages'] for r in m['source_rows']]


def key(hand):
    return f'{hand["table_id"]}:{hand["session_id"]}:{hand["hand_id"] or "PREFIX"}'


def stable(diagnostic):
    d = copy.deepcopy(diagnostic)
    d.pop('logic_ms', None)
    d['math'].pop('elapsed_ms', None)
    return d


def run(output, live=False, mcp=False, typesafe=False):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'summary.json').exists():
        raise ValueError('Fresh output directory required')
    began = time.perf_counter()
    manifest = json.loads((ROOT/'production_manifest.json').read_text(encoding='utf-8'))
    before_hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in manifest['hashes']}
    assert hashlib.sha256(INPUT.read_bytes()).hexdigest() == EXPECTED_SHA
    rows = [json.loads(l) for l in INPUT.read_text(encoding='utf-8').splitlines()]
    for row in rows:
        row['raw_text'] = row.get('raw_ocr_text', row.get('raw_text', row.get('text', '')))
        row['row_id'] = row.get('stable_row_id', row.get('row_id'))
        row['timestamp_ms'] = row.get('timestamp_ms', float(row.get('video_time', 0))*1000)
    hands = assemble_hands(rows)
    assert len(hands) == 196 and sum(h['hand_id'] is not None for h in hands) == 189
    # Reuse the independently reproduced cross-table contract. Single-table
    # runs must not recalibrate a three-table evidence rule on a one-table input.
    for h in hands:
        h['semantics'] = {'call': 'ADDED', 'bet': 'ADDED', 'evidence': 'FROZEN_FIXED_5_WITNESSES_4_TABLES'}
    diagnostics = [diagnose_hand(h) for h in hands]
    inventory = json.loads((BASE/'baseline/boundary_inventory.json').read_text(encoding='utf-8'))
    inventory_map = {(b['table_id'], b['hand_id']): b for b in inventory}
    old = [json.loads(l) for l in (BASE/'first_break_before_assembly_fixes.jsonl').read_text(encoding='utf-8').splitlines()]
    old_map = {key(v['hand']): v for v in old}
    repair_evidence = []
    for h, d in zip(hands, diagnostics):
        previous = old_map[key(h)]
        old_hand = previous['hand']
        new_map = {r: m for m in h['messages'] for r in m['source_row_ids']}
        for old_msg in old_hand['messages']:
            new_msg = new_map[old_msg['source_row_ids'][0]]
            if any(old_msg.get(k) != new_msg.get(k) for k in ['action', 'amount', 'all_in']):
                cause = ('MULTILINE_NOT_ASSEMBLED' if len(new_msg['source_row_ids']) > len(old_msg['source_row_ids'])
                         else 'PARSER_MISCLASSIFICATION')
                repair_evidence.append({'hand_key': key(h), 'table_id': h['table_id'], 'hand_id': h['hand_id'],
                    'PROVEN_CAUSE': cause, 'source_row_ids': new_msg['source_row_ids'],
                    'before': {k: old_msg.get(k) for k in ['raw_text', 'action', 'amount', 'all_in']},
                    'after': {k: new_msg.get(k) for k in ['raw_text', 'action', 'amount', 'all_in']},
                    'first_break_before': previous['diagnostics']['first_break'],
                    'first_break_after': d['first_break'],
                    'proves_full_hand_root_cause': False,
                    'evidence': 'Unmodified adjacent physical rows parse under explicit complete known grammar'})
    dump(output/'source_repair_evidence.json', repair_evidence)
    isolated = []
    for table in [f'T{i:02}' for i in range(1, 10)]:
        single = assemble_hands([r for r in rows if r['table_id'] == table])
        for h in single:
            h['semantics'] = copy.deepcopy(hands[0]['semantics'])
        expected = {key(h): stable(d) for h, d in zip(hands, diagnostics) if h['table_id'] == table}
        actual = {key(h): stable(diagnose_hand(h)) for h in single}
        mismatches = [k for k in set(expected)|set(actual) if expected.get(k) != actual.get(k)]
        isolated.append({'table_id': table, 'segments': len(single), 'logic_pass': sum(d['logic_status'] == 'PASS' for d in actual.values()),
                         'all_diagnostics_match': not mismatches, 'mismatches': mismatches})
    dump(output/'table_isolation.json', isolated)
    assert all(v['all_diagnostics_match'] for v in isolated)
    grouped = {table: [r for r in rows if r['table_id'] == table] for table in {r['table_id'] for r in rows}}
    context_positions = {table: {r['row_id']: i for i, r in enumerate(rs)} for table, rs in grouped.items()}
    if typesafe and mcp:
        raise ValueError('TypeSafe uses its official API; --jev-mcp is the legacy third-party transport')
    client_type = TypeSafeJevClient if typesafe else JevMCPClient if mcp else JevHTTPClient
    identity = 'FIXED_FIRST_BREAK_TYPESAFE_SYSTEMONE' if typesafe else 'FIXED_FIRST_BREAK_JEV_MCP' if mcp else 'FIXED_FIRST_BREAK_NATIVE_JEV'
    resolver = JevResolver(client_type({'api_key_env': 'JEV_API_KEY', 'timeout_seconds': 45,
                                      'prefer_windows_user_key': typesafe}),
                           cache_dir=output/'jev_cache', client_identity=identity) if live else None
    tabs = {name: [] for name in ['HAND_SUMMARY', 'FIRST_BREAKS', 'TRACE_ALL', 'MATH_FAILS',
                                  'MISSING_AMOUNTS', 'JEV_CASES', 'TABLE_STATS', 'ERROR_STATS', 'SOURCE_CONTEXT']}
    eligible = 0
    replay_records = []
    for h, d in zip(hands, diagnostics):
        common = {'table_id': h['table_id'], 'hand_id': h['hand_id'], 'hand_key': key(h)}
        b = d['first_break']
        first_msg = h['messages'][b['message_index']] if b and b['message_index'] >= 0 else None
        context = []
        if first_msg:
            table_rows = grouped[h['table_id']]
            pos = context_positions[h['table_id']][first_msg['source_row_ids'][0]]
            end_pos = context_positions[h['table_id']][first_msg['source_row_ids'][-1]]
            for i in range(max(0, pos-10), min(len(table_rows), end_pos+11)):
                row = table_rows[i]
                c = {**common, 'break_row_id': first_msg['source_row_ids'][0], 'relative_offset': i-pos,
                     'source_row_id': row['row_id'], 'frame_id': row['frame_id'], 'timestamp_ms': row['timestamp_ms'],
                     'raw_text': row['raw_text'], 'relation': 'BREAK' if row['row_id'] in first_msg['source_row_ids'] else 'BEFORE' if i < pos else 'AFTER',
                     'source_epoch': row.get('epoch'), 'session_id': row.get('session_id')}
                context.append(c)
            tabs['SOURCE_CONTEXT'].extend(context)
        boundary = inventory_map[(h['table_id'], h['hand_id'])]
        packet = {'table_id': h['table_id'], 'hand_id': h['hand_id'], 'FIRST_LOGIC_BREAK': b,
                  'physical_source_context': context, 'known_blind_level': h.get('blind_level'),
                  'observed_ledger': {k: v for k, v in d['math'].items() if k not in {'steps', 'elapsed_ms'}},
                  'boundary_context': {k: boundary[k] for k in ['start_marker', 'next_boundary', 'boundary_conflicts']}}
        resolution = None
        final = d
        is_eligible = bool(h['start_observed'] and h['end_observed'] and d['final_status'] != 'FULLY_VALID')
        eligible += is_eligible
        if is_eligible and resolver and not resolver.circuit_open:
            request_hand = copy.deepcopy(h)
            request_hand['diagnostic_packet'] = packet
            initial = validate_hand(request_hand)
            initial['issues'] = [b] if b else initial['card_issues'] if 'card_issues' in initial else []
            initial['math'] = packet['observed_ledger']
            resolution = resolver.resolve(request_hand, initial)
            if typesafe:
                print(json.dumps({'jev_progress': eligible, 'hand': key(h),
                                  'live_call': resolution.get('live_call'),
                                  'rejection': resolution.get('rejection')}, ensure_ascii=False), flush=True)
            checked = None
            if resolution.get('candidate_hand'):
                candidate = resolution.pop('candidate_hand')
                checked = revalidate_candidate(request_hand, candidate)
                resolution['candidate_view'] = candidate
                resolution['full_revalidation'] = checked
                if checked['accepted']:
                    final = checked['revalidation']
                    final['final_status'] = 'JEV_ASSISTED_CERTIFIED'
            decision = resolution.get('jev_decision') or {}
            jev_row = {**common, 'rule_trigger': b['code'] if b else None,
                'proven_cause_before': 'UNKNOWN', 'first_break_text': first_msg['raw_text'] if first_msg else 'END_OF_SEGMENT',
                'math_delta_before': numeric(d['math']['delta']), 'hypothesis_type': decision.get('decision'),
                'hypothesis': decision, 'evidence': decision.get('evidence_rows'),
                'logic_after': checked['revalidation']['logic_status'] if checked and checked.get('revalidation') else None,
                'math_delta_after': numeric(checked['revalidation']['math']['delta']) if checked and checked.get('revalidation') else None,
                'card_after': checked['revalidation']['card_status'] if checked and checked.get('revalidation') else None,
                'accepted': bool(checked and checked['accepted']),
                'reject_reason': resolution.get('rejection') or (checked['reject_reason'] if checked else 'NO_CANDIDATE'),
                'latency_ms': resolution.get('latency_ms'), 'api_error': resolution.get('api_error_detail'),
                'request_hash': resolution.get('request_hash'),
                'transport': 'TYPESAFE_SYSTEMONE' if typesafe else 'MCP_jev_decide' if mcp else 'REST_NATIVE_DECISIONS'}
            if resolution.get('live_call'):
                tabs['JEV_CASES'].append(jev_row)
        physical = source_rows(h)
        table_ids = sorted({r['table_id'] for r in physical})
        assert table_ids == [h['table_id']]
        s = {**common, 'numbered_hand': h['hand_id'] is not None,
             'start_frame': min(r['frame_id'] for r in physical), 'end_frame': max(r['frame_id'] for r in physical),
             'start_time_ms': min(r['timestamp_ms'] for r in physical), 'end_time_ms': max(r['timestamp_ms'] for r in physical),
             'source_row_count': len(physical), 'assembled_message_count': len(h['messages']),
             'complete_start': d['complete_start'], 'complete_end': d['complete_end'],
             'logic_status': d['logic_status'], 'has_first_logic_break': bool(b),
             'first_break_row_id': '; '.join(b['source_row_ids']) if b else None,
             'first_break_text': b['observed'] if b else None, 'first_break_rule_trigger': b['code'] if b else None,
             'proven_cause': d['PROVEN_CAUSE'], 'math_status': d['math_status'], 'math_delta': numeric(d['math']['delta']),
             'total_contributions': numeric(d['math']['total_contributions']), 'total_returns': numeric(d['math']['total_returns']),
             'total_payouts': numeric(d['math']['payout_total']), 'card_status': d['card_status'],
             'jev_attempted': bool(resolution and resolution.get('live_call')), 'jev_result': resolution.get('rejection') if resolution else 'CIRCUIT_OPEN_NOT_ATTEMPTED' if is_eligible and resolver and resolver.circuit_open else 'NOT_ATTEMPTED',
             'jev_hypothesis_type': (resolution.get('jev_decision') or {}).get('decision') if resolution else None,
             'jev_accepted': final is not d, 'final_status': final['final_status'],
             'unique_table_ids_inside_hand': len(table_ids), 'source_rows': [r['row_id'] for r in physical],
             'boundary_start': boundary['start_marker'], 'boundary_next': boundary['next_boundary'],
             'boundary_end': [{k: m.get(k) for k in ['action', 'amount', 'player', 'source_row_ids', 'raw_text', 'frame_id', 'timestamp_ms']}
                              for m in h['messages'] if m['action'] in {'WIN', 'HAND_END'}],
             'boundary_conflicts_current': h['boundary_conflicts'], 'unnumbered_kind': boundary['unnumbered_kind'],
             'ledger_certainty': d['math']['ledger_certainty']}
        tabs['HAND_SUMMARY'].append(s)
        if b:
            tabs['FIRST_BREAKS'].append({**common, 'row_number_in_hand': b['message_index']+1 if first_msg else None,
                'source_row_id': '; '.join(b['source_row_ids']), 'frame_id': first_msg['frame_id'] if first_msg else None,
                'timestamp_ms': first_msg['timestamp_ms'] if first_msg else None, 'raw_text': b['observed'],
                'parsed_action': first_msg['action'] if first_msg else 'END_OF_SEGMENT',
                'player': first_msg.get('player') if first_msg else None, 'amount': numeric(first_msg.get('amount')) if first_msg else None,
                'street': b['state_before']['street'], 'state_before': b['state_before'], 'state_after': b['state_after'],
                'expected': b['expected'], 'observed': b['observed'], 'rule_trigger': b['code'], 'proven_cause': 'UNKNOWN',
                'math_delta_at_break': numeric(b['math_delta_at_break']),
                'notes': 'Prefix net-minus-paid is not conservation failure; source cause UNKNOWN. '+('VIRTUAL END CHECK' if not first_msg else '')})
        for index, (msg, step) in enumerate(zip(h['messages'], d['trace'])):
            before, after = step['state_before'] or {}, step['state_after'] or {}
            math = step['observed_math']
            tabs['TRACE_ALL'].append({**common, 'seq': index+1, 'status': step['status'],
                'source_row_id': '; '.join(msg['source_row_ids']), 'assembled_message_id': f'{key(h)}:M{index+1:03}',
                'frame_id': msg['frame_id'], 'timestamp_ms': msg['timestamp_ms'], 'raw_text': msg['raw_text'],
                'normalized_text': msg['normalized_text'], 'action': msg['action'], 'player': msg.get('player'),
                'amount': numeric(msg.get('amount')), 'street': before.get('street'),
                'current_bet_before': numeric(before.get('current_bet')), 'current_bet_after': numeric(after.get('current_bet')),
                'player_street_contribution_before': numeric(before.get('player_street_contribution')),
                'player_street_contribution_after': numeric(after.get('player_street_contribution')),
                'logic_rule': b['code'] if step['is_first_break'] else None, 'logic_result': step['rule_result'],
                'math_effect': math['math_effect'], 'running_contribution_total': numeric(math['after']['total_contributions']),
                'running_return_total': numeric(math['after']['total_returns']), 'running_payout_total': numeric(math['after']['total_payouts']),
                'state_before': before or None, 'state_after': after or None, 'proven_cause': 'UNKNOWN',
                'assembly': msg['assembly'], 'raw_source_rows': msg['source_rows'],
                'math_scope': 'FULL_OBSERVED_BOOKKEEPING_INDEPENDENT_OF_LOGIC'} )
        for missing in d['math']['missing_fields']:
            msg = h['messages'][missing['message_index']]
            tabs['MISSING_AMOUNTS'].append({**common, 'source_row_id': '; '.join(msg['source_row_ids']),
                'frame_id': msg['frame_id'], 'timestamp_ms': msg['timestamp_ms'], 'raw_text': msg['raw_text'],
                'action_type': msg['action'], 'player': msg.get('player'), 'street': missing['street'],
                'expected_amount_source': 'NONE', 'candidate_amount': None, 'candidate_confidence': None, 'proven_cause': 'UNKNOWN'})
        if d['math_status'] == 'FAIL':
            suspect = d['math']['first_suspicious_action']
            msg = h['messages'][suspect['message_index']] if suspect else None
            missing = d['math']['missing_fields'][0] if d['math']['missing_fields'] else {}
            tabs['MATH_FAILS'].append({**common, 'contributions': numeric(d['math']['total_contributions']),
                'returns': numeric(d['math']['total_returns']), 'payouts': numeric(d['math']['payout_total']), 'delta': numeric(d['math']['delta']),
                'first_suspicious_row': '; '.join(msg['source_row_ids']) if msg else None,
                'first_suspicious_text': msg['raw_text'] if msg else None, 'rule_trigger': 'MATH_DELTA' if not d['math']['pass'] else 'LEDGER_NOT_VERIFIABLE',
                'proven_cause': 'UNKNOWN', 'missing_amount_type': missing.get('action'), 'missing_player': missing.get('player'),
                'missing_street': missing.get('street'), 'candidate_explanation': d['math']['localization_note'], 'candidate_amount': None,
                'jev_attempted': s['jev_attempted'], 'jev_result': s['jev_result'], 'missing_fields': d['math']['missing_fields'],
                'contributions_by_player': d['math']['players'], 'contributions_by_street': d['math']['contributions_by_street']})
        replay_records.append({'hand': h, 'diagnostics': d, 'final_diagnostics': final, 'resolution': resolution, 'packet': packet})
    for table in [f'T{i:02}' for i in range(1, 10)]:
        values = [s for s in tabs['HAND_SUMMARY'] if s['table_id'] == table]
        tabs['TABLE_STATS'].append({'table_id': table, 'segments': len(values), 'numbered': sum(v['numbered_hand'] for v in values),
            'unnumbered': sum(not v['numbered_hand'] for v in values), 'complete': sum(v['complete_start'] and v['complete_end'] for v in values),
            'logic_pass': sum(v['logic_status'] == 'PASS' for v in values), 'logic_fail': sum(v['logic_status'] == 'FAIL' for v in values),
            'math_pass': sum(v['math_status'] == 'PASS' for v in values), 'math_fail': sum(v['math_status'] == 'FAIL' for v in values),
            'fully_valid': sum(v['final_status'] == 'FULLY_VALID' for v in values), 'jev_assisted': sum(v['jev_accepted'] for v in values),
            'unresolved': sum(v['final_status'] not in {'FULLY_VALID', 'JEV_ASSISTED_CERTIFIED'} for v in values),
            'logic_not_verifiable': sum(v['logic_status'] == 'NOT_VERIFIABLE' for v in values),
            'math_not_verifiable': sum(v['math_status'] == 'NOT_VERIFIABLE' for v in values)})
    rules = collections.Counter(v['rule_trigger'] for v in tabs['FIRST_BREAKS'])
    math_rules = collections.Counter(v['rule_trigger'] for v in tabs['MATH_FAILS'])
    causes = collections.Counter(v['proven_cause'] for v in tabs['HAND_SUMMARY'] if v['final_status'] != 'FULLY_VALID')
    for domain, stats in [('FIRST_LOGIC_BREAK_RULE_TRIGGER', rules), ('MATH_RULE_TRIGGER', math_rules), ('PROVEN_CAUSE', causes)]:
        tabs['ERROR_STATS'].extend({'statistic_type': domain, 'rule_trigger': name if domain != 'PROVEN_CAUSE' else None,
                                  'proven_cause': name if domain == 'PROVEN_CAUSE' else None, 'count': count} for name, count in stats.items())
    after_hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in manifest['hashes']}
    assert before_hashes == after_hashes == manifest['hashes']
    old_times = [v['diagnostics']['logic_ms'] for v in old]
    new_times = [d['logic_ms'] for d in diagnostics]
    summary = {'input': str(INPUT), 'input_sha256': EXPECTED_SHA, 'physical_rows': len(rows), 'segments': len(hands),
        'numbered': sum(bool(h['hand_id']) for h in hands), 'unnumbered': sum(not h['hand_id'] for h in hands),
        'complete_start_end': sum(h['start_observed'] and h['end_observed'] for h in hands),
        'logic_status': dict(collections.Counter(d['logic_status'] for d in diagnostics)), 'first_break_count': sum(bool(d['first_break']) for d in diagnostics),
        'first_break_rules': dict(rules), 'proven_causes_current_problem_hands': dict(causes),
        'changed_before_message_records': len(repair_evidence),
        'source_proven_assembly_repairs': len({tuple(v['source_row_ids']) for v in repair_evidence}),
        'repair_types': dict(collections.Counter(v['PROVEN_CAUSE'] for v in {tuple(v['source_row_ids']): v for v in repair_evidence}.values())),
        'repaired_hand_count': len({v['hand_key'] for v in repair_evidence}), 'math_status': dict(collections.Counter(d['math_status'] for d in diagnostics)),
        'numeric_math_balanced': sum(d['math']['pass'] for d in diagnostics), 'missing_amount_count': len(tabs['MISSING_AMOUNTS']),
        'exact_delta_distribution': dict(collections.Counter(d['math']['delta'] for d in diagnostics)),
        'card_status': dict(collections.Counter(d['card_status'] for d in diagnostics)),
        'final_status': dict(collections.Counter(v['final_status'] for v in tabs['HAND_SUMMARY'])),
        'jev': {'eligible': eligible, 'attempted': resolver.live_calls if resolver else 0,
                'api_failures': resolver.errors if resolver else 0, 'successful_responses': len(resolver.success_latencies) if resolver else 0,
                'accepted': sum(v['accepted'] for v in tabs['JEV_CASES']), 'circuit_open': resolver.circuit_open if resolver else False,
                'success_latency_mean_ms': statistics.mean(resolver.success_latencies) if resolver and resolver.success_latencies else None,
                'preflight_32k_rejections': sum(bool(v['resolution'] and v['resolution'].get('rejection') == 'WHOLE_HAND_EXCEEDS_JEV_32K_LIMIT') for v in replay_records)},
        'table_isolation': isolated, 'production_registered_hashes_unchanged': len(after_hashes), 'production_code_changed': False,
        'performance': {'wall_seconds': time.perf_counter()-began, 'first_break_before_mean_ms': statistics.mean(old_times),
                        'first_break_after_mean_ms': statistics.mean(new_times), 'after_max_ms': max(new_times)},
        'A_G_current_problem_hands': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': sum(causes.values())},
        'verdict': 'PARTIAL_REPAIR_ONLY', 'rows_per_sheet': {n: len(v) for n, v in tabs.items()}}
    dump(output/'summary.json', summary)
    dump(output/'workbook_data.json', tabs)
    with (output/'hands.jsonl').open('w', encoding='utf-8') as f:
        for v in replay_records:
            f.write(json.dumps(v, ensure_ascii=False)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jev-live', action='store_true')
    parser.add_argument('--jev-mcp', action='store_true')
    parser.add_argument('--jev-typesafe', action='store_true')
    args = parser.parse_args()
    run(args.output, args.jev_live, args.jev_mcp, args.jev_typesafe)
