"""Saved JSONL only. Experimental logic/math/Jev replay; never decodes video."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.poker_hand_assembly import assemble_hands, VERSION as ASSEMBLY_VERSION
from parser_core.poker_logic_engine import validate_hand, resolve_hand, VERSION as LOGIC_VERSION
from parser_core.poker_math_engine import VERSION as MATH_VERSION
from parser_core.jev_poker_resolver import JevResolver, JevHTTPClient, VERSION as JEV_VERSION
from parser_core.poker_hand_assembly import MONEY_ACTIONS
from parser_core.edge_gate import classify_edges, rows_for_downstream, VERSION as EDGE_VERSION


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timing(values):
    if not values:
        return {'count': 0, 'mean_ms': None, 'p50_ms': None, 'p95_ms': None, 'max_ms': None, 'sum_ms': 0}
    values = sorted(values)
    return {'count': len(values), 'mean_ms': statistics.mean(values), 'p50_ms': values[(len(values)-1)//2],
            'p95_ms': values[max(0, int(.95*len(values)+.999999)-1)], 'max_ms': max(values), 'sum_ms': sum(values)}


def registered_hashes():
    manifest = json.loads((ROOT/'production_manifest.json').read_text(encoding='utf-8'))
    return {name: digest(ROOT/name) if (ROOT/name).exists() else None for name in manifest['hashes']}


def attach_blind_level(hand):
    levels = []
    for message in hand['messages']:
        for match in re.finditer(r'(?:blind(?: level)?|блайнды)\s*[:=]?\s*(\d+(?:[ \u00a0]\d{3})*)\s*/\s*(\d+(?:[ \u00a0]\d{3})*)', message['raw_text'], re.I):
            pair = [v.replace(' ', '').replace('\u00a0', '') for v in match.groups()]
            levels.append({'small_blind': pair[0], 'big_blind': pair[1], 'source_row_ids': message['source_row_ids'], 'source_type': 'OBSERVED'})
    if len(levels) == 1:
        hand['blind_level'] = levels[0]


def semantic_evidence(hands):
    """Report both hypotheses. Never silently promote arithmetic on dirty hands."""
    evidence = []
    for hand in hands:
        if not hand['start_observed'] or not hand['end_observed']:
            continue
        item = {'table_id': hand['table_id'], 'hand_id': hand['hand_id'], 'source_row_ids':
                [r for m in hand['messages'] for r in m['source_row_ids']], 'arms': {}}
        for call in ['ADDED', 'TOTAL']:
            view = dict(hand, semantics={'call': call, 'bet': 'ADDED'})
            checked = validate_hand(view)
            item['arms'][call] = {'delta': checked['math']['delta'], 'logic_pass': checked['logic_pass'],
                                    'issues': [i['code'] for i in checked['issues']]}
        evidence.append(item)
    clean = []
    for e in evidence:
        h = next(h for h in hands if h['table_id'] == e['table_id'] and h['hand_id'] == e['hand_id']
                 and [r for m in h['messages'] for r in m['source_row_ids']] == e['source_row_ids'])
        money = [m for m in h['messages'] if m['action'] in MONEY_ACTIONS]
        signatures = [(m['action'], m['player'], m['amount']) for m in money]
        money_errors = {'INVALID_CALL', 'INVALID_RAISE', 'MISSING_AMOUNT', 'RETURN_EXCEEDS_CONTRIBUTION',
                        'INVALID_UNCALLED_RETURN', 'BET_INTO_EXISTING_BET'}
        if (e['arms']['ADDED']['delta'] == '0' and e['arms']['TOTAL']['delta'] != '0' and
                len(signatures) == len(set(signatures)) and all(m['amount'] is not None for m in money) and
                not money_errors.intersection(e['arms']['ADDED']['issues'])):
            clean.append(e)
    # Financial semantics evidence is separate from hand certification: header,
    # card and row-identity anomalies stay blockers in the actual replay.
    status = 'UNPROVEN' if len(clean) < 3 or len({e['table_id'] for e in clean}) < 3 else 'ADDED_SUPPORTED'
    return {'status': status, 'monetary_clean_supporting_hands': len(clean), 'comparisons': evidence,
            'supporting_hands': [{'table_id': e['table_id'], 'hand_id': e['hand_id'],
                                 'source_row_ids': e['source_row_ids'], 'added_delta': e['arms']['ADDED']['delta'],
                                 'total_delta': e['arms']['TOTAL']['delta'],
                                 'remaining_hand_issues': e['arms']['ADDED']['issues']} for e in clean],
            'automatic_production_promotion': False,
            'bet_status': 'ADDED_ON_UNOPENED_STREET: street contribution starts at zero; total and added coincide'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jev-live', action='store_true')
    parser.add_argument('--key-stdin', action='store_true')
    parser.add_argument('--workers', type=int, choices=[1, 2, 4], default=4)
    parser.add_argument('--cache-dir', type=Path)
    args = parser.parse_args()
    if args.key_stdin:
        os.environ['JEV_API_KEY'] = sys.stdin.readline().strip()
    began = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'summary.json').exists():
        raise SystemExit('Use a fresh output directory to preserve prior replay evidence')
    before = registered_hashes()
    input_sha = digest(args.input)
    rows = [json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines() if line.strip()]
    for row in rows:
        row['raw_text'] = row.get('raw_ocr_text', row.get('raw_text', row.get('text', '')))
        row['row_id'] = row.get('stable_row_id', row.get('row_id'))
        if row['row_id'] is None:
            raise ValueError('Every physical row needs a source identity')
        row['timestamp_ms'] = row.get('timestamp_ms', float(row.get('video_time', 0))*1000)
    edge_result = classify_edges(rows)
    write_json(args.output/'edge_report.json', {'version': EDGE_VERSION,
        'input_rows': len(rows), 'sessions': edge_result.sessions})
    with (args.output/'edge_evidence.jsonl').open('w', encoding='utf-8') as stream:
        for row in edge_result.rows:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    input_row_count = len(rows)
    rows = rows_for_downstream(edge_result.rows)
    hands = assemble_hands(rows)
    for hand in hands:
        attach_blind_level(hand)
    semantics = semantic_evidence(hands)
    if semantics['status'] == 'ADDED_SUPPORTED':
        contract = {'call': 'ADDED', 'bet': 'ADDED', 'source_type': 'DERIVED', 'input_sha256': input_sha,
                    'evidence_file': str((args.output/'semantics_evidence.json').resolve()),
                    'rule': 'CALL proved by at least 3 monetary-clean complete observed transcripts on 3 tables; BET allowed only on unopened street'}
        for hand in hands:
            hand['semantics'] = dict(contract)
    initial = [validate_hand(hand) for hand in hands]
    for index, (hand, result) in enumerate(zip(hands, initial)):
        if any(i['code'] in {'HAND_BOUNDARY_CONFLICT', 'OLD_WIN'} for i in result['issues']):
            neighbors = {}
            if index > 0 and hands[index-1]['table_id'] == hand['table_id']:
                neighbors['previous_tail'] = [{k: m.get(k) for k in ['raw_text', 'action', 'player', 'amount', 'source_row_ids']}
                                              for m in hands[index-1]['messages'][-3:]]
            if index+1 < len(hands) and hands[index+1]['table_id'] == hand['table_id']:
                neighbors['next_head'] = [{k: m.get(k) for k in ['raw_text', 'action', 'player', 'amount', 'source_row_ids']}
                                          for m in hands[index+1]['messages'][:3]]
            hand['boundary_context'] = neighbors
    client = JevHTTPClient({'api_key_env': 'JEV_API_KEY', 'timeout_seconds': 30}) if args.jev_live else None
    cache_dir = args.cache_dir or args.output/'jev_cache'
    resolver = JevResolver(client, cache_dir, 'https://www.jevai.org/api/v1/decisions|typesafe-ai/jev')
    sent = [i for i, result in enumerate(initial) if result['complete_candidate'] and result['status'] != 'FULLY_VALID_HAND']
    resolve_began = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(lambda hand: resolve_hand(hand, resolver), hands))
    resolve_wall = time.perf_counter()-resolve_began
    cold = {'workers': args.workers, 'kind': 'LIVE_COLD' if args.jev_live else 'UNAVAILABLE',
            'wall_s': resolve_wall, 'live_calls': resolver.live_calls, 'cache_hits': resolver.cache_hits,
            'successful_responses': len(resolver.success_latencies),
            'throughput_successful_responses_s': len(resolver.success_latencies)/resolve_wall if resolver.success_latencies else None,
            'attempted_requests_s': resolver.live_calls/resolve_wall if resolver.live_calls else None}
    arms = [cold]
    for workers in [1, 2, 4]:
        cached_resolver = JevResolver(None, cache_dir, resolver.client_identity)
        began_arm = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            cached = list(pool.map(lambda hand: resolve_hand(hand, cached_resolver), hands))
        wall = time.perf_counter()-began_arm
        arms.append({'workers': workers, 'kind': 'CACHE_REPLAY' if cached_resolver.cache_hits else 'OFFLINE_FALLBACK_REPLAY', 'wall_s': wall,
                     'live_calls': cached_resolver.live_calls, 'cache_hits': cached_resolver.cache_hits,
                     'same_outcome_order': [(h['table_id'], h['hand_id'], o['final']['status']) for h, o in zip(hands, cached)] ==
                                           [(h['table_id'], h['hand_id'], o['final']['status']) for h, o in zip(hands, outcomes)]})
    patterns = {}
    before_patterns = Counter(code for result in initial for code in {i['code'] for i in result['issues']})
    for code, count in before_patterns.most_common():
        relevant = [i for i, result in enumerate(initial) if code in {x['code'] for x in result['issues']}]
        patterns[code] = {'before_jev': count, 'resolved_by_deterministic_rules': 0,
                          'sent_to_jev': sum(i in sent and bool(outcomes[i]['resolution'].get('live_call') or outcomes[i]['resolution'].get('cache_hit')) for i in relevant),
                          'resolved_by_jev_revalidation': sum(outcomes[i]['final']['accepted_inference'] for i in relevant),
                          'unresolved': sum(outcomes[i]['final']['status'] not in {'FULLY_VALID_HAND', 'JEV_ASSISTED_CERTIFIED'} for i in relevant)}
    tables = {}
    for table in sorted({h['table_id'] for h in hands}):
        indices = [i for i, h in enumerate(hands) if h['table_id'] == table]
        tables[table] = {'hands': len(indices), 'complete_candidates': sum(initial[i]['complete_candidate'] for i in indices),
                         'before': dict(Counter(initial[i]['status'] for i in indices)),
                         'after': dict(Counter(outcomes[i]['final']['status'] for i in indices))}
    after = registered_hashes()
    changed = [name for name in before if before[name] != after[name]]
    accepted = sum(o['final']['accepted_inference'] for o in outcomes)
    summary = {'schema': 'experimental.poker-logic-math-jev.v1', 'input': str(args.input.resolve()),
               'input_sha256': input_sha, 'input_rows': input_row_count,
               'downstream_rows': len(rows), 'edge_excluded_rows': input_row_count-len(rows),
               'edge_version': EDGE_VERSION, 'source_unchanged': digest(args.input) == input_sha,
               'versions': {'assembly': ASSEMBLY_VERSION, 'logic': LOGIC_VERSION, 'math': MATH_VERSION, 'jev': JEV_VERSION},
               'candidate_hands': len(hands), 'complete_candidate_hands': sum(r['complete_candidate'] for r in initial),
               'logic_pass': sum(r['logic_pass'] for r in initial), 'math_pass': sum(r['math']['pass'] for r in initial),
               'fully_valid': sum(r['status'] == 'FULLY_VALID_HAND' for r in initial),
               'eligible_for_jev': len(sent), 'sent_to_jev': resolver.live_calls+resolver.cache_hits,
               'jev_hypotheses_accepted': accepted,
               'jev_hypotheses_rejected': sum(bool(o['resolution'] and o['resolution'].get('jev_decision')) and not o['final']['accepted_inference'] for o in outcomes),
               'jev_assisted_certified': accepted,
               'still_unresolved': sum(o['final']['status'] not in {'FULLY_VALID_HAND', 'JEV_ASSISTED_CERTIFIED'} for o in outcomes),
               'jev_live_calls': resolver.live_calls, 'jev_cache_hits': resolver.cache_hits, 'jev_errors': resolver.errors,
               'jev_latency': timing(resolver.success_latencies), 'jev_attempt_latency': timing(resolver.latencies),
               'jev_successful_responses': len(resolver.success_latencies), 'jev_circuit_open': resolver.circuit_open,
               'jev_wall_s': resolve_wall,
               'logic_timing': timing([r['logic_ms'] for r in initial]),
               'math_timing': timing([r['math']['elapsed_ms'] for r in initial]),
               'parallel_arms': arms, 'live_parallel_comparison': 'NOT_MEASURED: identical requests reuse exact cache',
               'tables': tables, 'patterns': patterns, 'semantics': {k: v for k, v in semantics.items() if k != 'comparisons'},
               'cards_pass': sum(r['cards_pass'] for r in initial),
               'production_code_changed': bool(changed), 'changed_registered_files': changed,
               'verdict': 'INCONCLUSIVE', 'full_replay_wall_s': time.perf_counter()-began,
               'limitations': ['Experimental statuses do not promote production or export HH',
                               'Financial semantics evidence does not clear card, identity or boundary conflicts',
                               'Jev selects supplied evidence-linked hypotheses through native typed API',
                               'No nickname/boundary inference accepted without independent identity/boundary evidence']}
    write_json(args.output/'input_manifest.json', {'source': str(args.input.resolve()), 'sha256': input_sha,
                                                 'registered_before': before, 'registered_after': after})
    write_json(args.output/'summary.json', summary)
    write_json(args.output/'semantics_evidence.json', semantics)
    with (args.output/'hands.jsonl').open('w', encoding='utf-8') as stream:
        for hand, outcome in zip(hands, outcomes):
            stream.write(json.dumps({'observed_hand': hand, **outcome}, ensure_ascii=False)+'\n')
    lines = ['# Smoke17 experimental poker logic/math/Jev replay', '',
             'Input: '+str(args.input.resolve()), 'SHA256: '+input_sha, '',
             f"Rows {len(rows)}; candidate hands {len(hands)}; complete candidates {summary['complete_candidate_hands']}.",
             f"Deterministic FULLY_VALID {summary['fully_valid']}; logic PASS {summary['logic_pass']}; math PASS {summary['math_pass']}.",
             f"Jev calls {resolver.live_calls}; accepted {accepted}; rejected {summary['jev_hypotheses_rejected']}; unresolved {summary['still_unresolved']}.",
             'Production code changed: '+('YES' if changed else 'NO'), 'VERDICT: INCONCLUSIVE', '',
             '## Tables', '', '| Table | Candidates | Complete | Before | After |', '|---|---:|---:|---|---|']
    lines += [f"| {t} | {v['hands']} | {v['complete_candidates']} | {v['before']} | {v['after']} |" for t, v in tables.items()]
    lines += ['', '## Error patterns', '', '| Pattern | Before | Sent | Resolved | Unresolved |', '|---|---:|---:|---:|---:|']
    lines += [f"| {p} | {v['before_jev']} | {v['sent_to_jev']} | {v['resolved_by_jev_revalidation']} | {v['unresolved']} |" for p, v in patterns.items()]
    lines += ['', '## Performance', '', '```json', json.dumps({k: summary[k] for k in ['logic_timing', 'math_timing', 'jev_latency', 'parallel_arms', 'full_replay_wall_s']}, indent=2), '```',
              '', '## Limits', '', *['- '+x for x in summary['limitations']]]
    (args.output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ['input_rows', 'candidate_hands', 'complete_candidate_hands', 'fully_valid',
                                             'sent_to_jev', 'jev_hypotheses_accepted', 'jev_hypotheses_rejected', 'jev_errors',
                                             'still_unresolved', 'production_code_changed', 'full_replay_wall_s']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
