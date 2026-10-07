"""Saved Smoke17 OLD/NEW audit; no video, network, promotion or observed-row edits."""
from __future__ import annotations
import collections
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.poker_hand_assembly import assemble_hands, make_message
from parser_core.poker_logic_engine import validate_hand
from experiments.audit_poker_logic_engine import trace_validation, serial

BASE = ROOT/'runs/logic_engine_audit_smoke17_20261001'


def readlines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def ids(hand):
    return tuple(i for m in hand['messages'] for i in m['source_row_ids'])


def stable(value):
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k not in {'logic_ms', 'elapsed_ms'}}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


def stats(results):
    return {'candidate_windows': len(results), 'complete': sum(v['complete_candidate'] for v in results),
            'logic_pass': sum(v['logic_pass'] for v in results),
            'math_pass': sum(v['math']['pass'] for v in results),
            'card_pass': sum(v['cards_pass'] for v in results),
            'fully_valid': sum(v['status'] == 'FULLY_VALID_HAND' for v in results),
            'statuses': dict(collections.Counter(v['status'] for v in results)),
            'hands_containing_failure_codes': dict(collections.Counter(c for v in results for c in {i['code'] for i in v['issues']})),
            'action_logic_statuses': dict(collections.Counter(v.get('action_logic_status', 'PASS' if v['logic_pass'] else 'FAIL') for v in results)),
            'card_statuses': dict(collections.Counter(v.get('card_status', 'PASS' if v['cards_pass'] else 'CONFLICT') for v in results))}


def classify(issue, message, new_result, new_message=None):
    code = issue['code']
    action = message.get('action') if message else None
    if code == 'ROW_IDENTITY_UNRESOLVED':
        matching = [i for i in new_result['issues'] if i['code'] == code and
                    set(i['source_row_ids']) & set(issue['source_row_ids'])]
        if matching and all(i['severity'] == 'WARNING' for i in matching):
            return 'LOGIC_ENGINE_BUG', 'Harmless informational origin was a blanket betting-logic blocker; observation retained as NONCRITICAL warning, with no financial inference dependency.'
        return 'UPSTREAM_DIRTY_TIMELINE', 'Observation multiplicity/order or financial inference evidence is unresolved. Critical provenance remains a certification blocker; action verdict is reported separately.'
    if code == 'CARD_LOGIC_CONFLICT' and new_result['card_status'] == 'NOT_VERIFIABLE':
        return 'LOGIC_ENGINE_BUG', 'OLD called missing board/live hole-card evidence a contradiction. NEW keeps missing-evidence certification blocker.'
    if code == 'MISSING_ACTION' and new_message and new_message['action'] in {'TABLE_INFO', 'BLIND_LEVEL'}:
        return 'LOGIC_ENGINE_BUG', 'Anchored server notice was parsed as UNKNOWN_EVENT. It contains no betting action; response/round closure checks remain active.'
    if code == 'CHAT_OVERLAP_DUPLICATE' and action in {'HAND_HEADER', 'HAND_ID'}:
        return 'LOGIC_ENGINE_BUG', 'Repeated same explicit hand identifier is redundant metadata, retained as warning; no financial row is masked.'
    if code in {'INCOMPLETE_START', 'INCOMPLETE_END', 'HAND_BOUNDARY_CONFLICT', 'MISSING_ACTION', 'MISSING_AMOUNT',
                'PLAYER_IDENTITY_CONFLICT', 'BETTING_ROUND_INCOMPLETE', 'OLD_WIN', 'STALE_BLIND_OR_ANTE',
                'CHAT_OVERLAP_DUPLICATE', 'MISSING_UNCALLED_RETURN', 'SIDE_POT_ELIGIBILITY_CONFLICT'}:
        return 'UPSTREAM_DIRTY_TIMELINE', 'Saved sequence contains a gap/repeated observation or ambiguous boundary/identity. Delta=0 does not prove action completeness. No hidden action or stale financial row is inferred.'
    return 'TRUE_POKER_CONFLICT', 'Observed parsed sequence violates this rule given recorded BEFORE state. This does not prove the actual played hand was illegal; upstream corruption can produce the sequence.'


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=BASE/'final')
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    old = readlines(BASE/'baseline/baseline_hands.jsonl')
    freeze = json.loads((BASE/'baseline/freeze_manifest.json').read_text(encoding='utf-8'))
    timeline = ROOT/freeze['timeline_path']
    original = readlines(timeline)
    rows = copy.deepcopy(original)
    for row in rows:
        row['raw_text'] = row.get('raw_ocr_text', row.get('raw_text', row.get('text', '')))
        row['row_id'] = row.get('stable_row_id', row.get('row_id'))
        row['timestamp_ms'] = row.get('timestamp_ms', float(row.get('video_time', 0))*1000)
    baseline = {ids(r['observed_hand']): r for r in old}
    def configure(hand):
        previous = baseline[ids(hand)]['observed_hand']
        hand['semantics'] = copy.deepcopy(previous['semantics'])
        if 'blind_level' in previous:
            hand['blind_level'] = copy.deepcopy(previous['blind_level'])
        return hand
    hands = [configure(h) for h in assemble_hands(rows)]
    unchanged_hands = copy.deepcopy(hands)
    results = [validate_hand(h) for h in hands]
    old_results = [baseline[ids(h)]['initial'] for h in hands]
    records = [{'observed_hand': h, 'initial': v, 'old': b} for h, v, b in zip(hands, results, old_results)]
    (out/'hands.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    math_diff = [i for i, (b, v) in enumerate(zip(old_results, results)) if stable(b['math']) != stable(v['math'])]
    player_diff = [i for i, (b, v) in enumerate(zip(old_results, results)) if stable(b['players']) != stable(v['players'])]
    source_lookup = {r['stable_row_id']: r for r in original}
    table_manifest = []
    for h in hands:
        tables = sorted({source_lookup[i]['table_id'] for i in ids(h)})
        table_manifest.append({'hand_id': h['hand_id'], 'table_id': h['table_id'], 'hand_epoch': h['hand_epoch'],
                               'source_rows': list(ids(h)), 'unique_table_ids_inside_hand': tables})
    (out/'all_hand_table_manifest.json').write_text(json.dumps(table_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    table_results = {}
    global_lookup = {ids(h): v for h, v in zip(hands, results)}
    for table in sorted({r['table_id'] for r in rows}):
        local = [configure(h) for h in assemble_hands([r for r in rows if r['table_id'] == table])]
        validated = [validate_hand(h) for h in local]
        mismatch = [h['hand_id'] for h, v in zip(local, validated) if stable(v) != stable(global_lookup[ids(h)])]
        table_results[table] = {'hands': len(local), 'logic_pass_all': sum(global_lookup[ids(h)]['logic_pass'] for h in local),
                                'logic_pass_alone': sum(v['logic_pass'] for v in validated), 'mismatches': mismatch}
        (out/(table+'_replay.jsonl')).write_text(''.join(json.dumps({'observed_hand': h, 'validation': v}, ensure_ascii=False)+'\n' for h, v in zip(local, validated)), encoding='utf-8')
    matrix, classifications, traces, masks = [], [], [], []
    old_trace_lookup = {(t['table_id'], t['hand_epoch']): t for t in readlines(BASE/'old_traces/traces.jsonl')}
    for h, v, b in zip(hands, results, old_results):
        if b['math']['delta'] != '0':
            continue
        bh = baseline[ids(h)]['observed_hand']
        trace = trace_validation(h, validate_hand)
        traces.append(trace)
        name = f"{h['table_id']}_{h['hand_id'] or 'UNKNOWN'}_epoch{h['hand_epoch']}"
        (out/(name+'_trace.json')).write_text(json.dumps(serial(trace), ensure_ascii=False, indent=2), encoding='utf-8')
        failures = [i for i in b['issues'] if not i['code'].startswith('CARD_') and i['code'] not in b['math']['errors']]
        first = failures[0]['code'] if failures else 'NONE'
        matrix.append({'table_id': h['table_id'], 'hand_id': h['hand_id'], 'hand_epoch': h['hand_epoch'],
                       'start_complete': h['start_observed'], 'end_complete': h['end_observed'],
                       'old_complete_candidate': b['complete_candidate'], 'new_complete_candidate': v['complete_candidate'],
                       'old_logic_status': b['status'], 'new_logic_status': v['status'],
                       'action_logic_status': v['action_logic_status'], 'math_delta': v['math']['delta'],
                       'card_status': v['card_status'], 'identity_status': v['identity_status'],
                       'first_logic_failure': first, 'all_logic_failures': failures,
                       'affected_rows': [i for x in failures for i in x['source_row_ids']],
                       'timestamps': [m.get('timestamp_ms') for m in bh['messages']],
                       'raw_text': [m['raw_text'] for m in bh['messages']], 'new_issues': v['issues']})
        previous_trace = old_trace_lookup[(h['table_id'], h['hand_epoch'])]
        for issue in b['issues']:
            index = issue['message_index']
            message = bh['messages'][index] if index >= 0 else None
            new_message = next((m for m in h['messages'] if message and set(message['source_row_ids']) <= set(m['source_row_ids'])), None)
            category, reason = classify(issue, message, v, new_message)
            step = next((a for a in previous_trace['actions'] if a['message_index'] == index), None)
            classifications.append({'table_id': h['table_id'], 'hand_id': h['hand_id'], 'hand_epoch': h['hand_epoch'],
                                    'old_issue': issue, 'classification': category, 'evidence': reason,
                                    'raw_text': message['raw_text'] if message else None,
                                    'timestamp_ms': message.get('timestamp_ms') if message else None,
                                    'BEFORE': step['before'] if step else None, 'AFTER': step['after'] if step else None,
                                    'trace_file': 'old_traces/'+name+'.json'})
        # Candidate view only: identical pixels/text do not establish that two
        # money/action observations denote one event. Metadata repeats alone are
        # safely redundant given the same explicit hand identifier.
        seen, redundant, ambiguous = set(), [], []
        for i, m in enumerate(h['messages']):
            number = m.get('hand_number')
            if m['action'] in {'HAND_HEADER', 'HAND_ID'} and number == h['hand_id']:
                if number in seen:
                    redundant.append(i)
                seen.add(number)
        for issue in v['issues']:
            if issue['code'] in {'CHAT_OVERLAP_DUPLICATE', 'STALE_BLIND_OR_ANTE', 'OLD_WIN'} and issue['message_index'] not in redundant:
                ambiguous.append({'issue': issue, 'decision': 'AMBIGUOUS; retained. No independent stale/multiplicity proof.'})
        candidate = copy.deepcopy(h)
        candidate['excluded_message_indices'] = redundant
        cv = validate_hand(candidate)
        masks.append({'table_id': h['table_id'], 'hand_id': h['hand_id'], 'hand_epoch': h['hand_epoch'],
                      'masked_message_indices': redundant, 'proof': 'Repeated exact explicit hand identifier; redundant metadata only',
                      'raw': stable(v), 'cleaned_candidate': stable(cv), 'ambiguous_retained': ambiguous,
                      'observed_rows_unchanged': ids(candidate) == ids(h), 'adopted': False})
    matrix.sort(key=lambda m: (m['first_logic_failure'], m['table_id'], m['hand_epoch']))
    for filename, value in [('math_pass_22_matrix.json', matrix), ('per_failure_classification.json', classifications), ('raw_vs_cleaned_candidates.json', masks)]:
        (out/filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'new_traces.jsonl').write_text(''.join(json.dumps(serial(t), ensure_ascii=False)+'\n' for t in traces), encoding='utf-8')
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    changed_registered = [path for path, hash_ in freeze['registered_hashes'].items() if sha(ROOT/path) != hash_['actual']]
    unchanged_experimental = ['parser_core/jev_poker_resolver.py', 'parser_core/poker_math_engine.py', 'parser_core/poker_card_engine.py',
                              'experiments/replay_poker_logic_math_jev.py', 'tests/test_poker_logic_math_jev.py']
    unchanged_checks = {path: sha(ROOT/path) == freeze['experimental_source_hashes'][path] for path in unchanged_experimental}
    # Exact raw row objects/order per table; overall assembler groups by table.
    assembled = [r for h in hands for m in h['messages'] for r in m['source_rows']]
    expected = [r for table in dict.fromkeys(r['table_id'] for r in rows) for r in rows if r['table_id'] == table]
    parity = assembled == expected
    summary = {'before': stats(old_results), 'after': stats(results), 'fixed_22_total': len(matrix),
               'fixed_22_complete': sum(m['old_complete_candidate'] for m in matrix),
               'fixed_22_logic_pass_before': sum(b['logic_pass'] for b in old_results if b['math']['delta'] == '0'),
               'fixed_22_logic_pass_after': sum(v['logic_pass'] for b, v in zip(old_results, results) if b['math']['delta'] == '0'),
               'fixed_22_action_statuses': dict(collections.Counter(m['action_logic_status'] for m in matrix)),
               'fixed_22_card_statuses': dict(collections.Counter(m['card_status'] for m in matrix)),
               'fixed_22_old_code_frequencies': dict(collections.Counter(code for b in old_results if b['math']['delta'] == '0' for code in {i['code'] for i in b['issues']})),
               'classification_occurrences': dict(collections.Counter(c['classification'] for c in classifications)),
               'math_ledger_parity': {'pass': len(hands)-len(math_diff), 'fail': math_diff},
               'player_state_parity': {'pass': len(hands)-len(player_diff), 'fail': player_diff},
               'source_rows': len(rows), 'source_field_order_parity': parity,
               'source_hash_unchanged': sha(timeline) == freeze['timeline_sha256'],
               'registered_hashes_unchanged': len(freeze['registered_hashes'])-len(changed_registered),
               'registered_changed': changed_registered, 'unchanged_components': unchanged_checks,
               'table_isolation': table_results, 'unique_table_violations': sum(len(m['unique_table_ids_inside_hand']) != 1 for m in table_manifest),
               'new_trace_messages': sum(len(t['actions']) for t in traces), 'candidate_masks_adopted': 0,
               'video': 'NOT_DECODED', 'Jev': 'NOT_CALLED', 'production_code_changed': 'YES' if changed_registered else 'NO',
               'integrity_status': 'FAIL' if changed_registered or not all(unchanged_checks.values()) or not parity or sha(timeline) != freeze['timeline_sha256'] else 'PASS'}
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    assert hands == unchanged_hands and len(hands) == 199 and len(matrix) == 22
    assert not math_diff and not player_diff and parity and not changed_registered and all(unchanged_checks.values())
    assert summary['source_hash_unchanged'] and not summary['unique_table_violations']
    assert all(not t['mismatches'] for t in table_results.values())


if __name__ == '__main__':
    main()
