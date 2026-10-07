"""Append the separately measured MCP default-model probe to final review data."""
import collections
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]/'runs/logic_math_fixed_smoke17_20261001'
OUT = BASE/'final_diagnostics_v2'
MCP = BASE/'first_break_final_mcp'


def run():
    summary = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    data = json.loads((OUT/'workbook_data.json').read_text(encoding='utf-8'))
    mcp_data = json.loads((MCP/'workbook_data.json').read_text(encoding='utf-8'))
    data['JEV_CASES'] = mcp_data['JEV_CASES']
    for case in data['JEV_CASES']:
        case['diagnostic_snapshot'] = str(MCP)
    probe = json.loads((MCP/'service_default_probe.json').read_text(encoding='utf-8'))['result']
    if not any(v.get('transport') == 'MCP_jev_decide_SERVICE_DEFAULT' for v in data['JEV_CASES']):
        case = dict(data['JEV_CASES'][0])
        case.update(latency_ms=probe['latency_ms'], api_error=probe['api_error_detail'],
                    request_hash=probe['request_hash'], transport='MCP_jev_decide_SERVICE_DEFAULT')
        data['JEV_CASES'].append(case)
    summary['jev']['service_default_probe'] = {'attempted': 1, 'api_failures': 1, 'successful_responses': 0,
                                              'accepted': 0, 'latency_ms': probe['latency_ms'], 'error': probe['api_error_detail']}
    summary['jev']['all_mcp_calls_including_service_default_probe'] = 4
    summary['jev']['previous_diagnostic_snapshot'] = str(MCP)
    summary['jev']['previous_snapshot_attempted'] = 3
    summary['jev']['previous_snapshot_api_failures'] = 3
    summary['jev']['previous_snapshot_preflight_32k_rejections'] = 28
    summary['jev']['reason_no_new_calls_after_final_parser_fixes'] = 'MCP model access rejected, including service default probe; source fixes do not resolve service access'
    summary['rows_per_sheet']['JEV_CASES'] = len(data['JEV_CASES'])
    old = {}
    for line in (BASE/'first_break_before_assembly_fixes.jsonl').read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        h = record['hand']
        old[f'{h["table_id"]}:{h["session_id"]}:{h["hand_id"] or "PREFIX"}'] = record
    evidence = json.loads((OUT/'source_repair_evidence.json').read_text(encoding='utf-8'))
    statuses = {r['hand_key']: r['math_status'] for r in data['HAND_SUMMARY']}
    data['MATH_FAILS'] = [r for r in data['MATH_FAILS'] if statuses[r['hand_key']] == 'FAIL']
    data['ERROR_STATS'] = [r for r in data['ERROR_STATS'] if r.get('rule_trigger') != 'LEDGER_NOT_VERIFIABLE']
    records = [json.loads(l) for l in (OUT/'hands.jsonl').read_text(encoding='utf-8').splitlines()]
    grouped = collections.defaultdict(list)
    source = OUT.parents[1]/'temporal_overlap_boundary_smoke17_20261001/ordered_timeline.jsonl'
    for line in source.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        r['row_id'] = r.get('stable_row_id', r.get('row_id'))
        r['timestamp_ms'] = r.get('timestamp_ms', r.get('video_time', 0)*1000)
        grouped[r['table_id']].append(r)
    positions = {t: {r['row_id']: i for i, r in enumerate(rows)} for t, rows in grouped.items()}
    data['SOURCE_CONTEXT'] = []
    for record in records:
        hand, first = record['hand'], record['diagnostics']['first_break']
        if not first or first['message_index'] < 0:
            continue
        msg = hand['messages'][first['message_index']]
        table = hand['table_id']
        start, end = (positions[table][msg['source_row_ids'][i]] for i in [0, -1])
        for i in range(max(0, start-10), min(len(grouped[table]), end+11)):
            r = grouped[table][i]
            data['SOURCE_CONTEXT'].append({'table_id': table, 'hand_id': hand['hand_id'],
                'hand_key': f'{table}:{hand["session_id"]}:{hand["hand_id"] or "PREFIX"}',
                'break_row_id': msg['source_row_ids'][0], 'relative_offset': i-start,
                'source_row_id': r['row_id'], 'frame_id': r['frame_id'], 'timestamp_ms': r['timestamp_ms'],
                'raw_text': r.get('raw_ocr_text', r.get('raw_text', r.get('text', ''))),
                'relation': 'BREAK' if r['row_id'] in msg['source_row_ids'] else 'BEFORE' if i < start else 'AFTER',
                'source_epoch': r.get('epoch'), 'session_id': r.get('session_id')})
    for row in data['HAND_SUMMARY']:
        cases = [r for r in data['JEV_CASES'] if r['table_id'] == row['table_id'] and r['hand_id'] == row['hand_id']]
        row['jev_attempted'] = bool(cases)
        row['jev_result'] = 'MCP_MODEL_ACCESS_REJECTED' if cases else 'NOT_ATTEMPTED_MCP_MODEL_ACCESS_UNAVAILABLE'
        row['jev_diagnostic_snapshot'] = str(MCP) if cases else None
        before = old[row['hand_key']]['diagnostics']['first_break']
        row['first_break_before_parser_fixes'] = before['observed'] if before else None
        row['first_rule_before_parser_fixes'] = before['code'] if before else None
        unique = {tuple(v['source_row_ids']): v for v in evidence if v['hand_key'] == row['hand_key']}
        row['source_proven_repairs_in_hand'] = [{'PROVEN_CAUSE': v['PROVEN_CAUSE'], 'source_row_ids': v['source_row_ids'],
                                               'before': v['before'], 'after': v['after'], 'full_hand_cause_proven': False}
                                              for v in unique.values()]
    summary['rows_per_sheet'] = {name: len(rows) for name, rows in data.items()}
    for name, value in [('summary.json', summary), ('workbook_data.json', data)]:
        (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    run()
