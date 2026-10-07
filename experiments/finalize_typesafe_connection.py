"""Summarize actual TypeSafe replay without changing its recorded observations.

Older live-process error/transport labels are retained in separate fields;
current nonrepair classification is applied offline, with no extra API calls.
Only Jev columns/cases change in the previously finalized review workbook data.
"""
import collections
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'runs/logic_math_fixed_smoke17_20261001'
LIVE = BASE/'typesafe_live_20261001'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run():
    summary = load(LIVE/'summary.json')
    data = load(BASE/'final_diagnostics_v2/workbook_data.json')
    live_data = load(LIVE/'workbook_data.json')
    records = [json.loads(l) for l in (LIVE/'hands.jsonl').read_text(encoding='utf-8').splitlines()]
    # No accepted candidates expected in this measurement. A future successful
    # repair needs the full final-view report, rather than this narrow update.
    assert summary['jev']['accepted'] == 0
    cases = copy.deepcopy(live_data['JEV_CASES'])
    for row in cases:
        row['live_process_transport_label'] = row['transport']
        row['transport'] = 'TYPESAFE_SYSTEMONE'
        row['live_process_reject_reason'] = row['reject_reason']
        decision = row['hypothesis']
        if decision.get('decision') == 'UNRESOLVED' and decision.get('target_rows') == [] and decision.get('evidence_rows') == []:
            row['reject_reason'] = 'JEV_UNRESOLVED'
        elif decision.get('decision') == 'CARD_EVIDENCE_CONFLICT' and decision.get('target_rows') == [] and decision.get('evidence_rows') == []:
            row['reject_reason'] = 'CARD_EVIDENCE_CONFLICT'
        native = (decision.get('native_response') or {}).get('data', {})
        row['actual_model'] = native.get('model')
        row['token_usage'] = native.get('usage')
        row['diagnostic_snapshot'] = str(LIVE)
    data['JEV_CASES'] = cases
    indexed = {row['hand_key']: row for row in cases}
    for row in data['HAND_SUMMARY']:
        case = indexed.get(row['hand_key'])
        row['jev_attempted'] = bool(case)
        row['jev_result'] = case['reject_reason'] if case else 'NOT_ELIGIBLE_PARTIAL_OR_ALREADY_FULLY_VALID'
        row['jev_diagnostic_snapshot'] = str(LIVE) if case else None
        # Every other diagnostic/proof field is deliberately preserved.
    decisions = collections.Counter(row['hypothesis_type'] for row in cases)
    rejections = collections.Counter(row['reject_reason'] for row in cases)
    usage = {key: sum((row['token_usage'] or {}).get(key, 0) for row in cases)
             for key in ['input_tokens', 'output_tokens']}
    report = {'provider': 'OFFICIAL_TYPESAFE', 'endpoint': 'https://api.typesafe.ai/v1/systemone',
              'credential_location': 'Windows User environment TYPESAFE_API_KEY',
              'mcp': {'transport': 'local stdio bridge', 'registered_server': 'jev',
                      'actual_probe': str(BASE/'typesafe_mcp_probe_20261001.json'),
                      'tested_question_types': ['choice', 'noul', 'score'],
                      'native_current_chat_tool_catalog_reloaded': False},
              'source': {key: summary[key] for key in ['input', 'input_sha256', 'physical_rows', 'segments', 'numbered', 'unnumbered']},
              'jev': summary['jev'], 'actual_models': dict(collections.Counter(row['actual_model'] for row in cases)),
              'decision_counts': dict(decisions), 'evidence_gate_rejections': dict(rejections), 'token_usage': usage,
              'final_status': summary['final_status'],
              'all_nine_isolated_table_replays_match': all(v['all_diagnostics_match'] for v in summary['table_isolation']),
              'production_hashes_match': not summary['production_code_changed'],
              'production_registered_hash_count': summary['production_registered_hashes_unchanged'],
              'report_label_corrections': 'Explicit UNRESOLVED is a nonrepair answer, not invalid row data. Actual provider label corrected; live labels retained separately.',
              'repair_scope': 'Closed evidence-linked candidates: repeated observed message or explicit observed blind amount. No free-form invented repair.',
              'source_and_original_live_records_modified': False}
    assert len(cases) == summary['jev']['attempted'] == summary['jev']['successful_responses']
    assert all(row['actual_model'] for row in cases)
    for filename, value in [('connection_report.json', report), ('workbook_data_typesafe.json', data)]:
        (LIVE/filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
