"""Offline deterministic diagnostics. Traces the actual validator, never a second engine."""
from __future__ import annotations

import ast
import copy
from decimal import Decimal
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def serial(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (set, tuple, list)):
        return [serial(v) for v in value]
    return value


def trace_validation(hand, validator):
    """Line tracing sees the real BEFORE/AFTER state including every continue path."""
    filename = inspect.getsourcefile(validator)
    source = Path(filename).read_text(encoding='utf-8')
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == validator.__name__)
    main_loop = next(n for n in function.body if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple)
                     and [getattr(e, 'id', '') for e in n.target.elts] == ['index', 'msg'])
    first_body = main_loop.body[0].lineno
    source_lines = source.splitlines()
    check_lines = {n.lineno for n in ast.walk(function) if isinstance(n, (ast.If, ast.For))}
    entries = []
    current = None
    final_checks = []
    first_issues = []

    def state(locals_, name=None):
        players = locals_.get('players', {})
        player = players.get(name)
        return serial({'street': locals_.get('street'), 'current_bet': locals_.get('bet'),
                       'player': name, 'player_state': player.get('state') if player else 'NOT_YET_OBSERVED',
                       'player_street_contribution': player.get('street_contribution') if player else None,
                       'player_total_contribution': player.get('total_contribution') if player else None,
                       'players': copy.deepcopy(players), 'ended': locals_.get('ended'),
                       'board': list(locals_.get('board', []))})

    def finish(locals_):
        nonlocal current
        if current is not None:
            current['after'] = state(locals_, current['player'])
            current['failures'] = serial(locals_['issues'][current.pop('_issue_start'):])
            current = None

    def tracer(frame, event, arg):
        nonlocal current, first_issues
        if frame.f_code is not validator.__code__:
            return tracer
        data = frame.f_locals
        line = frame.f_lineno
        if event == 'line' and line == first_body:
            finish(data)
            index, message = data['index'], data['msg']
            if not entries:
                first_issues = serial(data['issues'])
            current = {'message_index': index, 'source_rows': copy.deepcopy(message.get('source_rows', [])),
                       'source_row_ids': message['source_row_ids'], 'timestamp_ms': message.get('timestamp_ms'),
                       'frame_id': message.get('frame_id'), 'raw_text': message['raw_text'],
                       'parsed_action': message['action'], 'player': message.get('player'),
                       'amount': message.get('amount'), 'before': state(data, message.get('player')),
                       'rules_evaluated': [], 'excluded': index in data.get('excluded', set()),
                       '_issue_start': len(data['issues'])}
            entries.append(current)
        if event == 'line' and line > main_loop.end_lineno:
            finish(data)
            if line in check_lines:
                final_checks.append({'line': line, 'check': source_lines[line-1].strip()})
        elif event == 'line' and current is not None and line in check_lines:
            check = {'line': line, 'check': source_lines[line-1].strip()}
            if check not in current['rules_evaluated']:
                current['rules_evaluated'].append(check)
        if event == 'return':
            finish(data)
        return tracer

    prior = sys.gettrace()
    try:
        sys.settrace(tracer)
        result = validator(hand)
    finally:
        sys.settrace(prior)
    message_issues = [issue for entry in entries for issue in entry['failures']]
    return {'table_id': hand['table_id'], 'hand_id': hand['hand_id'], 'hand_epoch': hand['hand_epoch'],
            'START': {'start_observed': hand.get('start_observed'), 'end_observed': hand.get('end_observed'),
                      'initial_failures': first_issues}, 'actions': entries,
            'END': {'rules_evaluated': final_checks, 'failures': [i for i in result['issues'] if i not in message_issues],
                    'result': result}, 'validator_file': str(Path(filename).resolve())}


def main():
    import argparse
    from parser_core.poker_logic_engine import validate_hand
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hands', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(x) for x in args.hands.read_text(encoding='utf-8').splitlines() if x]
    selected = [r for r in records if r.get('initial', r.get('result', {}))['math']['delta'] == '0']
    traces = []
    for record in selected:
        hand = record.get('observed_hand', record.get('hand'))
        trace = trace_validation(hand, validate_hand)
        traces.append(trace)
        name = f"{hand['table_id']}_{hand['hand_id'] or 'UNKNOWN'}_epoch{hand['hand_epoch']}"
        (args.output/(name+'.json')).write_text(json.dumps(serial(trace), ensure_ascii=False, indent=2), encoding='utf-8')
        lines = [f"# {name}", '', 'START', '```json', json.dumps(trace['START'], ensure_ascii=False, indent=2), '```']
        for action in trace['actions']:
            lines += ['', f"## Action {action['message_index']}: {action['parsed_action']}", '',
                      '```json', json.dumps(serial(action), ensure_ascii=False, indent=2), '```']
        lines += ['', 'END', '```json', json.dumps(serial(trace['END']), ensure_ascii=False, indent=2), '```', '']
        (args.output/(name+'.md')).write_text('\n'.join(lines), encoding='utf-8')
    (args.output/'traces.jsonl').write_text(''.join(json.dumps(serial(t), ensure_ascii=False)+'\n' for t in traces), encoding='utf-8')
    print(json.dumps({'math_pass_traces': len(traces), 'messages_traced': sum(len(t['actions']) for t in traces)}))


if __name__ == '__main__':
    main()
