"""Read-only audit of saved official live sidecar receipts; never OCR crops."""
import argparse
import collections
import hashlib
import json
import re
import time
from pathlib import Path


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='runs/hand_state_marker_gate_20261003/acceptance_ON')
    parser.add_argument('--adjudication', default='runs/hand_state_marker_gate_20261003/acceptance_adjudication.json')
    parser.add_argument('--output', default='runs/start_stack_live_path_20261003/existing_path_audit')
    args = parser.parse_args()
    started = time.perf_counter()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    hands = json.loads(Path(args.adjudication).read_text('utf-8'))['hands']
    headers = collections.defaultdict(dict)
    for r in rows(source / 'hand_state_receipts.jsonl'):
        if r['status'] == 'HAND_HEADER_CONFIRMED':
            key = (r['table_id'], r['session_id'])
            old = headers[key].get(r['hand_id'])
            headers[key][r['hand_id']] = min(old if old is not None else r['source_frame'], r['source_frame'])
    capture = collections.defaultdict(dict)
    total_crops = 0
    capture_status = collections.Counter()
    for r in rows(source / 'start_stack_capture.jsonl'):
        total_crops += 1
        capture_status[r.get('status', 'MISSING')] += 1
        capture[(r['table_id'], r['session_id'], r['seat'], r['field'])][r['source_frame']] = r['pixel_sha256']
    deals = list(rows(source / 'deal_complete_receipts.jsonl'))
    readings = list(rows(source / 'start_stack_readings.jsonl'))
    actions = collections.defaultdict(list)
    grammar = re.compile(r'^Dealer:\s+.+\s+(calls|bets|raises|is all-in|goes all-in)\b', re.I)
    for r in rows(source / 'raw_ocr_results.jsonl'):
        response = r.get('raw_ocr_response', {})
        if response.get('status') == 'OK' and grammar.match(response.get('raw_text', '')):
            actions[(r['table_id'], r['table_session_id'])].append(r)
    profile = json.loads(Path('configs/layout_847x404.json').read_text('utf-8'))
    seat_rows = []
    for h in hands:
        event = h['events'][0]
        table, session, hand = h['table'], event['session_id'], h['hand_id']
        first = event['trigger_source_frame']
        next_headers = [f for other, f in headers[(table, session)].items() if other != hand and f > first]
        end = min(next_headers) if next_headers else 5401
        action_candidates = [r for r in actions[(table, session)] if first <= r['frame_id'] < end]
        first_action = min(action_candidates, key=lambda r: (r['frame_id'], r['source_span_index'])) if action_candidates else None
        for seat in profile['slots'][f'slot_{table:02d}']['seats']:
            field_evidence = {}
            for field in ('nick', 'stack'):
                values = capture[(table, session, seat, field)]
                found = sorted(f for f in values if first <= f < end)
                triples = [[f, f+1, f+2] for f in found if f+1 in values and f+2 in values and f+2 < end]
                field_evidence[field] = {
                    'crop_count': len(found), 'first_frame': found[0] if found else None,
                    'last_frame': found[-1] if found else None,
                    'consecutive_crop_triple_count': len(triples),
                    'first_consecutive_crop_triple': triples[0] if triples else None,
                    'frame_hashes': [{'frame': f, 'pixel_sha256': values[f]} for f in found],
                }
            seat_rows.append({
                'table': table, 'session': session, 'hand_id': hand, 'seat': seat,
                'occupancy_player': None, 'occupancy_status': 'UNKNOWN',
                'dealing_cards_frame': first, 'deal_complete_frame': None,
                'stable_numeric_stack_frames': None, 'displayed_stack': None,
                'ANTE': None, 'SB_BB': None, 'START_STACK': None,
                'first_stack_changing_action_frame': None,
                'first_money_ocr_observation_frame_candidate': first_action['frame_id'] if first_action else None,
                'money_ocr_observation_is_action_onset': False,
                'PASS_UNKNOWN': 'UNKNOWN',
                'exact_blocker': 'TABLE_DEAL_COMPLETE_PROVIDER_UNAVAILABLE',
                'additional_blockers': ['OCCUPANCY_EPOCH_UNPROVEN', 'NUMERIC_STACK_OCR_PROVIDER_UNAVAILABLE', 'FORCED_BET_BINDING_UNPROVEN', 'ACTION_ONSET_TIMING_UNPROVEN'],
                'provisional_capture_interval': [first, end],
                'provisional_crops': field_evidence,
            })
    actual_deals = [r for r in deals if r.get('status') == 'DEAL_COMPLETE']
    actual_pass = [r for r in readings if r.get('status') == 'START_STACK_RECOVERED']
    result = {
        'scope': 'SAVED_OFFICIAL_LIVE_PRODUCTION_RECEIPTS_ONLY; NO_NEW_OCR_OR_VIDEO_RUN',
        'source': str(source), 'HANDS': len(hands), 'CONFIGURED_SEAT_GROUPS': len(seat_rows),
        'OCCUPIED_SEATS': None, 'OCCUPIED_SEATS_STATUS': 'UNKNOWN_NOT_MEASURED',
        'DEAL_COMPLETE_PASS': len(actual_deals), 'DEAL_COMPLETE_UNKNOWN_HANDS': len(hands),
        'DEAL_COMPLETE_FAIL': 0, 'DEAL_COMPLETE_FAIL_MEANING': 'No table-proof contradictions measured; UNKNOWN is not FAIL',
        'ACTUAL_NUMERIC_STACK_READINGS': len(readings), 'STACK_TRIPLES_PASS': 0,
        'START_STACK_PASS': len(actual_pass), 'START_STACK_UNKNOWN_CONFIGURED_GROUPS': len(seat_rows),
        'START_STACK_UNKNOWN_OCCUPIED_SEATS': None, 'CONFLICTS': 0,
        'CONFLICTS_MEANING': 'No contradictory proved results; underlying numeric/provider facts not measured',
        'RAW_PROVISIONAL_CROP_ROWS': total_crops, 'CROP_STATUSES': dict(capture_status),
        'CONFIGURED_GROUPS_WITH_THREE_CONSECUTIVE_STACK_CROPS': sum(bool(r['provisional_crops']['stack']['consecutive_crop_triple_count']) for r in seat_rows),
        'INDEPENDENT_MATH_CHECKS': 'NOT_MEASURED_NO_PASS_CASES',
        'WARNING': 'Configured seats are not proved occupied. Consecutive crop images are not numeric readings. Crop assignment uses source-frame hand intervals, not production hand binding. OCR observation frames are not proved action onset.',
        'seat_rows': seat_rows,
    }
    result['audit_wall_seconds'] = round(time.perf_counter() - started, 6)
    result['input_sha256'] = {}
    for name in ('dealing_cards_triggers.jsonl', 'deal_complete_receipts.jsonl', 'start_stack_readings.jsonl', 'dealing_start_stack_summary.json'):
        result['input_sha256'][name] = hashlib.sha256((source/name).read_bytes()).hexdigest()
    (output/'existing_path_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
    lines = ['# Saved live START_STACK path audit', '', result['WARNING'], '',
             f"HANDS {len(hands)}; configured seat groups {len(seat_rows)}; occupied seats UNKNOWN; DEAL_COMPLETE PASS 0 / UNKNOWN {len(hands)}; numeric readings 0; numeric triples PASS 0; START_STACK PASS 0 / UNKNOWN configured groups {len(seat_rows)}.", '',
             '|table|hand_id|seat|occupancy/player|Dealing cards|DEAL_COMPLETE|3 stable numeric frames|displayed|ANTE|SB/BB|START_STACK|first stack-changing action|status|exact blocker|provisional stack crops|first crop triple|',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in seat_rows:
        crops = r['provisional_crops']['stack']
        lines.append(f"|T{r['table']:02d}|{r['hand_id']}|{r['seat']}|UNKNOWN|{r['dealing_cards_frame']}|UNKNOWN|UNKNOWN|UNKNOWN|UNKNOWN|UNKNOWN|UNKNOWN|UNKNOWN|UNKNOWN|{r['exact_blocker']}|{crops['crop_count']}|{crops['first_consecutive_crop_triple']}|")
    (output/'EXISTING_PATH_AUDIT.md').write_text('\n'.join(lines)+'\n', 'utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('seat_rows', 'input_sha256')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
