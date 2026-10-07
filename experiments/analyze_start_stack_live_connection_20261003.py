"""Saved official ON connection analysis, no new recognition or repair."""
import collections
import hashlib
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/start_stack_live_path_20261003'
RUN = OUT/'connection_acceptance_ON'
BASE = ROOT/'runs/hand_state_marker_gate_20261003/acceptance_ON'
TIMING = {'elapsed_ms', 'hybrid_elapsed_seconds', 'logic_ms', 'pixel_elapsed_seconds', 'vl_elapsed_seconds'}
NUMERIC = re.compile(r'^\d+(?:[ \u00a0]\d{3})*$')


def read(path):
    return json.loads(path.read_text('utf-8'))


def rows(path):
    if not path.exists():
        return
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k,v in obj.items() if k not in TIMING}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    return obj


def main():
    started = time.perf_counter()
    prior = read(ROOT/'runs/hand_state_marker_gate_20261003/acceptance_adjudication.json')
    expected = {(h['table'],h['hand_id']): h for h in prior['hands']}
    live = read(RUN/'start_stack_live_summary.json') if (RUN/'start_stack_live_summary.json').exists() else None
    triggers = list(rows(RUN/'dealing_cards_triggers.jsonl'))
    by_hand = collections.defaultdict(list)
    for r in triggers:
        by_hand[r['table_id'],r['hand_id']].append(r)
    initial_expected = {(r['table_id'],r['hand_id'],r['trigger_source_frame'],r['pixel_hash']) for r in prior['initial_partial_events']}
    unexpected = [r for r in triggers if (r['table_id'],r['hand_id']) not in expected]
    false = [r for r in unexpected if (r['table_id'],r['hand_id'],r['trigger_source_frame'],r['pixel_hash']) not in initial_expected]
    marker_conflicts = [r for k, rr in by_hand.items() if k in expected for r in rr
        if (r['trigger_source_frame'],r['pixel_hash']) != (expected[k]['events'][0]['trigger_source_frame'],expected[k]['events'][0]['pixel_hash'])]
    live_hands = {(h['table_id'],h['hand_id']): h for h in (live or {}).get('hands',[])}
    receipts = list(rows(RUN/'start_stack_live_receipts.jsonl'))
    receipt_hands = collections.defaultdict(list)
    for r in receipts:
        receipt_hands[r['table_id'],r['hand_id']].append(r)
    profile = read(ROOT/'configs/layout_847x404.json')
    detector = read(OUT/'connection_candidate/models/start_stack_table_cards.json')
    seat_rows = []
    perhand = []
    for key, ref in expected.items():
        t, hand = key
        events = by_hand[key]
        h = live_hands.get(key, {})
        deal_frame = h.get('deal_complete_frame')
        deal_receipts = [r for r in receipt_hands[key] if r.get('stage') == 'DEAL_COMPLETE' and r.get('source_frame') == deal_frame]
        deal = deal_receipts[0] if deal_receipts else None
        triples = h.get('numeric_triples', {})
        stack_readings = [r for r in receipt_hands[key] if r.get('stage') == 'STACK_OCR']
        checked = [r for r in receipt_hands[key] if r.get('stage') == 'TABLE_CARD_NCC']
        perhand.append({'table':t,'hand_id':hand,'trigger_count':len(events),
            'deal_complete_frame':deal_frame,'DEAL_COMPLETE':'PASS' if deal else 'UNKNOWN',
            'table_frames_checked':len(checked),'numeric_triples':len(triples),
            'actual_stack_ocr_receipts':len(stack_readings),'START_STACK_PASS':h.get('start_stack_pass',0)})
        for seat in profile['slots'][f'slot_{t:02d}']['seats']:
            triple = triples.get(seat)
            sr = [r for r in stack_readings if r['seat'] == seat]
            numbers = [r for r in sr if r.get('ocr_status') == 'OK' and NUMERIC.fullmatch((r.get('raw_text') or '').strip())]
            strict_numbers = [r for r in numbers if r.get('confidence') is not None and r['confidence'] >= .95]
            numeric_confidences = [r['confidence'] for r in numbers if r.get('confidence') is not None]
            best_number = max(numbers,key=lambda r:r.get('confidence') or 0) if numbers else None
            card_frames = sorted(r['source_frame'] for r in checked if r.get('scores',{}).get(seat) is not None
                and r['scores'][seat] >= detector['threshold'])
            card_frame_set = set(card_frames)
            card_triples = [[f,f+1,f+2] for f in card_frames if f+1 in card_frame_set and f+2 in card_frame_set]
            blocked = ('TABLE_DEAL_COMPLETE_NOT_CONFIRMED' if not deal else
                       'NUMERIC_STACK_CONFIDENCE_BELOW_0_95' if not triple and numbers and not strict_numbers else
                       'NO_STABLE_NUMERIC_TRIPLE_IN_CAPTURED_OCR_JOBS' if not triple else
                       'OCCUPANCY_EPOCH_AND_FORCED_BET_BINDING_UNPROVEN')
            seat_rows.append({'table':t,'hand_id':hand,'seat':seat,
                'session':events[0]['session_id'] if events else ref['events'][0]['session_id'],
                'occupied':True if deal and seat in deal['occupied_seats'] else None,
                'occupancy_player':None,'occupancy_epoch':None,
                'occupancy_evidence':'ALL_CONFIGURED_SEATS_POSITIVE_FROZEN_CARD_NCC' if deal else 'UNKNOWN',
                'dealing_cards_frame':events[0]['trigger_source_frame'] if events else None,
                'DEAL_COMPLETE_frame':deal_frame,'card_stable_frames':deal.get('card_stable_frames') if deal else None,
                'stable_numeric_stack_frames':triple['frames'] if triple else None,
                'displayed_stack':triple['displayed_stack'] if triple else None,
                'ANTE':None,'SB_BB':None,'START_STACK':None,
                'first_stack_changing_action_frame':None,
                'first_stack_changing_ocr_observation_frame':h.get('first_stack_changing_observation_frame'),
                'action_onset_is_proven':False,
                'PASS_UNKNOWN':'UNKNOWN','exact_blocker':blocked,
                'numeric_OCR_receipt_count':len(sr),
                'numeric_format_OCR_count':len(numbers),'strict_confidence_numeric_OCR_count':len(strict_numbers),
                'numeric_confidence_min':min(numeric_confidences) if numeric_confidences else None,
                'numeric_confidence_max':max(numeric_confidences) if numeric_confidences else None,
                'best_numeric_observation':{k:best_number.get(k) for k in ('source_frame','raw_text','confidence')} if best_number else None,
                'capture_budget_last_eligible_frame':deal_frame+64 if deal_frame is not None else None,
                'last_actual_stack_OCR_frame':max((r['source_frame'] for r in sr),default=None),
                'raw_text_counts':dict(collections.Counter(r.get('raw_text') for r in sr)),
                'missing_player_epoch':True,'missing_forced_bet_proof':True,
                'numeric_OCR_frames':[r['source_frame'] for r in sr],
                'numeric_OCR_status_counts':dict(collections.Counter(r.get('ocr_status') for r in sr)),
                'triple_source_receipts':[{k:r.get(k) for k in ('source_frame','source_pts','time_base','raw_text','confidence','ocr_status','pixel_sha256','status')}
                    for r in sr if triple and r['source_frame'] in triple['frames']],
                'player_epoch_proven':False,'forced_bets_proven':False})
            seat_rows[-1].update(partial_seat_card_positive_frames=len(card_frames),
                partial_seat_card_triple_count=len(card_triples),
                partial_seat_first_card_triple=card_triples[0] if card_triples else None,
                partial_seat_match_is_complete_table_occupancy=False)
    parity = []
    for name in ['physical_rows','raw_ocr_results','ordered_timeline','context_decisions','removed_duplicates','validated_hands','visual_evidence','hand_analysis']:
        old,new = BASE/(name+'.jsonl'),RUN/(name+'.jsonl')
        if not new.exists():
            parity.append({'stream':name,'status':'NOT_MEASURED_MISSING_ARTIFACT'})
            continue
        exact = sha(old) == sha(new)
        aa,bb = list(rows(old)),list(rows(new))
        semantic = len(aa) == len(bb) and all(clean(a)==clean(b) for a,b in zip(aa,bb))
        parity.append({'stream':name,'exact_bytes':exact,'semantic_equal_timing_only_exclusions':semantic,
            'baseline_rows':len(aa),'new_rows':len(bb),'old_sha256':sha(old),'new_sha256':sha(new)})
    numeric_total = sum(len(h.get('numeric_triples',{})) for h in live_hands.values())
    result = {'scope':'OFFICIAL_LIVE_5401_ON_CONNECTION_ONLY_NO_PROMOTION',
        'HANDS':len(expected),'CONFIGURED_SEAT_GROUPS':len(seat_rows),
        'OCCUPIED_SEATS':None,'OCCUPIED_SEATS_STATUS':'TOTAL_UNKNOWN_WHEN_ANY_MASK_UNPROVEN',
        'OCCUPIED_SEATS_POSITIVELY_PROVEN_IN_ACCEPTANCE_HANDS':sum(r['occupied'] is True for r in seat_rows),
        'DEAL_COMPLETE_PASS':sum(h['DEAL_COMPLETE']=='PASS' for h in perhand),
        'DEAL_COMPLETE_UNKNOWN':sum(h['DEAL_COMPLETE']=='UNKNOWN' for h in perhand),
        'DEAL_COMPLETE_FAIL':0,'STACK_TRIPLES_PASS':0,
        'RAW_NUMERIC_TRIPLES':sum(r['stable_numeric_stack_frames'] is not None for r in seat_rows),
        'STACK_TRIPLES_PASS_MEANING':'Existing CleanStackWindow accepted triples only; epoch/forced binding false keeps this at zero. RAW_NUMERIC_TRIPLES are separate live OCR diagnostics.',
        'PARTIAL_SEAT_CARD_TRIPLE_GROUPS':sum(bool(r['partial_seat_card_triple_count']) for r in seat_rows),
        'START_STACK_PASS':sum(h['START_STACK_PASS'] for h in perhand),
        'START_STACK_UNKNOWN_CONFIGURED_GROUPS':sum(r['PASS_UNKNOWN']=='UNKNOWN' for r in seat_rows),
        'CONFLICTS':sum(r.get('stage')=='DEAL_COMPLETE_FINAL_CONTEXT' for r in receipts),
        'TOP_BLOCKERS':dict(collections.Counter(r['exact_blocker'] for r in seat_rows)),
        'MARKERS':{'HITS':sum(bool(by_hand[k]) for k in expected),'MISSES':sum(not by_hand[k] for k in expected),
            'FALSE_POSITIVES':len(false),'DUPLICATES':sum(max(0,len(rr)-1) for rr in by_hand.values()),
            'SOURCE_BINDING_CONFLICTS':len(marker_conflicts),'INITIAL_PARTIALS':len(unexpected)-len(false)},
        'ALL_HANDS_INCLUDING_INITIAL_PARTIALS':{'DEAL_COMPLETE':sum(h.get('deal_complete_frame') is not None for h in live_hands.values()),
            'STACK_NUMERIC_TRIPLES':numeric_total,'START_STACK_PASS':(live or {}).get('start_stack_pass')},
        'INDEPENDENT_MATH':'NOT_MEASURED_NO_START_STACK_PASS',
        'parity':parity,'parity_PASS':all(r.get('semantic_equal_timing_only_exclusions') for r in parity),
        'timing_exclusions':sorted(TIMING),
        'NCC_RECEIPTS_EXACT':sha(BASE/'dealing_ncc_receipts.jsonl')==sha(RUN/'dealing_ncc_receipts.jsonl') if (RUN/'dealing_ncc_receipts.jsonl').exists() else None,
        'live_summary':live,'perhand':perhand,'seat_rows':seat_rows,
        'errors_retries':read(RUN/'errors_retries.json') if (RUN/'errors_retries.json').exists() else None,
        'default_flags':read(ROOT/'configs/dealing_start_stack.json'),
        'png_files_in_acceptance':len(list(RUN.rglob('*.png'))),
        'analysis_seconds':round(time.perf_counter()-started,6)}
    OUT.joinpath('connection_acceptance_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    lines=['# Live START_STACK connection: per configured seat', '',
        'A configured seat is not automatically occupied. TABLE-positive masks prove present cards; occupant epochs and forced bets remain UNKNOWN. Money OCR source frames are observations, not independently proved action onset.', '',
        '|table|hand_id|seat|occupancy/player|Dealing cards|DEAL_COMPLETE|three numeric frames|displayed|ANTE|SB/BB|START_STACK|first money action|money OCR observation|PASS/UNKNOWN|exact blocker|',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in seat_rows:
        show=lambda v:'UNKNOWN' if v is None else str(v)
        lines.append('|'+ '|'.join([f"T{r['table']:02d}",r['hand_id'],r['seat'],'cards present / player UNKNOWN' if r['occupied'] else 'UNKNOWN',
            show(r['dealing_cards_frame']),show(r['DEAL_COMPLETE_frame']),show(r['stable_numeric_stack_frames']),show(r['displayed_stack']),
            'UNKNOWN','UNKNOWN','UNKNOWN','UNKNOWN',show(r['first_stack_changing_ocr_observation_frame']),r['PASS_UNKNOWN'],r['exact_blocker']])+'|')
    OUT.joinpath('CONNECTION_304_SEATS.md').write_text('\n'.join(lines)+'\n','utf-8')
    print(json.dumps({k:result[k] for k in ('HANDS','DEAL_COMPLETE_PASS','DEAL_COMPLETE_UNKNOWN','STACK_TRIPLES_PASS','RAW_NUMERIC_TRIPLES','START_STACK_PASS','TOP_BLOCKERS','MARKERS','parity_PASS','NCC_RECEIPTS_EXACT','analysis_seconds')},ensure_ascii=False))


if __name__ == '__main__':
    main()
