"""Read-only source-bound transport/quality audit, independent of segment counts."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.poker_reconstruction.timeline import parse_text


def read(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def table(value):
    return int(str(value).removeprefix('T'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--production', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    verified = json.loads((args.replay/'verified_704_after.json').read_text('utf-8'))
    replay_rows = {r['row_id']: r for r in read(args.replay/'stage_rows.jsonl')}
    physical = read(args.production/'physical_rows.jsonl')
    ocr = read(args.production/'raw_ocr_results.jsonl')
    timeline = read(args.production/'ordered_timeline.jsonl')
    hands = read(args.production/'hand_histories.jsonl')
    records = read(args.production/'logical_hand_analysis.jsonl')
    physical_by_id = defaultdict(list)
    ocr_by_id = defaultdict(list)
    timeline_by_id = defaultdict(list)
    for row in physical:
        physical_by_id[row['row_id']].append(row)
    for row in ocr:
        ocr_by_id[row['row_id']].append(row)
    for row in timeline:
        timeline_by_id[row['stable_row_id']].append(row)
    totals = defaultdict(Counter)
    events = []
    kinds = dict(HAND_START='START_HAND', ANTE='ANTE', SMALL_BLIND='SB', BIG_BLIND='BB',
                 DEAL='DEAL', FOLD='FOLD', UNCALLED_RETURN='RETURN', SHOW='SHOW', WIN='WIN',
                 NO_SHOW='INFO', PLAYER_LEAVE='INFO')
    for event in verified:
        ids = list(dict.fromkeys(rid for matches in event['supplemental_rows'] for rid in matches))
        admitted = bool(ids) and event['fallback_recoverable']
        receipts = []
        for rid in ids:
            expected = replay_rows.get(rid)
            rows = physical_by_id[rid]
            exact = bool(expected) and len(rows) == 1 and all(
                rows[0].get(k) == expected[k] for k in
                ('table_id', 'frame_id', 'source_span_index', 'y1', 'epoch', 'source_pts',
                 'time_base_num', 'time_base_den', 'row_pixel_sha256', 'pixel_shape'))
            admitted = admitted and exact
            receipts.append(dict(row_id=rid, exact_source_row=exact,
                                 physical_count=len(rows), ocr_count=len(ocr_by_id[rid]),
                                 timeline_count=len(timeline_by_id[rid])))
        ocr_ok = admitted and all(len(ocr_by_id[rid]) == 1 and
            ocr_by_id[rid][0].get('raw_ocr_response', {}).get('status') == 'OK' for rid in ids)
        timeline_ok = ocr_ok and all(len(timeline_by_id[rid]) == 1 and
            timeline_by_id[rid][0]['row_pixel_sha256'] == replay_rows[rid]['row_pixel_sha256'] for rid in ids)
        observed_rows = sorted((timeline_by_id[rid][0] for rid in ids if len(timeline_by_id[rid]) == 1),
                               key=lambda r: r['row_index'])
        observed_text = ' '.join(r['raw_ocr_text'] for r in observed_rows)
        parsed = parse_text(observed_text, {'rows': observed_rows})
        recognized = bool(timeline_ok and parsed.kind == kinds[event['kind']])
        if event['kind'] == 'HAND_START':
            recognized = recognized and parsed.source.get('parsed_hand_id') == event['hand_number']
        # INFO alone is not sufficient to establish the requested semantic type.
        if event['kind'] == 'PLAYER_LEAVE':
            recognized = recognized and 'has left the table' in observed_text.lower()
        if event['kind'] == 'NO_SHOW':
            recognized = recognized and any(s in observed_text.lower() for s in ("doesn't show", 'does not show'))
        counter = totals[event['kind']]
        counter.update(physical_verified=1, admitted=int(admitted), OCR=int(ocr_ok),
                       timeline=int(timeline_ok), semantic_type_recognized=int(recognized))
        events.append(dict(event_id=event['event_id'], table=event['table_id'], frame=event['frame'],
            kind=event['kind'], source_chat_sha256=event.get('source_chat_sha256'),
            source_span_provenance=event['source_span_provenance'], row_receipts=receipts,
            admitted=bool(admitted), OCR=bool(ocr_ok), timeline=bool(timeline_ok),
            semantic_type_recognized=bool(recognized), raw_ocr_text=observed_text,
            authority='EXACT_REPLAY_SOURCE_AND_LIVE_PIXEL_HASH; TYPE_CHECK_IS_NOT_FULL_EVENT_CORRECTNESS'))
    denominator = json.loads((ROOT/'runs/full_hh_production_20261006/denominator_review/reviewer_verdict.json').read_text('utf-8'))
    hand_audit = []
    for source in denominator['rows']:
        matched = [h for h in hands if table(h['table']) == source['table'] and
                   h['hand_number'] == source['hand']]
        hand_audit.append(dict(source=source, matching_hh_segments=[h['segment_id'] for h in matched],
            exported_statuses=[h['status'] for h in matched],
            certified_candidates=sum(h['certified'] for h in matched),
            independent_full_action_review='NOT_YET_PROVEN'))
    total = sum(c['physical_verified'] for c in totals.values())
    delivered = sum(c['timeline'] for c in totals.values())
    legacy = [r for r in physical if ':R' in r['row_id']]
    summary = dict(known_704={k:dict(v) for k,v in totals.items()},
        known_704_timeline=delivered, known_704_physical=total,
        known_704_transport_recall=delivered/total,
        PHYSICAL_CHAT_RECALL=None,
        PHYSICAL_CHAT_RECALL_status='704_VERIFIED_LOWER_BOUND_ONLY; FULL_SEMANTIC_UNIVERSE_NOT_PROVEN',
        physical_complete_hands_reviewed=denominator['PHYSICAL_COMPLETE_HANDS'],
        physical_denominator_scope=denominator['scope'], END_PARTIAL_reviewed=denominator['END_PARTIAL'],
        exported_statuses=dict(Counter(h['status'] for h in hands)),
        exported_certified_hh=sum(h['certified'] for h in hands),
        independently_correct_hh=0 if not any(h['certified'] for h in hands) else None,
        HH_COMPLETE_RATE=0.0 if not any(h['certified'] for h in hands) else None,
        HH_COMPLETE_RATE_scope='REVIEWED_57_COMPLETE_HAND_INVENTORY; EXHAUSTIVE_DENOMINATOR_NOT_PROVEN',
        logic_statuses=dict(Counter(r['diagnostics']['logic_status'] for r in records)),
        math_statuses=dict(Counter(r['diagnostics']['math_status'] for r in records)),
        unknown_actions=sum(m['action']=='UNKNOWN_EVENT' for r in records for m in r['observed_hand']['messages']),
        physical_rows=len(physical), OCR_rows=len(ocr), timeline_rows=len(timeline),
        legacy_admitted=len(legacy), legacy_OCR=sum(len(ocr_by_id[r['row_id']])==1 for r in legacy),
        legacy_timeline=sum(len(timeline_by_id[r['row_id']])==1 for r in legacy),
        duplicate_physical_ids=sum(len(v)-1 for v in physical_by_id.values()),
        duplicate_OCR_ids=sum(len(v)-1 for v in ocr_by_id.values()),
        png_count=sum(1 for _ in args.production.rglob('*.png')))
    for name, payload in [('known_704_transport.json',events),('physical_hand_matches.json',hand_audit),
                          ('summary.json',summary)]:
        (args.output/name).write_text(json.dumps(payload,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(summary,ensure_ascii=True))


if __name__ == '__main__':
    main()
