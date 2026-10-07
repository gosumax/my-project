"""Persist root's visual review; never rewrite source OCR or production events."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/full_hh_production_20261006/semantic_unknown_review_v1'
# Reviewed from all seven exact-frame contact sheets, 52 source rows.
LABELS = {
    'T01:E0001': ('INITIAL_TOP_FRAGMENT', 'Only clipped top glyphs; no new hand header.'),
    'T01:E0051': ('FOLD', 'Visible folds; OCR lost spaces.'),
    'T01:E0252': ('BOUNTY_AWARD', 'Award 100 p. for busted player(s), not chip payout.'),
    'T01:E0264': ('RESEATING_NOTICE', 'Please wait while players are reseating; no trailing digit.'),
    'T02:E0001': ('INITIAL_TOP_FRAGMENT', 'Clipped prior fold at video start.'),
    'T02:E0038': ('NO_SHOW', "Visible doesn't show cards."),
    'T02:E0132': ('REBUY_WAIT', 'Waiting for player to rebuy.'),
    'T02:E0133': ('REBUY', 'Player makes rebuy(s), gets 150000 chips.'),
    'T03:E0030': ('BOUNTY_AWARD', 'Award 200 p. for busted players, not chip payout.'),
    'T03:E0173': ('NO_SHOW', 'Two adjacent rows: does not show cards.'),
    'T04:E0001': ('INITIAL_TOP_FRAGMENT', 'Clipped prior uncalled return; visible tail 2985 back.'),
    'T04:E0134': ('FOLD', 'Fold visible; exact l/1 actor identity remains unproven.'),
    'T04:E0164': ('FOLD', 'Fold visible; exact l/1 actor identity remains unproven.'),
    'T04:E0180': ('NO_SHOW', "Visible doesn't show cards."),
    'T05:E0001': ('INITIAL_TOP_FRAGMENT', 'Clipped ante line at video start.'),
    'T05:E0008': ('FOLD', 'twixspb folds; initial context.'),
    'T05:E0085': ('FOLD', 'twixspb folds; spaces missing in source OCR.'),
    'T05:E0091': ('NO_SHOW', "Visible doesn't show cards."),
    'T05:E0142': ('SMALL_BLIND', 'Visible posts Small Blind, adjacent 350; source OCR lost one l.'),
    'T05:E0162': ('BOUNTY_AWARD', 'Award 100 p. for busted player, not chip payout.'),
    'T05:E0180': ('JOIN', 'One joined-the-table message; repeated table tail across frames 9677 and 9680 is an audit stitching duplicate.'),
    'T07:E0001': ('INITIAL_CONTINUATION', '5000 above initial Dealing cards; no actor or action in this row.'),
    'T07:E0012': ('TURN', 'Source says Dealing Turn plus four board cards; candidate Dealing cards is wrong.'),
    'T07:E0094': ('CALL', 'Visible calls 10000; OCR lost space after calls.'),
    'T07:E0201': ('REBUY_WAIT', 'Waiting for player to rebuy.'),
    'T07:E0202': ('REBUY', 'Player makes rebuy(s), gets 150000 chips.'),
    'T08:E0001': ('INITIAL_CONTINUATION', 'from the main pot above an already present initial header.'),
    'T08:E0018': ('RETURN', 'Two adjacent rows: gets uncalled bet 2500 back.'),
    'T08:E0049': ('REBUY_WAIT', 'Waiting for player to rebuy.'),
    'T08:E0073': ('NO_SHOW', "Visible doesn't show cards."),
    'T08:E0099': ('REBUY_WAIT', 'Waiting for player to rebuy.'),
    'T08:E0202': ('NO_SHOW', "Visible doesn't show cards."),
}


def main():
    source = OUT / 'receipts.json'
    receipt = json.loads(source.read_text(encoding='utf-8'))
    assert set(LABELS) == {e['event_id'] for e in receipt['events']}
    rows = []
    for event in receipt['events']:
        label, note = LABELS[event['event_id']]
        rows.append({
            'event_id': event['event_id'], 'table_id': event['table_id'],
            'frame': event['frame'], 'review_label': label, 'note': note,
            'new_inside_video_hand_start': False,
            'source_rows': [{k: r[k] for k in (
                'sequence_row_id', 'source_first_frame', 'source_span',
                'source_frame_sha256', 'crop_sha256')}
                for r in event['source_rows']],
        })
    result = {
        'schema': 'root.manual_unknown_source_review.v1',
        'authority': 'INDEPENDENT_AUDIT_ONLY; RAW_OCR_AND_PRODUCTION_UNCHANGED',
        'method': 'Root visually inspected every row of seven exact-frame contact sheets.',
        'receipt_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'contact_sheets': [{
            'path': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()
        } for p in sorted(OUT.glob('unknown_events_*.png'))],
        'reviewed_events': len(rows), 'reviewed_source_rows': receipt['source_row_count'],
        'extra_hand_starts_found': 0,
        'counts': dict(Counter(r['review_label'] for r in rows)),
        'physical_semantic_universe_exhaustive': False,
        'limitation': 'This resolves only these 32 candidates; it does not certify all audit events or source stitching.',
        'events': rows,
    }
    (OUT / 'root_visual_verdict.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in (
        'reviewed_events', 'reviewed_source_rows', 'extra_hand_starts_found', 'counts')}))


if __name__ == '__main__':
    main()
