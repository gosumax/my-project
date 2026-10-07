"""Copy a bounded native source context; do not infer any nickname binding."""
import hashlib
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/full_hh_production_20261006/short_v1'
OUTPUT = ROOT / 'tests/fixtures/roster_prefix_smoke19.json'
TABLE, HAND = 5, '2878365559'


def rows(name):
    with (SOURCE / name).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    global SOURCE,OUTPUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();SOURCE=args.source.resolve();OUTPUT=args.output.resolve()
    summary = json.loads((SOURCE / 'seat_forced_summary.json').read_text('utf-8'))
    snapshot = next(s for s in summary['forced_snapshots']
                    if (s['table'], s['hand_id']) == (TABLE, HAND))
    session = snapshot['session']
    def context(r):
        return (r.get('table_id', r.get('table')) == TABLE and
                r.get('session_id', r.get('session')) == session and r.get('hand_id') == HAND)
    names = ['seat_forced_inputs.jsonl', 'seat_forced_summary.json',
             'deal_complete_receipts.jsonl', 'dealing_cards_triggers.jsonl']
    fixture = dict(schema='native-roster-prefix-fixture-v1',
        authority='UNMODIFIED_SOURCE_ONLY; NO_NEW_BINDING_AUTHORITY',
        context=dict(table=TABLE, session=session, hand_id=HAND),
        source_files={n: dict(path=str((SOURCE/n).relative_to(ROOT)),
                             sha256=hashlib.sha256((SOURCE/n).read_bytes()).hexdigest()) for n in names},
        forced_snapshot=snapshot,
        input_rows=[r for r in rows('seat_forced_inputs.jsonl') if context(r)],
        deal_receipts=[r for r in rows('deal_complete_receipts.jsonl') if context(r)],
        triggers=[r for r in rows('dealing_cards_triggers.jsonl') if context(r)],
        state_receipts=[r for r in summary['state_receipts'] if context(r)],
        unresolved_state_events=[r for r in summary['unresolved_state_events']
            if r.get('table') == TABLE and r.get('session') == session],
        old_binding_receipts=[r for r in summary['historical_binding_receipts'] if context(r)],
        old_binding_blockers=[r for r in summary['historical_binding_blockers'] if context(r)])
    OUTPUT.write_text(json.dumps(fixture, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(path=str(OUTPUT), bytes=OUTPUT.stat().st_size,
        input_rows=len(fixture['input_rows']), deal_receipts=len(fixture['deal_receipts']),
        old_bindings=len(fixture['old_binding_receipts']), old_blockers=fixture['old_binding_blockers']),
        ensure_ascii=False))


if __name__ == '__main__':
    main()
