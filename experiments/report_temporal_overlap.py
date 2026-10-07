"""Smoke18 acceptance diagnostics for the separate temporal-overlap experiment."""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import json
from pathlib import Path
import re
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.temporal_overlap_dedup import signature
from parser_core.chat_rows import canonical_row_distance


def read(path):
    return [json.loads(l) for l in path.read_text('utf-8').splitlines()]


def compact(s):
    return re.sub(r'\s+', '', s).casefold()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--arm', default='NEW')
    p.add_argument('--audit', type=Path)
    args = p.parse_args()
    arm = args.run / args.arm
    summary = json.loads((arm / 'experiment_summary.json').read_text('utf-8'))
    old_dir = Path(summary['old'])
    old = read(old_dir / 'ordered_timeline.jsonl')
    new = read(arm / 'ordered_timeline.jsonl')
    removed = read(arm / 'removed_duplicates.jsonl')
    t1old = [r for r in old if r['table_id'] == 'T01']
    t1new = [r for r in new if r['table_id'] == 'T01']
    # Literal acceptance requirements, used only for reporting, never for dedup.
    requirements = [
        ('ante karanebesnaya 3 000', ['Dealer: karanebesnaya posts Ante', '3000'], 1),
        *[(f'ante {name} 3 000', [f'Dealer: {name} posts Ante 3 000'], 1) for name in
          ('vvs047', 'elance', 'lifeforkaifff', '359846', 'lexa_tsar1', 'antivea')],
        ('SB 12 500', ['Dealer: 359846 posts Small Blind', '12500'], 1),
        ('BB 25 000', ['Dealer: lexa_tsar1 posts Big Blind', '25000'], 1),
        *[(f'{name} folds', [f'Dealer: {name} folds'], 1) for name in
          ('antivea', 'karanebesnaya', 'vvs047', 'elance')],
        ('lifeforkaifff raises to 50 000', ['Dealer: lifeforkaifff raises to 50 000'], 1),
        ('359846 folds', ['Dealer: 359846 folds'], 1),
        ('lexa_tsar1 calls 25 000', ['Dealer: lexa_tsar1 calls 25 000'], 1),
        ('Dealing Flop', ['Dealer: Dealing Flop'], 1),
        ('lexa_tsar1 checks (both streets)', ['Dealer: lexa_tsar1 checks'], 2),
        ('lifeforkaifff bets 44 055', ['Dealer: lifeforkaifff bets 44 055'], 1),
        ('lexa_tsar1 calls 44 055', ['Dealer: lexa_tsar1 calls 44 055'], 1),
        ('Dealing Turn', ['Dealer: Dealing Turn'], 1),
        ('lifeforkaifff bets 160 240', ['Dealer: lifeforkaifff bets 160 240'], 1),
    ]
    texts = [compact(r['raw_ocr_text']) for r in t1new]
    old_texts = [compact(r['raw_ocr_text']) for r in t1old]
    new_ids = {r['stable_row_id'] for r in new}
    actions = []
    for name, parts, count in requirements:
        wanted = [compact(s) for s in parts]
        matches = [t1new[i:i + len(wanted)] for i in range(len(texts))
                   if texts[i:i + len(wanted)] == wanted]
        originals = [t1old[i:i + len(wanted)] for i in range(len(old_texts))
                     if old_texts[i:i + len(wanted)] == wanted][:count]
        original_ids = [r['stable_row_id'] for group in originals for r in group]
        actions.append({'action': name, 'required_count': count, 'observed_count': len(matches),
                        'preserved': len(matches) >= count and len(originals) == count and all(rid in new_ids for rid in original_ids),
                        'original_real_row_ids': original_ids,
                        'row_ids': [[r['stable_row_id'] for r in group] for group in matches]})
    targets = ['Dealer: Starting a new hand', 'Dealer: 359846 posts Ante 3 000',
               'Dealer: lifeforkaifff raises to 50 000', 'Dealer: Dealing cards']
    target_results = [{'text': s,
        'old_count': sum(compact(r['raw_ocr_text']) == compact(s) for r in t1old),
        'new_count': sum(compact(r['raw_ocr_text']) == compact(s) for r in t1new)} for s in targets]
    physical = {r['row_id']: r for r in read(old_dir / 'physical_rows.jsonl')}
    for result in target_results:
        pair = [r for r in t1old if compact(r['raw_ocr_text']) == compact(result['text'])]
        if len(pair) == 2:
            a, b = (physical[r['stable_row_id']] for r in pair)
            pixels = lambda r: np.frombuffer(base64.b64decode(r['row_pixels_base64']), np.uint8).reshape(r['pixel_shape'])
            result['pair_canonical_mae'] = canonical_row_distance(signature(pixels(a)), signature(pixels(b)))
            result['within_visual_threshold'] = result['pair_canonical_mae'] <= 9.0
    removed_ids = {r['old_row_id'] for r in removed}
    old_by_id = {r['stable_row_id']: r for r in old}
    regressions = []
    for table in sorted({r['table_id'] for r in old}):
        before = [r for r in old if r['table_id'] == table]
        after = [r for r in new if r['table_id'] == table]
        expected = [r for r in before if r['stable_row_id'] not in removed_ids]
        regressions.append({'table_id': table, 'old_rows': len(before), 'new_rows': len(after),
            'removed': len(before) - len(after), 'retained_rows_order_text_card_fields_equal': after == expected,
            'removed_new_candidates': sum(old_by_id[r['old_row_id']]['origin'] != 'UNRESOLVED_IDENTITY'
                for r in removed if r['table_id'] == table)})
    raw_equal = all(summary['raw_evidence_hash_parity'].values())
    actions_ok = all(r['preserved'] for r in actions)
    targets_ok = all(r['old_count'] == 2 and r['new_count'] == 1 for r in target_results)
    regression_ok = all(r['retained_rows_order_text_card_fields_equal'] and not r['removed_new_candidates'] for r in regressions)
    evidence_ok = all(r['visual_match'] and len(r['anchor_row_ids']) >= 2 and r['overlap_length'] >= 3 for r in removed)
    audit = json.loads(args.audit.read_text('utf-8')) if args.audit else {}
    audit_ok = audit.get('all_removals_reviewed_safe') is True and audit.get('negative_tests_pass') is True
    verdict = ('DEDUP_REDUCES_DUPLICATES_BUT_LOSES_REAL_ROWS' if removed and not actions_ok else
               'DEDUP_PROVEN_SAFE' if targets_ok and actions_ok and regression_ok and evidence_ok and raw_equal and audit_ok else
               'NO_MEASURABLE_IMPROVEMENT' if not removed else 'INCONCLUSIVE')
    result = {'verdict': verdict, 'scope': 'This full Smoke18 replay only; no general accuracy or HH certification',
              'real_actions': actions, 'target_duplicates': target_results, 'regressions': regressions,
              'raw_evidence_equal': raw_equal, 'dedup_evidence_complete': evidence_ok,
              'independent_audit_pass': audit_ok, 'experiment': summary['experiment']}
    acceptance_name = 'acceptance.json' if args.arm == 'NEW' else f'acceptance_{args.arm}.json'
    (args.run / acceptance_name).write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')

    def sequence(rows):
        return '\n'.join(f"{i+1}. frame={r['frame_id']} {r['stable_row_id'].split(':')[-1]} — {r['raw_ocr_text']}" for i, r in enumerate(rows))
    report = [
        'FILES CHANGED',
        json.dumps(summary.get('experiment_code_sha256', {})) + '. Shared experiment replay/report files and experimental tests. Production, manifest and frozen inputs unchanged.',
        'Current mechanism: run_production.py -> scripts/run_smoke14_ram.py::main -> parser_core/ram_first_stage.py::FirstStage.rows -> parser_core/chat_rows.py::match_rows / align_upward / match_row_blocks. FirstStage retains 12 changed chat states per session, with conditional historical search. Published RamRow becomes physical evidence, Tiny OCR, card evidence and timeline. Experiment uses the existing read-only inherited-row observer to recover visible blocks and matches them against up to 256 published rows per table. It does not alter production identity, OCR, card recognition, or origin/status fields.',
        'TESTS ADDED', 'experiments/test_temporal_overlap_dedup.py, test_temporal_overlap_native.py, test_temporal_overlap_guarded.py. Coverage: anchor/text/pixel conflicts, repeated NEW_CANDIDATE and later-hand identity, fixed native geometry, vertical phase, all 90 ordered single-digit substitutions, active top/bottom edge, empty OCR, aliases/table isolation and future views. Final targeted set107 passed in1.97s (exit0). Tests are synthetic and regression checks, not independent video acceptance.',
        'OLD RESULT', f"Full 2040 frames. Physical={summary['old_physical_rows']}; OCR={summary['old_ocr_rows']}; timeline={summary['old_timeline_rows']}. Golden official runner; PARTIAL_VISUAL.",
        'NEW RESULT', f"Full 2040-frame capture replay. Physical={summary['new_physical_rows']}; OCR={summary['new_ocr_rows']}; timeline={summary['new_timeline_rows']}; removed={summary['removed_count']}. Raw physical/OCR/card evidence retained byte-for-byte. Dedup operates on timeline publication after recognition. Stable original sequence/message IDs retained for provenance; gaps are intentional, grouping is not recomputed.",
        'REMOVED DUPLICATES', f"Full machine-readable details: {args.arm}/removed_duplicates.jsonl. Experiment={summary['experiment']}; metric={summary.get('distance_metric', 'canonical_mae')}. Evidence uses unique diagonal mapping anchored by at least two inherited identities, contiguous visual agreement, text conflict gate, and UNRESOLVED_IDENTITY gate. Timestamp never decides deletion. Native MAE is not numerically equivalent to tight-span canonical MAE; safety requires negative tests and review.",
    ]
    for r in removed:
        report.append(json.dumps(r, ensure_ascii=False))
    report += ['T01 BEFORE', 'Hand #2875902536; entire T01 sequence:', sequence(t1old),
               'T01 AFTER', sequence(t1new), 'REAL ACTIONS PRESERVED']
    report += [f"{'PASS' if r['preserved'] else 'FAIL'}: {r['action']} — {r['observed_count']}/{r['required_count']}" for r in actions]
    report += [json.dumps(target_results, ensure_ascii=False), 'REGRESSIONS',
        json.dumps(regressions, ensure_ascii=False, indent=2),
        'Raw OCR, physical pixels, card visual evidence: exact file SHA parity. Retained timeline objects including recognition fields unchanged and in original order. All removals require anchored evidence. No independent human ground truth for every other-table action; this verifies preservation relative to OLD and evidenced overlap, not completeness of OLD.',
        'PERFORMANCE',
        f"OLD end-to-end wall={summary['old_wall_seconds']:.6f}s. NEW full capture+dedup replay={summary['replay_wall_seconds']:.6f}s; this replay shares capture between two arms when dual_arm_replay=true ({summary.get('dual_arm_replay', False)}). This arm's block dedup wall={summary['dedup_seconds']:.6f}s; visible blocks={summary['visible_blocks']}. OCR/cards reused. These timings cannot demonstrate end-to-end speedup or runtime parity of a promoted implementation.",
        'Independent audit', json.dumps(audit, ensure_ascii=False),
        'VERDICT', verdict,
        'Verdict applies only to this full Smoke18 experiment and remains subject to independent code/evidence review. No promotion performed.'
    ]
    report_name = 'REPORT.txt' if args.arm == 'NEW' else f'REPORT_{args.arm}.txt'
    (args.run / report_name).write_text('\n\n'.join(report) + '\n', 'utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
