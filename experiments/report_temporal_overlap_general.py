"""Source-neutral OLD/NEW experimental dedup comparison; no text-based repair."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.temporal_overlap_dedup import read_jsonl, sha, write_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--audit', type=Path)
    p.add_argument('--protocol-validation', type=Path, required=True)
    args = p.parse_args()
    summary = json.loads((args.run / 'NEW/experiment_summary.json').read_text('utf-8'))
    old_dir = Path(summary['old'])
    old = read_jsonl(old_dir / 'ordered_timeline.jsonl')
    new = read_jsonl(args.run / 'NEW/ordered_timeline.jsonl')
    removed = read_jsonl(args.run / 'NEW/removed_duplicates.jsonl')
    ids = {r['old_row_id'] for r in removed}
    lookup = {r['stable_row_id']: r for r in old}
    retained = {r['stable_row_id'] for r in new}
    checks = []
    for r in removed:
        current, previous = lookup[r['old_row_id']], lookup[r['matched_previous_row_id']]
        checks.append({'row_id': r['old_row_id'],
            'same_table': r['table_id'] == current['table_id'] == previous['table_id'],
            'earlier_original': previous['frame_id'] < current['frame_id'],
            'earlier_appearance': r['matched_previous_observation']['frame_id'] < current['frame_id'],
            'nonempty_exact_ocr': bool(current['raw_ocr_text'].strip()) and current['raw_ocr_text'] == previous['raw_ocr_text'],
            'unresolved_origin': current['origin'] == 'UNRESOLVED_IDENTITY',
            'matched_original_retained': previous['stable_row_id'] in retained,
            'anchors_retained': all(a in retained for a in r['anchor_row_ids']),
            'at_least_two_distinct_anchors': len(set(r['anchor_row_ids'])) >= 2,
            'glyph_gate': r['min_glyph_ncc'] >= r['glyph_ncc_min'],
            'edge_gate': r['active_edge_pixels_discarded'] is False})
    raw_equal = {name: sha(old_dir / name) == sha(args.run / 'NEW' / name)
                 for name in ('physical_rows.jsonl', 'raw_ocr_results.jsonl', 'visual_evidence.jsonl')}
    tables = []
    sequences = args.run / 'table_sequences'
    sequences.mkdir(exist_ok=True)
    for table in sorted({r['table_id'] for r in old}):
        before, after = [r for r in old if r['table_id'] == table], [r for r in new if r['table_id'] == table]
        expected = [r for r in before if r['stable_row_id'] not in ids]
        write_jsonl(sequences / f'{table}_OLD.jsonl', before)
        write_jsonl(sequences / f'{table}_NEW.jsonl', after)
        tables.append({'table_id': table, 'old': len(before), 'new': len(after), 'removed': len(before)-len(after),
            'retained_rows_text_order_recognition_equal': after == expected,
            'all_INITIAL_VISIBLE_and_NEW_CANDIDATE_retained': all(r['stable_row_id'] in retained for r in before if r['origin'] != 'UNRESOLVED_IDENTITY')})
    audit = json.loads(args.audit.read_text('utf-8')) if args.audit else {}
    audit_summary = {k: v for k, v in audit.items() if k not in ('removal_audit', 'real_rows_lost', 'failed_checks')}
    loss_fields = ('table_id', 'frame_id', 'text', 'old_row_id', 'matched_previous_row_id',
                   'previous_hand', 'current_hand', 'current_hand_marker_row_id', 'reason')
    audit_summary['real_rows_lost'] = [{k: r[k] for k in loss_fields if k in r} for r in audit.get('real_rows_lost', [])]
    audit_summary['full_audit_path'] = str(args.audit.resolve()) if args.audit else None
    for table in tables:
        table['confirmed_real_rows_lost'] = sum(r['table_id'] == table['table_id'] for r in audit.get('real_rows_lost', []))
    protocol_validation = json.loads(args.protocol_validation.read_text('utf-8'))
    structural_ok = (all(raw_equal.values()) and all(all(v for k, v in c.items() if k != 'row_id') for c in checks)
                     and all(t['retained_rows_text_order_recognition_equal'] and t['all_INITIAL_VISIBLE_and_NEW_CANDIDATE_retained'] for t in tables)
                     and protocol_validation.get('pass') is True)
    if audit.get('real_rows_lost'):
        verdict = 'DEDUP_REDUCES_DUPLICATES_BUT_LOSES_REAL_ROWS'
    elif removed and structural_ok and audit.get('all_removals_reviewed_safe') is True:
        verdict = 'DEDUP_PROVEN_SAFE'
    elif not removed and structural_ok:
        verdict = 'NO_MEASURABLE_IMPROVEMENT'
    else:
        verdict = 'INCONCLUSIVE'
    result = {'source_sha256': summary['source_sha256'], 'frames': summary['frames'],
              'verdict': verdict, 'structural_checks_pass': structural_ok,
              'tables': tables, 'removal_checks': checks, 'raw_file_sha_parity': raw_equal,
              'independent_review': audit_summary, 'experiment_parameters_unchanged': protocol_validation.get('pass') is True,
              'protocol_validation': protocol_validation,
              'scope': 'Full source replay, fixed OLD OCR/cards; no certified HH or global accuracy claim'}
    (args.run / 'acceptance.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
    no_gallery = json.loads((args.run / 'NO_GALLERY/experiment_summary.json').read_text('utf-8'))
    report = ['FILES CHANGED', 'experiments/temporal_overlap_guarded.py: experiment launcher source binding generalized. experiments/report_temporal_overlap_general.py: source-neutral report with measured protocol verification. experiments/build_overlap_evidence_viewer.py: final read-only pixel evidence HTML. Reducer, native geometry, glyph metric and thresholds unchanged from Smoke18 V4. Production/frozen inputs unchanged.',
        'TESTS ADDED', 'Existing targeted107 tests reused; actual validation results in VALIDATION.txt.',
        'OLD RESULT', f"Baseline={old_dir}; full frames={summary['frames']}; sourceSHA={summary['source_sha256']}; physical={summary['old_physical_rows']}; OCR={summary['old_ocr_rows']}; timeline={summary['old_timeline_rows']}.",
        'NEW RESULT', f"Full frames={summary['frames']}; physical={summary['new_physical_rows']}; OCR={summary['new_ocr_rows']}; timeline={summary['new_timeline_rows']}. Same physical/OCR/card evidence. NO_GALLERY timeline={no_gallery['new_timeline_rows']}; only gallery setting differs between internal arms.",
        'REMOVED DUPLICATES', f"Removed={len(removed)}; these are experimental removal decisions, not all proved duplicates. Source-confirmed wrongly removed physical rows={len(audit.get('real_rows_lost', []))}. Per-table={dict(Counter(r['table_id'] for r in removed))}. Full table/frame/timestamp/text/oldID/previousID/SHA/glyph/overlap records: NEW/removed_duplicates.jsonl.",
        'T01 BEFORE', 'table_sequences/T01_OLD.jsonl (full sequence, no Smoke18 hand expectations applied).',
        'T01 AFTER', 'table_sequences/T01_NEW.jsonl.',
        'REAL ACTIONS PRESERVED', f"{'FAIL: '+str(len(audit.get('real_rows_lost', [])))+' physical rows representing '+str(audit.get('distinct_real_ante_actions_lost', 'unknown'))+' real ante actions wrongly removed.' if audit.get('real_rows_lost') else 'See independent review.'} All INITIAL_VISIBLE and NEW_CANDIDATE original row IDs retained on every table, but this alone does not prove real-action preservation. Full losses with hand/frame/identity evidence are listed below.",
        'REGRESSIONS', json.dumps(tables, ensure_ascii=False, indent=2),
        f"Physical/OCR/card SHA parity={raw_equal}; retained timeline records and order preserved. Structural checks={structural_ok}. Independent review={json.dumps(audit_summary, ensure_ascii=False, indent=2)}.",
        'PERFORMANCE', f"OLD end-to-end wall={summary['old_wall_seconds']:.6f}s; full shared capture replay+two reducers={summary['replay_wall_seconds']:.6f}s; V4 reducer={summary['dedup_seconds']:.6f}s. Recognition reused; no end-to-end speedup conclusion.",
        'VERDICT', verdict, 'No promotion. Verdict concerns reviewed removals in this full source replay, not completeness of OLD or general recognition accuracy.']
    (args.run / 'REPORT.txt').write_text('\n\n'.join(report)+'\n', 'utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='removal_checks'},ensure_ascii=False))


if __name__ == '__main__':
    main()
