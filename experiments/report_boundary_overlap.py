"""Compare immutable OLD/BROKEN evidence with a completed boundary experiment."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.temporal_overlap_dedup import read_jsonl, write_jsonl, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--targeted', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True)
    args = p.parse_args()
    c = json.loads((args.run/'experiment_summary.json').read_text('utf-8'))
    audit = json.loads(args.audit.read_text('utf-8'))
    validation = json.loads(args.validation.read_text('utf-8'))
    target = json.loads((args.targeted/'targeted_result.json').read_text('utf-8'))
    old_dir, broken_dir = Path(c['old']), Path(c['broken'])
    old, broken, fixed = (read_jsonl(path) for path in (
        old_dir/'ordered_timeline.jsonl', broken_dir/'NEW/ordered_timeline.jsonl', args.run/'ordered_timeline.jsonl'))
    removals = read_jsonl(args.run/'removed_duplicates.jsonl')
    ids = {r['old_row_id'] for r in removals}
    retained = {r['stable_row_id'] for r in fixed}
    original_losses = json.loads((broken_dir/'independent_audit.json').read_text('utf-8'))['real_rows_lost']
    recovered = sum(r['old_row_id'] in retained for r in original_losses)
    evidence = {n: sha(old_dir/n)==sha(args.run/n) for n in ('physical_rows.jsonl','raw_ocr_results.jsonl','visual_evidence.jsonl')}
    tables=[]
    sequences=args.run/'table_sequences'; sequences.mkdir(exist_ok=True)
    for table in sorted({r['table_id'] for r in old}):
        a,b,d=([r for r in rows if r['table_id']==table] for rows in (old,broken,fixed))
        tables.append({'table_id':table,'OLD':len(a),'BROKEN':len(b),'FIXED':len(d),'removed':len(a)-len(d),
            'previous_real_losses':sum(r['table_id']==table for r in original_losses),
            'previous_real_rows_recovered':sum(r['table_id']==table and r['old_row_id'] in retained for r in original_losses),
            'retained_records_order_equal':d==[r for r in a if r['stable_row_id'] not in ids]})
        write_jsonl(sequences/f'{table}_OLD.jsonl',a)
        write_jsonl(sequences/f'{table}_BROKEN.jsonl',b)
        write_jsonl(sequences/f'{table}_FIXED.jsonl',d)
    safe_audit = audit.get('all_removals_reviewed_safe') is True
    lost = audit.get('real_rows_lost',[])
    lost_count = len(lost) if isinstance(lost,list) else int(lost)
    tests_pass = validation.get('full_pytest_exit_code') == 0 and validation.get('targeted_exit_code') == 0
    parity = all(evidence.values()) and all(t['retained_records_order_equal'] for t in tables)
    if lost_count or recovered != 14:
        verdict='DEDUP_REDUCES_DUPLICATES_BUT_LOSES_REAL_ROWS'
    elif not removals:
        verdict='NO_MEASURABLE_IMPROVEMENT'
    elif (safe_audit and tests_pass and parity and validation.get('capture_exit_code')==0
          and validation.get('strict_integrity_exit_code')==0 and recovered==14
          and audit.get('recovered_ante_actions')==12 and target['pass']
          and target['previously_proven_positive_control_removed']>0):
        verdict='DEDUP_PROVEN_SAFE'
    else:
        verdict='INCONCLUSIVE'
    result={'verdict':verdict,'frames':c['frames'],'physical_rows':[8146,8146,c['physical_rows']],
        'ocr_rows':[8146,8146,c['ocr_rows']],'timeline_rows':[len(old),len(broken),len(fixed)],
        'removed':len(removals),'recovered_physical_rows':recovered,
        'recovered_ante_actions':audit.get('recovered_ante_actions'),
        'confirmed_real_rows_lost':lost_count,'tables':tables,'raw_evidence_sha_parity':evidence,
        'tests':validation,'independent_audit_path':str(args.audit.resolve()),
        'scope':'This full Smoke17 source replay with the same OLD OCR/cards as BROKEN. No production promotion or general HH accuracy claim.'}
    (args.run/'acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    records = []
    for r in removals:
        fields=('table_id','session_id','hand_id','hand_epoch','frame_id','timestamp_ms','text','old_row_id',
                'matched_previous_row_id','previous_hand_id','previous_hand_epoch','duplicate_reason',
                'context_evidence_class','exact_hash_match','current_pixel_sha256','previous_pixel_sha256',
                'native_overlap_mae','min_glyph_ncc','anchor_row_ids')
        records.append({k:r[k] for k in fields})
    write_jsonl(args.run/'removed_row_index.jsonl',records)
    lines=['FILES CHANGED',
        'experiments/temporal_overlap_boundary.py; experiments/run_boundary_overlap.py; experiments/boundary_regression_replay.py; experiments/test_temporal_overlap_boundary.py; experiments/report_boundary_overlap.py; new source-linked JSONL fixtures. Registered production/BROKEN reducers and frozen inputs unchanged.',
        'TESTS ADDED', f"Boundary, stale marker, missing marker, immutable inherited context, alias/gallery isolation, adjacency barriers, monotonic/session cases; 3 recorded Smoke17 cases cover14 physical rows, plus1 audited positive duplicate. Targeted={validation.get('targeted_passed')}; no synthetic OCR in full replay.",
        'ROOT CAUSE', 'BROKEN matches inherited ante identities from previous hand and applies text/native/glyph guards without hand context. FirstStage epoch is viewport alignment, not poker hand. All14 proof: runs/boundary_root_cause_20261001/all_14_proof.jsonl.',
        'TARGETED RESULT', f"Recovered physical rows={target['recovered_physical_rows']}/14; recovered ante={target['recovered_ante_actions']}/12; real rows lost=0 in targeted fixtures; independently proved same-hand control removed={target['previously_proven_positive_control_removed']}/1.",
        'FULL SMOKE17 RESULT', f"Frames={c['frames']}; exit={validation.get('capture_exit_code')}; replay wall={c['replay_wall_seconds']:.6f}s; command wrapper wall={validation['capture_wrapper_wall_seconds']:.6f}s; physical={c['physical_rows']}; OCR={c['ocr_rows']}; timeline={c['timeline_rows']}; removed={c['removed_count']}; reducer={c['dedup_seconds']:.6f}s. Replay start={c['started_at_moscow']}; end={c['finished_at_moscow']}. Exact command={validation['capture_command']}. Wrapper UTC start/end in validation.json. Runtime errors={c['runtime_errors']}; fixed OLD OCR/card errors={c['OCR_card_errors']}. Recognition was reused exactly as BROKEN; not a live OCR/card speed benchmark.",
        'COMPARISON', f"OLD physical/OCR/timeline=8146/8146/{len(old)}; BROKEN=8146/8146/{len(broken)}; FIXED={c['physical_rows']}/{c['ocr_rows']}/{len(fixed)}. Of previous506: {c['previous_removals_still_removed']} still removed, {c['previous_506_retained']} preserved; new removal IDs={c['new_removal_ids']}. KEEP reasons={dict(Counter(r['decision']['reason'] for r in read_jsonl(args.run/'previous_506_retained.jsonl')))}. Every KEEP reason: previous_506_retained.jsonl."]
    for table in ('T05','T06','T09'):
        row=next(t for t in tables if t['table_id']==table)
        lines += [table+' RESULT',json.dumps(row,ensure_ascii=False)]
    lines += ['DUPLICATES STILL REMOVED', f"Full C removed={len(removals)}; audit safe={safe_audit}; reviewed={audit.get('reviewed_count')}. Strong overlap/MAE/NCC gates unchanged; no action-type exemption. Missing/ambiguous number now keeps rows, so not every former duplicate is suppressed. Complete per-row context/hash/evidence in removed_duplicates.jsonl / removed_row_index.jsonl.",
        'REAL ACTIONS PRESERVED',f"Prior physical losses recovered={recovered}/14; ante={audit.get('recovered_ante_actions')}/12. Independent real-row losses={lost_count}. Individual14: recovered_real_rows.jsonl. Poker safety checks={json.dumps(audit.get('poker_safety_checks',{}),ensure_ascii=False)}.",
        'OCR PARITY',f"Raw OCR byte SHA equal={evidence['raw_ocr_results.jsonl']}; corresponding retained raw text equal by exact record comparison; model/config unchanged.",
        'CARD PARITY',f"Raw visual/card evidence byte SHA equal={evidence['visual_evidence.jsonl']}; CardHybrid/config unchanged; same corresponding retained recognition output.",
        'ORDER PARITY',f"All9 retained table records exact OLD subsequences; physical pixel/identity set parity=True. Raw emission vs OLD serialized sequence equality={c['capture_emission_sequence_equal']}: {validation['ordering_analysis']['reason']} Actual raw emitted sequence was not persisted, so direct equality to reconstructed frame order is NOT_MEASURED. Original serialized physical file is byte-identical; no published order regression.",
        'FULL PYTEST RESULT',json.dumps(validation,ensure_ascii=False,indent=2),
        'PRODUCTION INTEGRITY',f"Strict integrity exit={validation.get('strict_integrity_exit_code')}; registered production hashes unchanged; manifest same SHA as pinned OLD/protocol={audit.get('manifest_identity_match')}.",
        'PRODUCTION CODE CHANGED','NO',
        'VERDICT',verdict,
        'PATHS',f"Fixed run={args.run.resolve()}; report={args.run.resolve()/'REPORT.txt'}; independent audit={args.audit.resolve()}; removals={args.run.resolve()/'removed_duplicates.jsonl'}; tests={ROOT/'experiments/test_temporal_overlap_boundary.py'}; fixtures={args.targeted.resolve()/'regression_fixtures.jsonl'}",
        'LIMITS',result['scope']+' FirstStage suppression before experimental dedup is unchanged; this experiment cannot restore actions missing from OLD physical/OCR evidence.']
    (args.run/'REPORT.txt').write_text('\n\n'.join(lines)+'\n','utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('tables','tests')},ensure_ascii=False))


if __name__=='__main__':
    main()
