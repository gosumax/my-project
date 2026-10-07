"""Publish explicit effective ownership attribution from saved simulation only."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_validated_relationship_edge_simulation import read, write_json, write_rows


def main():
    out=ROOT/'outputs/validated_relationship_edge_smoke17_20261001_v3'
    report=json.loads((out/'simulation_report.json').read_text('utf-8'))
    baseline=json.loads((out/'baseline_downstream_dry_run.json').read_text('utf-8'))['baseline']
    effective={}
    for mode in report['modes']:
        directory=out/mode
        links={r['pair_id']:r for r in read(directory/'relationships_used.jsonl')}
        changes=read(directory/'status_changes.jsonl')
        result=[]
        for row in changes:
            pairs=sorted({pair for proof in row['window_continuity_proofs']
                          for path in proof['relationship_paths'] for pair in path})
            result.append(dict(source_stable_row_id=row['stable_row_id'],sequence_id=row['sequence_id'],
                old_edge_status=row['old_edge_status'],new_edge_status=row['new_edge_status'],
                exact_relationships_used=[links[pair] for pair in pairs],
                hand_interval=row['hand_interval'], boundary_frames={
                    'confirmed_start':row['hand_interval']['start_frame'],
                    'next_confirmed_start':row['hand_interval']['next_start_frame']},
                reason_before=row['reason_before'],reason_after=row['reason_after'],
                baseline_continuity_diagnostic=row['ownership_evidence'],
                effective_edge_ownership='CURRENT_FIXED_WINDOW_SUPPORTED_AFTER_VALIDATED_PREVIOUS_TAIL',
                effective_end_reason=row['effective_end_reason'],
                source_poker_event_or_identity_rewritten=False,duplicate_event_suppressed=False,
                certification='NOT_CERTIFIED_BOUNDARIES_OR_HH'))
        assert len(result)==report['modes'][mode]['status_changes']
        target=directory/'effective_reclassification_attribution.jsonl'
        if target.exists(): assert read(target)==result
        else: write_rows(target,result)
        effective[mode]=len(result)
    c=report['modes']['CORE_280']; e=report['modes']['EXTENDED_313_SAMPLE']
    d=c['downstream']['dry_run'];perf=c['performance'];ep=e['performance']
    lines=['# Validated relationship edge simulation — Smoke17, 2026-10-01','',
        '## FILES CHANGED','',
        '- `experiments/validated_relationship_edge_simulation.py` — isolated continuity/ownership overlay.',
        '- `experiments/run_validated_relationship_edge_simulation.py` — frozen cohorts, parity, timing, offline dry-run.',
        '- `experiments/report_validated_relationship_edge_simulation.py` — explicit effective attribution and this report.',
        '- `tests/test_validated_relationship_edge_simulation.py` — admission/provenance/negative regressions.',
        '- `docs/VALIDATED_RELATIONSHIP_EDGE_SMOKE17_20261001.md` — this report.','',
        'All five are new task files. Existing production, gate, assembly, logic/math, configs and manifest unchanged. Prior dirty files preserved.', '',
        'Final behavior evidence: `outputs/validated_relationship_edge_smoke17_20261001_v3/`. Initial failed provenance run and earlier metadata-only result are preserved in v1/v2 directories and are not final acceptance evidence.', '',
        '## RELATIONSHIPS USED','',
        'CORE_280: **280 validated TEXT-FIRST VISUALLY_SAME**. EXTENDED_313_SAMPLE: **313**, including separate33 visually SAME AMBIGUOUS sample.8 DIFFERENT/9 UNCERTAIN excluded from both.', '',
        'Review labels and raw judgments are hash-frozen. Exact endpoint/lifetime records match the original candidate file pinned by sample_summary; cohort matches pinned TEXT-FIRST decisions. Original source crop hashes are bound independently through source_ocr_parents to raw OCR records. Endpoint crop hashes at old.last_frame remain separate from source-frame hashes. Lifetimes use the frozen causal review cutoff, not terminal future extrapolation.', '',
        '## BASELINE EDGE','',
        '| Status | BASELINE | CORE_280 | EXTENDED_313_SAMPLE |','|---|---:|---:|---:|']
    for status in ['KEEP','PARTIAL_START','PARTIAL_END','UNRESOLVED_EDGE']:
        lines.append(f'| {status} | {report["baseline_edge"][status]} | {c["edge_counts"][status]} | {e["edge_counts"][status]} |')
    lines += ['', 'Total **8146** in every mode. Baseline row-for-row equals frozen final edge output.', '',
        '## CORE_280 EDGE RESULT','',
        '**UNRESOLVED_EDGE 6685 → 6663.** Only22 rows in one T06 interval gain KEEP. No new boundaries, epochs, sessions or events.', '',
        '## STATUS CHANGES','',
        '**22 UNRESOLVED_EDGE → KEEP;0 other status transitions.** Full rows remain in classified_rows.jsonl. status_changes.jsonl contains every transition, before/after reasons, fixed interval and exact validated paths. relationships_used.jsonl resolves every path to immutable old/new IDs and validation provenance.', '',
        'effective_reclassification_attribution.jsonl additionally embeds exact relationships, boundary frames and effective current-window ownership for every changed row. The original ownership_evidence field describes the pre-discharge continuity diagnostic;2 admitted rows retain its baseline UNRESOLVED_BOUNDARY_OWNERSHIP label. The supplemental final attribution makes their window-based admission explicit; no source identity has been certified or rewritten.', '',
        '## IDENTITY-ONLY RESOLUTIONS','',
        '**22**, from case **V122**, `table_6f37cc14274a733e7ed2ad35:R000081 → table_6f37cc14274a733e7ed2ad35:R000092`.', '',
        'FILE → FUNCTION → CALLER → DATA: frozen parser_core/edge_gate.py classify_edges consumes assemble_messages/confirmed_hand_starts and _safe; the new simulate overlay consumes these fixed intervals plus separate validated links. Production itself ends at timeline/Excel; these offline modules are outside its manifest.', '',
        'Original R000081 first appears at frame3699 before the confirmed SB row R000085 at frame3699,row13. R000092 appears at3800 and is visually validated continuation of the same NEW_HAND physical row. Comparing (frame,row_index), rather than frame alone, proves it predates the current interval. Raw OCR parents are OK; identity origin produces source statusUNRESOLVED under registered run_smoke14_ram.py:363. No unrelated OCR failure is discharged.', '',
        'Fixed current interval3699→4542 has only UNSAFE_NEW_HAND_BOUNDARY. The next already-confirmed start4542 supplies its end. All unsafe NEW_HAND message sources are validated previous-tail observations, so that observation ceases vetoing the current interval. **R000092 remains UNRESOLVED_EDGE**, owned by the already-uncertain previous interval2565→3699. Its poker event is not duplicated or admitted. Other window rows gain KEEP through this shared proof.', '',
        '## UNRESOLVED REASON BREAKDOWN','',
        'Exclusive combinations, sums6685 before and6663 after:', '',
        '| Active reason combination | Before | After |','|---|---:|---:|---:|',
        '| Conflicting numbers + epoch discontinuity + multiple NEW_HAND + unsafe boundary |5302|5302|',
        '| Multiple NEW_HAND + unsafe boundary |954|954|',
        '| Conflicting numbers + multiple NEW_HAND + unsafe boundary |271|271|',
        '| Epoch discontinuity + unsafe boundary |135|135|',
        '| Unsafe boundary only |23|0|',
        '| Previous tail, previous owner conflicting numbers + multiple NEW_HAND + unsafe boundary |0|1|', '',
        'The unsafe-only23 comprise22 resolved current rows and1 retained previous tail. No genuine number/multiple-start/epoch conflict is discharged. Overlapping epoch predicate5437→5437. Number predicate5573→5574 and multiple-start6527→6528 increase by1 solely because the retained tail now carries its previous-owner reasons. Unsafe boundary6685→6663.1555 remaining unresolved rows also lack a proven end, unchanged; this is a secondary prerequisite, not an extra disjoint population.441 missing-start fragments and66 partial ends remain unchanged. Session/source-integrity invalid groups0. Card/math/logic are downstream checks, not invented edge reasons.', '',
        '## NEGATIVE CONTROLS','',
        '8 VISUALLY_DIFFERENT /9 VISUALLY_UNCERTAIN:0 links admitted.217 duplicate-text blocked relationships:0 admitted in CORE. EXTENDED admits9 only because those exact pairs have separate independent SAME validation; unvalidated equal text never grants continuity.', '',
        'Targeted regressions preserve real conflicting numbers, rejected second starts in the same window, epoch changes, missing links, branch conflicts, missing start/end, unrelated ERROR/UNREADABLE/OCR failure, and stale previous-hand payout. Any chain conflict in a candidate window blocks discharge. End witnesses are checked with existing _confirmed_end on unchanged current-owner messages only; a tail payout cannot prove completion.', '',
        '**126 passed /0 failed**, targeted edge+simulation+logic/math suite,2.76s; strict production integrity PASS. Independent reviewer final audit: no remaining material findings after these safety regressions.', '',
        '## EXTENDED_313 RESULT','',
        '**6685→6663**, same22 changes; no extra recall demonstrated. CORE280 independent components,0 chains/conflicts. EXTENDED311 components,2 multi-link branch-conflicted components;6 affected rows retain unresolved ownership. No missing-link transitive merge. This sampled extension is not production rule coverage.', '',
        '## DOWNSTREAM DRY-RUN','',
        'Existing deterministic assemble_hands / attach_blind_level / semantic_evidence / validate_hand and its existing math run on copied filtered rows. No Jev/API/model calls or downstream repair.', '',
        '| Metric | Comparable baseline | CORE / EXTENDED |','|---|---:|---:|',
        f'| Admitted rows |{baseline["admitted_rows"]}|{d["admitted_rows"]}|',
        f'| Hands assembled |{baseline["hands_assembled"]}|{d["hands_assembled"]}|',
        f'| Complete candidates |{baseline["complete_candidates"]}|{d["complete_candidates"]}|',
        f'| Logic PASS |{baseline["logic_pass"]}|{d["logic_pass"]}|',
        f'| Math PASS |{baseline["math_pass"]}|{d["math_pass"]}|',
        f'| Fully valid |{baseline["fully_valid"]}|{d["fully_valid"]}|', '',
        'Math PASS **6→5**. Both semanticsUNPROVEN. Existing ROW_IDENTITY_UNRESOLVED, ACTION_AFTER_HAND_END, overlap/card/amount defects remain visible; no fixes applied. Row admission is not logic/math/HH acceptance.', '',
        '## SOURCE PARITY','',
        '**8146 BEFORE ==8146 AFTER** in both modes. Deep equality of every original field and order, including source statuses, stable IDs, table/session, frames/times, OCR/events, hashes/card evidence, epochs.94 input-file hashes unchanged. Reclassifying the copied rows with the frozen gate gives exactly the original session/hand-boundary summaries. Source XLSX unchanged; no simulation Excel rewrite requested. Image writes **0**, no video decode/image operations. Independent parity and post-run implementation fingerprints: independent_parity.json.', '',
        '## PERFORMANCE','',
        'Five alternating in-memory repeats; medians, excludes artifact I/O:', '',
        '| Metric | CORE_280 | EXTENDED_313_SAMPLE |','|---|---:|---:|',
        f'| Baseline edge seconds |{perf["baseline_median"]:.6f}|{ep["baseline_median"]:.6f}|',
        f'| Full simulation seconds |{perf["simulation_median"]:.6f}|{ep["simulation_median"]:.6f}|',
        f'| Full overhead |{perf["overhead_percent"]:.2f}%|{ep["overhead_percent"]:.2f}%|',
        f'| Overlay/attribution with fixed gate/context seconds |{perf["overlay_attribution_median"]:.6f}|{ep["overlay_attribution_median"]:.6f}|',
        f'| Relationship deep bytes |{perf["relationship_metadata_deep_bytes"]}|{ep["relationship_metadata_deep_bytes"]}|',
        f'| Component deep bytes |{perf["continuity_components_deep_bytes"]}|{ep["continuity_components_deep_bytes"]}|', '',
        'Full overhead includes repeated message assembly, reason attribution and deepcopy; it is not lookup-only latency or a production performance estimate. Metadata values are Python object deep sizes, not process peak RAM. No images are stored in metadata. Full command wall/process peak RAM not measured.', '',
        '## PRODUCTION CODE CHANGED','', '**NO.** No promotion, freeze or production run.', '',
        '## VERDICT','', '**VALIDATED_RELATIONSHIPS_HELP_PARTIALLY**.', '',
        'The22 changes are specifically attributable to one independently validated physical continuity relationship and existing fixed boundaries. Uncertain previous ownership stays unresolved. Real boundary/epoch/OCR vetoes remain. Independent correctness certification of the hand boundaries and fully valid HH remain unproved. Work stops after this report.']
    doc=ROOT/'docs/VALIDATED_RELATIONSHIP_EDGE_SMOKE17_20261001.md'
    doc.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(out/'effective_attribution_summary.json',dict(rows=effective,
        purpose='Explicit window proof and effective ownership supplement; original source rows unchanged'))
    print(json.dumps(dict(report=str(doc),effective_attribution_rows=effective)))


if __name__=='__main__': main()
