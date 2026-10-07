"""Readable per-hand blocker register and fixed-window experiment report."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_validated_relationship_edge_simulation import read,write_json,digest


def codes(issues):
    return ', '.join(sorted({i['code'] for i in issues})) or 'NONE'


def main():
    out=ROOT/'outputs/edge_hand_attribution_window_simulation_20261001_v2'
    summary=json.loads((out/'summary.json').read_text('utf-8'))
    core=read(out/'CORE_280.hands.jsonl');base=read(out/'BASELINE.hands.jsonl')
    fixed=read(out/'CORE_FIXED_WINDOWS.hands.jsonl')
    pairs=json.loads((out/'all_27_hand_comparison.json').read_text('utf-8'))
    delta=json.loads((out/'math_6_to_5_regression.json').read_text('utf-8'))
    lines=['# All27 hands: exact downstream attribution — Smoke17','',
        'CORE_280 after the frozen edge overlay, before the new assembly simulation. IDs below are one-based display indices; JSON indices remain zero-based. All27 are matched to baseline by exact source-row containment, table and unique match. Source raw OCR/messages, per-player ledger, issue domain/severity/evidence type and every issue source-row ID are in CORE_280.hands.jsonl. Diagnostic first break is supplementary, not the complete blocker list.', '',
        'Logic PASS requires no ACTION blocker plus established completeness. Math PASS requires Decimal net−paid=0 and no negative player net. Fully valid requires zero BLOCKER issues of any domain; warnings remain visible.', '',
        '| # | Table / candidate hand ID | Logic | Math | Delta | Fully valid |','|---|---|---|---|---:|---|']
    for r in core:
        lines.append(f'|{r["hand_index"]+1}|{r["table_id"]} / {r["hand_id"] or "UNKNOWN"}|{r["logic_pass"]}|{r["math_pass"]}|{r["validator_result"]["math"]["delta"]}|{r["fully_valid"]}|')
    for r in core:
        old=base[next(p['baseline_hand_index'] for p in pairs if p['core_hand_index']==r['hand_index'])]
        lines += ['',f'## Hand {r["hand_index"]+1}: {r["table_id"]} / {r["hand_id"] or "UNKNOWN"}', '',
            f'Fixed edge windows: {", ".join(r["fixed_edge_window_ids"])}. Source rows {len(r["source_row_ids"])}. Status `{r["status"]}`.', '',
            '**Logic blockers:** '+codes(r['logic_blocking_checks'])+'.',
            '**Math blockers:** '+(', '.join(r['math_blocking_checks']) or 'NONE; Math PASS')+'. '+
            f'Net={r["validator_result"]["math"]["net_total"]}, payout={r["validator_result"]["math"]["payout_total"]}, delta={r["validator_result"]["math"]["delta"]}.',
            '**Fully valid blockers:** '+codes(r['fully_valid_blocking_checks'])+'.',
            f'Baseline Logic={old["logic_pass"]}, Math={old["math_pass"]}, Fully valid={old["fully_valid"]}.', '',
            '| Check | Domain / severity | Source message index / IDs | Detail |','|---|---|---|---|']
        for i in r['fully_valid_blocking_checks']:
            detail=str(i['detail']).replace('|','\\|').replace('\n',' ')
            lines.append(f'|{i["code"]}|{i["domain"]} / {i["severity"]}|{i["message_index"]}: {", ".join(i["source_row_ids"]) or "aggregate EOF"}|{detail}|')
    attribution=ROOT/'docs/EDGE_ALL27_HAND_ATTRIBUTION_20261001.md'
    attribution.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    text=['# Frozen overlay and downstream fixed-window simulation — 2026-10-01','',
        'VERDICT: **FIXED_WINDOW_ASSEMBLY_HELPS_PARTIALLY; PRODUCTION_UNCHANGED**.', '',
        '## Experiment freeze','',
        'Overlayv3 source, tests, report, offline dependencies and all final artifacts copied byte-for-byte to provenance/experiments/validated_relationship_overlay_v3_20261001/files. freeze_manifest.json records hashes and proven scope. It is a proven offline edge experiment, not a production candidate or certified hand history. Prior artifacts and dirty working tree preserved.', '',
        '## All27 attribution','',
        'Readable register: docs/EDGE_ALL27_HAND_ATTRIBUTION_20261001.md. Machine-readable baseline/core results: BASELINE.hands.jsonl and CORE_280.hands.jsonl in final output. Each hand contains all exact Logic checks, all Math errors and player ledger, all Fully valid blockers, raw source-linked issue context and fixed interval IDs. No missing checks hidden behind a first-break-only diagnostic. Completeness and ACTION blockers are both included.', '',
        '## Math PASS6→5','',
        'Exactly one regressed hand: **T06 #2876718617**, baseline/core index4 (display hand5). Baseline fixed interval R000002 is balanced: net33200, payout33200, delta0. Flat downstream assembly appends22 rows from later fixed interval R000085(frame3699), without a retained NEW_HAND event. The prior street/players/ended state persist.', '',
        'After concatenation: net63800, payout68700, delta−4900. Added money: SB3000 + BB6000 + billmurray raise13800 + vesnushka raise net37500 − return29700 = net30600; payout35500. Source ledger of the added interval is itself short4900. Raises wrongly enter the old FLOP bucket in the combined hand. Removing precisely the added22 source rows restores the old hand delta0, without any OCR/amount correction. Complete evidence and counterfactual: math_6_to_5_regression.json.', '',
        '**Root cause of the lost previous-hand PASS:** edge interval ownership was lost between KEEP filtering and assemble_hands. The new22-row hand still has MATH_DELTA−4900 after correct separation; this task does not recover missing contributions. Fragmented uncalled-return R000104 lacks amount and stays an explicit blocker. The old hand regains Math PASS but still has two AMOUNT_SEMANTICS_UNPROVEN blockers.', '',
        'FILE → FUNCTION → CALLER → DATA: experiments/run_validated_relationship_edge_simulation.py dry_run → edge_gate.rows_for_downstream (flat KEEP list) → poker_hand_assembly.assemble_hands (opens boundaries only from candidate textual markers) → poker_logic_engine.validate_hand (state/ledger) → poker_math_engine.validate_math. Real data old R000002 and newly admitted R000085 share one assembled hand because the latter begins with confirmed SB/BB/DEAL rather than a retained textual hand marker.', '',
        '## Largest next class and bounded scope','',
        'Raw counts are not root causes. CORE top categories by blocker occurrences and affected hands:', '',
        '| Category | Occurrences | Affected hands |','|---|---:|---:|']
    arm=summary['arms']['CORE_280']
    for code,n in sorted(arm['blocker_occurrences'].items(),key=lambda item:-item[1])[:10]:
        text.append(f'|{code}|{n}|{arm["affected_hands_by_blocker"][code]}|')
    text += ['',
        'Selected one evidence-proven correctable class under the largest symptom ACTION_AFTER_HAND_END: **FIXED_EDGE_INTERVALS_LOST_AT_DOWNSTREAM_ASSEMBLY**.26 hands have identity blockers; those cannot be declared correctable merely from counts. Among27 legacy assembled hands,2 cross fixed intervals in CORE (1 in baseline); this directly proves transport loss. This is the largest currently proven fixable shared cause found in this bounded analysis, not a claim that all208 after-end errors have this cause. Only8/208 after-end errors are removed;200 remain.', '',
        '## Isolated simulation','',
        'Group unchanged KEEP rows by exact table/session/confirmed start_row_id from frozen edge attribution. Reuse existing assemble_hands inside each interval, requiring one candidate and exact admitted start row. Carry confirmed start as derived edge_start_evidence; do not inject NEW_HAND, number, epoch or payout. Keep end_observed tied to existing WIN/HAND_END, not merely next start. Preserve every admitted row, original field and per-table order; no dedup, text match or automatic relationship event suppression.', '',
        '| Arm | Admitted rows | Hands | Logic PASS | Math PASS | Fully valid |','|---|---:|---:|---:|---:|---:|']
    for name,a in summary['arms'].items():
        text.append(f'|{name}|{954 if name.startswith("BASELINE") else 976}|{a["hands"]}|{a["logic_pass"]}|{a["math_pass"]}|{a["fully_valid"]}|')
    text += ['',
        '**Fully valid0→1** and Logic0→1 in both fixed-window arms. Valid candidate: **T08**, hand number UNKNOWN, exact fixed start `table_fee1721aabe467d7bb7db090:R000002`. It has net29200/payout29200, unchanged antes/blinds/folds/raise/return/payout, no CALL/BET semantics-dependent actions and zero validator blockers. Source contains redundant DEAL observations retained as informational under the unchanged validator. This is FULLY_VALID_HAND under the existing offline checks, not independent visual/HH certification.', '',
        'This gain comes from fixed-window transport, **not relationship marginal coverage**: baseline fixed windows already has the same valid candidate. CORE contributes one extra22-row candidate with negative4900 math and unresolved amount/provenance. Math PASS5→6 restores the prior clean ledger without hiding the new deficit. All four arms have semanticsUNPROVEN; therefore the comparison does not silently promote CALL/BET semantics. Multi-window assembled hands2→0; candidate count27→29 reflects preserved intervals rather than creation of new source boundaries.', '',
        '## Safety and tests','',
        'Tester: original bounded suite131 passed/0failed, strict production integrity PASS, simulation exit0. Reviewer identified attribution completeness omission; fixed action+completeness listing and added two targeted tests. Final fixed-window module7 passed/0failed; final rerun artifacts in outputs/edge_hand_attribution_window_simulation_20261001_v2. Earlier outputs preserved. Reviewer found no material boundary risk; monetary semantics confound checked absent.', '',
        'Tests cover two forced-start hands with no NEW_HAND event, missing payout cannot be invented, missing admitted start row, missing interval, epoch veto, completeness+action attribution and interleaved table/session isolation. Tester independently verified94 source/production hashes, overlay freeze fingerprints,8146 original fields/order,976 admitted row multiset/value/per-table order and29 candidate starts matched to existing fixed intervals. Image writes0; no video/OCR/API/Jev calls.', '',
        '## Files and promotion gate','',
        'New task files: experiments/attribute_edge_hands_and_window_simulation.py, experiments/report_edge_hand_attribution.py, tests/test_fixed_edge_window_simulation.py, this report and the all27 attribution document; new freeze and output artifacts. Existing overlay files, production modules/configs/manifest, edge gate and validators are unchanged.', '',
        '**PRODUCTION CODE CHANGED: NO.** No promotion considered. One new isolated change improves existing-validator Fully valid count; several independent changes with preserved correctness must improve Fully valid hands before considering transfer of relationship logic. This is a future evaluation condition, not deployment authorization.']
    report=ROOT/'docs/EDGE_HAND_ATTRIBUTION_WINDOW_SIMULATION_20261001.md'
    report.write_text('\n'.join(text)+'\n',encoding='utf-8')
    hashes={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),
        ROOT/'experiments/attribute_edge_hands_and_window_simulation.py',
        ROOT/'tests/test_fixed_edge_window_simulation.py',attribution,report]}
    write_json(out/'report_manifest.json',dict(hashes=hashes,source_outputs_unchanged=True,
        production_promotion=False,fully_valid_gain_origin='FIXED_WINDOW_TRANSPORT_NOT_RELATIONSHIP_INCREMENT'))
    print(json.dumps(dict(report=str(report),all27=str(attribution),hands=len(core))))


if __name__=='__main__':main()
