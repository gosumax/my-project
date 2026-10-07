# Frozen overlay and downstream fixed-window simulation — 2026-10-01

VERDICT: **FIXED_WINDOW_ASSEMBLY_HELPS_PARTIALLY; PRODUCTION_UNCHANGED**.

## Experiment freeze

Overlayv3 source, tests, report, offline dependencies and all final artifacts copied byte-for-byte to provenance/experiments/validated_relationship_overlay_v3_20261001/files. freeze_manifest.json records hashes and proven scope. It is a proven offline edge experiment, not a production candidate or certified hand history. Prior artifacts and dirty working tree preserved.

## All27 attribution

Readable register: docs/EDGE_ALL27_HAND_ATTRIBUTION_20261001.md. Machine-readable baseline/core results: BASELINE.hands.jsonl and CORE_280.hands.jsonl in final output. Each hand contains all exact Logic checks, all Math errors and player ledger, all Fully valid blockers, raw source-linked issue context and fixed interval IDs. No missing checks hidden behind a first-break-only diagnostic. Completeness and ACTION blockers are both included.

## Math PASS6→5

Exactly one regressed hand: **T06 #2876718617**, baseline/core index4 (display hand5). Baseline fixed interval R000002 is balanced: net33200, payout33200, delta0. Flat downstream assembly appends22 rows from later fixed interval R000085(frame3699), without a retained NEW_HAND event. The prior street/players/ended state persist.

After concatenation: net63800, payout68700, delta−4900. Added money: SB3000 + BB6000 + billmurray raise13800 + vesnushka raise net37500 − return29700 = net30600; payout35500. Source ledger of the added interval is itself short4900. Raises wrongly enter the old FLOP bucket in the combined hand. Removing precisely the added22 source rows restores the old hand delta0, without any OCR/amount correction. Complete evidence and counterfactual: math_6_to_5_regression.json.

**Root cause of the lost previous-hand PASS:** edge interval ownership was lost between KEEP filtering and assemble_hands. The new22-row hand still has MATH_DELTA−4900 after correct separation; this task does not recover missing contributions. Fragmented uncalled-return R000104 lacks amount and stays an explicit blocker. The old hand regains Math PASS but still has two AMOUNT_SEMANTICS_UNPROVEN blockers.

FILE → FUNCTION → CALLER → DATA: experiments/run_validated_relationship_edge_simulation.py dry_run → edge_gate.rows_for_downstream (flat KEEP list) → poker_hand_assembly.assemble_hands (opens boundaries only from candidate textual markers) → poker_logic_engine.validate_hand (state/ledger) → poker_math_engine.validate_math. Real data old R000002 and newly admitted R000085 share one assembled hand because the latter begins with confirmed SB/BB/DEAL rather than a retained textual hand marker.

## Largest next class and bounded scope

Raw counts are not root causes. CORE top categories by blocker occurrences and affected hands:

| Category | Occurrences | Affected hands |
|---|---:|---:|
|ACTION_AFTER_HAND_END|208|20|
|ROW_IDENTITY_UNRESOLVED|83|26|
|MISSING_ACTION|33|13|
|AMOUNT_SEMANTICS_UNPROVEN|32|12|
|ACTION_AFTER_FOLD|31|11|
|CARD_EVIDENCE_MISSING|24|20|
|OLD_WIN|23|20|
|MATH_DELTA|22|22|
|SIDE_POT_NOT_VERIFIABLE|21|19|
|CHAT_OVERLAP_DUPLICATE|20|14|

Selected one evidence-proven correctable class under the largest symptom ACTION_AFTER_HAND_END: **FIXED_EDGE_INTERVALS_LOST_AT_DOWNSTREAM_ASSEMBLY**.26 hands have identity blockers; those cannot be declared correctable merely from counts. Among27 legacy assembled hands,2 cross fixed intervals in CORE (1 in baseline); this directly proves transport loss. This is the largest currently proven fixable shared cause found in this bounded analysis, not a claim that all208 after-end errors have this cause. Only8/208 after-end errors are removed;200 remain.

## Isolated simulation

Group unchanged KEEP rows by exact table/session/confirmed start_row_id from frozen edge attribution. Reuse existing assemble_hands inside each interval, requiring one candidate and exact admitted start row. Carry confirmed start as derived edge_start_evidence; do not inject NEW_HAND, number, epoch or payout. Keep end_observed tied to existing WIN/HAND_END, not merely next start. Preserve every admitted row, original field and per-table order; no dedup, text match or automatic relationship event suppression.

| Arm | Admitted rows | Hands | Logic PASS | Math PASS | Fully valid |
|---|---:|---:|---:|---:|---:|
|BASELINE|954|27|0|6|0|
|CORE_280|976|27|0|5|0|
|BASELINE_FIXED_WINDOWS|954|28|1|6|1|
|CORE_FIXED_WINDOWS|976|29|1|6|1|

**Fully valid0→1** and Logic0→1 in both fixed-window arms. Valid candidate: **T08**, hand number UNKNOWN, exact fixed start `table_fee1721aabe467d7bb7db090:R000002`. It has net29200/payout29200, unchanged antes/blinds/folds/raise/return/payout, no CALL/BET semantics-dependent actions and zero validator blockers. Source contains redundant DEAL observations retained as informational under the unchanged validator. This is FULLY_VALID_HAND under the existing offline checks, not independent visual/HH certification.

This gain comes from fixed-window transport, **not relationship marginal coverage**: baseline fixed windows already has the same valid candidate. CORE contributes one extra22-row candidate with negative4900 math and unresolved amount/provenance. Math PASS5→6 restores the prior clean ledger without hiding the new deficit. All four arms have semanticsUNPROVEN; therefore the comparison does not silently promote CALL/BET semantics. Multi-window assembled hands2→0; candidate count27→29 reflects preserved intervals rather than creation of new source boundaries.

## Safety and tests

Tester: original bounded suite131 passed/0failed, strict production integrity PASS, simulation exit0. Reviewer identified attribution completeness omission; fixed action+completeness listing and added two targeted tests. Final fixed-window module7 passed/0failed; final rerun artifacts in outputs/edge_hand_attribution_window_simulation_20261001_v2. Earlier outputs preserved. Reviewer found no material boundary risk; monetary semantics confound checked absent.

Tests cover two forced-start hands with no NEW_HAND event, missing payout cannot be invented, missing admitted start row, missing interval, epoch veto, completeness+action attribution and interleaved table/session isolation. Tester independently verified94 source/production hashes, overlay freeze fingerprints,8146 original fields/order,976 admitted row multiset/value/per-table order and29 candidate starts matched to existing fixed intervals. Image writes0; no video/OCR/API/Jev calls.

## Files and promotion gate

New task files: experiments/attribute_edge_hands_and_window_simulation.py, experiments/report_edge_hand_attribution.py, tests/test_fixed_edge_window_simulation.py, this report and the all27 attribution document; new freeze and output artifacts. Existing overlay files, production modules/configs/manifest, edge gate and validators are unchanged.

**PRODUCTION CODE CHANGED: NO.** No promotion considered. One new isolated change improves existing-validator Fully valid count; several independent changes with preserved correctness must improve Fully valid hands before considering transfer of relationship logic. This is a future evaluation condition, not deployment authorization.
