# Native temporal overlap promotion, 2026-10-02

User authorized promotion of NEW18 + NCC and downstream verification. The
manifest-defined entrypoint remains `run_production.py`; original golden
`provenance/production-golden-20260930/` is retained. Original manifest/docs are
also retained in `provenance/experiments/temporal_promotion_20261002/`.

## Shared cause and production integration

`parser_core/ram_first_stage.py -> FirstStage.rows/identity_observer ->
scripts/run_smoke14_ram.py::main -> native visible appearances + stable physical
row IDs + real Tiny OCR` provides the complete visible block in capture order.
Rows with `UNRESOLVED_IDENTITY` can be repeated visual appearances. They cannot
be removed solely because old visual anchors and identical OCR agree: old ante
anchors can survive into another poker hand.

`parser_core/temporal_overlap.py -> TemporalObservations.reduce ->
BoundaryReducer.process -> guarded native overlap` now runs after real OCR drain
and before timeline resequencing, `group_rows`, `associate_visual_evidence`,
`project_chat_card_events`, JSONL and the existing Excel exporter. This rebuilds
all downstream message/card annotations from the retained chronology.

The promoted native MAE/glyph NCC/gallery mechanics come from the guarded
experiment plus its hand-boundary successor. No experiment runner is imported
into production. Deletion requires one table/session/known immutable hand for
candidate, target, all anchors and overlap pairs, two inherited anchors, exact
nonempty OCR agreement, native MAE <=9 and glyph NCC >=0.90. Unknown/conflicting
contexts, malformed/stale/nonmonotonic markers and unnumbered boundaries KEEP.
Inherited markers are checked again against their own original pixels.

Independent review additionally proved two uncertainty gaps. Production now
passes explicit OCR status (timeline UNRESOLVED cannot hide OCR ERROR), and
empty/failed OCR is a spatial context barrier because it may be a hand marker.
Frame monotonicity, unique visible span indices and session binding are checked.
Assertions protecting runtime hand evidence are explicit exceptions, also under
optimized Python execution.

Raw physical/OCR/card evidence stays available. Removal and per-row context
proofs are exported as `removed_duplicates.jsonl` and `context_decisions.jsonl`.
Aliases do not transfer card evidence or upgrade unresolved statuses. Native
appearance pixels wait for OCR in compressed RAM; no PNG transport is added.

## Evidence

`runs/temporal_promotion_20261002_checks/`: full Smoke18 capture replay,
2040 frames; 618 physical rows exactly equal in ID/hash/pixels and documented
capture/export orders. Production and boundary experiment agree in all removal
IDs/evidence/context decisions. 26 proven duplicates removed, 592 retained;
347 UNKNOWN decisions kept. All 26 deletion context proofs pass. Original
guarded-only NEW18 retained 564; that count is not the acceptance target.

The targeted suite includes source-backed real Smoke17 losses and a positive
Smoke18 duplicate control, hand/session conflicts, markerless blocks, stale
marker OCR, empty OCR barriers and failed OCR with nonempty text. Latest targeted
suite: 73 passed. Full active pytest: 770 passed, 7 subtests, no failures.

Replay timing: capture 71.025s, production reduction 24.189s; the comparison
also ran the experimental reduction 29.796s. This is fixed-recognition replay,
not a live OCR throughput claim. 67,446 appearances occupied 98,138,551 compressed
bytes. Live integration and final golden checks are recorded separately below.

## Live production integration

Both candidate runs used official `run_production.py`, real Tiny OCR and
PIXEL_ONLY card workers. Their launch metadata has no modified registered files;
the exact candidate manifest is saved at
`runs/temporal_promotion_20261002_checks/candidate_manifest.json`.

| Run | Frames | Physical/OCR | Timeline | Removed | Wall seconds | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate500 | 500 | 298 | 291 | 7 | 44.200153 | 2.657324 |
| full candidate | 2040 | 618 | 592 | 26 | 115.658637 | 1.701696 |

Both exited0, exported Excel, retained PARTIAL_VISUAL, and produced zero errors
and intermediate PNG files/bytes. Full capture69.028583s; reduction23.879233s;
main peak RSS280.512MiB; peak pipeline1035.926MiB. Compared with the frozen OLD
run, full wall increased44.62%, capture18.20%, peak pipeline102.430MiB. These
numbers are observational, not an isolated microbenchmark.

`live_comparison_summary.json` verifies physical/OCR file SHA equality; complete
raw capture+hybrid fields for all45 card rows /164 tiles; exact retained IDs,
order, OCR text and pixel hashes; freshly reprojected event/cards/status fields;
Excel ALL_EVENTS592 rows, sequence/frame/span/text/hash equality to JSONL.

Independent review is complete. The two explicit OCR barriers supplement the
prior source-backed boundary proof. Freeze guards reject existing golden
destinations and a2040-frame run mislabeled as a500-frame benchmark (exit2,
without mutation). Reviewed freeze created unique
`production-golden-temporal-20261002`, with24 production modules and53 active
test files registered. Final current-manifest verification artifacts are
`runs/production_temporal_smoke18_500_golden_20261002/` and the final pytest log
under `runs/temporal_promotion_20261002_checks/`.

Post-freeze golden500 official exit0: version matches the new manifest,
modified_files=[]; 500 frames, 298 physical/OCR ->291 timeline, 7 removals;
wall42.017082s, RTF2.526077, capture20.050143s, peak pipeline938.543MiB,
zero errors/PNG. Physical/OCR/timeline/removal/context files exactly match
candidate500 by SHA. Visual fields match excluding timings; every workbook cell
matches except the intentional per-run evidence JSONL path. All77 prior-golden
snapshot hashes remain valid, and its manifest matches the pre-promotion backup
SHA `6ab17904d74b98e88edc1b8334d42cd32494fb7945f1b7102fae9bea9535b575`.
All26 live full-run removals and618 context decisions equal the audited replay
records exactly.

Final post-freeze full pytest:770 passed,7 subtests,0 failed, exit0,79.27s.
Post-test strict integrity: PASS `production-golden-temporal-20261002`, exit0.
All replay/live/golden test processes completed and their final artifacts were
verified; no pending production run remains.

## Limits

This promotion verifies bounded full Smoke18 and recorded Smoke17 regressions.
It does not certify hand histories or independent card accuracy. The official
pipeline still exports diagnostics/events through Excel, with PARTIAL_VISUAL
when applicable. Long-video RAM/performance is not established: observations
accumulate until OCR drains, and per-identity gallery limits do not bound total
run memory. No long Smoke17 live acceptance is claimed.
