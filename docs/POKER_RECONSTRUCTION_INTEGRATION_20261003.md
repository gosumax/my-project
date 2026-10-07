# Safe reconstruction integration, 2026-10-03

The current manifest-defined RTF1 candidate and adjacent-safe-v4 CHAT tracker were retained. The only runtime orchestration change is a call to `write_reconstruction(output, hand_records)` immediately after `analyze_timeline` and before the existing output writes. No existing assembly, validator, tracker, OCR or reconstruction algorithm was edited by this task.

## FILE -> FUNCTION -> CALLER -> DATA

`parser_core/poker_reconstruction_layer.py` -> `write_reconstruction` -> `scripts/run_smoke14_ram.py:main`, launched only through `run_production.py` -> existing `hand_records` with strict production boundaries and source rows. The adapter passes a separate indexed copy of each segment to the existing parser/assembly/solver; raw events and production assembly retain original values. The engine does not choose production hand boundaries.

## Output authority

`configs/poker_reconstruction.json` sets enabled=true and allow_logic_math_output=false, with a 20,000-node budget per segment. The engine verdict remains NOT_PROVEN. EXACT/AUTO_RECOVERED can provide additional validation only with unique exact eligible accounting, no deleted or invented source events, exact kept-source coverage, physical adjacency for grouped rows, no aliases and an existing fully_validated production hand. A result does not replace existing hand_analysis.jsonl, validated_hands.jsonl or HH. A fully validated hand is not certified HH.

RECOVERED_BY_LOGIC_MATH stays a candidate with full evidence. Even if its feature flag is later enabled, existing production evidence gates and the prohibition on row deletion remain effective. AMBIGUOUS, INVALID and TRUNCATED_START/TRUNCATED_END retain engine status and source evidence; failed output gates put all source rows into unresolved_rows. Diagnostic dropped_stale is a proposed engine removal, never a deletion from production.

Separate outputs are poker_reconstruction.jsonl, reconstructed_candidates.jsonl, reconstruction_validated_hands.jsonl and reconstruction_summary.json. Each segment includes raw_events, assembled_events, kept_events, dropped_stale, unresolved_rows, identity_hypotheses, player_contributions, returns, calculated_pot, observed_payout, pot_delta, reconstruction_status and reconstruction_reason. Engine alternatives, traces and grouping graph are also retained.

## Baseline and concurrent work

The before snapshot is runs/reconstruction_integration_20261003/before/. Initial integrity failed because chat_tracker.py was already imported but unregistered, and ram_first_stage.py/temporal_overlap.py hashes were stale. Registering those current files does not promote their algorithms or revert them to golden.

During this task another writer changed ram_first_stage.py, temporal_overlap.py and test_identity_observer_isolation.py. Their initial and later bytes are saved separately under before/ and concurrent_update/. This task preserved the later changes; live results cannot be attributed exclusively to reconstruction. Reconstruction package source bytes and poker_pipeline.py were unchanged.

## Acceptance evidence

Evidence lives under runs/reconstruction_integration_20261003/. Full active pytest includes existing production and reconstruction tests. scripts/audit_reconstruction_integration.py performs an offline full frozen Smoke17 JSONL replay using saved raw OCR statuses. It compares pre-integration/current hand records and summaries except timing, retains before/after hand records and source hashes, and writes reconstruction results. This is downstream replay, not a new full video/OCR run. A separate official live 500-frame smoke tests runtime registration and output/Excel integration.

No new golden is frozen: pre-existing test failures and engine NOT_PROVEN prevent certification. See REPORT.md in the evidence directory for actual numerical results and limitations.
