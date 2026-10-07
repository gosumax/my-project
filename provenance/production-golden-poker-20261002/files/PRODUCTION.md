THIS FILE DEFINES THE ONLY PRODUCTION PIPELINE.

DO NOT DISCOVER, GUESS OR RECONSTRUCT PRODUCTION FROM OTHER FILES.

EXPERIMENTAL OR LEGACY CODE MUST NEVER REPLACE A PRODUCTION MODULE UNLESS IT IS EXPLICITLY PROMOTED THROUGH THE PRODUCTION UPDATE PROCEDURE.

# CURRENT PRODUCTION

- Name/version: `production-golden-poker-20261002`.
- Promotion: strict source-preserving hand assembly, deterministic poker logic, Decimal observed ledger, physical card checks and first-break diagnostics run after verified card projection, before Excel. Native temporal overlap + glyph NCC retains immutable hand/session fences and UNKNOWN/failed-OCR barriers. See `docs/POKER_PRODUCTION_CONNECTION_20261002.md`.
- Prior golden snapshots `provenance/production-golden-temporal-20261002/` and `provenance/production-golden-20260930/` remain immutable. Prior manifest/docs are retained in `provenance/experiments/poker_pipeline_promotion_20261002/`.
- Git HEAD: `a887126816f4166395810fd369d2ad0bd0a115ff`. HEAD alone does **not** represent the dirty working implementation. No tag/commit created: unrelated tracked and untracked changes exist.
- Exact working implementation: `production_manifest.json` hashes and `provenance/production-golden-poker-20261002/files/` snapshot. The snapshot retains registered source/config/test bytes, tracked dirty patch, status and HEAD without changing Git staging or unrelated files.
- Official entrypoint: `run_production.py:main`. It verifies the manifest and delegates to the already audited `scripts/run_smoke14_ram.py:main`; the historical runner is **INTERNAL ONLY**, not an alternative launcher. No new capture/recognition architecture is assembled.
- Pipeline: RAM_FIRST; PP-OCRv6 Tiny text OCR; PIXEL_ONLY card recognition; exact verified rank masks plus frozen RankNCC `.80/.01`; existing raw_suit2 and suit NCC BACKUP `.98/.03`, with native component fallback. VL/Qwen for CHAT cards is disabled.

The single official command (change only video, requested frame count and fresh output directory):

```powershell
python run_production.py --video video/smoke_17.mp4 --frames 500 --output runs/production_smoke_fresh --ocr-workers 2 --visual-workers 2 --card-workers 2 --node C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe
```

Use the actual full decoded frame count for a full video; a 500-frame smoke is not a full-run acceptance. Configuration overrides pointing to other files are rejected. `--skip-cards` disables card work; `--skip-vl` is its historical alias, not a means to select another card recognizer. Standard production includes cards.

## Evidence and performance

Control source: `video/smoke_17.mp4`, SHA256 `27c0ba10544e0f6025e2d563baf7693d36c185cdbbfeb4517d3486116e9f8baf`.

Reviewed live candidate500: `runs/production_poker_smoke17_500_candidate_20261002/`, exit0, PARTIAL_VISUAL, 500 frames,349 physical/OCR rows ->348 timeline rows,1 proven duplicate. Wall55.444231s, capture21.284028s, RTF3.333320.16 assembled segments,9 numbered hands,3 complete candidates,1 fully validated hand,0 certified HH. Excel348 ALL_EVENTS and16 HAND_SUMMARY rows verified. Runtime errors and intermediate PNG files/bytes0. Exact saved candidate manifest: `runs/poker_pipeline_promotion_20261002_checks/candidate_smoke17_manifest.json`.

Full live integration: `runs/production_poker_smoke17_full_candidate_v2_20261002/`, actual exit0, all28,222 frames (940.7s),8,146 physical/OCR rows ->7,954 timeline rows,192 proven removals. Physical and OCR file SHA parity against prior full Smoke17; retained ID/order parity against safe boundary replay; all192 removal proofs, raw recognition/cards, source card projection and Excel checks pass.4506 UNKNOWN observations kept.329 card messages bind verified pixel evidence with0 binding conflicts. Excel7,954 ALL_EVENTS and196 HAND_SUMMARY rows, exact7,954/7,954 hand-analysis coverage.

Hand results:196 segments,189 unique numbered hands,163 complete candidates,1 fully validated hand,0 certified HH. Analysis status PARTIAL_UNRESOLVED; most blockers concern row identity, missing actions, unproven amount semantics and overlap/stale events;11 segments have card conflicts. Observed math195 NOT_VERIFIABLE and1 PASS. Raw validator statuses remain visible independently; raw MATH_FAIL on an incomplete/untrusted ledger is not proof of a mathematical contradiction. No external resolver or synthesized action/card/money facts are used.

Full measured wall1394.967493s, RTF1.482746; capture963.440512s, dedup249.696070s, hand analysis4.209739s, Excel37.886382s. Peak sampled pipeline RSS3633.8008MiB, main2227.8594MiB; compressed appearances1,412,825,440bytes. Runtime errors and intermediate PNG files/bytes0. Temporary gallery release before Excel was independently reviewed. Appearances still accumulate until OCR drains; this is not a global streaming memory bound.

Final manifest golden500 verification output: `runs/production_poker_smoke17_500_golden_20261002/`; final exit/artifacts and test evidence are recorded separately in `runs/poker_pipeline_promotion_20261002_checks/`. Earlier interrupted runs have explicit abort markers and are not completed acceptance runs. Historical temporal-only Smoke18 promotion evidence remains in `docs/TEMPORAL_PRODUCTION_PROMOTION_20261002.md`.

The early capture guard uses the reviewed Smoke17 500-frame capture21.284028s;4x slowdown with modified-path evidence stops the run. Total RTF range derives from that500-frame run (startup/export overhead included), not a universal SLA or HH accuracy gate.

Known Smoke15 DEV evidence: `runs/pixel_card_remaining38/replay_WITH_VL_CACHE/summary.json` has900/900 correct tiles,0wrong,0unresolved; projected level583correct,0wrong,317unresolved. It used cached OCR/VL and frozen FALLBACK, so **is not current PIXEL_ONLY independent validation**. Do not equate tile success with event projection, accuracy certification or complete HH. Production99.9% remains unproved; WRONG and UNRESOLVED both fail the full-card metric.

## PNG policy and outputs

Filesystem transport between processing stages is prohibited. Frames, ROIs, rows and card tiles remain in RAM. The golden intermediate PNG baseline is exactly **0 files /0 bytes**; the runtime guard rejects any unregistered `.png` in the fresh run directory. No debug image paths are currently registered. Explicit bounded debug images require a reviewed manifest allowlist change.

PNG **encoding in RAM** and PNG **base64 in final evidence JSONL** remain enabled. Read-only frozen template PNG files are inputs, not transport writes. This policy does not delete evidence or replace image encoding.

Outputs: `physical_rows.jsonl`, `raw_ocr_results.jsonl`, `visual_evidence.jsonl`, `removed_duplicates.jsonl`, `context_decisions.jsonl`, `ordered_timeline.jsonl`, `ordered_timeline.xlsx`, `hand_analysis.jsonl`, `validated_hands.jsonl`, `hand_summary.json`, workbook inspection output, `run_metadata.json`, `final_summary.json`, `module_timings.json`, `resource_samples.csv`, `errors_retries.json`, worker stderr logs and `production_guard.json`. These are diagnostics/events; no certified HH exporter exists in this entrypoint.

# PRODUCTION MODULE MAP

Paths below are relative to this project root. `production_manifest.json` lists the entire allowed dependency closure and hashes.

| STAGE | EXACT FILE | EXACT CLASS/FUNCTION |
|---|---|---|
| Official launch | run_production.py | main |
| Runtime/integrity/import/PNG/performance guard | parser_core/production_integrity.py | ProductionGuard; validate_integrity; RegisteredImportGuard |
| Orchestration | scripts/run_smoke14_ram.py | main |
| Decode/capture/change detection | parser_core/ram_first_stage.py | FirstStage.rows; png_compatible_gray |
| ROI coordinates | parser_core/layout.py | load_fixed_layout |
| Row spans/tracking/scroll identity | parser_core/chat_rows.py | physical_spans; exclude_first_visible_span; match_rows; match_row_blocks; align_upward |
| Native temporal overlap and immutable hand/session fence | parser_core/temporal_overlap.py | TemporalObservations.reduce; BoundaryReducer.process |
| Source/session stable identifiers | parser_core/journal.py | stable_id (journal storage is not used here) |
| Existing ROI helpers | parser_core/visual_crops.py | snapshot_ram_board (retain_board=False; no board PNG capture) |
| OCR queue/order linkage | parser_core/parallel_ocr.py | ParallelRowOCR; row_key |
| Tiny process client | parser_core/tiny_chat_client.py | TinyChatClient |
| Actual Tiny recognition | scripts/tiny_chat_worker.py | main |
| VL | NONE | disabled; no VL client/model/dispatcher |
| Card queue/candidate eligibility | parser_core/visual_tile_queue.py | VisualTileQueue.submit; VisualTileQueue._run; card_tile_candidate |
| Card geometry/glyphs | parser_core/board_tiles.py | card_spans; secondary_card_spans; _glyph_boxes; _glyphs |
| Rank/suit/full-card combination | parser_core/card_hybrid.py | CardHybrid.read_bytes |
| Frozen rank NCC | parser_core/rank_ncc.py | RankNCC.read |
| Suit BACKUP/conflict gate | parser_core/suit_backup.py | SuitBackup.read; decide_suit |
| Native component suit fallback | parser_core/component_suit.py | ComponentSuitBackup.read; component_feature |
| Registered shared NCC dependency | experiments/suit_template_experiment.py | features; predict ONLY (existing dependency, hash pinned; experiment main never invoked) |
| Text event candidates | parser_core/normalize_chat.py | normalize |
| Multi-row message diagnostics | parser_core/message_assembly.py | group_rows |
| Event binding/projection | parser_core/visual_cards.py | associate_visual_evidence; project_chat_card_events |
| Strict source hand assembly | parser_core/poker_hand_assembly.py | assemble_hands (pixel cards and observed adjacency required) |
| Deterministic hand analysis | parser_core/poker_pipeline.py | analyze_timeline |
| Poker action/state rules | parser_core/poker_logic_engine.py | validate_hand |
| Decimal observed money ledger | parser_core/poker_math_engine.py | audit_observed_ledger; validate_math |
| Physical Hold'em card evaluation | parser_core/poker_card_engine.py | best_rank |
| First-break and uncertainty diagnostics | parser_core/poker_diagnostics.py | diagnose_hand |
| Excel export | scripts/build_smoke14_ram_workbook.mjs | top-level SpreadsheetFile/Workbook export |
| Metrics | scripts/run_smoke14_ram.py | Telemetry.add; Telemetry.report |
| CI integrity | scripts/check_production_integrity.py | main |
| CI workflow | .github/workflows/production-integrity.yml | portable integrity; optional equipped Windows runner tests |
| Tests | tests/test_*.py exact list in manifest | python -m pytest; integrity tests in tests/test_production_integrity.py |

Production configs: `ocr_tiny_local.json`, `card_hybrid_local.json`, `suit_backup_local.json`, `rank_ncc_production.json`, `nine_table_layout.json`, `layout_847x404.json`, `chat_log_bottom_rois.json` in `configs/`. Fixed glyph model: `models/board_glyphs_v1.json`. Frozen TRAIN431 manifest and every template PNG, selected-method JSON and suit validation CSV are registered **read-only data inputs**. Their original config hashes/provenance remain checked by existing readers. Tiny Python/model and Node/export package are machine-local runtime dependencies recorded in the manifest; they are not substitute parser implementations.

# FORBIDDEN SOURCES

`experiments/` (except the one explicit shared helper), `legacy/`, `archive/`, saved old implementations, code under `runs/` and `outputs/`, benchmark copies, temporary scripts and `tests/fixtures/` are **not production implementations**. Frozen data files under runs are authorized only as named read-only inputs.

Other capture runners, `scripts/capture_video.py`, `scripts/capture_nine_tables.py`, `scripts/run_nine_tables_streaming.py`, benchmark runners and historic `run_xxx_v2` scripts are **LEGACY / EXPERIMENT / INTERNAL ONLY** relative to this production. They may be evidence or experiments; they are not interchangeable entrypoints. `scripts/run_smoke14_ram.py` is the registered internal implementation and refuses direct production launch.

**A newer timestamp or similar filename does NOT make a file production.** A search result, filename, report or model output cannot override this manifest.

# HOW TO CHANGE PRODUCTION

`GOLDEN PRODUCTION -> EXPERIMENT -> TARGETED TEST -> PARITY/CORRECTNESS -> PERFORMANCE TEST -> REVIEW -> MINIMAL DIFF INTO PRODUCTION -> FULL PYTEST -> GOLDEN SMOKE -> UPDATE PRODUCTION.md + manifest -> NEW GOLDEN`.

An experiment can test against production. It cannot replace the production runner. Transfer only the smallest proven change into the registered production implementation. Preserve pixels/OCR/provenance, unresolved outcomes and event gates. Never tune thresholds on untouched acceptance data.

Before a change: retain the existing golden snapshot and dirty worktree. After review/tests/smoke, deliberately update documentation, the exact stage/dependency list and hashes. `python scripts/freeze_production_manifest.py --reviewed-freeze --version production-golden-UNIQUE --benchmark-run runs/REVIEWED_500_FRAME_RUN --benchmark-manifest PATH_TO_SAVED_CANDIDATE_MANIFEST --benchmark-video video/smoke_17.mp4` is an **INTERNAL ONLY** explicit freeze utility; it is never invoked by runtime or CI. A new unique version and provenance destination are required, and an existing destination is rejected. The measured candidate must match the saved manifest, promoted runtime/config bytes and frozen source, finish 500 frames with cards enabled, and export Excel without errors. Never overwrite a previous golden with new code silently. CI strict hashes must pass. If unrelated dirty edits prevent an unambiguous commit, keep a byte snapshot/hash manifest; do not invent a clean tag.

# RUNTIME AND CI INTEGRITY

The official launcher validates existence, schema, exact registered stage paths/symbols, local import closure and config bindings before loading OCR or decoding video. Any project-local implementation must be registered, regardless of module name. A runtime import finder catches dynamic unregistered/foreign modules; an execution audit hook also checks direct file loaders against registered code and installed Python-library roots. This is an accidental-composition guard, not a hostile-code sandbox. Only the one already-used NCC helper under experiments is exempt. Changed hashes/HEAD print **WORKTREE MODIFIED** and file names; structural/config substitutions fail. CI uses strict hashes and rejects stale registration/documentation. Guard-abort metrics and FAILED status are retained in artifacts.

`pytest.ini` excludes archived provenance/run/output copies from recursive test discovery; every existing active test remains enabled. Golden snapshots are evidence, not a second executable test suite.

Runtime PNG checks run during decoding (every50 frames) and at finish. Early performance gate reports500-frame capture numbers. Fingerprint values come from manifest/current Git/files, not decorative constants. `production_guard.json` preserves measurements. The protection layer does not alter OCR/recognition/capture. The registered temporal reducer filters only proven duplicates before rebuilding message/event/card annotations and Excel; raw evidence stays available.

Local checks:

```powershell
python scripts/check_production_integrity.py
python -m pytest tests/test_production_integrity.py -q
python -m pytest -q
```

No prior tracked CI existed. The new portable workflow runs strict integrity and isolated negative guard tests. Full existing tests require this project's frozen ignored data and machine-local runtimes; the optional self-hosted Windows job is explicitly equipped for those, rather than pretending a clean hosted checkout contains them. Local full pytest and500-frame production smoke are required for golden promotion. See `runs/production_integrity_20261001/` for actual checks; workflow execution on a remote service is not claimed without run evidence.
