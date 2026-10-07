# Cross-track identity — Smoke17

## FILES CHANGED

New files only:

- `D:\Новый парсер\experiments\cross_track_linker.py`
- `D:\Новый парсер\experiments\run_cross_track_linker.py`
- `D:\Новый парсер\tests\test_cross_track_linker.py`
- `D:\Новый парсер\docs\CROSS_TRACK_IDENTITY_SMOKE17_20261001.md`

Diagnostic inspection script/data are under `runs/cross_track_inspection_20261001/`. New results belong to `runs/cross_track_linker_smoke17_20261001/`. These are experiments/evidence, not production. Only `run_production.py` remains the production entrypoint defined by unchanged `PRODUCTION.md` and manifest.

## IDENTITY BREAK CAUSES

Common primary mechanism, verified FILE → FUNCTION → CALLER → DATA:

`parser_core/chat_rows.py:match_rows/align_upward/match_row_blocks` → called by `parser_core/ram_first_stage.py:FirstStage.rows` → immediate viewport mapping, alignment minimum-match gate and up to 12 history states → unmatched rows not proven newly exposed become `UNRESOLVED_IDENTITY` → `_stable_id` assigns a fresh session-counter ID.

All 590 candidate new tracks have this origin. This proves the common failure-to-carry-identity path, not that all 590 represent the same physical row. Existing tracker requires geometry/order/pixel agreement; repeated single lines are deliberately insufficient. `epoch` changes on unproven viewport alignment and is not a poker hand ID.

Independent saved-data audit:

- 590 pairs, 972 involved IDs, missing IDs 0; all same table/session, and every proposed old first observation precedes new first observation.
- 355 full-lifetime handoffs end one frame before new first; 231 have a larger gap; four terminal envelopes interleave. All 590 have zero actual simultaneous visibility, independently checked against disjoint spans and 54,631 chat states.
- In four interleaved cases new occupies an old-track visibility gap of 8 / 311 / 21 / 1 frames; old reappears next frame. Later reappearance is retrospective audit information and cannot feed an earlier linking decision.
- Either endpoint has a visibility gap in 12 pairs. Canonical old origins: INITIAL_VISIBLE 14, NEW_CANDIDATE 334, UNRESOLVED_IDENTITY 242.
- 950,047 saved track views parsed; 68,900 involve the candidate IDs. Saved views carry per-view hash/position but not full viewport pixels or historical accepted mapping traces.

Thus a complete historical history-search rejection cause cannot be recovered from the saved metadata alone. The experiment regenerates causal source ROIs at exact PTS and records both immediate transition rejection and observed movement/pixel changes. Counts below will distinguish overlapping observations from exclusive proven causes; UNKNOWN remains explicit.

Real source-ROI observations: scroll/vertical shift 501, ROI change 590, pixel hash change 590, ambiguous immediate alignment 14, causal-prefix visibility gap 1. These overlap and do not establish accepted physical links.

## 590 PAIRS BEFORE

SAME: 0; DIFFERENT: 0; AMBIGUOUS_CROSS_TRACK_IDENTITY: 590. Sources are legacy boundary/guarded proposals and saved resolver hypothesis packets; none is an accepted physical link. Source memberships overlap: boundary 192, guarded 506, offline generator 69, saved-request replay 65.

## 590 PAIRS AFTER

Real 590-pair test completed: SAME=0, DIFFERENT=0, AMBIGUOUS=590. Conditions and rejection dependencies are audited separately in `CROSS_TRACK_CONDITION_AUDIT_SMOKE17_20261001.md`.

## RULES USED

Frozen `TrackSnapshot` copies nested visibility spans into immutable tuples. Snapshot endpoints/spans/source frames after the decision frame are rejected. Motion endpoints must match actual causal old last-view/new first-view frames, y positions and source row index. Boundary frames must lie strictly after old last and through new first. Terminal full-run metadata is retained separately as future audit information.

SAME_PHYSICAL_TRACK_PROVEN requires jointly:

1. Same table/session; no causal simultaneous visibility; one-frame handoff, or a two-frame handoff with observed one-frame local raster loss/restoration and unchanged whole context outside the target.
2. Existing `align_upward` confirms viewport geometry; existing `match_rows` and an ordered `match_row_blocks` block of at least three support the target correspondence.
3. Target raster is exact after the existing 12px scrollbar exclusion, unique across both viewports, and mapped to the proposed new row. At least two exact neighbor anchors move in the same viewport transform.
4. Cached normalized text agrees; no separating confirmed hand boundary. Text/time/similarity alone never qualifies.

DIFFERENT_PHYSICAL_TRACK_PROVEN requires a positive physical conflict: different table/session; distinct positions simultaneously visible at the decision frame; or adjacent physical continuity proving old continues at another row position with two neighbor anchors. A distant new onset requires immediate proven bottom append, two neighbor anchors and a confirmed boundary in the separation interval. Long gap or changed pixels/text alone is insufficient.

Everything else is AMBIGUOUS. Pixel/text similarities are diagnostic features, not probabilities or final acceptance scores. No tuning of existing matching thresholds or new tracker is introduced.

Images are regenerated through exact unique source PTS, not frame/fps; table/chat ROI/grayscale conversion reuse registered layout and `png_compatible_gray`. Three ROI cache slots support before/after/optional gap frame. Decoder buffers/RGB conversion remain separately owned and included in process RSS. All pixels stay in RAM, and source NumPy arrays are read-only; hash parity is checked before/after motion analysis. New first-view raw crop hash/y/span is compared to the original physical row. Original OCR parent frame/key/hash remains explicit.

## 60 CASE AUDIT

20 real AMBIGUOUS cases audited from saved causal evidence. SAME/DIFFERENT samples available: 0 each, shortfall 20 each. Read-only review found no missed classification under the frozen predicates; this is not independent visual ground truth.

The audit JSONL records old/new causal snapshot, terminal metadata separately, PTS-derived timestamps, frame/position deltas, raw/normalized OCR/confidence, pixel hashes/similarity, disjoint visibility spans, neighbor track mapping, alignment scores, boundaries, verdict, reason and all source frames used. No PNG review files.

## EDGE-GATE SIMULATION

Existing `parser_core/edge_gate.py:classify_edges` runs twice on the same original timeline: first without relationships, then on copies with only proven SAME relationship metadata attached. All source rows/fields/IDs/epochs remain intact. The current gate has no relationship consumer, so linking alone is not authority to remove repeated hand markers, normalize epochs, invent boundaries or change admission. Potential rows with useful relationship evidence are counted separately from safely reclassified rows.

UNRESOLVED_EDGE 6685 -> 6685; safely resolved 0; decisions changed 0; source rows 8146 -> 8146. Nothing applied.

## PERFORMANCE

Same-input 590-pair analysis, 20 repeats per mode: metadata median 0.025353s; linker median 0.047709s (+0.022356s /88.177%). Source ROI extraction/decode/matching 281.852324s; total 409.758975s. These repeated timings exclude extraction and are not video RTF. 1777 ROI requests /13153 seek-decoded frames; three live ROI arrays /260820 bytes maximum, zero at end. RSS extraction-phase sampled peak 393707520 bytes versus start 319062016; repeat RSS sampled at repeat boundaries only. Global image inventory unchanged; new PNG/JPG/WEBP 0.

## TESTS

`python -X utf8 -m pytest tests/test_cross_track_linker.py tests/test_ram_crop_metadata.py tests/test_chat_rows.py tests/test_temporal_evidence_analysis.py -q`: exit 0, 52 passed, 1.25s. Production integrity PASS.

13 new linker tests cover scroll with multiple neighbors, one-frame local loss, same text with separately proven new onset, simultaneous rows, different session, missing motion/anchors, uniqueness, future source frames/lifetimes, immutable nested spans, unchanged original pixels/IDs, stale motion endpoints, unrelated boundary rejection and runner current-frame coexistence.

Independent read-only audit found and corrected stale motion, future terminal lifetime, historical boundary and runner coexistence bugs; re-audit found no remaining material issues before the real experiment.

## SOURCE DATA CHANGED

NO. Completed source/hash assertions passed. No source row deletion or physical merge. No source OCR/pixels/stable-ID rewrite.

## PRODUCTION CODE CHANGED

NO. Manifest, frozen/vendor, OCR/cards/math/Jev and edge-gate implementations/decisions unchanged. Relationship data remains experimental.

## VERDICT

CROSS_TRACK_IDENTITY_NOT_ENOUGH: no proven physical relationship from the 590 candidates. No automatic edge-gate/math connection follows this experiment. The follow-up condition audit measures the missing evidence without changing the rules.
