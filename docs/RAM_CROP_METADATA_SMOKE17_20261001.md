# RAM crop metadata — Smoke17

## FILES CHANGED

New files only (existing dirty files preserved):

- `D:\Новый парсер\experiments\ram_crop_metadata.py`
- `D:\Новый парсер\experiments\run_ram_crop_metadata.py`
- `D:\Новый парсер\tests\test_ram_crop_metadata.py`
- `D:\Новый парсер\docs\RAM_CROP_METADATA_SMOKE17_20261001.md`

## CURRENT CROP LIFECYCLE

Authority: `PRODUCTION.md` and `production_manifest.json`, production-golden-20260930. Official production entrypoint remains `run_production.py:main`. The experiment is a capture-only diagnostic, not another production launcher or a promoted implementation.

Verified FILE → FUNCTION → CALLER → DATA:

1. `run_production.py:main` validates the manifest and loads registered `scripts/run_smoke14_ram.py:main`.
2. Runner constructs `parser_core/ram_first_stage.py:FirstStage` and consumes `FirstStage.rows`.
3. PyAV `source.decode(stream)` → `frame.to_ndarray(format="rgb24")`: native RGB NumPy frame. PTS × rational time_base defines source time.
4. Table/chat ROI slices are NumPy views. `previous_rgb` holds contiguous ROI arrays for change comparison; gray conversion creates a separate grayscale array. `chat_history` is a deque bounded to 12 states per session. It holds grayscale states, spans, identity and lazy canonical representations.
5. `physical_spans`, `exclude_first_visible_span`, `match_rows`, `align_upward`, `match_row_blocks` assign existing stable IDs. The excluded first visible source span does not become a new track/event.
6. New rows use `padded_row_span`, `np.ascontiguousarray(gray[slice])`, SHA256 of raw bytes, and immutable `RamRow`. A contiguous row can share its gray array allocation; byte payload and backing-array ownership are different measurements.
7. Inherited rows invoke `identity_observer(metadata, pixels)` with a separate `np.array(..., copy=True, order="C")`. New rows are yielded. Observer cannot alter capture pixels or identity.
8. `ParallelRowOCR.submit_batch` → bounded queue (32 batches, 8 rows each by current defaults) → `_run` → `cv2.imencode` → RAM base64 → `TinyChatClient.read_batch` → `tiny_chat_worker.py:main` decodes RAM payload and invokes Tiny OCR. No image file round trip.
9. `VisualTileQueue` uses bounded jobs and `image_transport="ram"` in registered runner. PNG encoding stays in RAM; visual evidence retains base64 PNG in final JSONL. Its separate file mode/write site is not selected by production.
10. Runner retains emitted `RamRow` objects in `rows` until processing/export. It writes raw row bytes as base64 in `physical_rows.jsonl`, raw OCR separately, normalizes/groups source rows, writes ordered timeline, then invokes `build_smoke14_ram_workbook.mjs` for Excel. Board snapshots are disabled (`retain_board=False`).

Thus registered production already creates zero intermediate image files, but its end-to-end image lifetime is not bounded by streaming queues alone: the `rows` list remains an owner. This task does not change that existing production retention or its exported evidence. No claim of bounded whole-production RSS follows from the new layer.

## EXISTING METADATA

| Field | Registered production | Existing experiment | Reuse |
|---|---|---|---|
| frame/source frame | EXISTS: `frame_id` | EXISTS | same integer |
| PTS/time base | EXISTS: `source_pts`, `time_base_num/den` | EXISTS | same source authority |
| timestamp | EXISTS: `video_time` seconds; ms representation PARTIAL | EXISTS: `timestamp_ms` | existing `source_ms`, no fps inference |
| table/session | EXISTS | EXISTS | same IDs; table label is presentation |
| row key | EXISTS: `parallel_ocr.row_key` | EXISTS | same table:frame:span value |
| stable row ID | EXISTS | EXISTS | track_id is the same identity, no independent counter |
| pixel hash | EXISTS: `pixel_sha256`/`row_pixel_sha256` | EXISTS | raw pixels; latest view and original OCR parent distinguished |
| OCR confidence | EXISTS: Tiny score → timeline confidence | EXISTS | source-linked cached confidence, no OCR rewrite |
| normalized text | EXISTS | EXISTS | cached unchanged text |
| first/last seen ms | MISSING | EXISTS | same lifetime definition |
| observation count | MISSING | EXISTS | decoded visible frames; explicit views counted separately |
| visibility spans | MISSING | EXISTS | disjoint frame intervals, no gap interpolation |
| ROI position | EXISTS: span index and y1; full absolute box PARTIAL | EXISTS | no new coordinate system |
| temporal class/context | MISSING | EXISTS: offline `classify` | existing research rule/cached analysis only |

`epoch` is alignment state, not a hand ID. First seen is visible observation, not server action time. Known temporal fields were reused, not duplicated with different semantics.

## IMPLEMENTATION

`RamMetadata` is an opt-in experimental sidecar over unchanged `FirstStage`. `CropView` borrows pixels only inside a synchronous context manager. It exposes a scalar metadata snapshot; nested spans are immutable copies. Track summaries use the existing stable_row_id. Table/session mismatches, invalid frame order and completion during a live borrow fail explicitly.

`watermark` formats table/session/frame/source time, row key/track, first/last seen, observation count and class as text. `debug_overlay` builds an independent NumPy canvas and draws only on that canvas. It performs no filesystem IO. Live views without confirmed temporal context are AMBIGUOUS. Cached offline labels are attached only after full capture parity with source-linked temporal inputs.

`run_ram_crop_metadata.py` replays the pinned capture code, validates source video and baseline hashes, compares every emitted row's raw bytes/hash/identity/PTS/order, and completes every decoded frame via existing telemetry. Changed empty ROIs remove active tracks; unchanged ROIs extend active metadata without materializing extra crop objects. Metadata survives images; images are not cached. Full lifetimes are compared against the prior independently executed temporal capture. Source OCR hash/frame/key are exposed separately from the latest visible crop hash/frame/key.

No alternative recognizer, poker boundary rule, automatic deletion/merge, edge-gate decision, Jev acceptance, math change or Excel row removal was introduced. No frozen/vendor files changed.

## RAM LIFECYCLE

Image: existing capture NumPy crop/copy → borrow → hash/metadata → release in `finally`. One live crop reference allowed in the new layer; cache bytes = 0. An optional overlay belongs to its caller and is released by dropping that RAM reference. The new layer does not retain completed-track images.

Metadata: one dictionary per existing track, active track IDs per session, and pending IDs for the current changed state. After finish active/pending state is cleared; track summaries remain for downstream text review. Metadata count scales with tracks, not all image observations. No arbitrary RAM cache limit was selected because no image cache is needed.

The experiment reuses stored source PTS records for unchanged frames after verifying the exact source video; inherited/new observations check frame/PTS linkage. This is diagnostic replay with cached OCR, not a fresh production recognition run.

## DISK IMAGE WRITES

Full filesystem audit PASS: 296,336 existing image files before and after under all `runs/` and `outputs/`. Existing inputs excluded by full path; modified files also checked.

```text
NEW PNG FILES = 0
NEW JPG FILES = 0
NEW WEBP FILES = 0
IMAGE FILES CREATED = 0
MODIFIED IMAGE FILES = 0
```

JPG includes `.jpg` and `.jpeg`. Receipt: `runs/ram_metadata_smoke17_20261001/image_inventory_audit.json`. New-layer output is text JSON/JSONL; no new pixel/base64 image payload is serialized. Capture process-write counter: 2,462 bytes of stdout progress text. Text artifacts before summary/audit completion: 11,255,144 bytes; this is not the full final directory byte count.

## SMOKE17 RESULT

Full replay exit 0: 28,222 / 28,222 frames, 8,146 / 8,146 original physical rows and track lifetimes, 950,047 synchronous crop observations. Raw bytes/hash/identity/PTS, capture chronology and production export order matched. OCR launches: 0.

Post-validation exit 0: all 178,633 original fields of 8,146 timeline rows preserved; every cached OCR text/confidence/normalized text and original frame/key/hash linked to its physical source. No CURRENT/tail attribution with unresolved identity or a visibility gap. Source/baseline/temporal-input hashes and unchanged registered-production hashes are retained in `summary.json`.

| Full-run metric | Value |
|---|---:|
| capture seconds | 856.419529 |
| preflight seconds | 60.442636 |
| total experiment seconds before summary | 967.722775 |
| PTS span (last minus first frame) | 940.800 s |
| capture RTF | 0.910310 |
| capture + preflight RTF | 0.974556 |
| total experiment RTF | 1.028617 |
| process RSS start / sampled peak / growth bytes | 348,700,672 / 499,273,728 / 150,573,056 |
| new-layer metadata objects | 8,146 |
| peak borrowed image references / payload bytes | 1 / 3,680 |
| image references / image cache bytes after finish | 0 / 0 |
| final and sampled metadata deep-size bytes | 15,878,009 |

Metadata deep-size is about 15.14 MiB. It includes all track summaries and cached OCR/context, not original images. Deep-size sampled every 1,000 frames and at final enrichment; exact allocation peak NOT_MEASURED. RSS is sampled every 100ms and includes decoder, pinned FirstStage and baseline oracle; it cannot be attributed entirely to this layer. No completed-track images remain in this layer.

Receipts: `runs/ram_metadata_smoke17_20261001/summary.json`, `post_validation.json`, `tracks.jsonl`. The diagnostic validator source `post_validate.py` is stored in that run as an evidence artifact with its SHA256; it is not a production dependency.

## TEMPORAL REVIEW

`runs/ram_metadata_smoke17_20261001/temporal_review.jsonl`: exactly 20 PREVIOUS_HAND_TAIL, 20 CURRENT_HAND and 20 AMBIGUOUS records; post-validation PASS. Each includes table/session, last explicit crop frame/key/PTS/time/hash, existing stable track ID, first/last seen, visible spans, observation count, classification/reason, original OCR parent frame/key/hash and boundary source IDs. Text watermark supplied for every record. No review images saved or new human adjudication claimed.

Existing raw OCR boundary authority is research evidence, not human certification. PREVIOUS labels mean earlier visible context; they do not identify the immediately preceding numbered hand and do not require that the original timeline row itself still be visible at a later start. Persistent track/start pairs are counted separately. Source candidate labels remain relative to their cached analysis interval; they are not a live classifier for every future hand.

| Existing research class | Tracks/source rows |
|---|---:|
| PREVIOUS_HAND_TAIL | 142 |
| CURRENT_HAND | 1,578 |
| NEXT_HAND_HEAD | 0 |
| AMBIGUOUS | 6,426 |
| TOTAL | 8,146 |

## EDGE-GATE USEFULNESS

Observed with the unchanged existing identity and temporal analysis:

- 296 persistent previous-context track/confirmed-start pairs: physical visibility extends across the boundary. This is distinct from the 142 cached original-source tail labels.
- 941,901 repeated explicit crop observations (950,047 views minus 8,146 tracks), and 4,403,347 repeated decoded-frame observations: lifetime observations, not additional poker events.
- 2,999 tracks originated as UNRESOLVED_IDENTITY; 61 have disjoint visibility gaps. Their temporal uncertainty is preserved. These counts may overlap and must not be added as disjoint categories.
- 590 cross-track candidate pairs; all 590 remain AMBIGUOUS_CROSS_TRACK_IDENTITY. Text/time/similarity does not prove a merge or new event.
- Parser/edge-gate decisions changed: 0. No timestamp-only hand attribution introduced.

Usefulness: provenance and repeats within existing stable IDs are measurable; cross-track identity and complete boundaries remain insufficient. Edge-gate automation remains PARTIAL and unchanged.

## PERFORMANCE

Identical 500-frame Smoke17 fragment; both exit 0, 349 emitted rows; exact pixel/hash/ID/PTS parity. Capture order SHA256 and both experiment file hashes are equal across A/B.

| Metric | BASELINE | RAM METADATA |
|---|---:|---:|
| capture seconds | 16.932287 | 17.007852 |
| capture overhead | — | +0.075566 s / +0.446281% |
| RSS peak bytes | 379,387,904 | 378,949,632 |
| capture RSS growth bytes | 102,129,664 | 102,113,280 |
| emitted physical rows | 349 | 349 |
| materialized inherited debug views | 0 (observer off) | 21,757 |
| total new-layer crop borrows | — | 22,106 |
| new image files / modified image files | 0 / 0 | 0 / 0 |
| process capture write bytes (stdout progress) | 83 | 83 |
| new-layer peak live images / payload bytes | — | 1 / 3,680 |
| new-layer images/cache bytes at end | — | 0 / 0 |
| final metadata deep-size bytes | — | 476,699 |
| preflight seconds | 57.801253 | 59.289759 |
| total process seconds before summary | 124.524301 | 129.056190 |

Receipts: `runs/ram_metadata_ab_baseline_20261001_v3/summary.json`, `runs/ram_metadata_ab_new_20261001_v3/summary.json`. Single paired measurement, not a statistical speed guarantee. The small negative RSS difference is sampling/startup variation, not proof of memory savings.

RSS includes test oracle and FirstStage state; it is not new-layer memory. New-layer image payload/reference metrics and sampled deep-size metadata are reported separately. Capture runtime excludes preflight hashes/inventory, source-baseline loading, OCR/export and final text serialization. Preflight and total process costs are exposed separately. Process write-byte counters include progress output; image filesystem counts are authoritative for the image prohibition. No large overhead was observed in this pair.

## TESTS

Targeted pytest after audit fixes: 51 passed. Production integrity PASS before replay. Independent reviewer reproduced snapshot sharing and future-frame acceptance; both fixed with isolation/order tests, plus completion-during-borrow rejection. Re-audit found no remaining material findings.

An initial harness-only baseline assertion compared capture order with sorted export order and failed after row bytes/count/PTS parity had passed. Corrected to validate each distinct ordering contract. The incomplete `_v2` artifact is preserved and excluded from successful A/B evidence. Capture/image/identity algorithms were not changed to accommodate that assertion.

## PRODUCTION CODE CHANGED

NO. Existing golden and dirty worktree preserved. No production promotion or registration was performed.

## VERDICT

**RAM_CROP_METADATA_PARTIAL**.

The implemented experimental row layer passed its full Smoke17 safety contracts: pixels/OCR preserved, zero image writes, identity/time/lifetime parity, synchronous image release, light measured A/B overhead, and complete text review. However it has not been promoted into the manifest-pinned production runner, and it does not wrap every table/chat ROI or card tile. Existing production still retains its `rows` image list to export. Whole-production bounded lifecycle and precise new-layer allocation peak are not proved by this capture/cached-analysis experiment. These scope limits prevent an unqualified RAM_CROP_METADATA_PROVEN_SAFE verdict for the entire requested crop/evidence pipeline.

No production promotion, automatic edge-gate/math connection, acceptance change, or new OCR run followed this report. Stop at the requested research/implementation/test result.
