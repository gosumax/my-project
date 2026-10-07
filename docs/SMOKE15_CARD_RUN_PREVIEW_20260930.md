# Smoke15: architecture and measurement contract before launch

Status: prepared, **full Smoke15 not started**. A user decision to launch is required.
The source is `video/smoke_15.mp4` (20,326 frames, 2560×1440, 30 fps;
sample-reported SHA-256 `290654e4b07fa693235d4840421ae539c8da869a81f7ae60c6ccfed1093533f8`).
The full command verifies the actual source SHA against the capture journal.

## FILE → FUNCTION → CALLER → DATA

1. `scripts/capture_nine_tables.py:main` reads the exact source video once into a
   fresh baseline journal with table/chat PNG, frame PTS, ROI boxes, source and
   crop SHA, and nine provisional slot sessions. `parser_core/visual_crops.py:
   crop_saved_table` uses this journal for BOARD crops. No full Smoke15 baseline
   exists yet; the bounded sample folders are not interchangeable with it.
2. `scripts/run_smoke14_ram.py:main` calls `parser_core/ram_first_stage.py:
   FirstStage.rows`. This second decode finds physical CHAT rows in RAM. It
   preserves table/session, frame, source span, PTS, pixel SHA, stable row ID,
   origin and epoch. First-visible-row suppression affects derived rows only.
3. The main loop sends Tiny OCR batches to `parser_core/parallel_ocr.py:
   ParallelRowOCR.submit_batch` (two persistent CPU workers, batch eight,
   bounded queue 32) and card candidates to `parser_core/visual_tile_queue.py:
   VisualTileQueue.submit` (one persistent VL worker, bounded queue 256).
   One VL response is tied to one saved full CHAT-row crop by row key and SHA.
4. `VisualTileQueue._run` keeps the existing `board_tiles.read_board_tiles`
   reading and raw VL response. `parser_core/card_hybrid.py:CardHybrid.read`
   obtains rank tokens from that **same whole-row VL response**, uses the
   existing fixed CHAT span geometry, and reads `raw_suit2` from the 431
   SHA-checked TRAIN tiles (Smoke12/13). Each derived CARD tile gets its own
   path, byte SHA, pixel SHA, x box, rank and distance evidence. Incorrect
   count, duplicate card, rank conflict, suit conflict, geometry failure or
   missing evidence is explicit `UNRESOLVED`.
5. `parser_core/suit_backup.py:SuitBackup` independently computes grayscale
   NCC on native CARD ROI `[10,19)×[2,13)`, linearly resized to 16×16.
   The exact feature/scoring functions are reused from
   `experiments/suit_template_experiment.py`. The only templates are the
   pinned Smoke12/13 TRAIN manifest. `OFF` disables NCC, `SHADOW` records it
   without changing the existing card result, and default `BACKUP` may recover
   a suit if score ≥0.98 and margin ≥0.03. Confident disagreement with the
   accepted primary reading yields `SUIT_CONFLICT`, with both readings kept.
   These scores are similarities, not probabilities. The thresholds passed
   312/318 checked validation tiles; all 318 predictions were correct before
   gating. Smoke16 figures are disclosed diagnostic results, not new blind
   acceptance.
6. `parser_core/visual_cards.py:associate_visual_evidence` joins the read
   to a nearby trigger only in the same table/session/epoch/frame and physical
   row order. It records hand, street, player and trigger row key even for a
   conflict. `board_roundtrip` checks exact BOARD crop→response identity;
   `project_board_events` requires its SHA round trip and accepted evidence.
   Neither card audit nor a partial timeline certifies HH.
7. `scripts/build_smoke14_ram_workbook.mjs` exports `ordered_timeline.xlsx`.
   `ALL_EVENTS` and `T01`–`T09` preserve the physical timeline. `CARD_AUDIT`
   has one line per candidate CARD tile (or unresolved card row), including
   row/crop/tile identity, raw VL text, primary and NCC suits, scores, margin,
   source, status, card and conflict. The complete structured raw VL response
   remains in `visual_evidence.jsonl`, and Tiny response in
   `raw_ocr_results.jsonl`; Excel is a review view, not the source journal.

## Launch commands after approval

Both output directories must be fresh. The capture is a prerequisite for
BOARD provenance and stable slot sessions. It is a full-video job itself.

```powershell
python scripts/capture_nine_tables.py video/smoke_15.mp4 `
  --output C:/ParserData/runs/smoke15_full_capture_20260930 `
  --max-frames 20326 --capture-workers 4
python scripts/run_smoke14_ram.py --video video/smoke_15.mp4 `
  --baseline C:/ParserData/runs/smoke15_full_capture_20260930 `
  --frames 20326 --output runs/smoke15_card_hybrid_full_20260930 `
  --ocr-workers 2 --ocr-queue-size 32 --vl-workers 1 --vl-queue-size 256 `
  --card-hybrid-config configs/card_hybrid_local.json `
  --suit-backup-config configs/suit_backup_local.json `
  --node C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe
```

## Measurements and acceptance boundaries

| Module | Saved measures | Quality / missing reference |
|---|---|---|
| Capture / frame decode | frames, source SHA/PTS, changed and reused ROI, PNG/DB write and commit seconds, source/crop bytes | frame/ROI/hash/order parity against source; no hand quality |
| RAM first stage | per-operation calls, total/mean/median/p95/max seconds and objects/s for decode, ndarray, ROI slice/change compare, grayscale, spans, alignment, history, row hash; changed/unchanged ROI, row count, first-stage fps | physical row identity, origins/epochs, GAP/UNRESOLVED; frame/table/row parity must be checked |
| Tiny CHAT OCR | rows, batch size, worker count, queue maximum, OCR wall, rows/s, capture overlap, errors, empty/UNKNOWN/UNRESOLVED counts; raw response per row | text quality requires human labels; coverage is not accuracy |
| VL whole-row / BOARD | candidate requests, replies, errors, max queue, request wall and requests/s, crop/response SHA, BOARD and SHOWDOWN counts | strict accepted/unresolved counts, rank alignment, trigger association, BOARD SHA round trip and continuity; no rank accuracy without reviewed truth |
| CARD segmentation / rank | candidate rows, span count, rank token count/alignment, rank conflicts, duplicate cards, derived CARD PNG/pixel hashes, per-row hybrid time and rows/worker-s | resolved cards per candidate card and reason counts; rank accuracy only after manual review |
| `raw_suit2` | nearest TRAIN card, best and other-card distances, tie count, method/manifest SHA | no calibrated confidence; discrepancy with NCC retained |
| NCC suit backup | best/second scores, margin, threshold pass, primary/template agreement, backup, low score, conflict; OFF/SHADOW/BACKUP in config | coverage and disagreements; suit/exact-card error rate after user checks Excel |
| Association / Excel | grouped messages, timeline rows, projected BOARD events, missing trigger/hand/player, workbook row counts and builder seconds | verify row key, crop SHA and response identity; HH remains partial |
| System and bottlenecks | end-to-end wall/fps/RTF, per-stage wall, process/group CPU and RSS samples, disk read/write bytes, PNG count/bytes, GPU utilization and used/total VRAM samples when `nvidia-smi` is available | `resource_samples.csv`, `module_timings.json`, `final_summary.json`; overlapping stage seconds must not be summed into total wall |

`run_metadata.json`, `final_summary.json`, `module_timings.json`,
`resource_samples.csv`, `errors_retries.json`, `ordered_timeline.jsonl`,
`ordered_timeline.xlsx`, `visual_evidence.jsonl`, `raw_ocr_results.jsonl`,
`visual_crops/`, `card_tiles/`, `board_crops/`, worker stderr logs and the
baseline capture journal/PNG form the output set. GPU metrics are null if
the utility is unavailable. A bottleneck verdict must use measured wall,
queue occupancy and resource samples from this run; validation accuracy or
Smoke16 diagnostics cannot substitute for Smoke15 review.
