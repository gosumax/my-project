# Smoke15: RAM, CHAT cards and parallel workers

This launch supersedes the disk capture prerequisite in
`SMOKE15_CARD_RUN_PREVIEW_20260930.md`. The user explicitly authorized parallel
worker changes and restricted VL inference to CHAT card rows.

## Actual path

`run_smoke14_ram.py` -> `FirstStage.rows` -> physical rows in RAM -> Tiny OCR
and independent CHAT card candidate queue -> one persistent VL model process
with batches of up to eight CHAT crops -> CardHybrid VL ranks + primary pixel
suits + existing NCC BACKUP -> strict row/frame/session association -> Excel.

No BOARD ROI is submitted to VL. A `BOARD` event can still be read from cards
shown inside a CHAT row. The standalone streaming runner is excluded.

Capture runs alongside Tiny and the visual queue. Eight visual callers share
the single GPU dispatcher; two card-processing slots bound CPU concurrency.
VL loads at startup rather than at the first queued crop. Event assembly runs
while outstanding VL work drains. Excel export waits for all evidence.

Tiny retains the existing globally ordered batches of eight. Only complete
T01 batches can be submitted before capture ends without changing those
batches; remaining tables drain afterward. Changing batch composition would
change model padding and has previously changed raw OCR. Worker parallelism
does not authorize that quality change.

## Preserved decisions and evidence

NCC config, templates and algorithm are unchanged: BACKUP, score >= 0.98,
margin >= 0.03. Full VL card strings remain diagnostic; their suit characters
cannot fill a rejected hybrid CHAT reading. Conflicts remain unresolved.

For short 18/19-pixel detected tiles, the original tile/box/SHA remains intact.
The existing glyph locator identifies the suit component and an independent
20-pixel NCC query is taken from the same physical row. Query bytes, query
SHA, native suit ROI and component box are retained separately. A query is
not a claim that the actual card boundary was recovered. Unsupported or
ambiguous geometry is rejected explicitly.

CHAT identity checks source SHA returned by VL, table/session/frame/epoch,
physical row ID and pixel SHA. Raw complete cards may be exported with
`PARTIAL_HAND_CONTEXT` if the hand ID is absent. This is `RAW_UNREVIEWED`
evidence and never a certified hand history or a fabricated PASS.

## Measured preflight evidence

FirstStage integer grayscale was compared to PNG decoding for all 16,777,216
RGB values: zero differences on this OpenCV build. On 500 Smoke15 frames,
186 rows and 4256 changed ROI retain exact pixel hashes, order, identity,
timestamps and decisions. Wall: 15.0635 -> 12.1936 seconds, zero PNG files.

Frozen eleven CHAT crops: VL batch 1/2/4/8 retained identical text/status/token
counts. Model wall: 46.2406/39.2733/33.2013/29.0190 seconds. Batch eight peaked
at 2955 MiB VRAM, minimum free 1090 MiB. The worker guards free memory at
768 MiB. These measurements do not predict full-run RTF.

Evidence lives under `runs/smoke15_parallel_chat_20260930/` and
`runs/smoke15_ram_port_20260930/`. Full Smoke15 and final pytest results must
be recorded separately after execution. Smoke15 is reused regression data,
not a new independent test.

## Live 500-frame integration gate

`runs/smoke15_ram_chat_parallel_500_20260930/`: exit zero, 67.7091 seconds,
14.7736 capture, 20.1223 Tiny, 22.8027 GPU startup, 34.3133 GPU batch work,
0.10422 CardHybrid. Eleven CHAT requests, no BOARD ROI requests, no PNG files.
The three GPU batches contained 8, 1 and 2 crops.

All 186 physical rows and complete Tiny raw responses match the preceding
RAM run exactly. Eleven CHAT SHA/text/status/token records also match exactly.
Audit checked 44 original tiles and four separate native NCC queries, their
PNG/pixel SHA and source-row boxes. Workbook: eleven sheets, 186 event rows,
44 CARD_AUDIT tile rows. Four projected events contain fourteen cards;
recognized but unprojected cards remain visible in evidence.

There are 34 individually resolved tiles, three accepted NCC backups, one
suit conflict and ten unresolved tiles. No scored threshold rejects and no
geometry rejects occur in this bounded run. The conflict remains unresolved.
These counts are not the result of the full video.

## Full launch

```powershell
python scripts/run_smoke14_ram.py --video video/smoke_15.mp4 --frames 20326 --output runs/smoke15_ram_chat_parallel_full_20260930 --ocr-workers 2 --ocr-queue-size 32 --vl-batch-size 8 --visual-workers 8 --card-workers 2 --vl-queue-size 256 --card-hybrid-config configs/card_hybrid_local.json --suit-backup-config configs/suit_backup_local.json --node C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe
```

Run metadata freezes the entrypoint, arguments, source/layout identities,
production file SHA and NCC config SHA. Production remains frozen during the
run. `progress.json` provides phase, frames, rows, completed OCR/VL work and
resource samples; final metrics and evidence are written separately.
