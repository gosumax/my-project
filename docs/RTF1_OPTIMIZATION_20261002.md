# Smoke17 RTF optimization

User authorized reducing production RTF toward1. Prior
`production-golden-poker-20261002` and all older snapshots remain immutable.
Candidate is `production-candidate-rtf1-20261002`; official entrypoint remains
`run_production.py`. No certified HH or accuracy claim follows from performance.

## Proven bottleneck and minimum changes

Baseline full Smoke17:28,222 frames,940.8s source,1394.967493s wall,
RTF1.482746. Capture963.440512s, temporal reduction249.696070s,
poker analysis4.209739s, Excel37.886382s. Target RTF1 requires saving454.167493s.

`scripts/run_smoke14_ram.py::main` previously submitted only table1 complete
OCR batches during capture, then globally sorted and submitted tables2–9 at EOF.
`parser_core/parallel_ocr.py::CaptureTableBatches` now submits complete
per-table/session batches of8 while capture advances; only per-table tails wait
for EOF. Original row-key mapping and final table/frame/span sorting remain.
Every raw OCR response and measured confidence is retained without substitution.
Full8146-row frozen-pixel OCR experiment had exact text/status/ID parity;
15scores differed, max absolute delta0.04212636. Confidence is a review field,
not a decision threshold in the registered downstream path.

`parser_core/ram_first_stage.py::FirstStage.rows` keeps one owner of frame,
table history, row identity and epochs. FRAME3 decoder threading prefetches
frames; it does not parallelize or reorder capture-state transitions.
The2000-frame decoder experiment preserved all RGB hashes, PTS/timebases/order
for FRAME2 and FRAME3. Decode-wait improvement alone is not an end-to-end speedup.

`parser_core/temporal_overlap.py::TemporalObservations.record` caches immutable
zlib payloads by shape/SHA; a cache hit requires exact decompressed-byte equality.
Every observation tuple and context remains separate. Logical packed byte volume
and unique allocated payload volume are reported separately. An ordered native
observation fingerprint permits comparison with the frozen950,047-view baseline.

`TemporalObservations.reduce` partitions by table into up to3 processes after
OCR has drained. Within each table frames remain serial. Worker state is isolated
by table/session/hand; merged state rejects duplicate keys. Stable sorting by
original visible-block ordinal preserves complete removal/decision order.
Each spawned worker checks the parent manifest fingerprint and installs the
registered production import/execution guard. Failure retains the source inputs
and closes the process pool before propagating; partial output is not accepted.

## Architecture and resource contract

One capture-state owner;3 decoder threads;2 persistent OCR workers with queue32
batches;2 visual threads and2 card slots with queue256;3 post-OCR dedup processes;
then strict assembly/logic/math/cards and Node Excel export. State transitions
within a table remain serial. Packed observations still accumulate until EOF;
this is not a globally bounded streaming collector. No filesystem image transport
or intermediate PNG writes are introduced.

New `ocr_batches.jsonl` is schedule provenance after OCR completion.
`process_samples.jsonl` records PID/create-time/executable, phase, CPU/RSS samples.
Final summary adds per-OCR-worker preparation/request/inference/startup timings
and per-table dedup PID/wall/CPU timings. Worker sums, process CPU work, enclosing
stage wall and lifetime intervals remain distinct measurements.

## Evidence before official live acceptance

`runs/rtf1_optimization_20261002_checks/` contains exact source/model pins,
all new OCR responses and bounded OLD/NEW experiments. Frozen500 temporal
inputs:22,106 observations,349 physical rows,348 retained,1 removal. Serial
reduction10.587898s vs parallel3 6.034152s, with exact complete evidence/order.
Packing3.921641s vs shared2.461077s; exact tuple/payload parity; logical33,527,828B,
unique13,902,308B and13,264 cache hits. These are isolated microbenchmarks.

An initial capture-mode test had exact row/observation parity but failed its
final source-pin check because root added worker instrumentation during the test.
Its timings are retained but not accepted as current implementation performance;
a repeat against stable candidate pins is required. Independent code review found
no material blocker. Actual full-pipeline RTF remains NOT_MEASURED for this candidate.
