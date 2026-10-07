# Smoke14: isolated local VL benchmark on saved BOARD crops

Measured on 2026-09-30. This is a benchmark, not a production change or an HH acceptance test.

## Hardware and runtime

- NVIDIA GeForce GTX 1650, 4096 MiB dedicated VRAM; 16,322 MiB system RAM.
- Local PaddleOCR-VL-1.5 model revision `2a4195faa5e7914c12f2fc601d72c81caf8d2da5`, weights SHA-256 `d557c9d8997ae57ed3b1b33bdf347be878cc335687f32ca105341c16973f8958`.
- PyTorch 2.7.1+cu126, CUDA 12.6, Transformers 5.3.0, FP16, SDPA, one model on `cuda:0`. Model load was 4.31 s in the batch 1/2/4 process, excluded from inference timing. The batch 8 process loaded the same model once separately.
- GPU memory before/after model load in the batch 1/2/4 run: 725/2628 MiB. All model parameters were on CUDA. No CUDA OOM was observed. Task Manager displayed about 0.1 GB shared GPU memory during inference, but peak shared-memory usage was not instrumented.

## Dataset and method

Source: `runs/smoke14_parallel_vl_full_20260929/visual_evidence.jsonl` and its saved `BOARD` PNGs. There are 106 unique crop PNG SHA-256 values; all 106 files were present. The benchmark selected 32 distinct SHA values in deterministic table/frame order, round robin across tables, and verified each PNG byte SHA before inference. The same 32 crops and their order were used at every batch size. The selection is recorded in each run's `dataset.json`.

Inference used the production model/prompt (`OCR:`), processor image-size settings and 256-token limit with one `model.generate` call per actual batch. There were two warm-up batches per size before timing. Timing includes measured-loop preprocessing, transfer, generation and output handling; model loading and warm-up are excluded. The listed median and p95 are **batch generation** times, not per-image response times. All batches were full because 32 is divisible by 1, 2, 4 and 8. Process RSS is reported as RAM; whole-system RAM peak was not measured. GPU memory and utilization were sampled through `nvidia-smi`; CPU percentage is for the benchmark Python process and can exceed 100% on multiple cores.

| Batch | Crops | Wall s | s/crop | Crops/s | Median batch s | p95 batch s | Peak VRAM MiB | Peak process RSS MiB | Raw parity vs 1 | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 32 | 91.283 | 2.853 | 0.3506 | 2.897 | 3.132 | 2732 | 3098 | baseline | OK |
| 2 | 32 | 80.218 | 2.507 | 0.3989 | 5.013 | 5.233 | 2658 | 3107 | 32/32 exact | OK |
| 4 | 32 | 71.268 | 2.227 | 0.4490 | 8.806 | 9.024 | 2735 | 3107 | 32/32 exact | OK |
| 8 | 32 | 66.684 | 2.084 | 0.4799 | 16.511 | 16.589 | 2919 | 3053 | 32/32 exact | OK |
| 16 | 0 | — | — | — | — | — | — | — | not measured | memory guard stopped generation |

Throughput relative to batch 1: 1.138x, 1.281x and 1.369x for batch 2, 4 and 8 respectively. Batch 8 was only four measured calls, so sustained stability beyond this bounded run is unproven. GPU utilization averaged 83.4%, 95.7%, 96.3% and 98.0% across these modes. The highest process CPU reading averaged by mode was about 97%; no steep process RAM increase was observed. The batch 8 peak left 1177 MiB of dedicated VRAM by `nvidia-smi`.

At the user's request, batch 16 was attempted on the same dataset in a separate single-model process. During its first warm-up generation, CUDA free memory fell below the pre-set 768 MiB guard and generation was stopped. There was no CUDA OOM. No 16-image batch completed, so its speed, peak memory, parity and quality are **NOT_MEASURED**; batch 16 is not a viable measured option under the selected memory margin. The benchmark process returned exit code 0 but its `summary.json` records `status=ERROR`; that status is authoritative.

## Quality

Every batch 2/4/8 raw text matched the batch 1 raw text for the same SHA (32/32 in each mode). No inference exception or empty result occurred. However, `parser_core.visual_cards.parse_cards` accepted **0/32** BOARD readings at every batch size. Typical raw outputs contain ranks without a complete suit sequence. Thus batching preserved the baseline output, while the baseline itself did not produce usable strict BOARD cards on this sample. `EXACT_SAME` is raw-output parity, not a claim of correct card recognition.

Per-SHA raw text, normalized cards, status, batch position and actual batch size are preserved in `results_batch*.jsonl` under the run folders. The `cards` field is null for all 32 images in every mode.

## Smoke14 estimate and queue window

The saved Smoke14 run recorded 106 unique direct BOARD crops and 332.4 s source-video duration. Multiplying benchmark s/crop by 106 gives a **full-queue theoretical estimate**, assuming the same crop mix and full batches:

| Batch | Estimated VL s | Theoretical VL RTF |
|---:|---:|---:|
| 1 | 302.4 | 0.910 |
| 2 | 265.7 | 0.799 |
| 4 | 236.1 | 0.710 |
| 8 | 220.9 | 0.665 |

The source-video average is 106/332.4 = 0.319 BOARD crops/s. Actual wall-clock crop arrivals at the VL worker were not logged, so this is not a measured production queue-arrival rate. A 100/200/300 ms grouping of source-video timestamps yields 92/86/83 singleton groups out of 97/94/92 groups respectively. This is a source-time proxy, not a live micro-batch throughput test; it suggests that a short waiting window would often fail to fill a batch. No production queue or worker was changed.

## Decision

`RECOMMENDED_BATCH = 4` for a **backlogged experimental BOARD queue**: it improved throughput 28.1% over batch 1 with 1361 MiB dedicated-VRAM headroom. Batch 8 gave another 6.9% throughput over batch 4, while median batch response time rose from 8.81 to 16.51 s; only four batch 8 calls were measured. This is not authorization to switch production to direct BOARD VL, because strict readings were valid for 0/32 images and live queue arrivals were not measured.

`VL_BATCHING_SPEEDUP = 1.28x` for the recommended batch 4 on these 32 crops. `CAN_VL_KEEP_UP_WITH_PIPELINE = UNCLEAR`. `NEED_BOARD_ARRIVAL_RATE_MEASUREMENT = YES`.

Artifacts: `runs/smoke14_board_vl_batch_benchmark_20260930_v2` (batch 1/2/4), `runs/smoke14_board_vl_batch8_20260930` (batch 8) and `runs/smoke14_board_vl_batch16_20260930` (guard stop). The earlier `runs/smoke14_board_vl_batch_benchmark_20260930` is retained as a failed pre-inference diagnostic (`KeyError('path')`), not used for measurements. Benchmark code: `experiments/benchmark_smoke14_board_vl_batches.py`.
