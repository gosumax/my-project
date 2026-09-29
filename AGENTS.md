# Новый парсер: работа Codex

Use the current code in this directory as the source of truth. Do not import architecture or paths from older PokerDom projects without verifying them here. This folder is the current project root even when it has no `.git` directory.

## Main agent

Use `gpt-6-sol` with `medium` reasoning by default. The main agent owns the task interpretation, architecture, shared production-code changes, integration, evidence sufficiency, and final verdict. Increase reasoning only for a real root-cause or architecture deadlock. Prove the common cause through `FILE -> FUNCTION -> CALLER -> DATA` before changing shared logic. Preserve raw pixels/OCR, provenance, and explicit `UNKNOWN`, `UNRESOLVED`, and partial outcomes.

## Routing

MUST delegate sufficiently broad repository searches, call-path tracing, multi-file reading, OLD/NEW implementation comparisons, PNG/write-site inventories, and existing-test discovery to `explorer`. Give it a narrow question and return contract; ask for paths, symbols, facts, and a short conclusion, not file dumps.

MUST delegate substantial pytest, smoke, parity, benchmark, hash/order/frame/table comparisons, performance measurement, and large-log analysis to `tester`. Set a bounded command and data range; request exit code, pass/fail count, numerical results, timings, and only decisive failures. Do not run full Smoke14 or a long video merely to validate agent routing.

The `tests/test_*.py` files contain pytest functions. Use `python -m pytest` for targeted tests rather than `unittest discover` unless the target is verified to use unittest.

MUST delegate an independent read-only audit to `reviewer` for risky changes to pipeline architecture, state machines, hand boundaries, event contracts, capture optimization, dedup/change detection, or logic that could silently alter parser output. Do not invoke it for minor edits.

The main agent decides the fix, edits related production components, resolves conflicting evidence, decides whether tests prove parity, and reports the final result. For an answer obtainable with one or two short tool calls, work directly. Do not chain agents or duplicate the same investigation. Use at most two subagents concurrently, only for independent work; prefer sequential exploration then testing when the test depends on findings.

## Current code and evidence

Start with `README.md`, `scripts/`, `parser_core/`, `configs/`, and `tests/` as relevant. Current capture and streaming paths include `scripts/capture_nine_tables.py`, `scripts/run_nine_tables_streaming.py`, and `parser_core/ram_first_stage.py`; verify the actual call path before relying on this map. Capture optimization must preserve observation identity, frame/table/order and pixel or hash parity with a valid baseline. Keep evidence artifacts when required; do not infer certified hand histories from partial pipeline diagnostics.
