# Production hand assembly, poker logic and math connection

User authorized connecting the existing stages and selected full Smoke17.
Previous golden snapshots remain immutable; no OCR, card recognition, capture,
temporal dedup or frozen training inputs are replaced.

## FILE -> FUNCTION -> CALLER -> DATA

`parser_core/poker_pipeline.py::analyze_timeline` is called by
`scripts/run_smoke14_ram.py::main` after temporal reduction and
`project_chat_card_events`, before final JSONL/Excel export. It receives the
retained timeline and the real per-stable-ID OCR status map.

Deep-copy source adapters retain every original field, add stable `row_id`, raw
text, PTS-derived timestamp and OCR status aliases. `assemble_hands` groups
source rows by table/session, then `validate_hand`, `diagnose_hand`,
`audit_observed_ledger` and deterministic Hold'em card evaluation execute.
Coverage must match every retained source ID exactly once. Timeline and raw
physical/OCR/card evidence stay unchanged.

## Strict production contracts

Historical helpers accept synthetic raw text cards in tests. Production uses
`require_pixel_cards=True`: raw OCR never supplies authoritative card facts.
Only existing projected PIXEL_ONLY evidence with matching original source row,
session/table/frame/epoch/SHA and native reader response can bind cards.
The verified projection allows physical rows one or two spans below its trigger.
Offset-two sources are referenced independently in card provenance, without
merging intervening unrelated OCR or duplicating source message coverage.

Grammar continuations require same table/session/epoch/frame and adjacent
physical row spans. A truncated action and an unrelated later number remain
separate observations. Viewport epoch changes alone do not define poker hands.
Original UNKNOWN/UNRESOLVED rows remain visible and can block validation.
Failed OCR remains a provenance blocker even if its text happens to parse.

No synthetic monetary semantics are supplied. Missing or unproven call/bet
interpretation stays explicit. Partial/untrusted contribution-minus-payout
deltas are NOT_VERIFIABLE, not proof of a conservation failure. Raw validator
results remain alongside the scoped diagnostics. A raw MATH_FAIL on an
unproven ledger is reported as an unresolved diagnostic conclusion.

No external Jev/HTTP client is invoked. The resolver module is registered only
as the static dependency of an unused diagnostics revalidation function.
There are no new inferred observations, actions, amounts or cards.

## Outputs and count contract

- `hand_analysis.jsonl`: every assembled segment, original source messages,
  validator blockers, first-break trace and observed ledger.
- `hand_summary.json`: segments, unique numbered hands, observed starts/ends,
  complete candidates, fully validated hands and separate certified HH count.
- `validated_hands.jsonl`: only candidates passing both deterministic validation
  and diagnostics with complete boundaries and no blockers. This is not a
  certified HH text exporter.
- Existing workbook adds `HAND_SUMMARY`; all prior timeline/card sheets remain.

`fully_validated_hands` and `complete_candidates` are separate. A visible payout
or numbered header alone does not prove a complete hand. `certified_hh` remains0
until a distinct certification/export contract exists.

## Source-backed pre-live evidence

`runs/poker_pipeline_promotion_20261002_checks/` retains targeted and offline
checks. On the previous full Smoke18 timeline:592/592 source rows preserved,
22segments,14numbered,7complete candidates,0fully validated.33/33 projected
card events bind exact pixel evidence, including the real T04 frame576 offset2
case; zero binding conflicts. All22 observed math conclusions remain
NOT_VERIFIABLE. Full live metrics and final golden acceptance are appended after
actual process completion.

## Smoke17 preflight

Source `video/smoke_17.mp4`: SHA256
`27c0ba10544e0f6025e2d563baf7693d36c185cdbbfeb4517d3486116e9f8baf`,
28,222 decoded frames. Official run uses two OCR, two visual and two card workers.
The temporal collector still accumulates compressed appearances until OCR drains;
after reduction its native matching galleries and marker templates are released
before downstream analysis/Excel. Removal/context JSON evidence is retained.
Independent review found no downstream consumer of the released structures.

Early interrupted Smoke18 and Smoke17 launches are retained with explicit abort
markers; neither is a completed full-run result. The final full Smoke17 uses a
fresh directory. Benchmark selection is explicit in the freeze utility and
requires the selected source path and SHA to match completed run metadata;
the earlier Smoke18 benchmark is not relabelled as Smoke17.

## Completed full Smoke17

`runs/production_poker_smoke17_full_candidate_v2_20261002/`: actual exit0,
28,222/28,222 frames,8,146 physical/OCR rows ->7,954 retained rows,
192 removals. Physical and OCR JSONL file SHA exactly match previous full
Smoke17. Retained stable IDs/order exactly match previous safe boundary replay;
all192 removal proofs pass,4,506 UNKNOWN observations are retained.
Raw card recognition and fresh projected-card replay pass;329 card messages bind
verified pixel evidence with0 binding conflicts. Workbook contains7,954
ALL_EVENTS and196 HAND_SUMMARY data rows, exact raw IDs/text/hash/blockers/flags.
Source coverage is7,954/7,954 exactly once.

196 assembled segments,189 unique numbered hands,163 complete candidates,
1 fully validated hand,0 certified HH. Analysis remains PARTIAL_UNRESOLVED.
Raw validator has2 FULLY_VALID_HAND and2 MATH_FAIL, while final combined gates
admit only1; these raw results are retained rather than renamed.
Scoped math195 NOT_VERIFIABLE/1 PASS; card184 NOT_VERIFIABLE/11 CONFLICT/1 PASS.
First-break distribution:69 overlap duplicates,42 missing actions,40 stale
OLD_WIN payouts,19 unproven amount semantics,11 missing amounts,5 missing blinds,
4 invalid raises,3 actions after fold,1 stale blind/ante,1 incomplete end.

Wall1394.967493s, RTF1.482746; capture963.440512s, reduction249.696070s,
hand analysis4.209739s, Excel37.886382s. Peak pipeline3633.8008MiB,
main2227.8594MiB; compressed appearances1,412,825,440bytes. Zero error rows,
zero retries and zero intermediate PNG files/bytes. Against previous full
temporal-free Smoke17, wall+42.447%, capture+24.473%, peak pipeline+669.891MiB.
The additional audited appearance reduction is a measured throughput cost.

Golden `production-golden-poker-20261002` is frozen against the exact completed
Smoke17 candidate500 manifest/runtime/source, with a new immutable provenance
snapshot. Final golden500 and active pytest evidence are recorded separately
under `runs/poker_pipeline_promotion_20261002_checks/`.

Final golden500 completed exit0:500 frames,348 retained rows,16 segments,
9 numbered hands,3 complete candidates,1 jointly validated,0 certified HH.
Wall40.652985s, capture17.855542s. Physical/OCR/timeline/removal/context file SHA
parity passes. Hand analysis/validated-hand semantic parity passes after excluding
measured `logic_ms`/`elapsed_ms`; visual evidence excludes measured elapsed times.
All workbook cells match after accounting for the output provenance path.
The first byte-level hand comparison correctly differed on runtime measurements;
no action, amount, card, blocker or validation-result differences were found.

Independent final audit: strict current hashes101/101, new snapshot102/102,
prior temporal snapshot94/94 and20260930 snapshot77/77 all intact. Source17
benchmark path/hash and all new output contracts are registered.

Final active pytest:781 passed and7 subtests passed, exit0,77.48s.
Final strict production integrity:PASS. Detailed logs/result JSON are retained
in the checks directory. No unrelated Git changes were staged or committed.
