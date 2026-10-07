# FULL_HH_PRODUCTION work log

Request: REQUEST.txt. Baseline: before/files, before/hashes.json and saved dirty/staged patches.
Initial strict integrity: FAIL only because AGENTS.md hash differs from manifest. No baseline code silently replaced.

## Proven data path

run_production.py:main -> registered scripts/run_smoke14_ram.py:main ->
FirstStage.rows -> FirstStage.chat_rows -> AdjacentChatTracker.process ->
OCR_queue_row_ids -> RamRow -> CaptureTableBatches -> Tiny -> timeline.
The legacy tracker admits only GLOBAL_SCROLL bottom births. Non-global physical
new rows have no ordinary OCR path. T01 2084->2085: UNRESOLVED /
INSUFFICIENT_CONFIDENT_ZONES / empty queue, retained source pixels in source audit DB.

No edits to chat_tracker.py, GLOBAL_SCROLL, thresholds, zone checks, symmetry,
legacy R counters or SAFE_ID_STATE. Supplemental IDs use F namespace and retain
UNKNOWN old identity, independent of proof of new bottom exposure.

## Candidate implementation, not yet accepted

- physical_chat_admission.py: unique pixel-supported suffix/prefix translation,
  two or more retained complete rows, per-row and whole overlap support, new
  bottom region only. Alternate translations include zero/no-birth hypotheses.
- FirstStage opt-in overlay: source pixel hashes, frame/span/PTS, UNKNOWN row
  identity, separate admission receipts; ordinary branch unchanged.
- FULL_HH_PRODUCTION profile joins existing direct hand-state/binding/START_STACK
  consumers with STACK timeline, validates connected module matrix before capture.
  Single observer explicitly fans out to both lanes.
- HH exporter serializes original observed messages and partials. Complete export
  requires source-bound seats/START_STACK, existing final validation and eligible
  reconstruction. Conflicting stack receipts cannot certify a hand.

## Reviews and tests so far

Independent reviewer found repeated identical retained rows + local bottom edit
could masquerade as a scroll. Fixed by comparing all no-birth translations over
the same support, with regression.
Reviewer also found row identity field collision with card crop roundtrip status;
now row_identity_status is separate. Conflicting PROVEN stack receipt reduction
was fixed before export. New regressions added.
First targeted run: 18 passed (physical admission + legacy tracker), 1.59s pytest.
Later integration tests and independent review still pending.
Saved-pixel replay v1 interrupted after P1 finding; its results are NOT acceptance.
Replay v2 running under tester, capped at 10099 and 15 minutes.

## Known proof gaps to investigate, not assumed solved

The current STACK evidence provider explicitly exports UNKNOWN without independent
seat/player/occupancy/action-interval proof. Live table completion requires all
configured seats positive, so variable occupancy is not an independently complete
mask. Existing START_STACK arithmetic candidates do not prove true starting stacks.
The source audit gives 64 reviewed starts, not an independent complete-hand
denominator. Explorer is locating source terminal evidence. No 95% claim.

No full Smoke19 until short official profile acceptance passes. No PNG transport
introduced. Frozen/vendor unchanged. Production candidate manifest not yet updated.

## Later iterations

Candidate manifest registered; strict integrity passed before v3-v5 refinements.
It requires a final exact hash refresh before any official launch.
Full pytest: 1454 passed, 178 failed, 19 subtests, 143.37s. Baseline reproduction
preloading the ORIGINAL FirstStage and using 53/53 unchanged other saved parser
modules reproduces exactly the same 178 failure IDs (58 pass in those families).
175 reconstruction expectations and three capture expectations already fail.
No regression expectations were changed to hide them.

Full saved v2: 17,658 views, legacy1461, supplemental336, known omissions266/704.
Common refusal root cause: ROI bottom differs from physical last-row bottom;
new rows may begin inside the translated old blank lower margin. Fixed geometry.
v3 target pairs681/704. Audit first-frame mapping fix removes best-frame span
misjoins (T08 two headers and T09 SHOW card continuation). v5 target pairs694/704,
906 physical supplemental rows on69 target pairs; tests151 passed. The sole10
unresolved events are T08frame1672, with only one short retained line and no
independent two-row proof. No event-specific admission or changed GLOBAL threshold.
Full v4 replay deliberately interrupted when v5 superseded it; no acceptance claim.

Independent reviewer manually checked all57 terminal source crops and seven EOF
partials, with saved per-hand receipts in denominator_review/reviewer_verdict.json.
57 complete hands are supported by the reviewed64-start inventory; exhaustive
discovery of all source boundaries is a separate scope caveat. Initial contexts
are excluded. No parser-segment denominator used.

## Official short and remaining HH path

Official short_v1, 2101 frames, is running through run_production.py with the
FULL_HH_PRODUCTION profile after strict integrity PASS. Registered runtime
sources remain frozen during this run. All 33 required connections are ON.
Full ON FirstStage saved-source replay is queued after the short, with exact
legacy IDs, source geometry, PTS, epochs and pixel hashes compared against1461
baseline rows. Audit span localization now uses only exact reviewed snapshot
frame/hash or stitched first-frame geometry, never a track's best-frame span.

Root cause tracing identified the 8-seat circular dependency: card completion
requires all configured seats, while NICK/OPEN SEAT OCR runs only after that
completion. Independent current-occupancy evidence is being added separately;
it requires all seats resolved from three exact source-continuous label crops,
preserves unclear/overlay/truncated/conflicting labels as UNKNOWN, and grants no
historical actor or START_STACK authority. It is not connected to the active run.

Source evidence also proves concrete nickname limitations: one truncated
krismoltisa... label, numeric7697, overlays and unresolved seats in short logs.
These observations cannot safely be silently merged with similar CHAT names.
Existing START_STACK clean-window/action-onset and full accounting proof remain
necessary; an observed CHAT action timestamp is not proof of actual onset.

Regression investigation found a literal numeric continuation suffix mismatch
(2500 back with a trailing bullet). Candidate graph already accepts the suffix;
literal parse_rows does not. Proposed fix preserves raw text and adjacency.
The 175 blanket reconstruction expectations include genuinely ambiguous source
sequences, so expectations will not be changed merely to make pytest green.


## Short v1 result and integrated correction iteration

Official short_v1 completed exit0: 2101 frames, wall941.8695s, RTF13.45528;
all33 mandatory connections ON, short acceptance18/18 PASS, physical496/OCR496,
timeline520 including24 canonical source joins, errors0, PNG0. HH23 records,
0 certified. Short proves connected transport, not the quality target.
Full saved-source FirstStage ON replay remains active with1461 legacy parity
checked at completion, separate from a native video run.

New independent current occupancy evidence is wired only for FULL profile;
three exact source-continuous reads per fixed seat, bounded pre-DEAL window,
UNKNOWN for overlays/truncations/conflicts, frozen NCC evaluated on proven
occupied seats. Fresh post-DEAL nickname binding remains independently required.
Strict FULL math disables dropped observations and inference-count ranking,
compares complete ordered economic histories and source groups, then revalidates
the unique derived candidate with all existing validators. SQLite snapshot joins
require exact frame/seat/pixel/value; absent clean action intervals remain UNKNOWN.

Literal real-source RETURN/SHOW wraps and first visible context regression cases
are saved under tests/fixtures/full_hh_*.json. First context avoids START_PARTIAL
only with newly admitted physical numbered header plus adjacent same-number start.
Canonical forced joins now preserve an existing physical prefix before DEAL or
first voluntary action, including wraps; delayed snapshot domain placement stays.
Focused six-file suite114 PASS after corrections. Earlier occupancy/math eight-file
suite225 PASS. Independent risky audit and final full pytest remain pending.

Integrity was refreshed only for explicitly owned changed paths and two new
modules/tests/fixtures; check_production_integrity.py PASS. This is an integration
candidate, NOT_PROMOTED/quality NOT_PROVEN. Runtime remains frozen for review.

Independent visual review inspected all31 additional unnumbered NEW_HAND candidates:
every candidate repeated a previously visible header. Three numbers outside64
reviewed starts were already visible at video frame0 (T01/T02/T09); excluded initial
contexts. Per-source review receipts are being finalized, no parser denominator.
