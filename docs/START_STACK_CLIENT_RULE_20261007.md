# First visible PokerDom stack: connected FULL candidate

The user clarified that a hand's first visible numeric stack is already net of
ANTE/SB/BB. The FULL profile now has an authority path for that specific initial
snapshot. HH START_STACK adds back the actual source-bound forced posts. A later
numeric snapshot, a predicted timer position, or an OCR guess cannot substitute
for the first observation. This is a candidate, not an HH95 or golden promotion.

## Measured source behavior

`runs/full_hh_production_20261006/stack_overlay_timing/` contains receipts and
contact sheets for 64 independently reviewed hand starts and 512 physical seats.
33,336 saved RGB crops were projected into the registered STACK ROI. A sequential
PyAV RGB24 decode reproduced all 952 saved-source comparisons byte for byte.

Two directly reviewed onset/end examples:

| Hand/table | First forced label | Last fading label | Blank interval | First numeric | Delay |
|---|---:|---:|---|---:|---:|
| 2878365559 / T05 | 238 | 282 | 283 | 284 | 46 frames / 1.533 s |
| 2878365523 / T08 | 130 | 176 | 177–180 | 181 | 51 frames / 1.700 s |

The delay is the difference between source frame ordinals. The interval including
both endpoint frames contains one more frame. These examples disprove a single
universal frame delay in this recording; the duration of the completely static
subphase was not independently established. Existing 64-frame capture budgets
cover both examples. No guessed fixed delay supplies START_STACK authority.

T03 hand 2878366904, seat1, contains a visible number at 6590 while the unchanged
digit gate first passes at 6600. Therefore neither a gate rejection nor its first
pass defines literal visibility. Initial carried video contexts remain partial.

## Source path and cause

`start_stack_live.py` preserves source-ordered direct stack OCR and its first
three-frame attempt. `seat_forced_binding.py` supplies physical seat, occupancy
epoch, participation, and forced-post facts; `canonical_events.py` binds the exact
actor and complete actual posts. `poker_math_engine.py` previously always withheld
START_STACK for lack of an independent voluntary-action interval, even for the
initial post-forced display. The new user rule removes that requirement only
after the first-visibility proof succeeds. Default non-FULL behavior is retained.

The original direct/SQLite join also required equal pixel hashes although all
72 routes used different ROI rectangles and two different grayscale conversions.
That comparison cannot prove equality between those two representations. FULL
now retains their common original RGB source and reproduces both projections;
the SQLite join checks exact table/session/seat/frame/PTS/timebase and projected
hash. Raw OCR and visual-state admission in each lane remain unchanged. A
conflicting confirmed SQLite number is withheld before solver use and remains
CONFLICT in the export audit; an UNKNOWN SQLite OCR is never relabelled as numeric.

## First-visibility authority

`start_stack_visibility.py` first verifies original RGB, direct grayscale PNG,
the refined SQLite grayscale, source identity, and continuous frame/time order.
The first three sources must positively match a native forced-overlay template.
The 62-template bank retains hashes, geometry, frame and source references.
Root visually reviewed its seed examples and a 72-cell variant sheet; empty,
all-in and two obstructed panels were excluded. Template recognition identifies
only the forced-overlay category: SB/BB ambiguity cannot supply a money amount.

Every subsequent pre-number source must retain the positive overlay/fade shape
on the fixed inner pixel region, or be near-uniform within the measured codec
noise range. Unexplained nonblank pixels, source gaps, wrong hand/seat/time,
earlier reliable numeric OCR, or a failed first numeric triple remain UNKNOWN.
Fading-label OCR ghosts such as `90` at confidence .24 are retained verbatim;
positive pixel evidence, not that uncertain string, classifies the phase.

The actual first numeric triple still requires the existing .95 confidence and
three identical numeric values. The proof and arithmetic are reproduced at the
consumer, and exact canonical actor/seat/epoch/post bindings are required before
the solver receives a bankroll. The HH exporter additionally requires the bound
SQLite source, complete lifecycle and existing final reconstruction validation.

## Bounded evidence before native acceptance

Real-source regression proves T05 seat1: 17,416 + 70 + 300 = 17,786; seat4:
11,055 + 70 = 11,125. Seat5's first numbers at confidence approximately .89 remain
UNKNOWN. No confidence threshold was lowered and no fixture OCR was rewritten.

The first five-file regression passed 129 tests; eight additional SB/BB dim-number
cases also passed. A subsequent small audit change retains a solver-withheld
conflict in export status; its separate regression and the full suite are required
before native acceptance. Current runs and later results are recorded in
`runs/full_hh_production_20261006/start_stack_domain_tests_v*/` and `full_pytest_v4`.

The template diagnostic identified 1,119 of 1,416 source candidates; that count
includes mixed occupancy and is not an accuracy denominator. There were no false
forced labels in 32 numeric negatives. A bounded search of 9,833 saved states
found no native literal 8/80/88. Explicitly derived native-font glyph tests cover
those values and multiple brightness levels: no false PROVEN result in 45
injections into actual SB/BB/ANTE prefixes. Maximum later-shape NCC was .355/.339/
.101 respectively, below .90. Derived examples do not establish coverage of all
native numeric variants; this limit remains explicit.

The required next steps are a fresh official short FULL run, its acceptance audit,
and the 10,100-frame official Smoke19 run with independent quality auditing.
