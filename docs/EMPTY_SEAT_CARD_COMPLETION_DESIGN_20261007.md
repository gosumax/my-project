# Positive empty-seat and card completion evidence

Status: FULL-only integration candidate; not golden promotion or full HH proof.

In native short_v3, four of five complete lifecycle contexts never reached
TABLE_DEAL_COMPLETE. Before the initial stack, the configured nickname ROI shows
the lower part of the ANTE/SB/BB overlay rather than a nickname. At genuinely
empty seats it clips OPEN SEAT into PEN SEAT/PENSEAT. Exact nickname occupancy
therefore cannot establish the early participant mask for these hands.

Root reviewed native source tables T08 frame131, T07 frame151 and T06 frame466,
and the first/max-confidence nickname crops for their eight seats. Source images,
ROIs, hashes and PTS are retained outside production outputs in
`runs/full_hh_production_20261006/occupancy_source_review/`.

## Completion contract

Each configured seat must have positive evidence of either the complete native
OPEN SEAT button (including geometry and borders), or the existing frozen card
NCC, or the separate positive card-back supplement when the frozen template is
absent. At least two seats must have positive cards. A missing EMPTY match does not
prove occupancy: every remaining seat still requires positive frozen CARD_NCC.

The same partition must be supported on three consecutive source frames with
the same table/session/hand and increasing PTS. Preserve per-frame native pixels,
hashes, scores, model identity, fixed ROIs and full configured-seat coverage.
The empty-seat indicator grants no actor, historical epoch, or START_STACK proof.

An EMPTY match together with positive cards at the same seat is a conflict and
blocks every FULL completion route, including the existing all-eight route.
Likewise a contradictory current proven nickname occupancy must remain UNKNOWN.
Stale receipts, gaps, context changes, partially obstructed buttons, or similar
nonempty panels must not supply authority. Unknown cases retain the old gates.

The new EMPTY ROI is separate from the unchanged nickname and stack OCR ROIs.
It must include the full button, not just the cropped PEN SEAT text. Candidate
geometry should be visually checked before freezing templates. The known cropped
static fragments PEN SEAT/PENSEAT cannot be used as nickname occupancy evidence
in this new FULL route; excluding them must not rewrite original OCR strings or
create an EMPTY classification by text substitution.

## Source validation and review

The independent reviewer accepted this design with the explicit global-conflict,
same-source three-frame, complete-button and positive-every-seat gates above.
The complete 128x48 EMPTY ROI requires full-image NCC, positive text agreement,
and separate outside-text border NCC, each >=.98, with at most two-pixel native
translation. A regression retaining only text exposed insufficient border
checking; the independent border test now rejects that altered source.
Root reviewed five positive and nineteen negative native button regions. In
three bounded 64-frame windows all 320 known empty-seat observations matched,
with zero false EMPTY among known occupied seats; 48 old nickname hashes agreed.
These counts are bounded diagnostics, not full-video classifier accuracy.

The original frozen card model has 70 of 72 table-seat templates. No generator
or metadata explains the missing T07/seat2 and T08/seat7. Root reviewed complete
two-back source regions at frames190 and170 respectively, and adjacent blank,
one-card, partial and complete observations. A separate registered supplement
matches these fixed ROIs at >=.98 only if the original frozen key is absent.
The frozen model/threshold/recognizer remains unchanged. The supplement carries
native source hashes/PTS and does not read experimental code or runtime images.
Any route using supplementation retains all three source-frame card proofs and
uses an explicit mixed proof label. Unknown and contradictory seats remain so.

Joint three-frame completion occurred at T08 143..145, T07 173..175 and T06
483..485, before initial numeric stacks at181,198,512. Per-frame EMPTY pixels,
card pixels, source identity and scores are retained. Replay preserves same-source
nickname conflict evidence; missing replay labels withhold completion. EMPTY
epoch transitions consume the proven deal receipt independently of nickname OCR.
The additional EMPTY cache is capped at48 frames per table with separate encoded
byte telemetry. A fresh official short remains required before the full run.

Native short_v4 independently exposed a roster-proof callback offset: card deal
source257 is followed by forced CHAT snapshot258. The nickname roster proof now
retains these separate observations, their ordered PTS and common membership
epoch. Both the old same-frame fixture and native next-frame fixture are retained.
