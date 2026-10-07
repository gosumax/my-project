# Source-bound truncated nickname design

Status: implemented as a FULL-only candidate after short_v3 finished. Targeted
tests and final implementation review are pending; no native acceptance claim.

The real T05 hand 2878365559 has a complete eight-actor forced-post roster,
eight occupied participant seats, seven exact nickname bindings, and seat2's
native `stoleyourbl...` at confidence .970–.977. The complete actor name
`stoleyourblind` is observed in CHAT. The current exact-label binding correctly
leaves this seat UNKNOWN. A guessed complement or fuzzy alias is not sufficient.

## Required proof

1. A physically source-bound complete forced snapshot has one proven ANTE per
   distinct participant actor. Distinct actor count equals the independently
   proven occupied participant seat count. SB/BB repetitions do not add actors.
2. All seats retain the same source-proven occupancy epoch and hand membership
   from the forced source through the label confirmation interval. Unknown or
   changed seat events withhold the complete roster matching proof.
3. Each seat has a fresh high-confidence, contiguous, three-source nickname
   observation. Exact labels remain exact. Truncated labels require explicit
   trailing ellipsis and a nonempty literal prefix; a bare ellipsis grants no edge.
   Case is preserved, with no fuzzy spelling or OCR alias substitution.
4. Candidate actor edges satisfy every eligible label observation for that seat
   in the interval. This is an intersection of constraints, never their union.
   For example, exact `Alice` followed by `Bob...` remains contradictory even if
   other seats would force a unique global assignment.
5. The complete seat-to-actor graph has exactly one perfect matching. Zero or
   multiple matchings remain UNKNOWN. Every selected actor is rechecked against
   all retained eligible label triples. Every newly bound seat must have direct
   literal label support; roster complement alone grants no authority.
6. The receipt preserves the whole graph, unique assignment, original nickname
   OCR, confidence, source frames/PTS/hashes/pixels, participant epochs, complete
   CHAT roster references, and the exact proof interval. Canonical consumption
   reproduces the proof and checks the same actor/seat/epoch/source context.

## Integration boundary

The FULL profile may enable this as a separate historical binding rule. Original
nickname observations and the existing exact-label route are not rewritten.
Canonical forced posts can accept the new rule only after replaying its proof.
The existing first-visible stack, arithmetic, SQLite source join, reconstruction
and HH validation gates remain required independently.

The independent reviewer accepted these premises and identified the union-of-
constraints counterexample above. Before implementation, inspect short_v3's
actual blockers; before another native run, add real-source and adversarial
regressions, obtain an implementation audit, and register the candidate hashes.

The implementation review additionally required replaying complete seat history
for participant membership, verifying raw forced-prefix pixels/OCR and contiguous
row indices, checking the configured nickname ROI, and matching frame/PTS between
the forced prefix, deal receipt and nickname observations. These gates and their
negative regression cases are included in the candidate.

short_v3 completed 2,101 frames with 18/18 connection checks and zero runtime
errors/PNG files. Its five complete lifecycle contexts produced zero certified
HH. T05 hand 2878365559 had seven proven first-visibility observations, but all
eight arithmetic facts were withheld because the unbound truncated BB actor left
the canonical forced corrections incomplete. The first stack for seat5 remained
below .95 OCR confidence and is independently UNKNOWN; nickname binding cannot
repair that missing numeric authority.
