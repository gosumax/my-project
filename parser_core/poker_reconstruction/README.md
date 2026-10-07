# Offline poker reconstruction

This package is isolated from the manifest-defined production pipeline. It reads
the supplied ALL_EVENTS workbook or JSONL without modifying the input, OCR,
tracker, frozen modules, production imports, configuration or validators.

Flow: raw rows → candidate grouping graph → assembly/identity hypotheses →
bounded poker solver → integer chip accounting → source-preserving decisions.
Returns are distinct events constrained to observed unmatched street excess,
not annotations added to a bet merely because the rows are adjacent.

Run from the project root:

```powershell
python scripts/reconstruct_timeline.py --input runs/production_rtf1_smoke17_full_candidate_20261002/ordered_timeline.xlsx --output reports/poker_reconstruction/fresh_run --start-stack-oracle --expectations tests/fixtures/poker_reconstruction_expected.json
python -m pytest tests/test_poker_reconstruction_state_machine.py tests/test_poker_reconstruction_accounting.py tests/test_poker_reconstruction_stale_filter.py tests/test_poker_reconstruction_assembly.py -q
python -m pytest tests/test_poker_reconstruction_regression.py -q
```

The regression expectations are read after reconstruction, solely for reporting
and tests. They never influence assembly, aliases, amounts or solver decisions.

Each candidate grouping retains original rows and indices. Orphan numeric and
all-in tokens are not complete semantic actions. Neighbor grouping requires
same table/session/epoch; context, state, contribution and payout constraints
then accept or reject that grouping. Unsupported text remains UNKNOWN. It can
block recovery; it is never silently discarded to make a pot match.

Nickname variants remain distinct by default. Optional `--identity-evidence`
accepts a JSON mapping `table:hand_id` to candidate objects with `a`, `b`,
`seat_a`, `seat_b`, `continuous_identity`, `same_table_hand_context`, and
`identity_evidence_refs`. These fields must describe independent observations;
the caller owns their provenance. Similar text or equal pot alone does not
authorize an alias. Both distinct and merged hypotheses enter the solver.

`--start-stacks` accepts exact nonnegative integer stacks keyed by table:hand_id
then player. Without supplied exact observations, `--start-stack-oracle`
reports capability limits and an unmeasured uplift, not invented stacks or +0.

Limitations: the solver implements bounded observed-event constraints, not a
complete NLHE theorem prover. It does not invent missing actions, infer seat
order, validate board/hole-card ranks, or certify HH. It does not assume SB is
always half BB. Missing blind amounts may be constrained by an observed future
raise-to and first call delta, and are accepted only with unique exact accounting.
Equal-ranked alternatives differing in source-row attribution remain AMBIGUOUS.
Search exhaustion is AMBIGUOUS. An absent amount or unsupported raw form prevents
an INVALID claim from being interpreted as a proof of impossible real poker.

Outputs retain raw/kept/dropped/inferred events, source decisions, per-step street
and total contributions, returns, layers, observed payouts, candidate grouping
graph, identity hypotheses, failures and five-control diagnostic traces. Candidate
prefix ledgers are diagnostics; only a unique completed accepted solution obtains
the hand's calculated pot. All unresolved source rows remain UNRESOLVED.
