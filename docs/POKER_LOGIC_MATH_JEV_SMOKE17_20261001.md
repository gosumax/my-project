# POKER LOGIC + MATH + JEV — Smoke17, 2026-10-01

VERDICT: **INCONCLUSIVE**. Experimental implementation and local safety contracts pass. No complete live Jev inference cycle was observed: verified HTTPS returned HTTP 502. Production promotion and final HH export were not performed.

## FILES CHANGED

- `parser_core/poker_hand_assembly.py` (new experimental file)
- `parser_core/poker_logic_engine.py` (new experimental file)
- `parser_core/poker_math_engine.py` (new experimental file)
- `parser_core/poker_card_engine.py` (new experimental file)
- `parser_core/jev_poker_resolver.py` (new experimental file)
- `experiments/replay_poker_logic_math_jev.py` (new experimental file)
- `tests/test_poker_logic_math_jev.py` (new experimental file)
- `docs/POKER_LOGIC_MATH_JEV_SMOKE17_20261001.md` (this report)

## TESTS ADDED

59 pytest cases cover normal hands, fold/check/bet/call/raise/reraise, short all-in calls, all-in raises, uncalled returns, split and side pots, legal card winners/classes, card collisions, wrong winners, forbidden player transitions, forced-bet/header duplicates, incomplete first/last hands, multiline amounts/all-in, punctuation identity conflict, accepted/rejected Jev candidates, hallucinated actions, candidate math/logic regressions, malformed decisions, exact-cache replay, concurrent same-hash requests, native request format, and whole-hand API size limits. Previous-hand payout and card changes without proof remain rejected.

## ARCHITECTURE

`saved ordered timeline -> source-preserving assembled messages -> hand windows -> deterministic state + Decimal ledger + card evaluator -> failed complete candidate -> Jev native typed choice -> evidence-gated immutable candidate view -> same deterministic revalidation -> FULLY_VALID_HAND / JEV_ASSISTED_CERTIFIED / UNRESOLVED_HAND`

Observed source rows retain original keys/values and source order. Parsed fields carry OBSERVED provenance; computations are DERIVED; accepted hypotheses remain JEV_INFERRED in the stored candidate view and inference record. Confidence never substitutes for evidence. Exclusion is a candidate index mask, not deletion of observed rows. Missing amounts require explicit observed raw blind-level evidence. Nickname merges and previous-hand reassignment cannot pass without independent identity/boundary proof. Boundary conflicts retain adjacent source tails/heads for review.

Jev is integrated through `resolve_hand`; it receives only failed complete candidates. The native service selects supplied source-linked hypotheses rather than producing unrestricted prose/new facts. Request identity contains hand data, conflicts, logic/math/resolver/prompt versions and client identity. Cache envelopes preserve exact request/response; identical concurrent hashes coalesce. No key or Authorization header is stored in code, requests, response cache, or report. HTTPS verification remains enabled.

## INPUT AND PROVENANCE

Input: `D:\Новый парсер\runs\production_smoke17_20261001_013909\ordered_timeline.jsonl`
SHA256: `2b7822db0fa4c14706a8ef8e576b5942e22275cd28c29b62c5174720ff566a5b`

Original saved production Smoke17: 28,222 decoded frames, 940.8 seconds; run status PARTIAL_VISUAL. This task did not decode video or rerun capture/OCR/CardHybrid/dedup. Replay consumed all 8,146 saved physical rows on T01–T09.

FILE → FUNCTION → CALLER → DATA: `run_production.py:main` → registered internal `run_smoke14_ram.py:main` → row `normalize` + diagnostic `group_rows` + card projection → saved JSONL. Experimental `replay_poker_logic_math_jev.py:main` reads that JSONL → `assemble_hands` → `validate_hand`/`validate_math`/`best_rank` → conditional `JevResolver.resolve`. Production does not import these new modules.

## DETERMINISTIC RESULT

- Candidate windows: 199; complete start/end candidates: 160.
- Logic PASS: 0; math delta=0: 22; card checks PASS: 81.
- FULLY_VALID_HAND: 0; still unresolved including incomplete windows: 199.
- Complete candidate means observed marker/blind/start/end evidence, not certified hand. It can still have missing amounts, identity, rule or card conflicts. First/last windows receive the same evidence gates.

## CALL / BET / RAISE SEMANTICS

Four financial witnesses on three tables support CALL = amount added now: ADDED arm delta=0, TOTAL arm nonzero. These witnesses still have nonmoney/identity/card conflicts and are not certified HH. No monetary/action rows were removed for the witness comparison.

| Table | Hand | ADDED delta | TOTAL delta |
|---|---|---:|---:|
| T02 | 2876720584 | 0 | -80000 |
| T04 | 2876727536 | 0 | -6000 |
| T06 | 2876722787 | 0 | -12000 |
| T06 | 2876724008 | 0 | -6000 |

RAISE TO contributes target minus the player’s prior street contribution. BET is allowed only on an unopened street, whose contribution starts at zero; added amount and street total coincide there. No visual pot is used as certification evidence. Exact financial evidence and all remaining conflicts are in `semantics_evidence.json`.

## JEV RESULT

- Eligible complete failed candidates: 160; live attempts: 5; cache hits: 0.
- Successful responses: 0; service errors: 5 (HTTP 502). Circuit breaker stopped the remaining 155 requests after repeated failures; already in-flight requests completed.
- Hypotheses accepted: 0; rejected: 0; JEV_ASSISTED_CERTIFIED: 0.
- Zero rejected hypotheses means no hypotheses were returned, not that failed requests passed. Accepted/rejected inference mechanics are verified by controlled test responses only.

## TOP ERROR PATTERNS

Counts below are hands containing a detected conflict, not proven upstream root causes. A hand may appear in multiple rows.

| Pattern | Before Jev | Deterministic resolved | Sent | Jev + revalidation resolved | Unresolved |
|---|---:|---:|---:|---:|---:|
| ROW_IDENTITY_UNRESOLVED | 197 | 0 | 5 | 0 | 197 |
| MATH_DELTA | 177 | 0 | 5 | 0 | 177 |
| MISSING_ACTION | 164 | 0 | 5 | 0 | 164 |
| CHAT_OVERLAP_DUPLICATE | 135 | 0 | 5 | 0 | 135 |
| MISSING_UNCALLED_RETURN | 132 | 0 | 4 | 0 | 132 |
| SIDE_POT_ELIGIBILITY_CONFLICT | 126 | 0 | 4 | 0 | 126 |
| CARD_LOGIC_CONFLICT | 118 | 0 | 3 | 0 | 118 |
| BETTING_ROUND_INCOMPLETE | 109 | 0 | 3 | 0 | 109 |
| OLD_WIN | 99 | 0 | 2 | 0 | 99 |
| STALE_BLIND_OR_ANTE | 89 | 0 | 4 | 0 | 89 |
| MISSING_AMOUNT | 84 | 0 | 5 | 0 | 84 |
| INVALID_CALL | 56 | 0 | 2 | 0 | 56 |
| ACTION_AFTER_FOLD | 50 | 0 | 0 | 0 | 50 |
| ACTION_AFTER_HAND_END | 48 | 0 | 0 | 0 | 48 |
| INVALID_RAISE | 46 | 0 | 1 | 0 | 46 |
| INVALID_UNCALLED_RETURN | 40 | 0 | 0 | 0 | 40 |
| CALL_WITH_NOTHING_TO_CALL | 39 | 0 | 2 | 0 | 39 |
| INCOMPLETE_START | 32 | 0 | 0 | 0 | 32 |
| BROKEN_STREET_ORDER | 21 | 0 | 0 | 0 | 21 |
| RETURN_EXCEEDS_CONTRIBUTION | 19 | 0 | 0 | 0 | 19 |
| BET_INTO_EXISTING_BET | 18 | 0 | 0 | 0 | 18 |
| ACTION_AFTER_ALLIN | 10 | 0 | 1 | 0 | 10 |
| INCOMPLETE_END | 10 | 0 | 0 | 0 | 10 |
| CHECK_FACING_BET | 7 | 0 | 0 | 0 | 7 |
| HAND_BOUNDARY_CONFLICT | 3 | 0 | 0 | 0 | 3 |
| PLAYER_IDENTITY_CONFLICT | 1 | 0 | 0 | 0 | 1 |
| INELIGIBLE_PAYOUT | 1 | 0 | 0 | 0 | 1 |

Multiline assembly (different unit: messages): 1129 assembled messages; 433 amount/all-in continuations resolved deterministically. This is message assembly progress, not hand certification. PLAYER_IDENTITY_OCR and HAND_BOUNDARY_BLEED are represented by their detected identity/boundary/OLD_WIN conflicts; no uncertain nickname merge or cross-hand repair is accepted.

## T01–T09

| Table | Candidates | Complete start/end | Before | After |
|---|---:|---:|---|---|
| T01 | 23 | 13 | {'INCOMPLETE_START': 9, 'CARD_CONFLICT': 9, 'LOGIC_FAIL': 4, 'INCOMPLETE_END': 1} | {'INCOMPLETE_START': 9, 'UNRESOLVED_HAND': 13, 'INCOMPLETE_END': 1} |
| T02 | 20 | 16 | {'INCOMPLETE_START': 3, 'CARD_CONFLICT': 15, 'LOGIC_FAIL': 1, 'INCOMPLETE_END': 1} | {'INCOMPLETE_START': 3, 'UNRESOLVED_HAND': 16, 'INCOMPLETE_END': 1} |
| T03 | 20 | 18 | {'INCOMPLETE_START': 1, 'CARD_CONFLICT': 11, 'LOGIC_FAIL': 7, 'INCOMPLETE_END': 1} | {'INCOMPLETE_START': 1, 'UNRESOLVED_HAND': 18, 'INCOMPLETE_END': 1} |
| T04 | 17 | 13 | {'INCOMPLETE_START': 3, 'CARD_CONFLICT': 4, 'LOGIC_FAIL': 9, 'INCOMPLETE_END': 1} | {'INCOMPLETE_START': 3, 'UNRESOLVED_HAND': 13, 'INCOMPLETE_END': 1} |
| T05 | 19 | 15 | {'INCOMPLETE_START': 3, 'CARD_CONFLICT': 14, 'LOGIC_FAIL': 1, 'INCOMPLETE_END': 1} | {'INCOMPLETE_START': 3, 'UNRESOLVED_HAND': 15, 'INCOMPLETE_END': 1} |
| T06 | 19 | 17 | {'INCOMPLETE_START': 1, 'LOGIC_FAIL': 13, 'CARD_CONFLICT': 4, 'INCOMPLETE_END': 1} | {'INCOMPLETE_START': 1, 'UNRESOLVED_HAND': 17, 'INCOMPLETE_END': 1} |
| T07 | 23 | 17 | {'CARD_CONFLICT': 14, 'INCOMPLETE_START': 6, 'LOGIC_FAIL': 3} | {'UNRESOLVED_HAND': 17, 'INCOMPLETE_START': 6} |
| T08 | 29 | 29 | {'LOGIC_FAIL': 6, 'CARD_CONFLICT': 23} | {'UNRESOLVED_HAND': 29} |
| T09 | 29 | 22 | {'INCOMPLETE_START': 6, 'LOGIC_FAIL': 15, 'CARD_CONFLICT': 7, 'INCOMPLETE_END': 1} | {'INCOMPLETE_START': 6, 'UNRESOLVED_HAND': 22, 'INCOMPLETE_END': 1} |

## CARD CONSISTENCY

Cards are read from source-linked saved PIXEL_ONLY projections only with nonempty valid hashes, matching physical card row/frame/session/table/epoch, verified identity, matching crop/response SHA, OK native recognizer response, and no development-only status. Unresolved projections are not repaired. Physical board/hole collisions, missing critical cards, showdown class contradictions, rank winners, side-pot eligibility and unsupported odd-chip allocation block certification. Contested payout requires observed hole cards for live players; absent evidence stays unresolved. Recognition candidates remain RAW_UNREVIEWED; deterministic consistency is not an independent OCR accuracy certificate.

## PERFORMANCE

| Stage | Mean ms | p50 ms | p95 ms | Max ms |
|---|---:|---:|---:|---:|
| Logic | 0.3388 | 0.2316 | 0.9997 | 2.4521 |
| Math | 0.0662 | 0.0625 | 0.0894 | 0.1959 |
| Jev successful response | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| Jev failed attempt | 1814.2532 | 1812.3162 | 1826.0306 | 1826.0306 |

Full live replay wall: 6.9755s; Jev stage wall: 3.6575s; failed attempt wall sum: 9.0713s.

| Workers | Arm | Wall s | Attempts | Cache hits | Successful responses |
|---:|---|---:|---:|---:|---:|
| 4 | LIVE_COLD | 3.6575 | 5 | 0 | 0 |
| 1 | OFFLINE_FALLBACK_REPLAY | 0.3618 | 0 | 0 | 0 |
| 2 | OFFLINE_FALLBACK_REPLAY | 0.3312 | 0 | 0 | 0 |
| 4 | OFFLINE_FALLBACK_REPLAY | 0.3350 | 0 | 0 | 0 |

Live 1/2/4 worker throughput comparison: NOT_MEASURED. The 1/2/4 arms have zero response-cache hits because the service returned no successful responses; they measure local fallback replay only. Failed-attempt rate is 1.3671/s; successful response throughput and latency remain unmeasured. Never use processed candidate count as successful Jev throughput.

## FULL PYTEST RESULT

`python -m pytest tests/test_poker_logic_math_jev.py -q`: exit 0, 59 passed, 0.40s.
`python -m pytest -q`: exit 0, 484 passed + 7 subtests, 57.48s; wall 58.284s.

Independent audit confirmed the missing-SHA, unresolved-header and unmatched-contribution certification gates. Tests include the audit reproducers.

## PRODUCTION CODE CHANGED

**NO**. All 76 registered production hashes match the frozen manifest before/after; strict integrity passes. Original source-row parity: 8,146 rows, zero missing/extra/modified original fields, identical global/per-table order and ordered canonical hash. Frozen inputs and the production manifest were not modified.

## VERDICT

**INCONCLUSIVE**. The experimental path, deterministic safety tests and lossless replay are implemented. Smoke17 produced 0 fully valid and 0 Jev-assisted certified hands. Live resolver effect cannot be established while the service returns HTTP 502. Identity/card/boundary conflicts are deliberately retained; no final HH is exported. This is not production promotion or proof of 99.9% accuracy.

API contract: https://www.jevai.org/docs (native typed Decisions, not chat completions).

## FOLLOW-UP AFTER CONTINUE: DOCUMENTED MODEL A/B

One fixed Smoke17 complete failed hand (T01 #2876727477, 16 messages) was submitted intact in both arms. Only the model-selection mode changed; this is transport diagnosis, not a second acceptance pool. Both modes returned HTTP 502; no typed response or inference was produced.

| Model selection | Requests | Result | Wall s |
|---|---:|---|---:|
| Explicit typesafe-ai/jev | 1 | JEV_HTTP_STATUS_502 | 6.6703 |
| Service default (model omitted) | 1 | JEV_HTTP_STATUS_502 | 6.3799 |

Windows HTTPS verifies the current public certificate. The failed Python trust-chain probe did not justify disabling TLS verification, and no bypass was used. Valid-key authentication cannot be concluded from a 502; these failures do not demonstrate that the supplied key is invalid.

Saved follow-up evidence: `runs/poker_logic_math_jev_smoke17_model_ab_20261001/results.json`. The selected main replay still records 5 attempts; the two diagnosis requests are listed separately. VERDICT remains INCONCLUSIVE.
